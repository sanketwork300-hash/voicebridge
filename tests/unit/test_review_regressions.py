"""Regression tests for issues found in code review of v0.2."""

import asyncio

import pytest

from voicebridge.core.metrics.metrics import SessionMetrics
from voicebridge.core.pipeline.s2st_pipeline import S2STRealtimePipeline
from voicebridge.core.pipeline.speech_gate import SpeechGate, SpeechGateConfig
from voicebridge.core.session.config import SessionConfig
from voicebridge.core.types import AudioChunk, EventType, SpeechDecision
from voicebridge.media.probe import input_args
from voicebridge.providers.base import ProviderError, S2STCapabilities


class CrashingS2ST:
    name = "crashy"
    capabilities = S2STCapabilities(name="crashy", model_license="test")

    async def translate_stream(self, audio_stream, source_language, target_language):
        await audio_stream.__anext__()
        raise ProviderError("worker died")
        yield  # pragma: no cover


def test_s2st_stop_does_not_hang_after_engine_failure():
    async def run():
        p = S2STRealtimePipeline("s", SessionConfig(), CrashingS2ST(), SessionMetrics("s"))
        await p.start()
        for _ in range(400):  # more than the audio queue holds
            await p.push_audio(AudioChunk(data=b"\x00" * 3200))
            await asyncio.sleep(0)
        await asyncio.wait_for(p.stop(), timeout=5)
        return [e.event_type async for e in p.stream_events()]

    events = asyncio.run(run())
    assert EventType.ERROR in events and events[-1] is EventType.SESSION_ENDED


class ScorelessClassifier:
    name = "energy"

    async def classify(self, chunk):
        return SpeechDecision(True, "UNKNOWN", 0.45, chunk.start, chunk.end)


def test_gate_handles_classifier_without_category_scores():
    gate = SpeechGate(classifier=ScorelessClassifier(), config=SpeechGateConfig())
    d = asyncio.run(gate.decide(AudioChunk(data=b"\x10\x00" * 1600)))
    assert d.should_transcribe and d.event_type == "UNKNOWN"


def test_ffmpeg_inputs_are_restricted_to_local_files_and_expected_demuxer():
    args = input_args("/x/uploads/abc.mp3")
    assert args[:2] == ["-protocol_whitelist", "file"]
    assert args[2:4] == ["-f", "mp3"] and args[-2:] == ["-i", "/x/uploads/abc.mp3"]
    assert input_args("a.webm")[3] == "matroska"


def test_job_manager_finish_is_idempotent_and_delete_removes_artifacts(tmp_path):
    from voicebridge.core.jobs.manager import JobManager, JobStatus
    from voicebridge.storage import LocalArtifactStore

    store = LocalArtifactStore(tmp_path)

    class Never:
        async def run(self, job, manager):
            await manager.progress(job, "transcribing", 0.1)
            await asyncio.sleep(10)

    async def run():
        m = JobManager(Never(), store=store)
        job = await m.submit({"upload_key": "uploads/abc.wav"})
        await store.save("uploads/abc.wav", b"x")
        await store.save(f"outputs/{job.job_id}/a.srt", b"x")
        await m.cancel(job.job_id)          # cancelled while queued
        await m.start()
        await asyncio.sleep(0.2)            # worker picks it up: must not re-run it
        assert job.status is JobStatus.CANCELLED
        assert await m.delete(job.job_id)
        await m.stop()
        return job

    job = asyncio.run(run())
    assert not (tmp_path / "uploads" / "abc.wav").exists()
    assert not (tmp_path / "outputs" / job.job_id).exists()


@pytest.mark.parametrize("field,value", [("target_language", "pt/BR"),
                                          ("source_language", "../x"),
                                          ("honorifics", "shout"),
                                          ("outputs", "audio,exe")])
def test_file_api_rejects_bad_parameters_before_processing(tmp_path, field, value):
    from fastapi.testclient import TestClient

    from voicebridge.apps.gateway.main import create_app
    from voicebridge.config import AppConfig

    cfg = AppConfig(storage={"root": str(tmp_path)}, runtime={"warmup": []})
    with TestClient(create_app(cfg)) as client:
        data = {"source_language": "ja", "target_language": "en", field: value}
        r = client.post("/api/v1/files/translate",
                        files={"file": ("a.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ")}, data=data)
        assert r.status_code == 400


def test_worker_is_stopped_cleanly_when_system_memory_runs_out(monkeypatch, tmp_path):
    """The kernel OOM killer took down the whole terminal scope in a real run."""
    import sys

    from voicebridge.workers import client

    script = tmp_path / "sleepy.py"
    script.write_text(
        "import sys, json, time\n"
        "print(json.dumps({'id': 0, 'result': {'ready': True}}), flush=True)\n"
        "sys.stdin.readline(); time.sleep(60)\n")
    worker = client.ModelWorker("sleepy", str(script), sys.executable)
    worker.script = str(script)
    monkeypatch.setattr(client, "system_memory_headroom_mb", lambda: 100.0)

    async def run():
        with pytest.raises(ProviderError, match="memory nearly exhausted"):
            await worker.call("anything")
        assert not worker.running

    asyncio.run(asyncio.wait_for(run(), timeout=15))
