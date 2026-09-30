"""Subtitles, timeline rendering, upload validation, storage keys, job lifecycle."""

import asyncio

import numpy as np
import pytest

from voicebridge.api.uploads import sanitize_filename, sniff
from voicebridge.core.jobs.errors import JobCancelled, PipelineError
from voicebridge.core.jobs.manager import JobManager, JobStatus
from voicebridge.core.types import TranslationSegment
from voicebridge.media.renderer import TimelineRenderer
from voicebridge.media.subtitles import render_srt, render_vtt
from voicebridge.storage import LocalArtifactStore
from voicebridge.storage.base import safe_key


def segs():
    return [
        TranslationSegment("a", 2.1, 4.8, "何してるの", "What are you doing?"),
        TranslationSegment("b", 5.1, 7.6, "来るなって言ったでしょ", "I told you not to come here."),
        TranslationSegment("c", 9.0, 9.0, None, "no timing", metadata={"timestamps_available": False}),
    ]


def test_srt_uses_segment_timestamps():
    srt = render_srt(segs())
    assert "1\n00:00:02,100 --> 00:00:04,800\nWhat are you doing?" in srt
    assert "2\n00:00:05,100 --> 00:00:07,600\nI told you not to come here." in srt
    assert "no timing" not in srt  # never invents times


def test_vtt_and_dual_subtitles():
    vtt = render_vtt(segs(), dual=True)
    assert vtt.startswith("WEBVTT")
    assert "00:00:02.100 --> 00:00:04.800\nWhat are you doing?\n何してるの" in vtt


def test_overlapping_cues_are_trimmed():
    s = [TranslationSegment("a", 0, 5, "x", "one"), TranslationSegment("b", 3, 6, "y", "two")]
    assert "00:00:00,000 --> 00:00:02,990" in render_srt(s)


def test_timeline_places_clips_at_source_time_without_overlap(tmp_path):
    r = TimelineRenderer(10.0, 1000, tmp_path)
    p1 = r.place("a", np.ones(3000, np.float32) * 0.5, 1000, 1.0)   # 1.0 - 4.0
    p2 = r.place("b", np.ones(1000, np.float32) * 0.5, 1000, 2.0)   # overlaps -> shifted
    p3 = r.place("c", np.ones(1000, np.float32) * 0.5, 1000, 7.0)   # back on time
    assert p1.actual_start == 1.0 and p1.drift == 0
    assert p2.actual_start == pytest.approx(4.05) and p2.drift > 0
    assert p3.actual_start == 7.0
    out = r.write_wav(tmp_path / "t.wav")
    import soundfile as sf

    audio, rate = sf.read(out)
    assert rate == 1000 and len(audio) == 10000
    assert abs(audio[500]) < 1e-3 and abs(audio[1500]) > 0.4
    r.close()


def test_upload_sniffing_and_filename_sanitising():
    assert sniff(b"RIFF\x00\x00\x00\x00WAVEfmt ") == "wav"
    assert sniff(b"\x00\x00\x00\x20ftypisom") == "isobmff"
    assert sniff(b"\x1a\x45\xdf\xa3") == "matroska"
    assert sniff(b"<html><body>") is None
    assert sanitize_filename("../../etc/pass wd.MP4") == "pass_wd.mp4"
    assert sanitize_filename(None) == "media"


def test_storage_rejects_path_traversal(tmp_path):
    store = LocalArtifactStore(tmp_path)
    for bad in ("../x", "outputs/../../x", "/etc/passwd", "a//b"):
        with pytest.raises(ValueError):
            safe_key(bad)
    asyncio.run(store.save("outputs/job/a.srt", b"x"))
    assert asyncio.run(store.exists("outputs/job/a.srt"))


class SlowExecutor:
    def __init__(self, fail=None):
        self.fail = fail

    async def run(self, job, manager):
        for i in range(50):
            await manager.progress(job, "transcribing", i / 50)
            await asyncio.sleep(0.01)
        if self.fail:
            raise self.fail
        return {"out.srt": "outputs/x/out.srt"}


def test_job_completes_with_monotonic_progress():
    async def run():
        m = JobManager(SlowExecutor())
        await m.start()
        job = await m.submit({"x": 1})
        seen = []
        async for ev in m.events(job.job_id):
            seen.append(ev["progress"])
        await m.stop()
        return job, seen

    job, seen = asyncio.run(run())
    assert job.status is JobStatus.COMPLETED and job.progress == 100
    assert seen == sorted(seen)


def test_job_cancellation_is_checked_by_worker():
    async def run():
        m = JobManager(SlowExecutor())
        await m.start()
        job = await m.submit({})
        await asyncio.sleep(0.1)
        await m.cancel(job.job_id)
        async for _ in m.events(job.job_id):
            pass
        await m.stop()
        return job

    job = asyncio.run(run())
    assert job.status is JobStatus.CANCELLED and job.error["code"] == "JOB_CANCELLED"


def test_structured_errors_and_no_stack_traces():
    async def run(exc):
        m = JobManager(SlowExecutor(fail=exc))
        await m.start()
        job = await m.submit({})
        async for _ in m.events(job.job_id):
            pass
        await m.stop()
        return job

    job = asyncio.run(run(PipelineError("translating", "TRANSLATION_TIMEOUT", "slow", True)))
    assert job.error == {"stage": "translating", "code": "TRANSLATION_TIMEOUT",
                         "message": "slow", "retryable": True}
    job = asyncio.run(run(RuntimeError("secret internal detail")))
    assert job.error["code"] == "INTERNAL_ERROR" and "secret" not in job.error["message"]
    assert isinstance(JobCancelled(), PipelineError)


def test_file_chunking_merges_short_gaps_and_splits_long_regions_at_a_pause(tmp_path):
    import soundfile as sf

    from voicebridge.core.pipeline.file_pipeline import FilePipeline, FilePipelineConfig
    from voicebridge.core.types import SpeechDecision
    from voicebridge.media.normalizer import NormalizedAudio
    from voicebridge.providers.registry import ProviderSet

    t = np.arange(16000 * 40) / 16000
    audio = 0.3 * np.sin(2 * np.pi * 200 * t)
    audio[int(16000 * 24.0):int(16000 * 24.4)] = 0.0  # a breath at 24.0-24.4 s
    sf.write(tmp_path / "a.wav", audio.astype(np.float32), 16000, subtype="PCM_16")
    na = NormalizedAudio(tmp_path / "a.wav")

    async def noop(*_a, **_k):
        return None

    fp = FilePipeline(ProviderSet(None, None, None), FilePipelineConfig(), noop, noop)
    d = [SpeechDecision(True, "DIALOGUE", 0.9, 1.0, 5.0), SpeechDecision(True, "DIALOGUE", 0.9, 5.4, 8.0),
         SpeechDecision(False, "MUSIC", 0.9, 8.0, 10.0), SpeechDecision(True, "DIALOGUE", 0.9, 10.0, 39.0)]
    chunks = fp.chunk(d, na)
    assert chunks[0] == (1.0, 8.0)                      # short gap merged
    assert 24.0 <= chunks[1][1] <= 24.4                 # long region cut inside the pause
    assert chunks[-1][1] == 39.0 and all(b - a <= 28.0 for a, b in chunks)
