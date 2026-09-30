"""File translation through the HTTP API with mock providers.

These prove the job/API/media plumbing (upload -> FFmpeg -> pipeline -> render
-> download) and the failure matrix. They use mock ASR/translation/TTS, so they
say nothing about recognition or translation quality; the real-model runs are
recorded in docs/evaluation.md.
"""

import shutil
import subprocess
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voicebridge.config import AppConfig, SecurityConfig
from voicebridge.core.types import SpeechTranslationResult, TranslationSegment
from voicebridge.providers.base import S2STCapabilities, S2STProvider
from voicebridge.providers.registry import s2st_registry

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="FFmpeg is required for file translation")


def _config(tmp_path, **media):
    return AppConfig(
        security=SecurityConfig(),
        providers={"asr": {"provider": "mock"}, "translation": {"provider": "mock"},
                   "tts": {"provider": "mock"},
                   "s2st": {"fake_s2st": {"enabled": True},
                            "seamless_streaming": {"enabled": False}}},
        storage={"root": str(tmp_path / "storage")},
        media={"max_file_size_mb": 5, **media},
        runtime={"device": "cpu", "dtype": "auto", "warmup": []},
    )


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "clip.wav"
    t = np.arange(16000 * 6) / 16000
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    import soundfile as sf

    sf.write(path, audio, 16000, subtype="PCM_16")
    return path


@pytest.fixture
def video_file(tmp_path, audio_file):
    path = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=320x240:rate=25:duration=6", "-i", str(audio_file),
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def _wait(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def _submit(client, path, **form):
    with open(path, "rb") as fh:
        return client.post("/api/v1/files/translate", files={"file": (path.name, fh)},
                           data={"source_language": "ja", "target_language": "en", **form})


def _client(tmp_path, **media):
    from voicebridge.apps.gateway.main import create_app

    return TestClient(create_app(_config(tmp_path, **media)))


def test_audio_file_to_translated_audio_and_subtitles(tmp_path, audio_file):
    with _client(tmp_path) as client:
        r = _submit(client, audio_file, output_format="mp3")
        assert r.status_code == 202
        job = _wait(client, r.json()["job_id"])
        assert job["status"] == "completed", job["error"]
        assert set(job["outputs"]) == {"clip.en.mp3", "clip.en.srt", "clip.en.vtt"}
        srt = client.get(f"/api/v1/jobs/{job['job_id']}/outputs/clip.en.srt").text
        assert "-->" in srt
        mp3 = client.get(f"/api/v1/jobs/{job['job_id']}/outputs/clip.en.mp3")
        assert mp3.status_code == 200 and len(mp3.content) > 1000
        assert job["log"]["engine"] == "cascade" and job["log"]["media"]["duration"] > 5


def test_video_file_to_translated_video(tmp_path, video_file):
    with _client(tmp_path) as client:
        job = _wait(client, _submit(client, video_file, audio_mode="keep").json()["job_id"])
        assert job["status"] == "completed", job["error"]
        assert "clip.en.mp4" in job["outputs"] and "clip.en.srt" in job["outputs"]
        out = tmp_path / "out.mp4"
        out.write_bytes(client.get(f"/api/v1/jobs/{job['job_id']}/outputs/clip.en.mp4").content)
        streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                  "stream=codec_type,codec_name,width", "-of", "csv=p=0", str(out)],
                                 capture_output=True, text=True).stdout.split()
        assert "h264,video,320" in streams  # video stream copied, resolution preserved
        assert sum(1 for s in streams if "audio" in s) == 2  # dub + original (keep mode)
        assert any("mov_text" in s for s in streams)


class FakeS2ST(S2STProvider):
    """Stands in for an S2ST engine to prove the shared output layer."""

    name = "fake_s2st"

    def __init__(self, enabled=False, **_):
        self.enabled = enabled

    @property
    def capabilities(self):
        return S2STCapabilities(name=self.name, model_license="test")

    async def translate_stream(self, audio_stream, source_language, target_language):
        yield  # pragma: no cover

    async def translate_file(self, audio_path, source_language, target_language, workdir=None,
                             progress=None, emit=None, cancel=None):
        import soundfile as sf

        out = f"{workdir}/fake.wav"
        sf.write(out, np.zeros(16000 * 6, np.float32), 16000)
        return SpeechTranslationResult(out, source_language, target_language, [
            TranslationSegment("s1", 1.0, 2.0, translated_text="hello",
                               metadata={"timestamps_available": True})], 6.0)


s2st_registry.register("fake_s2st", FakeS2ST)


def test_s2st_engine_uses_the_same_output_layer(tmp_path, video_file):
    with _client(tmp_path) as client:
        job = _wait(client, _submit(client, video_file, engine="seamless_m4t_v2").json()["job_id"])
        # seamless_m4t_v2 is not enabled in this config: explicit failure, no fallback.
        assert job["status"] == "failed" and job["error"]["code"] == "SEAMLESS_MODEL_ERROR"
    cfg_dir = tmp_path / "b"
    from voicebridge.apps.gateway.main import create_app

    config = _config(cfg_dir)
    with TestClient(create_app(config)) as client:
        client.app.state.factory.config["s2st"]["seamless_m4t_v2"] = {"enabled": True}
        # Route the engine name to the fake provider to exercise rendering.
        s2st_registry.register("seamless_m4t_v2", FakeS2ST)
        try:
            job = _wait(client, _submit(client, video_file, engine="seamless_m4t_v2").json()["job_id"])
        finally:
            from voicebridge.providers.s2st.seamless import SeamlessM4Tv2Provider

            s2st_registry.register("seamless_m4t_v2", SeamlessM4Tv2Provider)
        assert job["status"] == "completed", job["error"]
        assert {"clip.en.mp4", "clip.en.srt", "clip.en.wav"} <= set(job["outputs"])
        assert job["log"]["engine"] == "seamless_m4t_v2"


def test_disabled_seamless_fails_explicitly_without_fallback(tmp_path, audio_file):
    with _client(tmp_path) as client:
        job = _wait(client, _submit(client, audio_file, engine="seamless_streaming").json()["job_id"])
        assert job["status"] == "failed"
        assert job["error"]["code"] == "SEAMLESS_MODEL_ERROR"
        assert "disabled" in job["error"]["message"]
        assert "models" not in job["log"] or "asr" not in (job["log"].get("models") or {})


def test_corrupted_and_disguised_files(tmp_path):
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 100)
    fake = tmp_path / "page.mp4"
    fake.write_bytes(b"<html>not a video</html>")
    exe = tmp_path / "tool.exe"
    exe.write_bytes(b"MZ")
    with _client(tmp_path) as client:
        job = _wait(client, _submit(client, bad).json()["job_id"])
        assert job["status"] == "failed" and job["error"]["code"] in ("CORRUPTED_MEDIA",
                                                                       "NO_AUDIO_STREAM")
        assert _submit(client, fake).status_code == 415
        assert _submit(client, exe).status_code == 400


def test_size_limit_and_quota(tmp_path, audio_file):
    big = tmp_path / "big.wav"
    big.write_bytes(audio_file.read_bytes() + b"\x00" * (6 * 1024 * 1024))
    with _client(tmp_path) as client:
        assert _submit(client, big).status_code == 413
    with _client(tmp_path / "q", disk_quota_gb=0.000001) as client:
        assert _submit(client, audio_file).status_code == 507


def test_missing_ffmpeg_gives_clear_error(tmp_path, audio_file, monkeypatch):
    from voicebridge.media import probe

    monkeypatch.setattr(probe, "_find", lambda tool: None)
    with _client(tmp_path) as client:
        job = _wait(client, _submit(client, audio_file).json()["job_id"])
        assert job["status"] == "failed" and job["error"]["code"] == "FFMPEG_MISSING"
        assert "FFmpeg is required" in job["error"]["message"]


def test_unknown_job_and_output_names_are_404(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/api/v1/jobs/nope").status_code == 404
        assert client.get("/api/v1/jobs/nope/outputs/../../etc/passwd").status_code == 404


def test_engines_endpoint_lists_license_notice(tmp_path):
    with _client(tmp_path) as client:
        engines = {e["id"]: e for e in client.get("/api/v1/models/engines").json()["engines"]}
        assert engines["cascade"]["enabled"]
        assert "CC-BY-NC" in engines["seamless_streaming"]["license"]
        assert not engines["seamless_streaming"]["enabled"]


def test_live_seamless_session_is_refused_when_disabled(tmp_path):
    with _client(tmp_path) as client:
        r = client.post("/api/v1/realtime/sessions", json={
            "source_language": "ja", "target_language": "en", "engine": "seamless_streaming"})
        assert r.status_code == 400 and "disabled" in r.json()["detail"]
        ok = client.post("/api/v1/realtime/sessions", json={
            "source_language": "ja", "target_language": "en", "engine": "cascade"})
        assert ok.status_code == 201
