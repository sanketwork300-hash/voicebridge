"""Deterministic end-to-end demo.

Runs the complete control flow -- file audio -> ASR -> stabiliser -> segmenter
-> translation -> TTS -> subtitle and audio output -- with mock providers, so CI
exercises every stage without downloading a model or needing a GPU.

The fixture is generated rather than committed: a synthetic tone is enough to
drive the mock ASR's clock, and shipping real speech audio would mean shipping
someone's voice recording with its own licensing questions. No copyrighted
entertainment audio is distributed with this project.
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest

from voicebridge.adapters.input.file import FileAudioInput
from voicebridge.adapters.output.file import SubtitleWriter, WavFileOutput
from voicebridge.core.audio.format import float32_to_bytes
from voicebridge.core.audio.normalizer import decode_wav, encode_wav
from voicebridge.core.session.manager import SessionManager
from voicebridge.core.types import EventType, SynthesisedAudio

pytestmark = pytest.mark.asyncio

MOCK_PROVIDERS = {
    "asr": {"provider": "mock"},
    "translation": {"provider": "mock"},
    "tts": {"provider": "mock"},
}


@pytest.fixture
def fixture_audio(tmp_path: Path) -> Path:
    """30 s of synthetic tone, enough to drive several mock utterances."""
    seconds = 30
    t = np.linspace(0, seconds, 16000 * seconds, endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path = tmp_path / "fixture.wav"
    path.write_bytes(encode_wav(float32_to_bytes(tone)))
    return path


async def run_pipeline(audio_path: Path, payload: dict):
    manager = SessionManager(provider_config=MOCK_PROVIDERS)
    session = await manager.create(payload)
    await manager.start(session.session_id)

    events = []

    async def consume():
        async for event in session.pipeline.stream_events():
            events.append(event)

    consumer = asyncio.create_task(consume())
    source = FileAudioInput(str(audio_path))
    async for chunk in source.stream():
        await session.pipeline.push_audio(chunk)
    await asyncio.sleep(0.8)
    await manager.stop(session.session_id)
    await consumer
    return session, events


def by_type(events, event_type):
    return [e for e in events if e.event_type is event_type]


async def test_japanese_to_english_subtitles(fixture_audio):
    """MVP-A: Japanese tab audio -> English subtitles."""
    _, events = await run_pipeline(fixture_audio, {
        "source_language": "ja", "target_language": "en",
        "mode": "subtitles", "input": {"type": "file"},
    })
    translations = by_type(events, EventType.TRANSLATION_FINAL)
    assert translations, "no translations produced"
    first = translations[0].payload
    assert first["source_language"] == "ja"
    assert first["target_language"] == "en"
    assert first["committed"] is True
    assert first["translated_text"] == "Thank you all for coming together today."
    # Subtitle-only mode must not synthesise speech.
    assert by_type(events, EventType.TTS_AUDIO) == []


async def test_korean_to_english_with_speech(fixture_audio):
    """MVP-B: Korean -> English subtitles plus dubbed audio."""
    _, events = await run_pipeline(fixture_audio, {
        "source_language": "ko", "target_language": "en",
        "mode": "speech_and_subtitles", "input": {"type": "file"},
    })
    translations = by_type(events, EventType.TRANSLATION_FINAL)
    audio = by_type(events, EventType.TTS_AUDIO)
    assert translations and audio
    assert "Hello everyone" in translations[0].payload["translated_text"]

    # Audio must be released strictly in order.
    order = [e.payload["translation_sequence"] for e in audio]
    assert order == sorted(order)


async def test_english_to_hindi_speech(fixture_audio):
    """MVP-C: English -> Hindi subtitles and speech."""
    _, events = await run_pipeline(fixture_audio, {
        "source_language": "en", "target_language": "hi",
        "mode": "speech_and_subtitles", "input": {"type": "file"},
    })
    translations = by_type(events, EventType.TRANSLATION_FINAL)
    assert translations
    assert "मुझे लगता है" in translations[0].payload["translated_text"]


async def test_partials_precede_commitments_and_never_reach_tts(fixture_audio):
    """The core safety property: only committed text is ever spoken."""
    _, events = await run_pipeline(fixture_audio, {
        "source_language": "ja", "target_language": "en",
        "mode": "speech_and_subtitles", "input": {"type": "file"},
    })
    partials = by_type(events, EventType.ASR_PARTIAL)
    spoken = {e.payload["text"] for e in by_type(events, EventType.TTS_AUDIO)}
    committed = {e.payload["translated_text"] for e in by_type(events, EventType.TRANSLATION_FINAL)}

    assert partials, "expected provisional hypotheses"
    assert spoken, "expected synthesised speech"
    # Everything spoken came from a committed translation, never a partial.
    assert spoken <= committed


async def test_glossary_preserves_proper_nouns(fixture_audio):
    _, events = await run_pipeline(fixture_audio, {
        "source_language": "ja", "target_language": "en",
        "mode": "subtitles", "input": {"type": "file"},
        "glossary": {"terms": [
            {"source": "五条悟", "target": "Satoru Gojo", "romanized": "Satoru Gojo"},
        ]},
        "name_rendering": "preserve_and_romanize",
    })
    texts = " ".join(e.payload["translated_text"] for e in by_type(events, EventType.TRANSLATION_FINAL))
    if "五条悟" in " ".join(e.payload["source_text"] for e in by_type(events, EventType.TRANSLATION_FINAL)):
        assert "五条悟 (Satoru Gojo)" in texts


async def test_writes_subtitles_and_audio_files(fixture_audio, tmp_path):
    session, events = await run_pipeline(fixture_audio, {
        "source_language": "ko", "target_language": "en",
        "mode": "speech_and_subtitles", "input": {"type": "file"},
        "output": {"dual_subtitles": True},
    })

    subtitles = SubtitleWriter(dual=True)
    for event in by_type(events, EventType.TRANSLATION_FINAL):
        p = event.payload
        subtitles.add(p["start"], p["end"], p["translated_text"], p["source_text"])
    srt_path = tmp_path / "out.srt"
    subtitles.write(str(srt_path))

    text = srt_path.read_text(encoding="utf-8")
    assert " --> " in text
    assert "Hello everyone" in text
    assert "안녕하세요" in text          # dual subtitles keep the original

    import base64
    out = WavFileOutput(str(tmp_path / "dub.wav"))
    await out.start()
    for event in by_type(events, EventType.TTS_AUDIO):
        p = event.payload
        await out.write(SynthesisedAudio(
            audio=base64.b64decode(p["audio"]), sample_rate=p["sample_rate"],
            channels=p["channels"], duration=p["duration"],
            sequence_id=p["translation_sequence"],
        ))
    await out.stop()
    dub = decode_wav((tmp_path / "dub.wav").read_bytes())
    assert len(dub) > 16000, "dubbed audio is implausibly short"


async def test_metrics_are_populated(fixture_audio):
    session, _ = await run_pipeline(fixture_audio, {
        "source_language": "ja", "target_language": "en", "input": {"type": "file"},
    })
    snapshot = session.pipeline.metrics.snapshot()
    assert snapshot["latency"].get("translation_latency", {}).get("count", 0) > 0
    assert snapshot["gauges"]["audio_seconds_processed"] == pytest.approx(30.0, abs=0.5)


async def test_no_queue_is_unbounded(fixture_audio):
    session, _ = await run_pipeline(fixture_audio, {
        "source_language": "ja", "target_language": "en",
        "mode": "speech_and_subtitles", "input": {"type": "file"},
    })
    for name, stats in session.pipeline.queue_stats().items():
        if name == "scheduler":
            continue
        assert stats["maxsize"] > 0
