"""Mock TTS that produces real, audible PCM.

It must emit genuine audio rather than silence, because the parts of the system
most likely to be wrong -- playback ordering, scheduler backlog, browser
buffering, mixing against the original track -- are only exercised by audio that
actually plays and has a realistic duration.

The waveform is a deliberately synthetic tone sequence: one short tone per word,
pitched from a hash of the word so the output is deterministic and obviously not
speech. Duration tracks text length at a plausible speaking rate, which is what
the TTS scheduler's backlog accounting needs.
"""

from __future__ import annotations

import asyncio
import math

import numpy as np

from voicebridge.core.audio.format import float32_to_bytes
from voicebridge.core.types import SynthesisedAudio
from voicebridge.providers.base import TTSCapabilities, TTSEngine
from voicebridge.providers.registry import tts_registry

SAMPLE_RATE = 16000
#: Characters per second of synthesised speech. ~14 c/s is a normal rate for
#: English narration; used so backlog maths behaves like a real voice.
CHARS_PER_SECOND = 14.0
MIN_DURATION = 0.35
MAX_DURATION = 20.0


class MockTTSEngine(TTSEngine):
    name = "mock"

    def __init__(self, latency_seconds: float = 0.02, **_: object):
        self.latency_seconds = latency_seconds

    @property
    def capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            name=self.name,
            languages=[],  # any
            voices=["mock-a", "mock-b"],
            streaming=False,
            sample_rate=SAMPLE_RATE,
            model_license="not-a-model",
            commercial_use=True,
        )

    async def synthesize(
        self,
        text: str,
        language: str,
        speaker: str | None = None,
        speed: float = 1.0,
        sequence_id: int = 0,
    ) -> SynthesisedAudio:
        if self.latency_seconds:
            await asyncio.sleep(self.latency_seconds)
        speed = max(0.25, min(4.0, speed))
        clean = (text or "").strip()
        duration = max(MIN_DURATION, min(MAX_DURATION, len(clean) / CHARS_PER_SECOND / speed))
        pcm = _render(clean, duration, base_offset=0 if speaker != "mock-b" else 60)
        return SynthesisedAudio(
            audio=pcm,
            sample_rate=SAMPLE_RATE,
            channels=1,
            duration=duration,
            sequence_id=sequence_id,
            text=clean,
            language=language,
            provider=self.name,
        )


def _render(text: str, duration: float, base_offset: int = 0) -> bytes:
    n = max(1, int(duration * SAMPLE_RATE))
    out = np.zeros(n, dtype=np.float32)
    words = text.split() or [text or "."]
    per = n // len(words)
    if per <= 0:
        per = n
    for i, word in enumerate(words):
        start = i * per
        end = min(n, start + per)
        if start >= n:
            break
        length = end - start
        if length <= 0:
            continue
        # Deterministic pitch from the word, in a comfortable speech-like band.
        freq = 180.0 + base_offset + (hash(word) % 220)
        t = np.arange(length, dtype=np.float32) / SAMPLE_RATE
        tone = 0.25 * np.sin(2.0 * math.pi * freq * t).astype(np.float32)
        # Short attack/release so consecutive tones do not click.
        env = np.ones(length, dtype=np.float32)
        ramp = max(1, int(0.01 * SAMPLE_RATE))
        ramp = min(ramp, length // 2)
        if ramp > 0:
            env[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
            env[-ramp:] = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
        # Small inter-word gap keeps it legible as separate units.
        gap = int(0.04 * SAMPLE_RATE)
        usable = max(0, length - gap)
        out[start : start + usable] = (tone * env)[:usable]
    return float32_to_bytes(out)


tts_registry.register("mock", MockTTSEngine)
