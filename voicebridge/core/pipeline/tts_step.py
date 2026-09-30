"""Synthesise one translated segment and fit it to the source duration.

Dubbed speech has to fit where the original line was spoken. English is often
longer than the Japanese/Korean it translates, so a naive dub drifts further
behind the picture with every line. Fitting, when ``timing.enabled``:

1. synthesise at the requested rate;
2. if the result is longer than ``source_duration * (1 + duration_tolerance)``,
   compute the needed speed-up, capped at ``max_rate`` -- speech is never
   pushed past the point where it stops sounding natural;
3. apply it by **regenerating** at that rate when the engine controls rate
   natively and ``regenerate`` is on (file mode; costs a second synthesis), or
   by pitch-preserving **time-stretch** otherwise (realtime, and engines such
   as Qwen3-TTS Base that have no rate control);
4. whatever still overruns is left to the timeline renderer, which shifts the
   next line rather than overlapping two voices.

Too-short audio is not slowed down: silence after a short line is natural,
slowed speech is not. ``min_rate`` bounds any rate a style asks for.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from voicebridge.core.audio.dsp import float_to_pcm16, pcm16_to_float, time_stretch
from voicebridge.core.types import SynthesisedAudio, TTSRequest
from voicebridge.providers.base import TTSEngine


@dataclass
class TimingConfig:
    enabled: bool = True
    max_rate: float = 1.20
    min_rate: float = 0.85
    duration_tolerance: float = 0.15
    regenerate: bool = False

    @classmethod
    def from_dict(cls, data: dict | None) -> TimingConfig:
        data = data or {}
        return cls(**{k: type(getattr(cls(), k))(v) for k, v in data.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class FitInfo:
    source_duration: float
    raw_duration: float
    final_duration: float
    rate: float = 1.0
    method: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def mismatch(self) -> float:
        """final / source; 1.0 is a perfect fit."""
        return self.final_duration / self.source_duration if self.source_duration > 0 else 0.0


def _caps(engine: TTSEngine, language: str):
    fn = getattr(engine, "capabilities_for", None)
    return fn(language) if fn else engine.capabilities


async def synthesize_fitted(
    engine: TTSEngine, request: TTSRequest, timing: TimingConfig
) -> tuple[SynthesisedAudio, FitInfo]:
    request.speaking_rate = max(timing.min_rate, min(timing.max_rate, request.speaking_rate))
    audio = await engine.synthesize_request(request)
    source = max(0.0, request.source_end - request.source_start)
    info = FitInfo(source, audio.duration, audio.duration, request.speaking_rate)
    if not timing.enabled or source <= 0 or audio.duration <= 0:
        return audio, info
    ratio = audio.duration / source
    if ratio <= 1.0 + timing.duration_tolerance:
        return audio, info
    factor = min(ratio, timing.max_rate / max(request.speaking_rate, 1e-3))
    if factor <= 1.01:
        info.method = "capped"
        return audio, info
    caps = _caps(engine, request.language)
    if timing.regenerate and caps.speaking_rate:
        request.speaking_rate = min(timing.max_rate, request.speaking_rate * factor)
        audio = await engine.synthesize_request(request)
        info.method = "regenerate"
    else:
        stretched = await asyncio.to_thread(
            time_stretch, pcm16_to_float(audio.audio), audio.sample_rate, factor)
        audio.audio = float_to_pcm16(stretched)
        audio.duration = len(stretched) / audio.sample_rate
        info.method = "time_stretch"
    info.rate = request.speaking_rate if info.method == "regenerate" else factor
    info.final_duration = audio.duration
    return audio, info
