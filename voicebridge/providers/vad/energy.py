"""Small dependency-free VAD used as VoiceBridge's conservative speech gate default.

It is not a replacement for Silero or a model VAD. It exists so the pipeline has
a concrete VAD contract in base installs and tests, while real deployments can
swap in a stronger provider through the same interface.
"""

from __future__ import annotations

import audioop

from voicebridge.core.types import AudioChunk, AudioEventType, SpeechDecision
from voicebridge.providers.base import VADCapabilities, VADProvider
from voicebridge.providers.registry import vad_registry


class EnergyVADProvider(VADProvider):
    name = "energy"

    def __init__(self, rms_threshold: int = 120, **_: object):
        self.rms_threshold = int(rms_threshold)

    @property
    def capabilities(self) -> VADCapabilities:
        return VADCapabilities(name=self.name, streaming=True)

    async def detect(self, chunk: AudioChunk) -> SpeechDecision:
        rms = audioop.rms(chunk.data, chunk.fmt.sample_width) if chunk.data else 0
        confidence = min(1.0, rms / max(1, self.rms_threshold * 4))
        is_speech_candidate = rms >= self.rms_threshold
        return SpeechDecision(
            should_transcribe=is_speech_candidate,
            event_type=AudioEventType.DIALOGUE if is_speech_candidate else AudioEventType.SILENCE,
            confidence=confidence,
            start_time=chunk.start,
            end_time=chunk.end,
            reason=f"rms={rms}",
        )


vad_registry.register("energy", EnergyVADProvider)

