"""Conservative audio event classifier.

This deliberately marks ambiguous non-silent audio as DIALOGUE with moderate
confidence, because the gate should avoid silently discarding low-confidence
speech. Model-backed classifiers can replace it behind the same interface.
"""

from __future__ import annotations

import audioop

from voicebridge.core.types import AudioChunk, AudioEventType, SpeechDecision
from voicebridge.providers.base import AudioEventCapabilities, AudioEventClassifier
from voicebridge.providers.registry import audio_event_registry

EVENTS = [event.value for event in AudioEventType]


class EnergyAudioEventClassifier(AudioEventClassifier):
    name = "energy"

    def __init__(self, silence_rms: int = 80, loud_rms: int = 12000, **_: object):
        self.silence_rms = int(silence_rms)
        self.loud_rms = int(loud_rms)

    @property
    def capabilities(self) -> AudioEventCapabilities:
        return AudioEventCapabilities(name=self.name, event_types=EVENTS)

    async def classify(self, chunk: AudioChunk) -> SpeechDecision:
        rms = audioop.rms(chunk.data, chunk.fmt.sample_width) if chunk.data else 0
        if rms < self.silence_rms:
            event_type = AudioEventType.SILENCE
            confidence = 0.95
            transcribe = False
        elif rms > self.loud_rms:
            event_type = AudioEventType.UNKNOWN
            confidence = 0.45
            transcribe = True
        else:
            event_type = AudioEventType.DIALOGUE
            confidence = min(0.95, max(0.50, rms / max(1, self.loud_rms)))
            transcribe = True
        return SpeechDecision(
            should_transcribe=transcribe,
            event_type=event_type.value,
            confidence=confidence,
            start_time=chunk.start,
            end_time=chunk.end,
            reason=f"rms={rms}",
        )


audio_event_registry.register("energy", EnergyAudioEventClassifier)
audio_event_registry.register("basic", EnergyAudioEventClassifier)

