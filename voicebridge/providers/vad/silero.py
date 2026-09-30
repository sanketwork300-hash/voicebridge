"""Silero VAD, via the ONNX model that ships inside faster-whisper.

API used (verified against faster-whisper 1.2.1, ``faster_whisper/vad.py``)
--------------------------------------------------------------------------
``get_vad_model()``
    ``lru_cache``-d loader returning ``SileroVADModel`` for the bundled
    ``assets/silero_vad_v6.onnx``. Calling the model with a float32 mono array
    whose length is a multiple of 512 returns one speech probability per
    512-sample (32 ms) window.
``get_speech_timestamps(audio, VadOptions(...))``
    Converts those probabilities into ``[{"start": sample, "end": sample}]``
    regions with hysteresis, minimum-silence merging and padding.

faster-whisper is already a dependency of the WhisperLiveKit ASR extra, so this
adds no new install. Silero VAD itself is MIT licensed.

Two entry points, because the two pipeline modes need different things:

* :meth:`detect` scores one realtime chunk (``VADProvider`` contract);
* :meth:`speech_regions` segments a whole normalised file for the file
  pipeline, where hysteresis and padding over the full signal are what give
  dialogue-shaped regions rather than 200 ms fragments.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from voicebridge.core.types import AudioChunk, AudioEventType, SpeechDecision
from voicebridge.providers.base import ProviderUnavailable, VADCapabilities, VADProvider
from voicebridge.providers.registry import vad_registry

WINDOW = 512  # samples per Silero decision at 16 kHz


class SileroVADProvider(VADProvider):
    name = "silero"

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 200,
        max_speech_duration_s: float = 30.0,
        **_: object,
    ):
        self.threshold = float(threshold)
        self.min_speech_duration_ms = int(min_speech_duration_ms)
        self.min_silence_duration_ms = int(min_silence_duration_ms)
        self.speech_pad_ms = int(speech_pad_ms)
        self.max_speech_duration_s = float(max_speech_duration_s)
        self._model: Any = None

    @property
    def capabilities(self) -> VADCapabilities:
        return VADCapabilities(name=self.name, streaming=True)

    def _load(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper.vad import get_vad_model
            except ImportError as exc:
                raise ProviderUnavailable(
                    "Silero VAD uses the ONNX model bundled with faster-whisper. Install "
                    "the ASR extra: pip install 'voicebridge[asr-whisperlivekit]'"
                ) from exc
            self._model = get_vad_model()
        return self._model

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        """Per-32 ms speech probability for float32 mono 16 kHz audio."""
        if audio.size == 0:
            return np.zeros(0, dtype=np.float32)
        model = self._load()
        pad = (-audio.shape[0]) % WINDOW
        padded = np.pad(audio.astype(np.float32, copy=False), (0, pad))
        return np.asarray(model(padded), dtype=np.float32).reshape(-1)

    async def detect(self, chunk: AudioChunk) -> SpeechDecision:
        audio = _pcm16_to_float(chunk.data)
        probs = await asyncio.to_thread(self.probabilities, audio)
        peak = float(probs.max()) if probs.size else 0.0
        voiced = float((probs >= self.threshold).mean()) if probs.size else 0.0
        is_speech = peak >= self.threshold
        return SpeechDecision(
            should_transcribe=is_speech,
            event_type=(AudioEventType.DIALOGUE if is_speech else AudioEventType.SILENCE).value,
            confidence=peak if is_speech else 1.0 - peak,
            start_time=chunk.start,
            end_time=chunk.end,
            reason=f"silero peak={peak:.2f} voiced={voiced:.2f}",
        )

    def speech_regions(self, audio: np.ndarray, sample_rate: int = 16000) -> list[dict[str, float]]:
        """Speech regions in seconds with the mean probability inside each."""
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps
        except ImportError as exc:  # pragma: no cover - same dependency as _load
            raise ProviderUnavailable("faster-whisper is required for Silero VAD") from exc
        options = VadOptions(
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            max_speech_duration_s=self.max_speech_duration_s,
        )
        stamps = get_speech_timestamps(audio.astype(np.float32, copy=False), options, sample_rate)
        probs = self.probabilities(audio)
        regions = []
        for stamp in stamps:
            a, b = stamp["start"] // WINDOW, max(stamp["start"] // WINDOW + 1, stamp["end"] // WINDOW)
            window = probs[a:b]
            regions.append(
                {
                    "start": stamp["start"] / sample_rate,
                    "end": stamp["end"] / sample_rate,
                    "confidence": float(window.mean()) if window.size else 0.0,
                }
            )
        return regions


def _pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


vad_registry.register("silero", SileroVADProvider)
