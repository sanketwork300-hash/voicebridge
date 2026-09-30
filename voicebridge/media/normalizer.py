"""Normalised audio access for the file pipeline.

:func:`normalize_audio` produces the pipeline format (16 kHz, mono, PCM s16le
WAV). :class:`NormalizedAudio` then serves float32 regions of it by seeking in
the file, which is what keeps memory flat for long media.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from voicebridge.media.extractor import extract_audio

SAMPLE_RATE = 16000


async def normalize_audio(input_path: str | Path, output_path: str | Path) -> Path:
    return await extract_audio(input_path, output_path, SAMPLE_RATE)


class NormalizedAudio:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with wave.open(self.path, "rb") as wf:
            if (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                raise ValueError(f"{path} is not 16 kHz mono 16-bit PCM")
            self.frames = wf.getnframes()

    @property
    def duration(self) -> float:
        return self.frames / SAMPLE_RATE

    def read(self, start: float = 0.0, end: float | None = None) -> np.ndarray:
        a = max(0, int(start * SAMPLE_RATE))
        b = self.frames if end is None else min(self.frames, int(end * SAMPLE_RATE))
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        with wave.open(self.path, "rb") as wf:
            wf.setpos(a)
            raw = wf.readframes(b - a)
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    def blocks(self, seconds: float = 600.0, overlap: float = 0.0):
        """Yield ``(offset, samples)`` blocks of at most ``seconds``."""
        t = 0.0
        while t < self.duration:
            end = min(self.duration, t + seconds)
            yield t, self.read(max(0.0, t - overlap), end)
            t = end
