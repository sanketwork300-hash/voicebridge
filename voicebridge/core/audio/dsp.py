"""Small audio helpers used by TTS timing and rendering.

* :func:`resample` -- polyphase (``scipy.signal.resample_poly``) when SciPy is
  installed, linear interpolation otherwise. TTS engines emit 22.05/24 kHz and
  the renderers work at one fixed rate, so every clip passes through here.
* :func:`time_stretch` -- tempo change without pitch change through FFmpeg's
  ``atempo`` filter (WSOLA-style), which sounds far better than a phase
  vocoder on speech. ``atempo`` accepts 0.5..100 per instance; the factors used
  for duration fitting stay within 0.8..1.3.
"""

from __future__ import annotations

import io
import math
import subprocess
import wave

import numpy as np


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly

        g = math.gcd(src_rate, dst_rate)
        return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)
    except ImportError:
        n = int(round(audio.shape[0] * dst_rate / src_rate))
        x = np.linspace(0, audio.shape[0] - 1, n)
        return np.interp(x, np.arange(audio.shape[0]), audio).astype(np.float32)


def time_stretch(audio: np.ndarray, rate: int, factor: float, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Speed speech up by ``factor`` (>1 shorter) keeping pitch."""
    if abs(factor - 1.0) < 0.01 or audio.size == 0:
        return audio
    filters = []
    f = factor
    while f > 2.0:
        filters.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        filters.append("atempo=0.5")
        f /= 0.5
    filters.append(f"atempo={f:.4f}")
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(rate),
         "-ac", "1", "-i", "pipe:0", "-filter:a", ",".join(filters), "-f", "s16le",
         "-ar", str(rate), "-ac", "1", "pipe:1"],
        input=float_to_pcm16(audio), capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"time stretch failed: {proc.stderr.decode(errors='ignore')[:300]}")
    return pcm16_to_float(proc.stdout)


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Mono float32 samples and rate from a 16-bit PCM WAV (channels averaged)."""
    with wave.open(path, "rb") as wf:
        rate, channels, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported here")
    audio = pcm16_to_float(raw)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def write_wav(path: str, audio: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(float_to_pcm16(audio))


def wav_bytes(audio: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(float_to_pcm16(audio))
    return buf.getvalue()
