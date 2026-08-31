"""Low-level PCM manipulation helpers.

Deliberately dependency-light: these run per audio chunk on the hot path, so
they use numpy only, never scipy/librosa.
"""

from __future__ import annotations

import numpy as np

from voicebridge.core.types import INTERNAL_FORMAT, AudioFormat


def bytes_to_float32(data: bytes, sample_width: int = 2) -> np.ndarray:
    """Decode signed little-endian PCM to float32 in [-1, 1]."""
    if sample_width == 2:
        arr = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        arr = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        # 8-bit PCM in WAV is unsigned by convention.
        arr = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {sample_width}")
    return arr


def float32_to_bytes(arr: np.ndarray) -> bytes:
    """Encode float32 in [-1, 1] to signed 16-bit little-endian PCM."""
    clipped = np.clip(arr, -1.0, 1.0)
    # Asymmetric scaling avoids wrapping +1.0 round to -32768.
    scaled = np.where(clipped < 0, clipped * 32768.0, clipped * 32767.0)
    return scaled.astype("<i2").tobytes()


def downmix_to_mono(arr: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 1:
        return arr
    usable = (len(arr) // channels) * channels
    if usable != len(arr):
        arr = arr[:usable]
    return arr.reshape(-1, channels).mean(axis=1)


def resample_linear(arr: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampler.

    Chosen over a windowed-sinc filter because the consumer is an ASR model
    whose own front end is a mel filterbank: the aliasing introduced by linear
    interpolation sits far above the frequencies Whisper-family models weight
    heavily, and it costs a fraction of the CPU. Reference implementations do
    the same (kami-subs ``offscreen.js`` resamples this way in the browser).

    If a future benchmark shows measurable WER loss, swap this for
    ``soxr``/``torchaudio`` behind the same signature.
    """
    if src_rate == dst_rate or arr.size == 0:
        return arr
    ratio = src_rate / dst_rate
    out_len = int(np.floor(arr.size / ratio))
    if out_len <= 0:
        return np.zeros(0, dtype=np.float32)
    idx = np.arange(out_len, dtype=np.float64) * ratio
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, arr.size - 1)
    frac = (idx - lo).astype(np.float32)
    return (arr[lo] * (1.0 - frac) + arr[hi] * frac).astype(np.float32)


def convert_pcm(data: bytes, src: AudioFormat, dst: AudioFormat = INTERNAL_FORMAT) -> bytes:
    """Convert raw PCM between two formats (rate, channel count, width)."""
    if src == dst:
        return data
    arr = bytes_to_float32(data, src.sample_width)
    arr = downmix_to_mono(arr, src.channels)
    arr = resample_linear(arr, src.sample_rate, dst.sample_rate)
    if dst.channels > 1:
        arr = np.repeat(arr, dst.channels)
    return float32_to_bytes(arr)


def rms_dbfs(data: bytes, sample_width: int = 2) -> float:
    """Signal level in dBFS; used for the 'is the tab actually silent?' warning."""
    arr = bytes_to_float32(data, sample_width)
    if arr.size == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(np.square(arr))))
    if rms <= 1e-9:
        return -np.inf
    return 20.0 * float(np.log10(rms))
