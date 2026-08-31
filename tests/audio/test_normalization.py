"""Audio normalisation: the contract every input adapter must satisfy."""

import numpy as np
import pytest

from voicebridge.core.audio.format import (
    bytes_to_float32,
    convert_pcm,
    downmix_to_mono,
    float32_to_bytes,
    resample_linear,
    rms_dbfs,
)
from voicebridge.core.audio.normalizer import decode_wav, encode_wav
from voicebridge.core.types import INTERNAL_FORMAT, AudioFormat


def sine(seconds: float, rate: int, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_roundtrip_float_bytes_preserves_signal():
    original = sine(0.1, 16000)
    restored = bytes_to_float32(float32_to_bytes(original))
    # 16-bit quantisation error is bounded by 1 LSB.
    assert np.max(np.abs(original - restored)) < 1.5 / 32768


def test_positive_full_scale_does_not_wrap():
    """+1.0 must clamp to +32767, not wrap to -32768 (an audible click)."""
    data = float32_to_bytes(np.array([1.0, -1.0, 0.999], dtype=np.float32))
    samples = np.frombuffer(data, dtype="<i2")
    assert samples[0] == 32767
    assert samples[1] == -32768
    assert samples[2] > 0


def test_downmix_averages_channels():
    interleaved = np.array([1.0, 0.0, 0.5, 0.5], dtype=np.float32)
    assert np.allclose(downmix_to_mono(interleaved, 2), [0.5, 0.5])


def test_downmix_handles_truncated_frame():
    """A frame cut mid-sample must not raise or misalign the channels."""
    interleaved = np.array([1.0, 0.0, 0.5], dtype=np.float32)  # 1.5 frames
    assert len(downmix_to_mono(interleaved, 2)) == 1


@pytest.mark.parametrize("src_rate", [8000, 22050, 44100, 48000])
def test_resample_produces_expected_length(src_rate):
    signal = sine(1.0, src_rate)
    out = resample_linear(signal, src_rate, 16000)
    assert abs(len(out) - 16000) <= 1


def test_resample_preserves_level():
    """Resampling must not change the perceived loudness."""
    signal = sine(1.0, 48000, freq=220.0, amp=0.5)
    out = resample_linear(signal, 48000, 16000)
    before = float(np.sqrt(np.mean(signal**2)))
    after = float(np.sqrt(np.mean(out**2)))
    assert abs(before - after) < 0.01


def test_convert_pcm_48k_stereo_to_internal():
    stereo = np.repeat(sine(1.0, 48000), 2)
    src = AudioFormat(sample_rate=48000, channels=2)
    out = convert_pcm(float32_to_bytes(stereo), src, INTERNAL_FORMAT)
    assert len(out) == INTERNAL_FORMAT.bytes_per_second  # exactly 1 second


def test_convert_pcm_is_identity_for_matching_format():
    payload = b"\x01\x02" * 100
    assert convert_pcm(payload, INTERNAL_FORMAT, INTERNAL_FORMAT) is payload


def test_wav_roundtrip():
    pcm = float32_to_bytes(sine(0.5, 16000))
    assert decode_wav(encode_wav(pcm)) == pcm


def test_wav_decode_resamples_foreign_rate():
    pcm = float32_to_bytes(sine(1.0, 44100))
    wav = encode_wav(pcm, AudioFormat(sample_rate=44100, channels=1))
    assert len(decode_wav(wav)) == pytest.approx(INTERNAL_FORMAT.bytes_per_second, abs=4)


def test_rms_of_silence_is_negative_infinity():
    assert rms_dbfs(b"\x00\x00" * 100) == float("-inf")


def test_empty_input_is_safe():
    assert convert_pcm(b"", AudioFormat(sample_rate=44100), INTERNAL_FORMAT) == b""
