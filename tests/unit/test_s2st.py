"""SeamlessStreaming provider logic that does not need the model."""

import asyncio

import pytest

from voicebridge.providers.base import ProviderUnavailable
from voicebridge.providers.s2st.seamless_streaming import (
    SeamlessStreamingProvider,
    _asset_overrides,
    group_text_events,
)


def test_text_events_group_into_emission_timed_segments():
    events = [
        {"text": "Hello", "t": 1.0}, {"text": "everyone.", "t": 1.6},
        {"text": "Today", "t": 5.0}, {"text": "we", "t": 5.3}, {"text": "start", "t": 5.9},
    ]
    segs = group_text_events(events, "ja", "en", total=8.0)
    assert [s.translated_text for s in segs] == ["Hello everyone.", "Today we start"]
    assert segs[0].start_time == 1.0 and segs[0].end_time <= segs[1].start_time
    assert all(s.metadata["timing"] == "emission_offset" for s in segs)


def test_asset_overrides_point_fairseq2_at_local_snapshot(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    for name in ("seamless_streaming_unity.pt", "spm_char_lang38_tc.model",
                 "seamless_streaming_monotonic_decoder.pt", "vocoder_v2.pt"):
        (snap / name).write_bytes(b"x")
    out = _asset_overrides(snap, tmp_path / "cards")
    text = (out / "voicebridge_seamless.yaml").read_text()
    assert "name: seamless_streaming_unity@user" in text
    assert f'checkpoint: "file://{snap}/seamless_streaming_unity.pt"' in text
    assert "vocoder_v2@user" in text


def test_disabled_provider_refuses_to_run():
    p = SeamlessStreamingProvider(enabled=False, model="")
    with pytest.raises(ProviderUnavailable, match="disabled"):
        asyncio.run(p.translate_file("x.wav", "ja", "en"))
    assert p.capabilities.commercial_use is False
    assert "CC-BY-NC" in p.capabilities.model_license


def test_unsupported_target_language_is_explicit():
    p = SeamlessStreamingProvider(enabled=True, model="")
    with pytest.raises(ProviderUnavailable, match="cannot output"):
        asyncio.run(p.translate_file("x.wav", "ja", "xx"))
