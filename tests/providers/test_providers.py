"""Provider registry, capabilities and failure behaviour."""

import pytest

from voicebridge.core.types import AudioChunk, EventType
from voicebridge.providers.base import ProviderError, ProviderUnavailable
from voicebridge.providers.registry import (
    asr_registry,
    translation_registry,
    tts_registry,
)

pytestmark = pytest.mark.asyncio


async def test_all_builtin_providers_are_registered():
    assert {"mock", "whisperlivekit"} <= set(asr_registry.names())
    assert {"mock", "opus_mt", "indictrans2", "nllb"} <= set(translation_registry.names())
    assert {"mock", "piper"} <= set(tts_registry.names())


async def test_unknown_provider_raises_with_available_names():
    with pytest.raises(ProviderUnavailable) as exc:
        translation_registry.create("does-not-exist")
    assert "available" in str(exc.value)


async def test_registry_never_imports_heavy_dependencies():
    """Constructing a provider must not require torch to be installed."""
    for name in translation_registry.names():
        translation_registry.create(name)   # must not raise


class TestLicensing:
    """Licence metadata is part of the provider contract, not documentation."""

    async def test_nllb_declares_non_commercial(self):
        caps = translation_registry.create("nllb").capabilities
        assert caps.model_license == "CC-BY-NC-4.0"
        assert caps.commercial_use is False

    async def test_nllb_refuses_to_load_without_acknowledgement(self, monkeypatch):
        monkeypatch.delenv("VOICEBRIDGE_ALLOW_NC_MODELS", raising=False)
        engine = translation_registry.create("nllb")
        with pytest.raises(ProviderError) as exc:
            await engine.translate("hello", "ja", "en")
        assert "CC-BY-NC-4.0" in str(exc.value)

    async def test_permissive_providers_allow_commercial_use(self):
        assert translation_registry.create("opus_mt").capabilities.commercial_use is True
        assert translation_registry.create("indictrans2").capabilities.commercial_use is True


class TestCapabilityHonesty:
    """Capabilities must not over-claim: the pipeline routes on them."""

    async def test_indictrans2_rejects_japanese(self):
        engine = translation_registry.create("indictrans2")
        assert engine.supports("en", "hi") is True
        assert engine.supports("en", "ja") is False
        assert engine.supports("ja", "en") is False

    async def test_indictrans2_raises_a_helpful_error_for_unsupported_pair(self):
        engine = translation_registry.create("indictrans2")
        with pytest.raises(ProviderUnavailable) as exc:
            await engine.translate("こんにちは", "ja", "en")
        assert "opus_mt" in str(exc.value)

    async def test_opus_mt_advertises_verified_tier1_pairs(self):
        caps = translation_registry.create("opus_mt").capabilities
        assert ("ja", "en") in caps.pairs
        assert ("ko", "en") in caps.pairs


class TestMockProviders:
    async def test_mock_asr_emits_partials_then_stable(self):
        engine = asr_registry.create("mock")
        await engine.start_session("s", "en")
        events = []

        async def consume():
            async for event in engine.get_events("s"):
                events.append(event)

        import asyncio
        task = asyncio.create_task(consume())
        for _ in range(60):
            await engine.push_audio("s", AudioChunk(data=b"\x00\x00" * 1600))
        await asyncio.sleep(0.05)
        await engine.stop_session("s")
        await task

        kinds = [e.event_type for e in events]
        assert EventType.ASR_PARTIAL in kinds
        assert EventType.ASR_STABLE in kinds
        assert EventType.LANGUAGE_DETECTED in kinds
        # Partials must precede the commitment they build up to.
        assert kinds.index(EventType.ASR_PARTIAL) < kinds.index(EventType.ASR_STABLE)

    async def test_mock_translation_is_deterministic(self):
        engine = translation_registry.create("mock")
        a = await engine.translate("五条悟の領域展開は本当に印象的でした。", "ja", "en")
        b = await engine.translate("五条悟の領域展開は本当に印象的でした。", "ja", "en")
        assert a.translated_text == b.translated_text
        assert a.confidence == 1.0

    async def test_mock_translation_marks_unknown_text(self):
        engine = translation_registry.create("mock")
        result = await engine.translate("not in the phrase table", "ja", "en")
        assert result.translated_text.startswith("[en]")
        assert result.confidence == 0.0

    async def test_mock_tts_returns_real_audio_of_plausible_duration(self):
        engine = tts_registry.create("mock")
        audio = await engine.synthesize("Thank you all for coming today.", "en", sequence_id=1)
        assert audio.duration > 0.5
        assert len(audio.audio) == int(audio.duration * audio.sample_rate) * 2
        assert audio.audio != b"\x00" * len(audio.audio)   # not silence

    async def test_mock_tts_speed_shortens_output(self):
        engine = tts_registry.create("mock")
        text = "A reasonably long sentence to synthesise for the speed test."
        slow = await engine.synthesize(text, "en", speed=0.5)
        fast = await engine.synthesize(text, "en", speed=2.0)
        assert fast.duration < slow.duration

    async def test_mock_tts_handles_empty_text(self):
        engine = tts_registry.create("mock")
        audio = await engine.synthesize("   ", "en")
        assert audio.duration > 0     # clamped to the minimum, never negative
