"""Provider factory resolution, config compatibility, honorific aliases, loops."""

from voicebridge.core.asr_validation import ASRValidationEngine
from voicebridge.core.types import HonorificPolicy, TranscriptSegment
from voicebridge.providers.factory import ProviderFactory


def test_legacy_flat_config_still_resolves():
    f = ProviderFactory({"asr": {"provider": "mock"}, "translation": {"provider": "mock",
                                                                      "latency_seconds": 0}})
    assert f.resolve("translation") == ("mock", {"latency_seconds": 0})
    assert f.get("asr").name == "mock"


def test_nested_provider_options_and_named_override():
    cfg = {
        "asr": {"provider": "whisperlivekit",
                "whisperlivekit": {"model": "openai/whisper-large-v3-turbo"},
                "qwen3_asr": {"model": "Qwen/Qwen3-ASR-1.7B"}},
        "translation": {"provider": "contextual", "backend": "mock"},
        "fallback_translation": {"provider": "opus_mt", "num_beams": 2},
    }
    f = ProviderFactory(cfg)
    assert f.resolve("asr") == ("whisperlivekit", {"model": "openai/whisper-large-v3-turbo"})
    assert f.resolve("asr", "qwen3_asr") == ("qwen3_asr", {"model": "Qwen/Qwen3-ASR-1.7B"})
    assert f.resolve("translation", "opus_mt") == ("opus_mt", {"num_beams": 2})


def test_presets_and_runtime_defaults():
    f = ProviderFactory({"translation": {"provider": "mock"},
                         "translation_presets": {"fast": {"provider": "opus_mt"}}},
                        runtime={"device": "cpu", "dtype": "auto"})
    assert f.resolve("translation", preset="fast") == ("opus_mt", {})
    assert f.resolve("translation")[1] == {"device": "cpu"}  # 'auto' is not forced


def test_instances_are_cached_by_effective_config():
    f = ProviderFactory({"translation": {"provider": "mock"}})
    assert f.get("translation") is f.get("translation")
    assert f.get("translation") is not f.get("translation", latency_seconds=0.5)


def test_honorific_policy_accepts_config_short_names():
    assert HonorificPolicy("preserve") is HonorificPolicy.PRESERVE_HONORIFICS
    assert HonorificPolicy("naturalize") is HonorificPolicy.NATURAL_ENGLISH
    assert HonorificPolicy("remove") is HonorificPolicy.REMOVE


def test_phrase_loop_hallucination_is_rejected():
    # Observed in a real run: induced by passing the previous chunk as the prompt.
    text = ("The UN is a production of the UN, and the UN is a production of the UN, "
            "and the UN is a production of the UN.")
    assert ASRValidationEngine().validate(TranscriptSegment(text, 0, 8)).reason == "repetition"
    ok = ("Now widely available throughout the archipelago, Javanese cuisine features an "
          "array of simply seasoned dishes, especially Javanese coconut sugar and spices.")
    assert ASRValidationEngine().validate(TranscriptSegment(ok, 0, 12)).valid


def test_translation_by_target_overrides_default_but_not_explicit_choice():
    f = ProviderFactory({"translation": {"provider": "contextual", "backend": "mock"},
                         "translation_by_target": {"hi": {"provider": "mock"}},
                         "translation_presets": {"fast": {"provider": "mock",
                                                          "latency_seconds": 0}}})
    assert f.translation_for("en").name == "contextual"
    assert f.translation_for("hi").name == "mock"
    assert f.translation_for("hi", name="contextual").name == "contextual"
    assert f.translation_for("hi", preset="fast").latency_seconds == 0
