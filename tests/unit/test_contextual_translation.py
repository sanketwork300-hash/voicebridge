"""Context window, contextual provider prompting and the shared translate step."""

import asyncio

from voicebridge.core.context.glossary import SessionGlossary
from voicebridge.core.context.translation_context import GlobalTranslationContext, estimate_tokens
from voicebridge.core.context.window import ContextEntry
from voicebridge.core.pipeline.translate_step import translate_segment
from voicebridge.core.types import HonorificPolicy
from voicebridge.providers.translation.contextual import ContextualTranslationProvider, clean_output
from voicebridge.providers.translation.llm_backends import ChatBackend


class Recorder(ChatBackend):
    name = "recorder"
    model = "test"

    def __init__(self, reply="Don't tell him about it."):
        self.reply = reply
        self.calls = []

    async def complete(self, system, user, max_tokens):
        self.calls.append((system, user))
        return self.reply


def provider(reply="Don't tell him about it."):
    p = ContextualTranslationProvider(backend="mock")
    p.backend = Recorder(reply)
    return p


def test_context_window_is_bounded_and_compressed():
    ctx = GlobalTranslationContext(max_segments=2, max_tokens=10_000)
    for i in range(5):
        ctx.add(ContextEntry(f"文{i}", f"sentence {i}"))
    built = ctx.build()
    assert [e.translated_text for e in built.previous] == ["sentence 3", "sentence 4"]
    assert "sentence 0" in built.summary and "sentence 2" in built.summary


def test_context_token_budget_moves_old_entries_to_summary():
    ctx = GlobalTranslationContext(max_segments=8, max_tokens=40)
    for i in range(8):
        ctx.add(ContextEntry("あ" * 10, f"line {i}"))
    built = ctx.build(current_text="い" * 5)
    assert built.estimated_tokens() <= 40 + estimate_tokens(built.summary)
    assert built.summary  # nothing silently dropped


def test_prompt_contains_context_glossary_speaker_and_honorific_policy():
    p = provider()
    ctx = GlobalTranslationContext()
    ctx.add(ContextEntry("昨日のことだけど…", "About what happened yesterday..."))
    glossary = SessionGlossary.from_payload({"terms": [{"source": "五条", "target": "Gojo"}]})

    async def run():
        return await translate_segment(p, "五条には言わないで。", "ja", "en", glossary=glossary,
                                       context=ctx, honorifics=HonorificPolicy.PRESERVE_HONORIFICS,
                                       speaker=2)

    result = asyncio.run(run())
    system, user = p.backend.calls[0]
    assert "Japanese to English" in system and "-san" in system
    assert "<context>" in user and "About what happened yesterday" in user
    assert "五条 => Gojo" in user  # glossary as a constraint, not a string replacement
    assert "<current>[speaker 2] 五条には言わないで。</current>" in user
    assert result.translated_text == "Don't tell him about it."
    assert result.source_text == "五条には言わないで。"


def test_clean_output_strips_wrappers_and_echoed_context():
    from voicebridge.core.context.translation_context import TranslationContext

    ctx = TranslationContext(previous=[ContextEntry("a", "About yesterday...")])
    assert clean_output("<think>x</think>Translation: \"Hello.\"") == "Hello."
    assert clean_output("About yesterday...\nDon't tell him.", ctx) == "Don't tell him."


def test_remove_policy_strips_suffixes():
    p = provider("Tanaka-san, wait!")

    async def run():
        return await translate_segment(p, "田中さん、待って！", "ja", "en",
                                       glossary=SessionGlossary(),
                                       context=GlobalTranslationContext(),
                                       honorifics=HonorificPolicy.REMOVE)

    assert asyncio.run(run()).translated_text == "Tanaka, wait!"


def test_nmt_providers_keep_sentinel_glossary_path():
    from voicebridge.providers.translation.mock import MockTranslationEngine

    glossary = SessionGlossary.from_payload({"terms": [{"source": "五条悟", "target": "Satoru Gojo"}]})

    async def run():
        return await translate_segment(MockTranslationEngine(latency_seconds=0),
                                       "五条悟の領域展開は本当に印象的でした。", "ja", "en",
                                       glossary=glossary, context=GlobalTranslationContext())

    assert "Satoru Gojo" in asyncio.run(run()).translated_text
