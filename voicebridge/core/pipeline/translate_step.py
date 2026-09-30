"""One translation step, shared by the realtime and file pipelines.

Two provider families are handled differently on purpose:

* **Context-aware providers** (``glossary_as_constraints``) receive a
  :class:`TranslationContext` with the rolling history, the relevant glossary
  terms as constraints, the speaker, the style and the honorific policy. Their
  output is used as-is apart from the ``REMOVE`` honorific policy.
* **Segment NMT providers** (OPUS-MT, NLLB, IndicTrans2) keep the existing
  mechanics: glossary terms are masked with sentinels before translation and
  restored after, and the honorific heuristic rewrites titles afterwards.
"""

from __future__ import annotations

from voicebridge.core.context.glossary import SessionGlossary
from voicebridge.core.context.honorifics import apply_policy, strip_honorific_suffixes
from voicebridge.core.context.translation_context import GlobalTranslationContext
from voicebridge.core.types import (
    HonorificPolicy,
    NameRendering,
    TranslationResult,
)
from voicebridge.providers.base import TranslationEngine


async def translate_segment(
    engine: TranslationEngine,
    text: str,
    source_language: str,
    target_language: str,
    *,
    glossary: SessionGlossary,
    context: GlobalTranslationContext,
    honorifics: HonorificPolicy = HonorificPolicy.NATURAL_ENGLISH,
    name_rendering: NameRendering = NameRendering.TRANSLATE,
    speaker: int | str | None = None,
    style: str | None = None,
    sequence_id: int = 0,
) -> TranslationResult:
    if source_language == target_language:
        return TranslationResult(text, text, source_language, target_language,
                                 sequence_id=sequence_id, provider="passthrough")

    if getattr(engine, "glossary_as_constraints", False):
        ctx = context.build(glossary=glossary, source_language=source_language,
                            speaker=speaker, style=style, honorifics=honorifics,
                            current_text=text)
        result = await engine.translate(
            text, source_language, target_language, context=ctx,
            metadata={"speaker": speaker, "sequence_id": sequence_id,
                      "translation_context": ctx},
        )
        if honorifics is HonorificPolicy.REMOVE:
            result.translated_text = strip_honorific_suffixes(result.translated_text)
        elif honorifics is HonorificPolicy.PRESERVE_HONORIFICS:
            # Small models often ignore the instruction and write "Mr. Tanaka";
            # the (unambiguous-only) title heuristic is a safety net.
            result.translated_text = apply_policy(text, result.translated_text, honorifics,
                                                  source_language)
    else:
        protected, mapping = glossary.protect(text, source_language)
        legacy_context = context.window.source_context() if engine.capabilities.supports_context else None
        result = await engine.translate(
            protected, source_language, target_language, context=legacy_context,
            metadata={"speaker": speaker, "sequence_id": sequence_id},
        )
        restored = glossary.restore(result.translated_text, mapping, name_rendering)
        result.translated_text = apply_policy(text, restored, honorifics, source_language)

    result.source_text = text
    result.sequence_id = sequence_id
    return result
