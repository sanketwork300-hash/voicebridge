"""Context-aware (LLM) translation provider.

Segment-by-segment NMT (OPUS-MT) translates every unit in isolation, which
loses omitted subjects, pronoun reference, running jokes and the
relationships honorifics encode. This provider instead gives an
instruction-following model the current segment *plus*:

* the previous source segments and their translations (bounded window with
  compression, :mod:`voicebridge.core.context.translation_context`);
* glossary entries as **constraints in the prompt** -- the output is never
  string-replaced afterwards, so the model can inflect and place the term;
* speaker id, requested style, the honorific policy and the language pair.

The model is told to return only the translation. Output is then cleaned of the
usual wrapper noise (quotes, "Translation:" labels, reasoning blocks) and
checked for leaked context: if the model translated the previous lines as well,
only the last line is kept.

Which model runs is configuration (``backend`` + ``model``); see
:mod:`voicebridge.providers.translation.llm_backends`. Quality therefore depends
on the configured model, and the benchmark exists to measure it rather than
assume it.
"""

from __future__ import annotations

import re
from typing import Any

from voicebridge.core.context.honorifics import policy_instruction
from voicebridge.core.context.translation_context import TranslationContext
from voicebridge.core.context.window import ContextEntry
from voicebridge.core.types import HonorificPolicy, TranslationResult, now
from voicebridge.providers.base import TranslationCapabilities, TranslationEngine
from voicebridge.providers.registry import translation_registry
from voicebridge.providers.translation.llm_backends import ChatBackend, create_backend

LANGUAGE_NAMES = {
    "en": "English", "ja": "Japanese", "ko": "Korean", "hi": "Hindi", "zh": "Chinese",
    "mr": "Marathi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali", "es": "Spanish",
    "fr": "French", "de": "German", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
}

SYSTEM_PROMPT = """You are a professional audiovisual translator working on {src} to {tgt} dialogue (subtitles and dubbing).

Rules:
- Translate ONLY the text inside <current>. The <context> lines are earlier dialogue, given so you can resolve omitted subjects, pronouns, references and tone. Never translate or repeat them.
- Preserve meaning, tone, register and politeness level. Keep slang as slang and render idioms with a natural {tgt} equivalent instead of word-for-word.
- Restore subjects or objects the {src} omits only when the context makes them clear; do not invent information that is not implied.
- Do not add explanations, notes, alternatives, romanization or quotation marks.
- {honorifics}
- Use the glossary translations exactly as given when the source term appears.
- Keep names and terminology consistent with the earlier translations.
{style}
Output only the {tgt} translation of the current line."""


def _lang(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


class ContextualTranslationProvider(TranslationEngine):
    name = "contextual"

    def __init__(
        self,
        backend: str = "transformers",
        model: str | None = None,
        max_new_tokens: int = 160,
        backend_options: dict[str, Any] | None = None,
        **options: Any,
    ):
        opts = dict(backend_options or {})
        for key in ("device", "dtype", "base_url", "api_key_env", "timeout", "effort",
                    "fallbacks", "threads", "temperature"):
            if key in options:
                opts[key] = options.pop(key)
        if model:
            opts["model"] = model
        self.backend: ChatBackend = create_backend(backend, **opts)
        self.max_new_tokens = int(max_new_tokens)

    @property
    def capabilities(self) -> TranslationCapabilities:
        return TranslationCapabilities(
            name=self.name,
            source_languages=[],
            target_languages=[],
            supports_context=True,
            supports_glossary=True,
            model_license=f"{self.backend.name}:{self.backend.model} ({self.backend.license})",
            commercial_use=None,
        )

    @property
    def glossary_as_constraints(self) -> bool:
        """The pipeline passes glossary terms in the prompt instead of masking them."""
        return True

    async def warmup(self) -> None:
        await self.backend.warmup()

    def build_prompt(
        self, text: str, source_language: str, target_language: str,
        context: TranslationContext,
    ) -> tuple[str, str]:
        style = f"- Style: {context.style}." if context.style else ""
        system = SYSTEM_PROMPT.format(
            src=_lang(source_language), tgt=_lang(target_language),
            honorifics=policy_instruction(context.honorifics, source_language),
            style=style,
        )
        parts: list[str] = []
        if context.summary:
            parts.append(f"<summary>Earlier in this conversation: {context.summary}</summary>")
        if context.previous:
            lines = []
            for entry in context.previous:
                who = f"[speaker {entry.speaker}] " if entry.speaker is not None else ""
                lines.append(f"{who}{entry.source_text}\n  => {entry.translated_text}")
            parts.append("<context>\n" + "\n".join(lines) + "\n</context>")
        if context.glossary:
            terms = "\n".join(f"{t.source} => {t.target}" for t in context.glossary)
            parts.append(f"<glossary>\n{terms}\n</glossary>")
        who = f"[speaker {context.speaker}] " if context.speaker is not None else ""
        parts.append(f"<current>{who}{text.strip()}</current>")
        return system, "\n\n".join(parts)

    async def translate(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        context: list[str] | TranslationContext | None = None,
        metadata: dict[str, object] | None = None,
        *,
        glossary: Any = None,
        speaker: str | int | None = None,
        style: str | None = None,
    ) -> TranslationResult:
        metadata = metadata or {}
        ctx = metadata.get("translation_context")
        if isinstance(context, TranslationContext):
            ctx = context
        if not isinstance(ctx, TranslationContext):
            # Legacy callers pass a list of previous source strings.
            previous = [ContextEntry(s, "") for s in (context or []) if isinstance(s, str)]
            ctx = TranslationContext(previous=previous)
        if glossary is not None:
            ctx.glossary = list(glossary.applicable(source_language)) if hasattr(
                glossary, "applicable") else list(glossary)
        if speaker is not None:
            ctx.speaker = speaker
        if style is not None:
            ctx.style = style
        if metadata.get("honorifics") is not None:
            ctx.honorifics = HonorificPolicy(metadata["honorifics"])

        system, user = self.build_prompt(source_text, source_language, target_language, ctx)
        raw = await self.backend.complete(system, user, self.max_new_tokens)
        translated = clean_output(raw, ctx)
        return TranslationResult(
            source_text=source_text,
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            confidence=None,
            timestamp=now(),
            provider=f"{self.name}:{self.backend.name}:{self.backend.model}",
            speaker=ctx.speaker if isinstance(ctx.speaker, int) else None,
        )


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_LABEL_RE = re.compile(r"^\s*(translation|english|hindi|output|answer)\s*:\s*", re.IGNORECASE)
_TAG_RE = re.compile(r"</?(current|context|summary|glossary)>")


def clean_output(raw: str, context: TranslationContext | None = None) -> str:
    text = _THINK_RE.sub("", raw or "")
    text = _TAG_RE.sub("", text).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    # If the model echoed earlier translations too, keep only what is new.
    if context is not None and len(lines) > 1:
        known = {e.translated_text.strip() for e in context.previous}
        fresh = [ln for ln in lines if ln.lstrip("=> ").strip() not in known]
        lines = fresh or lines[-1:]
    text = " ".join(ln.lstrip("=> ").strip() for ln in lines)
    text = _LABEL_RE.sub("", text)
    text = re.sub(r"^\[speaker [^\]]*\]\s*", "", text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”「」":
        text = text[1:-1]
    text = text.strip().strip("“”").strip()
    return text


translation_registry.register("contextual", ContextualTranslationProvider)
translation_registry.register("llm", ContextualTranslationProvider)
translation_registry.register("contextual_llm", ContextualTranslationProvider)
