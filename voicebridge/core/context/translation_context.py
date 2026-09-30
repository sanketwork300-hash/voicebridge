"""Structured translation context handed to context-aware translation providers.

:class:`~voicebridge.core.context.window.ContextWindow` keeps recent
(source, translation) pairs; this module turns them into a bounded,
provider-neutral :class:`TranslationContext`:

* the most recent pairs, newest last, within ``max_segments``;
* a token budget (``max_tokens``). Tokens are *estimated* -- one per CJK /
  Hangul / Devanagari character, one per four other characters -- because the
  core must not depend on any particular tokenizer;
* compression: pairs that do not fit are not silently dropped. Their
  translations are folded, oldest first, into a short ``summary`` line
  ("Earlier: ...") that keeps names and topic in view at a fraction of the cost.

The same object is used by the realtime pipeline and, as
``GlobalTranslationContext``, across chunks of a long file, so chunk N is
translated knowing what chunks 1..N-1 said.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from voicebridge.core.context.glossary import GlossaryTerm, SessionGlossary
from voicebridge.core.context.window import ContextEntry, ContextWindow
from voicebridge.core.types import HonorificPolicy

_WIDE_RE = re.compile(r"[ऀ-ॿ぀-ヿ㐀-鿿가-힯]")


def estimate_tokens(text: str) -> int:
    wide = len(_WIDE_RE.findall(text))
    return wide + max(0, len(text) - wide) // 4 + 1


@dataclass
class TranslationContext:
    previous: list[ContextEntry] = field(default_factory=list)
    summary: str = ""
    glossary: list[GlossaryTerm] = field(default_factory=list)
    speaker: str | int | None = None
    style: str | None = None
    honorifics: HonorificPolicy = HonorificPolicy.NATURAL_ENGLISH
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def previous_source(self) -> list[str]:
        return [e.source_text for e in self.previous]

    @property
    def previous_translations(self) -> list[str]:
        return [e.translated_text for e in self.previous]

    def estimated_tokens(self) -> int:
        return sum(estimate_tokens(e.source_text) + estimate_tokens(e.translated_text)
                   for e in self.previous) + estimate_tokens(self.summary)


class GlobalTranslationContext:
    """Rolling context that persists for a whole session or file job."""

    def __init__(self, max_segments: int = 8, max_tokens: int = 2048, summary_chars: int = 400):
        self.window = ContextWindow(max_segments=max(1, max_segments) * 4,
                                    max_characters=10**9)
        self.max_segments = max(1, max_segments)
        self.max_tokens = max_tokens
        self.summary_chars = summary_chars
        self._summary: list[str] = []

    def add(self, entry: ContextEntry) -> None:
        self.window.add(entry)

    def build(
        self,
        glossary: SessionGlossary | None = None,
        source_language: str | None = None,
        speaker: str | int | None = None,
        style: str | None = None,
        honorifics: HonorificPolicy = HonorificPolicy.NATURAL_ENGLISH,
        current_text: str = "",
    ) -> TranslationContext:
        entries = self.window.entries
        recent = entries[-self.max_segments:]
        overflow = entries[: -self.max_segments] if len(entries) > self.max_segments else []
        budget = self.max_tokens - estimate_tokens(current_text)
        kept: list[ContextEntry] = []
        used = 0
        for entry in reversed(recent):
            cost = estimate_tokens(entry.source_text) + estimate_tokens(entry.translated_text)
            if used + cost > budget:
                overflow = overflow + [entry]
                continue
            kept.insert(0, entry)
            used += cost
        summary = self._compress(overflow)
        terms = glossary.applicable(source_language) if glossary else []
        # Only terms that appear in the current or recent source text: a long
        # glossary would otherwise dominate the prompt.
        haystack = current_text + "".join(e.source_text for e in kept)
        relevant = [t for t in terms if t.source in haystack]
        return TranslationContext(previous=kept, summary=summary, glossary=relevant,
                                  speaker=speaker, style=style, honorifics=honorifics)

    def _compress(self, dropped: list[ContextEntry]) -> str:
        if not dropped:
            return ""
        text = " ".join(e.translated_text.strip() for e in dropped if e.translated_text.strip())
        if len(text) > self.summary_chars:
            text = "..." + text[-self.summary_chars:]
        return text

    def clear(self) -> None:
        self.window.clear()

    def __len__(self) -> int:
        return len(self.window)
