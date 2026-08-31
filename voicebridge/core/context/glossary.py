"""Session glossaries: keeping proper nouns intact through translation.

The problem this solves is specific and very visible to users. A general NMT
model will happily render 五条悟 as "Gojo Satoru", "Gojou Satoru", "Five Article
Enlightenment", or a different one of those in each subtitle. For anime, drama
and K-pop content, character and artist names are exactly what the audience
cares most about getting right, and inconsistency is more jarring than a slight
grammatical error.

Mechanism: **placeholder protection**. Before translation each glossary source
term is replaced with an opaque sentinel that survives tokenisation; after
translation the sentinel is replaced with the desired target term. This works
with any provider, including remote ones, because it needs no model support.

The sentinel format matters. It must (a) not be split into subwords in a way
that loses it, (b) not look like natural language the model will translate, and
(c) survive being copied through the decoder. Digit-delimited ASCII sentinels do
this reliably across Marian, NLLB and IndicTrans2 style models. Where a model
drops a sentinel entirely, :meth:`SessionGlossary.restore` leaves the rest of
the sentence untouched rather than corrupting it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_SENTINEL = "VBX{index}X"
_SENTINEL_RE = re.compile(r"VBX(\d+)X")


@dataclass
class GlossaryTerm:
    source: str
    target: str
    #: Optional romanisation, used by NameRendering.PRESERVE_AND_ROMANIZE.
    romanized: str | None = None
    #: When set, only applies while translating out of this language.
    source_language: str | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("glossary term needs a non-empty source")


@dataclass
class SessionGlossary:
    """An ordered set of term mappings scoped to one session.

    Optional by design: an empty glossary is a no-op with no measurable cost.
    """

    terms: list[GlossaryTerm] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: dict | None) -> SessionGlossary:
        """Build from the wire format documented in ``docs/protocol.md``."""
        if not payload:
            return cls()
        raw = payload.get("terms", []) if isinstance(payload, dict) else payload
        terms: list[GlossaryTerm] = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            source = (item.get("source") or "").strip()
            target = (item.get("target") or "").strip()
            if not source or not target:
                continue
            terms.append(
                GlossaryTerm(
                    source=source,
                    target=target,
                    romanized=(item.get("romanized") or None),
                    source_language=(item.get("source_language") or None),
                )
            )
        return cls(terms=terms)

    def applicable(self, source_language: str | None) -> list[GlossaryTerm]:
        out = []
        for t in self.terms:
            if t.source_language and source_language and t.source_language != source_language:
                continue
            out.append(t)
        # Longest first: 五条悟先生 must win over 五条悟.
        return sorted(out, key=lambda t: len(t.source), reverse=True)

    def protect(self, text: str, source_language: str | None = None) -> tuple[str, dict[str, GlossaryTerm]]:
        """Replace known terms with sentinels. Returns (text, sentinel map)."""
        mapping: dict[str, GlossaryTerm] = {}
        if not self.terms or not text:
            return text, mapping
        out = text
        for idx, term in enumerate(self.applicable(source_language)):
            if term.source not in out:
                continue
            sentinel = _SENTINEL.format(index=idx)
            out = out.replace(term.source, sentinel)
            mapping[sentinel] = term
        return out, mapping

    def restore(
        self,
        text: str,
        mapping: dict[str, GlossaryTerm],
        rendering: object = None,
    ) -> str:
        """Replace sentinels with target terms.

        ``rendering`` is a :class:`~voicebridge.core.types.NameRendering`; it is
        typed loosely to keep this module free of a circular import.
        """
        from voicebridge.core.types import NameRendering

        if not mapping:
            return text
        out = text
        for sentinel, term in mapping.items():
            replacement = term.target
            if rendering is NameRendering.PRESERVE:
                replacement = term.source
            elif rendering is NameRendering.PRESERVE_AND_ROMANIZE:
                roman = term.romanized or term.target
                replacement = f"{term.source} ({roman})" if roman else term.source
            out = out.replace(sentinel, replacement)

        # A model may have mangled or dropped a sentinel. Strip any survivors
        # rather than showing "VBX3X" to the user.
        leftovers = _SENTINEL_RE.findall(out)
        if leftovers:
            logger.debug("glossary sentinels survived translation: %s", leftovers)
            out = _SENTINEL_RE.sub("", out)
            out = re.sub(r"\s{2,}", " ", out).strip()
        return out

    def __len__(self) -> int:
        return len(self.terms)

    def __bool__(self) -> bool:
        return bool(self.terms)
