"""Japanese/Korean honorific preservation.

For anime and drama audiences, "Tanaka-san" and "Mr. Tanaka" are not
interchangeable: the honorific encodes a relationship that the English title
flattens, and the fandom convention is overwhelmingly to keep it. VoiceBridge
therefore exposes two policies (``NATURAL_ENGLISH``, ``PRESERVE_HONORIFICS``)
and defaults the Anime/Drama preset to preserving.

What this module actually does, and what it does not
----------------------------------------------------
It is a **post-translation heuristic**, not a model capability. After the NMT
model has produced English, it:

1. detects which Japanese honorifics were present in the *source* text;
2. if exactly one distinct honorific is present, rewrites English politeness
   titles ("Mr. Tanaka", "Miss Sato") into the suffixed form ("Tanaka-san").

The "exactly one" restriction is deliberate. With two speakers addressed by
different honorifics in one segment there is no reliable way to attribute a
title in the output to an honorific in the input without word alignment, and
guessing would produce confidently wrong output. In that case the module leaves
the translation alone -- an unconverted "Mr. Tanaka" is a much smaller error
than "Tanaka-chan" attached to the wrong person.

A glossary entry always beats this heuristic: if the user supplies
``{"source": "田中さん", "target": "Tanaka-san"}`` the term never reaches the
model in the first place. This is the recommended path for recurring characters.
"""

from __future__ import annotations

import re

from voicebridge.core.types import HonorificPolicy

#: Japanese honorifics and the suffix used when preserving them.
JA_HONORIFICS: dict[str, str] = {
    "さん": "-san",
    "君": "-kun",
    "くん": "-kun",
    "ちゃん": "-chan",
    "様": "-sama",
    "さま": "-sama",
    "先生": "-sensei",
    "先輩": "-senpai",
    "殿": "-dono",
}

#: Korean address terms. Korean honorifics attach to relationships rather than
#: names as consistently as Japanese ones do, so only the clearest cases are
#: handled; everything else is left to the glossary.
KO_HONORIFICS: dict[str, str] = {
    "선배": "-seonbae",
    "후배": "-hubae",
    "언니": "-unnie",
    "오빠": "-oppa",
    "누나": "-noona",
    "형": "-hyung",
}

#: English titles the NMT model is likely to produce for an honorific.
_TITLE_RE = re.compile(
    r"\b(Mr\.?|Mrs\.?|Ms\.?|Miss|Master|Sir|Madam|Teacher|Professor)\s+"
    r"([A-Z][A-Za-z'’-]+)"
)


def detect_honorifics(source_text: str, language: str | None = None) -> list[str]:
    """Return the distinct honorific suffixes present in ``source_text``."""
    table = KO_HONORIFICS if (language or "").startswith("ko") else JA_HONORIFICS
    found: list[str] = []
    for token, suffix in table.items():
        if token in source_text and suffix not in found:
            found.append(suffix)
    return found


def apply_policy(
    source_text: str,
    translated_text: str,
    policy: HonorificPolicy,
    source_language: str | None = None,
) -> str:
    """Rewrite English titles into honorific suffixes when the policy asks for it.

    Returns ``translated_text`` unchanged when the policy is
    ``NATURAL_ENGLISH``, when no honorific was present in the source, or when
    the source is ambiguous (more than one distinct honorific).
    """
    if policy is HonorificPolicy.REMOVE:
        return strip_honorific_suffixes(translated_text)
    if policy is not HonorificPolicy.PRESERVE_HONORIFICS:
        return translated_text
    if not source_text or not translated_text:
        return translated_text

    suffixes = detect_honorifics(source_text, source_language)
    if len(suffixes) != 1:
        # Zero: nothing to preserve. More than one: ambiguous, do not guess.
        return translated_text

    suffix = suffixes[0]

    def _replace(match: re.Match) -> str:
        title, name = match.group(1), match.group(2)
        # "Teacher/Professor X" only maps onto -sensei; otherwise keep the title.
        if title in ("Teacher", "Professor") and suffix != "-sensei":
            return match.group(0)
        return f"{name}{suffix}"

    return _TITLE_RE.sub(_replace, translated_text)


_SUFFIX_RE = re.compile(
    r"(?<=[A-Za-z])-(?:"
    + "|".join(sorted({v[1:] for v in {**JA_HONORIFICS, **KO_HONORIFICS}.values()}, key=len, reverse=True))
    + r")\b"
)


def strip_honorific_suffixes(text: str) -> str:
    """``REMOVE`` policy: "Tanaka-san" -> "Tanaka". Titles are left alone."""
    return _SUFFIX_RE.sub("", text)


def policy_instruction(policy: HonorificPolicy, source_language: str | None) -> str:
    """How an instruction-following translator should treat honorifics.

    Used by the contextual (LLM) provider, which receives the policy as a
    constraint instead of having its output rewritten afterwards.
    """
    if policy is HonorificPolicy.PRESERVE_HONORIFICS:
        table = KO_HONORIFICS if (source_language or "").startswith("ko") else JA_HONORIFICS
        examples = ", ".join(sorted(set(table.values())))
        return (
            "Keep honorifics attached to names in romanized form "
            f"({examples}), e.g. 'Tanaka-san', instead of replacing them with Mr./Ms."
        )
    if policy is HonorificPolicy.REMOVE:
        return "Drop honorific suffixes and titles; use bare names."
    return (
        "Render honorifics naturally in the target language (e.g. Mr./Ms./teacher, "
        "or nothing) while keeping the politeness level and relationship they imply."
    )
