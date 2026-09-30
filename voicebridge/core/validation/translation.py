"""Lightweight translation output validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from voicebridge.core.types import TranslationResult


@dataclass
class TranslationValidationResult:
    valid: bool
    confidence: float
    issues: list[str] = field(default_factory=list)


class TranslationValidationEngine:
    def __init__(self, policy: str = "balanced"):
        self.policy = policy

    def validate(self, result: TranslationResult) -> TranslationValidationResult:
        issues: list[str] = []
        text = (result.translated_text or "").strip()
        if not text:
            issues.append("empty_output")
        if _repetition(text) > 0.50:
            issues.append("repetition")
        if result.target_language and not _looks_like_language(text, result.target_language):
            # Kept as a warning signal; mixed-script names and honorifics are valid.
            issues.append("possible_language_mismatch")
        if result.source_text.strip() and text == result.source_text.strip() and (
            result.source_language != result.target_language
        ):
            issues.append("unchanged_source_text")

        if self.policy == "fast":
            blocking = {"empty_output"}
        elif self.policy == "strict":
            blocking = set(issues)
        else:
            blocking = {"empty_output", "repetition"}
        confidence = 1.0 - min(0.8, 0.2 * len(issues))
        return TranslationValidationResult(not (blocking & set(issues)), confidence, issues)


def _repetition(text: str) -> float:
    tokens = re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE)
    if len(tokens) < 4:
        return 0.0
    return sum(1 for a, b in zip(tokens, tokens[1:], strict=False) if a == b) / (len(tokens) - 1)


def _looks_like_language(text: str, language: str) -> bool:
    if not text:
        return False
    if language in {"en", "hi"}:
        return True
    if language == "ja":
        return bool(re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text))
    if language == "ko":
        return bool(re.search(r"[\uac00-\ud7af]", text))
    return True

