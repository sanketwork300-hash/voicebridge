"""Language-detection stabilisation.

Whisper-family models report a language per decode window. On short windows,
music, silence or a single loanword, that guess flips readily -- and a session
that switches its ASR language, segmentation profile and translation direction
on every flip produces garbage.

The rule implemented here: a session's language changes only when a *different*
language has been reported consistently, ``min_evidence`` times in a row, above
``min_confidence``, and after ``min_seconds`` of audio. Evidence for the current
language resets the counter. This favours stability over responsiveness on
purpose: mid-stream language changes are rare, and the cost of a false switch is
far higher than the cost of a slow one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LanguageDetectionStabilizer:
    min_evidence: int = 3
    min_confidence: float = 0.5
    min_seconds: float = 2.0

    current: str | None = None
    _candidate: str | None = None
    _streak: int = 0
    _audio_seconds: float = 0.0
    _counts: dict[str, int] = field(default_factory=dict)

    def note_audio(self, seconds: float) -> None:
        self._audio_seconds += seconds

    def observe(self, language: str | None, confidence: float | None = None) -> str | None:
        """Feed one detection. Returns the new language if it just changed."""
        if not language or language == "auto":
            return None
        self._counts[language] = self._counts.get(language, 0) + 1

        if confidence is not None and confidence < self.min_confidence:
            return None

        if language == self.current:
            self._candidate = None
            self._streak = 0
            return None

        if language == self._candidate:
            self._streak += 1
        else:
            self._candidate = language
            self._streak = 1

        if self.current is None:
            # First detection: accept quickly, there is nothing to disrupt.
            if self._streak >= 1 and self._audio_seconds >= 0.0:
                self.current = language
                self._candidate = None
                self._streak = 0
                logger.info("session language detected: %s", language)
                return language
            return None

        if self._streak >= self.min_evidence and self._audio_seconds >= self.min_seconds:
            previous = self.current
            self.current = language
            self._candidate = None
            self._streak = 0
            logger.info("session language switched %s -> %s", previous, language)
            return language
        return None

    @property
    def observations(self) -> dict[str, int]:
        return dict(self._counts)
