"""Accumulate stable ASR text and flush it as translation-sized units."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from voicebridge.core.segmentation.profiles import (
    ALL_TERMINATORS,
    SOFT_BREAKS,
    LanguageSegmentationProfile,
    get_profile,
)
from voicebridge.core.transcript.stabilizer import _join
from voicebridge.core.types import LatencyProfile, TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class PendingSegment:
    text: str = ""
    start: float = 0.0
    end: float = 0.0
    speaker: int | None = None
    language: str | None = None

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class FlushDecision:
    """Why (or why not) the buffer was flushed -- surfaced in metrics and tests."""

    flush: bool
    reason: str = ""


class TranslationSegmenter:
    """Groups stable transcript spans into meaningful units for translation.

    Flush triggers, in priority order:

    1. ``max_seconds`` / ``max_chars`` -- hard safety valves; always fire.
    2. sentence-terminating punctuation (when ``punctuation_flush``).
    3. a sentence-final ending, for languages with ``sentence_final_bias``.
    4. a pause at least ``pause_threshold_ms`` long.
    5. a soft break, only once the buffer is already reasonably long.

    A buffer shorter than ``minimum_stable_chars`` is never flushed by rules
    2-5, which is what prevents "I", "think", "the" from being translated
    independently. Rule 1 overrides that, so a long unpunctuated monologue still
    produces output.
    """

    def __init__(
        self,
        language: str | None = None,
        latency: LatencyProfile = LatencyProfile.BALANCED,
        profile: LanguageSegmentationProfile | None = None,
    ):
        base = profile or get_profile(language)
        self.profile = base.scaled(latency)
        self.language = language
        self._pending = PendingSegment()
        self._last_end: float | None = None

    @property
    def pending_text(self) -> str:
        return self._pending.text

    def set_language(self, language: str, latency: LatencyProfile = LatencyProfile.BALANCED) -> None:
        """Switch profiles after language detection stabilises."""
        if language and language != self.language:
            self.language = language
            self.profile = get_profile(language).scaled(latency)
            logger.info("segmenter switched to language profile %s", language)

    def add(self, segment: TranscriptSegment) -> list[PendingSegment]:
        """Add newly stable text; return zero or more units ready to translate."""
        out: list[PendingSegment] = []
        text = (segment.text or "").strip()
        if not text:
            return out

        pause = 0.0
        if self._last_end is not None:
            pause = max(0.0, segment.start - self._last_end)
        self._last_end = segment.end

        # A pause *before* this segment closes whatever came before it.
        if not self._pending.empty and pause * 1000 >= self.profile.pause_threshold_ms:
            if self._long_enough():
                out.append(self._take("pause"))

        if self._pending.empty:
            self._pending = PendingSegment(
                text=text,
                start=segment.start,
                end=segment.end,
                speaker=segment.speaker,
                language=segment.language or self.language,
            )
        else:
            # A speaker change is a hard boundary: never merge two speakers into
            # one translation unit.
            if (
                segment.speaker is not None
                and self._pending.speaker is not None
                and segment.speaker != self._pending.speaker
            ):
                out.append(self._take("speaker_change"))
                self._pending = PendingSegment(
                    text=text,
                    start=segment.start,
                    end=segment.end,
                    speaker=segment.speaker,
                    language=segment.language or self.language,
                )
            else:
                self._pending.text += _join(self._pending.text, text)
                self._pending.end = segment.end

        decision = self._should_flush()
        if decision.flush:
            out.append(self._take(decision.reason))
        return out

    def on_silence(self, duration_seconds: float) -> list[PendingSegment]:
        """Called when the audio layer reports a gap with no speech."""
        if self._pending.empty:
            return []
        if duration_seconds * 1000 >= self.profile.pause_threshold_ms and self._long_enough():
            return [self._take("silence")]
        return []

    def flush(self, reason: str = "forced") -> list[PendingSegment]:
        """Flush unconditionally -- used at end of stream and on pause/stop."""
        if self._pending.empty:
            return []
        return [self._take(reason)]

    # -- internals ---------------------------------------------------------

    def _long_enough(self) -> bool:
        return len(self._pending.text.strip()) >= self.profile.minimum_stable_chars

    def _should_flush(self) -> FlushDecision:
        text = self._pending.text.strip()
        if not text:
            return FlushDecision(False)

        # 1. Hard safety valves. These ignore minimum length by design.
        if len(text) >= self.profile.max_chars:
            return FlushDecision(True, "max_chars")
        if self._pending.duration >= self.profile.max_seconds:
            return FlushDecision(True, "max_seconds")

        if not self._long_enough():
            return FlushDecision(False, "too_short")

        last = text[-1]

        # 2. Sentence-terminating punctuation.
        if self.profile.punctuation_flush and last in ALL_TERMINATORS:
            return FlushDecision(True, "punctuation")

        # 3. Language-specific sentence-final endings.
        if self.profile.sentence_final_bias:
            if any(text.endswith(e) for e in self.profile.sentence_final_endings):
                return FlushDecision(True, "sentence_final")
            # Under sentence-final bias nothing weaker may flush: a clause
            # without its ending is exactly what we must not translate.
            return FlushDecision(False, "awaiting_sentence_final")

        # 5. Soft break, only when already substantial.
        if last in SOFT_BREAKS and len(text) >= self.profile.max_chars // 2:
            return FlushDecision(True, "soft_break")

        return FlushDecision(False, "accumulating")

    def _take(self, reason: str) -> PendingSegment:
        seg = self._pending
        self._pending = PendingSegment()
        logger.debug("segmenter flush (%s): %r", reason, seg.text[:60])
        seg_reason = reason
        seg.reason = seg_reason
        return seg
