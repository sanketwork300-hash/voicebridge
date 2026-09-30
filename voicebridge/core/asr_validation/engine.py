"""Post-ASR validation: catch likely hallucinations before they are translated.

Whisper-family models hallucinate in well-documented ways on non-speech audio:
fluent sentences over music, looping repetitions, and a handful of stock
phrases learned from subtitle credits ("ご視聴ありがとうございました",
"Thanks for watching", "Subtitles by the Amara.org community"). Once such text
reaches translation and TTS it is spoken aloud, which is far worse than a gap.

Signals, in the order they are checked (each rejection names its reason):

* empty text, implausibly short span, abnormal characters-per-second;
* token repetition and character n-gram loops (CJK has no spaces, so a word
  tokenizer alone misses "ははははは" style loops); zlib compression ratio of
  the text when the decoder did not supply one;
* Whisper decoder signals when present: ``no_speech_prob`` (alone, and the
  decoder's own rule "no_speech_prob > 0.6 and avg_logprob < -1"),
  ``avg_logprob``, ``compression_ratio``, mean word probability;
* language mismatch against the expected source language;
* the speech gate's verdict for the same time span;
* stock hallucination phrases -- rejected only when some other signal is weak,
  because people do say "thank you for watching".

The confidence of an accepted segment combines exp(avg_logprob), 1 -
no_speech_prob and the mean word probability, whichever are available.
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass, field
from typing import Any

from voicebridge.core.types import AudioEventType, SpeechDecision, TranscriptSegment

#: Normalised (lower-case, no punctuation/space) stock phrases.
KNOWN_HALLUCINATIONS = {
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "ありがとうございました",
    "チャンネル登録お願いします",
    "チャンネル登録よろしくお願いします",
    "最後までご視聴いただきありがとうございました",
    "字幕作成者",
    "시청해주셔서감사합니다",
    "구독과좋아요부탁드립니다",
    # Added after the held-out Korean non-speech run (docs/evaluation.md §1),
    # where the validator initially let 5/8 clips through.
    "다음영상에서만나요",
    "감사합니다",
    "아멘",
    "thankyouforwatching",
    "thanksforwatching",
    "pleasesubscribe",
    "subtitlesbytheamaraorgcommunity",
    "subtitlesby",
}

_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)

#: Subtitle-credit lines learned from fan-subtitled training data
#: ("한글자막 by …", "字幕：…", "Subtitles by …"). Never real dialogue.
_CREDIT_RE = re.compile(r"(자막\s*(by|제공)|字幕\s*[:：by]|subtitles?\s+by|captions?\s+by)",
                        re.IGNORECASE)


@dataclass
class ASRValidationResult:
    valid: bool
    confidence: float
    reason: str | None = None
    signals: dict[str, Any] = field(default_factory=dict)


class ASRValidationEngine:
    def __init__(
        self,
        min_duration: float = 0.08,
        max_chars_per_second: float = 28.0,
        no_speech_threshold: float = 0.85,
        min_avg_logprob: float = -1.5,
        max_compression_ratio: float = 2.6,
        min_word_probability: float = 0.25,
        reject_non_dialogue_confidence: float = 0.60,
        language_mismatch_probability: float = 0.80,
    ):
        self.min_duration = min_duration
        self.max_chars_per_second = max_chars_per_second
        self.no_speech_threshold = no_speech_threshold
        self.min_avg_logprob = min_avg_logprob
        self.max_compression_ratio = max_compression_ratio
        self.min_word_probability = min_word_probability
        self.reject_non_dialogue_confidence = reject_non_dialogue_confidence
        self.language_mismatch_probability = language_mismatch_probability

    @classmethod
    def from_dict(cls, data: dict | None) -> ASRValidationEngine:
        data = data or {}
        allowed = cls.__init__.__code__.co_varnames[1 : cls.__init__.__code__.co_argcount]
        return cls(**{k: float(v) for k, v in data.items() if k in allowed})

    def validate(
        self,
        segment: TranscriptSegment,
        metadata: dict[str, Any] | None = None,
        speech_decision: SpeechDecision | None = None,
        expected_language: str | None = None,
    ) -> ASRValidationResult:
        metadata = metadata or {}
        text = (segment.text or "").strip()
        signals: dict[str, Any] = {"duration": round(segment.duration, 3), "chars": len(text)}

        def reject(reason: str, confidence: float) -> ASRValidationResult:
            return ASRValidationResult(False, confidence, reason, signals)

        if not text or not _PUNCT_RE.sub("", text):
            return reject("empty_transcript", 0.0)
        if segment.duration < self.min_duration:
            return reject("segment_too_short", 0.25)
        density = len(_PUNCT_RE.sub("", text)) / max(segment.duration, 1e-6)
        signals["text_density"] = round(density, 2)
        if density > self.max_chars_per_second:
            return reject("abnormal_text_density", 0.35)

        repeated = _repetition_score(text)
        loop = _ngram_loop_score(text)
        signals["repetition_score"] = round(repeated, 3)
        signals["ngram_loop_score"] = round(loop, 3)
        phrase = _phrase_loop_score(text)
        signals["phrase_loop_score"] = round(phrase, 3)
        if repeated > 0.55 or loop > 0.6 or phrase > 0.5:
            return reject("repetition", 0.30)

        no_speech = _float(metadata.get("no_speech_prob"))
        avg_logprob = _float(metadata.get("avg_logprob"))
        compression = _float(metadata.get("compression_ratio"))
        if compression is None and len(text.encode()) >= 24:
            compression = _compression_ratio(text)
        word_prob = _mean_word_probability(metadata.get("words") or segment.words)
        for key, value in (("no_speech_prob", no_speech), ("avg_logprob", avg_logprob),
                           ("compression_ratio", compression), ("word_probability", word_prob)):
            if value is not None:
                signals[key] = round(value, 3)

        if no_speech is not None and no_speech >= self.no_speech_threshold:
            return reject("high_no_speech_probability", 0.20)
        if no_speech is not None and avg_logprob is not None and no_speech > 0.6 and avg_logprob < -1.0:
            return reject("no_speech_and_low_logprob", 0.20)
        if avg_logprob is not None and avg_logprob < self.min_avg_logprob:
            return reject("low_average_log_probability", 0.35)
        if compression is not None and compression > self.max_compression_ratio:
            return reject("high_compression_ratio", 0.35)
        if word_prob is not None and word_prob < self.min_word_probability:
            return reject("low_word_probability", 0.35)

        detected = metadata.get("language") or segment.language
        detected_p = _float(metadata.get("language_probability"))
        if (
            expected_language and expected_language != "auto" and detected
            and detected != expected_language
            and (detected_p is None or detected_p >= self.language_mismatch_probability)
        ):
            signals["language"] = detected
            return reject("language_mismatch", 0.30)

        non_dialogue = False
        if speech_decision is not None:
            signals["audio_event_type"] = speech_decision.event_type
            signals["audio_event_confidence"] = round(speech_decision.confidence, 3)
            non_dialogue = speech_decision.event_type not in (
                AudioEventType.DIALOGUE.value, AudioEventType.UNKNOWN.value
            )
            if non_dialogue and speech_decision.confidence >= self.reject_non_dialogue_confidence:
                return reject("non_dialogue_audio_event", 0.25)

        confidence = _combined_confidence(segment.confidence, avg_logprob, no_speech, word_prob)
        normalised = _PUNCT_RE.sub("", text).lower()
        if _CREDIT_RE.search(text):
            return reject("subtitle_credit", 0.10)
        if normalised in KNOWN_HALLUCINATIONS:
            # Measured on real non-speech clips (laughter, crying, applause, choir):
            # Whisper large-v3-turbo emitted these phrases with no_speech_prob 0.00
            # and avg_logprob around -0.3, so decoder signals cannot be trusted
            # here. Only positive evidence of dialogue from the speech gate lets
            # a stock phrase through.
            dialogue_evidence = (
                speech_decision is not None
                and speech_decision.event_type == AudioEventType.DIALOGUE.value
                and speech_decision.confidence >= 0.5
            )
            weak = (
                not dialogue_evidence
                or non_dialogue
                or (no_speech is not None and no_speech > 0.3)
                or (avg_logprob is not None and avg_logprob < -0.6)
                or (speech_decision is not None
                    and speech_decision.event_type == AudioEventType.UNKNOWN.value)
                or confidence < 0.6
            )
            signals["known_hallucination_phrase"] = True
            if weak:
                return reject("known_hallucination_phrase", 0.30)
        return ASRValidationResult(True, confidence, None, signals)


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _mean_word_probability(words: Any) -> float | None:
    probs = []
    for w in words or []:
        p = w.get("probability") if isinstance(w, dict) else getattr(w, "probability", None)
        if p is not None:
            probs.append(float(p))
    return sum(probs) / len(probs) if probs else None


def _combined_confidence(
    base: float | None, avg_logprob: float | None, no_speech: float | None, word_prob: float | None
) -> float:
    parts = [p for p in (
        base,
        math.exp(avg_logprob) if avg_logprob is not None else None,
        (1.0 - no_speech) if no_speech is not None else None,
        word_prob,
    ) if p is not None]
    if not parts:
        return 0.80
    return max(0.0, min(1.0, sum(parts) / len(parts)))


def _compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    return len(raw) / max(1, len(zlib.compress(raw)))


def _repetition_score(text: str) -> float:
    tokens = re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE)
    if len(tokens) < 4:
        return 0.0
    repeated = sum(1 for a, b in zip(tokens, tokens[1:], strict=False) if a == b)
    return repeated / max(1, len(tokens) - 1)


def _ngram_loop_score(text: str) -> float:
    """Fraction of the text covered by one n-gram (n = 1..6) repeated back-to-back.

    "ははははははは" -> ~1.0, "ありがとうありがとうありがとう" -> 1.0,
    a normal sentence -> ~0.
    """
    s = _PUNCT_RE.sub("", text)
    if len(s) < 5:
        return 0.0
    best = 0
    for n in range(1, 7):
        i = 0
        while i + 2 * n <= len(s):
            unit = s[i : i + n]
            j = i + n
            while s[j : j + n] == unit:
                j += n
            run = j - i
            if run >= 3 * n and run > best:
                best = run
            i = j if j > i + n else i + 1
    return best / len(s)


def _phrase_loop_score(text: str) -> float:
    """Share of words inside a word n-gram (n = 2..8) that occurs 3+ times.

    Catches decoder loops such as "the UN is a production of the UN, and the
    UN is a production of the UN, and ..." that character-level checks miss
    because the repeating unit is long.
    """
    words = re.findall(r"\w+", text.lower())
    if len(words) < 9:
        return 0.0
    covered = [False] * len(words)
    for n in range(2, 9):
        counts: dict[tuple[str, ...], list[int]] = {}
        for i in range(len(words) - n + 1):
            counts.setdefault(tuple(words[i:i + n]), []).append(i)
        for starts in counts.values():
            if len(starts) >= 3:
                for i in starts:
                    for j in range(i, i + n):
                        covered[j] = True
    return sum(covered) / len(words)
