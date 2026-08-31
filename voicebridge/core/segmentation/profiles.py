"""Per-language segmentation profiles.

Why this exists
---------------
Translating every stable ASR fragment produces both bad translations and
wasted compute. The correct unit is a *meaningful language unit*, and what
counts as one is language-dependent.

For Japanese and Korean this is not a tuning detail, it is the difference
between usable and unusable output: both languages place tense, negation,
politeness and question/statement mood in sentence-final position. A clause
translated before its ending is not merely rougher -- it can assert the
opposite of what the speaker said ("he will go" vs "he won't go"). So the
profiles for ``ja`` and ``ko`` bias strongly toward waiting for a sentence-final
signal or a real pause.

Status of these numbers
-----------------------
The thresholds below are *defaults chosen from the linguistic argument above,
not from measurement*. They have not been benchmarked. ``benchmarks/`` contains
the harness intended to tune them; until it has been run against real audio,
treat every value here as a starting point and expect to change it. They are
config, not constants, for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from voicebridge.core.types import LatencyProfile

# Sentence-terminating punctuation, by script.
LATIN_TERMINATORS: set[str] = {".", "!", "?"}
CJK_TERMINATORS: set[str] = {"。", "！", "？", "…", "‥"}
ALL_TERMINATORS: set[str] = LATIN_TERMINATORS | CJK_TERMINATORS

# Clause-level punctuation: a weaker boundary, usable when a segment is already
# long and we would otherwise flush mid-phrase.
SOFT_BREAKS: set[str] = {",", "、", "，", ";", "；", ":", "："}

# Japanese sentence-final forms. Whisper-family models emit Japanese with
# inconsistent punctuation, so punctuation alone is not a reliable boundary;
# these endings are a secondary signal.
JA_SENTENCE_FINAL: tuple = (
    "です", "ます", "ました", "ません", "でした", "だった", "ですね", "ますね",
    "でしょう", "ましょう", "ください", "だよ", "だね", "かな", "のだ", "んだ",
)

# Korean sentence-final endings (declarative, polite, interrogative).
KO_SENTENCE_FINAL: tuple = (
    "습니다", "ㅂ니다", "습니까", "예요", "에요", "이에요", "아요", "어요",
    "해요", "네요", "지요", "죠", "는데요", "거든요", "잖아요", "십시오",
)


@dataclass
class LanguageSegmentationProfile:
    """Rules governing when accumulated stable text is flushed for translation."""

    language: str

    #: A silence at least this long is treated as an utterance boundary.
    pause_threshold_ms: int = 700

    #: Never flush a segment shorter than this (avoids one-word translations).
    minimum_stable_chars: int = 12

    #: Flush unconditionally once the buffer reaches this length -- the safety
    #: valve that stops a speaker who never pauses from producing no output.
    max_chars: int = 220

    #: Flush unconditionally after this long, regardless of content.
    max_seconds: float = 8.0

    #: Treat sentence-terminating punctuation as an immediate boundary.
    punctuation_flush: bool = True

    #: Require a sentence-final signal (punctuation, a known ending, or a pause)
    #: before flushing, even when the buffer is long. Set for ja/ko.
    sentence_final_bias: bool = False

    #: Endings that count as sentence-final when ``sentence_final_bias`` is set.
    sentence_final_endings: tuple = ()

    #: Characters-per-second used to convert profile lengths across scripts.
    #: CJK packs far more meaning per character than Latin script does.
    dense_script: bool = False

    def scaled(self, latency: LatencyProfile) -> LanguageSegmentationProfile:
        """Return this profile adjusted for a latency preference.

        Only the *waiting* parameters move. ``sentence_final_bias`` is never
        switched off by LOW_LATENCY, because doing so is what produces
        confidently-wrong Japanese and Korean translations.
        """
        if latency is LatencyProfile.LOW_LATENCY:
            factor, char_factor = 0.6, 0.55
        elif latency is LatencyProfile.ACCURATE:
            factor, char_factor = 1.6, 1.5
        else:
            factor, char_factor = 1.0, 1.0
        return LanguageSegmentationProfile(
            language=self.language,
            pause_threshold_ms=int(self.pause_threshold_ms * factor),
            minimum_stable_chars=max(1, int(self.minimum_stable_chars * char_factor)),
            max_chars=max(16, int(self.max_chars * char_factor)),
            max_seconds=round(self.max_seconds * factor, 2),
            punctuation_flush=self.punctuation_flush,
            sentence_final_bias=self.sentence_final_bias,
            sentence_final_endings=self.sentence_final_endings,
            dense_script=self.dense_script,
        )


DEFAULT_PROFILE = LanguageSegmentationProfile(language="default")

_PROFILES: dict[str, LanguageSegmentationProfile] = {
    "en": LanguageSegmentationProfile(
        language="en",
        pause_threshold_ms=600,
        minimum_stable_chars=14,
        max_chars=240,
        max_seconds=7.0,
    ),
    # Japanese: wait for the verb. Shorter char thresholds because Japanese
    # carries more content per character than Latin script.
    "ja": LanguageSegmentationProfile(
        language="ja",
        pause_threshold_ms=900,
        minimum_stable_chars=8,
        max_chars=90,
        max_seconds=9.0,
        punctuation_flush=True,
        sentence_final_bias=True,
        sentence_final_endings=JA_SENTENCE_FINAL,
        dense_script=True,
    ),
    # Korean: same reasoning; spaces exist but mood/negation are still final.
    "ko": LanguageSegmentationProfile(
        language="ko",
        pause_threshold_ms=900,
        minimum_stable_chars=10,
        max_chars=110,
        max_seconds=9.0,
        punctuation_flush=True,
        sentence_final_bias=True,
        sentence_final_endings=KO_SENTENCE_FINAL,
        dense_script=True,
    ),
    "zh": LanguageSegmentationProfile(
        language="zh",
        pause_threshold_ms=800,
        minimum_stable_chars=8,
        max_chars=90,
        max_seconds=8.0,
        dense_script=True,
    ),
    "hi": LanguageSegmentationProfile(
        language="hi", pause_threshold_ms=650, minimum_stable_chars=12, max_chars=220
    ),
}


def get_profile(language: str | None) -> LanguageSegmentationProfile:
    """Look up a profile, falling back to a conservative default."""
    if not language:
        return DEFAULT_PROFILE
    key = language.lower().split("-")[0].split("_")[0]
    return _PROFILES.get(key, DEFAULT_PROFILE)


def register_profile(profile: LanguageSegmentationProfile) -> None:
    """Override or add a profile (used by config loading and benchmarks)."""
    _PROFILES[profile.language.lower()] = profile
