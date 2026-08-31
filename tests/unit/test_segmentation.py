"""Language-aware translation segmentation."""

import pytest

from voicebridge.core.segmentation.profiles import get_profile
from voicebridge.core.segmentation.segmenter import TranslationSegmenter
from voicebridge.core.types import LatencyProfile
from voicebridge.core.types import TranscriptSegment as Seg


def feed(segmenter, words, start=0.0, step=0.3, language="en"):
    """Feed word-by-word and return every flushed unit."""
    out, t = [], start
    for word in words:
        out.extend(segmenter.add(Seg(word, t, t + step, language=language)))
        t += step
    return out


def test_individual_words_are_not_translated_separately():
    seg = TranslationSegmenter("en")
    flushed = feed(seg, "I think the main reason is that we should wait.".split())
    assert len(flushed) == 1
    assert flushed[0].text == "I think the main reason is that we should wait."


def test_short_text_is_held_below_minimum_length():
    seg = TranslationSegmenter("en")
    assert feed(seg, ["Yes."]) == []          # 4 chars < minimum_stable_chars
    assert seg.pending_text == "Yes."


def test_punctuation_flushes_english():
    seg = TranslationSegmenter("en")
    flushed = seg.add(Seg("This is a complete sentence.", 0.0, 2.0))
    assert len(flushed) == 1
    assert flushed[0].reason == "punctuation"


class TestJapanese:
    """Japanese places tense/negation last; a clause must not flush early."""

    def test_clause_without_ending_is_held(self):
        seg = TranslationSegmenter("ja")
        assert seg.add(Seg("私は昨日映画館に行って", 0.0, 2.0)) == []

    def test_sentence_final_verb_flushes(self):
        seg = TranslationSegmenter("ja")
        seg.add(Seg("私は昨日映画館に行って", 0.0, 2.0))
        flushed = seg.add(Seg("友達と会いました", 2.0, 3.5))
        assert len(flushed) == 1
        assert flushed[0].reason == "sentence_final"
        assert flushed[0].text == "私は昨日映画館に行って友達と会いました"

    def test_punctuation_also_flushes(self):
        seg = TranslationSegmenter("ja")
        flushed = seg.add(Seg("これは重要な点だと考えている。", 0.0, 2.0))
        assert len(flushed) == 1

    def test_low_latency_never_disables_sentence_final_bias(self):
        """Speed must not be bought by translating incomplete Japanese."""
        seg = TranslationSegmenter("ja", latency=LatencyProfile.LOW_LATENCY)
        assert seg.profile.sentence_final_bias is True
        assert seg.add(Seg("私は昨日映画館に行って", 0.0, 2.0)) == []


class TestKorean:
    def test_clause_without_ending_is_held(self):
        seg = TranslationSegmenter("ko")
        assert seg.add(Seg("어제 친구를 만나서", 0.0, 1.5)) == []

    def test_sentence_final_ending_flushes(self):
        seg = TranslationSegmenter("ko")
        seg.add(Seg("어제 친구를 만나서", 0.0, 1.5))
        flushed = seg.add(Seg("영화를 봤습니다", 1.5, 3.0))
        assert len(flushed) == 1
        assert flushed[0].text == "어제 친구를 만나서 영화를 봤습니다"


def test_max_chars_is_a_hard_safety_valve():
    """A speaker who never pauses or punctuates must still produce output."""
    seg = TranslationSegmenter("ja")
    flushed = feed(seg, ["あの時に"] * 40, language="ja")
    assert flushed, "safety valve never fired"
    assert flushed[0].reason == "max_chars"


def test_max_seconds_is_a_hard_safety_valve():
    seg = TranslationSegmenter("en")
    flushed = seg.add(Seg("a long stretch of speech without any punctuation", 0.0, 30.0))
    assert flushed and flushed[0].reason == "max_seconds"


def test_pause_closes_a_segment():
    seg = TranslationSegmenter("en")
    seg.add(Seg("This is long enough to flush", 0.0, 1.0))
    flushed = seg.add(Seg("and this follows later", 5.0, 6.0))
    assert flushed and flushed[0].reason == "pause"


def test_speaker_change_is_a_hard_boundary():
    seg = TranslationSegmenter("en")
    seg.add(Seg("First speaker says something", 0.0, 1.0, speaker=1))
    flushed = seg.add(Seg("second speaker replies", 1.0, 2.0, speaker=2))
    assert flushed and flushed[0].reason == "speaker_change"
    assert flushed[0].speaker == 1


def test_flush_emits_pending_text():
    seg = TranslationSegmenter("en")
    seg.add(Seg("incomplete", 0.0, 1.0))
    flushed = seg.flush("session_stop")
    assert flushed and flushed[0].text == "incomplete"
    assert seg.flush() == []          # nothing left


def test_language_switch_changes_profile():
    seg = TranslationSegmenter("en")
    assert seg.profile.sentence_final_bias is False
    seg.set_language("ja")
    assert seg.profile.sentence_final_bias is True


@pytest.mark.parametrize("latency,expected_order", [
    (LatencyProfile.LOW_LATENCY, "shorter"),
    (LatencyProfile.ACCURATE, "longer"),
])
def test_latency_profile_scales_waiting(latency, expected_order):
    base = get_profile("en")
    scaled = base.scaled(latency)
    if expected_order == "shorter":
        assert scaled.pause_threshold_ms < base.pause_threshold_ms
    else:
        assert scaled.pause_threshold_ms > base.pause_threshold_ms
