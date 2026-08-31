"""Transcript stabilisation: newly-stable text must be emitted exactly once."""

from voicebridge.core.transcript.stabilizer import TranscriptStabilizer, _is_cjk, _join
from voicebridge.core.types import TranscriptSegment as Seg


def test_cumulative_snapshots_emit_each_segment_once():
    st = TranscriptStabilizer()
    first = st.update([Seg("Hello there", 0.0, 1.0)], "how")
    repeat = st.update([Seg("Hello there", 0.0, 1.0)], "how are")
    second = st.update(
        [Seg("Hello there", 0.0, 1.0), Seg("how are you", 1.0, 2.0)], ""
    )
    assert [s.text for s in first.newly_stable] == ["Hello there"]
    assert repeat.newly_stable == []          # idempotent
    assert [s.text for s in second.newly_stable] == ["how are you"]
    assert st.state.committed_text == "Hello there how are you"


def test_partial_changes_are_reported_without_committing():
    st = TranscriptStabilizer()
    update = st.update([], "I think")
    assert update.newly_stable == []
    assert update.partial_text == "I think"
    assert update.partial_changed is True
    assert st.state.committed_text == ""


def test_rollback_is_detected_and_reported():
    st = TranscriptStabilizer()
    st.update([Seg("one two three", 0.0, 3.0)], "")
    rolled = st.update([Seg("one", 0.0, 1.0)], "")
    assert rolled.rolled_back is True
    assert rolled.newly_stable == []          # never re-emit retracted text


def test_empty_segments_are_ignored():
    st = TranscriptStabilizer()
    update = st.update([Seg("   ", 0.0, 1.0), Seg("", 1.0, 2.0)], "")
    assert update.newly_stable == []


def test_revision_id_advances_only_on_commit():
    st = TranscriptStabilizer()
    st.update([], "partial")
    assert st.state.revision_id == 0
    st.update([Seg("committed", 0.0, 1.0)], "")
    assert st.state.revision_id == 1


def test_sequence_ids_are_monotonic():
    st = TranscriptStabilizer()
    st.update([Seg("a", 0, 1)], "")
    st.update([Seg("a", 0, 1), Seg("b", 1, 2)], "")
    st.update([Seg("a", 0, 1), Seg("b", 1, 2), Seg("c", 2, 3)], "")
    ids = [s.sequence_id for s in st.committed_segments]
    assert ids == sorted(ids) and len(set(ids)) == 3


class TestScriptAwareJoining:
    """CJK is scriptio continua; Korean and Latin are not."""

    def test_japanese_joins_without_space(self):
        st = TranscriptStabilizer()
        st.update([Seg("こんにちは", 0, 1), Seg("世界です", 1, 2)], "")
        assert st.state.committed_text == "こんにちは世界です"

    def test_korean_keeps_word_space(self):
        st = TranscriptStabilizer()
        st.update([Seg("어제 친구를", 0, 1), Seg("만났습니다", 1, 2)], "")
        assert st.state.committed_text == "어제 친구를 만났습니다"

    def test_english_keeps_space(self):
        st = TranscriptStabilizer()
        st.update([Seg("hello", 0, 1), Seg("world", 1, 2)], "")
        assert st.state.committed_text == "hello world"

    def test_hangul_is_not_treated_as_scriptio_continua(self):
        assert _is_cjk("あ") is True
        assert _is_cjk("漢") is True
        assert _is_cjk("한") is False
        assert _is_cjk("a") is False

    def test_join_does_not_double_space(self):
        assert _join("hello ", "world") == "world"
