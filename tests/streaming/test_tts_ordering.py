"""TTS playback ordering and backlog accounting."""


from voicebridge.core.scheduling.tts_scheduler import TTSScheduler
from voicebridge.core.types import SynthesisedAudio


def audio(seq: int, duration: float = 0.5) -> SynthesisedAudio:
    return SynthesisedAudio(
        audio=b"\x00\x00" * int(16000 * duration),
        sample_rate=16000,
        channels=1,
        duration=duration,
        sequence_id=seq,
    )


def test_out_of_order_arrivals_are_reordered():
    s = TTSScheduler()
    assert s.submit(audio(3)) == []
    assert s.submit(audio(2)) == []
    released = s.submit(audio(1))
    assert [a.sequence_id for a in released] == [1, 2, 3]


def test_segment_five_never_precedes_segment_four():
    s = TTSScheduler()
    s.submit(audio(1))
    assert s.submit(audio(5)) == []
    assert s.submit(audio(3)) == []
    released = s.submit(audio(2))
    assert [a.sequence_id for a in released] == [2, 3]
    assert [a.sequence_id for a in s.submit(audio(4))] == [4, 5]


def test_failed_synthesis_does_not_deadlock_the_stream():
    s = TTSScheduler()
    s.submit(audio(1))
    assert s.submit(audio(3)) == []
    released = s.mark_failed(2)
    assert [a.sequence_id for a in released] == [3]
    assert s.stats().skipped == 1


def test_gap_timeout_releases_stalled_stream():
    s = TTSScheduler(gap_timeout=0.0)
    s.submit(audio(2))
    assert [a.sequence_id for a in s.tick()] == [2]


def test_late_arrival_is_discarded_not_replayed():
    s = TTSScheduler()
    s.submit(audio(1))
    s.submit(audio(2))
    assert s.submit(audio(1)) == []      # already played


def test_pending_buffer_is_bounded():
    s = TTSScheduler(max_pending=4)
    for seq in range(10, 30):
        s.submit(audio(seq))
    assert s.stats().pending <= 4


def test_backlog_reflects_queued_audio():
    s = TTSScheduler(max_backlog_seconds=1.0)
    s.submit(audio(1, duration=2.0))
    assert s.backlog_seconds > 1.0
    assert s.overloaded is True


def test_backlog_is_zero_before_any_audio():
    assert TTSScheduler().backlog_seconds == 0.0
