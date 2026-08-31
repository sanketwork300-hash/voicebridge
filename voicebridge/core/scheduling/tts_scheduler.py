"""Ordered playback scheduling for synthesised speech.

Two hard requirements drive this design:

1. **Order is absolute.** Segment 5 must never be heard before segment 4.
   Synthesis may run concurrently and finish out of order, so a reorder buffer
   is required; arrival order cannot be trusted.
2. **Backlog must never grow silently.** If TTS runs slower than real time the
   dub drifts steadily further behind the video. Accumulating that delay
   invisibly is the single worst failure mode for a dubbing system, because the
   output stays plausible while becoming completely desynchronised. The
   scheduler therefore measures backlog continuously and reports overload so the
   pipeline can degrade deliberately (drop to subtitles, shrink segments, use a
   faster voice) instead of drifting.

A third, less obvious requirement: a **failed** synthesis must not deadlock the
stream. If segment 4 errors, segment 5 is stuck behind a sequence number that
will never arrive. ``mark_failed`` closes that gap explicitly, and
``gap_timeout`` closes it as a backstop when a provider neither returns nor
raises.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from voicebridge.core.types import SynthesisedAudio

logger = logging.getLogger(__name__)


@dataclass
class SchedulerStats:
    emitted: int = 0
    pending: int = 0
    skipped: int = 0
    backlog_seconds: float = 0.0
    overloaded: bool = False
    next_expected: int = 1


class TTSScheduler:
    """Reorder buffer that releases synthesised audio strictly in sequence."""

    def __init__(
        self,
        first_sequence: int = 1,
        max_backlog_seconds: float = 6.0,
        gap_timeout: float = 5.0,
        max_pending: int = 64,
    ):
        self.next_expected = first_sequence
        self.max_backlog_seconds = max_backlog_seconds
        self.gap_timeout = gap_timeout
        self.max_pending = max_pending

        self._buffer: dict[int, SynthesisedAudio] = {}
        self._failed: set[int] = set()
        self._arrived_at: dict[int, float] = {}
        self._emitted = 0
        self._skipped = 0
        #: Total audio duration handed downstream but not yet finished playing.
        self._queued_audio_seconds = 0.0
        self._playhead: float | None = None

    # -- submission --------------------------------------------------------

    def submit(self, audio: SynthesisedAudio) -> list[SynthesisedAudio]:
        """Add synthesised audio; return whatever is now releasable, in order."""
        if audio.sequence_id < self.next_expected:
            # Late arrival for something already released or skipped: dropping
            # is correct, replaying it would corrupt the order.
            logger.debug("discarding late TTS segment %s", audio.sequence_id)
            return []
        if len(self._buffer) >= self.max_pending:
            # Bounded like every other buffer in the pipeline. Losing the
            # furthest-future segment is the least damaging choice.
            furthest = max(self._buffer)
            if furthest > audio.sequence_id:
                self._buffer.pop(furthest, None)
                self._arrived_at.pop(furthest, None)
                self._skipped += 1
            else:
                self._skipped += 1
                return []
        self._buffer[audio.sequence_id] = audio
        self._arrived_at[audio.sequence_id] = time.monotonic()
        return self._release()

    def mark_failed(self, sequence_id: int) -> list[SynthesisedAudio]:
        """Record that synthesis for ``sequence_id`` will never arrive."""
        if sequence_id >= self.next_expected:
            self._failed.add(sequence_id)
        return self._release()

    def tick(self) -> list[SynthesisedAudio]:
        """Advance time-based logic: close stale gaps, decay the backlog."""
        self._decay_backlog()
        released = self._release()
        if released:
            return released
        # Nothing released and something is waiting behind a hole that has been
        # open too long: give up on the missing sequence.
        if self._buffer and self.next_expected not in self._buffer:
            oldest_wait = min(
                (time.monotonic() - t for t in self._arrived_at.values()),
                default=0.0,
            )
            if oldest_wait >= self.gap_timeout:
                logger.warning(
                    "TTS sequence %s missing for %.1fs; skipping to keep audio flowing",
                    self.next_expected,
                    oldest_wait,
                )
                self._failed.add(self.next_expected)
                return self._release()
        return []

    # -- internals ---------------------------------------------------------

    def _release(self) -> list[SynthesisedAudio]:
        out: list[SynthesisedAudio] = []
        while True:
            seq = self.next_expected
            if seq in self._buffer:
                audio = self._buffer.pop(seq)
                self._arrived_at.pop(seq, None)
                out.append(audio)
                self._emitted += 1
                self._register_playback(audio.duration)
                self.next_expected += 1
                continue
            if seq in self._failed:
                self._failed.discard(seq)
                self._skipped += 1
                self.next_expected += 1
                continue
            break
        return out

    def _register_playback(self, duration: float) -> None:
        now = time.monotonic()
        if self._playhead is None or self._playhead < now:
            self._playhead = now
        self._playhead += max(0.0, duration)
        self._queued_audio_seconds = max(0.0, self._playhead - now)

    def _decay_backlog(self) -> None:
        if self._playhead is None:
            return
        self._queued_audio_seconds = max(0.0, self._playhead - time.monotonic())

    # -- observation -------------------------------------------------------

    @property
    def backlog_seconds(self) -> float:
        """How far behind real time the dubbed audio currently is."""
        self._decay_backlog()
        return self._queued_audio_seconds

    @property
    def overloaded(self) -> bool:
        return self.backlog_seconds >= self.max_backlog_seconds

    def stats(self) -> SchedulerStats:
        return SchedulerStats(
            emitted=self._emitted,
            pending=len(self._buffer),
            skipped=self._skipped,
            backlog_seconds=round(self.backlog_seconds, 3),
            overloaded=self.overloaded,
            next_expected=self.next_expected,
        )
