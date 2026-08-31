"""Turn a stream of ASR snapshots into an append-only stable transcript.

Streaming ASR backends report a *snapshot*: the text they have committed so far
plus a retractable hypothesis tail. WhisperLiveKit is explicit about this split
-- ``FrontData.lines`` holds committed segments and ``FrontData.buffer_transcription``
holds the unstable tail (``whisperlivekit/audio_processor.py::results_formatter``).
Backends using LocalAgreement or SimulStreaming decide *when* text moves from
tail to committed; that policy lives in the backend, not here.

This class does the part the backends do not: convert successive snapshots into
a monotonic stream of newly-stable text, so that downstream stages translate
each piece of source text exactly once.

It deliberately does **not** re-implement a commitment policy. Inventing one on
top of a backend that already has one produces double-buffering and latency
that is hard to attribute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from voicebridge.core.types import TranscriptSegment, TranscriptState, Word, now

logger = logging.getLogger(__name__)


@dataclass
class StabilizerUpdate:
    """Result of feeding one ASR snapshot to the stabiliser."""

    newly_stable: list[TranscriptSegment] = field(default_factory=list)
    partial_text: str = ""
    partial_changed: bool = False
    rolled_back: bool = False
    state: TranscriptState | None = None


class TranscriptStabilizer:
    """Tracks committed text across snapshots and emits only the new suffix.

    The backend is trusted about *what* is committed. Our job is bookkeeping:

    * emit each newly committed span once, in order;
    * notice when a backend retracts committed text (it should not, but a
      backend restart or a session reset can cause it) and report it rather than
      silently emitting duplicate or overlapping text;
    * keep the partial tail available for display without ever letting it reach
      the TTS path.
    """

    def __init__(self) -> None:
        self.state = TranscriptState()
        self._committed_segments: list[TranscriptSegment] = []
        self._committed_end: float = 0.0
        self._sequence = 0

    @property
    def committed_segments(self) -> list[TranscriptSegment]:
        return list(self._committed_segments)

    def update(
        self,
        stable_segments: list[TranscriptSegment],
        partial_text: str = "",
        language: str | None = None,
    ) -> StabilizerUpdate:
        """Feed one ASR snapshot.

        ``stable_segments`` is the backend's full list of committed segments
        (cumulative, as WhisperLiveKit's ``lines`` is). Segments already seen are
        recognised by their end timestamp and skipped.
        """
        result = StabilizerUpdate()
        rolled_back = False

        # A snapshot whose committed text ends earlier than what we already
        # emitted means the backend retracted. Report it; do not re-emit.
        if stable_segments:
            latest_end = max(seg.end for seg in stable_segments)
            if latest_end + 1e-6 < self._committed_end:
                rolled_back = True
                logger.warning(
                    "ASR retracted committed text (was %.2fs, now %.2fs)",
                    self._committed_end,
                    latest_end,
                )

        newly_stable: list[TranscriptSegment] = []
        for seg in stable_segments:
            if not seg.text or not seg.text.strip():
                continue
            # Strictly-after comparison on the sample clock is what makes this
            # idempotent under repeated identical snapshots.
            if seg.end <= self._committed_end + 1e-6:
                continue
            self._sequence += 1
            seg.sequence_id = self._sequence
            newly_stable.append(seg)
            self._committed_segments.append(seg)
            self._committed_end = max(self._committed_end, seg.end)

        partial = (partial_text or "").strip()
        result.partial_changed = partial != self.state.partial_text
        result.newly_stable = newly_stable
        result.partial_text = partial
        result.rolled_back = rolled_back

        if newly_stable:
            self.state.revision_id += 1
            # Accumulate incrementally: each join must see the text produced by
            # the previous segment, otherwise a batch of two segments loses the
            # separator between them.
            combined = self.state.committed_text
            for segment in newly_stable:
                combined += _join(combined, segment.text)
            self.state.committed_text = combined
            self.state.stable_text = self.state.committed_text
            words: list[Word] = []
            for s in newly_stable:
                words.extend(s.words)
            self.state.words.extend(words)
        self.state.partial_text = partial
        if language:
            self.state.language = language
        self.state.updated_at = now()
        result.state = self.state
        return result

    def reset(self) -> None:
        self.state = TranscriptState()
        self._committed_segments.clear()
        self._committed_end = 0.0


def _join(existing: str, addition: str) -> str:
    """Join transcript pieces, inserting a space only where scripts need one.

    CJK text is written without inter-word spaces; blindly joining with ' '
    corrupts Japanese and Korean transcripts and changes tokenisation for the
    translation model downstream.
    """
    addition = addition.strip()
    if not addition:
        return ""
    if not existing:
        return addition
    if existing.endswith((" ", "\n")):
        return addition
    if _is_cjk(existing[-1]) or _is_cjk(addition[0]):
        return addition
    return " " + addition


def _is_cjk(ch: str) -> bool:
    """True for scripts written *without* inter-word spacing.

    Japanese and Chinese are scriptio continua: inserting a space between two
    ASR segments corrupts the text and changes how the translation model
    tokenises it.

    Hangul is deliberately **excluded**. Korean is written with spaces between
    eojeol, so joining two Korean segments without one produces a single
    malformed token ("만나서영화를"), which degrades translation quality on a
    Tier-1 language pair. Korean therefore takes the normal space-joining path.
    """
    code = ord(ch)
    return (
        0x3000 <= code <= 0x303F      # CJK punctuation
        or 0x3040 <= code <= 0x30FF   # hiragana + katakana
        or 0x3400 <= code <= 0x4DBF   # CJK ext A
        or 0x4E00 <= code <= 0x9FFF   # CJK unified
        or 0xFF00 <= code <= 0xFFEF   # fullwidth forms
    )
