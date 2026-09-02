"""The WhisperLiveKit adapter's snapshot handling, without WhisperLiveKit.

``FrontData.lines`` is cumulative; its trailing text segment keeps growing
until a five-second silence closes it (``tokens_alignment.py::get_lines``,
``audio_processor.py::MIN_DURATION_REAL_SILENCE``), while the tokens inside it
are already committed by the policy. These tests pin the mapping of that shape
onto VoiceBridge's stable/partial contract using plain stand-ins for
``Segment`` and ``FrontData``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from voicebridge.core.types import EventType
from voicebridge.providers.asr.whisperlivekit import WhisperLiveKitASREngine, _Session


@dataclass
class Seg:
    start: float
    end: float
    text: str | None
    speaker: int = -1
    detected_language: str | None = None


@dataclass
class Front:
    lines: list = field(default_factory=list)
    buffer_transcription: str = ""
    error: str = ""


class _Proc:
    sep = " "


def _drain(session: _Session) -> list:
    out = []
    while not session.queue.empty():
        out.append(session.queue.get_nowait())
    return out


def _stable_texts(events) -> list[str]:
    return [e.payload["segment"]["text"] for e in events if e.event_type is EventType.ASR_STABLE]


@pytest.fixture
def engine_and_session():
    engine = WhisperLiveKitASREngine()
    session = _Session(_Proc(), "en")
    return engine, session


def test_growing_line_emits_only_new_text(engine_and_session):
    engine, session = engine_and_session
    run = asyncio.run

    run(engine._handle_front_data(session, Front([Seg(0.1, 2.0, "I think the")], "main")))
    events = _drain(session)
    assert _stable_texts(events) == ["I think the"]
    assert [e.payload["text"] for e in events if e.event_type is EventType.ASR_PARTIAL] == ["main"]
    first = [e for e in events if e.event_type is EventType.ASR_STABLE][0].payload["segment"]
    assert (first["start"], first["end"]) == (0.1, 2.0)

    # Same line, grown: only the suffix is stable, and it starts where the
    # previous delta ended so the segmenter sees no pause.
    run(engine._handle_front_data(session, Front([Seg(0.1, 4.7, "I think the main reason")], "")))
    events = _drain(session)
    assert _stable_texts(events) == ["main reason"]
    seg = [e for e in events if e.event_type is EventType.ASR_STABLE][0].payload["segment"]
    assert (seg["start"], seg["end"]) == (2.0, 4.7)

    # Identical snapshot: nothing.
    run(engine._handle_front_data(session, Front([Seg(0.1, 4.7, "I think the main reason")], "")))
    assert _drain(session) == []


def test_terminator_with_unchanged_end_is_not_lost(engine_and_session):
    engine, session = engine_and_session
    run = asyncio.run
    run(engine._handle_front_data(session, Front([Seg(0.1, 4.7, "simpler than before")], "")))
    _drain(session)
    run(engine._handle_front_data(session, Front([Seg(0.1, 4.7, "simpler than before.")], "")))
    events = _drain(session)
    assert _stable_texts(events) == ["."]
    seg = [e for e in events if e.event_type is EventType.ASR_STABLE][0].payload["segment"]
    assert seg["end"] > 4.7


def test_silence_and_new_line_after_it(engine_and_session):
    engine, session = engine_and_session
    run = asyncio.run
    run(engine._handle_front_data(session, Front([Seg(0.1, 4.7, "First sentence.")], "")))
    _drain(session)

    closed = [Seg(0.1, 4.7, "First sentence."), Seg(4.7, 10.0, None, speaker=-2)]
    run(engine._handle_front_data(session, Front(closed, "")))
    assert _drain(session) == []  # silence segments are never text

    run(engine._handle_front_data(session, Front([*closed, Seg(10.0, 11.0, "Second")], "one")))
    events = _drain(session)
    assert _stable_texts(events) == ["Second"]
    seg = [e for e in events if e.event_type is EventType.ASR_STABLE][0].payload["segment"]
    assert seg["start"] == 10.0  # a real pause the segmenter can see


def test_revised_committed_text_is_not_duplicated(engine_and_session):
    engine, session = engine_and_session
    run = asyncio.run
    run(engine._handle_front_data(session, Front([Seg(0.0, 1.0, "we under estimated")], "")))
    _drain(session)
    run(engine._handle_front_data(session, Front([Seg(0.0, 2.0, "we underestimated it")], "")))
    events = _drain(session)
    assert _stable_texts(events) == ["estimated it"]
    assert session.revisions == 1


def test_language_announced_once_and_speakers_normalised(engine_and_session):
    engine, session = engine_and_session
    run = asyncio.run
    lines = [
        Seg(0.0, 1.0, "こんにちは。", speaker=-1, detected_language="ja"),
        Seg(1.0, 2.0, "元気ですか。", speaker=0, detected_language="ja"),
        Seg(2.0, 3.0, None, speaker=-2),
    ]
    run(engine._handle_front_data(session, Front(lines, "")))
    events = _drain(session)
    kinds = [e.event_type for e in events]
    assert kinds.count(EventType.LANGUAGE_DETECTED) == 1
    stables = [e for e in events if e.event_type is EventType.ASR_STABLE]
    assert [s.payload["segment"]["speaker"] for s in stables] == [None, 0]
