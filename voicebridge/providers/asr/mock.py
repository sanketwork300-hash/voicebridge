"""Deterministic mock ASR.

Purpose: let the entire pipeline, protocol and UI be exercised with no GPU, no
model download and no network. Mock mode is a first-class requirement, not a
test fixture -- ``docker compose --profile mock up`` must produce a working
demo on a clean machine.

Behaviour is intentionally *realistic where it matters*: it emits a growing
partial hypothesis that is then committed as stable text, so the transcript
stabiliser, the segmenter and the partial-vs-committed subtitle distinction are
all genuinely exercised rather than bypassed.

It is **not** speech recognition. It ignores the audio content entirely and
replays a fixed script, advancing on the session's sample clock. Nothing it
produces says anything about real ASR accuracy.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from voicebridge.core.types import (
    AudioChunk,
    Event,
    EventType,
)
from voicebridge.providers.base import ASRCapabilities, ASREngine
from voicebridge.providers.registry import asr_registry

#: Scripts keyed by language. Each entry is one utterance.
SCRIPTS: dict[str, list[str]] = {
    "en": [
        "I think the main reason is that we underestimated the problem.",
        "But once the team started measuring it, the picture changed completely.",
        "So the plan for next quarter is much simpler than before.",
    ],
    "ja": [
        "今日はお集まりいただきありがとうございます。",
        "この作品の一番の魅力はキャラクターの関係性だと思います。",
        "五条悟の領域展開は本当に印象的でした。",
    ],
    "ko": [
        "안녕하세요 여러분 오늘도 방송에 와주셔서 감사합니다.",
        "이번 앨범은 정말 오래 준비한 작업이었습니다.",
        "방탄소년단의 무대를 다시 보고 싶습니다.",
    ],
}

DEFAULT_SCRIPT = SCRIPTS["en"]

#: Seconds of audio consumed before one more word is revealed. Roughly a
#: natural speaking rate, so mock latency numbers are not absurd.
SECONDS_PER_WORD = 0.32


class _MockSession:
    def __init__(self, language: str):
        self.language = language
        self.script = SCRIPTS.get((language or "en").split("-")[0], DEFAULT_SCRIPT)
        self.utterance_index = 0
        self.word_index = 0
        self.audio_seconds = 0.0
        self.committed_end = 0.0
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.sequence = 0
        self.closed = False
        self.announced_language = False

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence


class MockASREngine(ASREngine):
    name = "mock"

    def __init__(self, language: str | None = None, **_: object):
        self._default_language = language or "en"
        self._sessions: dict[str, _MockSession] = {}

    @property
    def capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(
            name=self.name,
            streaming=True,
            language_detection=True,
            word_timestamps=False,
            diarization=False,
            languages=sorted(SCRIPTS),
            emits_partial_and_stable=True,
        )

    async def start_session(
        self, session_id: str, language: str | None = None, **options: object
    ) -> None:
        lang = language if language and language != "auto" else self._default_language
        self._sessions[session_id] = _MockSession(lang or "en")

    async def push_audio(self, session_id: str, chunk: AudioChunk) -> None:
        session = self._sessions.get(session_id)
        if session is None or session.closed:
            return
        session.audio_seconds += chunk.duration

        if not session.announced_language:
            session.announced_language = True
            await self._emit(
                session,
                EventType.LANGUAGE_DETECTED,
                {"language": session.language, "confidence": 0.99},
            )

        # Reveal words at a fixed rate against the session's audio clock.
        target_words = int(session.audio_seconds / SECONDS_PER_WORD)
        while session.word_index < target_words and not session.closed:
            await self._advance(session)

    async def _advance(self, session: _MockSession) -> None:
        if session.utterance_index >= len(session.script):
            # Loop the script so long streams keep producing output.
            session.utterance_index = 0
        utterance = session.script[session.utterance_index]
        words = _split(utterance, session.language)
        local_index = session.word_index - _words_before(
            session.script, session.utterance_index, session.language
        )

        if local_index >= len(words):
            # Utterance complete: commit it as stable text.
            start = session.committed_end
            end = session.audio_seconds
            session.committed_end = end
            await self._emit(
                session,
                EventType.ASR_STABLE,
                {
                    "segment": {
                        "text": utterance,
                        "start": start,
                        "end": end,
                        "language": session.language,
                    }
                },
            )
            session.utterance_index += 1
            session.word_index += 1  # consume the boundary
            return

        partial = _join_words(words[: local_index + 1], session.language)
        session.word_index += 1
        await self._emit(
            session,
            EventType.ASR_PARTIAL,
            {"text": partial, "start": session.committed_end, "end": session.audio_seconds},
        )

    async def _emit(self, session: _MockSession, event_type: EventType, payload: dict) -> None:
        event = Event(
            event_type=event_type,
            sequence_id=session.next_sequence(),
            payload=payload,
        )
        try:
            session.queue.put_nowait(event)
        except asyncio.QueueFull:
            # Bounded like everywhere else: drop the oldest partial.
            try:
                session.queue.get_nowait()
                session.queue.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    async def get_events(self, session_id: str) -> AsyncIterator[Event]:
        session = self._sessions.get(session_id)
        if session is None:
            return
        while True:
            event = await session.queue.get()
            if event is None:  # sentinel
                return
            yield event

    async def stop_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        session.closed = True
        try:
            session.queue.put_nowait(None)  # type: ignore[arg-type]
        except asyncio.QueueFull:
            pass


def _split(text: str, language: str) -> list[str]:
    """Split into reveal units: words for spaced scripts, characters for CJK."""
    if language.startswith("ja") or language.startswith("zh"):
        return list(text)
    return text.split()


def _join_words(units: list[str], language: str) -> str:
    if language.startswith("ja") or language.startswith("zh"):
        return "".join(units)
    return " ".join(units)


def _words_before(script: list[str], index: int, language: str) -> int:
    # +1 per completed utterance for the boundary token consumed on commit.
    return sum(len(_split(script[i], language)) + 1 for i in range(index))


asr_registry.register("mock", MockASREngine)
