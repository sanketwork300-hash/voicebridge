"""Streaming ASR backed by WhisperLiveKit.

Why this backend
----------------
VoiceBridge does not implement its own streaming policy. Running Whisper
independently over fixed chunks and concatenating the results produces
duplicated and truncated words at every boundary, and no amount of
post-processing fixes it. WhisperLiveKit already implements the published
approaches to this problem (LocalAgreement-style commitment and
SimulStreaming/Simul-Whisper policies, with VAD and optional diarization), and
is Apache-2.0 licensed. We adapt it rather than reinvent it.

API used (verified against WhisperLiveKit 0.2.26, commit b781ce9)
-----------------------------------------------------------------
``whisperlivekit.TranscriptionEngine``
    Process-wide model holder. It is a **singleton**: constructing a second one
    returns the first (``core.py``). We honour that and share one engine across
    sessions, which is also what its own server does.
``whisperlivekit.AudioProcessor(transcription_engine=..., language=...,
    target_language=..., mode=..., pcm_input=...)``
    Per-session processor.
``await processor.create_tasks()``
    Returns an async generator of ``FrontData`` snapshots.
``await processor.process_audio(bytes)``
    Feeds audio. An empty/None message signals end of stream.

``FrontData`` (``whisperlivekit/timed_objects.py``) carries:
``lines``
    Cumulative list of committed ``Segment``s -> our ``ASR_STABLE``.
``buffer_transcription``
    The unstable hypothesis tail -> our ``ASR_PARTIAL``.

That committed/tail split is exactly VoiceBridge's stable/partial contract,
which is why the adapter is thin.

PCM input
---------
We always pass ``pcm_input=True``. WhisperLiveKit otherwise spawns FFmpeg to
decode a container, and FFmpeg is the most common missing dependency in
self-hosted setups. The browser extension already sends 16 kHz mono PCM, so the
decode step buys nothing.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from typing import Any

from voicebridge.core.types import (
    AudioChunk,
    Event,
    EventType,
)
from voicebridge.providers.base import ASRCapabilities, ASREngine, ProviderUnavailable
from voicebridge.providers.registry import asr_registry

logger = logging.getLogger(__name__)

#: Whisper's own language inventory is large; these are the ones VoiceBridge
#: lists as supported sources. Kept short deliberately -- we only advertise
#: what we have a segmentation profile or a Tier-1/2 translation path for.
ADVERTISED_LANGUAGES = [
    "auto", "en", "ja", "ko", "zh", "hi", "mr", "ta", "te", "bn", "gu", "kn",
    "ml", "pa", "es", "fr", "de", "pt", "ru", "ar",
]


class _Session:
    def __init__(self, processor: Any, language: str | None):
        self.processor = processor
        self.language = language
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.reader_task: asyncio.Task | None = None
        self.sequence = 0
        self.closed = False
        #: End timestamp of the last committed segment we forwarded, so
        #: cumulative snapshots are not re-emitted.
        self.committed_end = 0.0
        self.last_partial = ""
        self.announced_language: str | None = None

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence


class WhisperLiveKitASREngine(ASREngine):
    name = "whisperlivekit"

    def __init__(
        self,
        model: str = "small",
        language: str | None = None,
        backend: str | None = None,
        diarization: bool = False,
        vac: bool = True,
        min_chunk_size: float = 0.5,
        warmup_file: str | None = None,
        **extra: object,
    ):
        self.model = model
        self.language = language
        self.backend = backend
        self.diarization = diarization
        self.vac = vac
        self.min_chunk_size = min_chunk_size
        self.warmup_file = warmup_file
        self.extra = extra
        self._engine: Any = None
        self._sessions: dict[str, _Session] = {}
        self._engine_lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------

    @property
    def capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(
            name=self.name,
            streaming=True,
            language_detection=True,
            word_timestamps=False,
            diarization=self.diarization,
            languages=ADVERTISED_LANGUAGES,
            emits_partial_and_stable=True,
        )

    async def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        async with self._engine_lock:
            if self._engine is not None:
                return self._engine
            try:
                from whisperlivekit import TranscriptionEngine
            except ImportError as exc:
                raise ProviderUnavailable(
                    "whisperlivekit is not installed. Install the ASR extra:\n"
                    "    pip install 'voicebridge[asr-whisperlivekit]'\n"
                    f"Note: WhisperLiveKit requires Python >=3.11,<3.14; you are on "
                    f"{sys.version_info.major}.{sys.version_info.minor}. On 3.14+ the "
                    "extra is skipped by its environment marker, so the install "
                    "succeeds without providing it."
                ) from exc

            kwargs: dict[str, Any] = {
                "model": self.model,
                "diarization": self.diarization,
                "vac": self.vac,
                "min_chunk_size": self.min_chunk_size,
                # Always PCM: see module docstring.
                "pcm_input": True,
            }
            if self.language and self.language != "auto":
                kwargs["lan"] = self.language
            if self.backend:
                kwargs["backend"] = self.backend
            if self.warmup_file:
                kwargs["warmup_file"] = self.warmup_file
            kwargs.update(self.extra)  # type: ignore[arg-type]

            logger.info("loading WhisperLiveKit engine: %s", kwargs)
            # Model loading is blocking and can take tens of seconds; keep the
            # event loop responsive so health checks still answer.
            self._engine = await asyncio.to_thread(TranscriptionEngine, **kwargs)
            return self._engine

    async def warmup(self) -> None:
        await self._ensure_engine()

    async def start_session(
        self, session_id: str, language: str | None = None, **options: object
    ) -> None:
        engine = await self._ensure_engine()
        from whisperlivekit import AudioProcessor

        lang = language if language and language != "auto" else None
        processor = AudioProcessor(
            transcription_engine=engine,
            language=lang,
            mode="full",
            pcm_input=True,
        )
        session = _Session(processor, lang)
        self._sessions[session_id] = session
        results = await processor.create_tasks()
        session.reader_task = asyncio.create_task(
            self._read_results(session_id, session, results)
        )

    async def push_audio(self, session_id: str, chunk: AudioChunk) -> None:
        session = self._sessions.get(session_id)
        if session is None or session.closed:
            return
        await session.processor.process_audio(chunk.data)

    async def stop_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        session.closed = True
        try:
            # Empty message is WhisperLiveKit's documented end-of-stream signal.
            await session.processor.process_audio(b"")
        except Exception as exc:  # pragma: no cover - best effort teardown
            logger.debug("error signalling end of stream: %s", exc)
        if session.reader_task is not None:
            try:
                await asyncio.wait_for(session.reader_task, timeout=10)
            except (TimeoutError, asyncio.CancelledError):
                session.reader_task.cancel()
        try:
            await session.processor.cleanup()
        except Exception as exc:  # pragma: no cover
            logger.debug("error during processor cleanup: %s", exc)
        await self._put(session, Event(event_type=EventType.ASR_FINAL, payload={}))
        await self._put(session, None)  # type: ignore[arg-type]

    # -- result translation ------------------------------------------------

    async def _read_results(self, session_id: str, session: _Session, results: Any) -> None:
        try:
            async for front in results:
                await self._handle_front_data(session, front)
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception as exc:
            logger.exception("WhisperLiveKit result stream failed")
            await self._put(
                session,
                Event(
                    event_type=EventType.ERROR,
                    sequence_id=session.next_sequence(),
                    payload={"message": f"ASR stream failed: {exc}", "recoverable": False},
                ),
            )

    async def _handle_front_data(self, session: _Session, front: Any) -> None:
        error = getattr(front, "error", "")
        if error:
            await self._put(
                session,
                Event(
                    event_type=EventType.ERROR,
                    sequence_id=session.next_sequence(),
                    payload={"message": str(error), "recoverable": True},
                ),
            )
            return

        for line in getattr(front, "lines", []) or []:
            text = (getattr(line, "text", "") or "").strip()
            if not text:
                continue
            start = float(getattr(line, "start", 0.0) or 0.0)
            end = float(getattr(line, "end", 0.0) or 0.0)
            if end <= session.committed_end + 1e-6:
                continue  # already forwarded; snapshots are cumulative
            session.committed_end = end

            speaker = getattr(line, "speaker", None)
            # WhisperLiveKit uses -1 for "no diarization" and -2 for silence.
            if speaker is not None and int(speaker) < 0:
                speaker = None

            detected = getattr(line, "detected_language", None)
            if detected and detected != session.announced_language:
                session.announced_language = detected
                await self._put(
                    session,
                    Event(
                        event_type=EventType.LANGUAGE_DETECTED,
                        sequence_id=session.next_sequence(),
                        payload={"language": detected, "confidence": None},
                    ),
                )

            await self._put(
                session,
                Event(
                    event_type=EventType.ASR_STABLE,
                    sequence_id=session.next_sequence(),
                    payload={
                        "segment": {
                            "text": text,
                            "start": start,
                            "end": end,
                            "speaker": speaker,
                            "language": detected,
                        }
                    },
                ),
            )

        partial = (getattr(front, "buffer_transcription", "") or "").strip()
        if partial != session.last_partial:
            session.last_partial = partial
            await self._put(
                session,
                Event(
                    event_type=EventType.ASR_PARTIAL,
                    sequence_id=session.next_sequence(),
                    payload={"text": partial, "start": session.committed_end, "end": session.committed_end},
                ),
            )

    async def _put(self, session: _Session, event: Event | None) -> None:
        try:
            session.queue.put_nowait(event)
        except asyncio.QueueFull:
            # Never drop a stable/final result: drain a partial to make room.
            drained = None
            try:
                drained = session.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            if drained is not None and drained.event_type not in (
                EventType.ASR_PARTIAL,
            ):
                logger.warning("ASR event queue overflow dropped %s", drained.event_type)
            try:
                session.queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover
                logger.error("ASR event queue still full; dropping event")

    async def get_events(self, session_id: str) -> AsyncIterator[Event]:
        session = self._sessions.get(session_id)
        if session is None:
            return
        while True:
            event = await session.queue.get()
            if event is None:
                return
            yield event


asr_registry.register("whisperlivekit", WhisperLiveKitASREngine)
