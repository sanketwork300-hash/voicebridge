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
``whisperlivekit.TranscriptionEngine(**kwargs)``
    Process-wide model holder. It is a **singleton**: constructing a second one
    returns the first (``core.py``). We honour that and share one engine across
    sessions, which is also what its own server does. Keyword arguments are
    the field names of ``WhisperLiveKitConfig`` (``config.py``): ``model_size``,
    ``lan``, ``backend``, ``vac``, ``min_chunk_size``, ``diarization``,
    ``pcm_input``, ``warmup_file``. Unknown keys are dropped with a warning,
    not rejected.
``whisperlivekit.AudioProcessor(transcription_engine=..., language=...,
    target_language=..., mode=..., pcm_input=...)``
    Per-session processor.
``await processor.create_tasks()``
    Returns an async generator of ``FrontData`` snapshots.
``await processor.process_audio(bytes)``
    Feeds audio. An empty/None message signals end of stream.

``FrontData`` (``whisperlivekit/timed_objects.py``) carries:
``lines``
    Cumulative list of ``Segment``s. **Grouping is not commitment.** Read from
    ``tokens_alignment.py::get_lines``: committed tokens accumulate in
    ``current_line_tokens`` and are appended as the *last* element of
    ``lines`` on every snapshot, with the same ``start`` and a growing ``end``
    and ``text``; that line moves to ``validated_segments`` only when a
    ``Silence`` token arrives, and ``audio_processor.py`` sets
    ``MIN_DURATION_REAL_SILENCE = 5`` seconds for that. Silence gaps appear
    as segments with ``speaker == -2`` and empty text. The *tokens* inside a
    line are committed by the policy (LocalAgreement's
    ``committed_in_buffer`` in ``local_agreement/online_asr.py``; AlignAtt's
    emitted tokens for SimulStreaming) and are not revised afterwards.
``buffer_transcription``
    The unstable hypothesis tail beyond the committed tokens.

Mapping to VoiceBridge's contract therefore is per *line index*: the text a
line has grown by since the previous snapshot -> ``ASR_STABLE`` (so stable
text flows token by token, and the segmenter decides translation units);
``buffer_transcription`` -> ``ASR_PARTIAL``. Forwarding whole lines whenever
their end time moves -- the obvious reading of the field name -- re-emits the
entire growing sentence on every snapshot, and waiting for the line to close
holds every sentence back until a five-second pause.

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
        #: Text already forwarded per line index in ``FrontData.lines``.
        #: Lines are cumulative and keep their index, so the delta since the
        #: last snapshot is what is new.
        self.forwarded: dict[int, str] = {}
        self.revisions = 0

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

            # Field names are those of ``WhisperLiveKitConfig`` (``config.py``).
            # ``from_kwargs`` silently drops unknown keys with only a log
            # warning, so a wrong name here does not fail -- it quietly loads
            # the default ``base`` model. The size field is ``model_size``,
            # not ``model``; ``lan`` is the language field.
            kwargs: dict[str, Any] = {
                "model_size": self.model,
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

        for index, line in enumerate(getattr(front, "lines", []) or []):
            if _is_silence(line):
                continue
            fields = _line_fields(line)
            text = fields["text"]
            if not text:
                continue
            seen = session.forwarded.get(index, "")
            if text == seen:
                continue
            if seen and not text.startswith(seen):
                # The policy is not supposed to revise committed tokens. If it
                # does, never re-emit what was already forwarded: emit only the
                # part beyond the common prefix and count the revision.
                common = _common_prefix_len(seen, text)
                session.revisions += 1
                logger.warning(
                    "WhisperLiveKit revised committed text on line %d (%d so far): %r -> %r",
                    index, session.revisions, seen[-40:], text[-40:],
                )
                delta = text[common:].strip()
            else:
                delta = text[len(seen):].strip()
            session.forwarded[index] = text
            if not delta:
                continue
            start = fields["start"] if not seen else session.committed_end
            end = fields["end"]
            if end <= session.committed_end + 1e-6:
                # Same end time as the last delta (a trailing punctuation mark
                # attached to the final token). The stabiliser keys on end
                # time, so nudge it forward rather than lose the terminator
                # the segmenter needs.
                end = session.committed_end + 1e-3
            await self._commit(session, delta, start, end, fields)

        partial = (getattr(front, "buffer_transcription", "") or "").strip()
        if partial != session.last_partial:
            session.last_partial = partial
            await self._put(
                session,
                Event(
                    event_type=EventType.ASR_PARTIAL,
                    sequence_id=session.next_sequence(),
                    payload={
                        "text": partial,
                        "start": session.committed_end,
                        "end": session.committed_end,
                    },
                ),
            )

    async def _commit(
        self, session: _Session, text: str, start: float, end: float, fields: dict[str, Any]
    ) -> None:
        session.committed_end = end
        detected = fields["language"]
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
                        "speaker": fields["speaker"],
                        "language": detected,
                    }
                },
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


def _is_silence(line: Any) -> bool:
    """WhisperLiveKit marks silence segments with speaker -2 (``timed_objects.py``)."""
    speaker = getattr(line, "speaker", None)
    try:
        return speaker is not None and int(speaker) == -2
    except (TypeError, ValueError):
        return False


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _line_fields(line: Any) -> dict[str, Any]:
    speaker = getattr(line, "speaker", None)
    # -1 means "no diarization"; only non-negative ids are real speakers.
    try:
        speaker = int(speaker) if speaker is not None and int(speaker) >= 0 else None
    except (TypeError, ValueError):
        speaker = None
    return {
        "text": (getattr(line, "text", "") or "").strip(),
        "start": float(getattr(line, "start", 0.0) or 0.0),
        "end": float(getattr(line, "end", 0.0) or 0.0),
        "speaker": speaker,
        "language": getattr(line, "detected_language", None),
    }


asr_registry.register("whisperlivekit", WhisperLiveKitASREngine)
