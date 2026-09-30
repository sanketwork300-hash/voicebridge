"""Realtime pipeline for direct speech-to-speech engines (Meta SeamlessStreaming).

Same public surface as :class:`~voicebridge.core.pipeline.pipeline.TranslationPipeline`
(``start``/``stop``/``push_audio``/``stream_events``/``queue_stats``), so the
WebSocket protocol, the browser demo and the extension work unchanged; only the
engine behind it differs. No Whisper, no LLM translation and no cascade TTS run
here. Speech gating is the engine's own business (SeamlessStreaming's agent has
its own Silero VAD stage).

Events emitted: ``SESSION_STARTED``, ``TRANSLATION_FINAL`` (text pieces as the
agent emits them; ``start``/``end`` are emission offsets on the source clock),
``TTS_AUDIO`` (translated speech as it is produced), ``WARNING``/``ERROR`` and
``SESSION_ENDED``. If the engine fails the session reports
``SEAMLESS_MODEL_ERROR``; it never falls back to the cascade silently.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from typing import Any

from voicebridge.core.metrics.metrics import SessionMetrics
from voicebridge.core.session.config import SessionConfig
from voicebridge.core.types import AudioChunk, Event, EventType, now
from voicebridge.providers.base import ProviderError

logger = logging.getLogger(__name__)


class S2STRealtimePipeline:
    def __init__(self, session_id: str, config: SessionConfig, provider: Any,
                 metrics: SessionMetrics):
        self.session_id = session_id
        self.config = config
        self.provider = provider
        self.metrics = metrics
        self.source_language = config.source_language
        self.target_language = config.target_language
        self.events: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self._audio: asyncio.Queue[AudioChunk | None] = asyncio.Queue(maxsize=256)
        self._task: asyncio.Task | None = None
        self._sequence = 0
        self._translation_sequence = 0
        self._audio_seconds = 0.0
        self._running = False
        self._paused = False
        self._started_wall = 0.0
        self._failed = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._started_wall = now()
        self._task = asyncio.create_task(self._run(), name="vb-s2st")
        await self._emit(EventType.SESSION_STARTED, {
            "config": self.config.to_dict(),
            "providers": {"s2st": {"name": self.provider.name,
                                   "license": self.provider.capabilities.model_license}}})

    async def push_audio(self, chunk: AudioChunk) -> None:
        if not self._running or self._paused or self._failed:
            return
        chunk.start = self._audio_seconds
        self._audio_seconds += chunk.duration
        chunk.session_id = self.session_id
        try:
            self._audio.put_nowait(chunk)
        except asyncio.QueueFull:
            self.metrics.increment("dropped_audio_chunks")

    async def _chunks(self) -> AsyncIterator[AudioChunk]:
        while True:
            chunk = await self._audio.get()
            if chunk is None:
                return
            yield chunk

    async def _run(self) -> None:
        try:
            async for part in self.provider.translate_stream(
                    self._chunks(), self.source_language, self.target_language):
                for seg in part.segments:
                    self._translation_sequence += 1
                    latency = self._audio_seconds - seg.start_time
                    self.metrics.observe("end_to_end_latency", max(0.0, latency))
                    await self._emit(EventType.TRANSLATION_FINAL, {
                        "source_text": "", "translated_text": seg.translated_text,
                        "source_language": self.source_language,
                        "target_language": self.target_language,
                        "start": seg.start_time, "end": seg.end_time,
                        "translation_sequence": self._translation_sequence,
                        "committed": True, "timing": "emission_offset",
                        "engine": self.provider.name})
                pcm = part.metadata.get("pcm16") or b""
                if pcm:
                    rate = int(part.metadata.get("sample_rate", 16000))
                    await self._emit(EventType.TTS_AUDIO, {
                        "audio": base64.b64encode(pcm).decode("ascii"), "encoding": "pcm_s16le",
                        "sample_rate": rate, "channels": 1, "duration": len(pcm) / 2 / rate,
                        "translation_sequence": self._translation_sequence,
                        "source_start": part.duration, "source_end": part.duration, "text": ""})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("S2ST stream failed")
            self._failed = True
            reason = str(exc).splitlines()[0][:200] if isinstance(exc, ProviderError) else (
                "the model worker failed; see server logs")
            await self._emit(EventType.ERROR, {
                "code": "SEAMLESS_MODEL_ERROR", "recoverable": False,
                "message": f"{self.provider.name} failed: {reason}. The session was not "
                           "switched to another engine."})

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        # Never block here: if the engine task died, nothing drains the queue.
        while True:
            try:
                self._audio.put_nowait(None)
                break
            except asyncio.QueueFull:
                self._audio.get_nowait()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=120)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        await self._emit(EventType.SESSION_ENDED, {"metrics": self.metrics.summary()})
        self._put_nowait(None)

    async def pause(self) -> None:
        self._paused = True

    async def resume(self) -> None:
        self._paused = False

    async def _emit(self, event_type: EventType, payload: dict[str, Any]) -> None:
        self._sequence += 1
        event = Event(event_type=event_type, session_id=self.session_id,
                      sequence_id=self._sequence, payload=payload)
        self._put_nowait(event)

    def _put_nowait(self, event: Event | None) -> None:
        """Enqueue without blocking; if no client is reading, drop the oldest."""
        while True:
            try:
                self.events.put_nowait(event)
                return
            except asyncio.QueueFull:
                self.events.get_nowait()

    async def stream_events(self) -> AsyncIterator[Event]:
        while True:
            event = await self.events.get()
            if event is None:
                return
            yield event

    def queue_stats(self) -> dict[str, Any]:
        return {"audio": {"depth": self._audio.qsize()}}
