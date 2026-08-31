"""The VoiceBridge translation pipeline.

Stage graph (each arrow is a bounded queue with a declared overflow policy):

    push_audio -> [audio_queue: DROP_OLDEST]
               -> ASR provider
               -> [asr events]
               -> TranscriptStabilizer -> TranslationSegmenter
               -> [translation_queue: DROP_NEWEST_DISPOSABLE]
               -> TranslationEngine (+ glossary, context, honorifics)
               -> [tts_queue: BLOCK]
               -> TTSEngine -> TTSScheduler (reorder) -> TTS_AUDIO events

Two invariants the rest of the system depends on:

**Partial results never reach TTS.** Partial ASR text is display-only. Speech
cannot be un-played, so a retracted partial that has already been spoken is an
unrecoverable error; a retracted subtitle is just a flicker. This is why the
dubbing path runs one commitment stage behind the subtitle path, and why
``subtitle_latency`` and ``speech_latency`` are reported separately.

**Every stage degrades rather than fails.** A dead TTS provider must still leave
working subtitles; a dead translation provider must still leave a working
source-language transcript. Failures raise a WARNING event and disable the
failing stage for the session instead of tearing the session down.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from voicebridge.core.context.glossary import SessionGlossary
from voicebridge.core.context.honorifics import apply_policy
from voicebridge.core.context.window import ContextEntry, ContextWindow
from voicebridge.core.metrics.metrics import SessionMetrics
from voicebridge.core.scheduling.tts_scheduler import TTSScheduler
from voicebridge.core.segmentation.segmenter import PendingSegment, TranslationSegmenter
from voicebridge.core.session.config import SessionConfig
from voicebridge.core.session.language import LanguageDetectionStabilizer
from voicebridge.core.streaming.queue import BoundedQueue, OverflowPolicy
from voicebridge.core.transcript.stabilizer import TranscriptStabilizer
from voicebridge.core.types import (
    AudioChunk,
    Event,
    EventType,
    SynthesisedAudio,
    TranscriptSegment,
    TranslationResult,
    now,
)
from voicebridge.providers.base import ProviderError
from voicebridge.providers.registry import ProviderSet

logger = logging.getLogger(__name__)

#: Queue capacities. Small on purpose: a deep queue converts a throughput
#: problem into an invisible latency problem.
AUDIO_QUEUE_SIZE = 64

#: Inputs that can be back-pressured instead of dropped (see ``live_input``).
OFFLINE_INPUT_TYPES = frozenset({"file", "batch"})
TRANSLATION_QUEUE_SIZE = 32
TTS_QUEUE_SIZE = 16

#: Minimum gap between two warnings carrying the same code.
WARNING_THROTTLE_SECONDS = 10.0


@dataclass
class TranslationJob:
    """A unit of source text awaiting translation."""

    text: str
    start: float
    end: float
    sequence_id: int
    speaker: int | None = None
    language: str | None = None
    is_partial: bool = False
    enqueued_at: float = field(default_factory=now)
    #: Only partial jobs may be dropped under backpressure.
    @property
    def disposable(self) -> bool:
        return self.is_partial


@dataclass
class TTSJob:
    text: str
    language: str
    sequence_id: int
    source_start: float
    source_end: float
    enqueued_at: float = field(default_factory=now)

    @property
    def disposable(self) -> bool:
        return False  # committed speech is never dropped


class TranslationPipeline:
    """Owns all per-session state and the stage tasks."""

    def __init__(
        self,
        session_id: str,
        config: SessionConfig,
        providers: ProviderSet,
        metrics: SessionMetrics,
    ):
        self.session_id = session_id
        self.config = config
        self.providers = providers
        self.metrics = metrics

        self.source_language = config.source_language
        self.target_language = config.target_language

        self.stabilizer = TranscriptStabilizer()
        self.segmenter = TranslationSegmenter(
            language=None if config.source_language == "auto" else config.source_language,
            latency=config.profile,
        )
        self.context = ContextWindow(
            max_segments=config.context_segments,
            max_characters=config.context_characters,
        )
        self.language_detector = LanguageDetectionStabilizer()
        self.scheduler = TTSScheduler()
        self.glossary: SessionGlossary = config.glossary

        # Overflow policy depends on whether the source is live.
        #
        # A live source (tab, microphone, network stream) cannot be slowed down:
        # if we cannot keep up, the only options are to drop audio or to fall
        # further behind forever, and for live translation the freshest audio is
        # the useful one -- so DROP_OLDEST.
        #
        # An offline source (a file) *can* be slowed down, and dropping from it
        # would silently produce an incomplete transcript of a file the user
        # expected to be translated in full. There BLOCK is correct: it simply
        # reads the file more slowly.
        self.live_input = config.input.type not in OFFLINE_INPUT_TYPES
        self.audio_queue: BoundedQueue[AudioChunk | None] = BoundedQueue(
            "audio",
            AUDIO_QUEUE_SIZE,
            OverflowPolicy.DROP_OLDEST if self.live_input else OverflowPolicy.BLOCK,
        )
        self.translation_queue: BoundedQueue[TranslationJob | None] = BoundedQueue(
            "translation", TRANSLATION_QUEUE_SIZE, OverflowPolicy.DROP_NEWEST_DISPOSABLE
        )
        self.tts_queue: BoundedQueue[TTSJob | None] = BoundedQueue(
            "tts", TTS_QUEUE_SIZE, OverflowPolicy.BLOCK
        )
        self.events: asyncio.Queue = asyncio.Queue(maxsize=512)

        self._tasks: list[asyncio.Task] = []
        self._sequence = 0
        self._translation_sequence = 0
        self._audio_sequence = 0
        self._audio_seconds = 0.0
        self._running = False
        self._paused = False
        self._stopping = False

        #: Degradation flags. Once set, the stage is skipped for this session.
        self.translation_available = True
        self.tts_available = bool(providers.tts) and config.mode.wants_tts
        self._last_partial_emit = 0.0
        self._warned_at: dict[str, float] = {}
        self._warn_counts: dict[str, int] = {}

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        lang = None if self.source_language == "auto" else self.source_language
        await self.providers.asr.start_session(self.session_id, language=lang)
        self._running = True
        self._tasks = [
            asyncio.create_task(self._audio_pump(), name="vb-audio"),
            asyncio.create_task(self._asr_consumer(), name="vb-asr"),
            asyncio.create_task(self._translation_worker(), name="vb-translate"),
            asyncio.create_task(self._scheduler_tick(), name="vb-sched"),
        ]
        if self.tts_available:
            self._tasks.append(asyncio.create_task(self._tts_worker(), name="vb-tts"))
        await self._emit(
            EventType.SESSION_STARTED,
            {
                "config": self.config.to_dict(),
                "providers": self.providers.describe(),
            },
        )

    async def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True

        # Flush whatever the segmenter is still holding so the final sentence
        # is not lost on stop.
        for pending in self.segmenter.flush("session_stop"):
            await self._enqueue_translation(pending)

        await self.audio_queue.put(None)
        try:
            await self.providers.asr.stop_session(self.session_id)
        except Exception as exc:
            logger.warning("error stopping ASR session: %s", exc)

        # Let in-flight work drain briefly before cancelling.
        await self.translation_queue.put(None)
        if self.tts_available:
            await self.tts_queue.put(None)
        await asyncio.sleep(0)

        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._running = False
        await self._emit(EventType.SESSION_ENDED, {"metrics": self.metrics.summary()})
        await self.events.put(None)

    async def pause(self) -> None:
        self._paused = True

    async def resume(self) -> None:
        self._paused = False

    # -- ingress -----------------------------------------------------------

    async def push_audio(self, chunk: AudioChunk) -> None:
        """Accept audio from an input adapter."""
        if not self._running or self._stopping:
            return
        if self._paused:
            return  # deliberately discarded: paused means "stop listening"
        self._audio_sequence += 1
        chunk.sequence_id = self._audio_sequence
        chunk.session_id = self.session_id
        chunk.start = self._audio_seconds
        self._audio_seconds += chunk.duration
        self.metrics.set_gauge("audio_seconds_processed", self._audio_seconds)
        self.language_detector.note_audio(chunk.duration)

        accepted = await self.audio_queue.put(chunk)
        if not accepted:
            self.metrics.increment("dropped_audio_chunks")
        if self.audio_queue.overloaded:
            await self._warn("audio_backlog", "Audio queue is backing up; ASR cannot keep up.")

    # -- stage: audio ------------------------------------------------------

    async def _audio_pump(self) -> None:
        while True:
            chunk = await self.audio_queue.get()
            if chunk is None:
                return
            try:
                with self.metrics.timer("asr_queue_latency"):
                    pass
                self.metrics.observe("capture_latency", max(0.0, now() - chunk.created_at))
                await self.providers.asr.push_audio(self.session_id, chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("ASR push failed")
                await self._error(f"Speech recognition failed: {exc}", recoverable=False)
                return

    # -- stage: ASR events -------------------------------------------------

    async def _asr_consumer(self) -> None:
        try:
            async for event in self.providers.asr.get_events(self.session_id):
                await self._handle_asr_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("ASR event stream failed")
            await self._error(f"Speech recognition unavailable: {exc}", recoverable=False)

    async def _handle_asr_event(self, event: Event) -> None:
        if event.event_type is EventType.LANGUAGE_DETECTED:
            language = event.payload.get("language")
            confidence = event.payload.get("confidence")
            changed = self.language_detector.observe(language, confidence)
            if changed:
                self.source_language = changed
                self.segmenter.set_language(changed, self.config.profile)
                await self._emit(
                    EventType.LANGUAGE_DETECTED,
                    {"language": changed, "confidence": confidence, "stable": True},
                )
            return

        if event.event_type is EventType.ASR_PARTIAL:
            text = (event.payload.get("text") or "").strip()
            if text:
                await self._emit(
                    EventType.ASR_PARTIAL,
                    {
                        "text": text,
                        "start": event.payload.get("start", 0.0),
                        "end": event.payload.get("end", 0.0),
                        "language": self.source_language,
                    },
                )
            return

        if event.event_type is EventType.ASR_STABLE:
            payload = event.payload.get("segment") or {}
            segment = TranscriptSegment(
                text=payload.get("text", ""),
                start=float(payload.get("start", 0.0) or 0.0),
                end=float(payload.get("end", 0.0) or 0.0),
                language=payload.get("language") or self.source_language,
                speaker=payload.get("speaker"),
            )
            update = self.stabilizer.update([segment], partial_text="")
            if update.rolled_back:
                await self._warn("asr_rollback", "Speech recogniser retracted committed text.")
            for stable in update.newly_stable:
                await self._emit(
                    EventType.ASR_STABLE,
                    {
                        "text": stable.text,
                        "start": stable.start,
                        "end": stable.end,
                        "speaker": stable.speaker,
                        "language": stable.language or self.source_language,
                    },
                )
                for ready in self.segmenter.add(stable):
                    await self._enqueue_translation(ready)
            return

        if event.event_type is EventType.SPEAKER_CHANGED:
            await self._emit(EventType.SPEAKER_CHANGED, dict(event.payload))
            return

        if event.event_type is EventType.ASR_FINAL:
            for pending in self.segmenter.flush("asr_final"):
                await self._enqueue_translation(pending)
            return

        if event.event_type is EventType.ERROR:
            await self._error(
                str(event.payload.get("message", "ASR error")),
                recoverable=bool(event.payload.get("recoverable", True)),
            )

    async def _enqueue_translation(self, pending: PendingSegment) -> None:
        text = (pending.text or "").strip()
        if not text:
            return
        self._translation_sequence += 1
        job = TranslationJob(
            text=text,
            start=pending.start,
            end=pending.end,
            sequence_id=self._translation_sequence,
            speaker=pending.speaker,
            language=pending.language or self.source_language,
        )
        if not self.translation_available:
            # Degraded: no translation, but the source transcript still flows,
            # so the user keeps original-language subtitles.
            return
        accepted = await self.translation_queue.put(job)
        if not accepted:
            self.metrics.increment("dropped_partial_results")
        if self.translation_queue.overloaded:
            await self._warn("translation_backlog", "Translation is falling behind.")

    # -- stage: translation ------------------------------------------------

    async def _translation_worker(self) -> None:
        while True:
            job = await self.translation_queue.get()
            if job is None:
                return
            self.metrics.observe("translation_queue_latency", max(0.0, now() - job.enqueued_at))
            try:
                result = await self._translate(job)
            except asyncio.CancelledError:
                raise
            except ProviderError as exc:
                logger.warning("translation provider failed: %s", exc)
                self.translation_available = False
                await self._warn(
                    "translation_unavailable",
                    f"Translation unavailable ({exc}). Continuing with source subtitles only.",
                )
                continue
            except Exception as exc:
                logger.exception("translation failed")
                await self._warn("translation_error", f"Translation failed: {exc}")
                continue

            if result is None:
                continue

            latency = now() - job.enqueued_at
            self.metrics.observe("subtitle_latency", latency)
            self.metrics.observe("end_to_end_latency", latency)

            await self._emit(
                EventType.TRANSLATION_FINAL,
                {
                    "source_text": result.source_text,
                    "translated_text": result.translated_text,
                    "source_language": result.source_language,
                    "target_language": result.target_language,
                    "start": job.start,
                    "end": job.end,
                    "speaker": job.speaker,
                    "translation_sequence": job.sequence_id,
                    "committed": True,
                },
            )

            self.context.add(
                ContextEntry(
                    source_text=result.source_text,
                    translated_text=result.translated_text,
                    start=job.start,
                    end=job.end,
                    speaker=job.speaker,
                )
            )

            if self.tts_available and result.translated_text.strip():
                await self.tts_queue.put(
                    TTSJob(
                        text=result.translated_text,
                        language=self.target_language,
                        sequence_id=job.sequence_id,
                        source_start=job.start,
                        source_end=job.end,
                    )
                )

    async def _translate(self, job: TranslationJob) -> TranslationResult | None:
        engine = self.providers.translation
        source_language = job.language or self.source_language
        if source_language == "auto":
            # Nothing detected yet; translating from an unknown language would
            # be a guess. Skip rather than mistranslate.
            return None
        if source_language == self.target_language:
            return TranslationResult(
                source_text=job.text,
                translated_text=job.text,
                source_language=source_language,
                target_language=self.target_language,
                sequence_id=job.sequence_id,
                provider="passthrough",
            )

        protected, mapping = self.glossary.protect(job.text, source_language)
        context = self.context.source_context() if engine.capabilities.supports_context else None

        with self.metrics.timer("translation_latency"):
            result = await engine.translate(
                protected,
                source_language,
                self.target_language,
                context=context,
                metadata={"speaker": job.speaker, "sequence_id": job.sequence_id},
            )

        text = self.glossary.restore(
            result.translated_text, mapping, self.config.name_rendering
        )
        text = apply_policy(job.text, text, self.config.honorifics, source_language)

        result.translated_text = text
        result.source_text = job.text  # report the original, not the protected form
        result.sequence_id = job.sequence_id
        result.source_start = job.start
        result.source_end = job.end
        return result

    # -- stage: TTS --------------------------------------------------------

    async def _tts_worker(self) -> None:
        while True:
            job = await self.tts_queue.get()
            if job is None:
                return
            self.metrics.observe("tts_queue_latency", max(0.0, now() - job.enqueued_at))
            engine = self.providers.tts
            if engine is None:
                continue
            try:
                with self.metrics.timer("tts_generation_latency"):
                    audio = await engine.synthesize(
                        job.text,
                        job.language,
                        speaker=self.config.voice,
                        speed=self.config.speed,
                        sequence_id=job.sequence_id,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("TTS failed for segment %s: %s", job.sequence_id, exc)
                await self._warn(
                    "tts_error",
                    f"Speech synthesis failed for one segment ({exc}); subtitles continue.",
                )
                # Close the ordering gap so later segments are not stuck.
                for ready in self.scheduler.mark_failed(job.sequence_id):
                    await self._emit_audio(ready)
                continue

            self.metrics.observe("speech_latency", max(0.0, now() - job.enqueued_at))
            audio.source_start = job.source_start
            audio.source_end = job.source_end
            for ready in self.scheduler.submit(audio):
                await self._emit_audio(ready)

            if self.scheduler.overloaded:
                await self._warn(
                    "tts_backlog",
                    f"Dubbed audio is {self.scheduler.backlog_seconds:.1f}s behind; "
                    "consider subtitle-only mode or a faster voice.",
                )

    async def _emit_audio(self, audio: SynthesisedAudio) -> None:
        """Publish one synthesised segment.

        ``source_start``/``source_end`` are positions on the source sample
        clock, which is what lets a client align the dub to the original media
        even when synthesis ran behind real time.
        """
        await self._emit(
            EventType.TTS_AUDIO,
            {
                "audio": base64.b64encode(audio.audio).decode("ascii"),
                "encoding": "pcm_s16le",
                "sample_rate": audio.sample_rate,
                "channels": audio.channels,
                "duration": audio.duration,
                "translation_sequence": audio.sequence_id,
                "source_start": audio.source_start,
                "source_end": audio.source_end,
                "text": audio.text,
            },
        )

    async def _scheduler_tick(self) -> None:
        """Periodically close TTS ordering gaps and publish queue health."""
        while True:
            await asyncio.sleep(0.5)
            if self.tts_available:
                for ready in self.scheduler.tick():
                    await self._emit_audio(ready)
            self.metrics.set_gauge("audio_queue_depth", self.audio_queue.qsize())
            self.metrics.set_gauge("translation_queue_depth", self.translation_queue.qsize())
            self.metrics.set_gauge("tts_backlog_seconds", self.scheduler.backlog_seconds)

    # -- events ------------------------------------------------------------

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    async def _emit(self, event_type: EventType, payload: dict[str, Any]) -> None:
        event = Event(
            event_type=event_type,
            session_id=self.session_id,
            sequence_id=self._next_sequence(),
            payload=payload,
        )
        try:
            self.events.put_nowait(event)
        except asyncio.QueueFull:
            # The client is not reading. Drop display-only events first.
            if event_type in (EventType.ASR_PARTIAL, EventType.METRIC):
                self.metrics.increment("dropped_partial_results")
                return
            await self.events.put(event)

    async def _warn(self, code: str, message: str) -> None:
        """Emit a warning, rate-limited per code.

        Backlog conditions persist for many chunks, so an un-throttled warning
        floods the client with hundreds of identical events and buries anything
        else. One per code per ``WARNING_THROTTLE_SECONDS`` is enough to convey
        a sustained condition; the count is reported so nothing is hidden.
        """
        last = self._warned_at.get(code, 0.0)
        current = now()
        self._warn_counts[code] = self._warn_counts.get(code, 0) + 1
        if current - last < WARNING_THROTTLE_SECONDS:
            return
        self._warned_at[code] = current
        repeats = self._warn_counts[code]
        payload = {"code": code, "message": message}
        if repeats > 1:
            payload["repeats"] = repeats
        await self._emit(EventType.WARNING, payload)

    async def _error(self, message: str, recoverable: bool = True) -> None:
        await self._emit(
            EventType.ERROR, {"message": message, "recoverable": recoverable}
        )

    async def stream_events(self) -> AsyncIterator[Event]:
        while True:
            event = await self.events.get()
            if event is None:
                return
            yield event

    def queue_stats(self) -> dict[str, Any]:
        return {
            "audio": self.audio_queue.stats(),
            "translation": self.translation_queue.stats(),
            "tts": self.tts_queue.stats(),
            "scheduler": self.scheduler.stats().__dict__,
        }
