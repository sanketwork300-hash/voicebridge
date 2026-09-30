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
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from voicebridge.core.asr_validation import ASRValidationEngine
from voicebridge.core.context.glossary import SessionGlossary
from voicebridge.core.context.translation_context import GlobalTranslationContext
from voicebridge.core.context.window import ContextEntry
from voicebridge.core.metrics.metrics import SessionMetrics
from voicebridge.core.pipeline.speech_gate import (
    SpeechGate,
    SpeechGateConfig,
    decision_for_span,
)
from voicebridge.core.pipeline.translate_step import translate_segment
from voicebridge.core.pipeline.tts_step import TimingConfig, synthesize_fitted
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
    SpeechDecision,
    SynthesisedAudio,
    TranscriptSegment,
    TranslationResult,
    TTSRequest,
    now,
)
from voicebridge.core.validation import TranslationValidationEngine
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
    event_type: str | None = None
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
    speaker: int | None = None
    emotion: str | None = None
    style: str | None = None
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
        self.context = GlobalTranslationContext(
            max_segments=config.context_segments,
            max_tokens=config.context_tokens,
        )
        self.language_detector = LanguageDetectionStabilizer()
        self.scheduler = TTSScheduler()
        self.glossary: SessionGlossary = config.glossary
        self.speech_gate = SpeechGate(
            providers.vad,
            providers.audio_event,
            SpeechGateConfig(
                enabled=config.speech_gate_enabled,
                dialogue_threshold=config.speech_gate_dialogue_threshold,
                unknown_threshold=config.speech_gate_unknown_threshold,
                non_speech_threshold=config.speech_gate_non_speech_threshold,
            ),
        )
        self.asr_validator = ASRValidationEngine()
        # Realtime duration fitting uses time-stretch only: regenerating would
        # double TTS latency on the live path.
        self._timing = TimingConfig(
            enabled=config.tts_timing_enabled, max_rate=config.tts_max_rate,
            min_rate=config.tts_min_rate, duration_tolerance=config.tts_duration_tolerance,
            regenerate=False)
        self._recent_audio: deque[tuple[float, bytes]] = deque()
        self.translation_validator = TranslationValidationEngine(config.translation_validation)
        self._speech_decisions: list[SpeechDecision] = []

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

    async def stop(self, drain_timeout: float = 60.0) -> None:
        """Stop the session, draining committed work first.

        Order matters: audio pump -> ASR (its final text arrives during
        ``stop_session``) -> segmenter flush -> translation -> TTS. Each stage
        gets up to ``drain_timeout`` to finish what is already committed, so
        the last sentence is translated and spoken instead of being cancelled
        mid-flight; only then are the remaining tasks cancelled.
        """
        if self._stopping:
            return
        self._stopping = True

        async def drain(name: str, timeout: float) -> None:
            task = next((t for t in self._tasks if t.get_name() == name), None)
            if task is None or task.done():
                return
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            except (TimeoutError, asyncio.CancelledError):
                logger.warning("%s did not drain within %.0fs", name, timeout)
            except Exception:  # the stage already reported its own failure
                pass

        await self.audio_queue.put(None)
        await drain("vb-audio", 30)
        try:
            await self.providers.asr.stop_session(self.session_id)
        except Exception as exc:
            logger.warning("error stopping ASR session: %s", exc)
        await drain("vb-asr", 15)

        # Flush whatever the segmenter is still holding so the final sentence
        # is not lost on stop.
        for pending in self.segmenter.flush("session_stop"):
            await self._enqueue_translation(pending)
        await self.translation_queue.put(None)
        await drain("vb-translate", drain_timeout)
        if self.tts_available:
            await self.tts_queue.put(None)
            await drain("vb-tts", drain_timeout)
            for ready in self.scheduler.tick():
                await self._emit_audio(ready)

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
            try:
                if chunk is None:
                    # End of input: decide whatever the gate is still buffering.
                    for forwarded, decision in await self.speech_gate.flush():
                        await self._forward(forwarded, decision)
                    return
                self.metrics.observe("capture_latency", max(0.0, now() - chunk.created_at))
                with self.metrics.timer("speech_gate_latency"):
                    gated = await self.speech_gate.gate_chunk(chunk)
                self._remember_audio(chunk)
                for forwarded, decision in gated:
                    await self._forward(forwarded, decision)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("ASR push failed")
                await self._error(f"Speech recognition failed: {exc}", recoverable=False)
                return

    async def _forward(self, chunk: AudioChunk, decision: SpeechDecision) -> None:
        """Send one gated chunk to ASR, announcing each new gate decision once."""
        if self._speech_decisions and self._speech_decisions[-1] is decision:
            pass
        else:
            self._remember_speech_decision(decision)
            await self._emit(
                EventType.SPEECH_EVENT,
                {
                    "event_type": decision.event_type,
                    "confidence": round(decision.confidence, 3),
                    "should_transcribe": decision.should_transcribe,
                    "start": decision.start_time,
                    "end": decision.end_time,
                    "reason": decision.reason,
                },
            )
            if not decision.should_transcribe:
                self.metrics.increment("speech_gate_rejected_windows")
        # Rejected audio arrives here already replaced by silence of equal
        # length, so the ASR clock never drifts from the source clock.
        await self.providers.asr.push_audio(self.session_id, chunk)

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
                confidence=payload.get("confidence"),
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
        # ASR validation runs here, on the complete unit. Streaming ASR commits
        # text in token-sized deltas whose individual durations are meaningless
        # (validating those rejected real speech as "too dense"/"too short").
        validation = self.asr_validator.validate(
            TranscriptSegment(text=text, start=pending.start, end=pending.end,
                              language=pending.language or self.source_language),
            speech_decision=self._speech_decision_for(pending.start, pending.end),
        )
        self.metrics.observe("asr_validation_confidence", validation.confidence)
        if not validation.valid:
            self.metrics.increment("asr_validation_rejected_segments")
            await self._emit(EventType.WARNING, {
                "code": "asr_hallucination_rejected", "text": text, "reason": validation.reason,
                "start": pending.start, "end": pending.end,
                "message": f"Rejected likely ASR hallucination ({validation.reason})."})
            return
        self._translation_sequence += 1
        job = TranslationJob(
            text=text,
            start=pending.start,
            end=pending.end,
            sequence_id=self._translation_sequence,
            speaker=pending.speaker,
            language=pending.language or self.source_language,
            event_type=getattr(pending, "event_type", None),
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
                await self._emit(
                    EventType.TRANSLATION_STARTED,
                    {
                        "source_text": job.text,
                        "source_language": job.language or self.source_language,
                        "target_language": self.target_language,
                        "start": job.start,
                        "end": job.end,
                        "translation_sequence": job.sequence_id,
                    },
                )
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
            validation = self.translation_validator.validate(result)
            self.metrics.observe("translation_validation_confidence", validation.confidence)
            if not validation.valid:
                self.metrics.increment("translation_validation_failures")
                await self._warn(
                    "translation_validation_failed",
                    f"Translation validation failed: {', '.join(validation.issues)}",
                )
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
                    "validation": {
                        "valid": validation.valid,
                        "confidence": validation.confidence,
                        "issues": validation.issues,
                    },
                },
            )
            await self._emit(
                EventType.TRANSLATION_COMPLETED,
                {
                    "translation_sequence": job.sequence_id,
                    "provider": result.provider,
                    "confidence": result.confidence,
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
                        speaker=job.speaker,
                        style=self.config.translation_style,
                    )
                )

    async def _translate(self, job: TranslationJob) -> TranslationResult | None:
        engine = self.providers.translation
        source_language = job.language or self.source_language
        if source_language == "auto":
            # Nothing detected yet; translating from an unknown language would
            # be a guess. Skip rather than mistranslate.
            return None
        with self.metrics.timer("translation_latency"):
            result = await translate_segment(
                engine,
                job.text,
                source_language,
                self.target_language,
                glossary=self.glossary,
                context=self.context,
                honorifics=self.config.honorifics,
                name_rendering=self.config.name_rendering,
                speaker=job.speaker,
                style=self.config.translation_style,
                sequence_id=job.sequence_id,
            )
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
                await self._emit(
                    EventType.TTS_STARTED,
                    {"translation_sequence": job.sequence_id, "source_start": job.source_start},
                )
                request = TTSRequest(
                    text=job.text,
                    language=job.language,
                    speaker=self.config.voice,
                    emotion=job.emotion,
                    speaking_rate=self.config.speed,
                    style=job.style,
                    sequence_id=job.sequence_id,
                    source_start=job.source_start,
                    source_end=job.source_end,
                )
                caps_for = getattr(engine, "capabilities_for", None)
                caps = caps_for(job.language) if caps_for else engine.capabilities
                if caps.voice_cloning and self.config.voice in (None, "source"):
                    request.reference_audio = self._reference_clip(job)
                with self.metrics.timer("tts_generation_latency"):
                    audio, fit = await synthesize_fitted(engine, request, self._timing)
                self.metrics.observe("tts_duration_mismatch", fit.mismatch)
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
            await self._emit(
                EventType.TTS_COMPLETED,
                {
                    "translation_sequence": job.sequence_id,
                    "duration": audio.duration,
                    "source_duration": max(0.0, job.source_end - job.source_start),
                },
            )
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

    def _remember_speech_decision(self, decision: SpeechDecision) -> None:
        self._speech_decisions.append(decision)
        if len(self._speech_decisions) > 256:
            del self._speech_decisions[:128]

    def _speech_decision_for(self, start: float, end: float) -> SpeechDecision | None:
        return decision_for_span(self._speech_decisions, start, end)

    def _remember_audio(self, chunk: AudioChunk) -> None:
        """Keep the last ~60 s of source audio for voice-cloning references."""
        self._recent_audio.append((chunk.start, chunk.data))
        while self._recent_audio and chunk.end - self._recent_audio[0][0] > 60.0:
            self._recent_audio.popleft()

    def _reference_clip(self, job: TTSJob) -> str | None:
        """Write the source speech of this segment (>= 3 s) as a WAV for cloning."""
        if not self._recent_audio:
            return None
        start = job.source_start
        end = max(job.source_end, start + 3.0)
        parts = [data for t, data in self._recent_audio if t + len(data) / 32000 > start and t < end]
        if not parts:
            return None
        import tempfile

        from voicebridge.core.audio.dsp import pcm16_to_float, write_wav

        path = tempfile.NamedTemporaryFile(prefix="vb-ref-", suffix=".wav", delete=False).name
        write_wav(path, pcm16_to_float(b"".join(parts))[: int(12 * 16000)], 16000)
        return path

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
