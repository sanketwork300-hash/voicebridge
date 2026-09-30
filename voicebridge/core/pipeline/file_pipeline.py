"""File-mode cascade pipeline: normalised audio -> timed translated segments + dub.

Same business logic as the realtime pipeline (speech gate, ASR validation,
language-aware segmentation, context-aware translation, translation
validation, TTS with timing fit), different orchestration. The providers do not
know which mode they are in.

Stages and why they are shaped this way:

1. **Speech detection** -- Silero regions + AudioSet classification per window
   over the whole file (in 10-minute blocks, so memory stays flat). Only
   dialogue (and conservatively, uncertain audio) goes further.
2. **Dialogue-aware chunking** -- ASR chunks are built from speech regions,
   never from a fixed N-second grid: adjacent regions merge while the gap is
   short and the chunk stays under Whisper's 30 s window, so chunk edges fall
   in pauses.
3. **ASR** -- offline decoding per chunk with the previous chunk's text as the
   prompt; every Whisper segment is validated (hallucination filter) against
   decoder signals *and* the gate's verdict for that time span.
4. **Segmentation** -- the existing language-aware segmenter; for Japanese and
   Korean it holds clauses until a sentence-final form, across chunk edges.
5. **Translation** -- one :class:`GlobalTranslationContext` for the whole file,
   so chunk N sees chunks 1..N-1.
6. **TTS** -- per segment, fitted to the source duration, then placed at its
   source timestamp by :class:`~voicebridge.media.renderer.TimelineRenderer`.

Every stage checks the cancel token between units of work.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from voicebridge.core.asr_validation import ASRValidationEngine
from voicebridge.core.audio.dsp import pcm16_to_float, write_wav
from voicebridge.core.context.glossary import SessionGlossary
from voicebridge.core.context.translation_context import GlobalTranslationContext
from voicebridge.core.context.window import ContextEntry
from voicebridge.core.jobs.errors import CancelToken, PipelineError
from voicebridge.core.pipeline.speech_gate import SpeechGate, SpeechGateConfig, decision_for_span
from voicebridge.core.pipeline.translate_step import translate_segment
from voicebridge.core.pipeline.tts_step import TimingConfig, synthesize_fitted
from voicebridge.core.segmentation.segmenter import TranslationSegmenter
from voicebridge.core.session.language import LanguageDetectionStabilizer
from voicebridge.core.types import (
    EventType,
    HonorificPolicy,
    LatencyProfile,
    NameRendering,
    SpeechDecision,
    SpeechTranslationResult,
    TranscriptSegment,
    TranslationSegment,
    TTSRequest,
    new_id,
)
from voicebridge.core.validation import TranslationValidationEngine
from voicebridge.media.normalizer import SAMPLE_RATE, NormalizedAudio
from voicebridge.media.renderer import TimelineRenderer
from voicebridge.providers.base import ProviderError
from voicebridge.providers.registry import ProviderSet

logger = logging.getLogger(__name__)

EventSink = Callable[[EventType, dict[str, Any]], Awaitable[None]]


@dataclass
class FilePipelineConfig:
    source_language: str = "auto"
    target_language: str = "en"
    synthesize: bool = True
    voice: str | None = "source"
    voice_map: dict[str, str] = field(default_factory=dict)
    honorifics: HonorificPolicy = HonorificPolicy.PRESERVE_HONORIFICS
    name_rendering: NameRendering = NameRendering.TRANSLATE
    glossary: SessionGlossary = field(default_factory=SessionGlossary)
    style: str | None = None
    context_segments: int = 8
    context_tokens: int = 2048
    translation_validation: str = "balanced"
    speech_gate: SpeechGateConfig = field(default_factory=SpeechGateConfig)
    asr_validation: dict[str, Any] = field(default_factory=dict)
    timing: TimingConfig = field(default_factory=lambda: TimingConfig(regenerate=True))
    max_chunk_seconds: float = 28.0
    merge_gap_seconds: float = 0.8
    render_sample_rate: int = 24000
    reference_min_seconds: float = 3.0
    reference_max_seconds: float = 12.0
    #: Pass the previous chunk's transcript as the ASR prompt. Off by default:
    #: on the evaluation media it induced a Whisper decoding loop ("the UN is
    #: a production of the UN, ...") on an unrelated following utterance.
    asr_prompt_previous_text: bool = False
    #: Release the in-process translation model before TTS (low-memory hosts).
    release_between_stages: bool = False


@dataclass
class StageTimer:
    timings: dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, seconds: float) -> None:
        self.timings[stage] = round(self.timings.get(stage, 0.0) + seconds, 3)


class FilePipeline:
    def __init__(
        self,
        providers: ProviderSet,
        config: FilePipelineConfig,
        emit: EventSink,
        progress: Callable[[str, float], Awaitable[None]],
        cancel: CancelToken | None = None,
    ):
        self.providers = providers
        self.config = config
        self.emit = emit
        self.progress = progress
        self.cancel = cancel or CancelToken()
        self.gate = SpeechGate(providers.vad, providers.audio_event, config.speech_gate)
        self.asr_validator = ASRValidationEngine.from_dict(config.asr_validation)
        self.translation_validator = TranslationValidationEngine(config.translation_validation)
        self.context = GlobalTranslationContext(config.context_segments, config.context_tokens)
        self.timer = StageTimer()
        self.stats: dict[str, Any] = {
            "speech_regions": 0, "rejected_regions": 0, "rejected_by_event": {},
            "asr_segments": 0, "asr_rejected": 0, "asr_rejected_reasons": {},
            "translation_rejected": 0, "tts_failures": 0,
        }

    # -- stage 1 ---------------------------------------------------------------

    async def detect_speech(self, audio: NormalizedAudio) -> list[SpeechDecision]:
        t0 = time.perf_counter()
        decisions: list[SpeechDecision] = []
        for offset, block in audio.blocks(600.0):
            self.cancel.check()
            decisions.extend(await self.gate.detect_regions(block, offset))
            await self.progress("speech_detection", min(1.0, (offset + 600) / audio.duration))
        for d in decisions:
            await self.emit(EventType.SPEECH_EVENT, {
                "event_type": d.event_type, "confidence": round(d.confidence, 3),
                "should_transcribe": d.should_transcribe, "start": round(d.start_time, 3),
                "end": round(d.end_time, 3), "reason": d.reason})
            if d.should_transcribe:
                self.stats["speech_regions"] += 1
            else:
                self.stats["rejected_regions"] += 1
                by = self.stats["rejected_by_event"]
                by[d.event_type] = by.get(d.event_type, 0) + 1
        kept = sum(d.end_time - d.start_time for d in decisions if d.should_transcribe)
        self.stats["speech_seconds"] = round(kept, 2)
        self.stats["non_speech_seconds"] = round(max(0.0, audio.duration - kept), 2)
        self.timer.record("speech_detection", time.perf_counter() - t0)
        return decisions

    def chunk(self, decisions: list[SpeechDecision],
              audio: NormalizedAudio | None = None) -> list[tuple[float, float]]:
        chunks: list[list[float]] = []
        for d in decisions:
            if not d.should_transcribe:
                continue
            a, b = d.start_time, d.end_time
            while b - a > self.config.max_chunk_seconds:
                # Split an over-long region at its quietest point (a breath or
                # pause) rather than at a fixed offset that may cut a word.
                limit = self.config.max_chunk_seconds
                cut = _quietest_point(audio, a + limit / 2, a + limit) \
                    if audio is not None else a + limit
                chunks.append([a, cut])
                a = cut
            if (chunks and a - chunks[-1][1] <= self.config.merge_gap_seconds
                    and b - chunks[-1][0] <= self.config.max_chunk_seconds):
                chunks[-1][1] = b
            else:
                chunks.append([a, b])
        return [(a, b) for a, b in chunks]

    # -- stage 2/3 -------------------------------------------------------------

    async def transcribe(self, audio: NormalizedAudio, decisions: list[SpeechDecision]
                         ) -> list[dict[str, Any]]:
        """ASR + validation + segmentation. Returns translation units in order."""
        t0 = time.perf_counter()
        await self.progress("transcribing", 0.0)
        chunks = self.chunk(decisions, audio)
        src = self.config.source_language
        detector = LanguageDetectionStabilizer()
        language = None if src == "auto" else src
        segmenter = TranslationSegmenter(language=language, latency=LatencyProfile.ACCURATE)
        units: list[dict[str, Any]] = []
        prompt = ""
        last_end: float | None = None

        def take(pending_list):
            for p in pending_list:
                if p.text.strip():
                    units.append({"text": p.text.strip(), "start": p.start, "end": p.end,
                                  "speaker": p.speaker, "language": p.language or language})

        for i, (a, b) in enumerate(chunks):
            self.cancel.check()
            if last_end is not None:
                take(segmenter.on_silence(a - last_end))
            samples = audio.read(a, b)
            try:
                raw_segments = await self.providers.asr.transcribe_array(
                    samples, language=language or None,
                    initial_prompt=(prompt[-200:] or None)
                    if self.config.asr_prompt_previous_text else None)
            except ProviderError as exc:
                raise PipelineError("transcribing", "ASR_FAILED", str(exc)) from exc
            for raw in raw_segments:
                seg = TranscriptSegment(
                    text=str(raw.get("text", "")).strip(), start=a + float(raw["start"]),
                    end=a + float(raw["end"]), language=raw.get("language") or language,
                    speaker=raw.get("speaker"), confidence=raw.get("confidence"),
                )
                self.stats["asr_segments"] += 1
                decision = decision_for_span(decisions, seg.start, seg.end)
                result = self.asr_validator.validate(seg, raw, decision, expected_language=language)
                if not result.valid:
                    self.stats["asr_rejected"] += 1
                    reasons = self.stats["asr_rejected_reasons"]
                    reasons[result.reason] = reasons.get(result.reason, 0) + 1
                    await self.emit(EventType.WARNING, {
                        "code": "asr_hallucination_rejected", "text": seg.text,
                        "reason": result.reason, "start": seg.start, "end": seg.end,
                        "signals": result.signals})
                    continue
                if src == "auto" and raw.get("language"):
                    detector.note_audio(seg.duration)
                    stable = detector.observe(raw["language"], raw.get("language_probability"))
                    if stable and stable != language:
                        language = stable
                        segmenter.set_language(stable, LatencyProfile.ACCURATE)
                        await self.emit(EventType.LANGUAGE_DETECTED,
                                        {"language": stable, "stable": True})
                    elif language is None:
                        language = raw["language"]
                        segmenter.set_language(language, LatencyProfile.ACCURATE)
                seg.language = seg.language or language
                await self.emit(EventType.ASR_STABLE, {
                    "text": seg.text, "start": round(seg.start, 3), "end": round(seg.end, 3),
                    "language": seg.language, "confidence": round(result.confidence, 3)})
                prompt += seg.text
                take(segmenter.add(seg))
            last_end = b
            await self.progress("transcribing", (i + 1) / max(1, len(chunks)))
        take(segmenter.flush("end_of_file"))
        self.detected_language = language
        for unit in units:
            unit["language"] = unit["language"] or language
            await self.emit(EventType.SEGMENT_CREATED, dict(unit))
        self.timer.record("transcribing", time.perf_counter() - t0)
        return units

    # -- stage 4 ---------------------------------------------------------------

    async def translate(self, units: list[dict[str, Any]]) -> list[TranslationSegment]:
        t0 = time.perf_counter()
        await self.progress("translating", 0.0)
        engine = self.providers.translation
        out: list[TranslationSegment] = []
        latencies = []
        for i, unit in enumerate(units, start=1):
            self.cancel.check()
            src_lang = unit["language"] or self.config.source_language
            if not src_lang or src_lang == "auto":
                continue
            await self.emit(EventType.TRANSLATION_STARTED, {
                "source_text": unit["text"], "start": unit["start"], "end": unit["end"],
                "translation_sequence": i})
            started = time.perf_counter()
            try:
                result = await translate_segment(
                    engine, unit["text"], src_lang, self.config.target_language,
                    glossary=self.config.glossary, context=self.context,
                    honorifics=self.config.honorifics, name_rendering=self.config.name_rendering,
                    speaker=unit.get("speaker"), style=self.config.style, sequence_id=i)
            except ProviderError as exc:
                raise PipelineError("translating", "TRANSLATION_FAILED", str(exc),
                                    retryable=True) from exc
            latencies.append(time.perf_counter() - started)
            validation = self.translation_validator.validate(result)
            if not validation.valid:
                self.stats["translation_rejected"] += 1
                await self.emit(EventType.WARNING, {
                    "code": "translation_validation_failed", "issues": validation.issues,
                    "source_text": unit["text"], "translated_text": result.translated_text})
                continue
            self.context.add(ContextEntry(unit["text"], result.translated_text, unit["start"],
                                          unit["end"], unit.get("speaker")))
            seg = TranslationSegment(
                id=f"seg-{i:05d}", start_time=round(unit["start"], 3),
                end_time=round(unit["end"], 3), source_text=unit["text"],
                translated_text=result.translated_text, source_language=src_lang,
                target_language=self.config.target_language, speaker_id=unit.get("speaker"),
                confidence=validation.confidence, event_type="DIALOGUE",
                metadata={"provider": result.provider, "validation_issues": validation.issues,
                          "timestamps_available": True},
            )
            out.append(seg)
            await self.emit(EventType.TRANSLATION_COMPLETED, {
                "segment_id": seg.id, "source_text": seg.source_text,
                "translated_text": seg.translated_text, "start": seg.start_time,
                "end": seg.end_time, "translation_sequence": i})
            await self.emit(EventType.SUBTITLE_CREATED, {
                "segment_id": seg.id, "start": seg.start_time, "end": seg.end_time,
                "text": seg.translated_text})
            await self.progress("translating", i / max(1, len(units)))
        self.stats["translation_latency_mean"] = (
            round(float(np.mean(latencies)), 3) if latencies else None)
        self.timer.record("translating", time.perf_counter() - t0)
        return out

    # -- stage 5 ---------------------------------------------------------------

    def _reference_clip(self, audio: NormalizedAudio, seg: TranslationSegment,
                        segments: list[TranslationSegment], workdir: Path) -> str:
        """Source audio of this speaker for voice cloning (voice mode ``source``).

        The segment's own audio is preferred; very short lines borrow the
        neighbouring same-speaker lines so the speaker embedding is stable.
        """
        a, b = seg.start_time, seg.end_time
        if b - a < self.config.reference_min_seconds:
            for other in segments:
                if other.speaker_id != seg.speaker_id:
                    continue
                if abs(other.start_time - a) < 20.0:
                    a, b = min(a, other.start_time), max(b, other.end_time)
                if b - a >= self.config.reference_min_seconds:
                    break
        b = min(b, a + self.config.reference_max_seconds)
        path = workdir / f"ref-{seg.id}.wav"
        write_wav(str(path), audio.read(a, b), SAMPLE_RATE)  # <= 12 s: cheap
        return str(path)

    async def synthesize(self, audio: NormalizedAudio, segments: list[TranslationSegment],
                         workdir: Path) -> tuple[Path | None, dict[str, Any]]:
        t0 = time.perf_counter()
        engine = self.providers.tts
        if engine is None or not self.config.synthesize or not segments:
            return None, {}
        await self.progress("synthesizing", 0.0)
        renderer = TimelineRenderer(audio.duration, self.config.render_sample_rate,
                                    workdir / "render")
        mismatches, fits = [], {}
        tts_time = 0.0
        tts_audio = 0.0
        try:
            for i, seg in enumerate(segments, start=1):
                self.cancel.check()
                voice = self.config.voice_map.get(str(seg.speaker_id), self.config.voice)
                request = TTSRequest(
                    text=seg.translated_text or "", language=self.config.target_language,
                    speaker=voice, style=self.config.style, sequence_id=i,
                    source_start=seg.start_time, source_end=seg.end_time)
                caps = getattr(engine, "capabilities_for", lambda _l: engine.capabilities)(
                    request.language)
                if voice in (None, "source") and caps.voice_cloning:
                    request.reference_audio = self._reference_clip(audio, seg, segments, workdir)
                await self.emit(EventType.TTS_STARTED, {"segment_id": seg.id})
                started = time.perf_counter()
                try:
                    synth, fit = await synthesize_fitted(engine, request, self.config.timing)
                except ProviderError as exc:
                    self.stats["tts_failures"] += 1
                    await self.emit(EventType.WARNING, {"code": "tts_error", "segment_id": seg.id,
                                                        "message": str(exc)})
                    continue
                tts_time += time.perf_counter() - started
                tts_audio += synth.duration
                placement = await asyncio.to_thread(
                    renderer.place, seg.id, pcm16_to_float(synth.audio), synth.sample_rate,
                    seg.start_time)
                seg.metadata.update({
                    "tts_duration": round(synth.duration, 3), "tts_fit": fit.method,
                    "tts_rate": round(fit.rate, 3), "placed_at": round(placement.actual_start, 3),
                    "drift": round(placement.drift, 3)})
                fits[fit.method] = fits.get(fit.method, 0) + 1
                if fit.source_duration > 0:
                    mismatches.append(fit.mismatch)
                await self.emit(EventType.TTS_COMPLETED, {
                    "segment_id": seg.id, "duration": round(synth.duration, 3),
                    "source_duration": round(fit.source_duration, 3), "fit": fit.method,
                    "placed_at": round(placement.actual_start, 3)})
                await self.progress("synthesizing", i / len(segments))
            out = await asyncio.to_thread(renderer.write_wav, workdir / "dub.wav")
        finally:
            renderer.close()
        info = {
            "tts_rtf": round(tts_time / tts_audio, 3) if tts_audio else None,
            "duration_mismatch_mean": round(float(np.mean(mismatches)), 3) if mismatches else None,
            "duration_mismatch_max": round(float(np.max(mismatches)), 3) if mismatches else None,
            "fit_methods": fits,
            "max_drift_seconds": round(renderer.stats.max_drift, 3),
            "mean_drift_seconds": round(renderer.stats.mean_drift, 3),
        }
        self.timer.record("synthesizing", time.perf_counter() - t0)
        return out, info

    # -- whole run ---------------------------------------------------------------

    async def run(self, audio: NormalizedAudio, workdir: Path) -> SpeechTranslationResult:
        workdir.mkdir(parents=True, exist_ok=True)
        decisions = await self.detect_speech(audio)
        units = await self.transcribe(audio, decisions)
        segments = await self.translate(units)
        if self.config.release_between_stages:
            backend = getattr(self.providers.translation, "backend", None)
            if backend is not None:
                backend.unload()
            import gc

            gc.collect()
        dub, tts_info = await self.synthesize(audio, segments, workdir)
        return SpeechTranslationResult(
            audio_path=str(dub) if dub else None,
            source_language=getattr(self, "detected_language", None)
            or self.config.source_language,
            target_language=self.config.target_language,
            segments=segments,
            duration=audio.duration,
            metadata={"engine": "cascade", "stats": self.stats, "tts": tts_info,
                      "stage_seconds": self.timer.timings, "run_id": new_id(),
                      "timestamps_available": True},
        )


def _quietest_point(audio: NormalizedAudio, a: float, b: float, frame: float = 0.1) -> float:
    samples = audio.read(a, b)
    n = int(frame * SAMPLE_RATE)
    if samples.size < 2 * n:
        return (a + b) / 2
    frames = samples[: samples.size // n * n].reshape(-1, n)
    quietest = int(np.argmin(np.sqrt(np.mean(frames**2, axis=1))))
    return a + (quietest + 0.5) * frame
