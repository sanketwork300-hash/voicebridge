"""Benchmark runners.

Every runner takes the *same* dataset items and a list of candidate providers
or engines, measures, and returns rows; nothing is ranked or declared a winner.

``asr``          WER, CER, language-ID accuracy, latency, RTF, hallucination rate
                 (share of non-speech items that produce accepted text).
``translation``  BLEU, chrF (COMET when installed) against reference
                 translations, from *reference* source text so MT is isolated
                 from ASR errors; latency and characters/second. Contextual
                 providers are run in document order so context is used.
``tts``          RTF, duration, and intelligibility as round-trip WER/CER
                 (synthesised audio transcribed back by Whisper).
``engines``      whole systems on identical audio: VoiceBridge cascade vs
                 SeamlessStreaming vs SeamlessM4T v2 -> final translation
                 quality, output speech duration ratio, RTF, time to first
                 text/speech where the engine exposes it, peak memory.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from voicebridge.benchmark.datasets import DatasetItem
from voicebridge.benchmark.metrics import (
    ResourceSampler,
    audio_stats,
    cer,
    comet_scores,
    percentiles,
    translation_scores,
    wer,
)
from voicebridge.core.asr_validation import ASRValidationEngine
from voicebridge.core.types import TranscriptSegment


def _read(path: str) -> np.ndarray:
    import soundfile as sf

    from voicebridge.core.audio.dsp import resample

    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample(audio, rate, 16000)


def _is_spaced(lang: str) -> bool:
    return lang not in ("ja", "zh", "ko", "th")


async def _gated_regions(gate: Any, audio: np.ndarray) -> list[tuple[float, float]]:
    decisions = await gate.detect_regions(audio, 0.0)
    return [(d.start_time, d.end_time) for d in decisions if d.should_transcribe]


async def bench_asr(items: list[DatasetItem], providers: dict[str, Any],
                    force_language: bool = True, gate: Any = None,
                    validate: bool = True) -> list[dict[str, Any]]:
    """``gate``: when given, only speech-gate-approved regions are transcribed
    (the pipeline's behaviour); without it the whole clip goes to ASR, which is
    how the gate's effect on hallucination rate is measured."""
    validator = ASRValidationEngine()
    rows = []
    for name, engine in providers.items():
        loaded_at = time.perf_counter()
        await engine.warmup() if hasattr(engine, "warmup") else None
        load_seconds = time.perf_counter() - loaded_at
        errs_w, errs_c, lat, rtf, lang_ok, halluc, nonspeech = [], [], [], [], 0, 0, 0
        per_item = []
        with ResourceSampler() as res:
            for item in items:
                audio = _read(item.audio)
                started = time.perf_counter()
                lang = item.source_language if force_language else None
                if gate is not None:
                    segs = []
                    for a, b in await _gated_regions(gate, audio):
                        for s in await engine.transcribe_array(
                                audio[int(a * 16000):int(b * 16000)], language=lang):
                            segs.append({**s, "start": s["start"] + a, "end": s["end"] + a})
                else:
                    segs = await engine.transcribe_array(audio, language=lang)
                elapsed = time.perf_counter() - started
                duration = len(audio) / 16000
                accepted = []
                for raw in segs:
                    seg = TranscriptSegment(raw["text"], raw["start"], raw["end"],
                                            language=raw.get("language"))
                    if not validate or validator.validate(seg, raw, None, item.source_language
                                                          if force_language else None).valid:
                        accepted.append(raw["text"])
                hyp = " ".join(accepted) if _is_spaced(item.source_language) else "".join(accepted)
                lat.append(elapsed)
                rtf.append(elapsed / duration if duration else 0)
                detected = next((s.get("language") for s in segs if s.get("language")), None)
                lang_ok += int(detected == item.source_language)
                row = {"id": item.id, "hyp": hyp, "latency": round(elapsed, 3)}
                if item.reference_text:
                    row["cer"] = cer(item.reference_text, hyp)
                    errs_c.append(row["cer"])
                    if _is_spaced(item.source_language):  # WER is meaningless for ja/zh/th
                        row["wer"] = wer(item.reference_text, hyp)
                        errs_w.append(row["wer"])
                else:
                    nonspeech += 1
                    halluc += int(bool(hyp.strip()))
                per_item.append(row)
        speech_items = [i for i in items if i.reference_text]
        rows.append({
            "component": "asr", "model": name, "items": len(items),
            "gate": gate is not None, "validation": validate,
            "wer": round(float(np.mean(errs_w)), 4) if errs_w else None,
            "cer": round(float(np.mean(errs_c)), 4) if errs_c else None,
            "language_id_accuracy": (round(lang_ok / len(items), 3) if not force_language
                                     else "n/a (language forced)"),
            "hallucination_rate": round(halluc / nonspeech, 3) if nonspeech else None,
            "latency": percentiles(lat), "rtf": round(float(np.mean(rtf)), 3) if rtf else None,
            "load_seconds": round(load_seconds, 2), "speech_items": len(speech_items),
            "resources": res.summary(), "per_item": per_item,
        })
    return rows


async def bench_translation(items: list[DatasetItem], providers: dict[str, Any],
                            with_comet: bool = False, use_context: bool = True,
                            honorifics: str = "preserve") -> list[dict[str, Any]]:
    from voicebridge.core.context.glossary import SessionGlossary
    from voicebridge.core.context.translation_context import GlobalTranslationContext
    from voicebridge.core.context.window import ContextEntry
    from voicebridge.core.pipeline.translate_step import translate_segment
    from voicebridge.core.types import HonorificPolicy

    rows = []
    usable = [i for i in items if i.reference_text and i.reference_translation]
    for name, engine in providers.items():
        started = time.perf_counter()
        await engine.warmup()
        load_seconds = time.perf_counter() - started
        ctx = GlobalTranslationContext(8, 2048)
        hyps, lat, chars = [], [], 0
        with ResourceSampler() as res:
            for item in usable:
                t0 = time.perf_counter()
                result = await translate_segment(
                    engine, item.reference_text, item.source_language, item.target_language,
                    glossary=SessionGlossary(), context=ctx,
                    honorifics=HonorificPolicy(honorifics))
                lat.append(time.perf_counter() - t0)
                hyps.append(result.translated_text)
                chars += len(result.translated_text)
                if use_context:  # ablation: without it every line is translated alone
                    ctx.add(ContextEntry(item.reference_text, result.translated_text))
        refs = [i.reference_translation for i in usable]
        scores = translation_scores(hyps, refs)
        if with_comet:
            scores["comet"] = comet_scores([i.reference_text for i in usable], hyps, refs)
        rows.append({
            "component": "translation", "model": name + ("" if use_context else " (no context)"),
            "context": use_context, "items": len(usable), **scores,
            "latency": percentiles(lat),
            "chars_per_second": round(chars / sum(lat), 1) if lat else None,
            "load_seconds": round(load_seconds, 2), "resources": res.summary(),
            "per_item": [{"id": i.id, "src": i.reference_text, "hyp": h,
                          "ref": i.reference_translation} for i, h in zip(usable, hyps,
                                                                        strict=True)],
        })
    return rows


async def bench_tts(texts: list[tuple[str, str]], providers: dict[str, Any], asr: Any,
                    reference_audio: str | None = None) -> list[dict[str, Any]]:
    from voicebridge.core.audio.dsp import pcm16_to_float, resample
    from voicebridge.core.types import TTSRequest

    rows = []
    for name, engine in providers.items():
        started = time.perf_counter()
        await engine.warmup()
        load_seconds = time.perf_counter() - started
        rtf, lat, dur, w, c = [], [], [], [], []
        with ResourceSampler() as res:
            for lang, text in texts:
                req = TTSRequest(text=text, language=lang, reference_audio=reference_audio)
                t0 = time.perf_counter()
                out = await engine.synthesize_request(req)
                elapsed = time.perf_counter() - t0
                lat.append(elapsed)
                dur.append(out.duration)
                rtf.append(elapsed / out.duration if out.duration else 0)
                audio = resample(pcm16_to_float(out.audio), out.sample_rate, 16000)
                back = await asr.transcribe_array(audio, language=lang)
                hyp = " ".join(s["text"] for s in back)
                w.append(wer(text, hyp))
                c.append(cer(text, hyp))
        rows.append({
            "component": "tts", "model": name, "items": len(texts),
            "rtf": round(float(np.mean(rtf)), 3), "latency": percentiles(lat),
            "mean_audio_seconds": round(float(np.mean(dur)), 3),
            "roundtrip_wer": round(float(np.mean([x for x in w if x is not None])), 4),
            "roundtrip_cer": round(float(np.mean([x for x in c if x is not None])), 4),
            "load_seconds": round(load_seconds, 2), "resources": res.summary(),
        })
    return rows


async def bench_engines(items: list[DatasetItem], engines: dict[str, Any], factory: Any,
                        out_dir: Path) -> list[dict[str, Any]]:
    """Whole-system comparison on identical audio files.

    ``engines`` maps a label to ``"cascade"`` or an S2ST provider name. Each
    engine runs through the same executor path as a user's file job.
    """
    from voicebridge.core.jobs.errors import CancelToken
    from voicebridge.core.jobs.executor import PipelineExecutor
    from voicebridge.core.jobs.manager import JobRecord
    from voicebridge.storage import LocalArtifactStore

    store = LocalArtifactStore(out_dir / "store")
    executor = PipelineExecutor(factory, store, {"media": {}})
    rows = []

    class _Mgr:
        async def progress(self, job, stage, fraction=0.0):
            job.cancel.check()

        async def pipeline_event(self, job, event_type, payload):
            return None

    for label, engine in engines.items():
        for item in items:
            key = f"uploads/{label}-{item.id}{Path(item.audio).suffix}"
            await store.save_file(key, Path(item.audio))
            job = JobRecord(job_id=f"{label}-{item.id}", payload={
                "upload_key": key, "filename": Path(item.audio).name,
                "source_language": item.source_language,
                "target_language": item.target_language, "engine": engine,
                "outputs": ["audio", "srt"], "output_format": "wav"}, cancel=CancelToken())
            row: dict[str, Any] = {"component": "engine", "engine": label, "id": item.id}
            with ResourceSampler() as res:
                started = time.perf_counter()
                try:
                    outputs = await executor.run(job, _Mgr())
                    row["status"] = "ok"
                except Exception as exc:  # recorded, not hidden
                    outputs = {}
                    row["status"] = f"error: {getattr(exc, 'code', type(exc).__name__)}: {exc}"
                wall = time.perf_counter() - started
            source_seconds = job.log.get("duration_seconds") or 0.0
            row.update({"wall_seconds": round(wall, 2),
                        "rtf": round(wall / source_seconds, 3) if source_seconds else None,
                        "resources": res.summary(), "models": job.log.get("models")})
            hyp = " ".join(p["translation"] or "" for p in job.result.get("preview", []))
            row["hypothesis"] = hyp
            if item.reference_translation:
                row.update(translation_scores([hyp], [item.reference_translation]))
            wav = next((k for k in outputs if k.endswith(".wav")), None)
            if wav:
                stats = audio_stats(await store.get(outputs[wav]))
                row["output_audio"] = stats
                row["duration_ratio"] = (round(stats["duration"] / source_seconds, 3)
                                         if source_seconds else None)
            row["stage_seconds"] = job.result.get("stage_seconds")
            row["tts"] = job.result.get("tts")
            rows.append(row)
    return rows


def run(coro):
    return asyncio.run(coro)


def tempdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="vb-bench-"))


async def bench_realtime(items: list[DatasetItem], engines: dict[str, Any], factory: Any,
                         chunk_seconds: float = 0.2, pace: bool = True) -> list[dict[str, Any]]:
    """Live-mode latency on identical audio, streamed at real-time pace.

    * **TTFT** -- seconds from stream start to the first translated text;
    * **TTFA** -- seconds from stream start to the first translated audio;
    * **segment latency** -- for each translated segment, wall-clock time of
      its emission minus the source time its audio *ended* (how far behind the
      speaker the translation is).

    Cascade runs through the same :class:`TranslationPipeline` as a browser
    session; SeamlessStreaming through its provider's ``translate_stream``.
    """
    rows = []
    for label, engine in engines.items():
        for item in items:
            rows.append(await _realtime_one(label, engine, item, factory, chunk_seconds, pace))
    return rows


class _LiveRun:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.ttft: float | None = None
        self.ttfa: float | None = None
        self.seg_lat: list[float] = []
        self.texts: list[str] = []

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def text(self, translated: str, source_end: float) -> None:
        now = self.now()
        self.ttft = self.ttft if self.ttft is not None else now
        self.seg_lat.append(now - source_end)
        self.texts.append(translated)

    def audio(self) -> None:
        if self.ttfa is None:
            self.ttfa = self.now()


async def _realtime_one(label: str, engine: str, item: DatasetItem, factory: Any,
                        chunk_seconds: float, pace: bool) -> dict[str, Any]:
    from voicebridge.core.audio.dsp import float_to_pcm16
    from voicebridge.core.metrics.metrics import MetricsRegistry
    from voicebridge.core.session.manager import SessionManager
    from voicebridge.core.types import AudioChunk, EventType

    pcm = float_to_pcm16(_read(item.audio))
    step = int(16000 * 2 * chunk_seconds)
    chunks = [pcm[i:i + step] for i in range(0, len(pcm), step)]
    run = _LiveRun()

    async def feed():
        for i, c in enumerate(chunks):
            if pace:
                delay = run.t0 + i * chunk_seconds - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            yield AudioChunk(data=c)

    if engine == "cascade":
        manager = SessionManager(factory.config, metrics=MetricsRegistry(), factory=factory)
        session = await manager.create({
            "source_language": item.source_language, "target_language": item.target_language,
            "mode": "speech_and_subtitles", "input": {"type": "benchmark"}})
        await manager.start(session.session_id)

        async def consume():
            async for ev in session.pipeline.stream_events():
                if ev.event_type is EventType.TRANSLATION_FINAL:
                    run.text(ev.payload.get("translated_text", ""), float(ev.payload.get("end", 0.0)))
                elif ev.event_type is EventType.TTS_AUDIO:
                    run.audio()

        consumer = asyncio.create_task(consume())
        async for chunk in feed():
            await session.pipeline.push_audio(chunk)
        await manager.stop(session.session_id)
        await consumer
    else:
        provider = factory.s2st(engine)
        async for part in provider.translate_stream(feed(), item.source_language,
                                                    item.target_language):
            for seg in part.segments:
                run.text(seg.translated_text or "", seg.start_time)
            if part.metadata.get("pcm16"):
                run.audio()
    row = {"component": "realtime", "engine": label, "id": item.id,
           "ttft": round(run.ttft, 3) if run.ttft is not None else None,
           "ttfa": round(run.ttfa, 3) if run.ttfa is not None else None,
           "segment_latency": percentiles(run.seg_lat), "segments": len(run.seg_lat),
           "audio_seconds": round(len(pcm) / 32000, 2), "hypothesis": " ".join(run.texts)}
    if item.reference_translation:
        row.update(translation_scores([row["hypothesis"]], [item.reference_translation]))
    return row
