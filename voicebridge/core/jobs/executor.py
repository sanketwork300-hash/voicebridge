"""Executes one file-translation job.

    upload -> probe/validate -> extract + normalise (16 kHz mono WAV on disk)
           -> engine (VoiceBridge cascade | S2ST engine such as SeamlessStreaming)
           -> SpeechTranslationResult (common contract)
           -> subtitles, audio encodes, video mux -> artifact store

The engine is chosen by the job and is **never switched silently**: if the
user asked for SeamlessStreaming and it fails, the job fails with
``SEAMLESS_MODEL_ERROR`` and says why. Everything after the engine is shared,
so the renderer does not care which engine produced the speech.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from voicebridge.core.context.glossary import SessionGlossary
from voicebridge.core.jobs.errors import JobCancelled, PipelineError
from voicebridge.core.pipeline.file_pipeline import FilePipeline, FilePipelineConfig
from voicebridge.core.pipeline.speech_gate import SpeechGateConfig
from voicebridge.core.pipeline.tts_step import TimingConfig
from voicebridge.core.types import (
    EventType,
    HonorificPolicy,
    SpeechTranslationResult,
    TranslationEngineMode,
)
from voicebridge.media.backends import (
    FFmpegMediaProcessor,
    FFmpegVideoRenderer,
    TextSubtitleRenderer,
)
from voicebridge.media.muxer import output_container
from voicebridge.media.normalizer import NormalizedAudio
from voicebridge.media.probe import MediaInfo
from voicebridge.providers.base import ProviderError, ProviderUnavailable
from voicebridge.runtime import describe_runtime
from voicebridge.storage.base import ArtifactStore

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm"}
AUDIO_OUTPUTS = {"wav", "mp3", "m4a", "flac", "ogg"}


class PipelineExecutor:
    def __init__(self, factory: Any, store: ArtifactStore, app_config: dict[str, Any] | None = None,
                 media: Any = None, subtitles: Any = None, video: Any = None):
        self.factory = factory
        self.store = store
        self.cfg = app_config or {}
        self.media = media or FFmpegMediaProcessor()
        self.subtitles = subtitles or TextSubtitleRenderer()
        self.video = video or FFmpegVideoRenderer()

    # -- helpers --------------------------------------------------------------------

    def _pipeline_config(self, job_payload: dict[str, Any]) -> FilePipelineConfig:
        prov = self.factory.config
        gate = dict(prov.get("speech_gate") or {})
        tts_cfg = prov.get("tts") or {}
        tcfg = prov.get("translation") or {}
        glossary = SessionGlossary.from_payload(
            {"terms": list(self.cfg.get("glossary") or []) + list(job_payload.get("glossary") or [])})
        honor = job_payload.get("honorifics") or (self.cfg.get("honorific_policy") or {}).get(
            "mode") or "preserve"
        timing = TimingConfig.from_dict({"regenerate": True, **(tts_cfg.get("timing") or {})})
        return FilePipelineConfig(
            source_language=job_payload.get("source_language") or "auto",
            target_language=job_payload.get("target_language") or "en",
            synthesize=job_payload.get("synthesize", True),
            voice=job_payload.get("voice") or tts_cfg.get("default_voice") or "source",
            voice_map=dict(tts_cfg.get("voice_map") or {}),
            honorifics=HonorificPolicy(honor),
            glossary=glossary,
            style=job_payload.get("style"),
            context_segments=int(tcfg.get("context_segments", 8)),
            context_tokens=int((tcfg.get("context") or {}).get("max_tokens", 2048)),
            translation_validation=str(job_payload.get("validation")
                                       or tcfg.get("validation", "balanced")),
            speech_gate=SpeechGateConfig.from_dict(
                {k: v for k, v in gate.items() if k in SpeechGateConfig.__dataclass_fields__}),
            asr_validation=dict(prov.get("asr_validation") or {}),
            timing=timing,
            release_between_stages=bool(self.factory.runtime.get("low_memory", False)),
            asr_prompt_previous_text=bool((prov.get("asr") or {}).get("prompt_previous_text",
                                                                       False)),
        )

    def _describe_models(self, providers: Any, engine: str) -> dict[str, Any]:
        def name(p: Any) -> str | None:
            if p is None:
                return None
            model = getattr(p, "model", None) or getattr(getattr(p, "backend", None), "model", None)
            return f"{p.name}:{model}" if model else p.name
        if engine == "cascade":
            return {"asr": name(providers.asr), "translation": name(providers.translation),
                    "tts": name(providers.tts), "vad": name(providers.vad),
                    "audio_event": name(providers.audio_event)}
        return {"s2st": name(providers)}

    # -- run ---------------------------------------------------------------------

    async def run(self, job, manager) -> dict[str, str]:
        p = job.payload
        engine = TranslationEngineMode(p.get("engine") or "cascade")
        if engine is TranslationEngineMode.BENCHMARK:
            raise PipelineError("job", "BENCHMARK_NOT_A_JOB",
                                "Benchmark mode runs through `python -m voicebridge.benchmark`.")
        started = time.perf_counter()
        scratch = self.store.scratch(job.job_id)
        upload = await self.store.get(p["upload_key"])

        await manager.progress(job, "preprocessing", 0.0)
        info = await self.media.probe(str(upload))
        max_seconds = float((self.cfg.get("media") or {}).get("max_duration_seconds", 4 * 3600))
        if info.duration > max_seconds:
            raise PipelineError("preprocessing", "MEDIA_TOO_LONG",
                                f"Media is {info.duration:.0f} s; the limit is {max_seconds:.0f} s.")
        is_video = Path(p["filename"]).suffix.lower() in VIDEO_EXTENSIONS and info.has_video
        await manager.progress(job, "preprocessing", 0.3)
        normalized = await self.media.extract_audio(str(upload), str(scratch / "source.16k.wav"))
        audio = NormalizedAudio(normalized)
        await manager.progress(job, "preprocessing", 1.0)
        job.log.update({
            "engine": engine.value, "source_language": p.get("source_language"),
            "target_language": p.get("target_language"), "media": info.summary(),
            "duration_seconds": round(audio.duration, 3),
            "runtime": describe_runtime(self.factory.runtime),
        })

        async def emit(event_type: EventType, payload: dict[str, Any]) -> None:
            await manager.pipeline_event(job, event_type.value, payload)

        async def progress(stage: str, fraction: float) -> None:
            await manager.progress(job, stage, fraction)

        if engine is TranslationEngineMode.CASCADE:
            result = await self._run_cascade(job, audio, scratch, emit, progress)
        else:
            result = await self._run_s2st(job, normalized, audio, scratch, emit, progress)

        await manager.progress(job, "rendering", 0.0)
        outputs = await self._render(job, result, upload, info, is_video, scratch)
        await manager.progress(job, "rendering", 1.0)
        wall = time.perf_counter() - started
        job.log["processing_seconds"] = round(wall, 2)
        job.log["rtf"] = round(wall / audio.duration, 3) if audio.duration else None
        job.result = {
            "segments": len(result.segments),
            "detected_language": result.source_language,
            "subtitles_available": any(s.metadata.get("timestamps_available", True)
                                       for s in result.segments),
            "preview": [{"start": s.start_time, "end": s.end_time, "source": s.source_text,
                         "translation": s.translated_text} for s in result.segments[:200]],
            **{k: v for k, v in result.metadata.items() if k in ("stats", "tts",
                                                                 "stage_seconds")},
        }
        return outputs

    async def _run_cascade(self, job, audio, scratch, emit, progress) -> SpeechTranslationResult:
        p = job.payload
        wants_audio = p.get("synthesize", True)
        try:
            providers = self.factory.provider_set(
                wants_tts=wants_audio, translation=p.get("translation_provider"),
                translation_preset=p.get("translation_quality"), tts=p.get("tts_provider"),
                target_language=p.get("target_language"))
        except ProviderUnavailable as exc:
            raise PipelineError("preprocessing", "PROVIDER_UNAVAILABLE", str(exc)) from exc
        if wants_audio and providers.tts is None:
            raise PipelineError("synthesizing", "TTS_UNAVAILABLE",
                                "No TTS provider is available for dubbed output.")
        job.log["models"] = self._describe_models(providers, "cascade")
        pipeline = FilePipeline(providers, self._pipeline_config(p), emit, progress, job.cancel)
        return await pipeline.run(audio, scratch)

    async def _run_s2st(self, job, normalized, audio, scratch, emit, progress
                        ) -> SpeechTranslationResult:
        p = job.payload
        name = p.get("engine")
        try:
            provider = self.factory.s2st(name)
            if hasattr(provider, "gate") and provider.gate is None:
                # Offline S2ST backends reuse the shared speech gate so music/SFX
                # are skipped the same way as in the cascade.
                from voicebridge.core.pipeline.speech_gate import SpeechGate

                vad, classifier = self.factory.speech_gate_providers()
                provider.gate = SpeechGate(vad, classifier,
                                           self._pipeline_config(p).speech_gate)
            job.log["models"] = self._describe_models(provider, name)
            job.log["license_notice"] = provider.capabilities.model_license
            await progress("transcribing", 0.0)
            result = await provider.translate_file(
                str(normalized), p.get("source_language") or "auto",
                p.get("target_language") or "en", workdir=str(scratch),
                progress=progress, emit=emit, cancel=job.cancel)
        except (JobCancelled, PipelineError):
            raise
        except (ProviderError, ProviderUnavailable, RuntimeError, ValueError) as exc:
            raise PipelineError("translating", "SEAMLESS_MODEL_ERROR", str(exc)) from exc
        await progress("synthesizing", 1.0)
        return result

    async def _render(self, job, result: SpeechTranslationResult, upload: Path, info: MediaInfo,
                      is_video: bool, scratch: Path) -> dict[str, str]:
        p = job.payload
        stem = Path(p["filename"]).stem or "media"
        tgt = p.get("target_language") or "en"
        wanted = set(p.get("outputs") or (["video", "srt", "vtt", "audio"] if is_video
                                           else ["audio", "srt", "vtt"]))
        subtitles_on = p.get("subtitle_enabled", True)
        outputs: dict[str, str] = {}

        async def keep(name: str, path: Path) -> None:
            outputs[name] = await self.store.save_file(f"outputs/{job.job_id}/{name}", path)

        srt_path = None
        timed = [s for s in result.segments if s.metadata.get("timestamps_available", True)]
        if subtitles_on and timed:
            srt_path = scratch / f"{stem}.{tgt}.srt"
            dual = bool(p.get("dual_subtitles"))
            srt_path.write_text(self.subtitles.render(timed, "srt", dual), "utf-8")
            vtt_path = scratch / f"{stem}.{tgt}.vtt"
            vtt_path.write_text(self.subtitles.render(timed, "vtt", dual), "utf-8")
            if "srt" in wanted:
                await keep(srt_path.name, srt_path)
            if "vtt" in wanted:
                await keep(vtt_path.name, vtt_path)
        elif subtitles_on:
            job.log["subtitles"] = "unavailable: engine returned no timed text"

        dub = Path(result.audio_path) if result.audio_path else None
        if dub is not None and "audio" in wanted:
            fmt = (p.get("output_format") or "").lower()
            formats = [fmt] if fmt in AUDIO_OUTPUTS else ["wav", "mp3"]
            if is_video and fmt not in AUDIO_OUTPUTS:
                formats = ["wav"]
            for f in formats:
                target = scratch / f"{stem}.{tgt}.{f}"
                await self.media.encode_audio(str(dub), str(target))
                await keep(target.name, target)

        if is_video:
            container = output_container(Path(p["filename"]).suffix)
            mode = p.get("audio_mode") or "replace"
            if dub is not None and "video" in wanted:
                target = scratch / f"{stem}.{tgt}{container}"
                await self.video.render(
                    str(upload), str(target), audio_path=str(dub),
                    subtitle_path=str(srt_path) if srt_path and p.get("embed_subtitles", True)
                    else None,
                    mode=mode, original_gain=float(p.get("original_gain", 0.25)),
                    audio_language=tgt, subtitle_language=tgt)
                await keep(target.name, target)
            if srt_path is not None and ("subtitled_video" in wanted or dub is None):
                target = scratch / f"{stem}.{tgt}.subtitled{container}"
                await self.video.render(str(upload), str(target), audio_path=None,
                                        subtitle_path=str(srt_path), subtitle_language=tgt)
                await keep(target.name, target)
        if not outputs:
            raise PipelineError("rendering", "NO_DIALOGUE_DETECTED",
                                "Nothing translatable was found (no dialogue detected).")
        return outputs
