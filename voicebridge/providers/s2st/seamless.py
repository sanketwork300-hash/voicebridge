"""SeamlessM4T v2 (offline) speech-to-speech reference backend.

Model: ``facebook/seamless-m4t-v2-large`` through ``transformers``
(``SeamlessM4TProcessor`` + ``SeamlessM4Tv2Model``). API used, verified in
transformers 5.16: ``processor(audio=..., sampling_rate=16000, return_tensors="pt")``
and ``model.generate(input_features=..., tgt_lang="eng", generate_speech=True,
return_intermediate_token_ids=True)`` -> ``SeamlessM4Tv2GenerationOutput``
with ``waveform``, ``waveform_lengths`` and ``sequences`` (the translated text
tokens). Output speech is 16 kHz.

This is *not* streaming: it translates one speech region at a time. Regions come
from the shared speech gate (Silero VAD + event classifier), so music/SFX is
skipped exactly as in the cascade, and every segment carries the **source**
timestamps of its region (``metadata.timing = "source_region"``).

Licence: CC-BY-NC-4.0 weights. Disabled unless ``enabled: true``; weights are
fetched only by an explicit operator download.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from voicebridge.core.types import AudioChunk, SpeechTranslationResult, TranslationSegment
from voicebridge.providers.base import ProviderUnavailable, S2STCapabilities, S2STProvider
from voicebridge.providers.registry import s2st_registry
from voicebridge.runtime import resolve_device, resolve_dtype

logger = logging.getLogger(__name__)

LICENSE = "CC-BY-NC-4.0 (model weights; research/evaluation)"
LANG3 = {"en": "eng", "ja": "jpn", "ko": "kor", "hi": "hin", "zh": "cmn", "es": "spa",
         "fr": "fra", "de": "deu", "pt": "por", "ru": "rus", "it": "ita"}


class SeamlessM4Tv2Provider(S2STProvider):
    name = "seamless_m4t_v2"

    def __init__(self, enabled: bool = False, model: str = "facebook/seamless-m4t-v2-large",
                 device: str = "auto", dtype: str = "auto", max_region_seconds: float = 20.0,
                 gate: Any = None, **_: object):
        self.enabled = bool(enabled)
        self.model_name = model
        self.model = model
        self.device_pref = device
        self.dtype_pref = dtype
        self.max_region_seconds = float(max_region_seconds)
        self.gate = gate
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.Lock()

    @property
    def capabilities(self) -> S2STCapabilities:
        return S2STCapabilities(name=self.name, streaming=False, file=True,
                                source_languages=list(LANG3), target_languages=list(LANG3),
                                model_license=LICENSE, commercial_use=False)

    async def initialize(self, config: dict[str, object] | None = None) -> None:
        if not self.enabled:
            raise ProviderUnavailable(
                "SeamlessM4T v2 is disabled. Set providers.s2st.seamless_m4t_v2.enabled: true "
                "after reviewing the CC-BY-NC-4.0 model licence.")

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import SeamlessM4TProcessor, SeamlessM4Tv2Model

            device = resolve_device(self.device_pref)
            dtype = resolve_dtype(self.dtype_pref, device)
            logger.info("loading %s on %s (%s)", self.model_name, device, dtype)
            self._processor = SeamlessM4TProcessor.from_pretrained(self.model_name)
            model = SeamlessM4Tv2Model.from_pretrained(self.model_name, dtype=dtype)
            model.eval().to(device)
            self._torch, self._device, self._dtype = torch, device, dtype
            self._model = model

    async def warmup(self) -> None:
        await self.initialize()
        await asyncio.to_thread(self._load)

    async def shutdown(self) -> None:
        self._model = None
        self._processor = None

    close = shutdown

    def translate_region(self, audio: np.ndarray, target: str) -> tuple[str, np.ndarray]:
        self._load()
        inputs = self._processor(audio=audio, sampling_rate=16000, return_tensors="pt")
        feats = inputs["input_features"].to(self._device, self._dtype)
        mask = inputs.get("attention_mask")
        with self._torch.inference_mode():
            out = self._model.generate(
                input_features=feats,
                attention_mask=mask.to(self._device) if mask is not None else None,
                tgt_lang=LANG3.get(target, target), generate_speech=True,
                return_intermediate_token_ids=True)
        # For a single clip the model may return 0-dim lengths / a 1-D waveform.
        length = (int(out.waveform_lengths.reshape(-1)[0])
                  if out.waveform_lengths is not None else None)
        wav = out.waveform
        wav = (wav[0] if wav.dim() > 1 else wav).float().cpu().numpy()
        if length:
            wav = wav[:length]
        seq = out.sequences
        text = self._processor.decode((seq[0] if seq.dim() > 1 else seq).tolist(),
                                      skip_special_tokens=True)
        return text.strip(), wav

    async def translate_stream(self, audio_stream: AsyncIterator[AudioChunk], source_language: str,
                               target_language: str) -> AsyncIterator[SpeechTranslationResult]:
        raise ProviderUnavailable("seamless_m4t_v2 is offline-only; use seamless_streaming live")
        yield  # pragma: no cover

    async def translate_file(self, audio_path: str, source_language: str, target_language: str,
                             workdir: str | None = None, progress=None, emit=None,
                             cancel=None) -> SpeechTranslationResult:
        from voicebridge.media.normalizer import NormalizedAudio
        from voicebridge.media.renderer import TimelineRenderer

        await self.initialize()
        audio = NormalizedAudio(audio_path)
        regions: list[tuple[float, float]] = []
        if self.gate is not None:
            for offset, block in audio.blocks(600.0):
                for d in await self.gate.detect_regions(block, offset):
                    if d.should_transcribe:
                        regions.append((d.start_time, d.end_time))
        else:
            regions = [(0.0, audio.duration)]
        merged: list[list[float]] = []
        for a, b in regions:
            if merged and a - merged[-1][1] < 0.6 and b - merged[-1][0] <= self.max_region_seconds:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        work = Path(workdir or ".")
        renderer = TimelineRenderer(audio.duration, 16000, work / "render_s2st")
        segments: list[TranslationSegment] = []
        try:
            for i, (a, b) in enumerate(merged, start=1):
                if cancel is not None:
                    cancel.check()
                text, wav = await asyncio.to_thread(self.translate_region, audio.read(a, b),
                                                    target_language)
                if wav.size:
                    await asyncio.to_thread(renderer.place, f"m4t-{i}", wav, 16000, a)
                if text:
                    segments.append(TranslationSegment(
                        id=f"m4t-{i:05d}", start_time=round(a, 3), end_time=round(b, 3),
                        translated_text=text, source_language=source_language,
                        target_language=target_language, event_type="DIALOGUE",
                        metadata={"timing": "source_region", "timestamps_available": True}))
                if progress is not None:
                    await progress("translating", i / max(1, len(merged)))
            out = await asyncio.to_thread(renderer.write_wav, work / "seamless_m4t_dub.wav")
        finally:
            renderer.close()
        return SpeechTranslationResult(
            audio_path=str(out), source_language=source_language, target_language=target_language,
            segments=segments, duration=audio.duration,
            metadata={"engine": self.name, "timing": "source_region", "regions": len(merged),
                      "license": LICENSE})


s2st_registry.register("seamless_m4t_v2", SeamlessM4Tv2Provider)
s2st_registry.register("seamless", SeamlessM4Tv2Provider)
