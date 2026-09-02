"""Piper TTS.

Piper is the default real TTS backend: it is MIT-licensed, runs comfortably on
CPU (it is an ONNX VITS variant built for a Raspberry Pi), and ships voices for
many languages. For a project whose main deployment target is a self-hosted box
without a GPU, CPU-real-time synthesis matters more than voice quality.

Voice licensing is **per voice, not per engine.** The Piper software is MIT, but
individual voice models are trained on different corpora and carry their own
terms (varying across CC0, CC-BY, and dataset-specific licences). VoiceBridge
therefore ships no voices and downloads none automatically; the operator points
at a voice file and is responsible for its terms. ``docs/license-matrix.md``
records this explicitly rather than implying "Piper is MIT" covers the audio.

Piper voices are ``.onnx`` files with a sidecar ``.onnx.json`` config. Sample
rate is a property of the voice (commonly 16000 or 22050), so the engine reports
the loaded voice's rate rather than assuming one; the output adapter resamples
to the session's playback rate.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from pathlib import Path
from typing import Any

from voicebridge.core.types import SynthesisedAudio
from voicebridge.providers.base import (
    ProviderError,
    ProviderUnavailable,
    TTSCapabilities,
    TTSEngine,
)
from voicebridge.providers.registry import tts_registry

logger = logging.getLogger(__name__)


class PiperTTSEngine(TTSEngine):
    name = "piper"

    def __init__(
        self,
        voices: dict[str, str] | None = None,
        default_voice: str | None = None,
        **_: object,
    ):
        #: language code -> path to a ``.onnx`` voice file.
        self.voice_paths: dict[str, str] = dict(voices or {})
        self.default_voice = default_voice
        self._loaded: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    @property
    def capabilities(self) -> TTSCapabilities:
        rate = 22050
        for voice in self._loaded.values():
            rate = getattr(getattr(voice, "config", None), "sample_rate", rate)
            break
        return TTSCapabilities(
            name=self.name,
            languages=sorted(self.voice_paths),
            voices=sorted(self.voice_paths.values()),
            streaming=False,
            sample_rate=rate,
            model_license="MIT (engine); voice models carry their own licences",
            commercial_use=None,  # depends on the chosen voice
        )

    def _resolve_voice(self, language: str, speaker: str | None) -> str:
        if speaker and Path(speaker).exists():
            return speaker
        if speaker and speaker in self.voice_paths:
            return self.voice_paths[speaker]
        base = (language or "").split("-")[0]
        path = self.voice_paths.get(language) or self.voice_paths.get(base)
        if path:
            return path
        if self.default_voice:
            return self.default_voice
        raise ProviderError(
            f"no Piper voice configured for language {language!r}. Configure one under "
            "`providers.tts.piper.voices` in config, e.g. "
            "{'en': '/models/piper/en_US-amy-medium.onnx'}. Download voices from "
            "https://huggingface.co/rhasspy/piper-voices and check each voice's licence."
        )

    async def _load(self, path: str) -> Any:
        async with self._lock:
            voice = self._loaded.get(path)
            if voice is not None:
                return voice
            try:
                from piper import PiperVoice
            except ImportError as exc:
                raise ProviderUnavailable(
                    "piper-tts is not installed. Install the TTS extra:\n"
                    "    pip install 'voicebridge[tts-piper]'"
                ) from exc
            if not Path(path).exists():
                raise ProviderError(f"Piper voice file not found: {path}")
            voice = await asyncio.to_thread(PiperVoice.load, path)
            self._loaded[path] = voice
            logger.info("loaded Piper voice %s", path)
            return voice

    async def synthesize(
        self,
        text: str,
        language: str,
        speaker: str | None = None,
        speed: float = 1.0,
        sequence_id: int = 0,
    ) -> SynthesisedAudio:
        clean = (text or "").strip()
        if not clean:
            return SynthesisedAudio(
                audio=b"", sample_rate=16000, channels=1, duration=0.0,
                sequence_id=sequence_id, text="", language=language, provider=self.name,
            )
        path = self._resolve_voice(language, speaker)
        voice = await self._load(path)
        pcm, sample_rate = await asyncio.to_thread(self._synth_sync, voice, clean, speed)
        duration = (len(pcm) / 2) / sample_rate if sample_rate else 0.0
        return SynthesisedAudio(
            audio=pcm,
            sample_rate=sample_rate,
            channels=1,
            duration=duration,
            sequence_id=sequence_id,
            text=clean,
            language=language,
            provider=self.name,
        )

    def _synth_sync(self, voice: Any, text: str, speed: float) -> tuple:
        """Render to PCM.

        Piper's Python API has changed shape across releases (``synthesize`` vs
        ``synthesize_wav``, generator vs WAV writer). Rather than pin one
        version's private behaviour, try the documented entry points in order
        and normalise the result to raw PCM bytes.
        """
        sample_rate = int(getattr(getattr(voice, "config", None), "sample_rate", 22050))
        # length_scale > 1 slows speech down; speed is its inverse.
        length_scale = 1.0 / max(0.25, min(4.0, speed))

        # piper-tts >= 1.3 (verified on 1.7.0, ``piper/voice.py``) takes the
        # rate through ``syn_config=SynthesisConfig(length_scale=...)``;
        # ``synthesize(text, syn_config=None, include_alignments=False)`` and
        # ``synthesize_wav(text, wav_file, syn_config=None, ...)``. Older
        # releases accepted ``length_scale`` as a keyword. Build the config
        # when the class exists so speed is honoured rather than silently
        # dropped by the TypeError fallback below.
        syn_kwargs: dict[str, Any] = {"length_scale": length_scale}
        try:
            from piper.config import SynthesisConfig

            syn_kwargs = {"syn_config": SynthesisConfig(length_scale=length_scale)}
        except ImportError:
            pass

        synth_wav = getattr(voice, "synthesize_wav", None)
        if callable(synth_wav):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                try:
                    synth_wav(text, wf, **syn_kwargs)
                except TypeError:
                    synth_wav(text, wf)
            buf.seek(0)
            with wave.open(buf, "rb") as wf:
                sample_rate = wf.getframerate()
                return wf.readframes(wf.getnframes()), sample_rate

        synth = getattr(voice, "synthesize", None)
        if callable(synth):
            chunks: list[bytes] = []
            try:
                result = synth(text, **syn_kwargs)
            except TypeError:
                result = synth(text)
            if isinstance(result, (bytes, bytearray)):
                return bytes(result), sample_rate
            for item in result:
                data = getattr(item, "audio_int16_bytes", None)
                if data is None:
                    data = item if isinstance(item, (bytes, bytearray)) else None
                if data is None:  # pragma: no cover - unknown shape
                    continue
                rate = getattr(item, "sample_rate", None)
                if rate:
                    sample_rate = int(rate)
                chunks.append(bytes(data))
            return b"".join(chunks), sample_rate

        raise ProviderError(
            "installed piper-tts exposes neither synthesize_wav nor synthesize; "
            "unsupported version"
        )


tts_registry.register("piper", PiperTTSEngine)
