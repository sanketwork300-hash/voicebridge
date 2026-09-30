"""Qwen3-TTS provider.

The model runs in a worker process (:mod:`voicebridge.workers.qwen_tts_worker`)
inside its own virtualenv, because ``qwen-tts`` pins a transformers version the
gateway environment does not use. See :mod:`voicebridge.workers.client`.

Voices
------
``Qwen/Qwen3-TTS-12Hz-1.7B-Base`` is a *voice-cloning* model: it has no built-in
speakers and needs a reference clip. VoiceBridge resolves the reference per
request, in this order:

1. ``TTSRequest.reference_audio`` -- set by the pipeline when the voice is
   ``source``: the source speaker's own dialogue, so the dub keeps each
   speaker's timbre (speaker-embedding-only cloning; no transcript needed);
2. a configured named voice: ``voices: {narrator: {ref_audio: path, ref_text: ...}}``
   (with ``ref_text`` the model uses in-context cloning, usually closer);
3. otherwise the request fails with an explanation -- nothing is invented.

CustomVoice checkpoints (``...-CustomVoice``) have preset speakers and accept an
``instruct`` string, which is where ``emotion``/``style`` go. The Base model
ignores style, and has no rate control, so ``speaking_rate`` is applied by the
pipeline's time-stretch step (``capabilities.speaking_rate = False``).

Model licence: Apache-2.0 (checked on the Hugging Face model card).
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from voicebridge.core.audio.dsp import float_to_pcm16, read_wav
from voicebridge.core.types import SynthesisedAudio, TTSRequest
from voicebridge.providers.base import ProviderUnavailable, TTSCapabilities, TTSEngine
from voicebridge.providers.registry import tts_registry
from voicebridge.workers.client import ModelWorker, resolve_python

logger = logging.getLogger(__name__)

SUPPORTED = ["en", "ja", "ko", "zh", "de", "fr", "ru", "pt", "es", "it"]


class Qwen3TTSEngine(TTSEngine):
    name = "qwen3"

    def __init__(
        self,
        model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device: str = "auto",
        dtype: str = "auto",
        python: str | None = None,
        threads: int = 0,
        voices: dict[str, dict[str, str]] | None = None,
        default_voice: str | None = "source",
        idle_unload_seconds: float | None = None,
        max_new_tokens: int = 1024,
        **_: object,
    ):
        self.model = model
        self.voices = dict(voices or {})
        self.default_voice = default_voice
        self.max_new_tokens = int(max_new_tokens)
        self.custom_voice = "customvoice" in model.lower().replace("-", "").replace("_", "")
        self.worker = ModelWorker(
            "qwen3-tts", "qwen_tts_worker.py",
            resolve_python(python, "VOICEBRIDGE_QWEN_PYTHON", "qwen"),
            args=["--model", model, "--device", device, "--dtype", dtype,
                  "--threads", str(int(threads or 0))],
            idle_unload_seconds=idle_unload_seconds,
        )
        self._tmp = Path(tempfile.mkdtemp(prefix="vb-qwen-tts-"))
        self.load_info: dict[str, Any] = {}

    @property
    def capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            name=self.name,
            languages=SUPPORTED,
            voices=(["source"] if not self.custom_voice else []) + sorted(self.voices),
            streaming=False,
            sample_rate=24000,
            model_license="Apache-2.0",
            commercial_use=True,
            speaking_rate=False,
            voice_cloning=not self.custom_voice,
            style_control=self.custom_voice,
        )

    async def warmup(self) -> None:
        self.load_info = await self.worker.call("load")
        logger.info("Qwen3-TTS loaded: %s", self.load_info)

    async def close(self) -> None:
        await self.worker.stop()

    async def synthesize(self, text: str, language: str, speaker: str | None = None,
                         speed: float = 1.0, sequence_id: int = 0) -> SynthesisedAudio:
        return await self.synthesize_request(
            TTSRequest(text=text, language=language, speaker=speaker, speaking_rate=speed,
                       sequence_id=sequence_id))

    def _reference(self, request: TTSRequest) -> tuple[str | None, str | None]:
        voice = request.speaker or self.default_voice
        if voice and voice != "source" and voice in self.voices:
            cfg = self.voices[voice]
            return cfg.get("ref_audio"), cfg.get("ref_text")
        if request.reference_audio:
            return request.reference_audio, request.reference_text
        if self.default_voice and self.default_voice in self.voices:
            cfg = self.voices[self.default_voice]
            return cfg.get("ref_audio"), cfg.get("ref_text")
        return None, None

    async def synthesize_request(self, request: TTSRequest) -> SynthesisedAudio:
        text = (request.text or "").strip()
        if not text:
            return SynthesisedAudio(b"", 24000, 1, 0.0, request.sequence_id, provider=self.name)
        if request.language not in SUPPORTED:
            raise ProviderUnavailable(f"Qwen3-TTS does not support language {request.language!r}")
        params: dict[str, Any] = {"text": text, "language": request.language,
                                  "max_new_tokens": self.max_new_tokens}
        if self.custom_voice:
            params["speaker"] = request.speaker if request.speaker not in (None, "source") else None
            instruct = ", ".join(x for x in (request.emotion, request.style) if x)
            params["instruct"] = instruct or None
        else:
            ref_audio, ref_text = self._reference(request)
            if not ref_audio:
                raise ProviderUnavailable(
                    "Qwen3-TTS Base needs a reference voice: use voice 'source' (clone the "
                    "original speaker) or configure tts.voices.<name>.ref_audio")
            params["ref_audio"] = str(ref_audio)
            params["ref_text"] = ref_text
        out = self._tmp / f"{uuid.uuid4().hex}.wav"
        params["out_path"] = str(out)
        result = await self.worker.call("synthesize", **params)
        audio, rate = read_wav(result["path"])
        out.unlink(missing_ok=True)
        return SynthesisedAudio(
            audio=float_to_pcm16(audio), sample_rate=rate, channels=1,
            duration=len(audio) / rate, sequence_id=request.sequence_id, text=text,
            language=request.language, provider=self.name,
            source_start=request.source_start, source_end=request.source_end,
        )


tts_registry.register("qwen3", Qwen3TTSEngine)
tts_registry.register("qwen3_tts", Qwen3TTSEngine)
