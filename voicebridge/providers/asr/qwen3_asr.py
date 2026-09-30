"""Qwen3-ASR provider (``Qwen/Qwen3-ASR-1.7B``, Apache-2.0).

Offline only in VoiceBridge: it implements :meth:`transcribe_array` for the file
pipeline and the benchmark. The model runs in the ``qwen`` worker virtualenv
(:mod:`voicebridge.workers.qwen_asr_worker`) because ``qwen-asr`` pins its own
transformers version.

For live sessions keep WhisperLiveKit. (WhisperLiveKit 0.2.26 itself ships a
``qwen3-streaming`` backend; selecting it through ``providers.asr.backend`` is the
supported way to stream with Qwen3-ASR, and is independent of this provider.)

Without the optional forced aligner, a region yields one segment spanning the
region, whose times come from the speech gate. With ``aligner:
Qwen/Qwen3-ForcedAligner-0.6B`` the words carry model timestamps.
Qwen3-ASR exposes no decoder confidence, so ASR validation relies on text
and speech-gate signals for this provider.
"""

from __future__ import annotations

import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from voicebridge.core.audio.dsp import write_wav
from voicebridge.core.types import AudioChunk, Event
from voicebridge.providers.base import ASRCapabilities, ASREngine, ProviderUnavailable
from voicebridge.providers.registry import asr_registry
from voicebridge.workers.client import ModelWorker, resolve_python


class Qwen3ASREngine(ASREngine):
    name = "qwen3_asr"

    def __init__(self, model: str = "Qwen/Qwen3-ASR-1.7B", aligner: str | None = None,
                 device: str = "auto", dtype: str = "auto", python: str | None = None,
                 threads: int = 0, idle_unload_seconds: float | None = None, **_: object):
        self.model = model
        self.aligner = aligner
        self.worker = ModelWorker(
            "qwen3-asr", "qwen_asr_worker.py",
            resolve_python(python, "VOICEBRIDGE_QWEN_PYTHON", "qwen"),
            args=["--model", model, "--aligner", aligner or "", "--device", device,
                  "--dtype", dtype, "--threads", str(int(threads or 0))],
            idle_unload_seconds=idle_unload_seconds,
        )
        self._tmp = Path(tempfile.mkdtemp(prefix="vb-qwen-asr-"))

    @property
    def capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(
            name=self.name, streaming=False, language_detection=True,
            word_timestamps=bool(self.aligner),
            languages=["auto", "en", "ja", "ko", "zh", "hi", "de", "fr", "es", "ru", "pt", "it"],
            emits_partial_and_stable=False,
        )

    async def warmup(self) -> None:
        await self.worker.call("load")

    async def close(self) -> None:
        await self.worker.stop()

    async def transcribe_array(self, audio: Any, language: str | None = None,
                               initial_prompt: str | None = None) -> list[dict[str, Any]]:
        path = self._tmp / f"{uuid.uuid4().hex}.wav"
        write_wav(str(path), audio, 16000)
        try:
            out = await self.worker.call("transcribe", path=str(path),
                                         language=language if language != "auto" else None,
                                         context=(initial_prompt or "")[-200:])
        finally:
            path.unlink(missing_ok=True)
        text = (out.get("text") or "").strip()
        if not text:
            return []
        duration = len(audio) / 16000
        words = out.get("words") or []
        start = words[0]["start"] if words else 0.0
        end = words[-1]["end"] if words else duration
        return [{"text": text, "start": float(start), "end": float(max(end, start + 0.1)),
                 "language": out.get("language"), "language_probability": None,
                 "words": words}]

    # -- streaming session API: not provided by this provider ------------------

    async def start_session(self, session_id: str, language: str | None = None,
                            **options: object) -> None:
        raise ProviderUnavailable(
            "qwen3_asr is an offline provider (file mode and benchmarks). For live sessions "
            "use whisperlivekit, optionally with WhisperLiveKit's qwen3-streaming backend.")

    async def push_audio(self, session_id: str, chunk: AudioChunk) -> None:
        return None

    async def get_events(self, session_id: str) -> AsyncIterator[Event]:
        return
        yield  # pragma: no cover

    async def stop_session(self, session_id: str) -> None:
        return None


asr_registry.register("qwen3_asr", Qwen3ASREngine)
