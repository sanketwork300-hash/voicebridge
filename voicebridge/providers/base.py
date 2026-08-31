"""Provider interfaces.

Every model backend in VoiceBridge sits behind one of these three ABCs. The
pipeline imports *only* these; no module under ``core/`` may import a concrete
provider. That is what allows a mock, a local CPU model and a remote GPU worker
to be swapped by configuration alone.

Capability declaration
----------------------
Providers differ in what they can do, and the pipeline must adapt rather than
assume. Each provider therefore publishes a ``capabilities`` object: whether it
can detect language, whether it emits word timestamps, which language pairs it
supports. The pipeline queries this instead of hard-coding per-provider
behaviour, and the gateway exposes it at ``GET /v1/providers``.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from voicebridge.core.types import (
    AudioChunk,
    Event,
    SynthesisedAudio,
    TranslationResult,
)


class ProviderError(RuntimeError):
    """Raised when a provider fails in a way the pipeline should degrade around."""


class ProviderUnavailable(ProviderError):
    """The provider's dependencies or models are not installed/reachable."""


# --------------------------------------------------------------------------
# Capabilities
# --------------------------------------------------------------------------


@dataclass
class ASRCapabilities:
    name: str = ""
    streaming: bool = True
    language_detection: bool = False
    word_timestamps: bool = False
    diarization: bool = False
    languages: Sequence[str] = field(default_factory=list)
    #: True when the backend has its own commitment policy (LocalAgreement,
    #: SimulStreaming...). When False the pipeline must treat every result as
    #: final-on-arrival and cannot distinguish partial from stable.
    emits_partial_and_stable: bool = True


@dataclass
class TranslationCapabilities:
    name: str = ""
    #: Explicit directed pairs, e.g. ``[("ja", "en"), ("ko", "en")]``. Empty
    #: means "ask :meth:`TranslationEngine.supports` instead".
    pairs: Sequence[tuple] = field(default_factory=list)
    source_languages: Sequence[str] = field(default_factory=list)
    target_languages: Sequence[str] = field(default_factory=list)
    supports_context: bool = False
    supports_glossary: bool = True  # via placeholder protection, provider-agnostic
    #: Licence of the model weights this provider loads by default. Surfaced in
    #: ``GET /v1/providers`` so an operator can see NC restrictions at runtime.
    model_license: str = "unknown"
    commercial_use: bool | None = None


@dataclass
class TTSCapabilities:
    name: str = ""
    languages: Sequence[str] = field(default_factory=list)
    voices: Sequence[str] = field(default_factory=list)
    streaming: bool = False
    sample_rate: int = 16000
    model_license: str = "unknown"
    commercial_use: bool | None = None


# --------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------


class ASREngine(abc.ABC):
    """Streaming speech recognition.

    Lifecycle: ``start_session`` → many ``push_audio`` → ``get_events`` consumed
    concurrently → ``stop_session``.

    ``get_events`` yields :class:`~voicebridge.core.types.Event` objects of type
    ``ASR_PARTIAL``, ``ASR_STABLE``, ``ASR_FINAL``, ``LANGUAGE_DETECTED``,
    ``SPEAKER_CHANGED`` or ``ERROR``.
    """

    name: str = "asr"

    @property
    @abc.abstractmethod
    def capabilities(self) -> ASRCapabilities: ...

    @abc.abstractmethod
    async def start_session(
        self,
        session_id: str,
        language: str | None = None,
        **options: object,
    ) -> None: ...

    @abc.abstractmethod
    async def push_audio(self, session_id: str, chunk: AudioChunk) -> None: ...

    @abc.abstractmethod
    def get_events(self, session_id: str) -> AsyncIterator[Event]: ...

    @abc.abstractmethod
    async def stop_session(self, session_id: str) -> None: ...

    async def warmup(self) -> None:
        """Optionally preload weights so the first session is not slow."""
        return None


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------


class TranslationEngine(abc.ABC):
    name: str = "translation"

    @property
    @abc.abstractmethod
    def capabilities(self) -> TranslationCapabilities: ...

    @abc.abstractmethod
    async def translate(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        context: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TranslationResult: ...

    def supports(self, source_language: str, target_language: str) -> bool:
        caps = self.capabilities
        if caps.pairs:
            return (source_language, target_language) in {tuple(p) for p in caps.pairs}
        src_ok = not caps.source_languages or source_language in caps.source_languages
        dst_ok = not caps.target_languages or target_language in caps.target_languages
        return src_ok and dst_ok

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None


# --------------------------------------------------------------------------
# TTS
# --------------------------------------------------------------------------


class TTSEngine(abc.ABC):
    name: str = "tts"

    @property
    @abc.abstractmethod
    def capabilities(self) -> TTSCapabilities: ...

    @abc.abstractmethod
    async def synthesize(
        self,
        text: str,
        language: str,
        speaker: str | None = None,
        speed: float = 1.0,
        sequence_id: int = 0,
    ) -> SynthesisedAudio: ...

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None
