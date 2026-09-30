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
from typing import Any

from voicebridge.core.types import (
    AudioChunk,
    Event,
    SpeechDecision,
    SpeechTranslationResult,
    SynthesisedAudio,
    TranslationResult,
    TTSRequest,
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
    #: The engine honours ``speaking_rate`` natively. When False the pipeline
    #: time-stretches the output instead.
    speaking_rate: bool = True
    #: The engine clones a voice from ``TTSRequest.reference_audio``.
    voice_cloning: bool = False
    #: The engine interprets ``emotion``/``style`` text.
    style_control: bool = False


@dataclass
class VADCapabilities:
    name: str = ""
    streaming: bool = True


@dataclass
class AudioEventCapabilities:
    name: str = ""
    event_types: Sequence[str] = field(default_factory=list)


@dataclass
class S2STCapabilities:
    name: str = ""
    streaming: bool = True
    file: bool = True
    source_languages: Sequence[str] = field(default_factory=list)
    target_languages: Sequence[str] = field(default_factory=list)
    model_license: str = "model-specific"
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

    async def transcribe_file(
        self,
        audio_path: str,
        source_language: str | None = None,
        **options: object,
    ) -> list[dict[str, object]]:
        raise ProviderUnavailable(f"{self.name} does not support file transcription")

    async def transcribe_array(
        self,
        audio: Any,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Offline transcription of one float32 16 kHz mono region (file mode).

        Returns segments with times relative to the start of ``audio``::

            {"text", "start", "end", "language", "language_probability",
             "words": [{"text", "start", "end", "probability"}],
             # optional decoder quality signals, used by ASR validation:
             "avg_logprob", "compression_ratio", "no_speech_prob"}

        The file pipeline calls this instead of the streaming session API
        because a file does not need a commitment policy, and batch decoding
        exposes the quality signals the hallucination filter depends on.
        """
        raise ProviderUnavailable(f"{self.name} does not support offline transcription")


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

    async def synthesize_request(self, request: TTSRequest) -> SynthesisedAudio:
        return await self.synthesize(
            request.text,
            request.language,
            speaker=request.speaker,
            speed=request.speaking_rate,
            sequence_id=request.sequence_id,
        )


# --------------------------------------------------------------------------
# Speech gate providers
# --------------------------------------------------------------------------


class VADProvider(abc.ABC):
    name: str = "vad"

    @property
    @abc.abstractmethod
    def capabilities(self) -> VADCapabilities: ...

    @abc.abstractmethod
    async def detect(self, chunk: AudioChunk) -> SpeechDecision: ...


class AudioEventClassifier(abc.ABC):
    name: str = "audio_event_classifier"

    @property
    @abc.abstractmethod
    def capabilities(self) -> AudioEventCapabilities: ...

    @abc.abstractmethod
    async def classify(self, chunk: AudioChunk) -> SpeechDecision: ...


class MediaProcessor(abc.ABC):
    """Probe untrusted media and decode its audio to the pipeline format."""

    name: str = "media_processor"

    @abc.abstractmethod
    async def probe(self, path: str) -> Any: ...

    @abc.abstractmethod
    async def extract_audio(self, path: str, output_path: str) -> Any: ...

    @abc.abstractmethod
    async def encode_audio(self, wav_path: str, output_path: str) -> Any: ...


class SubtitleRenderer(abc.ABC):
    """Render timed segments to a subtitle format; never invents timing."""

    name: str = "subtitle_renderer"
    extensions: tuple[str, ...] = ()

    @abc.abstractmethod
    def render(self, segments: Sequence[Any], fmt: str, dual: bool = False) -> str: ...


class VideoRenderer(abc.ABC):
    """Combine source video with translated audio and/or subtitles."""

    name: str = "video_renderer"

    @abc.abstractmethod
    async def render(self, video_path: str, output_path: str, audio_path: str | None = None,
                     subtitle_path: str | None = None, mode: str = "replace",
                     **options: Any) -> Any: ...


# --------------------------------------------------------------------------
# Direct speech-to-speech translation
# --------------------------------------------------------------------------


class S2STProvider(abc.ABC):
    name: str = "s2st"

    @property
    @abc.abstractmethod
    def capabilities(self) -> S2STCapabilities: ...

    async def initialize(self, config: dict[str, object] | None = None) -> None:
        return None

    @abc.abstractmethod
    async def translate_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        source_language: str,
        target_language: str,
    ) -> AsyncIterator[SpeechTranslationResult]: ...

    @abc.abstractmethod
    async def translate_file(
        self,
        audio_path: str,
        source_language: str,
        target_language: str,
    ) -> SpeechTranslationResult: ...

    async def shutdown(self) -> None:
        return None
