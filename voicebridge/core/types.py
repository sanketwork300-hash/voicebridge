"""Shared value types for the VoiceBridge pipeline.

These types are the contract between every stage. They deliberately contain no
provider-specific fields: an ASR backend that knows about speaker turns and one
that does not must both be expressible here.

Terminology used throughout the codebase, and fixed here once:

``partial``
    Text the ASR may still retract. Safe to display (visually marked as
    provisional), never safe to synthesise into speech.
``stable``
    Text the streaming ASR policy has committed to. Safe to translate.
``committed``
    A translation produced from stable text that has been emitted downstream.
    Only committed translations are ever sent to TTS -- audio cannot be
    retracted once it has been played.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


def now() -> float:
    """Wall-clock seconds. Centralised so tests can monkeypatch one symbol."""
    return time.time()


def new_id() -> str:
    return str(uuid.uuid4())


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioFormat:
    """Describes a PCM audio layout.

    VoiceBridge's *internal* format is fixed at 16 kHz, mono, signed 16-bit
    little-endian, because that is what every supported ASR backend consumes.
    External inputs may use anything; :class:`~voicebridge.core.audio.normalizer.AudioNormalizer`
    is responsible for the conversion.
    """

    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # bytes per sample
    encoding: str = "pcm_s16le"

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * self.sample_width

    def duration_of(self, n_bytes: int) -> float:
        return n_bytes / self.bytes_per_second


INTERNAL_FORMAT = AudioFormat()
"""The single internal audio contract: PCM s16le, 16 kHz, mono."""


@dataclass
class AudioChunk:
    """A block of PCM audio with a position on the session's sample clock.

    ``start``/``end`` are seconds since the start of the session's audio, not
    wall-clock. They are what lets subtitles and dubbed speech be aligned to the
    source media even when processing runs behind real time.
    """

    data: bytes
    fmt: AudioFormat = INTERNAL_FORMAT
    start: float = 0.0
    sequence_id: int = 0
    session_id: str = ""
    created_at: float = field(default_factory=now)

    @property
    def duration(self) -> float:
        return self.fmt.duration_of(len(self.data))

    @property
    def end(self) -> float:
        return self.start + self.duration

    def __len__(self) -> int:
        return len(self.data)


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class EventType(StrEnum):
    """Every event the pipeline can emit. Mirrors ``docs/protocol.md``."""

    AUDIO_CHUNK = "AUDIO_CHUNK"
    SESSION_STARTED = "SESSION_STARTED"
    SPEECH_EVENT = "SPEECH_EVENT"
    ASR_PARTIAL = "ASR_PARTIAL"
    ASR_STABLE = "ASR_STABLE"
    ASR_FINAL = "ASR_FINAL"
    SEGMENT_CREATED = "SEGMENT_CREATED"
    LANGUAGE_DETECTED = "LANGUAGE_DETECTED"
    SPEAKER_CHANGED = "SPEAKER_CHANGED"
    TRANSLATION_STARTED = "TRANSLATION_STARTED"
    TRANSLATION_PARTIAL = "TRANSLATION_PARTIAL"
    TRANSLATION_COMPLETED = "TRANSLATION_COMPLETED"
    TRANSLATION_FINAL = "TRANSLATION_FINAL"
    TTS_STARTED = "TTS_STARTED"
    TTS_COMPLETED = "TTS_COMPLETED"
    TTS_AUDIO = "TTS_AUDIO"
    SUBTITLE_CREATED = "SUBTITLE_CREATED"
    JOB_PROGRESS = "JOB_PROGRESS"
    PIPELINE_ERROR = "PIPELINE_ERROR"
    METRIC = "METRIC"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SESSION_ENDED = "SESSION_ENDED"


class PipelineMode(StrEnum):
    REALTIME = "realtime"
    FILE = "file"


class TranslationEngineMode(StrEnum):
    CASCADE = "cascade"
    SEAMLESS_STREAMING = "seamless_streaming"
    #: Offline SeamlessM4T v2 reference backend (file mode only).
    SEAMLESS_M4T_V2 = "seamless_m4t_v2"
    BENCHMARK = "benchmark"


class TranslationMode(StrEnum):
    SUBTITLES = "subtitles"
    SPEECH = "speech"
    SPEECH_AND_SUBTITLES = "speech_and_subtitles"

    @property
    def wants_tts(self) -> bool:
        return self in (TranslationMode.SPEECH, TranslationMode.SPEECH_AND_SUBTITLES)

    @property
    def wants_subtitles(self) -> bool:
        return self in (
            TranslationMode.SUBTITLES,
            TranslationMode.SPEECH_AND_SUBTITLES,
        )


class LatencyProfile(StrEnum):
    """Latency/quality trade-off.

    ``ACCURATE`` exists in addition to ``QUALITY`` because for Japanese and
    Korean the dominant cost of low latency is *not* model size but premature
    segmentation -- see ``docs/architecture.md``.
    """

    LOW_LATENCY = "low_latency"
    BALANCED = "balanced"
    ACCURATE = "accurate"


class AudioMixMode(StrEnum):
    ORIGINAL_ONLY = "original_only"
    TRANSLATION_ONLY = "translation_only"
    MIXED = "mixed"


class NameRendering(StrEnum):
    """How proper nouns in the source script should be rendered in output."""

    TRANSLATE = "translate"          # "Satoru Gojo"
    PRESERVE = "preserve"            # "五条悟"
    PRESERVE_AND_ROMANIZE = "preserve_and_romanize"  # "五条悟 (Satoru Gojo)"


class HonorificPolicy(StrEnum):
    NATURAL_ENGLISH = "natural_english"      # "Mr. Tanaka"
    PRESERVE_HONORIFICS = "preserve_honorifics"  # "Tanaka-san"
    REMOVE = "remove"                        # "Tanaka"

    @classmethod
    def _missing_(cls, value: object):
        # Short names used in the YAML config (``honorific_policy.mode``).
        aliases = {"preserve": "preserve_honorifics", "naturalize": "natural_english",
                   "naturalise": "natural_english"}
        name = aliases.get(str(value).lower())
        return cls(name) if name else None


class SessionStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ENDED = "ended"
    FAILED = "failed"


class AudioEventType(StrEnum):
    DIALOGUE = "DIALOGUE"
    VOCAL_NON_SPEECH = "VOCAL_NON_SPEECH"
    MUSIC = "MUSIC"
    SFX = "SFX"
    BACKGROUND = "BACKGROUND"
    SILENCE = "SILENCE"
    UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------
# Transcript / translation
# --------------------------------------------------------------------------


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float | None = None


@dataclass
class TranscriptSegment:
    """A stable span of recognised source text."""

    text: str
    start: float
    end: float
    language: str | None = None
    speaker: int | None = None
    words: list[Word] = field(default_factory=list)
    confidence: float | None = None
    sequence_id: int = 0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class TranscriptState:
    """The stabiliser's view of the transcript at one instant.

    ``committed_text`` is everything already handed to the segmenter;
    ``stable_text`` is committed-and-not-yet-flushed plus newly stable text;
    ``partial_text`` is the retractable tail.
    """

    partial_text: str = ""
    stable_text: str = ""
    committed_text: str = ""
    revision_id: int = 0
    words: list[Word] = field(default_factory=list)
    confidence: float | None = None
    language: str | None = None
    updated_at: float = field(default_factory=now)


@dataclass
class TranslationResult:
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float | None = None
    timestamp: float = field(default_factory=now)
    sequence_id: int = 0
    is_partial: bool = False
    source_start: float = 0.0
    source_end: float = 0.0
    provider: str = ""
    speaker: int | None = None


@dataclass
class SynthesisedAudio:
    """Output of a TTS provider."""

    audio: bytes
    sample_rate: int
    channels: int
    duration: float
    sequence_id: int
    text: str = ""
    language: str = ""
    provider: str = ""
    source_start: float = 0.0
    source_end: float = 0.0


@dataclass
class TTSRequest:
    text: str
    language: str
    speaker: str | None = None
    emotion: str | None = None
    speaking_rate: float = 1.0
    pitch: float | None = None
    style: str | None = None
    sequence_id: int = 0
    source_start: float = 0.0
    source_end: float = 0.0
    #: Path to a WAV clip of the voice to reproduce (voice-cloning engines):
    #: a configured reference voice, or the source speaker's own audio when
    #: the voice mode is ``source``.
    reference_audio: str | None = None
    #: Transcript of ``reference_audio`` when known (enables ICL cloning).
    reference_text: str | None = None


@dataclass
class SpeechDecision:
    should_transcribe: bool
    event_type: str
    confidence: float
    start_time: float
    end_time: float
    reason: str | None = None
    #: Per-category scores when a classifier produced them (multi-label).
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class TranslationSegment:
    id: str
    start_time: float
    end_time: float
    source_text: str | None = None
    translated_text: str | None = None
    source_language: str | None = None
    target_language: str | None = None
    speaker_id: str | int | None = None
    confidence: float | None = None
    event_type: str | None = None
    audio_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeechTranslationResult:
    audio_path: str | None
    source_language: str
    target_language: str
    segments: list[TranslationSegment] = field(default_factory=list)
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


@dataclass
class Event:
    """A pipeline event.

    Every event carries ``session_id``, ``sequence_id``, ``timestamp`` and
    ``event_type`` as required by the protocol, so a client can order and
    de-duplicate without relying on arrival order.
    """

    event_type: EventType
    session_id: str = ""
    sequence_id: int = 0
    timestamp: float = field(default_factory=now)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "sequence_id": self.sequence_id,
            "timestamp": self.timestamp,
            **self.payload,
        }
