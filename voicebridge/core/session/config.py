"""Session configuration and presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from voicebridge.core.context.glossary import SessionGlossary
from voicebridge.core.types import (
    AudioMixMode,
    HonorificPolicy,
    LatencyProfile,
    NameRendering,
    TranslationEngineMode,
    TranslationMode,
)


@dataclass
class InputConfig:
    type: str = "browser_tab"
    sample_rate: int = 16000
    channels: int = 1
    encoding: str = "pcm_s16le"


@dataclass
class OutputConfig:
    audio: bool = False
    subtitles: bool = True
    #: How the original programme audio is treated on the client.
    mix: AudioMixMode = AudioMixMode.ORIGINAL_ONLY
    #: Gain applied to the original track when mixing (0.0-1.0).
    original_gain: float = 0.25
    dual_subtitles: bool = False


@dataclass
class PrivacyConfig:
    """Privacy defaults. All persistence is off unless explicitly enabled."""

    save_audio: bool = False
    save_transcripts: bool = False
    save_translation: bool = False


@dataclass
class SessionConfig:
    source_language: str = "auto"
    target_language: str = "en"
    mode: TranslationMode = TranslationMode.SUBTITLES
    profile: LatencyProfile = LatencyProfile.BALANCED
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)

    name_rendering: NameRendering = NameRendering.TRANSLATE
    honorifics: HonorificPolicy = HonorificPolicy.NATURAL_ENGLISH
    glossary: SessionGlossary = field(default_factory=SessionGlossary)

    voice: str | None = None
    speed: float = 1.0
    style: str | None = None
    translation_style: str | None = None
    context_segments: int = 4
    context_characters: int = 600
    context_tokens: int = 2048
    #: fast | balanced | high_quality -> providers.translation_presets
    translation_quality: str | None = None
    engine: TranslationEngineMode = TranslationEngineMode.CASCADE
    speech_gate_enabled: bool = True
    speech_gate_dialogue_threshold: float = 0.35
    speech_gate_unknown_threshold: float = 0.35
    speech_gate_non_speech_threshold: float = 0.60
    translation_validation: str = "balanced"
    tts_timing_enabled: bool = True
    tts_max_rate: float = 1.20
    tts_min_rate: float = 0.85
    tts_duration_tolerance: float = 0.15
    preset: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> SessionConfig:
        payload = dict(payload or {})
        preset_name = payload.get("preset")
        if preset_name:
            payload = {**PRESETS.get(preset_name, {}), **payload}

        inp = payload.get("input") or {}
        out = payload.get("output") or {}
        priv = payload.get("privacy") or {}
        speech_gate = payload.get("speech_gate") or {}
        translation = payload.get("translation") or {}
        tts = payload.get("tts") or {}
        timing = tts.get("timing") or {}

        cfg = cls(
            source_language=payload.get("source_language", "auto"),
            target_language=payload.get("target_language", "en"),
            mode=_enum(TranslationMode, payload.get("mode"), TranslationMode.SUBTITLES),
            profile=_enum(LatencyProfile, payload.get("profile"), LatencyProfile.BALANCED),
            input=InputConfig(
                type=inp.get("type", "browser_tab"),
                sample_rate=int(inp.get("sample_rate", 16000)),
                channels=int(inp.get("channels", 1)),
                encoding=inp.get("encoding", "pcm_s16le"),
            ),
            output=OutputConfig(
                audio=bool(out.get("audio", False)),
                subtitles=bool(out.get("subtitles", True)),
                mix=_enum(AudioMixMode, out.get("mix"), AudioMixMode.ORIGINAL_ONLY),
                original_gain=float(out.get("original_gain", 0.25)),
                dual_subtitles=bool(out.get("dual_subtitles", False)),
            ),
            privacy=PrivacyConfig(
                save_audio=bool(priv.get("save_audio", False)),
                save_transcripts=bool(priv.get("save_transcripts", False)),
                save_translation=bool(priv.get("save_translation", False)),
            ),
            name_rendering=_enum(
                NameRendering, payload.get("name_rendering"), NameRendering.TRANSLATE
            ),
            honorifics=_enum(
                HonorificPolicy, payload.get("honorifics"), HonorificPolicy.NATURAL_ENGLISH
            ),
            glossary=SessionGlossary.from_payload(payload.get("glossary")),
            voice=payload.get("voice"),
            speed=float(payload.get("speed", 1.0)),
            style=payload.get("style"),
            translation_style=payload.get("translation_style") or payload.get("style"),
            context_segments=int(payload.get("context_segments", 4)),
            context_characters=int(payload.get("context_characters", 600)),
            translation_quality=(payload.get("translation_quality")
                                 or translation.get("quality") or None),
            context_tokens=int(payload.get("context_tokens", translation.get("max_tokens", 2048))),
            engine=_enum(
                TranslationEngineMode,
                payload.get("engine") or payload.get("translation_engine"),
                TranslationEngineMode.CASCADE,
            ),
            speech_gate_enabled=bool(speech_gate.get("enabled", True)),
            speech_gate_dialogue_threshold=float(speech_gate.get("dialogue_threshold", 0.35)),
            speech_gate_unknown_threshold=float(speech_gate.get("unknown_threshold", 0.35)),
            speech_gate_non_speech_threshold=float(
                speech_gate.get("non_speech_threshold", 0.60)
            ),
            translation_validation=str(translation.get("validation", "balanced")),
            tts_timing_enabled=bool(timing.get("enabled", True)),
            tts_max_rate=float(timing.get("max_rate", 1.20)),
            tts_min_rate=float(timing.get("min_rate", 0.85)),
            tts_duration_tolerance=float(timing.get("duration_tolerance", 0.15)),
            preset=preset_name,
        )
        # Mode and output flags must agree; mode is authoritative.
        if cfg.mode.wants_tts:
            cfg.output.audio = True
        if cfg.mode is TranslationMode.SPEECH:
            cfg.output.subtitles = out.get("subtitles", False)
        if cfg.mode.wants_subtitles:
            cfg.output.subtitles = True
        return cfg

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        d["profile"] = self.profile.value
        d["engine"] = self.engine.value
        d["name_rendering"] = self.name_rendering.value
        d["honorifics"] = self.honorifics.value
        d["output"]["mix"] = self.output.mix.value
        d["glossary"] = {"terms": len(self.glossary)}  # never echo term contents
        return d


def _enum(enum_cls, value, default):
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        return default


#: UX presets. These configure defaults only -- the pipeline stays generic and
#: has no knowledge of which preset produced a configuration.
PRESETS: dict[str, dict[str, Any]] = {
    "anime_drama": {
        "source_language": "auto",
        "target_language": "en",
        "mode": "subtitles",
        "profile": "balanced",
        "name_rendering": "preserve_and_romanize",
        "honorifics": "preserve_honorifics",
        "context_segments": 6,
        "output": {"subtitles": True, "dual_subtitles": True, "audio": False},
    },
    "kpop_live": {
        "source_language": "ko",
        "target_language": "en",
        "mode": "subtitles",
        "profile": "low_latency",
        "name_rendering": "translate",
        "honorifics": "preserve_honorifics",
        "context_segments": 4,
        "output": {"subtitles": True, "dual_subtitles": True, "audio": False},
    },
    "conference": {
        "source_language": "en",
        "target_language": "hi",
        "mode": "speech_and_subtitles",
        "profile": "balanced",
        "output": {"subtitles": True, "audio": True, "mix": "mixed"},
    },
}
