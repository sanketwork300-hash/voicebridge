"""Provider registry.

Providers register a *factory*, not an instance, so that importing the registry
never imports torch. A backend whose dependencies are missing must fail at
``create()`` with :class:`ProviderUnavailable`, not at import time -- otherwise
mock mode cannot run on a machine without ML dependencies, which is the whole
point of mock mode.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Generic, TypeVar

from voicebridge.providers.base import (
    ASREngine,
    AudioEventClassifier,
    ProviderUnavailable,
    S2STProvider,
    TranslationEngine,
    TTSEngine,
    VADProvider,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class _Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T]) -> None:
        if name in self._factories:
            logger.debug("overriding %s provider %r", self.kind, name)
        self._factories[name] = factory

    def create(self, name: str, **kwargs: object) -> T:
        try:
            factory = self._factories[name]
        except KeyError:
            raise ProviderUnavailable(
                f"unknown {self.kind} provider {name!r}; available: {sorted(self._factories)}"
            ) from None
        return factory(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories


asr_registry: _Registry[ASREngine] = _Registry("asr")
translation_registry: _Registry[TranslationEngine] = _Registry("translation")
tts_registry: _Registry[TTSEngine] = _Registry("tts")
vad_registry: _Registry[VADProvider] = _Registry("vad")
audio_event_registry: _Registry[AudioEventClassifier] = _Registry("audio_event")
s2st_registry: _Registry[S2STProvider] = _Registry("s2st")


def load_builtin_providers() -> None:
    """Register every provider shipped with VoiceBridge.

    Each import is guarded: a provider module that cannot even be imported
    (missing optional dependency at module scope) must not prevent the others
    from registering.
    """
    modules = [
        "voicebridge.providers.asr.mock",
        "voicebridge.providers.asr.whisperlivekit",
        "voicebridge.providers.asr.qwen3_asr",
        "voicebridge.providers.translation.mock",
        "voicebridge.providers.translation.opus_mt",
        "voicebridge.providers.translation.indictrans2",
        "voicebridge.providers.translation.nllb",
        "voicebridge.providers.translation.contextual",
        "voicebridge.providers.tts.mock",
        "voicebridge.providers.tts.piper",
        "voicebridge.providers.tts.qwen3",
        "voicebridge.providers.vad.energy",
        "voicebridge.providers.vad.silero",
        "voicebridge.providers.audio_events.energy",
        "voicebridge.providers.audio_events.ast",
        "voicebridge.providers.s2st.seamless_streaming",
        "voicebridge.providers.s2st.seamless",
    ]
    for module in modules:
        try:
            __import__(module)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("provider module %s failed to import: %s", module, exc)


class ProviderSet:
    """The three providers bound to one session."""

    def __init__(
        self,
        asr: ASREngine,
        translation: TranslationEngine,
        tts: TTSEngine | None,
        vad: VADProvider | None = None,
        audio_event: AudioEventClassifier | None = None,
        s2st: S2STProvider | None = None,
    ):
        self.asr = asr
        self.translation = translation
        self.tts = tts
        self.vad = vad
        self.audio_event = audio_event
        self.s2st = s2st

    def describe(self) -> dict:
        return {
            "asr": {"name": self.asr.name, **_caps(self.asr.capabilities)},
            "translation": {"name": self.translation.name, **_caps(self.translation.capabilities)},
            "tts": (
                {"name": self.tts.name, **_caps(self.tts.capabilities)} if self.tts else None
            ),
            "vad": (
                {"name": self.vad.name, **_caps(self.vad.capabilities)} if self.vad else None
            ),
            "audio_event": (
                {"name": self.audio_event.name, **_caps(self.audio_event.capabilities)}
                if self.audio_event else None
            ),
            "s2st": (
                {"name": self.s2st.name, **_caps(self.s2st.capabilities)} if self.s2st else None
            ),
        }


def _caps(caps: object) -> dict:
    from dataclasses import asdict, is_dataclass

    if is_dataclass(caps):
        d = asdict(caps)  # type: ignore[arg-type]
        d.pop("name", None)
        return d
    return {}
