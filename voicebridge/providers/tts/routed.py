"""Language-routed TTS: a preferred engine with a per-language fallback.

Qwen3-TTS covers en/ja/ko/zh/de/fr/ru/pt/es/it but not Hindi, while the English
-> Hindi direction is a core VoiceBridge use case. Rather than making the
pipeline aware of which engine speaks which language, the session is given one
``TTSEngine`` that routes each request by language: the primary engine when it
lists the language, else the fallback (typically Piper with a Hindi voice).
Routing is announced in capabilities and logged once per language; it is never
a silent quality switch for a language the primary supports.
"""

from __future__ import annotations

import logging

from voicebridge.core.types import SynthesisedAudio, TTSRequest
from voicebridge.providers.base import TTSCapabilities, TTSEngine

logger = logging.getLogger(__name__)


class LanguageRoutedTTS(TTSEngine):
    def __init__(self, primary: TTSEngine, fallback: TTSEngine):
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}+{fallback.name}"
        self._announced: set[str] = set()

    def engine_for(self, language: str) -> TTSEngine:
        langs = self.primary.capabilities.languages
        engine = self.primary if (not langs or language in langs) else self.fallback
        if language not in self._announced:
            self._announced.add(language)
            logger.info("TTS for %s -> %s", language, engine.name)
        return engine

    @property
    def capabilities(self) -> TTSCapabilities:
        p, f = self.primary.capabilities, self.fallback.capabilities
        return TTSCapabilities(
            name=self.name,
            languages=sorted(set(p.languages) | set(f.languages)),
            voices=list(p.voices),
            sample_rate=p.sample_rate,
            model_license=f"{p.model_license} / fallback: {f.model_license}",
            commercial_use=p.commercial_use,
            speaking_rate=p.speaking_rate,
            voice_cloning=p.voice_cloning,
            style_control=p.style_control,
        )

    def capabilities_for(self, language: str) -> TTSCapabilities:
        return self.engine_for(language).capabilities

    async def warmup(self) -> None:
        await self.primary.warmup()
        await self.fallback.warmup()

    async def close(self) -> None:
        await self.primary.close()
        await self.fallback.close()

    async def synthesize(self, text: str, language: str, speaker: str | None = None,
                         speed: float = 1.0, sequence_id: int = 0) -> SynthesisedAudio:
        return await self.engine_for(language).synthesize(text, language, speaker, speed,
                                                          sequence_id)

    async def synthesize_request(self, request: TTSRequest) -> SynthesisedAudio:
        return await self.engine_for(request.language).synthesize_request(request)
