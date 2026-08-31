"""Deterministic mock translation.

This is **not** a translation model. It looks up the mock ASR scripts in a fixed
phrase table and otherwise returns a clearly-marked placeholder. Its job is to
make the end-to-end demo, the subtitle UI and the CI pipeline work without
downloading a model, and to give deterministic output that e2e tests can assert
on exactly.

Anything it emits for text outside the phrase table is prefixed with the target
language tag so nobody can mistake mock output for a real translation.
"""

from __future__ import annotations

import asyncio

from voicebridge.core.types import TranslationResult, now
from voicebridge.providers.base import TranslationCapabilities, TranslationEngine
from voicebridge.providers.registry import translation_registry

#: (source_language, target_language, source_text) -> translation
PHRASES: dict[tuple[str, str, str], str] = {
    ("en", "hi", "I think the main reason is that we underestimated the problem."):
        "मुझे लगता है कि मुख्य कारण यह है कि हमने समस्या को कम आंका।",
    ("en", "hi", "But once the team started measuring it, the picture changed completely."):
        "लेकिन जैसे ही टीम ने इसे मापना शुरू किया, तस्वीर पूरी तरह बदल गई।",
    ("en", "hi", "So the plan for next quarter is much simpler than before."):
        "इसलिए अगली तिमाही की योजना पहले से कहीं अधिक सरल है।",
    ("ja", "en", "今日はお集まりいただきありがとうございます。"):
        "Thank you all for coming together today.",
    ("ja", "en", "この作品の一番の魅力はキャラクターの関係性だと思います。"):
        "I think this work's greatest appeal is the relationships between the characters.",
    ("ja", "en", "五条悟の領域展開は本当に印象的でした。"):
        "Satoru Gojo's Domain Expansion was truly striking.",
    ("ko", "en", "안녕하세요 여러분 오늘도 방송에 와주셔서 감사합니다."):
        "Hello everyone, thank you for coming to the broadcast again today.",
    ("ko", "en", "이번 앨범은 정말 오래 준비한 작업이었습니다."):
        "This album was a project we prepared for a really long time.",
    ("ko", "en", "방탄소년단의 무대를 다시 보고 싶습니다."):
        "I want to see BTS's stage again.",
}


class MockTranslationEngine(TranslationEngine):
    name = "mock"

    def __init__(self, latency_seconds: float = 0.02, **_: object):
        #: Simulated compute time, so latency plumbing is exercised.
        self.latency_seconds = latency_seconds

    @property
    def capabilities(self) -> TranslationCapabilities:
        return TranslationCapabilities(
            name=self.name,
            source_languages=[],   # accepts anything
            target_languages=[],
            supports_context=True,
            supports_glossary=True,
            model_license="not-a-model",
            commercial_use=True,
        )

    def supports(self, source_language: str, target_language: str) -> bool:
        return True

    async def translate(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        context: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TranslationResult:
        if self.latency_seconds:
            await asyncio.sleep(self.latency_seconds)
        key = (source_language, target_language, source_text.strip())
        translated = PHRASES.get(key)
        if translated is None:
            translated = f"[{target_language}] {source_text.strip()}"
        return TranslationResult(
            source_text=source_text,
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            confidence=1.0 if key in PHRASES else 0.0,
            timestamp=now(),
            provider=self.name,
        )


translation_registry.register("mock", MockTranslationEngine)
