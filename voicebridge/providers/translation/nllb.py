"""NLLB-200 translation -- opt-in only, non-commercial licence.

NLLB-200 covers 200 languages in one model, including every VoiceBridge Tier-1
and Tier-2 direction, which makes it technically attractive as a single default.
It is **not** the default, for one reason:

    ``facebook/nllb-200-distilled-600M`` is licensed **CC-BY-NC-4.0**
    (verified on the model card). Commercial use is prohibited.

Shipping it as the default would silently push that restriction onto every
VoiceBridge deployment. Instead this provider refuses to load unless the
operator explicitly acknowledges the licence, via config
(``acknowledge_non_commercial: true``) or ``VOICEBRIDGE_ALLOW_NC_MODELS=1``.
That way the restriction is a decision someone made, not a default they
inherited.

For commercial deployments use ``opus_mt`` (Apache-2.0) or ``indictrans2`` (MIT).
"""

from __future__ import annotations

import logging
import os

from voicebridge.core.types import TranslationResult, now
from voicebridge.providers.base import (
    ProviderError,
    TranslationCapabilities,
    TranslationEngine,
)
from voicebridge.providers.registry import translation_registry
from voicebridge.providers.translation._hf import (
    ModelCache,
    require_transformers,
    select_device,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"

#: ISO-639-1 -> NLLB (FLORES-200) code, for the languages VoiceBridge advertises.
NLLB_CODES: dict[str, str] = {
    "en": "eng_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans",
    "hi": "hin_Deva", "mr": "mar_Deva", "ta": "tam_Taml", "te": "tel_Telu",
    "bn": "ben_Beng", "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym",
    "pa": "pan_Guru", "ur": "urd_Arab", "es": "spa_Latn", "fr": "fra_Latn",
    "de": "deu_Latn", "pt": "por_Latn", "ru": "rus_Cyrl", "ar": "arb_Arab",
    "id": "ind_Latn", "vi": "vie_Latn", "th": "tha_Thai", "tr": "tur_Latn",
}

LICENCE_MESSAGE = (
    "NLLB-200 weights are licensed CC-BY-NC-4.0 (non-commercial use only). "
    "VoiceBridge will not load them unless you acknowledge this explicitly: set "
    "`acknowledge_non_commercial: true` on the provider config, or the "
    "environment variable VOICEBRIDGE_ALLOW_NC_MODELS=1. For commercial use "
    "choose the 'opus_mt' (Apache-2.0) or 'indictrans2' (MIT) provider instead."
)


class NLLBTranslationEngine(TranslationEngine):
    name = "nllb"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        num_beams: int = 4,
        max_new_tokens: int = 256,
        acknowledge_non_commercial: bool = False,
        **_: object,
    ):
        self.model_id = model
        self.device = select_device(device)
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.acknowledged = bool(acknowledge_non_commercial) or os.environ.get(
            "VOICEBRIDGE_ALLOW_NC_MODELS"
        ) in ("1", "true", "yes")
        self._cache = ModelCache(max_models=1)

    @property
    def capabilities(self) -> TranslationCapabilities:
        langs = sorted(NLLB_CODES)
        return TranslationCapabilities(
            name=self.name,
            source_languages=langs,
            target_languages=langs,
            supports_context=False,
            supports_glossary=True,
            model_license="CC-BY-NC-4.0",
            commercial_use=False,
        )

    def supports(self, source_language: str, target_language: str) -> bool:
        return (
            source_language in NLLB_CODES
            and target_language in NLLB_CODES
            and source_language != target_language
        )

    async def _load(self):
        if not self.acknowledged:
            raise ProviderError(LICENCE_MESSAGE)

        def loader():
            transformers, torch = require_transformers()
            from voicebridge.providers.translation._hf import LoadedModel

            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            model.eval()
            model.to(self.device)
            logger.warning(
                "loaded %s -- CC-BY-NC-4.0 weights, non-commercial use only", self.model_id
            )
            return LoadedModel(tokenizer, model, self.device)

        return await self._cache.get_or_load(self.model_id, loader)

    async def translate(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        context: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TranslationResult:
        text = (source_text or "").strip()
        if not text:
            return TranslationResult(
                source_text=source_text,
                translated_text="",
                source_language=source_language,
                target_language=target_language,
                provider=self.name,
            )
        if not self.supports(source_language, target_language):
            raise ProviderError(
                f"NLLB provider has no code mapping for {source_language}->{target_language}"
            )
        loaded = await self._load()
        src_code = NLLB_CODES[source_language]
        tgt_code = NLLB_CODES[target_language]

        import asyncio

        async with loaded.lock:
            translated = await asyncio.to_thread(
                self._translate_sync, loaded, text, src_code, tgt_code
            )
        return TranslationResult(
            source_text=source_text,
            translated_text=translated.strip(),
            source_language=source_language,
            target_language=target_language,
            timestamp=now(),
            provider=self.name,
        )

    def _translate_sync(self, loaded, text: str, src_code: str, tgt_code: str) -> str:
        import torch

        loaded.tokenizer.src_lang = src_code
        inputs = loaded.tokenizer(
            [text], return_tensors="pt", truncation=True, max_length=512
        ).to(loaded.device)
        # NLLB selects the output language with a forced BOS token; the helper
        # name differs across transformers versions, so resolve defensively.
        bos_id = _target_bos_id(loaded.tokenizer, tgt_code)
        kwargs = {"max_new_tokens": self.max_new_tokens, "num_beams": self.num_beams}
        if bos_id is not None:
            kwargs["forced_bos_token_id"] = bos_id
        with torch.inference_mode():
            tokens = loaded.model.generate(**inputs, **kwargs)
        return loaded.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]


def _target_bos_id(tokenizer, tgt_code: str):
    getter = getattr(tokenizer, "convert_tokens_to_ids", None)
    lang_map = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_map, dict) and tgt_code in lang_map:
        return lang_map[tgt_code]
    if getter is not None:
        token_id = getter(tgt_code)
        unk = getattr(tokenizer, "unk_token_id", None)
        if token_id is not None and token_id != unk:
            return token_id
    logger.warning("could not resolve NLLB target token for %s", tgt_code)
    return None


translation_registry.register("nllb", NLLBTranslationEngine)
