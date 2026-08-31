"""IndicTrans2 translation (AI4Bharat).

VoiceBridge's default provider for Indic targets. Verified licensing: the
repository code is MIT, and the README's artifact table lists **model
checkpoints as MIT** too -- unusually permissive for a model of this quality,
and the reason it is preferred over NLLB for Indic pairs.

Covers the 22 scheduled Indian languages. It does **not** support Japanese or
Korean; those go to OPUS-MT. Nothing in the pipeline assumes otherwise -- the
provider's capabilities are the single source of truth.

API (read from ``huggingface_interface/example.py`` in the IndicTrans2 repo)
---------------------------------------------------------------------------
Checkpoints load with ``trust_remote_code=True`` because the architecture ships
with the model. Text must be passed through ``IndicProcessor`` from
``IndicTransToolkit`` -- ``preprocess_batch(batch, src_lang=, tgt_lang=)`` before
tokenisation and ``postprocess_batch(decoded, lang=)`` after. Skipping the
processor produces badly degraded output; it handles script normalisation and
entity placeholder restoration, not just formatting. Language codes are
FLORES-style (``eng_Latn``, ``hin_Deva``), not ISO-639-1.
"""

from __future__ import annotations

import logging
from typing import Any

from voicebridge.core.types import TranslationResult, now
from voicebridge.providers.base import (
    ProviderUnavailable,
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

#: ISO-639-1 -> IndicTrans2 (FLORES-200 style) code.
FLORES_CODES: dict[str, str] = {
    "en": "eng_Latn",
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "brx": "brx_Deva",
    "doi": "doi_Deva",
    "gom": "gom_Deva",
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ks": "kas_Arab",
    "mai": "mai_Deva",
    "ml": "mal_Mlym",
    "mni": "mni_Mtei",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "sa": "san_Deva",
    "sat": "sat_Olck",
    "sd": "snd_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
}

INDIC_LANGUAGES = [code for code in FLORES_CODES if code != "en"]

CHECKPOINTS = {
    "en-indic": "ai4bharat/indictrans2-en-indic-dist-200M",
    "indic-en": "ai4bharat/indictrans2-indic-en-dist-200M",
    "indic-indic": "ai4bharat/indictrans2-indic-indic-dist-320M",
}

#: Larger, higher-quality variants selectable via the ``quality`` profile.
CHECKPOINTS_LARGE = {
    "en-indic": "ai4bharat/indictrans2-en-indic-1B",
    "indic-en": "ai4bharat/indictrans2-indic-en-1B",
    "indic-indic": "ai4bharat/indictrans2-indic-indic-1B",
}


class IndicTrans2TranslationEngine(TranslationEngine):
    name = "indictrans2"

    def __init__(
        self,
        device: str | None = None,
        large: bool = False,
        num_beams: int = 5,
        max_length: int = 256,
        **_: object,
    ):
        self.device = select_device(device)
        self.checkpoints = CHECKPOINTS_LARGE if large else CHECKPOINTS
        self.num_beams = num_beams
        self.max_length = max_length
        self._cache = ModelCache(max_models=2)
        self._processor: Any = None

    @property
    def capabilities(self) -> TranslationCapabilities:
        langs = ["en"] + INDIC_LANGUAGES
        return TranslationCapabilities(
            name=self.name,
            source_languages=langs,
            target_languages=langs,
            supports_context=False,
            supports_glossary=True,
            model_license="MIT",
            commercial_use=True,
        )

    def supports(self, source_language: str, target_language: str) -> bool:
        if source_language not in FLORES_CODES or target_language not in FLORES_CODES:
            return False
        return source_language != target_language

    def _direction(self, source_language: str, target_language: str) -> str:
        if source_language == "en":
            return "en-indic"
        if target_language == "en":
            return "indic-en"
        return "indic-indic"

    def _get_processor(self) -> Any:
        if self._processor is not None:
            return self._processor
        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError as exc:
            raise ProviderUnavailable(
                "IndicTrans2 needs IndicTransToolkit for pre/post-processing.\n"
                "    pip install 'voicebridge[translation-indic]'\n"
                "See https://github.com/VarunGumma/IndicTransToolkit"
            ) from exc
        self._processor = IndicProcessor(inference=True)
        return self._processor

    async def _load(self, direction: str):
        model_id = self.checkpoints[direction]

        def loader():
            transformers, torch = require_transformers()
            from voicebridge.providers.translation._hf import LoadedModel

            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                model_id, trust_remote_code=True
            )
            model.eval()
            model.to(self.device)
            logger.info("loaded %s on %s", model_id, self.device)
            return LoadedModel(tokenizer, model, self.device)

        return await self._cache.get_or_load(model_id, loader)

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
            raise ProviderUnavailable(
                f"IndicTrans2 does not support {source_language}->{target_language}. "
                "It covers English and the 22 scheduled Indian languages only; "
                "use the opus_mt provider for Japanese/Korean."
            )

        src = FLORES_CODES[source_language]
        tgt = FLORES_CODES[target_language]
        processor = self._get_processor()
        loaded = await self._load(self._direction(source_language, target_language))

        import asyncio

        async with loaded.lock:
            translated = await asyncio.to_thread(
                self._translate_sync, loaded, processor, text, src, tgt
            )

        return TranslationResult(
            source_text=source_text,
            translated_text=translated.strip(),
            source_language=source_language,
            target_language=target_language,
            timestamp=now(),
            provider=self.name,
        )

    def _translate_sync(self, loaded, processor, text: str, src: str, tgt: str) -> str:
        import torch

        batch = processor.preprocess_batch([text], src_lang=src, tgt_lang=tgt)
        inputs = loaded.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(loaded.device)
        with torch.inference_mode():
            tokens = loaded.model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=self.max_length,
                num_beams=self.num_beams,
                num_return_sequences=1,
            )
        decoded = loaded.tokenizer.batch_decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return processor.postprocess_batch(decoded, lang=tgt)[0]


translation_registry.register("indictrans2", IndicTrans2TranslationEngine)
