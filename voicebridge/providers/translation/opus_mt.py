"""OPUS-MT (Marian) translation.

Chosen as VoiceBridge's **default** translation provider for the Tier-1
entertainment pairs, on licence and latency grounds:

* ``Helsinki-NLP/opus-mt-ja-en`` -- Apache-2.0 (verified on the model card)
* ``Helsinki-NLP/opus-mt-ko-en`` -- Apache-2.0 (verified on the model card)

Apache-2.0 matters here. The obvious alternative, NLLB-200, is CC-BY-NC-4.0 and
therefore cannot be the default for a project people may deploy commercially --
see ``docs/license-matrix.md``. Marian models are also small (~75M parameters)
and fast on CPU, which is what makes a GPU-free self-hosted deployment viable.

Model naming
------------
OPUS-MT publishes one bilingual checkpoint per direction, named
``Helsinki-NLP/opus-mt-{src}-{tgt}``. **Not every direction exists**, and the
set changes over time, so this provider does not claim a fixed language list.
It resolves the checkpoint at load time and raises a clear
:class:`ProviderUnavailable` naming the missing model if the direction has no
published checkpoint. Do not infer support from the naming scheme.
"""

from __future__ import annotations

import logging

from voicebridge.core.types import TranslationResult, now
from voicebridge.providers.base import (
    ProviderUnavailable,
    TranslationCapabilities,
    TranslationEngine,
)
from voicebridge.providers.registry import translation_registry
from voicebridge.providers.translation._hf import (
    ModelCache,
    generate,
    require_transformers,
    select_device,
)

logger = logging.getLogger(__name__)

#: Directions whose checkpoints were verified to exist and be Apache-2.0 at the
#: time of writing. Others may work; these are the ones we advertise.
VERIFIED_PAIRS = [("ja", "en"), ("ko", "en")]

#: A few directions use a non-obvious language token in the checkpoint name.
_MODEL_OVERRIDES: dict[tuple, str] = {}


class OpusMTTranslationEngine(TranslationEngine):
    name = "opus_mt"

    def __init__(
        self,
        device: str | None = None,
        max_new_tokens: int = 256,
        num_beams: int = 2,
        model_cache_size: int = 4,
        preload_pairs: list | None = None,
        **_: object,
    ):
        self.device = select_device(device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self._cache = ModelCache(max_models=model_cache_size)
        #: Directions to load at warmup, e.g. ``[["ja", "en"], ["en", "hi"]]``.
        #: A checkpoint is ~300 MB; downloading it inside the first session
        #: delays that session's first translation by minutes on a cold cache.
        self.preload_pairs = [tuple(p) for p in (preload_pairs or [])]

    async def warmup(self) -> None:
        for source, target in self.preload_pairs:
            await self._load(source, target)

    @property
    def capabilities(self) -> TranslationCapabilities:
        return TranslationCapabilities(
            name=self.name,
            pairs=VERIFIED_PAIRS,
            supports_context=False,  # Marian has no context conditioning
            supports_glossary=True,
            model_license="Apache-2.0",
            commercial_use=True,
        )

    def supports(self, source_language: str, target_language: str) -> bool:
        # Permissive: attempt any direction, fail loudly at load if absent.
        return bool(source_language) and bool(target_language) and source_language != target_language

    def model_name(self, source_language: str, target_language: str) -> str:
        override = _MODEL_OVERRIDES.get((source_language, target_language))
        if override:
            return override
        return f"Helsinki-NLP/opus-mt-{source_language}-{target_language}"

    async def _load(self, source_language: str, target_language: str):
        model_id = self.model_name(source_language, target_language)

        def loader():
            transformers, torch = require_transformers()
            from voicebridge.providers.translation._hf import LoadedModel

            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
                model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_id)
            except Exception as exc:
                raise ProviderUnavailable(
                    f"could not load OPUS-MT checkpoint {model_id!r} for "
                    f"{source_language}->{target_language}: {exc}. "
                    "OPUS-MT does not publish a checkpoint for every direction; "
                    "pick another provider for this pair."
                ) from exc
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
        loaded = await self._load(source_language, target_language)

        def encode(lm):
            batch = lm.tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
            return {k: v.to(lm.device) for k, v in batch.items()}

        def decode(lm, tokens):
            return lm.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

        translated = await generate(
            loaded,
            encode,
            decode,
            {"max_new_tokens": self.max_new_tokens, "num_beams": self.num_beams},
        )
        return TranslationResult(
            source_text=source_text,
            translated_text=translated.strip(),
            source_language=source_language,
            target_language=target_language,
            timestamp=now(),
            provider=self.name,
        )


translation_registry.register("opus_mt", OpusMTTranslationEngine)
