"""Shared Hugging Face plumbing for the local translation providers.

Two rules the concrete providers rely on:

* **Import lazily.** ``transformers`` and ``torch`` are optional extras. The
  module must import cleanly on a machine that has neither, so mock mode works
  everywhere.
* **Never block the event loop.** Generation is CPU/GPU-bound and takes tens to
  hundreds of milliseconds. It runs in a worker thread, and concurrent calls per
  model are serialised with a lock because a single HF model is not safe to
  call re-entrantly from several threads.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from voicebridge.providers.base import ProviderUnavailable

logger = logging.getLogger(__name__)


def require_transformers() -> tuple[Any, Any]:
    try:
        import torch  # noqa: F401
        import transformers
    except ImportError as exc:
        raise ProviderUnavailable(
            "This translation provider needs PyTorch and transformers. Install:\n"
            "    pip install 'voicebridge[translation-local]'"
        ) from exc
    import torch

    return transformers, torch


def select_device(preferred: str | None = None) -> str:
    """Pick a compute device, honouring an explicit override."""
    if preferred:
        return preferred
    env = os.environ.get("VOICEBRIDGE_DEVICE")
    if env:
        return env
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LoadedModel:
    """One tokenizer+model pair with a lock serialising generation."""

    def __init__(self, tokenizer: Any, model: Any, device: str):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.lock = asyncio.Lock()


class ModelCache:
    """Caches models by key so repeated sessions do not reload weights."""

    def __init__(self, max_models: int = 4):
        self.max_models = max_models
        self._models: dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()

    async def get_or_load(self, key: str, loader) -> LoadedModel:
        async with self._lock:
            existing = self._models.get(key)
            if existing is not None:
                return existing
            if len(self._models) >= self.max_models:
                # Simple eviction; translation models are 300MB-2.5GB each and
                # holding every pair a long session touches will OOM a small box.
                evicted, _ = next(iter(self._models.items())), None
                self._models.pop(evicted[0] if isinstance(evicted, tuple) else evicted, None)
                logger.info("evicted cached translation model %s", evicted)
            loaded = await asyncio.to_thread(loader)
            self._models[key] = loaded
            return loaded

    def clear(self) -> None:
        self._models.clear()


async def generate(loaded: LoadedModel, encode, decode, generate_kwargs: dict[str, Any]) -> str:
    """Run a guarded, off-loop generation step."""
    async with loaded.lock:
        return await asyncio.to_thread(_generate_sync, loaded, encode, decode, generate_kwargs)


def _generate_sync(loaded: LoadedModel, encode, decode, generate_kwargs: dict[str, Any]) -> str:
    import torch

    inputs = encode(loaded)
    with torch.inference_mode():
        tokens = loaded.model.generate(**inputs, **generate_kwargs)
    return decode(loaded, tokens)
