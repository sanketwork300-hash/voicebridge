"""Chat-completion backends for the contextual translation provider.

The contextual provider builds a prompt; a backend turns ``(system, user)``
into text. Backends are selected by configuration so the pipeline never depends
on one vendor, model or runtime:

``transformers``
    A local instruction-tuned causal LM loaded with Hugging Face transformers,
    lazily, on the configured device/dtype. The default model is
    ``Qwen/Qwen3-1.7B`` (Apache-2.0) because it runs on a CPU; any chat model
    with a chat template works. Qwen3's reasoning mode is switched off through
    the chat template's ``enable_thinking=False`` flag.
``openai``
    Any OpenAI-compatible ``/v1/chat/completions`` endpoint -- vLLM,
    llama.cpp server, Ollama, LM Studio, or a hosted service. Plain HTTP via
    the standard library; the key comes from an environment variable.
``anthropic``
    Claude through the official ``anthropic`` Python SDK (optional dependency).
``mock``
    Deterministic, for tests only.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import os
import threading
import urllib.error
import urllib.request
from typing import Any

from voicebridge.providers.base import ProviderError, ProviderUnavailable
from voicebridge.runtime import resolve_device, resolve_dtype

logger = logging.getLogger(__name__)


class ChatBackend(abc.ABC):
    name = "backend"
    model = ""
    license = "depends on configured model"

    @abc.abstractmethod
    async def complete(self, system: str, user: str, max_tokens: int) -> str: ...

    async def warmup(self) -> None:
        return None

    def unload(self) -> None:
        return None


class TransformersBackend(ChatBackend):
    name = "transformers"

    def __init__(self, model: str = "Qwen/Qwen3-1.7B", device: str = "auto",
                 dtype: str = "auto", threads: int | None = None, **_: object):
        self.model = model
        self.device_pref = device
        self.dtype_pref = dtype
        self.threads = threads
        self.license = "see model card"
        self._lm: Any = None
        self._tok: Any = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._lm is not None:
            return
        with self._lock:
            if self._lm is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise ProviderUnavailable(
                    "The transformers LLM backend needs torch and transformers: "
                    "pip install 'voicebridge[translation]'"
                ) from exc
            if self.threads:
                torch.set_num_threads(int(self.threads))
            device = resolve_device(self.device_pref)
            dtype = resolve_dtype(self.dtype_pref, device)
            logger.info("loading translation LLM %s on %s (%s)", self.model, device, dtype)
            self._tok = AutoTokenizer.from_pretrained(self.model)
            lm = AutoModelForCausalLM.from_pretrained(self.model, dtype=dtype)
            lm.eval().to(device)
            self._device = device
            self._torch = torch
            self._lm = lm

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    def unload(self) -> None:
        with self._lock:
            self._lm = None
            self._tok = None
        import gc

        gc.collect()

    def _generate(self, system: str, user: str, max_tokens: int) -> str:
        self._load()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": False}
        try:
            prompt = self._tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:  # templates without the Qwen3 thinking switch
            prompt = self._tok.apply_chat_template(messages, **kwargs)
        inputs = self._tok(prompt, return_tensors="pt").to(self._device)
        with self._lock, self._torch.inference_mode():
            out = self._lm.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
        return self._tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        return await asyncio.to_thread(self._generate, system, user, max_tokens)


class OpenAICompatibleBackend(ChatBackend):
    name = "openai"

    def __init__(self, model: str = "", base_url: str = "http://127.0.0.1:8080/v1",
                 api_key_env: str = "VOICEBRIDGE_LLM_API_KEY", timeout: float = 60.0,
                 temperature: float = 0.0, **_: object):
        if not model:
            raise ProviderUnavailable("openai backend needs translation.model to be set")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = float(timeout)
        self.temperature = float(temperature)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"LLM endpoint returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderUnavailable(f"LLM endpoint unreachable: {exc}") from exc

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        data = await asyncio.to_thread(self._post, body)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("unexpected response shape from LLM endpoint") from exc


class AnthropicBackend(ChatBackend):
    """Claude via the official SDK.

    Credentials resolve the SDK's usual way (``ANTHROPIC_API_KEY``, an auth
    token, or an ``ant auth login`` profile). Server-side refusal fallbacks are
    enabled by default so a safety decline is retried on a fallback model
    inside the same call; a final ``refusal`` still surfaces as an error.
    """

    name = "anthropic"
    license = "commercial API (Anthropic terms)"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low",
                 fallbacks: bool = True, timeout: float = 60.0, **_: object):
        self.model = model
        self.effort = effort
        self.fallbacks = bool(fallbacks)
        self.timeout = float(timeout)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ProviderUnavailable(
                    "The anthropic backend needs the SDK: pip install anthropic"
                ) from exc
            self._client = anthropic.AsyncAnthropic(timeout=self.timeout)
        return self._client

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        import anthropic

        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(1024, max_tokens * 4),  # room for adaptive thinking
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.effort},
        }
        try:
            if self.fallbacks:
                response = await client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            else:
                response = await client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            raise ProviderError("Anthropic rate limit reached") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderUnavailable(f"Anthropic API unreachable: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error {exc.status_code}") from exc
        if response.stop_reason == "refusal":
            raise ProviderError("translation declined by the model (refusal)")
        return "".join(block.text for block in response.content if block.type == "text")


class MockBackend(ChatBackend):
    """Echoes the current segment from the prompt. Tests only."""

    name = "mock"
    model = "mock"

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        from voicebridge.providers.translation.mock import PHRASES

        current = user.rsplit("<current>", 1)[-1].split("</current>", 1)[0].strip()
        for (_, _, src), tgt in PHRASES.items():
            if src == current:
                return tgt
        return f"[translated] {current}"


BACKENDS: dict[str, type[ChatBackend]] = {
    "transformers": TransformersBackend,
    "local": TransformersBackend,
    "openai": OpenAICompatibleBackend,
    "openai_compatible": OpenAICompatibleBackend,
    "anthropic": AnthropicBackend,
    "mock": MockBackend,
}


def create_backend(name: str, **options: Any) -> ChatBackend:
    cls = BACKENDS.get(name)
    if cls is None:
        raise ProviderUnavailable(f"unknown LLM backend {name!r}; choose from {sorted(BACKENDS)}")
    return cls(**options)
