"""Device and dtype selection shared by every model-backed provider.

``device: auto`` resolves CUDA -> MPS -> CPU; ``dtype: auto`` resolves to
float16 on CUDA (bfloat16 when the GPU supports it), float32 on MPS and
bfloat16 on CPUs that advertise AVX-512 BF16, else float32. Providers call these
helpers instead of probing torch themselves so the policy lives in one place.

Nothing here imports torch at module import time: the base install has no torch.
"""

from __future__ import annotations

import functools
import logging
import os

logger = logging.getLogger(__name__)

GPU_WARNING = "GPU acceleration unavailable. Translation may be significantly slower."


@functools.lru_cache(maxsize=1)
def _torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def resolve_device(preference: str | None = "auto") -> str:
    pref = (preference or "auto").lower()
    if pref != "auto":
        return pref
    torch = _torch()
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


@functools.lru_cache(maxsize=1)
def cpu_supports_bf16() -> bool:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            flags = fh.read()
        return "avx512_bf16" in flags or "amx_bf16" in flags
    except OSError:
        return False


def resolve_dtype_name(preference: str | None, device: str) -> str:
    pref = (preference or "auto").lower()
    if pref != "auto":
        return pref
    if device.startswith("cuda"):
        torch = _torch()
        if torch is not None and torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"
    if device == "cpu" and cpu_supports_bf16():
        return "bfloat16"
    return "float32"


def resolve_dtype(preference: str | None, device: str):
    torch = _torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    return getattr(torch, resolve_dtype_name(preference, device))


def describe_runtime(config: dict | None = None) -> dict:
    """Snapshot recorded in job logs and benchmark reports."""
    config = config or {}
    device = resolve_device(config.get("device", "auto"))
    info = {
        "device": device,
        "dtype": resolve_dtype_name(config.get("dtype", "auto"), device),
        "cpu_count": os.cpu_count(),
        "torch": None,
        "cuda": None,
        "gpu": None,
    }
    torch = _torch()
    if torch is not None:
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    return info


def warn_if_cpu(config: dict | None = None) -> bool:
    """Log the CPU-mode warning once at startup. Returns True when on CPU."""
    if resolve_device((config or {}).get("device", "auto")) == "cpu":
        logger.warning(GPU_WARNING)
        return True
    return False
