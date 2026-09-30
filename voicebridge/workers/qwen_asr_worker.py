"""Qwen3-ASR model worker (runs in the ``qwen`` worker virtualenv).

API used (verified against ``qwen-asr`` 0.0.6, ``qwen_asr/inference/qwen3_asr.py``):

* ``Qwen3ASRModel.from_pretrained(repo, forced_aligner=None, max_new_tokens=..., dtype=...,
  device_map=...)`` -- extra kwargs go to ``AutoModel.from_pretrained``.
* ``transcribe(audio, context="", language=None, return_time_stamps=False)``
  returns ``[ASRTranscription(language, text, time_stamps)]``. ``audio`` may be a
  path or ``(np.ndarray, sr)``; ``language`` is a name such as "Japanese".
* Timestamps require a forced aligner (``Qwen/Qwen3-ForcedAligner-0.6B``); its
  items carry ``text``, ``start_time``, ``end_time`` in seconds.

Qwen3-ASR exposes no decoder confidence (no logprob / no-speech probability),
so the ASR validation layer can only use text-level and speech-gate signals
for it -- one of the things the benchmark should surface.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol  # noqa: E402

LANG_NAMES = {"en": "English", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
              "hi": "Hindi", "de": "German", "fr": "French", "es": "Spanish",
              "ru": "Russian", "pt": "Portuguese", "it": "Italian"}
CODES = {v: k for k, v in LANG_NAMES.items()}
def _cpu_bf16() -> bool:
    """bf16 halves memory and is fast on CPUs with native BF16 (AVX-512 BF16 / AMX)."""
    try:
        flags = open("/proc/cpuinfo", encoding="utf-8").read()
    except OSError:
        return False
    return "avx512_bf16" in flags or "amx_bf16" in flags


STATE: dict = {"model": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--aligner", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    def _load():
        if STATE["model"] is None:
            import torch
            from qwen_asr import Qwen3ASRModel

            if args.threads:
                torch.set_num_threads(args.threads)
            device = args.device
            if device == "auto":
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = args.dtype if args.dtype != "auto" else (
                "bfloat16" if device.startswith("cuda") or _cpu_bf16() else "float32")
            started = time.time()
            kwargs = {"dtype": getattr(torch, dtype), "device_map": device}
            aligner_kwargs = dict(kwargs) if args.aligner else None
            STATE["model"] = Qwen3ASRModel.from_pretrained(
                args.model, forced_aligner=args.aligner or None,
                forced_aligner_kwargs=aligner_kwargs, max_new_tokens=512, **kwargs)
            STATE.update(device=device, dtype=dtype, load_seconds=time.time() - started)
        return STATE["model"]

    def load() -> dict:
        _load()
        return {"model": args.model, "device": STATE["device"], "dtype": STATE["dtype"],
                "load_seconds": round(STATE["load_seconds"], 2),
                "timestamps": bool(args.aligner)}

    def transcribe(path: str, language: str | None = None, context: str = "") -> dict:
        model = _load()
        started = time.time()
        lang = LANG_NAMES.get(language or "", None) if language else None
        out = model.transcribe(audio=path, context=context or "", language=lang,
                               return_time_stamps=bool(args.aligner))[0]
        words = []
        if out.time_stamps is not None:
            words = [{"text": it.text, "start": float(it.start_time), "end": float(it.end_time)}
                     for it in out.time_stamps]
        detected = (out.language or "").split(",")[0].strip()
        return {"text": out.text, "language": CODES.get(detected, detected.lower() or None),
                "words": words, "compute_seconds": round(time.time() - started, 3)}

    _protocol.serve({"load": load, "transcribe": transcribe},
                    {"worker": "qwen_asr", "model": args.model})


if __name__ == "__main__":
    main()
