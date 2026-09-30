"""Qwen3-TTS model worker (runs in the ``qwen`` worker virtualenv).

API used (verified against ``qwen-tts`` 0.1.1, ``qwen_tts/inference/qwen3_tts_model.py``):

* ``Qwen3TTSModel.from_pretrained(repo, dtype=..., device_map=...)``
* Base checkpoints (``...-Base``) only support voice cloning:
  ``create_voice_clone_prompt(ref_audio, ref_text=None, x_vector_only_mode=...)``
  and ``generate_voice_clone(text, language, voice_clone_prompt=..., ...)``.
  ``x_vector_only_mode=True`` conditions on the speaker embedding alone and
  needs no reference transcript; ICL mode (``False``) also uses the reference
  transcript and codec tokens.
* CustomVoice checkpoints use ``generate_custom_voice(text, speaker, language,
  instruct=...)`` -- ``instruct`` is where emotion/style text goes.
* Languages are passed by name ("English", "Japanese", ...), validated against
  ``get_supported_languages()``. Output is ``(list[np.ndarray], sample_rate)``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol  # noqa: E402

LANG_NAMES = {
    "en": "English", "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "de": "German",
    "fr": "French", "ru": "Russian", "pt": "Portuguese", "es": "Spanish", "it": "Italian",
}

def _cpu_bf16() -> bool:
    """bf16 halves memory and is fast on CPUs with native BF16 (AVX-512 BF16 / AMX)."""
    try:
        flags = open("/proc/cpuinfo", encoding="utf-8").read()
    except OSError:
        return False
    return "avx512_bf16" in flags or "amx_bf16" in flags


STATE: dict = {"model": None, "prompts": {}}


def _load(model: str, device: str, dtype: str, threads: int):
    if STATE["model"] is not None:
        return STATE["model"]
    import torch
    from qwen_tts import Qwen3TTSModel

    if threads:
        torch.set_num_threads(threads)
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        dtype = "bfloat16" if device.startswith("cuda") or _cpu_bf16() else "float32"
    started = time.time()
    tts = Qwen3TTSModel.from_pretrained(model, dtype=getattr(torch, dtype), device_map=device)
    STATE.update(model=tts, device=device, dtype=dtype, load_seconds=time.time() - started,
                 kind=tts.model.tts_model_type)
    return tts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    def load() -> dict:
        tts = _load(args.model, args.device, args.dtype, args.threads)
        return {"model": args.model, "device": STATE["device"], "dtype": STATE["dtype"],
                "kind": STATE["kind"], "load_seconds": round(STATE["load_seconds"], 2),
                "languages": tts.get_supported_languages(),
                "speakers": tts.get_supported_speakers()}

    def synthesize(text: str, language: str, out_path: str, ref_audio: str | None = None,
                   ref_text: str | None = None, speaker: str | None = None,
                   instruct: str | None = None, max_new_tokens: int = 2048,
                   seed: int = 1234) -> dict:
        import numpy as np
        import soundfile as sf
        import torch

        tts = _load(args.model, args.device, args.dtype, args.threads)
        torch.manual_seed(seed)
        lang = LANG_NAMES.get(language, language)
        started = time.time()
        if STATE["kind"] == "base":
            if not ref_audio:
                raise ValueError("Qwen3-TTS Base clones a voice: a reference clip is required")
            key = (ref_audio, ref_text or "")
            prompt = STATE["prompts"].get(key)
            if prompt is None:
                prompt = tts.create_voice_clone_prompt(
                    ref_audio=ref_audio, ref_text=ref_text or None,
                    x_vector_only_mode=not bool(ref_text))
                if len(STATE["prompts"]) > 64:
                    STATE["prompts"].clear()
                STATE["prompts"][key] = prompt
            wavs, sr = tts.generate_voice_clone(text=text, language=lang,
                                                voice_clone_prompt=prompt,
                                                max_new_tokens=max_new_tokens)
        else:
            wavs, sr = tts.generate_custom_voice(text=text, speaker=speaker or "Ryan",
                                                 language=lang, instruct=instruct or None,
                                                 max_new_tokens=max_new_tokens)
        wav = np.asarray(wavs[0], dtype=np.float32)
        sf.write(out_path, wav, sr, subtype="PCM_16")
        return {"path": out_path, "sample_rate": int(sr), "duration": len(wav) / sr,
                "compute_seconds": round(time.time() - started, 3)}

    _protocol.serve({"load": load, "synthesize": synthesize},
                    {"worker": "qwen_tts", "model": args.model})


if __name__ == "__main__":
    main()
