"""Resource report: hardware, software, and (optionally) model load/inference cost.

    python -m voicebridge.benchmark.resources
    python -m voicebridge.benchmark.resources --load asr,translation,tts \\
        --config config/examples/cascade-cpu.yaml

Records CPU, RAM, GPU/VRAM, CUDA, PyTorch and FFmpeg; with ``--load`` it loads
each configured provider, measures load time and peak memory, and runs one
small inference. Values are measurements on this machine, not requirements.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np

from voicebridge.benchmark.metrics import ResourceSampler, environment
from voicebridge.config import load_config
from voicebridge.media.probe import ffmpeg_available
from voicebridge.providers.factory import ProviderFactory


async def measure(kinds: list[str], config_path: str | None) -> list[dict]:
    config = load_config(config_path)
    factory = ProviderFactory(config.providers, config.runtime)
    rows = []
    for kind in kinds:
        row: dict = {"kind": kind}
        with ResourceSampler() as res:
            try:
                provider = factory.tts() if kind == "tts" else factory.get(kind)
                t0 = time.perf_counter()
                await provider.warmup()
                row["load_seconds"] = round(time.perf_counter() - t0, 2)
                t0 = time.perf_counter()
                if kind == "asr":
                    await provider.transcribe_array(np.zeros(16000 * 5, np.float32), "en")
                elif kind == "translation":
                    await provider.translate("Hello, how are you?", "en", "hi")
                elif kind == "tts":
                    from voicebridge.core.types import TTSRequest

                    await provider.synthesize_request(TTSRequest("Hello there.", "en"))
                row["inference_seconds"] = round(time.perf_counter() - t0, 2)
                row["provider"] = getattr(provider, "name", kind)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
        row.update(res.summary())
        rows.append(row)
    await factory.unload()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default="", help="comma list: asr,translation,tts")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    report = environment()
    report["ffmpeg_available"] = ffmpeg_available()
    if args.load:
        report["models"] = asyncio.run(measure(args.load.split(","), args.config))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
