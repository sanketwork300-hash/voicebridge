"""A/B: VoiceBridge cascade vs Meta SeamlessStreaming on the same audio.

    python -m voicebridge.benchmark.compare --input ./samples/test.wav --source ja --target en
    python -m voicebridge.benchmark.compare --dataset ./benchmark/datasets/ja_en.json

Writes ``results/cascade/``, ``results/seamless/`` (the engines' output audio
and subtitles), ``comparison.json`` and ``comparison.md``. Both engines get
byte-identical input; the report lists measured values side by side and does
not declare a winner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path

from voicebridge.benchmark.datasets import DatasetItem, load_manifest, manifest_version
from voicebridge.benchmark.metrics import environment
from voicebridge.benchmark.reports import render_markdown
from voicebridge.benchmark.runner import bench_engines, bench_realtime
from voicebridge.config import load_config
from voicebridge.providers.factory import ProviderFactory


async def compare(items: list[DatasetItem], out: Path, config_path: str | None,
                  engines: list[str], realtime: bool) -> dict:
    config = load_config(config_path)
    factory = ProviderFactory(config.providers, config.runtime)
    mapping = {e: e for e in engines}
    rows = await bench_engines(items, mapping, factory, out / "work")
    if realtime:
        rows += await bench_realtime(items, mapping, factory)
    await factory.unload()
    store = out / "work" / "store" / "outputs"
    for engine in engines:
        dest = out / ("seamless" if engine.startswith("seamless") and len(engines) == 2
                      else engine)
        dest.mkdir(parents=True, exist_ok=True)
        for job_dir in store.glob(f"{engine}-*"):
            for f in job_dir.iterdir():
                shutil.copy2(f, dest / f"{job_dir.name}.{f.name}")
    return {"title": " vs ".join(engines), "environment": environment(
        {"providers": config.providers, "runtime": config.runtime}), "results": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input")
    src.add_argument("--dataset")
    ap.add_argument("--source", default="ja")
    ap.add_argument("--target", default="en")
    ap.add_argument("--config", default=None)
    ap.add_argument("--engines", default="cascade,seamless_streaming")
    ap.add_argument("--realtime", action="store_true", help="also measure live latency")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default="benchmark/results")
    args = ap.parse_args()
    if args.dataset:
        items = load_manifest(args.dataset)[: args.limit or None]
    else:
        items = [DatasetItem(Path(args.input).stem, str(Path(args.input).resolve()),
                             args.source, args.target)]
    out = Path(args.output)
    report = asyncio.run(compare(items, out, args.config,
                                 [e.strip() for e in args.engines.split(",")], args.realtime))
    report["dataset"] = args.dataset or args.input
    report["dataset_version"] = manifest_version(args.dataset) if args.dataset else None
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                    default=str), encoding="utf-8")
    (out / "comparison.md").write_text(render_markdown(report), encoding="utf-8")
    print(out / "comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
