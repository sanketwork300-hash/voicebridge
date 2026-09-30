"""Benchmark CLI.

    python -m voicebridge.benchmark --dataset benchmark/datasets/ja_en.json \\
        --components asr,translation,tts \\
        --asr whisperlivekit,qwen3_asr \\
        --translation opus_mt,contextual:Qwen/Qwen3-1.7B,contextual:Qwen/Qwen3-4B-Instruct-2507 \\
        --tts piper,qwen3 --config config/examples/cascade-cpu.yaml

    python -m voicebridge.benchmark --input ./benchmark/audio --source ja --target en

``--input`` accepts a directory of audio files without references (latency,
RTF and resources are still measured; accuracy columns stay empty).
Provider specs are ``name`` or ``name:model``. Nothing is ranked.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from voicebridge.benchmark.datasets import DatasetItem, load_manifest, manifest_version
from voicebridge.benchmark.metrics import environment
from voicebridge.benchmark.reports import write_reports
from voicebridge.benchmark.runner import (
    bench_asr,
    bench_engines,
    bench_realtime,
    bench_translation,
    bench_tts,
)
from voicebridge.config import load_config
from voicebridge.providers.factory import ProviderFactory

AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def _spec(factory: ProviderFactory, kind: str, spec: str):
    name, _, model = spec.partition(":")
    overrides = {"model": model} if model else {}
    if kind == "translation" and name in ("contextual", "llm"):
        overrides.setdefault("backend", "transformers")
    return spec, factory.get(kind, name, **overrides)


def _items(args) -> list[DatasetItem]:
    if args.dataset:
        return load_manifest(args.dataset)[: args.limit or None]
    files = sorted(p for p in Path(args.input).rglob("*") if p.suffix.lower() in AUDIO)
    return [DatasetItem(p.stem, str(p), args.source, args.target) for p in files][: args.limit or None]


async def main_async(args) -> dict:
    config = load_config(args.config)
    factory = ProviderFactory(config.providers, config.runtime)
    items = _items(args)
    components = [c.strip() for c in args.components.split(",") if c.strip()]
    results: list[dict] = []
    notes: list[str] = []
    if "asr" in components:
        providers = dict(_spec(factory, "asr", s) for s in args.asr.split(","))
        gate = None
        if args.gate:
            from voicebridge.core.pipeline.speech_gate import SpeechGate, SpeechGateConfig

            vad, classifier = factory.speech_gate_providers()
            gate = SpeechGate(vad, classifier, SpeechGateConfig.from_dict(
                {k: v for k, v in (config.providers.get("speech_gate") or {}).items()
                 if k in SpeechGateConfig.__dataclass_fields__}))
        for validate in ([True, False] if args.ablate_validation else [True]):
            results += await bench_asr(items, providers, force_language=not args.detect_language,
                                       gate=gate, validate=validate)
    if "translation" in components:
        providers = dict(_spec(factory, "translation", s) for s in args.translation.split(","))
        for use_context in ([True, False] if args.ablate_context else [True]):
            results += await bench_translation(items, providers, with_comet=args.comet,
                                               use_context=use_context)
        if not args.comet:
            notes.append("COMET not computed (pass --comet with unbabel-comet installed).")
    if "tts" in components:
        providers = dict(_spec(factory, "tts", s) for s in args.tts.split(","))
        texts = [(i.target_language, i.reference_translation) for i in items
                 if i.reference_translation][: args.tts_items]
        asr = factory.get("asr")
        results += await bench_tts(texts, providers, asr, reference_audio=args.tts_reference)
    engines = {e: e for e in (args.engines.split(",") if args.engines else [])}
    if "engines" in components:
        results += await bench_engines(items, engines, factory, Path(args.output) / "engines")
    if "realtime" in components:
        results += await bench_realtime(items, engines, factory)
    await factory.unload()
    return {
        "title": args.title or ",".join(components),
        "dataset": args.dataset or args.input,
        "dataset_version": manifest_version(args.dataset) if args.dataset else None,
        "environment": environment({"providers": config.providers, "runtime": config.runtime}),
        "results": results, "notes": notes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="manifest JSON")
    src.add_argument("--input", help="directory of audio files (no references)")
    ap.add_argument("--source", default="ja")
    ap.add_argument("--target", default="en")
    ap.add_argument("--config", default=None)
    ap.add_argument("--components", default="asr,translation")
    ap.add_argument("--asr", default="whisperlivekit")
    ap.add_argument("--translation", default="opus_mt,contextual")
    ap.add_argument("--tts", default="qwen3")
    ap.add_argument("--tts-items", type=int, default=5)
    ap.add_argument("--tts-reference", default=None, help="voice-clone reference WAV")
    ap.add_argument("--engines", default="", help="cascade,seamless_streaming,seamless_m4t_v2")
    ap.add_argument("--detect-language", action="store_true")
    ap.add_argument("--gate", action="store_true", help="ASR only on speech-gate regions")
    ap.add_argument("--ablate-context", action="store_true",
                    help="also run translation with no conversational context")
    ap.add_argument("--ablate-validation", action="store_true",
                    help="also report ASR without the hallucination filter")
    ap.add_argument("--comet", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--title", default="")
    ap.add_argument("--output", default="benchmark/results")
    args = ap.parse_args()
    report = asyncio.run(main_async(args))
    files = write_reports(report, args.output)
    print(json.dumps(files, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
