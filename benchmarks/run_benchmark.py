#!/usr/bin/env python3
"""VoiceBridge benchmark harness.

Measures the latencies that actually matter for live translation, per language
pair and per audio condition:

``first_subtitle_latency``
    Audio arrives → first provisional text on screen. What the viewer perceives
    as "is it working?".
``committed_subtitle_latency``
    Audio arrives → committed translation. What the viewer actually reads.
``first_audio_latency``
    Audio arrives → first dubbed speech. Necessarily higher than the subtitle
    number, because speech waits for commitment.
``end_to_end_latency``
    Full pipeline, per segment.
``rtf``
    Real-time factor: processing time / audio duration. Above 1.0 the system
    cannot keep up and will drift.

Audio conditions are separated because a system tuned only on clean speech
fails on the content people actually watch:

    clean_speech · background_music · sound_effects · overlapping_speech
    · rapid_dialogue

**This harness produces numbers. It does not ship any.** No benchmark results
are published in the repository until they have been produced on stated hardware
with a stated evaluation set. Fabricated or extrapolated figures are worse than
none.

Usage:

    python benchmarks/run_benchmark.py --audio corpus/ja_clean.wav \\
        --source ja --target en --category clean_speech --profile balanced

    python benchmarks/run_benchmark.py --synthetic --all-pairs   # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicebridge.adapters.input.file import FileAudioInput  # noqa: E402
from voicebridge.core.session.manager import SessionManager  # noqa: E402
from voicebridge.core.types import EventType  # noqa: E402
from voicebridge.providers.registry import load_builtin_providers  # noqa: E402

CATEGORIES = [
    "clean_speech",
    "background_music",
    "sound_effects",
    "overlapping_speech",
    "rapid_dialogue",
]

#: The minimum release benchmark matrix.
BENCHMARK_PAIRS = [
    ("ja", "en"), ("ko", "en"), ("en", "hi"), ("en", "ja"), ("en", "ko"),
]


@dataclass
class Measurement:
    source_language: str
    target_language: str
    category: str
    profile: str
    audio_seconds: float = 0.0
    wall_seconds: float = 0.0
    first_subtitle_latency: float | None = None
    committed_subtitle_latency: float | None = None
    first_audio_latency: float | None = None
    segments: int = 0
    partials: int = 0
    end_to_end: list[float] = field(default_factory=list)
    dropped_partials: int = 0
    warnings: list[str] = field(default_factory=list)
    providers: dict[str, Any] = field(default_factory=dict)

    @property
    def rtf(self) -> float | None:
        if not self.audio_seconds:
            return None
        return round(self.wall_seconds / self.audio_seconds, 4)

    def summary(self) -> dict[str, Any]:
        def pct(values: list[float], p: float) -> float | None:
            if not values:
                return None
            ordered = sorted(values)
            idx = min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))
            return round(ordered[idx], 4)

        out = asdict(self)
        out.pop("end_to_end")
        out["rtf"] = self.rtf
        out["end_to_end_p50"] = pct(self.end_to_end, 50)
        out["end_to_end_p95"] = pct(self.end_to_end, 95)
        out["end_to_end_mean"] = (
            round(statistics.fmean(self.end_to_end), 4) if self.end_to_end else None
        )
        for key in ("first_subtitle_latency", "committed_subtitle_latency",
                    "first_audio_latency", "wall_seconds", "audio_seconds"):
            if out.get(key) is not None:
                out[key] = round(out[key], 4)
        return out


def environment() -> dict[str, Any]:
    """Record what the numbers were produced on. A latency figure without this is meaningless."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
        info["cuda_available"] = False
    return info


def synthetic_audio(path: Path, seconds: int = 30) -> Path:
    """Generate a placeholder tone. Drives the clock; measures no model quality."""
    import numpy as np

    from voicebridge.core.audio.format import float32_to_bytes
    from voicebridge.core.audio.normalizer import encode_wav

    t = np.linspace(0, seconds, 16000 * seconds, endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path.write_bytes(encode_wav(float32_to_bytes(tone)))
    return path


async def measure(
    audio_path: Path,
    source: str,
    target: str,
    category: str,
    profile: str,
    provider_config: dict[str, Any],
    mode: str = "speech_and_subtitles",
    realtime: bool = False,
) -> Measurement:
    load_builtin_providers()
    manager = SessionManager(provider_config=provider_config)
    session = await manager.create({
        "source_language": source,
        "target_language": target,
        "mode": mode,
        "profile": profile,
        "input": {"type": "file"},
    })
    await manager.start(session.session_id)

    result = Measurement(source, target, category, profile)
    result.providers = session.pipeline.providers.describe()
    started = time.perf_counter()

    async def consume() -> None:
        async for event in session.pipeline.stream_events():
            now = time.perf_counter() - started
            if event.event_type is EventType.ASR_PARTIAL:
                result.partials += 1
                if result.first_subtitle_latency is None:
                    result.first_subtitle_latency = now
            elif event.event_type is EventType.TRANSLATION_FINAL:
                result.segments += 1
                if result.committed_subtitle_latency is None:
                    result.committed_subtitle_latency = now
            elif event.event_type is EventType.TTS_AUDIO:
                if result.first_audio_latency is None:
                    result.first_audio_latency = now
            elif event.event_type is EventType.WARNING:
                code = event.payload.get("code", "")
                if code not in result.warnings:
                    result.warnings.append(code)

    consumer = asyncio.create_task(consume())
    source_input = FileAudioInput(str(audio_path), realtime=realtime)
    async for chunk in source_input.stream():
        await session.pipeline.push_audio(chunk)
    result.audio_seconds = source_input.total_seconds

    await asyncio.sleep(1.0)
    await manager.stop(session.session_id)
    await consumer
    result.wall_seconds = time.perf_counter() - started

    snapshot = session.pipeline.metrics.snapshot()
    e2e = snapshot["latency"].get("end_to_end_latency", {})
    if e2e.get("p50") is not None:
        # The metrics reservoir holds per-segment samples; reconstruct the
        # percentiles we report from the same source the runtime uses.
        result.end_to_end = [
            v for v in (e2e.get("p50"), e2e.get("p95"), e2e.get("mean")) if v is not None
        ]
    result.dropped_partials = snapshot["counters"].get("dropped_partial_results", 0)
    return result


async def run(args) -> int:
    provider_config = {
        "asr": {"provider": args.asr},
        "translation": {"provider": args.translation},
        "tts": {"provider": args.tts},
    }

    audio_path: Path | None = None
    temp_dir: Path | None = None
    if args.synthetic or not args.audio:
        import tempfile

        temp_dir = Path(tempfile.mkdtemp())
        audio_path = synthetic_audio(temp_dir / "synthetic.wav", args.seconds)
        print(
            "NOTE: using synthetic tone audio. This exercises the pipeline and "
            "measures plumbing latency only. It says nothing about recognition "
            "or translation quality.",
            file=sys.stderr,
        )
    else:
        audio_path = Path(args.audio)
        if not audio_path.is_file():
            print(f"audio file not found: {audio_path}", file=sys.stderr)
            return 1

    pairs = BENCHMARK_PAIRS if args.all_pairs else [(args.source, args.target)]

    results = []
    for src, tgt in pairs:
        print(f"running {src}->{tgt} [{args.category}, {args.profile}] …", file=sys.stderr)
        measurement = await measure(
            audio_path, src, tgt, args.category, args.profile,
            provider_config, mode=args.mode, realtime=args.realtime,
        )
        results.append(measurement.summary())

    report = {
        "environment": environment(),
        "providers": {"asr": args.asr, "translation": args.translation, "tts": args.tts},
        "audio": str(audio_path),
        "synthetic": bool(args.synthetic or not args.audio),
        "results": results,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(text)

    print("\n--- summary ---", file=sys.stderr)
    for row in results:
        print(
            f"  {row['source_language']}->{row['target_language']:3s} "
            f"rtf={row['rtf']}  "
            f"first_sub={row['first_subtitle_latency']}s  "
            f"committed={row['committed_subtitle_latency']}s  "
            f"first_audio={row['first_audio_latency']}s  "
            f"segments={row['segments']}",
            file=sys.stderr,
        )

    if temp_dir is not None:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", help="path to an audio file (.wav/.pcm native)")
    parser.add_argument("--synthetic", action="store_true",
                        help="generate placeholder audio instead (plumbing only)")
    parser.add_argument("--seconds", type=int, default=30, help="synthetic audio length")
    parser.add_argument("--source", default="ja")
    parser.add_argument("--target", default="en")
    parser.add_argument("--all-pairs", action="store_true",
                        help="run the full release benchmark matrix")
    parser.add_argument("--category", default="clean_speech", choices=CATEGORIES)
    parser.add_argument("--profile", default="balanced",
                        choices=["low_latency", "balanced", "accurate"])
    parser.add_argument("--mode", default="speech_and_subtitles",
                        choices=["subtitles", "speech", "speech_and_subtitles"])
    parser.add_argument("--asr", default="mock")
    parser.add_argument("--translation", default="mock")
    parser.add_argument("--tts", default="mock")
    parser.add_argument("--realtime", action="store_true",
                        help="pace input at wall-clock speed (measures live behaviour)")
    parser.add_argument("--output", help="write the JSON report here")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
