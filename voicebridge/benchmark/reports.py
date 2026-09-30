"""Benchmark report writers: ``benchmark.json`` (everything), ``benchmark.csv``
(one flat row per model/engine) and ``benchmark.md`` (human summary).

The Markdown report shows measured values side by side and never ranks them.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _flat(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in ("per_item",):
            continue
        if isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(v, (dict, list)):
                    out[f"{key}.{k}"] = v
        elif not isinstance(value, list):
            out[key] = value
    return out


def write_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "benchmark.json", "csv": out / "benchmark.csv",
             "markdown": out / "benchmark.md"}
    paths["json"].write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                             encoding="utf-8")
    rows = [_flat(r) for r in report.get("results", [])]
    keys = sorted({k for r in rows for k in r})
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def _table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(h for h, _ in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        f = _flat(r)
        lines.append("| " + " | ".join(_fmt(f.get(k)) for _, k in cols) + " |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    env = report.get("environment", {})
    lines = [f"# VoiceBridge benchmark: {report.get('title', '')}", "",
             "Measured values only. No ranking is implied; choose by your own constraints.", "",
             "## Environment", ""]
    for key in ("timestamp", "commit", "commit_dirty", "cpu", "cpu_count", "ram_gb", "device",
                "gpu", "cuda", "torch", "transformers", "faster_whisper", "ffmpeg", "python",
                "os"):
        lines.append(f"- **{key}**: `{env.get(key)}`")
    if report.get("dataset"):
        lines += ["", f"- **dataset**: `{report['dataset']}` (version `{report.get('dataset_version')}`)"]
    results = report.get("results", [])
    sections = {
        "asr": [("Model", "model"), ("Items", "items"), ("WER", "wer"), ("CER", "cer"),
                ("Halluc. rate", "hallucination_rate"), ("P50 s", "latency.p50"),
                ("P95 s", "latency.p95"), ("RTF", "rtf"), ("Load s", "load_seconds"),
                ("Peak RSS GB", "resources.peak_rss_gb")],
        "translation": [("Model", "model"), ("Items", "items"), ("BLEU", "bleu"),
                        ("chrF", "chrf"), ("COMET", "comet"), ("P50 s", "latency.p50"),
                        ("P95 s", "latency.p95"), ("chars/s", "chars_per_second"),
                        ("Peak RSS GB", "resources.peak_rss_gb")],
        "tts": [("Model", "model"), ("Items", "items"), ("RTF", "rtf"), ("P50 s", "latency.p50"),
                ("Round-trip WER", "roundtrip_wer"), ("Round-trip CER", "roundtrip_cer"),
                ("Load s", "load_seconds"), ("Peak RSS GB", "resources.peak_rss_gb")],
        "engine": [("Engine", "engine"), ("Item", "id"), ("Status", "status"), ("BLEU", "bleu"),
                   ("chrF", "chrf"), ("Wall s", "wall_seconds"), ("RTF", "rtf"),
                   ("Duration ratio", "duration_ratio"), ("Silence ratio", "output_audio.silence_ratio"),
                   ("Peak RSS GB", "resources.peak_rss_gb")],
        "realtime": [("Engine", "engine"), ("Item", "id"), ("TTFT s", "ttft"), ("TTFA s", "ttfa"),
                     ("Seg P50 s", "segment_latency.p50"), ("Seg P95 s", "segment_latency.p95"),
                     ("Segments", "segments")],
    }
    for comp, cols in sections.items():
        rows = [r for r in results if r.get("component") == comp]
        if rows:
            lines += ["", f"## {comp}", ""] + _table(rows, cols)
    for r in results:
        if r.get("per_item") and r.get("component") == "translation":
            lines += ["", f"### Samples: {r['model']}", ""]
            for s in r["per_item"][:5]:
                lines.append(f"- src: {s['src']}\n  - hyp: {s['hyp']}\n  - ref: {s['ref']}")
    if report.get("notes"):
        lines += ["", "## Notes", ""] + [f"- {n}" for n in report["notes"]]
    return "\n".join(lines) + "\n"
