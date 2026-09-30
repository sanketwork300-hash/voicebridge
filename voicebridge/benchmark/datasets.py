"""Benchmark datasets.

Manifest format (JSON list, or ``{"version": ..., "items": [...]}``)::

    {"id": "ja_001", "audio": "samples/ja/001.wav", "source_language": "ja",
     "target_language": "en", "reference_text": "...", "reference_translation": "...",
     "category": "clean_speech"}

``audio`` is relative to the manifest. ``category`` groups results (clean
speech, music, laughter, ...); non-speech items have an empty
``reference_text`` and are used to measure hallucination rate.

FLEURS
------
Google FLEURS (CC-BY-4.0) is n-way parallel: the same FLoRes sentence is read in
every language under the same numeric id. So for a Japanese recording, the
English FLEURS transcription of the same id is a human reference translation.
:func:`build_fleurs_manifest` uses that to build ja->en, ko->en and en->hi
manifests from the downloaded dev splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

FLEURS_CODES = {"ja": "ja_jp", "ko": "ko_kr", "en": "en_us", "hi": "hi_in"}


@dataclass
class DatasetItem:
    id: str
    audio: str
    source_language: str
    target_language: str
    reference_text: str | None = None
    reference_translation: str | None = None
    category: str = "speech"


def load_manifest(path: str | Path) -> list[DatasetItem]:
    base = Path(path).resolve().parent
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("items", [])
    return [
        DatasetItem(
            id=str(row["id"]),
            audio=str((base / row["audio"]).resolve()) if row.get("audio") else "",
            source_language=row["source_language"],
            target_language=row["target_language"],
            reference_text=row.get("reference_text"),
            reference_translation=row.get("reference_translation"),
            category=row.get("category", "speech"),
        )
        for row in rows
    ]


def manifest_version(path: str | Path) -> str:
    """Content hash of the manifest, recorded with every result."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _fleurs_rows(root: Path, lang: str, split: str) -> dict[str, list[str]]:
    rows = {}
    with open(root / f"{FLEURS_CODES[lang]}.{split}.tsv", encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows.setdefault(row[0], row)  # several speakers per id; keep the first
    return rows


def build_fleurs_manifest(root: str | Path, source: str, target: str, n: int,
                          out: str | Path, split: str = "dev") -> Path:
    root = Path(root)
    src = _fleurs_rows(root, source, split)
    tgt = _fleurs_rows(root, target, split)
    shared = sorted(set(src) & set(tgt), key=int)[:n]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for sid in shared:
        audio = root / FLEURS_CODES[source] / split / src[sid][1]
        items.append(asdict(DatasetItem(
            id=f"fleurs_{source}_{sid}",
            audio=os.path.relpath(Path(audio).resolve(), out.parent.resolve()),
            source_language=source, target_language=target,
            reference_text=src[sid][2], reference_translation=tgt[sid][2],
            category="read_speech")))
    out.write_text(json.dumps({"version": f"fleurs-{split}-{source}-{target}-{n}",
                               "license": "FLEURS: CC-BY-4.0", "items": items},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build benchmark manifests")
    ap.add_argument("kind", choices=["fleurs"])
    ap.add_argument("--root", default="benchmark/data/fleurs")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(build_fleurs_manifest(args.root, args.source, args.target, args.n, args.out))


if __name__ == "__main__":
    main()
