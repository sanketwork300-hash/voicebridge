"""Download freely licensed non-speech clips for speech-gate / hallucination tests.

    python scripts/fetch_nonspeech.py --out benchmark/data/nonspeech

Clips come from Wikimedia Commons; each file's licence and author are read from
the Commons API at download time and written to ``ATTRIBUTION.json``. The
clips are not committed to the repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

CLIPS = {
    "laughter": "File:72844 lonemonk approx-800-laughter-and-clapter-1.wav",
    "applause": "File:277021 sandermotions applause-2.wav",
    "crying": "File:Newborn baby crying sounds 20190831 061900.wav",
    "vocalise": "File:Female-vibrato-1.wav",
    "choir_music": "File:Joyful and Triumphant - Singing Sergeants - United States Air Force Band.mp3",
}
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "VoiceBridge-benchmark/0.1 (test fixtures)"}


def info(title: str) -> dict:
    q = urllib.parse.urlencode({"action": "query", "format": "json", "titles": title,
                                "prop": "imageinfo", "iiprop": "url|extmetadata",
                                "iiextmetadatafilter": "LicenseShortName|Artist|LicenseUrl"})
    with urllib.request.urlopen(urllib.request.Request(f"{API}?{q}", headers=UA)) as r:
        page = next(iter(json.load(r)["query"]["pages"].values()))
    ii = page["imageinfo"][0]
    meta = ii.get("extmetadata", {})
    return {"title": title, "url": ii["url"], "page": ii.get("descriptionurl"),
            "license": (meta.get("LicenseShortName") or {}).get("value"),
            "author": (meta.get("Artist") or {}).get("value")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/data/nonspeech")
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    attribution = {}
    for name, title in CLIPS.items():
        meta = info(title)
        raw = out / f"{name}.orig{Path(meta['url']).suffix}"
        req = urllib.request.Request(meta["url"], headers=UA)
        with urllib.request.urlopen(req) as r:
            raw.write_bytes(r.read())
        # 16 kHz mono excerpt from 5 s in (skips lead-in silence)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "5", "-t", str(args.seconds),
                        "-i", str(raw), "-ac", "1", "-ar", "16000", str(out / f"{name}.wav")],
                       check=True)
        raw.unlink()
        attribution[name] = meta
        print(name, meta["license"])
    (out / "ATTRIBUTION.json").write_text(json.dumps(attribution, indent=2), "utf-8")


if __name__ == "__main__":
    main()
