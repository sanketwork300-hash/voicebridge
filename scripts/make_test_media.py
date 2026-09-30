"""Build end-to-end test media from FLEURS speech plus synthetic non-speech.

    python scripts/make_test_media.py --lang ja_jp --count 5 --out benchmark/data/media

Produces ``<lang>_episode.wav`` / ``.mp3`` / ``.mp4`` and ``<lang>_episode.json``
listing where speech, music and noise were placed (ground truth for the speech
gate). Speech is real read speech from Google FLEURS (CC-BY-4.0); the music bed
and the noise burst are synthesised here, so no third-party audio is bundled.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 16000


def read_wav(path: Path) -> np.ndarray:
    import soundfile as sf  # FLEURS WAVs are 32-bit float

    audio, rate = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SR:
        n = int(len(audio) * SR / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def music(seconds: float, seed: int = 0) -> np.ndarray:
    """A chord progression with a kick drum: unmistakably music, no speech."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    out = np.zeros_like(t)
    chords = [[261.6, 329.6, 392.0], [220.0, 261.6, 329.6], [174.6, 220.0, 261.6],
              [196.0, 246.9, 293.7]]
    beat = 0.5
    for i in range(int(seconds / beat)):
        a, b = int(i * beat * SR), int((i + 1) * beat * SR)
        env = np.exp(-np.linspace(0, 3, b - a))
        for f in chords[(i // 4) % 4]:
            for h, amp in ((1, 1.0), (2, 0.4), (3, 0.2)):
                out[a:b] += 0.08 * amp * env * np.sin(2 * np.pi * f * h * t[a:b])
        kick = np.sin(2 * np.pi * 60 * t[: int(0.12 * SR)]) * np.exp(-np.linspace(0, 8, int(0.12 * SR)))
        out[a:a + len(kick)] += 0.5 * kick
    return (out + 0.002 * rng.standard_normal(len(out))).astype(np.float32)


def burst(seconds: float, seed: int = 1) -> np.ndarray:
    """Explosion-like broadband burst with a fast attack and long decay."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    noise = rng.standard_normal(n)
    noise = np.convolve(noise, np.ones(8) / 8, mode="same")
    return (0.6 * noise * np.exp(-np.linspace(0, 5, n))).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleurs", default="benchmark/data/fleurs")
    ap.add_argument("--lang", default="ja_jp")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--out", default="benchmark/data/media")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--name", default="", help="output stem (default <lang>_episode)")
    ap.add_argument("--music-after", type=int, default=1, help="utterance index followed by music")
    ap.add_argument("--sfx-after", type=int, default=3)
    args = ap.parse_args()
    root = Path(args.fleurs)
    rows = list(csv.reader(open(root / f"{args.lang}.dev.tsv", encoding="utf-8"),
                           delimiter="\t", quoting=csv.QUOTE_NONE))
    rows = sorted(rows, key=lambda r: r[1])[: args.count]
    pieces, truth, t = [], [], 0.0

    def add(kind: str, audio: np.ndarray, **meta):
        nonlocal t
        pieces.append(audio)
        truth.append({"type": kind, "start": round(t, 3), "end": round(t + len(audio) / SR, 3), **meta})
        t += len(audio) / SR

    add("silence", np.zeros(int(1.0 * SR), np.float32))
    for i, row in enumerate(rows):
        add("speech", read_wav(root / args.lang / "dev" / row[1]), id=row[0], text=row[2])
        add("silence", np.zeros(int(1.2 * SR), np.float32))
        if i == args.music_after:
            add("music", music(6.0))
            add("silence", np.zeros(int(0.8 * SR), np.float32))
        if i == args.sfx_after:
            add("sfx", burst(2.0))
            add("silence", np.zeros(int(0.8 * SR), np.float32))
    audio = np.concatenate(pieces)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / (args.name or f"{args.lang.split('_')[0]}_episode")
    with wave.open(str(stem.with_suffix(".wav")), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    stem.with_suffix(".json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), "utf-8")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(stem.with_suffix(".wav")),
                    "-c:a", "libmp3lame", "-b:a", "128k", str(stem.with_suffix(".mp3"))], check=True)
    if not args.no_video:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"testsrc2=size=1280x720:rate=25:duration={t:.3f}",
                        "-i", str(stem.with_suffix(".wav")), "-c:v", "libx264", "-preset",
                        "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                        str(stem.with_suffix(".mp4"))], check=True)
    print(stem, f"{t:.1f}s", len(rows), "utterances")


if __name__ == "__main__":
    main()
