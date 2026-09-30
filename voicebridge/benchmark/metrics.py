"""Benchmark metrics: text accuracy, translation quality, audio, latency, resources.

* **WER / CER** after light normalisation (Unicode NFKC, case-fold, punctuation
  removed). CER is the meaningful ASR metric for Japanese and Korean, which
  have no reliable word boundaries; WER is reported for space-delimited text.
* **BLEU / chrF** via sacrebleu (the ``ja``/``ko`` tokenisers are not needed
  for English/Hindi targets). **COMET** only when ``unbabel-comet`` and its
  model are installed; otherwise recorded as ``null`` with the reason. No single
  automatic metric is treated as definitive.
* Audio: duration, peak, RMS, clipped-sample ratio, silence ratio.
* Latency percentiles are computed only from measured samples.
* Resources: a background sampler records RSS (process + children, which
  includes model workers) and CPU%, and CUDA memory when available.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", text)).strip()


def edit_distance(a: list[str], b: list[str]) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def wer(reference: str, hypothesis: str) -> float | None:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    return edit_distance(ref, hyp) / len(ref) if ref else None


def cer(reference: str, hypothesis: str) -> float | None:
    ref = list(normalize(reference).replace(" ", ""))
    hyp = list(normalize(hypothesis).replace(" ", ""))
    return edit_distance(ref, hyp) / len(ref) if ref else None


def translation_scores(hypotheses: list[str], references: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"bleu": None, "chrf": None, "comet": None}
    try:
        import sacrebleu
    except ImportError:
        out["note"] = "sacrebleu not installed"
        return out
    if hypotheses:
        out["bleu"] = round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 2)
        out["chrf"] = round(sacrebleu.corpus_chrf(hypotheses, [references]).score, 2)
    return out


def comet_scores(sources: list[str], hypotheses: list[str], references: list[str],
                 model: str = "Unbabel/wmt22-comet-da") -> float | None:
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        return None
    checkpoint = load_from_checkpoint(download_model(model))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references,
                                                              strict=True)]
    return round(float(checkpoint.predict(data, batch_size=8, gpus=0).system_score), 4)


def percentiles(samples: list[float]) -> dict[str, float | None]:
    if not samples:
        return {"p50": None, "p95": None, "mean": None, "n": 0}
    arr = np.asarray(samples, dtype=float)
    return {"p50": round(float(np.percentile(arr, 50)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "mean": round(float(arr.mean()), 3), "n": int(arr.size)}


def audio_stats(path: str | Path, silence_threshold: float = 0.01) -> dict[str, Any]:
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return {"duration": 0.0, "peak": 0.0, "rms": 0.0, "clipping": 0.0, "silence_ratio": 1.0}
    frame = max(1, int(rate * 0.02))
    n = audio.size // frame
    frames = audio[: n * frame].reshape(n, frame) if n else audio.reshape(1, -1)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    return {
        "duration": round(audio.size / rate, 3),
        "peak": round(float(np.max(np.abs(audio))), 4),
        "rms": round(float(np.sqrt(np.mean(audio**2))), 4),
        "clipping": round(float(np.mean(np.abs(audio) >= 0.999)), 6),
        "silence_ratio": round(float(np.mean(frame_rms < silence_threshold)), 4),
    }


class ResourceSampler:
    """Samples RSS/CPU of this process and its children (model workers)."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.rss: list[int] = []
        self.cpu: list[float] = []
        self.vram: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> ResourceSampler:
        try:
            import psutil
        except ImportError:
            return self
        proc = psutil.Process()

        def loop() -> None:
            procs: dict[int, Any] = {}
            while not self._stop.is_set():
                try:
                    tree = [proc] + proc.children(recursive=True)
                    rss = 0
                    cpu = 0.0
                    for p in tree:
                        p = procs.setdefault(p.pid, p)
                        rss += p.memory_info().rss
                        cpu += p.cpu_percent(None)
                    self.rss.append(rss)
                    self.cpu.append(cpu)
                    vram = _cuda_memory()
                    if vram is not None:
                        self.vram.append(vram)
                except Exception:
                    pass
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> dict[str, Any]:
        gib = 1024**3
        return {
            "peak_rss_gb": round(max(self.rss) / gib, 3) if self.rss else None,
            "mean_rss_gb": round(float(np.mean(self.rss)) / gib, 3) if self.rss else None,
            "mean_cpu_percent": round(float(np.mean(self.cpu[1:] or [0])), 1) if self.cpu else None,
            "peak_vram_gb": round(max(self.vram) / gib, 3) if self.vram else None,
            "mean_vram_gb": round(float(np.mean(self.vram)) / gib, 3) if self.vram else None,
        }


def _cuda_memory() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated())
    except ImportError:
        pass
    return None


def environment(config: dict | None = None) -> dict[str, Any]:
    """Everything needed to reproduce (or distrust) a benchmark number."""
    info: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": _cpu_name(),
        "cpu_count": _cpu_count(),
        "ram_gb": _ram_gb(),
    }
    try:
        info["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                                 stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True,
                                        stderr=subprocess.DEVNULL).strip()
        info["commit_dirty"] = bool(dirty)
    except Exception:
        info["commit"] = None
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        info.update(torch=None, cuda_available=False, device="cpu")
    for pkg in ("transformers", "faster_whisper", "ctranslate2", "whisperlivekit", "sacrebleu"):
        try:
            from importlib.metadata import version

            info[pkg] = version(pkg.replace("_", "-"))
        except Exception:
            info[pkg] = None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        first = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout
        info["ffmpeg"] = first.splitlines()[0] if first else "present"
    else:
        info["ffmpeg"] = None
    if config is not None:
        info["config"] = config
    return info


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _cpu_count() -> int | None:
    import os

    return os.cpu_count()


def _ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        return None
