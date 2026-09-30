"""Latency and throughput instrumentation.

Design notes:

* Every stage records its own latency separately. A single end-to-end number
  cannot tell you whether you are ASR-bound, translation-bound or TTS-bound,
  and those have opposite fixes.
* Percentiles are computed from a bounded reservoir of recent samples. An
  unbounded list of every sample ever taken is a slow memory leak in a service
  meant to run for hours.
* Nothing here records transcript or audio content. Metrics carry timings and
  counts only -- see ``docs/security.md``.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

#: Stage names, matching the metric list in the requirements.
STAGES = (
    "capture_latency",
    "network_latency",
    "vad_latency",
    "asr_queue_latency",
    "asr_inference_latency",
    "asr_emit_latency",
    "translation_queue_latency",
    "translation_latency",
    "tts_queue_latency",
    "tts_first_audio_latency",
    "tts_generation_latency",
    "playback_latency",
    "subtitle_latency",
    "speech_latency",
    "end_to_end_latency",
)

_MAX_SAMPLES = 512


class Histogram:
    """Fixed-capacity latency reservoir with percentile queries."""

    def __init__(self, name: str, capacity: int = _MAX_SAMPLES):
        self.name = name
        self._samples: deque[float] = deque(maxlen=capacity)
        self._count = 0
        self._sum = 0.0

    def observe(self, seconds: float) -> None:
        if seconds < 0:
            return
        self._samples.append(seconds)
        self._count += 1
        self._sum += seconds

    @property
    def count(self) -> int:
        return self._count

    def percentile(self, p: float) -> float | None:
        if not self._samples:
            return None
        ordered = sorted(self._samples)
        if len(ordered) == 1:
            return ordered[0]
        # Nearest-rank on the retained window.
        idx = min(len(ordered) - 1, max(0, int(round(p / 100.0 * (len(ordered) - 1)))))
        return ordered[idx]

    def snapshot(self) -> dict[str, float | None]:
        return {
            "count": self._count,
            "mean": (self._sum / self._count) if self._count else None,
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "last": self._samples[-1] if self._samples else None,
        }


class SessionMetrics:
    """Per-session metric collection."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.started_at = time.time()
        self._hist: dict[str, Histogram] = {s: Histogram(s) for s in STAGES}
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._lock = threading.Lock()

    def observe(self, stage: str, seconds: float) -> None:
        with self._lock:
            if stage not in self._hist:
                self._hist[stage] = Histogram(stage)
            self._hist[stage].observe(seconds)

    def increment(self, counter: str, by: int = 1) -> None:
        with self._lock:
            self._counters[counter] += by

    def set_gauge(self, gauge: str, value: float) -> None:
        with self._lock:
            self._gauges[gauge] = value

    def timer(self, stage: str) -> StageTimer:
        return StageTimer(self, stage)

    @property
    def audio_seconds_processed(self) -> float:
        return self._gauges.get("audio_seconds_processed", 0.0)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "uptime_seconds": time.time() - self.started_at,
                "latency": {
                    name: h.snapshot() for name, h in self._hist.items() if h.count
                },
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
            }

    def summary(self) -> dict[str, float | None]:
        """Compact form for the client status panel."""
        with self._lock:
            e2e = self._hist["end_to_end_latency"]
            sub = self._hist["subtitle_latency"]
            speech = self._hist["speech_latency"]
            return {
                "end_to_end_p50": e2e.percentile(50),
                "end_to_end_p95": e2e.percentile(95),
                "subtitle_latency_p50": sub.percentile(50),
                "speech_latency_p50": speech.percentile(50),
                "dropped_partials": self._counters.get("dropped_partial_results", 0),
            }


@dataclass
class StageTimer:
    metrics: SessionMetrics
    stage: str
    _t0: float = field(default_factory=time.perf_counter)

    def __enter__(self) -> StageTimer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.metrics.observe(self.stage, time.perf_counter() - self._t0)
        return None


class MetricsRegistry:
    """Process-wide aggregation, exposed at ``/metrics``."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionMetrics] = {}
        self._totals: dict[str, int] = defaultdict(int)
        self._jobs: dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def record_job(self, status: str, timings: dict[str, float] | None = None) -> None:
        """File-job outcome and per-stage seconds (exported as job histograms)."""
        with self._lock:
            self._totals[f"jobs_{status}"] += 1
            for stage, seconds in (timings or {}).items():
                if seconds is None:
                    continue
                hist = self._jobs.setdefault(stage, Histogram(stage))
                hist.observe(float(seconds))

    def session(self, session_id: str) -> SessionMetrics:
        with self._lock:
            m = self._sessions.get(session_id)
            if m is None:
                m = SessionMetrics(session_id)
                self._sessions[session_id] = m
                self._totals["sessions_started"] += 1
            return m

    def release(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._totals["sessions_ended"] += 1

    @property
    def sessions_active(self) -> int:
        with self._lock:
            return len(self._sessions)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            sessions = list(self._sessions.values())
            totals = dict(self._totals)
        return {
            "sessions_active": len(sessions),
            "totals": totals,
            "sessions": [s.snapshot() for s in sessions],
        }

    def prometheus(self) -> str:
        """Render an OpenMetrics-compatible text exposition."""
        snap = self.snapshot()
        lines: list[str] = [
            "# HELP voicebridge_sessions_active Number of active translation sessions.",
            "# TYPE voicebridge_sessions_active gauge",
            f"voicebridge_sessions_active {snap['sessions_active']}",
        ]
        totals = snap["totals"]  # type: ignore[index]
        for key, value in totals.items():  # type: ignore[union-attr]
            lines.append(f"# TYPE voicebridge_{key} counter")
            lines.append(f"voicebridge_{key} {value}")

        agg: dict[str, list[float]] = defaultdict(list)
        counters: dict[str, int] = defaultdict(int)
        for s in snap["sessions"]:  # type: ignore[index]
            for stage, values in s["latency"].items():  # type: ignore[index]
                if values.get("p50") is not None:
                    agg[stage].append(values["p50"])
            for name, count in s["counters"].items():  # type: ignore[index]
                counters[name] += count
        for stage, values in agg.items():
            ordered = sorted(values)
            lines.append(f"# TYPE voicebridge_{stage}_seconds gauge")
            lines.append(f"voicebridge_{stage}_seconds{{quantile=\"0.5\"}} {ordered[len(ordered)//2]:.6f}")
        for name, value in counters.items():
            lines.append(f"# TYPE voicebridge_{name} counter")
            lines.append(f"voicebridge_{name} {value}")
        with self._lock:
            jobs = {k: v.snapshot() for k, v in self._jobs.items()}
        for stage, snap in jobs.items():
            name = f"voicebridge_job_{stage}"
            lines.append(f"# TYPE {name} summary")
            for q, key in (("0.5", "p50"), ("0.95", "p95")):
                if snap.get(key) is not None:
                    lines.append(f'{name}{{quantile="{q}"}} {snap[key]:.6f}')
        return "\n".join(lines) + "\n"


registry = MetricsRegistry()
"""Process-wide registry. Imported by the gateway; injected in tests."""
