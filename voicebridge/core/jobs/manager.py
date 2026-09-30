"""Asynchronous file-translation jobs: queue, workers, progress, cancellation.

API server -> :class:`JobQueue` -> worker task -> executor -> pipeline.

The queue is an interface with an in-process implementation, which is right for
a single-node deployment; a Redis/Arq/Celery-backed queue can implement the
same three methods without the API or executor changing. HTTP handlers only
enqueue; nothing long-running happens inside a request.

Progress is reported per stage and converted to one overall percentage with
fixed stage weights, so the bar never jumps backwards when a stage starts. ETA
is extrapolated from elapsed time and overall progress once progress is
meaningful (>= 5%), and is ``None`` before that rather than a guess.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from voicebridge.core.jobs.errors import CancelToken, JobCancelled, PipelineError
from voicebridge.core.types import new_id

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    SPEECH_DETECTION = "speech_detection"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    SYNTHESIZING = "synthesizing"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


#: (start, end) of each stage on the overall 0..100 bar.
STAGE_SPAN: dict[str, tuple[float, float]] = {
    "queued": (0, 0),
    "preprocessing": (0, 5),
    "speech_detection": (5, 15),
    "transcribing": (15, 45),
    "translating": (45, 62),
    "synthesizing": (62, 92),
    "rendering": (92, 100),
    "completed": (100, 100),
}


@dataclass
class JobRecord:
    job_id: str
    payload: dict[str, Any]
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    stage: str = "queued"
    stage_progress: float = 0.0
    outputs: dict[str, str] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    updated_at: float = field(default_factory=time.time)
    cancel: CancelToken = field(default_factory=CancelToken, repr=False)
    log: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def eta_seconds(self) -> int | None:
        if self.status.terminal or not self.started_at or self.progress < 5:
            return None
        elapsed = time.time() - self.started_at
        return int(elapsed * (100 - self.progress) / self.progress)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": int(self.progress),
            "stage": self.stage,
            "stage_progress": round(self.stage_progress, 3),
            "eta_seconds": self.eta_seconds,
            "outputs": sorted(self.outputs),
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "input": {k: self.payload.get(k) for k in (
                "filename", "source_language", "target_language", "engine", "output_format",
                "audio_mode", "subtitle_enabled", "translation_quality", "voice")},
            "log": self.log,
            "result": self.result,
        }


class JobQueue(abc.ABC):
    @abc.abstractmethod
    async def put(self, job_id: str) -> None: ...

    @abc.abstractmethod
    async def get(self) -> str: ...

    @abc.abstractmethod
    def qsize(self) -> int: ...


class InMemoryJobQueue(JobQueue):
    def __init__(self) -> None:
        self._q: asyncio.Queue[str] = asyncio.Queue()

    async def put(self, job_id: str) -> None:
        await self._q.put(job_id)

    async def get(self) -> str:
        return await self._q.get()

    def qsize(self) -> int:
        return self._q.qsize()


class JobManager:
    def __init__(self, executor: Any, queue: JobQueue | None = None, workers: int = 1,
                 timeout_seconds: float | None = None, store: Any = None,
                 retain_jobs: int = 200):
        self.executor = executor
        self.queue = queue or InMemoryJobQueue()
        self.n_workers = workers
        self.timeout_seconds = timeout_seconds
        self.store = store
        self.retain_jobs = retain_jobs
        self._jobs: dict[str, JobRecord] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._workers: list[asyncio.Task] = []
        self._running: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        if not self._workers:
            self._workers = [asyncio.create_task(self._worker(), name=f"vb-job-worker-{i}")
                             for i in range(self.n_workers)]

    async def stop(self) -> None:
        for job in self._jobs.values():
            job.cancel.cancel()
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit(self, payload: dict[str, Any]) -> JobRecord:
        job = JobRecord(job_id=payload.get("job_id") or new_id(), payload=payload)
        self._jobs[job.job_id] = job
        self._trim()
        await self.queue.put(job.job_id)
        await self._publish(job, "job.created")
        return job

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def list(self) -> list[JobRecord]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    async def cancel(self, job_id: str) -> JobRecord | None:
        job = self.get(job_id)
        if job is None:
            return None
        if job.status.terminal:
            return job
        job.cancel.cancel()
        if job.status is JobStatus.QUEUED:
            await self._finish(job, JobStatus.CANCELLED, error=JobCancelled().to_dict())
        return job

    # -- progress ----------------------------------------------------------------

    async def progress(self, job: JobRecord, stage: str, fraction: float = 0.0) -> None:
        job.cancel.check()  # also stops work for a job cancelled before it started
        if job.status.terminal:
            return
        fraction = max(0.0, min(1.0, fraction))
        a, b = STAGE_SPAN.get(stage, (job.progress, job.progress))
        job.stage = stage
        job.stage_progress = fraction
        job.status = JobStatus(stage) if stage in JobStatus._value2member_map_ else job.status
        job.progress = max(job.progress, a + (b - a) * fraction)
        job.updated_at = time.time()
        await self._publish(job, "job.progress")

    async def pipeline_event(self, job: JobRecord, event_type: str, payload: dict[str, Any]
                             ) -> None:
        """Forward a pipeline event (same types as realtime) to job subscribers."""
        await self._broadcast(job.job_id, {"event": event_type, "job_id": job.job_id, **payload})

    async def _publish(self, job: JobRecord, event: str) -> None:
        await self._broadcast(job.job_id, {
            "event": event, "job_id": job.job_id, "status": job.status.value,
            "stage": job.stage, "progress": int(job.progress),
            "stage_progress": round(job.stage_progress, 3), "eta_seconds": job.eta_seconds,
            "error": job.error, "outputs": sorted(job.outputs)})

    async def _broadcast(self, job_id: str, message: dict[str, Any]) -> None:
        for q in list(self._subscribers.get(job_id, [])):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(message)

    async def events(self, job_id: str):
        """Async iterator of events for one job, starting with its current state."""
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._subscribers.setdefault(job_id, []).append(q)
        try:
            yield {"event": "job.progress", **job.to_dict()}
            if job.status.terminal:
                return
            while True:
                message = await q.get()
                yield message
                if message.get("event") == "job.progress" and message.get("status") in (
                        "completed", "failed", "cancelled"):
                    return
        finally:
            self._subscribers[job_id].remove(q)

    # -- worker -------------------------------------------------------------------

    async def _worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            job = self.get(job_id)
            if job is None or job.status.terminal:
                continue
            job.started_at = time.time()
            task = asyncio.create_task(self.executor.run(job, self))
            self._running[job_id] = task
            try:
                outputs = await asyncio.wait_for(task, timeout=self.timeout_seconds)
                job.outputs.update(outputs or {})
                await self._finish(job, JobStatus.COMPLETED)
            except JobCancelled as exc:
                await self._finish(job, JobStatus.CANCELLED, error=exc.to_dict())
            except TimeoutError:
                job.cancel.cancel()
                await self._finish(job, JobStatus.FAILED, error=PipelineError(
                    job.stage, "PROCESSING_TIMEOUT",
                    f"Processing exceeded {self.timeout_seconds:.0f} s.").to_dict())
            except PipelineError as exc:
                logger.warning("job %s failed: %s", job_id, exc)
                await self._finish(job, JobStatus.FAILED, error=exc.to_dict())
            except Exception:
                logger.exception("job %s crashed", job_id)
                await self._finish(job, JobStatus.FAILED, error=PipelineError(
                    job.stage, "INTERNAL_ERROR", "Unexpected error; see server logs.").to_dict())
            finally:
                self._running.pop(job_id, None)

    async def _finish(self, job: JobRecord, status: JobStatus, error: dict | None = None) -> None:
        if job.status.terminal and job.finished_at is not None:
            return  # already finished (e.g. cancelled while queued)
        job.status = status
        job.error = error
        job.finished_at = time.time()
        if status is JobStatus.COMPLETED:
            job.progress, job.stage, job.stage_progress = 100.0, "completed", 1.0
        else:
            job.stage = status.value
        job.updated_at = time.time()
        if job.started_at:
            job.log["processing_seconds"] = round(job.finished_at - job.started_at, 2)
        from voicebridge.core.metrics.metrics import registry

        timings = {f"{k}_seconds": v for k, v in (job.result.get("stage_seconds") or {}).items()}
        timings["processing_seconds"] = job.log.get("processing_seconds")
        timings["rtf"] = job.log.get("rtf")
        registry.record_job(status.value, timings if status is JobStatus.COMPLETED else None)
        if self.store is not None:
            try:
                await self.store.save(f"jobs/{job.job_id}.json",
                                      json.dumps(job.to_dict(), ensure_ascii=False,
                                                 indent=2).encode())
                self.store.cleanup_scratch(job.job_id)
            except Exception as exc:  # pragma: no cover
                logger.warning("could not persist job %s: %s", job.job_id, exc)
        await self._publish(job, "job.progress")

    async def delete(self, job_id: str) -> bool:
        """Cancel if running and remove the job's upload, outputs and record."""
        job = self.get(job_id)
        if job is None:
            return False
        if not job.status.terminal:
            await self.cancel(job_id)
            task = self._running.get(job_id)
            if task is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=30)
                except (TimeoutError, Exception):
                    pass
        if self.store is not None:
            await self.store.delete_prefix(f"outputs/{job_id}")
            await self.store.delete_prefix(f"jobs/{job_id}.json")
            if job.payload.get("upload_key"):
                await self.store.delete_prefix(job.payload["upload_key"])
            self.store.cleanup_scratch(job_id)
        self._jobs.pop(job_id, None)
        return True

    async def purge_older_than(self, seconds: float) -> int:
        """Retention: delete finished jobs whose results are older than ``seconds``."""
        cutoff = time.time() - seconds
        old = [j.job_id for j in self.list() if j.status.terminal and (j.finished_at or 0) < cutoff]
        for job_id in old:
            await self.delete(job_id)
        return len(old)

    def _trim(self) -> None:
        done = [j for j in self.list() if j.status.terminal]
        for job in done[self.retain_jobs:]:
            self._jobs.pop(job.job_id, None)
