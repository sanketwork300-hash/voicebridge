"""Client for model workers running in isolated interpreters.

Why workers exist: some model runtimes pin dependencies that cannot coexist
with the main environment (``qwen-tts`` pins ``transformers==4.57.x`` while the
gateway runs transformers 5; Meta's ``seamless_communication`` needs
``fairseq2`` 0.2 on Python <= 3.11 and torch 2.1). Running each in its own
virtualenv behind a tiny JSON-lines protocol keeps them installable, and gives
two properties for free:

* **unloading** -- stopping the process returns all of its memory, which a
  ``del model`` inside one Python process does not reliably do;
* **fault isolation** -- a native crash in a model kills the worker, not the
  gateway; the next call restarts it.

A worker handles one request at a time (the models are compute-bound, and
concurrent generation on one CPU/GPU only adds contention), enforced here by a
lock.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from voicebridge.providers.base import ProviderError, ProviderUnavailable

logger = logging.getLogger(__name__)

WORKERS_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKERS_DIR.parent.parent


def resolve_python(configured: str | None, env_var: str, default_venv: str) -> str:
    """Worker interpreter: config > environment variable > repo-local venv > current."""
    for candidate in (configured, os.environ.get(env_var),
                      str(REPO_ROOT / ".venv-workers" / default_venv / "bin" / "python")):
        if candidate and Path(candidate).exists():
            return candidate
    return sys.executable


def system_memory_headroom_mb() -> float | None:
    """MemAvailable + SwapFree in MiB (Linux), or None when unknown."""
    try:
        fields = {}
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                fields[key] = float(rest.split()[0])
        return (fields["MemAvailable"] + fields.get("SwapFree", 0.0)) / 1024
    except (OSError, KeyError, ValueError, IndexError):
        return None


#: Headroom below which a worker is stopped instead of letting the kernel's
#: OOM killer act. On a 14 GB machine the SeamlessStreaming worker (fp32,
#: 12+ GB) was OOM-killed and systemd tore down the whole terminal scope with
#: it; a clean, explained failure is much better than that.
MEMORY_RESERVE_MB = float(os.environ.get("VOICEBRIDGE_WORKER_MEMORY_RESERVE_MB", "1536"))


class WorkerError(ProviderError):
    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind


class ModelWorker:
    def __init__(
        self,
        name: str,
        script: str,
        python: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        startup_timeout: float = 900.0,
        request_timeout: float = 1800.0,
        idle_unload_seconds: float | None = None,
    ):
        self.name = name
        self.script = str(WORKERS_DIR / script)
        self.python = python
        self.args = args or []
        self.env = env or {}
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.idle_unload_seconds = idle_unload_seconds
        self.info: dict[str, Any] = {}
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._next_id = 0
        self._last_used = 0.0
        self._reaper: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=8)
        self._memory_stop: str | None = None

    async def _memory_guard(self) -> None:
        """Stop the worker when system memory is nearly exhausted."""
        while True:
            await asyncio.sleep(0.5)
            headroom = system_memory_headroom_mb()
            if headroom is not None and headroom < MEMORY_RESERVE_MB and self.running:
                self._memory_stop = (
                    f"{self.name} worker stopped: system memory nearly exhausted "
                    f"({headroom:.0f} MiB of RAM+swap left). This model needs more memory "
                    "than this machine has available.")
                logger.error(self._memory_stop)
                self._proc.kill()
                return

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def _start(self) -> None:
        if self.running:
            return
        if not Path(self.python).exists():
            raise ProviderUnavailable(f"{self.name} worker interpreter not found: {self.python}")
        env = {**os.environ, **self.env, "PYTHONUNBUFFERED": "1"}
        logger.info("starting %s worker: %s %s", self.name, self.python, self.script)
        self._proc = await asyncio.create_subprocess_exec(
            self.python, self.script, *self.args,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=env, limit=64 * 1024 * 1024,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr(self._proc))
        try:
            hello = await asyncio.wait_for(self._read(), timeout=self.startup_timeout)
        except TimeoutError as exc:
            await self._shutdown()
            raise ProviderUnavailable(f"{self.name} worker did not start in time") from exc
        if not hello or not (hello.get("result") or {}).get("ready"):
            await self._shutdown()
            logger.warning("%s worker failed to start; last stderr: %s", self.name,
                           " | ".join(self._stderr_tail))
            raise ProviderUnavailable(
                f"{self.name} worker failed to start (see log). Check that its virtualenv "
                f"({self.python}) has the model dependencies installed."
            )
        self.info = hello["result"]
        if self.idle_unload_seconds and self._reaper is None:
            self._reaper = asyncio.create_task(self._idle_reaper())

    async def _drain_stderr(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr is not None
        async for raw in proc.stderr:
            line = raw.decode(errors="replace").rstrip()
            if line:
                self._stderr_tail.append(line[:300])
                logger.debug("[%s worker] %s", self.name, line)

    async def _read(self) -> dict[str, Any] | None:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                continue

    async def call(self, method: str, timeout: float | None = None, **params: Any) -> Any:
        async with self._lock:
            await self._start()
            assert self._proc is not None and self._proc.stdin is not None
            self._next_id += 1
            rid = self._next_id
            self._proc.stdin.write((json.dumps({"id": rid, "method": method,
                                                "params": params}) + "\n").encode())
            await self._proc.stdin.drain()
            self._memory_stop = None
            guard = asyncio.create_task(self._memory_guard())
            try:
                while True:
                    reply = await asyncio.wait_for(self._read(),
                                                   timeout=timeout or self.request_timeout)
                    if reply is None:
                        proc, self._proc = self._proc, None
                        code = await proc.wait() if proc is not None else None
                        if self._memory_stop:
                            raise ProviderError(self._memory_stop)
                        raise ProviderError(f"{self.name} worker exited (code {code}); "
                                            f"last output: {' | '.join(self._stderr_tail)}")
                    if reply.get("id") == rid:
                        break
            except TimeoutError as exc:
                await self._kill()
                raise ProviderError(f"{self.name} worker timed out on {method}") from exc
            except (BrokenPipeError, ConnectionResetError) as exc:
                self._proc = None
                raise ProviderError(f"{self.name} worker connection lost") from exc
            finally:
                guard.cancel()
            self._last_used = time.monotonic()
            if "error" in reply:
                err = reply["error"] or {}
                raise WorkerError(err.get("type", "Error"), err.get("message", ""))
            return reply.get("result")

    async def _idle_reaper(self) -> None:
        while True:
            await asyncio.sleep(max(5.0, (self.idle_unload_seconds or 60) / 4))
            if not (self.running and self._last_used):
                continue
            async with self._lock:  # never stop a worker while a call is in flight
                if (self.running and time.monotonic() - self._last_used
                        > (self.idle_unload_seconds or 0)):
                    logger.info("unloading idle %s worker", self.name)
                    await self._shutdown()

    async def _kill(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.kill()
            await self._proc.wait()
        self._proc = None

    async def stop(self) -> None:
        async with self._lock:
            await self._shutdown()
        if self._reaper is not None:
            self._reaper.cancel()
            self._reaper = None

    async def _shutdown(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.returncode is None and proc.stdin is not None:
            try:
                proc.stdin.write(b'{"id": -1, "method": "shutdown"}\n')
                await proc.stdin.drain()
                await asyncio.wait_for(proc.wait(), timeout=10)
            except (TimeoutError, BrokenPipeError, ConnectionResetError):
                pass
        await self._kill()
