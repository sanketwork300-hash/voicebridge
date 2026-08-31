"""Bounded queues with explicit overflow policies.

Rule, enforced by ``tests/streaming/test_backpressure.py``: **no queue in
VoiceBridge is unbounded.** A real-time pipeline that buffers without limit does
not fail, it just silently drifts further behind until the output is useless --
which is worse than dropping, because the user cannot tell it is happening.

Each stage therefore declares what may be sacrificed under overload:

``DROP_OLDEST``
    Ring-buffer behaviour. Correct for audio capture, where the freshest audio
    matters and ancient audio is worthless.
``DROP_NEWEST_DISPOSABLE``
    Drop incoming items that declare themselves disposable (partial ASR
    hypotheses, partial translations) while never dropping committed items.
``BLOCK``
    Apply real backpressure to the producer. Correct for TTS, where ordering
    and completeness are mandatory.
"""

from __future__ import annotations

import asyncio
import logging
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, runtime_checkable

logger = logging.getLogger(__name__)


class OverflowPolicy(StrEnum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST_DISPOSABLE = "drop_newest_disposable"
    BLOCK = "block"


@runtime_checkable
class Disposable(Protocol):
    """An item that may be discarded under load without breaking correctness."""

    @property
    def disposable(self) -> bool: ...


T = TypeVar("T")


class BoundedQueue(Generic[T]):
    """An ``asyncio.Queue`` with a declared overflow policy and drop counters."""

    def __init__(
        self,
        name: str,
        maxsize: int,
        policy: OverflowPolicy = OverflowPolicy.BLOCK,
    ):
        if maxsize <= 0:
            raise ValueError("BoundedQueue requires a positive maxsize; unbounded queues are forbidden")
        self.name = name
        self.maxsize = maxsize
        self.policy = policy
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        self.enqueued = 0

    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def depth_ratio(self) -> float:
        return self._queue.qsize() / self.maxsize

    @property
    def overloaded(self) -> bool:
        """True once the queue is ≥80% full -- surfaced to the client as a warning."""
        return self.depth_ratio >= 0.8

    def _is_disposable(self, item: T) -> bool:
        return isinstance(item, Disposable) and bool(item.disposable)

    async def put(self, item: T) -> bool:
        """Enqueue ``item``. Returns False when the item was dropped."""
        if self.policy is OverflowPolicy.BLOCK:
            await self._queue.put(item)
            self.enqueued += 1
            return True

        try:
            self._queue.put_nowait(item)
            self.enqueued += 1
            return True
        except asyncio.QueueFull:
            pass

        if self.policy is OverflowPolicy.DROP_OLDEST:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self.dropped += 1
            except asyncio.QueueEmpty:  # drained concurrently
                pass
            try:
                self._queue.put_nowait(item)
                self.enqueued += 1
                return True
            except asyncio.QueueFull:
                self.dropped += 1
                return False

        # DROP_NEWEST_DISPOSABLE
        if self._is_disposable(item):
            self.dropped += 1
            logger.debug("queue %s full; dropped disposable item", self.name)
            return False
        # Non-disposable items must survive: fall back to blocking.
        await self._queue.put(item)
        self.enqueued += 1
        return True

    async def get(self) -> T:
        item = await self._queue.get()
        self._queue.task_done()
        return item

    def get_nowait(self) -> T | None:
        try:
            item = self._queue.get_nowait()
            self._queue.task_done()
            return item
        except asyncio.QueueEmpty:
            return None

    def stats(self) -> dict:
        return {
            "name": self.name,
            "depth": self.qsize(),
            "maxsize": self.maxsize,
            "policy": self.policy.value,
            "enqueued": self.enqueued,
            "dropped": self.dropped,
            "overloaded": self.overloaded,
        }
