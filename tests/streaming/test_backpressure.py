"""Queue bounding and overflow behaviour.

The governing rule: no queue in VoiceBridge may be unbounded, and each must
declare what it sacrifices under load.
"""

import asyncio
from dataclasses import dataclass

import pytest

from voicebridge.core.streaming.queue import BoundedQueue, OverflowPolicy

pytestmark = pytest.mark.asyncio


@dataclass
class Item:
    value: int
    is_partial: bool = False

    @property
    def disposable(self) -> bool:
        return self.is_partial


async def test_unbounded_queue_is_rejected():
    with pytest.raises(ValueError):
        BoundedQueue("bad", 0)
    with pytest.raises(ValueError):
        BoundedQueue("bad", -1)


async def test_drop_oldest_keeps_freshest_audio():
    q = BoundedQueue("audio", 3, OverflowPolicy.DROP_OLDEST)
    for i in range(6):
        await q.put(i)
    assert [q.get_nowait() for _ in range(3)] == [3, 4, 5]
    assert q.dropped == 3


async def test_drop_newest_disposable_protects_committed_items():
    q = BoundedQueue("translation", 2, OverflowPolicy.DROP_NEWEST_DISPOSABLE)
    await q.put(Item(1))
    await q.put(Item(2))
    # A disposable item is dropped rather than blocking.
    assert await q.put(Item(3, is_partial=True)) is False
    assert q.dropped == 1

    # A non-disposable item must survive: the put blocks until space frees up.
    async def drain():
        await asyncio.sleep(0.01)
        q.get_nowait()

    drainer = asyncio.create_task(drain())
    assert await asyncio.wait_for(q.put(Item(4)), timeout=1.0) is True
    await drainer


async def test_block_policy_applies_real_backpressure():
    q = BoundedQueue("tts", 1, OverflowPolicy.BLOCK)
    await q.put(Item(1))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(q.put(Item(2)), timeout=0.05)
    assert q.dropped == 0          # BLOCK never loses anything


async def test_overloaded_flag_tracks_depth():
    q = BoundedQueue("audio", 10, OverflowPolicy.DROP_OLDEST)
    for i in range(7):
        await q.put(i)
    assert q.overloaded is False
    for i in range(2):
        await q.put(i)
    assert q.overloaded is True     # >= 80%


async def test_stats_are_reported():
    q = BoundedQueue("audio", 2, OverflowPolicy.DROP_OLDEST)
    for i in range(4):
        await q.put(i)
    stats = q.stats()
    assert stats["maxsize"] == 2
    assert stats["dropped"] == 2
    assert stats["policy"] == "drop_oldest"
