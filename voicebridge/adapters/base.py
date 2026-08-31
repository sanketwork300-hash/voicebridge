"""Input and output adapter interfaces.

The core pipeline never learns which adapter produced its audio or where its
output goes. That is what lets the same engine serve a browser tab, a
microphone, a file and a WebRTC track without conditionals inside the pipeline.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator

from voicebridge.core.types import AudioChunk, SynthesisedAudio


class AudioInput(abc.ABC):
    """A source of audio."""

    name: str = "input"

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def read(self) -> AudioChunk | None:
        """Return the next chunk, or None when the source is exhausted."""

    @abc.abstractmethod
    async def stop(self) -> None: ...

    async def stream(self) -> AsyncIterator[AudioChunk]:
        await self.start()
        try:
            while True:
                chunk = await self.read()
                if chunk is None:
                    return
                yield chunk
        finally:
            await self.stop()


class AudioOutput(abc.ABC):
    """A sink for synthesised audio."""

    name: str = "output"

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def write(self, audio: SynthesisedAudio) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...
