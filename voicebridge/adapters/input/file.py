"""File input adapter.

Reads WAV directly (stdlib) and anything else through FFmpeg. Used by the CLI,
the deterministic e2e test and the benchmark harness, where reproducibility
matters more than real-time behaviour.

``realtime=True`` paces reads to wall-clock speed, which is what you want when
reproducing a latency problem; the default reads as fast as the pipeline
accepts, which is what you want in CI.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from voicebridge.adapters.base import AudioInput
from voicebridge.core.audio.normalizer import FFmpegDecoder, decode_wav
from voicebridge.core.types import INTERNAL_FORMAT, AudioChunk


class FileAudioInput(AudioInput):
    name = "file"

    def __init__(self, path: str, chunk_seconds: float = 0.2, realtime: bool = False):
        self.path = Path(path)
        self.chunk_seconds = chunk_seconds
        self.realtime = realtime
        self._pcm = b""
        self._offset = 0
        self._decoder: FFmpegDecoder | None = None
        self._elapsed = 0.0

    @property
    def _chunk_bytes(self) -> int:
        return int(self.chunk_seconds * INTERNAL_FORMAT.bytes_per_second) // 2 * 2

    async def start(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"audio file not found: {self.path}")
        payload = self.path.read_bytes()
        if self.path.suffix.lower() == ".wav":
            self._pcm = decode_wav(payload)
        elif self.path.suffix.lower() in (".pcm", ".raw"):
            self._pcm = payload
        else:
            decoder = FFmpegDecoder()
            await decoder.start()
            self._decoder = decoder
            await decoder.write(payload)
            await decoder.close_stdin()
            chunks = []
            async for block in decoder.stream():
                chunks.append(block)
            self._pcm = b"".join(chunks)
            await decoder.stop()
        self._offset = 0
        self._elapsed = 0.0
        self._total_bytes = len(self._pcm)

    async def read(self) -> AudioChunk | None:
        if self._offset >= len(self._pcm):
            return None
        block = self._pcm[self._offset : self._offset + self._chunk_bytes]
        self._offset += len(block)
        chunk = AudioChunk(data=block, start=self._elapsed)
        self._elapsed += chunk.duration
        if self.realtime:
            await asyncio.sleep(chunk.duration)
        return chunk

    async def stop(self) -> None:
        if self._decoder is not None:
            await self._decoder.stop()
            self._decoder = None
        self._pcm = b""
        self._offset = 0

    @property
    def total_seconds(self) -> float:
        """Duration of the decoded audio, valid after ``stop()``."""
        return INTERNAL_FORMAT.duration_of(self._total_bytes)
