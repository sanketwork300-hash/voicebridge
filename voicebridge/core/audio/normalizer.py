"""Normalise arbitrary external audio into the single internal PCM contract.

Two paths exist, and which one is used matters a lot operationally:

1. **Raw PCM path** (no external process). The client already sends
   ``pcm_s16le``; we only downmix/resample. This is what the browser extension
   does, and it is the default.
2. **Container path** (FFmpeg subprocess). Needed for WebM/Opus, MP3, MP4 and
   anything else with a container or a compressed codec.

WhisperLiveKit makes exactly this split -- its ``--pcm-input`` flag bypasses
FFmpeg entirely (see ``whisperlivekit/ffmpeg_manager.py``, which prints install
instructions and points at ``--pcm-input`` as the alternative). VoiceBridge
follows it because FFmpeg is the single most common missing dependency in
self-hosted deployments, and requiring it for the primary browser flow would
make the MVP fail on a clean machine.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import wave
from collections.abc import AsyncIterator

from voicebridge.core.audio.format import convert_pcm
from voicebridge.core.types import INTERNAL_FORMAT, AudioFormat

logger = logging.getLogger(__name__)

FFMPEG_MISSING_MESSAGE = (
    "FFmpeg was not found on PATH. VoiceBridge needs it only for compressed or "
    "containerised audio (WebM/Opus, MP3, MP4). The browser extension sends raw "
    "PCM and does not require FFmpeg. Install it with "
    "`sudo apt install ffmpeg` / `brew install ffmpeg`, or send PCM instead."
)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


class AudioNormalizer:
    """Stateless-per-chunk normaliser for raw PCM inputs."""

    def __init__(self, source_format: AudioFormat, target_format: AudioFormat = INTERNAL_FORMAT):
        self.source_format = source_format
        self.target_format = target_format

    def normalize(self, data: bytes) -> bytes:
        if not data:
            return b""
        return convert_pcm(data, self.source_format, self.target_format)


def decode_wav(payload: bytes, target: AudioFormat = INTERNAL_FORMAT) -> bytes:
    """Decode a RIFF/WAVE byte string to the internal format, without FFmpeg."""
    with wave.open(io.BytesIO(payload), "rb") as wf:
        src = AudioFormat(
            sample_rate=wf.getframerate(),
            channels=wf.getnchannels(),
            sample_width=wf.getsampwidth(),
        )
        frames = wf.readframes(wf.getnframes())
    return convert_pcm(frames, src, target)


def encode_wav(pcm: bytes, fmt: AudioFormat = INTERNAL_FORMAT) -> bytes:
    """Wrap raw PCM in a WAV container (used by the file output adapter)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(fmt.channels)
        wf.setsampwidth(fmt.sample_width)
        wf.setframerate(fmt.sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class FFmpegDecoder:
    """Streaming decoder for containerised/compressed audio.

    Writes encoded bytes to ffmpeg's stdin and reads internal-format PCM from
    stdout. The process is restarted by the caller on failure; we surface state
    rather than silently swallowing a dead decoder.
    """

    def __init__(self, target: AudioFormat = INTERNAL_FORMAT):
        self.target = target
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        if not ffmpeg_available():
            raise RuntimeError(FFMPEG_MISSING_MESSAGE)
        self._proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", str(self.target.channels),
            "-ar", str(self.target.sample_rate),
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def write(self, data: bytes) -> None:
        if not self.running or self._proc is None or self._proc.stdin is None:
            raise RuntimeError("FFmpeg decoder is not running")
        self._proc.stdin.write(data)
        await self._proc.stdin.drain()

    async def read(self, n: int = 4096) -> bytes:
        if self._proc is None or self._proc.stdout is None:
            return b""
        return await self._proc.stdout.read(n)

    async def stream(self, n: int = 4096) -> AsyncIterator[bytes]:
        while True:
            chunk = await self.read(n)
            if not chunk:
                return
            yield chunk

    async def close_stdin(self) -> None:
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except (BrokenPipeError, RuntimeError):  # already gone
                pass

    async def stop(self) -> None:
        if self._proc is None:
            return
        await self.close_stdin()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5)
        except TimeoutError:
            self._proc.kill()
            await self._proc.wait()
        finally:
            self._proc = None
