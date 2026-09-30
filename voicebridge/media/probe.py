"""FFmpeg discovery and media probing.

FFmpeg is a *system* dependency, resolved in this order: ``media.ffmpeg_path``
/ ``media.ffprobe_path`` in configuration, ``VOICEBRIDGE_FFMPEG`` /
``VOICEBRIDGE_FFPROBE``, then ``PATH``. When it is missing every media entry
point raises :class:`FFmpegMissing` with an actionable message instead of a
``FileNotFoundError`` traceback from ``subprocess``.

Uploaded files are untrusted: ffprobe is run with a timeout, only on paths
VoiceBridge generated, and its answer decides whether the file is usable
(``CORRUPTED_MEDIA`` / ``NO_AUDIO_STREAM``) before any decoding happens.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from voicebridge.core.jobs.errors import PipelineError

FFMPEG_REQUIRED = (
    "FFmpeg is required for audio/video file translation. "
    "Install FFmpeg and ensure it is available in PATH."
)

_CONFIG: dict[str, str | None] = {"ffmpeg": None, "ffprobe": None}


class FFmpegMissing(PipelineError):
    def __init__(self) -> None:
        super().__init__("preprocessing", "FFMPEG_MISSING", FFMPEG_REQUIRED, retryable=False)


def configure(ffmpeg_path: str | None = None, ffprobe_path: str | None = None) -> None:
    _CONFIG["ffmpeg"] = ffmpeg_path
    _CONFIG["ffprobe"] = ffprobe_path


def _find(tool: str) -> str | None:
    for candidate in (_CONFIG.get(tool), os.environ.get(f"VOICEBRIDGE_{tool.upper()}")):
        if candidate and Path(candidate).exists():
            return candidate
    return shutil.which(tool)


def ffmpeg_binary() -> str:
    path = _find("ffmpeg")
    if not path:
        raise FFmpegMissing()
    return path


def ffprobe_binary() -> str:
    path = _find("ffprobe")
    if not path:
        sibling = Path(ffmpeg_binary()).with_name("ffprobe")
        if sibling.exists():
            return str(sibling)
        raise FFmpegMissing()
    return path


#: Demuxer forced per extension, so FFmpeg never sniffs an untrusted file into
#: a different (e.g. playlist) format. Combined with ``-protocol_whitelist
#: file`` an input can only ever read its own local file.
DEMUXERS = {".wav": "wav", ".mp3": "mp3", ".m4a": "mov", ".mp4": "mov", ".mov": "mov",
            ".mkv": "matroska", ".webm": "matroska", ".flac": "flac", ".ogg": "ogg"}


def input_args(path: str | Path) -> list[str]:
    """Safe FFmpeg/ffprobe input options for a local media file."""
    fmt = DEMUXERS.get(Path(path).suffix.lower())
    return ["-protocol_whitelist", "file"] + (["-f", fmt] if fmt else []) + ["-i", str(path)]


def ffmpeg_available() -> bool:
    return _find("ffmpeg") is not None and (_find("ffprobe") is not None)


def require_ffmpeg() -> None:
    ffmpeg_binary()
    ffprobe_binary()


@dataclass
class MediaInfo:
    path: str
    duration: float = 0.0
    format_name: str | None = None
    bitrate: int | None = None
    audio_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    video_codec: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    audio_streams: int = 0
    video_streams: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def has_video(self) -> bool:
        return self.video_streams > 0

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        d["path"] = Path(self.path).name
        return d


async def run_tool(args: list[str], timeout: float, stage: str) -> tuple[bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise PipelineError(stage, "MEDIA_TIMEOUT", f"{Path(args[0]).name} timed out") from exc
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
    if proc.returncode != 0:
        message = err.decode(errors="ignore").strip().splitlines()
        raise PipelineError(stage, "MEDIA_TOOL_FAILED",
                            (message[-1] if message else f"exit {proc.returncode}")[:300])
    return out, err


async def probe(path: str | Path, timeout: float = 60.0) -> MediaInfo:
    try:
        out, _ = await run_tool(
            [ffprobe_binary(), "-v", "error", "-print_format", "json", "-show_format",
             "-show_streams", *input_args(path)],
            timeout, "preprocessing",
        )
    except PipelineError as exc:
        if exc.code == "MEDIA_TOOL_FAILED":
            raise PipelineError("preprocessing", "CORRUPTED_MEDIA",
                                "The file could not be read as audio or video.") from exc
        raise
    data = json.loads(out.decode() or "{}")
    info = MediaInfo(path=str(path), raw=data)
    fmt = data.get("format") or {}
    info.duration = float(fmt.get("duration") or 0.0)
    info.format_name = fmt.get("format_name")
    info.bitrate = int(fmt["bit_rate"]) if str(fmt.get("bit_rate", "")).isdigit() else None
    for stream in data.get("streams") or []:
        kind = stream.get("codec_type")
        if kind == "audio":
            info.audio_streams += 1
            if info.audio_codec is None:
                info.audio_codec = stream.get("codec_name")
                info.sample_rate = int(stream["sample_rate"]) if stream.get("sample_rate") else None
                info.channels = int(stream["channels"]) if stream.get("channels") else None
        elif kind == "video" and (stream.get("disposition") or {}).get("attached_pic") != 1:
            info.video_streams += 1
            if info.video_codec is None:
                info.video_codec = stream.get("codec_name")
                info.width = stream.get("width")
                info.height = stream.get("height")
                num, _, den = (stream.get("avg_frame_rate") or "0/1").partition("/")
                try:
                    info.fps = round(float(num) / float(den or 1), 3) if float(den or 1) else None
                except ValueError:
                    info.fps = None
    if info.audio_streams == 0:
        raise PipelineError("preprocessing", "NO_AUDIO_STREAM", "The file has no audio track.")
    if info.duration <= 0:
        raise PipelineError("preprocessing", "CORRUPTED_MEDIA", "The file reports no duration.")
    return info
