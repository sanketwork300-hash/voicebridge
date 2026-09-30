"""Upload validation for file translation.

Uploaded files are untrusted input. They are:

* limited by extension (allow-list) *and* by content signature ("magic
  bytes"), so a renamed executable or HTML page is refused before FFmpeg ever
  sees it; ffprobe then validates the container properly;
* streamed to disk in 1 MiB chunks with a hard size cap -- never read whole
  into memory;
* stored under a server-generated UUID name; the client's filename is used
  only, after sanitising, to name the output files;
* refused when the storage quota would be exceeded.

Nothing uploaded is ever executed.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

from voicebridge.core.jobs.executor import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from voicebridge.storage.base import ArtifactStore

ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
CHUNK = 1024 * 1024


def sniff(head: bytes) -> str | None:
    """Coarse container type from the first bytes, or None if unrecognised."""
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "mp3"
    if head[4:8] == b"ftyp":
        return "isobmff"  # mp4 / m4a / mov
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "matroska"  # mkv / webm
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    return None


EXPECTED = {
    ".wav": {"wav"}, ".mp3": {"mp3"}, ".m4a": {"isobmff"}, ".flac": {"flac"}, ".ogg": {"ogg"},
    ".mp4": {"isobmff"}, ".mov": {"isobmff"}, ".mkv": {"matroska"}, ".webm": {"matroska"},
}


def sanitize_filename(name: str | None) -> str:
    base = Path(name or "media").name
    stem, suffix = Path(base).stem, Path(base).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:80] or "media"
    return f"{stem}{suffix}"


async def save_upload(file: UploadFile, store: ArtifactStore, max_bytes: int,
                      quota_bytes: int | None = None) -> tuple[str, str, int]:
    """Validate and store an upload. Returns (storage key, clean filename, size)."""
    filename = sanitize_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"unsupported file type {suffix or '(none)'}; allowed: "
                                 f"{', '.join(sorted(ALLOWED_EXTENSIONS))}")
    limit = max_bytes
    if quota_bytes is not None:
        free = quota_bytes - await store.usage_bytes_async()
        if free <= 0:
            raise HTTPException(507, "storage quota exhausted; delete old jobs "
                                     "(DELETE /api/v1/jobs/{id}) and retry")
        limit = min(max_bytes, free)
    head = await file.read(64)
    if sniff(head) not in EXPECTED[suffix]:
        raise HTTPException(415, f"file content does not look like {suffix}")
    key = f"uploads/{uuid.uuid4().hex}{suffix}"

    async def chunks():
        yield head
        while True:
            chunk = await file.read(CHUNK)
            if not chunk:
                return
            yield chunk

    try:
        size = await store.save_stream(key, chunks(), limit)
    except ValueError as exc:
        if str(exc) == "too_large":
            if limit < max_bytes:
                raise HTTPException(507, "not enough storage quota left for this file") from None
            raise HTTPException(413, f"file exceeds {max_bytes // (1024 * 1024)} MB") from None
        raise
    finally:
        await file.close()
    if size == 0:
        await store.delete(key)
        raise HTTPException(400, "empty upload")
    return key, filename, size
