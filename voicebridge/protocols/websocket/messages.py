"""WebSocket message definitions and validation.

The wire protocol is specified in ``docs/protocol.md``; this module is its
executable form. Two framing rules:

* **Binary frames are audio.** A binary WebSocket frame is raw PCM for the
  current session, with no envelope. Wrapping audio in JSON+base64 inflates it
  by a third and adds encode/decode cost to the hottest path in the system.
* **Text frames are JSON control messages**, each with a ``type`` field.

Server-to-client audio *is* base64-in-JSON, deliberately: it is far lower volume
than the input stream, and keeping it in the ordered event channel is what
guarantees it arrives interleaved correctly with the subtitle events it belongs
with.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ClientMessage(StrEnum):
    SESSION_CONFIG = "SESSION_CONFIG"
    AUDIO_START = "AUDIO_START"
    AUDIO_STOP = "AUDIO_STOP"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    PING = "PING"


class ServerMessage(StrEnum):
    SESSION_STARTED = "SESSION_STARTED"
    ASR_PARTIAL = "ASR_PARTIAL"
    ASR_STABLE = "ASR_STABLE"
    TRANSLATION_PARTIAL = "TRANSLATION_PARTIAL"
    TRANSLATION_FINAL = "TRANSLATION_FINAL"
    TTS_AUDIO = "TTS_AUDIO"
    LANGUAGE_DETECTED = "LANGUAGE_DETECTED"
    SPEAKER_CHANGED = "SPEAKER_CHANGED"
    METRIC = "METRIC"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SESSION_ENDED = "SESSION_ENDED"
    PONG = "PONG"


#: Hard cap on a single inbound audio frame. At 16 kHz mono s16le this is ~8
#: seconds of audio -- far more than any sane client sends, while still bounding
#: what a malicious client can allocate in one frame.
MAX_AUDIO_FRAME_BYTES = 256 * 1024

#: Hard cap on a JSON control frame.
MAX_CONTROL_FRAME_BYTES = 64 * 1024


class ProtocolError(ValueError):
    """Raised for a malformed or oversized client message."""


@dataclass
class ParsedControl:
    type: ClientMessage
    payload: dict[str, Any]


def parse_control(raw: str) -> ParsedControl:
    """Validate and parse a JSON control frame."""
    import json

    if len(raw.encode("utf-8", errors="ignore")) > MAX_CONTROL_FRAME_BYTES:
        raise ProtocolError("control frame too large")
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ProtocolError("control frame must be a JSON object")
    raw_type = data.get("type")
    if not isinstance(raw_type, str):
        raise ProtocolError("missing 'type'")
    try:
        message_type = ClientMessage(raw_type)
    except ValueError:
        raise ProtocolError(f"unknown message type {raw_type!r}") from None
    payload = {k: v for k, v in data.items() if k != "type"}
    return ParsedControl(type=message_type, payload=payload)


def validate_audio_frame(data: bytes, sample_width: int = 2) -> bytes:
    """Validate an inbound binary audio frame."""
    if len(data) > MAX_AUDIO_FRAME_BYTES:
        raise ProtocolError(
            f"audio frame of {len(data)} bytes exceeds the {MAX_AUDIO_FRAME_BYTES} byte limit"
        )
    if len(data) % sample_width:
        # A truncated sample means the client's framing is wrong; silently
        # padding it would inject a click into the audio every frame.
        raise ProtocolError("audio frame is not aligned to the sample width")
    return data


def error_message(message: str, code: str = "error", recoverable: bool = True) -> dict[str, Any]:
    return {
        "event_type": ServerMessage.ERROR.value,
        "code": code,
        "message": message,
        "recoverable": recoverable,
    }
