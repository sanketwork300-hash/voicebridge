"""WebSocket session handler: transport-agnostic core of the streaming API.

Kept free of FastAPI types so the same logic can serve a different transport.
The handler owns exactly one session's socket lifetime.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from voicebridge.core.audio.format import convert_pcm
from voicebridge.core.session.manager import Session
from voicebridge.core.types import INTERNAL_FORMAT, AudioChunk, AudioFormat
from voicebridge.protocols.websocket.messages import (
    ClientMessage,
    ProtocolError,
    ServerMessage,
    parse_control,
    validate_audio_frame,
)

logger = logging.getLogger(__name__)


class SessionSocket:
    """Bridges a WebSocket to a session pipeline.

    ``send_json`` is injected so the transport (FastAPI, websockets, a test
    double) is irrelevant to this logic.
    """

    def __init__(
        self,
        session: Session,
        send_json: Callable[[dict], Awaitable[None]],
    ):
        self.session = session
        self.send_json = send_json
        self._forward_task: asyncio.Task | None = None
        #: Client-declared input format; normalised to INTERNAL_FORMAT on entry.
        self._input_format = AudioFormat(
            sample_rate=session.config.input.sample_rate,
            channels=session.config.input.channels,
        )

    stop_requested = False

    async def wait_forwarded(self, timeout: float) -> None:
        """Wait until every remaining pipeline event has been sent."""
        if self._forward_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._forward_task), timeout=timeout)
            except (TimeoutError, asyncio.CancelledError):
                pass

    async def start(self) -> None:
        self._forward_task = asyncio.create_task(self._forward_events())

    async def _forward_events(self) -> None:
        try:
            async for event in self.session.pipeline.stream_events():
                await self.send_json(event.to_dict())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - transport failures
            logger.info("event forwarding stopped: %s", exc)

    async def handle_text(self, raw: str) -> bool:
        """Handle a control frame. Returns False when the socket should close."""
        try:
            message = parse_control(raw)
        except ProtocolError as exc:
            await self.send_json(
                {
                    "event_type": ServerMessage.ERROR.value,
                    "session_id": self.session.session_id,
                    "code": "protocol_error",
                    "message": str(exc),
                    "recoverable": True,
                }
            )
            return True

        if message.type is ClientMessage.PING:
            await self.send_json(
                {
                    "event_type": ServerMessage.PONG.value,
                    "session_id": self.session.session_id,
                    "echo": message.payload.get("echo"),
                }
            )
            return True

        if message.type is ClientMessage.SESSION_CONFIG:
            audio = message.payload.get("input") or {}
            if audio:
                self._input_format = AudioFormat(
                    sample_rate=int(audio.get("sample_rate", self._input_format.sample_rate)),
                    channels=int(audio.get("channels", self._input_format.channels)),
                )
            return True

        if message.type is ClientMessage.AUDIO_START:
            return True

        if message.type is ClientMessage.PAUSE:
            await self.session.pipeline.pause()
            return True

        if message.type is ClientMessage.RESUME:
            await self.session.pipeline.resume()
            return True

        if message.type is ClientMessage.AUDIO_STOP:
            # The client is done sending: the gateway drains the pipeline and
            # delivers the remaining events before closing the socket.
            self.stop_requested = True
            return False

        return True

    async def handle_binary(self, data: bytes) -> None:
        try:
            validate_audio_frame(data)
        except ProtocolError as exc:
            await self.send_json(
                {
                    "event_type": ServerMessage.ERROR.value,
                    "session_id": self.session.session_id,
                    "code": "invalid_audio",
                    "message": str(exc),
                    "recoverable": True,
                }
            )
            return
        if not data:
            return
        if self._input_format != INTERNAL_FORMAT:
            data = convert_pcm(data, self._input_format, INTERNAL_FORMAT)
        await self.session.pipeline.push_audio(AudioChunk(data=data))

    async def close(self) -> None:
        if self._forward_task is not None:
            self._forward_task.cancel()
            try:
                await self._forward_task
            except (asyncio.CancelledError, Exception):
                pass
            self._forward_task = None
