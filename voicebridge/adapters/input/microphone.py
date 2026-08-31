"""Microphone / system-audio input via ``sounddevice`` (PortAudio).

Optional dependency. On Linux, selecting a PulseAudio/PipeWire *monitor* source
here is what gives system-audio capture, which is why the adapter exposes raw
device enumeration rather than hiding it: virtual-audio routing differs enough
per OS that pretending otherwise would be wrong.
"""

from __future__ import annotations

import asyncio
import queue
from typing import Any

from voicebridge.adapters.base import AudioInput
from voicebridge.core.audio.format import convert_pcm
from voicebridge.core.types import INTERNAL_FORMAT, AudioChunk, AudioFormat
from voicebridge.providers.base import ProviderUnavailable


def _require_sounddevice():
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        # OSError happens when PortAudio itself is missing, which is a different
        # fix from a missing Python package, so say both.
        raise ProviderUnavailable(
            "microphone input needs the 'sounddevice' package and the PortAudio "
            "library.\n    pip install 'voicebridge[audio-device]'\n"
            "    Debian/Ubuntu: sudo apt install libportaudio2"
        ) from exc
    return sounddevice


def list_devices() -> list[dict[str, Any]]:
    """Enumerate audio devices. Monitor/loopback devices give system audio."""
    sd = _require_sounddevice()
    out = []
    for index, device in enumerate(sd.query_devices()):
        if device.get("max_input_channels", 0) <= 0:
            continue
        name = device.get("name", "")
        out.append(
            {
                "index": index,
                "name": name,
                "channels": device["max_input_channels"],
                "default_samplerate": device.get("default_samplerate"),
                "likely_system_audio": "monitor" in name.lower()
                or "loopback" in name.lower()
                or "stereo mix" in name.lower(),
            }
        )
    return out


class MicrophoneAudioInput(AudioInput):
    name = "microphone"

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_seconds: float = 0.1,
        max_queued_chunks: int = 50,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_seconds = chunk_seconds
        # Bounded: PortAudio's callback thread must never block, and an
        # unbounded queue would just hide that we cannot keep up.
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=max_queued_chunks)
        self._stream: Any = None
        self._elapsed = 0.0
        self._dropped = 0

    async def start(self) -> None:
        sd = _require_sounddevice()
        frames = int(self.sample_rate * self.chunk_seconds)

        def callback(indata, _frames, _time, status):
            if status:
                pass  # over/underflow; audio is best-effort by design
            try:
                self._queue.put_nowait(bytes(indata))
            except queue.Full:
                self._dropped += 1

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=frames,
            device=self.device,
            channels=self.channels,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()

    async def read(self) -> AudioChunk | None:
        if self._stream is None:
            return None
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(None, self._queue.get, True, 1.0)
        except queue.Empty:
            return AudioChunk(data=b"", start=self._elapsed)
        src = AudioFormat(sample_rate=self.sample_rate, channels=self.channels)
        if src != INTERNAL_FORMAT:
            data = convert_pcm(data, src, INTERNAL_FORMAT)
        chunk = AudioChunk(data=data, start=self._elapsed)
        self._elapsed += chunk.duration
        return chunk

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def dropped_chunks(self) -> int:
        return self._dropped
