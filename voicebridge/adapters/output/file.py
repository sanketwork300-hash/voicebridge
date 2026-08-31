"""File output adapters: dubbed audio as WAV, subtitles as SRT/VTT.

Subtitle timings come from the *source* sample clock, not from when the
translation happened to be produced, so an exported subtitle file lines up with
the original media even if the pipeline ran slower than real time.
"""

from __future__ import annotations

from pathlib import Path

from voicebridge.adapters.base import AudioOutput
from voicebridge.core.audio.normalizer import encode_wav
from voicebridge.core.types import INTERNAL_FORMAT, SynthesisedAudio


class WavFileOutput(AudioOutput):
    name = "file"

    def __init__(self, path: str):
        self.path = Path(path)
        self._chunks: list[bytes] = []

    async def start(self) -> None:
        self._chunks = []

    async def write(self, audio: SynthesisedAudio) -> None:
        self._chunks.append(audio.audio)

    async def stop(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(encode_wav(b"".join(self._chunks), INTERNAL_FORMAT))


def _timestamp(seconds: float, comma: bool = True) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s, m, h = total_s % 60, (total_s // 60) % 60, total_s // 3600
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


class SubtitleWriter:
    """Collects committed cues and writes SRT or WebVTT."""

    def __init__(self, dual: bool = False):
        self.dual = dual
        self.cues: list[dict] = []

    def add(
        self,
        start: float,
        end: float,
        translation: str,
        original: str | None = None,
        speaker: int | None = None,
    ) -> None:
        if end <= start:
            # A zero-length cue is invalid in both formats; give it a floor so
            # players do not skip or reject it.
            end = start + 0.8
        self.cues.append(
            {
                "start": start,
                "end": end,
                "translation": translation.strip(),
                "original": (original or "").strip(),
                "speaker": speaker,
            }
        )

    def _body(self, cue: dict) -> str:
        prefix = f"[Speaker {cue['speaker']}] " if cue["speaker"] is not None else ""
        if self.dual and cue["original"]:
            return f"{prefix}{cue['translation']}\n{cue['original']}"
        return f"{prefix}{cue['translation']}"

    def to_srt(self) -> str:
        blocks = []
        for index, cue in enumerate(self.cues, start=1):
            blocks.append(
                f"{index}\n"
                f"{_timestamp(cue['start'])} --> {_timestamp(cue['end'])}\n"
                f"{self._body(cue)}\n"
            )
        return "\n".join(blocks)

    def to_vtt(self) -> str:
        blocks = ["WEBVTT\n"]
        for cue in self.cues:
            blocks.append(
                f"{_timestamp(cue['start'], comma=False)} --> "
                f"{_timestamp(cue['end'], comma=False)}\n"
                f"{self._body(cue)}\n"
            )
        return "\n".join(blocks)

    def write(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = self.to_vtt() if target.suffix.lower() == ".vtt" else self.to_srt()
        target.write_text(text, encoding="utf-8")
