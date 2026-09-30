"""Timeline audio rendering and audio encoding.

:class:`TimelineRenderer` builds the dubbed track by placing every synthesised
segment at its **source timestamp** on a track exactly as long as the source.
It does not concatenate clips: concatenation drifts further from the picture
with every line and destroys pauses.

Placement rules:

* a clip starts at ``segment.start_time``;
* if the previous clip is still playing, the new one waits for it (voices are
  never overlapped) and the delay is recorded as drift;
* drift is absorbed by the next natural pause, since every later clip is again
  placed at its own source time;
* clips are resampled to the track rate and peak-limited.

The track is a disk-backed ``numpy.memmap``, so memory use does not grow with
media length. An optional *bed* (the original audio, attenuated) can be mixed
under the dub for the "mix" output mode.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from voicebridge.core.audio.dsp import resample
from voicebridge.media.probe import ffmpeg_binary, input_args, run_tool


@dataclass
class Placement:
    segment_id: str
    requested_start: float
    actual_start: float
    duration: float

    @property
    def drift(self) -> float:
        return self.actual_start - self.requested_start


@dataclass
class RenderStats:
    placements: list[Placement] = field(default_factory=list)

    @property
    def max_drift(self) -> float:
        return max((p.drift for p in self.placements), default=0.0)

    @property
    def mean_drift(self) -> float:
        return float(np.mean([p.drift for p in self.placements])) if self.placements else 0.0


class TimelineRenderer:
    def __init__(self, duration: float, sample_rate: int = 24000, workdir: str | Path | None = None,
                 min_gap: float = 0.05):
        self.sample_rate = sample_rate
        self.duration = duration
        self.min_gap = min_gap
        self._dir = Path(workdir or tempfile.mkdtemp(prefix="vb-render-"))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._raw = self._dir / "timeline.f32"
        self._capacity = int((duration + 30.0) * sample_rate)
        self.track = np.memmap(self._raw, dtype=np.float32, mode="w+", shape=(self._capacity,))
        self._cursor = 0.0
        self.stats = RenderStats()

    def place(self, segment_id: str, audio: np.ndarray, rate: int, start: float) -> Placement:
        clip = resample(audio, rate, self.sample_rate)
        actual = max(start, self._cursor + (self.min_gap if self._cursor > 0 else 0.0))
        a = int(actual * self.sample_rate)
        b = min(self._capacity, a + clip.shape[0])
        if b > a:
            self.track[a:b] += clip[: b - a]
        duration = clip.shape[0] / self.sample_rate
        self._cursor = actual + duration
        placement = Placement(segment_id, start, actual, duration)
        self.stats.placements.append(placement)
        return placement

    @property
    def end(self) -> float:
        return max(self.duration, self._cursor)

    def write_wav(self, path: str | Path, bed: np.ndarray | None = None, bed_rate: int = 16000,
                  bed_gain: float = 0.25) -> Path:
        """Write the track (length = max(source, last clip end)) as 16-bit WAV."""
        import wave

        n = int(self.end * self.sample_rate)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        block = self.sample_rate * 60
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            for i in range(0, n, block):
                chunk = np.array(self.track[i : min(n, i + block)], dtype=np.float32)
                if bed is not None:
                    t0, t1 = i / self.sample_rate, min(n, i + block) / self.sample_rate
                    src = bed[int(t0 * bed_rate): int(t1 * bed_rate)]
                    if src.size:
                        up = resample(src, bed_rate, self.sample_rate)[: chunk.shape[0]]
                        chunk[: up.shape[0]] += bed_gain * up
                peak = np.max(np.abs(chunk)) if chunk.size else 0.0
                if peak > 0.99:
                    chunk *= 0.99 / peak
                wf.writeframes((chunk * 32767.0).astype("<i2").tobytes())
        return path

    def close(self) -> None:
        del self.track
        self._raw.unlink(missing_ok=True)


AUDIO_CODECS = {
    ".wav": ["-c:a", "pcm_s16le"],
    ".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
    ".m4a": ["-c:a", "aac", "-b:a", "192k"],
    ".flac": ["-c:a", "flac"],
    ".ogg": ["-c:a", "libvorbis", "-q:a", "5"],
}


async def convert_audio(input_path: str | Path, output_path: str | Path,
                        timeout: float = 1800.0) -> Path:
    output = Path(output_path)
    codec = AUDIO_CODECS.get(output.suffix.lower())
    if codec is None:
        raise ValueError(f"unsupported audio output format {output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    await run_tool([ffmpeg_binary(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                    *input_args(input_path), *codec, str(output)], timeout, "rendering")
    return output
