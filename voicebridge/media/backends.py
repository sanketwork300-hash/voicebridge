"""FFmpeg-backed implementations of the media interfaces in ``providers.base``.

The executor talks to :class:`MediaProcessor`, :class:`SubtitleRenderer` and
:class:`VideoRenderer`; these are the default (and only built-in) backends.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from voicebridge.media.muxer import mux_video
from voicebridge.media.normalizer import normalize_audio
from voicebridge.media.probe import probe
from voicebridge.media.renderer import convert_audio
from voicebridge.media.subtitles import render_srt, render_vtt
from voicebridge.providers.base import MediaProcessor, SubtitleRenderer, VideoRenderer


class FFmpegMediaProcessor(MediaProcessor):
    name = "ffmpeg"

    async def probe(self, path: str) -> Any:
        return await probe(path)

    async def extract_audio(self, path: str, output_path: str) -> Any:
        return await normalize_audio(path, output_path)

    async def encode_audio(self, wav_path: str, output_path: str) -> Any:
        return await convert_audio(wav_path, output_path)


class TextSubtitleRenderer(SubtitleRenderer):
    name = "srt_vtt"
    extensions = ("srt", "vtt")

    def render(self, segments: Sequence[Any], fmt: str, dual: bool = False) -> str:
        if fmt == "srt":
            return render_srt(list(segments), dual)
        if fmt == "vtt":
            return render_vtt(list(segments), dual)
        raise ValueError(f"unsupported subtitle format {fmt!r}")


class FFmpegVideoRenderer(VideoRenderer):
    name = "ffmpeg"

    async def render(self, video_path: str, output_path: str, audio_path: str | None = None,
                     subtitle_path: str | None = None, mode: str = "replace",
                     **options: Any) -> Any:
        return await mux_video(video_path, output_path, audio_path=audio_path,
                               subtitle_path=subtitle_path, mode=mode, **options)
