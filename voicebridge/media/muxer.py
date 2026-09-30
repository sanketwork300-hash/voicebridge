"""Combine the original video with translated audio and/or subtitles.

The video stream is always stream-copied (``-c:v copy``): no re-encode, so
resolution, frame rate, quality and duration are preserved exactly. Audio modes:

``replace``  dubbed track only.
``keep``     dubbed track first (default), original as a second selectable track.
``mix``      one track: dub over the original attenuated by ``original_gain``.

Subtitles are embedded as a soft track in the codec the container supports
(``mov_text`` for MP4/MOV, SubRip for MKV, WebVTT for WebM). Output container
follows the input: MP4/MOV -> .mp4, MKV -> .mkv, WebM -> .webm (WebM requires
Opus audio).
"""

from __future__ import annotations

from pathlib import Path

from voicebridge.media.probe import ffmpeg_binary, input_args, run_tool

MUX_MODES = ("replace", "keep", "mix")


def output_container(input_suffix: str) -> str:
    s = input_suffix.lower()
    return {".mkv": ".mkv", ".webm": ".webm"}.get(s, ".mp4")


def _audio_codec(container: str) -> list[str]:
    return ["-c:a", "libopus", "-b:a", "160k"] if container == ".webm" else ["-c:a", "aac",
                                                                           "-b:a", "192k"]


def _subtitle_codec(container: str) -> str:
    return {".mkv": "srt", ".webm": "webvtt"}.get(container, "mov_text")


async def mux_video(
    video_path: str | Path,
    output_path: str | Path,
    audio_path: str | Path | None = None,
    subtitle_path: str | Path | None = None,
    mode: str = "replace",
    original_gain: float = 0.25,
    audio_language: str | None = None,
    subtitle_language: str | None = None,
    timeout: float = 3600.0,
) -> Path:
    if mode not in MUX_MODES:
        raise ValueError(f"mode must be one of {MUX_MODES}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    container = output.suffix.lower()
    args = [ffmpeg_binary(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            *input_args(video_path)]
    inputs = 1
    audio_idx = sub_idx = None
    if audio_path is not None:
        args += input_args(audio_path)
        audio_idx = inputs
        inputs += 1
    if subtitle_path is not None:
        args += ["-protocol_whitelist", "file", "-f", "srt", "-i", str(subtitle_path)]
        sub_idx = inputs
        inputs += 1

    maps = ["-map", "0:v:0"]
    codecs = ["-c:v", "copy"]
    meta: list[str] = []
    if audio_idx is None:
        maps += ["-map", "0:a?"]
        codecs += ["-c:a", "copy"]
    elif mode == "mix":
        args += ["-filter_complex",
                 f"[0:a:0]volume={original_gain:.3f}[bed];"
                 f"[{audio_idx}:a:0][bed]amix=inputs=2:duration=first:normalize=0[dub]"]
        maps += ["-map", "[dub]"]
        codecs += _audio_codec(container)
    else:
        maps += ["-map", f"{audio_idx}:a:0"]
        if mode == "keep":
            maps += ["-map", "0:a:0"]
        codecs += _audio_codec(container)
        if mode == "keep":
            meta += ["-disposition:a:0", "default", "-disposition:a:1", "0",
                     "-metadata:s:a:1", "title=Original"]
    if audio_idx is not None and audio_language:
        meta += ["-metadata:s:a:0", f"language={_iso639_2(audio_language)}",
                 "-metadata:s:a:0", "title=Translated"]
    if sub_idx is not None:
        maps += ["-map", f"{sub_idx}:0"]
        codecs += ["-c:s", _subtitle_codec(container)]
        if subtitle_language:
            meta += ["-metadata:s:s:0", f"language={_iso639_2(subtitle_language)}"]
    extra = ["-movflags", "+faststart"] if container in (".mp4", ".mov") else []
    await run_tool(args + maps + codecs + meta + extra + [str(output)], timeout, "rendering")
    return output


def _iso639_2(code: str) -> str:
    return {"en": "eng", "ja": "jpn", "ko": "kor", "hi": "hin", "zh": "zho", "es": "spa",
            "fr": "fra", "de": "deu"}.get(code, code)
