"""Audio extraction: any container -> the pipeline's normalised WAV on disk.

FFmpeg decodes the first audio stream straight to 16 kHz mono PCM s16le. The
result lives on disk (about 115 MB per hour) and is read region by region, so
a two-hour film never has to be held in memory as a whole.
"""

from __future__ import annotations

from pathlib import Path

from voicebridge.media.probe import ffmpeg_binary, input_args, run_tool


async def extract_audio(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
    timeout: float = 3600.0,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    await run_tool(
        [ffmpeg_binary(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
         *input_args(input_path), "-map", "0:a:0", "-vn", "-sn", "-dn",
         "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", "-f", "wav", str(output)],
        timeout, "preprocessing",
    )
    return output
