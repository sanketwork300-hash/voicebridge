"""SRT / WebVTT generation from translated segments.

Cue times are the segments' **source** timestamps -- the times the line was
spoken in the original -- never re-derived. Segments without timing
(``metadata.timestamps_available == False``, e.g. an S2ST engine that exposes
no alignment) are skipped rather than given invented times; the caller reports
that subtitles were unavailable.

Text is wrapped at ``max_line_chars`` into at most two lines, the usual
broadcast convention; overlong lines are left on the second line rather than
truncated.
"""

from __future__ import annotations

import textwrap

from voicebridge.core.types import TranslationSegment


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _wrap(text: str, width: int) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    lines = textwrap.wrap(text, width=width)
    if len(lines) <= 2:
        return "\n".join(lines)
    return lines[0] + "\n" + " ".join(lines[1:])


def timed_cues(segments: list[TranslationSegment], dual: bool = False,
               max_line_chars: int = 42) -> list[tuple[float, float, str]]:
    cues = []
    for seg in segments:
        if seg.metadata.get("timestamps_available") is False:
            continue
        if seg.start_time is None or seg.end_time is None or not (seg.translated_text or "").strip():
            continue
        end = seg.end_time if seg.end_time > seg.start_time else seg.start_time + 1.0
        body = _wrap(seg.translated_text or "", max_line_chars)
        if dual and seg.source_text:
            body += "\n" + seg.source_text.strip()
        cues.append((seg.start_time, end, body))
    cues.sort(key=lambda c: c[0])
    # Never let a cue overlap the next one; players stack overlapping cues.
    fixed = []
    for i, (a, b, text) in enumerate(cues):
        if i + 1 < len(cues) and b > cues[i + 1][0]:
            b = max(a + 0.3, cues[i + 1][0] - 0.01)
        fixed.append((a, b, text))
    return fixed


def render_srt(segments: list[TranslationSegment], dual: bool = False) -> str:
    blocks = [f"{i}\n{_ts(a, ',')} --> {_ts(b, ',')}\n{text}\n"
              for i, (a, b, text) in enumerate(timed_cues(segments, dual), start=1)]
    return "\n".join(blocks)


def render_vtt(segments: list[TranslationSegment], dual: bool = False) -> str:
    blocks = ["WEBVTT\n"] + [f"{_ts(a, '.')} --> {_ts(b, '.')}\n{text}\n"
                             for a, b, text in timed_cues(segments, dual)]
    return "\n".join(blocks)
