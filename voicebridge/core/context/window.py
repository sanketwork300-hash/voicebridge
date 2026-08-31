"""Bounded translation context.

Translation quality on conversational media depends heavily on prior context:
Japanese and Korean omit subjects freely, so "行きました" alone is "went" with no
subject at all, while the previous sentence usually establishes who.

The window is bounded on both segment count and character count. An unbounded
context window is a latency bug: every extra token is encoder work on the hot
path, and models degrade rather than improve past a few sentences of prompt.

Prior *translations* are kept only to steer terminology consistency. They are
never treated as authoritative output and are never re-emitted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class ContextEntry:
    source_text: str
    translated_text: str
    start: float = 0.0
    end: float = 0.0
    speaker: int | None = None


class ContextWindow:
    def __init__(self, max_segments: int = 4, max_characters: int = 600):
        if max_segments <= 0:
            raise ValueError("max_segments must be positive")
        self.max_segments = max_segments
        self.max_characters = max_characters
        self._entries: deque[ContextEntry] = deque()

    def add(self, entry: ContextEntry) -> None:
        self._entries.append(entry)
        self._trim()

    def _trim(self) -> None:
        while len(self._entries) > self.max_segments:
            self._entries.popleft()
        while self._entries and self.source_length > self.max_characters:
            self._entries.popleft()

    @property
    def source_length(self) -> int:
        return sum(len(e.source_text) for e in self._entries)

    @property
    def entries(self) -> list[ContextEntry]:
        return list(self._entries)

    def source_context(self) -> list[str]:
        return [e.source_text for e in self._entries]

    def target_context(self) -> list[str]:
        return [e.translated_text for e in self._entries if e.translated_text]

    def as_pairs(self) -> list[tuple[str, str]]:
        return [(e.source_text, e.translated_text) for e in self._entries]

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
