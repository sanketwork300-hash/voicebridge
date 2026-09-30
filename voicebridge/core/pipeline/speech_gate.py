"""Speech gate: decide which audio is translatable dialogue.

VAD answers "is there voice energy?"; it cannot tell dialogue from laughter,
screaming, singing or a crowd. The audio event classifier answers "what kind of
sound is this?" but is multi-label, so speech over a soundtrack scores high on
both *Speech* and *Music*. The gate combines them:

1. VAD says no speech            -> reject (SILENCE / top classifier category).
2. dialogue score >= dialogue_threshold -> DIALOGUE, transcribe.
3. top non-dialogue category >= non_speech_threshold -> reject as that category.
4. otherwise                     -> UNKNOWN, transcribe when ``transcribe_unknown``
                                    (conservative: ASR validation gets a second look).

Rule 2 precedes rule 3 on purpose: dialogue over music must be translated.

Two front-ends share :meth:`SpeechGate.combine`:

* realtime (:meth:`gate_chunk`) buffers ``window_seconds`` of audio, decides once
  per window and returns the audio to forward. Rejected windows are forwarded as
  **silence of the same length** instead of being dropped: a streaming ASR's
  timestamps are derived from the samples it has received, so dropping audio
  would shift every later subtitle, and its own VAD needs the silence to close
  an utterance. The window adds at most ``window_seconds`` of latency.
* file (:meth:`detect_regions`) runs Silero over the whole normalised signal,
  classifies each speech region in classifier-sized windows, and returns
  timestamped decisions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from voicebridge.core.types import AudioChunk, AudioEventType, SpeechDecision
from voicebridge.providers.base import AudioEventClassifier, VADProvider

SAMPLE_RATE = 16000


@dataclass
class SpeechGateConfig:
    enabled: bool = True
    dialogue_threshold: float = 0.35
    unknown_threshold: float = 0.35
    non_speech_threshold: float = 0.60
    transcribe_unknown: bool = True
    window_seconds: float = 0.6
    #: File mode: classifier window inside long VAD regions.
    classify_window_seconds: float = 2.0

    @classmethod
    def from_dict(cls, data: dict | None) -> SpeechGateConfig:
        data = data or {}
        fields = cls.__dataclass_fields__
        return cls(**{k: type(getattr(cls(), k))(v) for k, v in data.items() if k in fields})


class SpeechGate:
    def __init__(
        self,
        vad: VADProvider | None = None,
        classifier: AudioEventClassifier | None = None,
        config: SpeechGateConfig | None = None,
    ):
        self.vad = vad
        self.classifier = classifier
        self.config = config or SpeechGateConfig()
        self._pending: list[AudioChunk] = []

    # -- decision logic ------------------------------------------------------

    def combine(
        self,
        start: float,
        end: float,
        vad_confidence: float | None,
        vad_is_speech: bool | None,
        scores: dict[str, float] | None,
    ) -> SpeechDecision:
        cfg = self.config
        if vad_is_speech is False:
            event = AudioEventType.SILENCE.value
            confidence = 1.0 - (vad_confidence or 0.0)
            candidates = [k for k in (scores or {}) if k != AudioEventType.DIALOGUE.value]
            if candidates:
                top = max(candidates, key=lambda k: scores[k])
                if scores[top] >= cfg.non_speech_threshold:
                    event, confidence = top, scores[top]
            return SpeechDecision(False, event, confidence, start, end, "vad:no_speech", scores or {})

        if not scores:
            return SpeechDecision(
                True, AudioEventType.DIALOGUE.value if vad_is_speech else AudioEventType.UNKNOWN.value,
                vad_confidence or 0.0, start, end, "vad_only", {},
            )

        dialogue = scores.get(AudioEventType.DIALOGUE.value, 0.0)
        if dialogue >= cfg.dialogue_threshold:
            return SpeechDecision(True, AudioEventType.DIALOGUE.value, dialogue, start, end,
                                  "dialogue", scores)
        others = {k: v for k, v in scores.items()
                  if k not in (AudioEventType.DIALOGUE.value, AudioEventType.UNKNOWN.value)}
        top = max(others, key=lambda k: others[k]) if others else None
        if top is not None and others[top] >= cfg.non_speech_threshold:
            return SpeechDecision(False, top, others[top], start, end, "non_speech", scores)
        uncertain = 1.0 - max(dialogue, others[top] if top is not None else 0.0)
        return SpeechDecision(
            cfg.transcribe_unknown or dialogue >= cfg.unknown_threshold,
            AudioEventType.UNKNOWN.value, uncertain, start, end, "uncertain", scores,
        )

    async def decide(self, chunk: AudioChunk) -> SpeechDecision:
        """Decide one block of audio with no buffering."""
        if not self.config.enabled:
            return SpeechDecision(True, AudioEventType.UNKNOWN.value, 0.0, chunk.start, chunk.end,
                                  "gate_disabled")
        vad_conf = vad_speech = None
        if self.vad is not None:
            v = await self.vad.detect(chunk)
            vad_speech = v.should_transcribe
            vad_conf = v.confidence if v.should_transcribe else 1.0 - v.confidence
            if not vad_speech:
                return self.combine(chunk.start, chunk.end, vad_conf, False, None)
        scores = None
        if self.classifier is not None:
            c = await self.classifier.classify(chunk)
            scores = c.scores or {c.event_type: c.confidence}
        return self.combine(chunk.start, chunk.end, vad_conf, vad_speech, scores)

    # -- realtime ------------------------------------------------------------

    async def gate_chunk(self, chunk: AudioChunk) -> list[tuple[AudioChunk, SpeechDecision]]:
        """Buffer to a window; return ``(chunk_to_forward, decision)`` pairs."""
        if not self.config.enabled or (self.vad is None and self.classifier is None):
            return [(chunk, SpeechDecision(True, AudioEventType.UNKNOWN.value, 0.0,
                                           chunk.start, chunk.end, "gate_disabled"))]
        self._pending.append(chunk)
        if sum(c.duration for c in self._pending) < self.config.window_seconds:
            return []
        return await self.flush()

    async def flush(self) -> list[tuple[AudioChunk, SpeechDecision]]:
        if not self._pending:
            return []
        chunks, self._pending = self._pending, []
        window = AudioChunk(
            data=b"".join(c.data for c in chunks),
            fmt=chunks[0].fmt,
            start=chunks[0].start,
            sequence_id=chunks[0].sequence_id,
            session_id=chunks[0].session_id,
            created_at=chunks[0].created_at,
        )
        decision = await self.decide(window)
        out = []
        for c in chunks:
            forwarded = c if decision.should_transcribe else AudioChunk(
                data=bytes(len(c.data)), fmt=c.fmt, start=c.start,
                sequence_id=c.sequence_id, session_id=c.session_id, created_at=c.created_at,
            )
            out.append((forwarded, decision))
        return out

    # -- file ----------------------------------------------------------------

    async def detect_regions(self, audio: np.ndarray, offset: float = 0.0) -> list[SpeechDecision]:
        """Timestamped decisions for a float32 16 kHz signal.

        Without a VAD the whole signal is one candidate region. Each region is
        split into ``classify_window_seconds`` windows for the classifier, then
        adjacent windows with the same verdict are merged back.
        """
        duration = audio.shape[0] / SAMPLE_RATE
        if not self.config.enabled:
            return [SpeechDecision(True, AudioEventType.UNKNOWN.value, 0.0, offset,
                                   offset + duration, "gate_disabled")]
        if self.vad is not None and hasattr(self.vad, "speech_regions"):
            regions = await asyncio.to_thread(self.vad.speech_regions, audio, SAMPLE_RATE)
        else:
            regions = [{"start": 0.0, "end": duration, "confidence": None}]

        windows: list[tuple[float, float, float | None]] = []
        step = self.config.classify_window_seconds
        for region in regions:
            t = region["start"]
            while t < region["end"] - 1e-3:
                end = min(region["end"], t + step)
                if region["end"] - end < step / 2:  # fold a short tail into this window
                    end = region["end"]
                windows.append((t, end, region.get("confidence")))
                t = end

        scores_list: list[dict[str, float] | None] = [None] * len(windows)
        if self.classifier is not None and hasattr(self.classifier, "score_batch") and windows:
            batch = 8
            for i in range(0, len(windows), batch):
                part = windows[i : i + batch]
                arrays = [audio[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)] for a, b, _ in part]
                scored = await asyncio.to_thread(self.classifier.score_batch, arrays)
                scores_list[i : i + batch] = scored
        elif self.classifier is not None:
            for i, (a, b, _) in enumerate(windows):
                pcm = (np.clip(audio[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)], -1, 1)
                       * 32767).astype("<i2").tobytes()
                c = await self.classifier.classify(AudioChunk(data=pcm, start=a))
                scores_list[i] = c.scores or {c.event_type: c.confidence}

        decisions = [
            self.combine(offset + a, offset + b, conf,
                         True if self.vad is not None else None, scores)
            for (a, b, conf), scores in zip(windows, scores_list, strict=True)
        ]
        return merge_decisions(decisions)


def merge_decisions(decisions: list[SpeechDecision], max_gap: float = 0.05) -> list[SpeechDecision]:
    merged: list[SpeechDecision] = []
    for d in decisions:
        last = merged[-1] if merged else None
        if (
            last is not None
            and last.should_transcribe == d.should_transcribe
            and last.event_type == d.event_type
            and d.start_time - last.end_time <= max_gap
        ):
            span_a = last.end_time - last.start_time
            span_b = d.end_time - d.start_time
            last.confidence = (last.confidence * span_a + d.confidence * span_b) / max(
                1e-6, span_a + span_b
            )
            last.end_time = d.end_time
        else:
            merged.append(SpeechDecision(d.should_transcribe, d.event_type, d.confidence,
                                         d.start_time, d.end_time, d.reason, dict(d.scores)))
    return merged


def decision_for_span(decisions: list[SpeechDecision], start: float, end: float,
                      min_speech_fraction: float = 0.25) -> SpeechDecision | None:
    """The gate's verdict for an ASR segment spanning several gate windows.

    A segment of a few seconds overlaps many short windows, some of them
    pauses. Picking the single best-overlapping window let a silent window veto
    a real sentence (two utterances were lost that way in a live run). Instead:
    if windows the gate approved cover at least ``min_speech_fraction`` of the
    segment, the verdict is the strongest approving window; otherwise it is the
    non-speech verdict with the largest overlap. Streaming ASR timestamps are
    approximate, so the span is widened by 0.5 s on each side.
    """
    lo, hi = start - 0.5, end + 0.5
    span = max(1e-6, end - start)
    approved = 0.0
    best_ok: SpeechDecision | None = None
    best_no: SpeechDecision | None = None
    best_no_overlap = 0.0
    for d in decisions:
        overlap = min(hi, d.end_time) - max(lo, d.start_time)
        if overlap <= 0:
            continue
        if d.should_transcribe:
            approved += overlap
            if best_ok is None or d.confidence > best_ok.confidence:
                best_ok = d
        elif overlap > best_no_overlap:
            best_no, best_no_overlap = d, overlap
    if best_ok is not None and approved / span >= min_speech_fraction:
        return best_ok
    return best_no or best_ok
