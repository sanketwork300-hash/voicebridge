"""Meta SeamlessStreaming as a direct speech-to-speech engine.

Runs Meta's own streaming agent in the ``seamless`` worker virtualenv
(:mod:`voicebridge.workers.seamless_streaming_worker`); nothing else in
VoiceBridge sees fairseq2 or SimulEval types. When this engine is selected no
Whisper, contextual translation or Qwen3-TTS runs -- the audio goes straight
to SeamlessStreaming, which is what makes a cascade-vs-direct comparison
meaningful.

Licence: the model weights are **CC-BY-NC-4.0**. The provider refuses to run
unless ``enabled: true`` is set explicitly. Weights are fetched only by an
explicit operator action: ``hf download facebook/seamless-streaming --include
"*.pt" --include "*.model"``. When that snapshot exists the provider points fairseq2's
asset cards at it (``FAIRSEQ2_USER_ASSET_DIR`` with ``@user`` cards); otherwise
fairseq2 downloads on the first load, which happens only after ``enabled: true``.

Timing: SeamlessStreaming does not align output to source timestamps. Segments
carry the **emission offset** (source seconds consumed when the text was
emitted) and ``metadata.timing = "emission_offset"``; nothing is invented.
"""

from __future__ import annotations

import base64
import os
import re
import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from voicebridge.core.types import AudioChunk, SpeechTranslationResult, TranslationSegment
from voicebridge.providers.base import ProviderUnavailable, S2STCapabilities, S2STProvider
from voicebridge.providers.registry import s2st_registry
from voicebridge.workers.client import ModelWorker, resolve_python

LICENSE = "CC-BY-NC-4.0 (model weights; review before commercial use)"
SOURCES = ["en", "ja", "ko", "hi", "zh", "es", "fr", "de", "pt", "ru", "it"]
TARGETS = ["en", "hi", "ja", "ko", "zh", "es", "fr", "de", "pt", "ru", "it"]
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")
_SENTENCE_END = re.compile(r"[.!?。！？]$")


def _hf_snapshot(repo: str) -> Path | None:
    base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snaps = base / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return None
    for snap in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if (snap / "seamless_streaming_unity.pt").exists():
            return snap
    return None


def _asset_overrides(snapshot: Path, target: Path) -> Path:
    """fairseq2 ``@user`` cards pointing the checkpoints at a local snapshot."""
    target.mkdir(parents=True, exist_ok=True)
    cards = {
        "seamless_streaming_unity": {
            "checkpoint": snapshot / "seamless_streaming_unity.pt",
            "char_tokenizer": snapshot / "spm_char_lang38_tc.model",
        },
        "seamless_streaming_monotonic_decoder": {
            "checkpoint": snapshot / "seamless_streaming_monotonic_decoder.pt"},
        "vocoder_v2": {"checkpoint": snapshot / "vocoder_v2.pt"},
    }
    docs = []
    for name, fields in cards.items():
        present = {k: v for k, v in fields.items() if v.exists()}
        if present:
            lines = [f"name: {name}@user"] + [f'{k}: "file://{v}"' for k, v in present.items()]
            docs.append("\n".join(lines))
    (target / "voicebridge_seamless.yaml").write_text("\n---\n".join(docs) + "\n")
    return target


class SeamlessStreamingProvider(S2STProvider):
    name = "seamless_streaming"

    def __init__(self, enabled: bool = False, model: str = "facebook/seamless-streaming",
                 device: str = "auto", dtype: str = "auto", python: str | None = None,
                 threads: int = 0, segment_ms: int = 320, decision_threshold: float = 0.5,
                 idle_unload_seconds: float | None = None, restart_per_file: bool = True,
                 **_: object):
        self.enabled = bool(enabled)
        self.restart_per_file = bool(restart_per_file)
        self.model = model
        env = {}
        snapshot = _hf_snapshot(model) if model else None
        if snapshot is not None:
            overrides = _asset_overrides(snapshot, Path(tempfile.gettempdir()) / "vb-fairseq2-assets")
            env["FAIRSEQ2_USER_ASSET_DIR"] = str(overrides)
        self.snapshot = snapshot
        self.worker = ModelWorker(
            "seamless-streaming", "seamless_streaming_worker.py",
            resolve_python(python, "VOICEBRIDGE_SEAMLESS_PYTHON", "seamless"),
            args=["--device", device, "--dtype", dtype, "--threads", str(int(threads or 0)),
                  "--segment-ms", str(int(segment_ms)),
                  "--decision-threshold", str(decision_threshold)],
            env=env, idle_unload_seconds=idle_unload_seconds, startup_timeout=120,
        )
        self.segment_ms = int(segment_ms)
        self._tmp = Path(tempfile.mkdtemp(prefix="vb-seamless-"))
        self.load_info: dict[str, Any] = {}

    @property
    def capabilities(self) -> S2STCapabilities:
        return S2STCapabilities(name=self.name, streaming=True, file=True,
                                source_languages=SOURCES, target_languages=TARGETS,
                                model_license=LICENSE, commercial_use=False)

    async def initialize(self, config: dict[str, object] | None = None) -> None:
        if config and "enabled" in config:
            self.enabled = bool(config["enabled"])
        if not self.enabled:
            raise ProviderUnavailable(
                "SeamlessStreaming is disabled. Set providers.s2st.seamless_streaming.enabled: "
                "true after reviewing the CC-BY-NC-4.0 model licence.")

    async def warmup(self) -> None:
        await self.initialize()
        self.load_info = await self.worker.call("load", tgt_lang="en")

    async def shutdown(self) -> None:
        await self.worker.stop()

    close = shutdown

    def _check_langs(self, source: str, target: str) -> None:
        if target not in TARGETS:
            raise ProviderUnavailable(f"SeamlessStreaming cannot output {target!r}")
        if source not in ("auto", *SOURCES):
            raise ProviderUnavailable(f"SeamlessStreaming does not list {source!r} as a source")

    async def translate_stream(self, audio_stream: AsyncIterator[AudioChunk], source_language: str,
                               target_language: str) -> AsyncIterator[SpeechTranslationResult]:
        """Push live audio; yield partial results as the agent emits them.

        Each yielded result carries new text in ``segments`` (emission-timed)
        and new speech in ``metadata['pcm16']`` at ``metadata['sample_rate']``.
        """
        await self.initialize()
        self._check_langs(source_language, target_language)
        stream_id = uuid.uuid4().hex
        await self.worker.call("open_stream", stream_id=stream_id, tgt_lang=target_language)
        buf = b""
        step = int(16000 * 2 * self.segment_ms / 1000)
        finished = False
        index = 0
        while not finished:
            try:
                chunk = await audio_stream.__anext__()
                buf += chunk.data
            except StopAsyncIteration:
                finished = True
            if len(buf) < step and not finished:
                continue
            send, buf = (buf, b"") if finished else (buf[:step], buf[step:])
            out = await self.worker.call("push", stream_id=stream_id, finished=finished,
                                         pcm16_b64=base64.b64encode(send).decode())
            segments = []
            for event in out.get("text") or []:
                index += 1
                segments.append(TranslationSegment(
                    id=f"ss-{index}", start_time=event["t"], end_time=event["t"],
                    translated_text=event["text"], target_language=target_language,
                    source_language=source_language,
                    metadata={"timing": "emission_offset", "timestamps_available": False}))
            pcm = base64.b64decode(out["speech_pcm16_b64"]) if out.get("speech_pcm16_b64") else b""
            if segments or pcm:
                yield SpeechTranslationResult(
                    audio_path=None, source_language=source_language,
                    target_language=target_language, segments=segments,
                    duration=out.get("t", 0.0),
                    metadata={"pcm16": pcm, "sample_rate": out.get("speech_sample_rate", 16000),
                              "compute_seconds": out.get("compute_seconds")})

    async def translate_file(self, audio_path: str, source_language: str, target_language: str,
                             workdir: str | None = None, progress=None, emit=None,
                             cancel=None) -> SpeechTranslationResult:
        await self.initialize()
        self._check_langs(source_language, target_language)
        out_dir = Path(workdir) if workdir else self._tmp
        out_path = out_dir / "seamless_dub.wav"
        try:
            result = await self.worker.call("translate_file", path=str(audio_path),
                                            tgt_lang=target_language, out_path=str(out_path))
        finally:
            if self.restart_per_file:
                # Observed: worker memory grew from one file to the next (10.5 ->
                # 12.5 GB) until the kernel OOM-killed it. A fresh process per
                # file costs one model load (~30 s) and keeps memory bounded.
                await self.worker.stop()
        if cancel is not None:
            cancel.check()
        segments = group_text_events(result.get("text") or [], source_language, target_language,
                                     result.get("source_seconds", 0.0))
        return SpeechTranslationResult(
            audio_path=result["path"], source_language=source_language,
            target_language=target_language, segments=segments,
            duration=result.get("source_seconds", 0.0),
            metadata={"engine": self.name, "timing": "emission_offset",
                      "timestamps_available": True,
                      "compute_seconds": result.get("compute_seconds"),
                      "time_to_first_text": result.get("time_to_first_text"),
                      "time_to_first_speech": result.get("time_to_first_speech"),
                      "speech_clips": len(result.get("speech_clips") or []),
                      "license": LICENSE},
        )


def group_text_events(events: list[dict[str, Any]], source: str, target: str,
                      total: float, max_gap: float = 1.5) -> list[TranslationSegment]:
    """Group streamed text pieces into subtitle-sized segments.

    A segment closes at sentence punctuation, when the agent reports the
    utterance finished, or after ``max_gap`` seconds without new text. Its
    start/end are the emission offsets of its first and last pieces (end is
    extended to the next segment's start, or +1.5 s, so cues stay readable).
    """
    groups: list[list[dict[str, Any]]] = []
    for ev in events:
        text = (ev.get("text") or "").strip()
        if not text:
            continue
        if groups and ev["t"] - groups[-1][-1]["t"] <= max_gap and not groups[-1][-1].get("_closed"):
            groups[-1].append(ev)
        else:
            groups.append([ev])
        if _SENTENCE_END.search(text) or ev.get("finished"):
            groups[-1][-1]["_closed"] = True
    segments = []
    for i, g in enumerate(groups, start=1):
        start = g[0]["t"]
        nxt = groups[i][0]["t"] if i < len(groups) else total + 1.5
        end = min(max(g[-1]["t"] + 1.5, start + 1.0), nxt)
        segments.append(TranslationSegment(
            id=f"ss-{i:05d}", start_time=round(start, 3), end_time=round(end, 3),
            translated_text=_SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(e["text"].strip() for e in g)),
            source_language=source,
            target_language=target,
            metadata={"timing": "emission_offset", "timestamps_available": True}))
    return segments


s2st_registry.register("seamless_streaming", SeamlessStreamingProvider)
