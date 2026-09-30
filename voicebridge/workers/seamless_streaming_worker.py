"""Meta SeamlessStreaming worker (runs in the ``seamless`` worker virtualenv:
Python 3.10, torch 2.1.1, fairseq2 0.2.1, seamless_communication, simuleval).

API used (verified against seamless_communication @ main and simuleval 1.1.4):

* ``SeamlessStreamingS2STJointVADAgent`` (``streaming/agents/seamless_streaming_s2st.py``)
  is a SimulEval ``TreeAgentPipeline``: Silero VAD -> feature extractor ->
  w2v-BERT encoder -> monotonic text decoder -> {detokenizer (text),
  NAR unit decoder -> vocoder (speech)}. It is the agent Meta's streaming demo
  uses; the VAD resets decoder state between utterances, which is what makes
  unbounded input work.
* Built with ``Agent.add_args(parser)``; ``Agent.from_args(args)``; per-stream
  state from ``agent.build_states()``; audio fed with
  ``agent.pushpop(SpeechSegment(content, sample_rate, finished, tgt_lang), states)``,
  which returns a list of output segments (``TextSegment`` / ``SpeechSegment``).
* Model configuration mirrors ``cli/streaming/evaluate.py`` (``source_segment_size=320``,
  ``min_starting_wait_w2vbert=192``, ``decision_threshold=0.5``,
  ``min_unit_chunk_size=50``, ``no_early_stop``, ``max_len_a=0``, ``max_len_b=100``).
* Weights are resolved by fairseq2 asset cards (``seamless_streaming_unity``,
  ``seamless_streaming_monotonic_decoder``, ``vocoder_v2``) and downloaded by
  fairseq2 on first load. Licence: CC-BY-NC-4.0.

Timing: SeamlessStreaming does not align its output to source timestamps. What
it does expose -- and what this worker records -- is the **emission offset**:
how many seconds of source audio had been consumed when each piece of text or
speech was produced. File mode places output at those offsets (what a live
listener would hear) and labels the timing as ``emission_offset``.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol  # noqa: E402

LANG3 = {"en": "eng", "ja": "jpn", "ko": "kor", "hi": "hin", "zh": "cmn", "es": "spa",
         "fr": "fra", "de": "deu", "pt": "por", "ru": "rus", "it": "ita"}
STATE: dict = {"agent": None, "streams": {}}
#: Silence appended at end of input so the agent's VAD closes the last utterance.
FLUSH_SECONDS = 3.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--segment-ms", type=int, default=320)
    ap.add_argument("--decision-threshold", type=float, default=0.5)
    opts = ap.parse_args()

    def _agent(tgt: str):
        if STATE["agent"] is not None:
            return STATE["agent"]
        import torch
        from seamless_communication.streaming.agents.seamless_streaming_s2st import (
            SeamlessStreamingS2STJointVADAgent,
        )

        if opts.threads:
            torch.set_num_threads(opts.threads)
        device = opts.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = opts.dtype if opts.dtype != "auto" else ("fp16" if device != "cpu" else "fp32")
        parser = argparse.ArgumentParser(conflict_handler="resolve")
        # SimulEval's general parser normally supplies these two; the agents
        # read args.device / args.fp16 in load_model.
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--fp16", action="store_true", default=False)
        SeamlessStreamingS2STJointVADAgent.add_args(parser)
        args, _ = parser.parse_known_args([
            "--task", "s2st", "--tgt-lang", tgt, "--device", device, "--dtype", dtype,
            "--source-segment-size", str(opts.segment_ms),
            "--min-starting-wait-w2vbert", "192",
            "--decision-threshold", str(opts.decision_threshold),
            "--min-unit-chunk-size", "50", "--no-early-stop",
            "--max-len-a", "0", "--max-len-b", "100",
        ])
        started = time.time()
        STATE["agent"] = SeamlessStreamingS2STJointVADAgent.from_args(args)
        STATE.update(device=device, dtype=dtype, load_seconds=time.time() - started)
        return STATE["agent"]

    def _collect(outputs, t_source: float, events: list, speech: list) -> None:
        segs = outputs if isinstance(outputs, list) else [outputs]
        for seg in segs:
            if seg is None or getattr(seg, "is_empty", False):
                continue
            if seg.data_type == "text" and seg.content:
                events.append({"type": "text", "text": str(seg.content), "t": t_source,
                               "finished": bool(seg.finished)})
            elif seg.data_type == "speech" and seg.content:
                speech.append((t_source, list(seg.content), int(seg.sample_rate)))

    def load(tgt_lang: str = "en") -> dict:
        _agent(LANG3.get(tgt_lang, tgt_lang))
        return {"device": STATE["device"], "dtype": STATE["dtype"],
                "load_seconds": round(STATE["load_seconds"], 2),
                "models": ["seamless_streaming_unity", "seamless_streaming_monotonic_decoder",
                           "vocoder_v2"]}

    def open_stream(stream_id: str, tgt_lang: str) -> dict:
        agent = _agent(LANG3.get(tgt_lang, tgt_lang))
        STATE["streams"][stream_id] = {"states": agent.build_states(),
                                       "tgt": LANG3.get(tgt_lang, tgt_lang), "t": 0.0}
        return {"ok": True}

    def push(stream_id: str, pcm16_b64: str, finished: bool = False) -> dict:
        import numpy as np
        from simuleval.data.segments import SpeechSegment

        stream = STATE["streams"][stream_id]
        audio = np.frombuffer(base64.b64decode(pcm16_b64), dtype="<i2").astype(np.float32) / 32768
        stream["t"] += len(audio) / 16000
        events: list = []
        speech: list = []
        started = time.time()
        step = int(16000 * opts.segment_ms / 1000)
        blocks = [audio[i:i + step] for i in range(0, len(audio), step)] or []
        if finished:  # the VAD closes an utterance on silence, not on a flag
            blocks += [np.zeros(step, np.float32)] * int(FLUSH_SECONDS * 1000 / opts.segment_ms)
        for block in blocks:
            if len(block) < step:
                block = np.pad(block, (0, step - len(block)))
            out = STATE["agent"].pushpop(
                SpeechSegment(content=block.astype(np.float32), sample_rate=16000,
                              finished=False, tgt_lang=stream["tgt"]), stream["states"])
            _collect(out, stream["t"], events, speech)
        result = {"text": events, "compute_seconds": round(time.time() - started, 4),
                  "t": stream["t"]}
        if speech:
            wav = np.concatenate([np.asarray(s[1], dtype=np.float32) for s in speech])
            result["speech_pcm16_b64"] = base64.b64encode(
                (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()).decode()
            result["speech_sample_rate"] = speech[0][2]
        if finished:
            STATE["streams"].pop(stream_id, None)
        return result

    def translate_file(path: str, tgt_lang: str, out_path: str) -> dict:
        """Stream a whole file through the agent in ``segment_ms`` steps."""
        import numpy as np
        import soundfile as sf
        from simuleval.data.segments import SpeechSegment

        agent = _agent(LANG3.get(tgt_lang, tgt_lang))
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert sr == 16000, "expects 16 kHz input"
        states = agent.build_states()
        step = int(16000 * opts.segment_ms / 1000)
        events: list = []
        speech: list = []
        started = time.time()
        first_text = first_speech = None
        total_in = len(audio)
        audio = np.concatenate([audio, np.zeros(int(FLUSH_SECONDS * 16000), np.float32)])
        for i in range(0, len(audio), step):
            block = audio[i:i + step]
            if len(block) < step:
                block = np.pad(block, (0, step - len(block)))
            out = agent.pushpop(SpeechSegment(content=block.astype(np.float32),
                                              sample_rate=16000, finished=False,
                                              tgt_lang=LANG3.get(tgt_lang, tgt_lang)), states)
            n_text, n_speech = len(events), len(speech)
            _collect(out, (i + len(block)) / 16000, events, speech)
            now = time.time() - started
            if first_text is None and len(events) > n_text:
                first_text = now
            if first_speech is None and len(speech) > n_speech:
                first_speech = now
        # Place speech at emission offsets; never overlap (a listener hears it in order).
        rate = speech[0][2] if speech else 16000
        cursor = 0.0
        placed = []
        total = total_in / 16000
        for t, samples, r in speech:
            start = max(t, cursor)
            placed.append((start, np.asarray(samples, dtype=np.float32)))
            cursor = start + len(samples) / r
        n = int(max(total, cursor) * rate) + 1
        track = np.zeros(n, dtype=np.float32)
        for start, clip in placed:
            a = int(start * rate)
            track[a:a + len(clip)] += clip[: max(0, n - a)]
        sf.write(out_path, np.clip(track, -1, 1), rate, subtype="PCM_16")
        return {"path": out_path, "sample_rate": rate, "text": events,
                "speech_clips": [{"t": p[0], "duration": len(p[1]) / rate} for p in placed],
                "compute_seconds": round(time.time() - started, 3),
                "time_to_first_text": first_text, "time_to_first_speech": first_speech,
                "source_seconds": total}

    _protocol.serve({"load": load, "open_stream": open_stream, "push": push,
                     "translate_file": translate_file}, {"worker": "seamless_streaming"})


if __name__ == "__main__":
    main()
