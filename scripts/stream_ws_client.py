"""Stream an audio file to a running gateway exactly like the browser page does.

    python scripts/stream_ws_client.py benchmark/data/media/ja_episode.wav \\
        --source ja --target en --engine cascade --out /tmp/live

Creates a realtime session (``POST /api/v1/realtime/sessions``), opens its
WebSocket, sends ``SESSION_CONFIG`` + ``AUDIO_START``, then 100 ms frames of
16 kHz mono PCM at wall-clock pace (as a microphone would deliver them), then
``AUDIO_STOP``. It records every event with its arrival time and writes:

* ``events.jsonl``   -- all events (audio payloads elided)
* ``dub.wav``        -- the TTS_AUDIO it received, placed at arrival time
* ``summary.json``   -- TTFT, TTFA, per-segment latency P50/P95, subtitles

Latency here is observed wall-clock latency on the machine running the
gateway, not a benchmark claim.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import urllib.request
from pathlib import Path

import numpy as np


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


async def run(args) -> dict:
    import soundfile as sf
    import websockets

    audio, rate = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert rate == 16000, "use a 16 kHz file (the browser page resamples to 16 kHz)"
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    session = _post(f"{args.url}/api/v1/realtime/sessions", {
        "source_language": args.source, "target_language": args.target,
        "engine": args.engine, "mode": "speech_and_subtitles", "profile": "balanced",
        **({"translation_quality": args.quality} if args.quality else {}),
        "input": {"type": "microphone", "sample_rate": 16000, "channels": 1},
        "output": {"subtitles": True, "audio": True}})
    ws_url = args.url.replace("http", "ws") + f"/api/v1/realtime/sessions/{session['session_id']}/stream"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    events, clips = [], []
    t0 = time.perf_counter()
    audio_seconds = len(pcm) / 32000

    async with websockets.connect(ws_url, max_size=2**24, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "SESSION_CONFIG", "input": {"sample_rate": 16000,
                                                                       "channels": 1}}))
        await ws.send(json.dumps({"type": "AUDIO_START"}))

        async def sender():
            step = 3200  # 100 ms
            for i in range(0, len(pcm), step):
                delay = t0 + i / 32000 - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                await ws.send(pcm[i:i + step])
            await asyncio.sleep(args.tail)
            await ws.send(json.dumps({"type": "AUDIO_STOP"}))

        async def receiver():
            async for raw in ws:
                ev = json.loads(raw)
                ev["_t"] = round(time.perf_counter() - t0, 3)
                if ev.get("event_type") == "TTS_AUDIO":
                    clips.append((ev["_t"], base64.b64decode(ev["audio"]), ev["sample_rate"]))
                    ev = {k: v for k, v in ev.items() if k != "audio"}
                events.append(ev)
                if ev.get("event_type") == "SESSION_ENDED":
                    return

        send_task = asyncio.create_task(sender())
        closed = None
        try:
            await asyncio.wait_for(receiver(), timeout=audio_seconds + args.tail + args.grace)
        except TimeoutError:
            closed = "client timeout waiting for SESSION_ENDED"
        except websockets.exceptions.ConnectionClosed as exc:
            closed = f"connection closed at {time.perf_counter() - t0:.1f}s: {exc}"
        send_task.cancel()
        await asyncio.gather(send_task, return_exceptions=True)

    with (out / "events.jsonl").open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    finals = [e for e in events if e.get("event_type") == "TRANSLATION_FINAL"]
    tts = [e for e in events if e.get("event_type") == "TTS_AUDIO"]
    lat = [e["_t"] - float(e.get("end", 0.0)) for e in finals]
    if clips:
        rate = clips[0][2]
        n = int((max(t for t, _, _ in clips) + 30) * rate)
        track = np.zeros(n, np.float32)
        cursor = 0
        for t, data, r in clips:
            x = np.frombuffer(data, "<i2").astype(np.float32) / 32768
            a = max(int(t * r), cursor)
            track[a:a + len(x)] += x[: n - a]
            cursor = a + len(x)
        sf.write(out / "dub.wav", track[:cursor], rate, subtype="PCM_16")
    summary = {
        "engine": args.engine, "audio_seconds": round(audio_seconds, 2), "closed": closed,
        "ttft": finals[0]["_t"] if finals else None,
        "ttfa": tts[0]["_t"] if tts else None,
        "segment_latency_p50": round(float(np.percentile(lat, 50)), 3) if lat else None,
        "segment_latency_p95": round(float(np.percentile(lat, 95)), 3) if lat else None,
        "segments": len(finals), "tts_clips": len(tts),
        "gate_rejections": sum(1 for e in events if e.get("event_type") == "SPEECH_EVENT"
                               and not e.get("should_transcribe")),
        "warnings": [e.get("message") for e in events if e.get("event_type") == "WARNING"],
        "errors": [e.get("message") for e in events if e.get("event_type") == "ERROR"],
        "subtitles": [{"t": e["_t"], "start": e.get("start"), "end": e.get("end"),
                       "source": e.get("source_text"), "translation": e.get("translated_text")}
                      for e in finals],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--source", default="ja")
    ap.add_argument("--target", default="en")
    ap.add_argument("--engine", default="cascade")
    ap.add_argument("--quality", default="", help="fast | balanced | high_quality")
    ap.add_argument("--tail", type=float, default=3.0, help="silence time before AUDIO_STOP")
    ap.add_argument("--grace", type=float, default=240.0, help="max wait for trailing output")
    ap.add_argument("--out", default="live_run")
    args = ap.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
