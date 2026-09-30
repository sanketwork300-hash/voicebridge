"""VoiceBridge command line interface.

Subcommands:

``serve``      run the gateway
``translate``  translate an audio file offline (writes subtitles and/or a dub)
``devices``    list audio input devices, flagging likely system-audio monitors
``providers``  show registered providers, their availability and model licences
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from voicebridge import __version__
from voicebridge.core.types import EventType


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicebridge",
        description="Real-time speech-to-speech translation engine.",
    )
    parser.add_argument("--version", action="version", version=f"VoiceBridge {__version__}")
    parser.add_argument("--config", help="path to a YAML/JSON config file")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP + WebSocket gateway")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)

    translate = sub.add_parser("translate", help="translate an audio file")
    translate.add_argument("audio", help="path to a .wav/.pcm file (others need FFmpeg)")
    translate.add_argument("--source", default="auto", help="source language (default: auto)")
    translate.add_argument("--target", default="en", help="target language (default: en)")
    translate.add_argument("--mode", default="subtitles",
                           choices=["subtitles", "speech", "speech_and_subtitles"])
    translate.add_argument("--profile", default="balanced",
                           choices=["low_latency", "balanced", "accurate"])
    translate.add_argument("--subtitles", help="write subtitles here (.srt or .vtt)")
    translate.add_argument("--audio-out", help="write dubbed audio here (.wav)")
    translate.add_argument("--dual", action="store_true", help="include original text in subtitles")
    translate.add_argument("--realtime", action="store_true",
                           help="pace input at wall-clock speed to observe live latency")
    translate.add_argument("--json", action="store_true", help="emit events as JSON lines")

    tf = sub.add_parser("translate-file",
                        help="translate an audio/video file with the file pipeline (no server)")
    tf.add_argument("input", help="audio or video file")
    tf.add_argument("--source", default="auto")
    tf.add_argument("--target", default="en")
    tf.add_argument("--engine", default="cascade",
                    choices=["cascade", "seamless_streaming", "seamless_m4t_v2"])
    tf.add_argument("--outputs", default="", help="audio,video,srt,vtt,subtitled_video")
    tf.add_argument("--output-format", default="", help="wav | mp3 | m4a")
    tf.add_argument("--audio-mode", default="replace", choices=["replace", "keep", "mix"])
    tf.add_argument("--quality", default="", help="fast | balanced | high_quality")
    tf.add_argument("--no-speech", action="store_true", help="subtitles only, no TTS")
    tf.add_argument("--out", default=".", help="directory for the outputs")

    sub.add_parser("devices", help="list audio input devices")
    sub.add_parser("providers", help="list registered providers")
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def cmd_serve(args) -> int:
    import uvicorn

    from voicebridge.config import load_config

    config = load_config(args.config)
    host = args.host or config.server.host
    port = args.port or config.server.port
    print(f"VoiceBridge {__version__} on http://{host}:{port}  "
          f"({'mock mode' if config.mock_mode else 'live models'})")
    uvicorn.run("voicebridge.apps.gateway.main:app", host=host, port=port,
                log_level=config.server.log_level)
    return 0


async def _translate(args) -> int:
    from voicebridge.adapters.input.file import FileAudioInput
    from voicebridge.adapters.output.file import SubtitleWriter, WavFileOutput
    from voicebridge.config import load_config
    from voicebridge.core.session.manager import SessionManager
    from voicebridge.providers.registry import load_builtin_providers

    load_builtin_providers()
    config = load_config(args.config)

    manager = SessionManager(provider_config=config.providers)
    session = await manager.create(
        {
            "source_language": args.source,
            "target_language": args.target,
            "mode": args.mode,
            "profile": args.profile,
            "input": {"type": "file"},
            "output": {"subtitles": True, "dual_subtitles": args.dual},
        }
    )
    await manager.start(session.session_id)

    subtitles = SubtitleWriter(dual=args.dual)
    audio_out: WavFileOutput | None = None
    if args.audio_out:
        audio_out = WavFileOutput(args.audio_out)
        await audio_out.start()

    async def consume() -> None:
        async for event in session.pipeline.stream_events():
            if args.json:
                print(json.dumps(event.to_dict(), ensure_ascii=False), flush=True)
                continue
            if event.event_type is EventType.TRANSLATION_FINAL:
                p = event.payload
                subtitles.add(
                    p.get("start", 0.0), p.get("end", 0.0),
                    p.get("translated_text", ""), p.get("source_text", ""),
                    p.get("speaker"),
                )
                print(f"[{p.get('start', 0):7.2f}s] {p.get('source_text','')}")
                print(f"           -> {p.get('translated_text','')}")
            elif event.event_type is EventType.TTS_AUDIO and audio_out is not None:
                import base64

                from voicebridge.core.types import SynthesisedAudio

                await audio_out.write(
                    SynthesisedAudio(
                        audio=base64.b64decode(event.payload["audio"]),
                        sample_rate=event.payload["sample_rate"],
                        channels=event.payload["channels"],
                        duration=event.payload["duration"],
                        sequence_id=event.payload["translation_sequence"],
                    )
                )
            elif event.event_type is EventType.WARNING:
                print(f"warning: {event.payload.get('message')}", file=sys.stderr)
            elif event.event_type is EventType.ERROR:
                print(f"error: {event.payload.get('message')}", file=sys.stderr)

    consumer = asyncio.create_task(consume())

    source = FileAudioInput(args.audio, realtime=args.realtime)
    async for chunk in source.stream():
        await session.pipeline.push_audio(chunk)

    # Let the tail of the pipeline drain before tearing it down.
    await asyncio.sleep(1.0)
    await manager.stop(session.session_id)
    await consumer

    if args.subtitles:
        subtitles.write(args.subtitles)
        print(f"wrote {len(subtitles.cues)} cues to {args.subtitles}", file=sys.stderr)
    if audio_out is not None:
        await audio_out.stop()
        print(f"wrote dubbed audio to {args.audio_out}", file=sys.stderr)

    metrics = session.pipeline.metrics.summary()
    if metrics.get("end_to_end_p50"):
        print(
            f"latency p50={metrics['end_to_end_p50']:.2f}s "
            f"p95={metrics.get('end_to_end_p95') or float('nan'):.2f}s",
            file=sys.stderr,
        )
    return 0


def cmd_devices(_args) -> int:
    from voicebridge.adapters.input.microphone import list_devices
    from voicebridge.providers.base import ProviderUnavailable

    try:
        devices = list_devices()
    except ProviderUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1
    if not devices:
        print("no input devices found")
        return 0
    for device in devices:
        flag = "  <- likely system audio" if device["likely_system_audio"] else ""
        print(f"[{device['index']:2d}] {device['name']}  "
              f"({device['channels']}ch){flag}")
    return 0


async def cmd_translate_file(args) -> int:
    """Run one file job in-process: the same executor the HTTP API uses."""
    import shutil
    import tempfile
    from pathlib import Path

    from voicebridge.api.uploads import sanitize_filename
    from voicebridge.config import load_config
    from voicebridge.core.jobs.errors import PipelineError
    from voicebridge.core.jobs.executor import PipelineExecutor
    from voicebridge.core.jobs.manager import JobRecord
    from voicebridge.providers.factory import ProviderFactory
    from voicebridge.runtime import warn_if_cpu
    from voicebridge.storage import LocalArtifactStore

    config = load_config(args.config)
    warn_if_cpu(config.runtime)
    factory = ProviderFactory(config.providers, config.runtime)
    store = LocalArtifactStore(tempfile.mkdtemp(prefix="vb-cli-"))
    src = Path(args.input)
    key = f"uploads/input{src.suffix.lower()}"
    await store.save_file(key, src)
    job = JobRecord(job_id="cli", payload={
        "upload_key": key, "filename": sanitize_filename(src.name),
        "source_language": args.source, "target_language": args.target, "engine": args.engine,
        "outputs": [o for o in args.outputs.split(",") if o] or None,
        "output_format": args.output_format or None, "audio_mode": args.audio_mode,
        "translation_quality": args.quality or None, "synthesize": not args.no_speech})

    class Progress:
        async def progress(self, job, stage, fraction=0.0):
            print(f"\r{stage:18s} {fraction * 100:5.1f}%", end="", flush=True)

        async def pipeline_event(self, job, event_type, payload):
            if event_type == "SUBTITLE_CREATED":
                print(f"\n  {payload['start']:7.2f}s  {payload['text']}")

    try:
        outputs = await PipelineExecutor(factory, store, config.raw | {"media": config.media}).run(
            job, Progress())
    except PipelineError as exc:
        print(f"\nfailed: {exc.code}: {exc.message}")
        return 1
    finally:
        await factory.unload()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print()
    for name, artifact in sorted(outputs.items()):
        shutil.copy2(await store.get(artifact), out / name)
        print(out / name)
    print(f"processing {job.log.get('processing_seconds')} s, RTF {job.log.get('rtf')}")
    return 0


def cmd_providers(_args) -> int:
    from voicebridge.providers.registry import (
        asr_registry,
        load_builtin_providers,
        translation_registry,
        tts_registry,
    )

    load_builtin_providers()
    for kind, registry in (
        ("ASR", asr_registry),
        ("TRANSLATION", translation_registry),
        ("TTS", tts_registry),
    ):
        print(f"\n{kind}")
        for name in registry.names():
            try:
                caps = registry.create(name).capabilities
                licence = getattr(caps, "model_license", "n/a")
                commercial = getattr(caps, "commercial_use", None)
                note = {True: "commercial ok", False: "NON-COMMERCIAL", None: "see voice licence"}[
                    commercial
                ]
                print(f"  {name:16s} licence={licence:12s} {note}")
            except Exception as exc:
                print(f"  {name:16s} unavailable: {str(exc).splitlines()[0]}")
    return 0


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "translate":
        return asyncio.run(_translate(args))
    if args.command == "translate-file":
        return asyncio.run(cmd_translate_file(args))
    if args.command == "devices":
        return cmd_devices(args)
    if args.command == "providers":
        return cmd_providers(args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
