"""VoiceBridge gateway: HTTP + WebSocket API.

Runs the whole pipeline in-process. For local and small self-hosted
deployments that is the right shape: splitting ASR, translation and TTS into
separate services adds a network hop to each stage of a latency-critical path,
and the requirement is explicit that microservices must not be introduced where
they cost latency without buying anything. ``deploy/`` documents how to split
the stages once a deployment is large enough to need independent scaling.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from voicebridge import __version__
from voicebridge.api import ApiContext
from voicebridge.api import files as files_api
from voicebridge.api import jobs as jobs_api
from voicebridge.api import models as models_api
from voicebridge.apps.gateway.languages import catalogue
from voicebridge.apps.gateway.security import (
    RateLimiter,
    extract_token,
    origin_allowed,
    token_valid,
)
from voicebridge.config import AppConfig, load_config
from voicebridge.core.jobs.executor import PipelineExecutor
from voicebridge.core.jobs.manager import JobManager
from voicebridge.core.metrics.metrics import registry as metrics_registry
from voicebridge.core.session.config import PRESETS
from voicebridge.core.session.manager import SessionLimitExceeded, SessionManager
from voicebridge.media import probe as media_probe
from voicebridge.protocols.websocket.handler import SessionSocket
from voicebridge.providers.factory import ProviderFactory
from voicebridge.providers.registry import (
    asr_registry,
    audio_event_registry,
    load_builtin_providers,
    s2st_registry,
    translation_registry,
    tts_registry,
    vad_registry,
)
from voicebridge.runtime import warn_if_cpu
from voicebridge.storage import LocalArtifactStore

logger = logging.getLogger(__name__)

WS_CLOSE_UNAUTHORIZED = 4401
WS_CLOSE_FORBIDDEN_ORIGIN = 4403
WS_CLOSE_NOT_FOUND = 4404
WS_CLOSE_RATE_LIMITED = 4429


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or load_config()
    load_builtin_providers()

    factory = ProviderFactory(config.providers, config.runtime)
    manager = SessionManager(
        provider_config=config.providers,
        metrics=metrics_registry,
        max_sessions=config.security.max_sessions,
        max_session_seconds=config.security.max_session_seconds,
        factory=factory,
    )
    media_cfg = config.media or {}
    media_probe.configure(media_cfg.get("ffmpeg_path"), media_cfg.get("ffprobe_path"))
    artifact_store = LocalArtifactStore((config.storage or {}).get("root", "storage"))
    job_manager = JobManager(
        PipelineExecutor(factory, artifact_store, config.raw | {"media": media_cfg}),
        workers=int(media_cfg.get("job_workers", 1)),
        timeout_seconds=float(media_cfg["processing_timeout_seconds"])
        if media_cfg.get("processing_timeout_seconds") else None,
        store=artifact_store,
    )
    limiter = RateLimiter(config.security.max_audio_frames_per_second)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _warn_on_insecure_bind(config)
        warn_if_cpu(config.runtime)
        if not media_probe.ffmpeg_available():
            logger.warning(media_probe.FFMPEG_REQUIRED + " File translation is disabled.")
        if not config.mock_mode:
            # Load real models before accepting sessions; see
            # SessionManager.warmup for why this cannot wait for the first one.
            logger.info("warming up providers: %s", {
                kind: (config.providers.get(kind) or {}).get("provider", "mock")
                for kind in ("asr", "translation", "tts")
            })
            started = asyncio.get_running_loop().time()
            failed = await manager.warmup()
            logger.info(
                "provider warmup finished in %.1fs%s",
                asyncio.get_running_loop().time() - started,
                f"; unavailable: {failed}" if failed else "",
            )
        await job_manager.start()
        reaper = asyncio.create_task(_reaper(manager))
        retention = asyncio.create_task(_job_retention(
            job_manager, float(media_cfg.get("retention_hours", 72)) * 3600))
        try:
            yield
        finally:
            reaper.cancel()
            retention.cancel()
            await asyncio.gather(reaper, retention, return_exceptions=True)
            await job_manager.stop()
            await manager.stop_all()
            await factory.unload()

    app = FastAPI(
        title="VoiceBridge",
        version=__version__,
        description="Real-time speech-to-speech translation gateway",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.manager = manager
    app.state.jobs = job_manager
    app.state.artifacts = artifact_store
    app.state.factory = factory

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.security.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _authorize(authorization: str | None, token: str | None) -> None:
        if not token_valid(config.security, extract_token(authorization, token)):
            raise HTTPException(status_code=401, detail="invalid or missing API token")

    # -- meta --------------------------------------------------------------

    @app.get("/health")
    @app.get("/api/v1/health")
    @app.get("/api/v1/health/live")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "version": __version__,
                "mock_mode": config.mock_mode,
                "sessions_active": manager.active,
            }
        )

    @app.get("/ready")
    @app.get("/api/v1/health/ready")
    async def ready() -> JSONResponse:
        """Readiness: can we actually build the configured providers?"""
        problems = []
        for kind in ("asr", "translation"):
            try:
                factory.get(kind)
            except Exception as exc:
                problems.append(f"{kind}: {exc}")
        status = 200 if not problems else 503
        return JSONResponse(
            {"ready": not problems, "problems": problems}, status_code=status
        )

    @app.get("/metrics")
    @app.get("/api/v1/health/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(
            metrics_registry.prometheus(), media_type="text/plain; version=0.0.4"
        )

    @app.get("/v1/languages")
    @app.get("/api/v1/models/languages")
    async def languages() -> JSONResponse:
        return JSONResponse(catalogue())

    @app.get("/v1/providers")
    @app.get("/api/v1/models/providers")
    async def providers() -> JSONResponse:
        def describe(registry, kind: str):
            configured = (config.providers.get(kind) or {}).get("provider", "mock")
            out = []
            for name in registry.names():
                entry: dict[str, Any] = {"name": name, "active": name == configured}
                try:
                    instance = registry.create(name)
                    caps = instance.capabilities
                    entry["capabilities"] = {
                        k: v
                        for k, v in vars(caps).items()
                        if not k.startswith("_")
                    }
                except Exception as exc:
                    entry["available"] = False
                    entry["reason"] = str(exc).split("\n")[0]
                else:
                    entry["available"] = True
                out.append(entry)
            return out

        return JSONResponse(
            {
                "asr": describe(asr_registry, "asr"),
                "translation": describe(translation_registry, "translation"),
                "tts": describe(tts_registry, "tts"),
                "vad": describe(vad_registry, "vad"),
                "audio_event": describe(audio_event_registry, "audio_event"),
                "s2st": describe(s2st_registry, "s2st"),
                "presets": sorted(PRESETS),
            }
        )

    api_ctx = ApiContext(config=config, jobs=job_manager, store=artifact_store,
                         factory=factory, authorize=_authorize)
    for module in (files_api, jobs_api, models_api):
        app.include_router(module.build_router(api_ctx))

    # -- sessions ----------------------------------------------------------

    @app.post("/v1/sessions", status_code=201)
    @app.post("/api/v1/realtime/sessions", status_code=201)
    async def create_session(
        request: Request,
        authorization: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> JSONResponse:
        _authorize(authorization, token)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if isinstance(payload, dict) and not (payload.get("engine")
                                               or payload.get("translation_engine")):
            payload["engine"] = config.default_engine
        try:
            session = await manager.create(payload, client=request.client.host if request.client else None)
        except SessionLimitExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from None
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return JSONResponse(
            {
                **session.describe(),
                "stream_url": f"/v1/sessions/{session.session_id}/stream",
            },
            status_code=201,
        )

    @app.get("/v1/sessions")
    @app.get("/api/v1/realtime/sessions")
    async def list_sessions(
        authorization: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> JSONResponse:
        _authorize(authorization, token)
        return JSONResponse({"sessions": manager.list()})

    @app.get("/v1/sessions/{session_id}")
    @app.get("/api/v1/realtime/sessions/{session_id}")
    async def get_session(
        session_id: str,
        authorization: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> JSONResponse:
        _authorize(authorization, token)
        session = manager.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return JSONResponse(session.describe())

    @app.delete("/v1/sessions/{session_id}")
    @app.delete("/api/v1/realtime/sessions/{session_id}")
    async def delete_session(
        session_id: str,
        authorization: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> JSONResponse:
        _authorize(authorization, token)
        session = await manager.stop(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return JSONResponse(session.describe())

    # -- streaming ---------------------------------------------------------

    @app.websocket("/v1/sessions/{session_id}/stream")
    @app.websocket("/api/v1/realtime/sessions/{session_id}/stream")
    async def stream(websocket: WebSocket, session_id: str) -> None:
        supplied = extract_token(
            websocket.headers.get("authorization"),
            websocket.query_params.get("token"),
        )
        if not token_valid(config.security, supplied):
            await websocket.close(code=WS_CLOSE_UNAUTHORIZED, reason="invalid or missing API token")
            return
        if not origin_allowed(config.security, websocket.headers.get("origin")):
            await websocket.close(code=WS_CLOSE_FORBIDDEN_ORIGIN, reason="origin not allowed")
            return

        session = manager.get(session_id)
        if session is None:
            await websocket.close(code=WS_CLOSE_NOT_FOUND, reason="unknown session")
            return

        await websocket.accept()
        await manager.start(session_id)

        async def send_json(payload: dict) -> None:
            await websocket.send_json(payload)

        socket = SessionSocket(session, send_json)
        await socket.start()

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (text := message.get("text")) is not None:
                    keep_open = await socket.handle_text(text)
                    if not keep_open:
                        break
                elif (data := message.get("bytes")) is not None:
                    if not limiter.allow(session_id):
                        # Throttle rather than disconnect: a client briefly
                        # sending too fast should recover, not lose its session.
                        continue
                    await socket.handle_binary(data)
        except WebSocketDisconnect:
            logger.info("client disconnected from session %s", session_id)
        except Exception:
            logger.exception("stream error for session %s", session_id)
        finally:
            if socket.stop_requested:
                # Graceful end: finish committed translations/speech, send
                # them, then close. (A dropped socket keeps the session.)
                try:
                    await manager.stop(session_id)
                    await socket.wait_forwarded(timeout=10)
                    await websocket.close()
                except Exception:
                    logger.debug("error during graceful stream end", exc_info=True)
            await socket.close()
            limiter.reset(session_id)
            # The session is intentionally *not* destroyed here: a dropped
            # socket is usually a transient network failure, and the client
            # reconnects to the same session id. The reaper removes it if the
            # client never comes back.

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        from voicebridge.apps.gateway.demo import load_ui

        return HTMLResponse(load_ui())

    return app


async def _reaper(manager: SessionManager, interval: float = 60.0) -> None:
    while True:
        try:
            await asyncio.sleep(interval)
            reaped = await manager.reap_expired()
            if reaped:
                logger.info("reaped %d expired session(s)", reaped)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover
            logger.exception("session reaper failed")


async def _job_retention(jobs: JobManager, max_age: float, interval: float = 600.0) -> None:
    """Delete finished jobs (uploads, outputs, records) older than ``max_age``."""
    while True:
        try:
            await asyncio.sleep(interval)
            removed = await jobs.purge_older_than(max_age)
            if removed:
                logger.info("retention: removed %d finished job(s)", removed)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover
            logger.exception("job retention failed")


def _warn_on_insecure_bind(config: AppConfig) -> None:
    public = config.server.host not in ("127.0.0.1", "localhost", "::1")
    if public and not config.security.api_token:
        logger.warning(
            "VoiceBridge is bound to %s with no API token configured. Anyone who "
            "can reach this port can open sessions and stream audio. Set "
            "VOICEBRIDGE_API_TOKEN before exposing it.",
            config.server.host,
        )
    if public and not config.security.allowed_ws_origins:
        logger.warning(
            "No allowed_ws_origins configured on a non-localhost bind; any web "
            "page will be able to open a WebSocket session."
        )


app = create_app()


def main() -> None:
    import uvicorn

    config = load_config()
    logging.basicConfig(
        level=getattr(logging, config.server.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        "voicebridge.apps.gateway.main:app",
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
    )


if __name__ == "__main__":
    main()
