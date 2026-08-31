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
from voicebridge.apps.gateway.languages import catalogue
from voicebridge.apps.gateway.security import (
    RateLimiter,
    extract_token,
    origin_allowed,
    token_valid,
)
from voicebridge.config import AppConfig, load_config
from voicebridge.core.metrics.metrics import registry as metrics_registry
from voicebridge.core.session.config import PRESETS
from voicebridge.core.session.manager import SessionLimitExceeded, SessionManager
from voicebridge.protocols.websocket.handler import SessionSocket
from voicebridge.providers.registry import (
    asr_registry,
    load_builtin_providers,
    translation_registry,
    tts_registry,
)

logger = logging.getLogger(__name__)

WS_CLOSE_UNAUTHORIZED = 4401
WS_CLOSE_FORBIDDEN_ORIGIN = 4403
WS_CLOSE_NOT_FOUND = 4404
WS_CLOSE_RATE_LIMITED = 4429


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or load_config()
    load_builtin_providers()

    manager = SessionManager(
        provider_config=config.providers,
        metrics=metrics_registry,
        max_sessions=config.security.max_sessions,
        max_session_seconds=config.security.max_session_seconds,
    )
    limiter = RateLimiter(config.security.max_audio_frames_per_second)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _warn_on_insecure_bind(config)
        reaper = asyncio.create_task(_reaper(manager))
        try:
            yield
        finally:
            reaper.cancel()
            await asyncio.gather(reaper, return_exceptions=True)
            await manager.stop_all()

    app = FastAPI(
        title="VoiceBridge",
        version=__version__,
        description="Real-time speech-to-speech translation gateway",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.manager = manager

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
    async def ready() -> JSONResponse:
        """Readiness: can we actually build the configured providers?"""
        problems = []
        try:
            manager._provider("asr", asr_registry, "asr")
        except Exception as exc:
            problems.append(f"asr: {exc}")
        try:
            manager._provider("translation", translation_registry, "translation")
        except Exception as exc:
            problems.append(f"translation: {exc}")
        status = 200 if not problems else 503
        return JSONResponse(
            {"ready": not problems, "problems": problems}, status_code=status
        )

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(
            metrics_registry.prometheus(), media_type="text/plain; version=0.0.4"
        )

    @app.get("/v1/languages")
    async def languages() -> JSONResponse:
        return JSONResponse(catalogue())

    @app.get("/v1/providers")
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
                "presets": sorted(PRESETS),
            }
        )

    # -- sessions ----------------------------------------------------------

    @app.post("/v1/sessions", status_code=201)
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
    async def list_sessions(
        authorization: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> JSONResponse:
        _authorize(authorization, token)
        return JSONResponse({"sessions": manager.list()})

    @app.get("/v1/sessions/{session_id}")
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
            await socket.close()
            limiter.reset(session_id)
            # The session is intentionally *not* destroyed here: a dropped
            # socket is usually a transient network failure, and the client
            # reconnects to the same session id. The reaper removes it if the
            # client never comes back.

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        from voicebridge.apps.gateway.demo import DEMO_HTML

        return HTMLResponse(DEMO_HTML)

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
