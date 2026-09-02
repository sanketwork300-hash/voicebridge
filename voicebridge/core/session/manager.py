"""Session lifecycle management.

Sessions are the unit of observability, authorisation and resource accounting.
Everything the gateway enforces -- quotas, maximum duration, concurrency limits
-- is enforced here so that a second protocol (gRPC, a CLI, a desktop host)
inherits the same rules instead of reimplementing them.
"""

from __future__ import annotations

import asyncio
import builtins
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from voicebridge.core.metrics.metrics import MetricsRegistry
from voicebridge.core.pipeline.pipeline import TranslationPipeline
from voicebridge.core.session.config import SessionConfig
from voicebridge.core.types import SessionStatus, new_id, now
from voicebridge.providers.base import ProviderUnavailable
from voicebridge.providers.registry import (
    ProviderSet,
    asr_registry,
    translation_registry,
    tts_registry,
)

logger = logging.getLogger(__name__)


@dataclass
class Session:
    session_id: str
    config: SessionConfig
    pipeline: TranslationPipeline
    status: SessionStatus = SessionStatus.CREATED
    started_at: float = field(default_factory=now)
    ended_at: float | None = None
    client: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "source_language": self.pipeline.source_language,
            "target_language": self.config.target_language,
            "input_type": self.config.input.type,
            "output_type": "browser_audio" if self.config.output.audio else "subtitles",
            "mode": self.config.mode.value,
            "profile": self.config.profile.value,
            "preset": self.config.preset,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status.value,
            "queues": self.pipeline.queue_stats(),
            "metrics": self.pipeline.metrics.summary(),
        }


class SessionLimitExceeded(RuntimeError):
    pass


class SessionManager:
    """Creates, tracks and tears down sessions."""

    def __init__(
        self,
        provider_config: dict[str, Any] | None = None,
        metrics: MetricsRegistry | None = None,
        max_sessions: int = 8,
        max_session_seconds: float = 4 * 3600,
    ):
        from voicebridge.core.metrics.metrics import registry as default_registry

        self.provider_config = provider_config or {}
        self.metrics = metrics or default_registry
        self.max_sessions = max_sessions
        self.max_session_seconds = max_session_seconds
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        #: Providers are process-wide: models are large and loading one per
        #: session would be both slow and a quick route to OOM.
        self._shared: dict[str, Any] = {}

    # -- providers ---------------------------------------------------------

    def _provider(self, kind: str, registry, config_key: str):
        cfg = dict(self.provider_config.get(config_key, {}))
        name = cfg.pop("provider", None) or cfg.pop("name", None) or "mock"
        cache_key = f"{kind}:{name}"
        existing = self._shared.get(cache_key)
        if existing is not None:
            return existing
        instance = registry.create(name, **cfg)
        self._shared[cache_key] = instance
        return instance

    async def warmup(self) -> list[str]:
        """Preload the configured providers before the first session.

        Model loading is the slow, GIL-heavy part of going live: WhisperLiveKit
        downloads and converts weights on first construction, which took long
        enough on a live WebSocket that the client's keepalive gave up before a
        single event was sent. Doing it once at startup means a session only
        ever pays the streaming cost. Returns the names of providers that could
        not be warmed so the caller can log them; failures are not fatal
        because mock providers and subtitle-only sessions must keep working.
        """
        failed: list[str] = []
        for kind, registry in (
            ("asr", asr_registry),
            ("translation", translation_registry),
            ("tts", tts_registry),
        ):
            try:
                provider = self._provider(kind, registry, kind)
                await provider.warmup()
            except Exception as exc:
                logger.warning("%s provider warmup failed: %s", kind, exc)
                failed.append(f"{kind}: {exc}")
        return failed

    def build_providers(self, config: SessionConfig) -> ProviderSet:
        asr = self._provider("asr", asr_registry, "asr")
        translation = self._provider("translation", translation_registry, "translation")
        tts = None
        if config.mode.wants_tts:
            try:
                tts = self._provider("tts", tts_registry, "tts")
            except ProviderUnavailable as exc:
                # Degrade rather than refuse: subtitles still work.
                logger.warning("TTS unavailable, continuing without it: %s", exc)
        return ProviderSet(asr=asr, translation=translation, tts=tts)

    # -- lifecycle ---------------------------------------------------------

    async def create(
        self, payload: dict[str, Any] | None = None, client: str | None = None
    ) -> Session:
        config = SessionConfig.from_payload(payload)
        async with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise SessionLimitExceeded(
                    f"server is at its session limit ({self.max_sessions})"
                )
            session_id = new_id()
            providers = self.build_providers(config)
            metrics = self.metrics.session(session_id)
            pipeline = TranslationPipeline(session_id, config, providers, metrics)
            session = Session(
                session_id=session_id,
                config=config,
                pipeline=pipeline,
                client=client,
            )
            self._sessions[session_id] = session
        return session

    async def start(self, session_id: str) -> Session:
        session = self.require(session_id)
        if session.status is SessionStatus.RUNNING:
            return session
        await session.pipeline.start()
        session.status = SessionStatus.RUNNING
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def require(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown session {session_id}")
        return session

    def list(self) -> builtins.list[dict[str, Any]]:
        return [s.describe() for s in self._sessions.values()]

    async def stop(self, session_id: str) -> Session | None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return None
        session.status = SessionStatus.STOPPING
        try:
            await session.pipeline.stop()
        except Exception as exc:  # pragma: no cover
            logger.warning("error stopping session %s: %s", session_id, exc)
        session.status = SessionStatus.ENDED
        session.ended_at = now()
        self.metrics.release(session_id)
        return session

    async def stop_all(self) -> None:
        for session_id in list(self._sessions):
            await self.stop(session_id)

    def expired(self) -> builtins.list[str]:
        """Sessions past the maximum duration, for the reaper task."""
        cutoff = time.time() - self.max_session_seconds
        return [s.session_id for s in self._sessions.values() if s.started_at < cutoff]

    async def reap_expired(self) -> int:
        expired = self.expired()
        for session_id in expired:
            logger.info("reaping expired session %s", session_id)
            await self.stop(session_id)
        return len(expired)

    @property
    def active(self) -> int:
        return len(self._sessions)
