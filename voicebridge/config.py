"""Application configuration.

Precedence, highest first: environment variables, the YAML config file, then
built-in defaults. Secrets are read from the environment only -- a token in a
config file tends to end up in a git repository.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATHS = [
    Path("config/voicebridge.yaml"),
    Path("/etc/voicebridge/config.yaml"),
]


@dataclass
class SecurityConfig:
    #: Shared bearer token. When None the server is open -- acceptable for
    #: localhost-only LOCAL mode, never for a network deployment.
    api_token: str | None = None
    #: Allowed HTTP origins for CORS.
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    #: Allowed WebSocket Origin headers. ``chrome-extension://*`` entries let a
    #: browser extension connect. Empty list means "accept any origin".
    allowed_ws_origins: list[str] = field(default_factory=list)
    max_sessions: int = 8
    max_session_seconds: float = 4 * 3600
    #: Inbound audio frames per second per session, before throttling kicks in.
    max_audio_frames_per_second: int = 100
    require_auth_for_localhost: bool = False


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    providers: dict[str, Any] = field(
        default_factory=lambda: {
            "asr": {"provider": "mock"},
            "translation": {"provider": "mock"},
            "tts": {"provider": "mock"},
        }
    )
    #: Presented at GET /v1/languages.
    languages: dict[str, Any] = field(default_factory=dict)

    @property
    def mock_mode(self) -> bool:
        return all(
            (self.providers.get(kind) or {}).get("provider", "mock") == "mock"
            for kind in ("asr", "translation", "tts")
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except ImportError:
        # PyYAML is not a hard dependency; JSON config still works everywhere.
        if path.suffix in (".json",):
            return json.loads(text)
        raise RuntimeError(
            f"reading {path} requires PyYAML (pip install pyyaml), or use a .json config"
        ) from None


def load_config(path: str | None = None) -> AppConfig:
    data: dict[str, Any] = {}
    candidates = [Path(path)] if path else list(DEFAULT_CONFIG_PATHS)
    env_path = os.environ.get("VOICEBRIDGE_CONFIG")
    if env_path and not path:
        candidates.insert(0, Path(env_path))
    for candidate in candidates:
        if candidate.is_file():
            data = _load_yaml(candidate)
            logger.info("loaded config from %s", candidate)
            break

    server = data.get("server") or {}
    security = data.get("security") or {}
    providers = data.get("providers") or {}

    config = AppConfig(
        server=ServerConfig(
            host=os.environ.get("VOICEBRIDGE_HOST", server.get("host", "127.0.0.1")),
            port=int(os.environ.get("VOICEBRIDGE_PORT", server.get("port", 8000))),
            log_level=os.environ.get("VOICEBRIDGE_LOG_LEVEL", server.get("log_level", "info")),
        ),
        security=SecurityConfig(
            api_token=os.environ.get("VOICEBRIDGE_API_TOKEN") or security.get("api_token"),
            cors_origins=_split(os.environ.get("VOICEBRIDGE_CORS_ORIGINS"))
            or security.get("cors_origins", ["*"]),
            allowed_ws_origins=_split(os.environ.get("VOICEBRIDGE_WS_ORIGINS"))
            or security.get("allowed_ws_origins", []),
            max_sessions=int(
                os.environ.get("VOICEBRIDGE_MAX_SESSIONS", security.get("max_sessions", 8))
            ),
            max_session_seconds=float(
                os.environ.get(
                    "VOICEBRIDGE_MAX_SESSION_SECONDS", security.get("max_session_seconds", 14400)
                )
            ),
            max_audio_frames_per_second=int(
                security.get("max_audio_frames_per_second", 100)
            ),
            require_auth_for_localhost=bool(security.get("require_auth_for_localhost", False)),
        ),
        providers=providers or AppConfig().providers,
        languages=data.get("languages") or {},
    )

    # Environment overrides for the provider choice make Docker profiles simple.
    for kind in ("asr", "translation", "tts"):
        env_key = f"VOICEBRIDGE_{kind.upper()}_PROVIDER"
        chosen = os.environ.get(env_key)
        if chosen:
            config.providers.setdefault(kind, {})
            config.providers[kind]["provider"] = chosen
    return config


def _split(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]
