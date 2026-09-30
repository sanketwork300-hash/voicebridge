"""REST API routers: ``/api/v1/files``, ``/api/v1/jobs``, ``/api/v1/models``.

Realtime session routes (``/api/v1/realtime/*`` and the legacy ``/v1/sessions``)
live in :mod:`voicebridge.apps.gateway.main` next to the WebSocket handler.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ApiContext:
    config: Any
    jobs: Any
    store: Any
    factory: Any
    authorize: Callable[[str | None, str | None], None]
