"""Authentication, origin validation and rate limiting for the gateway."""

from __future__ import annotations

import fnmatch
import hmac
import time
from collections import defaultdict, deque

from voicebridge.config import SecurityConfig

LOCALHOST_HOSTS = {"127.0.0.1", "::1", "localhost"}


def token_valid(config: SecurityConfig, candidate: str | None) -> bool:
    """Constant-time token comparison. No configured token means open access."""
    if not config.api_token:
        return True
    if not candidate:
        return False
    return hmac.compare_digest(candidate, config.api_token)


def extract_token(authorization: str | None, query_token: str | None) -> str | None:
    """Read a bearer token from the header, falling back to a query parameter.

    The query fallback exists because the browser WebSocket API cannot set
    request headers. It is why ``docs/security.md`` recommends short-lived
    tokens for remote mode: a URL token is more exposed (proxy logs, history)
    than a header one.
    """
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return query_token


def origin_allowed(config: SecurityConfig, origin: str | None) -> bool:
    """Validate a WebSocket Origin header against the allow-list.

    An empty allow-list accepts anything, which is correct for a localhost-only
    deployment and wrong for a public one; ``docs/security.md`` says so and the
    gateway logs a warning at startup when a public bind has no allow-list.
    """
    patterns = config.allowed_ws_origins
    if not patterns:
        return True
    if origin is None:
        # Non-browser clients (CLI, desktop app) send no Origin. The token is
        # what authenticates them.
        return True
    return any(fnmatch.fnmatch(origin, pattern) for pattern in patterns)


class RateLimiter:
    """Fixed-window rate limiter keyed by client identity."""

    def __init__(self, limit: int, window_seconds: float = 1.0):
        self.limit = limit
        self.window = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        bucket = self._events[key]
        cutoff = now - self.window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return False
        bucket.append(now)
        return True

    def reset(self, key: str) -> None:
        self._events.pop(key, None)
