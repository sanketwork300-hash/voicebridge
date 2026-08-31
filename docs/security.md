# Security

## Threat model

VoiceBridge processes live audio, which is among the most sensitive data a user
can hand to a service. Two deployment modes with very different exposure:

| Mode | Exposure |
| --- | --- |
| **LOCAL** — gateway on `127.0.0.1`, models local | Nothing leaves the machine. Default. |
| **REMOTE** — gateway reachable over a network | Audio crosses the network; the gateway is attackable. |

The dangerous case is a LOCAL configuration accidentally exposed: bound to
`0.0.0.0` without a token, any host on the network can open sessions and stream
audio through your GPU. The gateway **logs a warning at startup** when it binds
publicly without a token or origin allow-list.

## Checklist for network deployments

1. **Set `VOICEBRIDGE_API_TOKEN`.** Without one, the server is open.
2. **Set `VOICEBRIDGE_WS_ORIGINS`.** An empty allow-list accepts any origin,
   which lets any web page open a session against your gateway.
3. **Terminate TLS.** Browsers require `wss://` from an `https://` page.
   `deploy/caddy/Caddyfile` does this with automatic certificates.
4. **Narrow CORS.** `cors_origins: ["*"]` is a local-development default.
5. **Set limits.** `max_sessions`, `max_session_seconds`,
   `max_audio_frames_per_second` bound what one client can consume.
6. **Do not put a token in extension code.** Extension source is readable by
   anyone who installs it.

## Controls implemented

| Control | Where |
| --- | --- |
| Constant-time token comparison (`hmac.compare_digest`) | `apps/gateway/security.py` |
| WebSocket origin allow-list with glob patterns | `apps/gateway/security.py` |
| Per-session audio frame rate limiting | `apps/gateway/main.py` |
| Max audio frame 256 KiB, control frame 64 KiB | `protocols/websocket/messages.py` |
| Sample alignment validation | `protocols/websocket/messages.py` |
| Bounded queues at every stage | `core/streaming/queue.py` |
| Session count and duration limits, background reaper | `core/session/manager.py` |
| Container runs as non-root uid 10001 | `deploy/docker/Dockerfile` |

### Rate limiting throttles, it does not disconnect

A client briefly sending too fast should recover, not lose its session. Excess
frames are dropped; the session survives.

### Token in a query parameter

The browser WebSocket API cannot set headers, so `?token=` is supported. It is
more exposed than a header — proxy logs, browser history — so for remote mode
prefer **short-lived tokens** minted per session by your own auth layer rather
than one long-lived shared secret.

## Input validation

Both sides of the socket are treated as untrusted:

* Control frames must be JSON objects with a known `type`; anything else yields
  a recoverable `ERROR` and the socket stays usable.
* Audio frames are size-checked and alignment-checked. A misaligned frame is
  rejected, not padded — padding injects a click into every frame.
* In the extension, subtitle text is inserted with `textContent`, never
  `innerHTML`. Subtitle text is remote content and must never be able to inject
  markup into a page the user is viewing.

## Extension permissions

Requested: `tabCapture`, `offscreen`, `storage`, `activeTab`, `scripting`.
Host permissions are limited to `127.0.0.1` and `localhost`; remote gateways
require the user to grant an optional host permission explicitly.

Tab capture requires user invocation on the target tab — the extension cannot
start capturing on its own.

## Logging

Raw audio is **never** logged. Transcripts are not logged by default. Metrics
carry timings and counts only, never content. Nothing is persisted unless the
session explicitly enables it.

## Reporting a vulnerability

Report privately to the maintainers rather than opening a public issue. Include
a description of the class of problem and its impact.

## Out of scope, permanently

VoiceBridge translates audio that the browser or OS legitimately provides to it.
It does not, and will not, bypass DRM, authentication, paywalls or access
controls. Contributions in that direction will be declined.
