"""Worker-side half of the model-worker protocol.

This file is imported by worker scripts that run in *another* virtualenv (one
with the model's pinned dependencies), so it must not import ``voicebridge`` or
anything outside the standard library.

Protocol: one JSON object per line.

    request   {"id": 1, "method": "synthesize", "params": {...}}
    response  {"id": 1, "result": {...}}
           or {"id": 1, "error": {"type": "ValueError", "message": "..."}}

The worker announces itself with ``{"id": 0, "result": {"ready": true, ...}}``
once imports succeed. Bulk audio never goes through the pipe: requests and
results carry file paths in a directory both processes can see.

Libraries love printing to stdout, which would corrupt the protocol stream.
:func:`serve` therefore moves the real stdout to a private file descriptor and
points fd 1 at stderr before any model code runs.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from collections.abc import Callable
from typing import Any


def serve(handlers: dict[str, Callable[..., Any]], info: dict[str, Any] | None = None) -> None:
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    out = os.fdopen(proto_fd, "w", buffering=1, encoding="utf-8")

    def send(obj: dict[str, Any]) -> None:
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out.flush()

    send({"id": 0, "result": {"ready": True, "pid": os.getpid(), **(info or {})}})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = request.get("id")
        method = request.get("method")
        if method == "shutdown":
            send({"id": rid, "result": {"bye": True}})
            break
        handler = handlers.get(method or "")
        if handler is None:
            send({"id": rid, "error": {"type": "UnknownMethod", "message": str(method)}})
            continue
        try:
            send({"id": rid, "result": handler(**(request.get("params") or {}))})
        except Exception as exc:  # reported to the client, never fatal
            traceback.print_exc()
            send({"id": rid, "error": {"type": type(exc).__name__, "message": str(exc)}})
