"""The browser UI served at ``/``.

The page itself is ``static/index.html`` (one self-contained file: no build
step, no external assets, works offline). It is designed against Jakob
Nielsen's ten usability heuristics; the comment block at the top of the HTML
maps each heuristic to what the page does about it.

Views: **Live** (microphone -> WebSocket session), **Translate file** (upload,
progress over Server-Sent Events, downloads, recent jobs), **Compare**
(benchmark reports found on the server; nothing is filled in) and **Help**.
"""

from pathlib import Path

UI_PATH = Path(__file__).parent / "static" / "index.html"


def load_ui() -> str:
    """Read the page on each request: it is small, and edits show up on reload."""
    return UI_PATH.read_text(encoding="utf-8")


DEMO_HTML = load_ui()
