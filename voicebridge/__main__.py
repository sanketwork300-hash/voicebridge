"""``python -m voicebridge`` starts the gateway (same as ``voicebridge serve``)."""

from voicebridge.apps.cli.main import main

if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:] or ["serve"]))
