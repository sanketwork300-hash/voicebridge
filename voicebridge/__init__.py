"""VoiceBridge: a self-hostable, real-time speech-to-speech translation engine.

The package is organised as:

* ``voicebridge.core``      -- source/sink agnostic pipeline machinery
* ``voicebridge.providers`` -- pluggable ASR / translation / TTS backends
* ``voicebridge.adapters``  -- input and output adapters
* ``voicebridge.protocols`` -- wire protocols (WebSocket)
* ``voicebridge.apps``      -- runnable entry points (gateway, CLI)

Nothing in ``core`` may import from ``apps`` or ``adapters``; the dependency
direction is always inward.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
