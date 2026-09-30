"""Machine-readable model matrix.

One entry per model VoiceBridge knows how to run: what it is for, which
provider runs it, its **weights** licence (checked on each model card; code and
weights often differ), whether that licence allows commercial use, and the
worker environment it needs. ``config/voicebridge.yaml`` may add or override
entries under ``models:``.

Hardware figures are deliberately absent: actual memory depends on precision,
quantisation, audio length, batch size and concurrency. ``python -m
voicebridge.benchmark.resources`` measures them on the machine at hand.
"""

from __future__ import annotations

from typing import Any

MODELS: dict[str, dict[str, Any]] = {
    "whisper_large_v3_turbo": {
        "type": "asr", "provider": "whisperlivekit", "model": "openai/whisper-large-v3-turbo",
        "purpose": "Streaming + offline speech recognition (99 languages)",
        "license": "MIT", "commercial_use": True, "default": True,
    },
    "qwen3_asr_1_7b": {
        "type": "asr", "provider": "qwen3_asr", "model": "Qwen/Qwen3-ASR-1.7B",
        "purpose": "Alternative offline ASR (benchmark against Whisper)",
        "license": "Apache-2.0", "commercial_use": True, "default": False, "worker": "qwen",
    },
    "silero_vad": {
        "type": "vad", "provider": "silero", "model": "silero_vad_v6 (bundled with faster-whisper)",
        "purpose": "Voice activity detection", "license": "MIT", "commercial_use": True,
        "default": True,
    },
    "ast_audioset": {
        "type": "audio_event", "provider": "ast",
        "model": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "purpose": "Dialogue vs laughter/music/SFX classification (speech gate)",
        "license": "BSD-3-Clause", "commercial_use": True, "default": True,
    },
    "opus_mt": {
        "type": "translation", "provider": "opus_mt", "model": "Helsinki-NLP/opus-mt-*",
        "purpose": "Fast segment-level NMT fallback", "license": "Apache-2.0 (CC-BY-4.0 for some pairs; check each model card)",
        "commercial_use": True, "default": "fallback",
    },
    "contextual_qwen3_1_7b": {
        "type": "translation", "provider": "contextual", "backend": "transformers",
        "model": "Qwen/Qwen3-1.7B", "purpose": "Context-aware LLM translation (CPU-capable)",
        "license": "Apache-2.0", "commercial_use": True, "default": "balanced",
    },
    "contextual_qwen3_4b": {
        "type": "translation", "provider": "contextual", "backend": "transformers",
        "model": "Qwen/Qwen3-4B-Instruct-2507", "purpose": "Context-aware LLM translation",
        "license": "Apache-2.0", "commercial_use": True, "default": "high_quality",
    },
    "contextual_remote": {
        "type": "translation", "provider": "contextual", "backend": "openai | anthropic",
        "model": "configured", "purpose": "Context-aware translation via an API endpoint",
        "license": "depends on the service/model", "commercial_use": None, "default": False,
    },
    "nllb_200_distilled_600m": {
        "type": "translation", "provider": "nllb", "model": "facebook/nllb-200-distilled-600M",
        "purpose": "Multilingual research baseline", "license": "CC-BY-NC-4.0",
        "commercial_use": False, "default": False, "mode": "research",
    },
    "indictrans2": {
        "type": "translation", "provider": "indictrans2", "model": "ai4bharat/indictrans2-*",
        "purpose": "English <-> Indic NMT", "license": "MIT", "commercial_use": True,
        "default": False,
    },
    "piper": {
        "type": "tts", "provider": "piper", "model": "rhasspy/piper-voices (per voice)",
        "purpose": "Fast CPU TTS; fallback for languages Qwen3-TTS lacks (Hindi)",
        "license": "engine MIT/GPL-3.0 (piper-tts 1.3+); each voice has its own licence",
        "commercial_use": None, "default": "fallback",
    },
    "qwen3_tts_1_7b_base": {
        "type": "tts", "provider": "qwen3", "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "purpose": "Higher-quality voice-cloning TTS (en, ja, ko, zh, de, fr, ru, pt, es, it)",
        "license": "Apache-2.0", "commercial_use": True, "default": True, "worker": "qwen",
    },
    "seamless_streaming": {
        "type": "s2st", "provider": "seamless_streaming", "model": "facebook/seamless-streaming",
        "purpose": "Meta SeamlessStreaming direct speech-to-speech translation",
        "license": "CC-BY-NC-4.0 (weights); code MIT", "commercial_use": False,
        "default": False, "worker": "seamless",
        "notice": "Non-commercial licence. Review before any commercial deployment.",
    },
    "seamless_m4t_v2_large": {
        "type": "s2st", "provider": "seamless_m4t_v2", "model": "facebook/seamless-m4t-v2-large",
        "purpose": "Offline SeamlessM4T v2 reference backend", "license": "CC-BY-NC-4.0",
        "commercial_use": False, "default": False,
        "notice": "Non-commercial licence. Research and evaluation only.",
    },
}


def model_matrix(overrides: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    merged = {k: dict(v) for k, v in MODELS.items()}
    for key, value in (overrides or {}).items():
        merged.setdefault(key, {}).update(value or {})
    return merged
