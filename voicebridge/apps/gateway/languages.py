"""Language and language-pair catalogue exposed at ``GET /v1/languages``.

Tiers reflect *product* priority, not capability: they say which pairs are
benchmarked and supported for launch, not which pairs the models can attempt.
The pipeline never reads them -- they are advertising metadata for the UI.
"""

from __future__ import annotations

from typing import Any

LANGUAGE_NAMES: dict[str, str] = {
    "auto": "Auto-detect",
    "en": "English", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "hi": "Hindi", "mr": "Marathi", "ta": "Tamil", "te": "Telugu",
    "bn": "Bengali", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "ur": "Urdu", "as": "Assamese", "or": "Odia",
    "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
    "ru": "Russian", "ar": "Arabic",
}

NATIVE_NAMES: dict[str, str] = {
    "en": "English", "ja": "日本語", "ko": "한국어", "zh": "中文",
    "hi": "हिन्दी", "mr": "मराठी", "ta": "தமிழ்", "te": "తెలుగు",
    "bn": "বাংলা", "gu": "ગુજરાતી", "kn": "ಕನ್ನಡ", "ml": "മലയാളം",
    "pa": "ਪੰਜਾਬੀ", "ur": "اردو",
}

TIER_1: list[tuple] = [("ja", "en"), ("ko", "en"), ("en", "hi")]
TIER_2: list[tuple] = [
    ("en", "ja"), ("en", "ko"), ("en", "mr"), ("en", "ta"),
    ("en", "te"), ("en", "bn"),
]


def catalogue(providers: dict[str, Any] | None = None) -> dict[str, Any]:
    def pair(src: str, dst: str, tier: int) -> dict[str, Any]:
        return {
            "source": src,
            "target": dst,
            "source_name": LANGUAGE_NAMES.get(src, src),
            "target_name": LANGUAGE_NAMES.get(dst, dst),
            "tier": tier,
        }

    return {
        "languages": [
            {
                "code": code,
                "name": name,
                "native_name": NATIVE_NAMES.get(code),
            }
            for code, name in sorted(LANGUAGE_NAMES.items())
        ],
        "pairs": (
            [pair(s, t, 1) for s, t in TIER_1] + [pair(s, t, 2) for s, t in TIER_2]
        ),
        "note": (
            "Tier 1 and 2 pairs are the benchmarked launch set. Other combinations "
            "may work depending on the configured translation provider; query "
            "/v1/providers for the authoritative capability list."
        ),
    }
