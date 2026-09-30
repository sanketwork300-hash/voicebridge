"""Model, engine and option discovery for the UI.

``GET /api/v1/models``          model matrix with licences
``GET /api/v1/models/engines``  translation engines the user can pick, with
                                whether each is enabled and its licence notice
``GET /api/v1/models/options``  quality presets and voices
"""

from __future__ import annotations

from fastapi import APIRouter, Header, Query
from fastapi.responses import JSONResponse

from voicebridge.api import ApiContext
from voicebridge.core.jobs.executor import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from voicebridge.model_registry import model_matrix


def build_router(ctx: ApiContext) -> APIRouter:
    router = APIRouter()
    providers = ctx.config.providers

    def cascade_summary() -> dict[str, str]:
        def pick(kind: str) -> str:
            cfg = providers.get(kind) or {}
            name = cfg.get("provider", "mock")
            nested = cfg.get(name) if isinstance(cfg.get(name), dict) else {}
            model = nested.get("model") or cfg.get("model")
            return f"{name} ({model})" if model else name
        return {"asr": pick("asr"), "translation": pick("translation"), "tts": pick("tts")}

    @router.get("/api/v1/models")
    @router.get("/v1/models")
    async def models(authorization: str | None = Header(None),
                    token: str | None = Query(None)) -> JSONResponse:
        ctx.authorize(authorization, token)
        return JSONResponse({
            "models": model_matrix(ctx.config.raw.get("models")),
            "runtime_profiles": ctx.config.raw.get("runtime_profiles") or {},
            "runtime": ctx.config.runtime,
        })

    @router.get("/api/v1/models/engines")
    async def engines(authorization: str | None = Header(None),
                     token: str | None = Query(None)) -> JSONResponse:
        ctx.authorize(authorization, token)
        s2st = providers.get("s2st") or {}
        out = [{
            "id": "cascade", "label": "VoiceBridge Cascade",
            "description": "ASR + contextual translation + TTS", "enabled": True,
            "components": cascade_summary(), "notice": None,
        }]
        matrix = model_matrix(ctx.config.raw.get("models"))
        for engine_id, label, key in (
            ("seamless_streaming", "Meta SeamlessStreaming", "seamless_streaming"),
            ("seamless_m4t_v2", "Meta SeamlessM4T v2 (offline)", "seamless_m4t_v2_large"),
        ):
            opts = s2st.get(engine_id) or {}
            out.append({
                "id": engine_id, "label": label,
                "description": "Direct speech-to-speech translation",
                "enabled": bool(opts.get("enabled")),
                "model": opts.get("model"),
                "license": matrix[key]["license"],
                "notice": "Model licensing and resource requirements apply: "
                          + matrix[key].get("notice", ""),
            })
        return JSONResponse({"engines": out, "default": ctx.config.default_engine})

    @router.get("/api/v1/models/options")
    async def options(authorization: str | None = Header(None),
                     token: str | None = Query(None)) -> JSONResponse:
        ctx.authorize(authorization, token)
        presets = providers.get("translation_presets") or {}
        tts = providers.get("tts") or {}
        voices = ["source"] + sorted((tts.get("voices") or {}).keys())
        return JSONResponse({
            "translation_quality": [
                {"id": k, "provider": v.get("provider"), "model": v.get("model")}
                for k, v in presets.items()],
            "voices": voices,
            "audio_modes": ["replace", "keep", "mix"],
            "languages": ctx.config.languages,
            # Lets the UI reject a wrong or oversized file before uploading it.
            "limits": {
                "max_file_size_mb": int((ctx.config.media or {}).get("max_file_size_mb", 2048)),
                "audio_extensions": sorted(AUDIO_EXTENSIONS),
                "video_extensions": sorted(VIDEO_EXTENSIONS),
            },
        })

    @router.get("/api/v1/benchmarks")
    async def benchmarks(authorization: str | None = Header(None),
                        token: str | None = Query(None)) -> JSONResponse:
        ctx.authorize(authorization, token)
        """Benchmark and comparison reports found under ``benchmark.results_dir``.

        The UI's comparison page shows only these files, i.e. only numbers that
        were actually measured.
        """
        import json
        from pathlib import Path

        root = Path((ctx.config.raw.get("benchmark") or {}).get("results_dir",
                                                                 "benchmark/results"))
        reports = []
        for path in sorted(root.rglob("*.json")):
            if path.name not in ("benchmark.json", "comparison.json"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            rows = [{k: v for k, v in r.items() if k not in ("per_item", "hypothesis")}
                    for r in data.get("results", [])]
            reports.append({"path": str(path), "title": data.get("title"),
                            "timestamp": (data.get("environment") or {}).get("timestamp"),
                            "environment": {k: (data.get("environment") or {}).get(k)
                                            for k in ("cpu", "ram_gb", "device", "gpu",
                                                      "commit", "torch")},
                            "results": rows})
        return JSONResponse({"reports": reports})

    return router
