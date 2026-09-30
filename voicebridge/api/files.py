"""File upload and translation endpoints.

``POST /api/v1/files/upload``      store a file, return an ``upload_id``
``POST /api/v1/files/translate``   start a job from ``file`` or ``upload_id``
``POST /api/v1/translate/file``    alias of the above

Both return immediately; processing happens in the job worker.
"""

from __future__ import annotations

import json
import re

from fastapi import APIRouter, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from voicebridge.api import ApiContext
from voicebridge.api.uploads import save_upload
from voicebridge.core.jobs.executor import AUDIO_OUTPUTS

ENGINES = {"cascade", "seamless_streaming", "seamless_m4t_v2"}
AUDIO_MODES = {"replace", "keep", "mix"}
QUALITY = {"fast", "balanced", "high_quality"}
_UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{32}\.[a-z0-9]{2,5}$")
#: BCP-47-ish language code: "ja", "en", "pt-BR", "zh-Hant". Validated up front
#: because it ends up in output file names and storage keys.
_LANG_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})?$")
HONORIFICS = {"preserve", "naturalize", "naturalise", "remove", "preserve_honorifics",
              "natural_english"}
OUTPUTS = {"audio", "video", "srt", "vtt", "subtitled_video"}


def build_router(ctx: ApiContext) -> APIRouter:
    router = APIRouter()
    media_cfg = ctx.config.media or {}
    max_bytes = int(media_cfg.get("max_file_size_mb", 2048)) * 1024 * 1024
    quota = media_cfg.get("disk_quota_gb")
    quota_bytes = int(float(quota) * 1024**3) if quota else None

    @router.post("/api/v1/files/upload")
    async def upload(file: UploadFile = File(...), authorization: str | None = Header(None),
                     token: str | None = Query(None)) -> JSONResponse:
        ctx.authorize(authorization, token)
        key, filename, size = await save_upload(file, ctx.store, max_bytes, quota_bytes)
        return JSONResponse({"upload_id": key.split("/", 1)[1], "filename": filename,
                             "size": size}, status_code=201)

    @router.post("/api/v1/files/translate")
    @router.post("/api/v1/translate/file")
    async def translate(
        file: UploadFile | None = File(None),
        upload_id: str | None = Form(None),
        filename: str | None = Form(None),
        source_language: str = Form("auto"),
        target_language: str = Form("en"),
        engine: str | None = Form(None),
        output_format: str | None = Form(None),
        outputs: str | None = Form(None),
        audio_mode: str = Form("replace"),
        voice: str | None = Form(None),
        translation_provider: str | None = Form(None),
        translation_quality: str | None = Form(None),
        tts_provider: str | None = Form(None),
        subtitle_enabled: bool = Form(True),
        dual_subtitles: bool = Form(False),
        embed_subtitles: bool = Form(True),
        synthesize: bool = Form(True),
        honorifics: str | None = Form(None),
        glossary: str | None = Form(None),
        style: str | None = Form(None),
        authorization: str | None = Header(None),
        token: str | None = Query(None),
    ) -> JSONResponse:
        ctx.authorize(authorization, token)
        if source_language != "auto" and not _LANG_RE.match(source_language):
            raise HTTPException(400, "invalid source_language")
        if not _LANG_RE.match(target_language):
            raise HTTPException(400, "invalid target_language")
        if honorifics and honorifics not in HONORIFICS:
            raise HTTPException(400, f"honorifics must be one of {sorted(HONORIFICS)}")
        if outputs and not set(o.strip() for o in outputs.split(",") if o.strip()) <= OUTPUTS:
            raise HTTPException(400, f"outputs must be a subset of {sorted(OUTPUTS)}")
        engine = engine or ctx.config.default_engine
        if engine not in ENGINES:
            raise HTTPException(400, f"engine must be one of {sorted(ENGINES)}")
        if audio_mode not in AUDIO_MODES:
            raise HTTPException(400, f"audio_mode must be one of {sorted(AUDIO_MODES)}")
        if translation_quality and translation_quality not in QUALITY:
            raise HTTPException(400, f"translation_quality must be one of {sorted(QUALITY)}")
        if output_format and output_format not in AUDIO_OUTPUTS | {"mp4", "mkv", "webm", "mov"}:
            raise HTTPException(400, "unsupported output_format")
        if file is not None and file.filename:
            key, clean_name, _ = await save_upload(file, ctx.store, max_bytes, quota_bytes)
        elif upload_id and _UPLOAD_ID_RE.match(upload_id):
            key = f"uploads/{upload_id}"
            if not await ctx.store.exists(key):
                raise HTTPException(404, "unknown upload_id")
            from voicebridge.api.uploads import sanitize_filename

            clean_name = sanitize_filename(filename or f"media{upload_id[32:]}")
            if not clean_name.endswith(upload_id[32:]):
                clean_name = f"{clean_name.rsplit('.', 1)[0]}{upload_id[32:]}"
        else:
            raise HTTPException(400, "provide a file or a valid upload_id")
        terms = []
        if glossary:
            try:
                parsed = json.loads(glossary)
                terms = parsed.get("terms", parsed) if isinstance(parsed, dict) else parsed
            except json.JSONDecodeError:
                raise HTTPException(400, "glossary must be JSON") from None
        payload = {
            "upload_key": key, "filename": clean_name,
            "source_language": source_language, "target_language": target_language,
            "engine": engine, "output_format": output_format,
            "outputs": [o.strip() for o in outputs.split(",")] if outputs else None,
            "audio_mode": audio_mode, "voice": voice,
            "translation_provider": translation_provider,
            "translation_quality": translation_quality, "tts_provider": tts_provider,
            "subtitle_enabled": subtitle_enabled, "dual_subtitles": dual_subtitles,
            "embed_subtitles": embed_subtitles, "synthesize": synthesize,
            "honorifics": honorifics, "glossary": terms, "style": style,
        }
        job = await ctx.jobs.submit(payload)
        return JSONResponse({"job_id": job.job_id, "status": job.status.value},
                            status_code=202)

    return router
