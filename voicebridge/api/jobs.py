"""Job status, progress streams, cancellation and downloads."""

from __future__ import annotations

import json

from fastapi import APIRouter, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from voicebridge.api import ApiContext
from voicebridge.apps.gateway.security import extract_token, token_valid


def build_router(ctx: ApiContext) -> APIRouter:
    router = APIRouter()

    def job_or_404(job_id: str):
        job = ctx.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job

    @router.get("/api/v1/jobs")
    async def list_jobs(authorization: str | None = Header(None),
                        token: str | None = Query(None)) -> JSONResponse:
        ctx.authorize(authorization, token)
        return JSONResponse({"jobs": [
            {k: v for k, v in j.to_dict().items() if k not in ("log", "result")}
            for j in ctx.jobs.list()]})

    @router.get("/api/v1/jobs/{job_id}")
    async def get_job(job_id: str, authorization: str | None = Header(None),
                      token: str | None = Query(None)) -> JSONResponse:
        ctx.authorize(authorization, token)
        return JSONResponse(job_or_404(job_id).to_dict())

    @router.post("/api/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, authorization: str | None = Header(None),
                         token: str | None = Query(None)) -> JSONResponse:
        ctx.authorize(authorization, token)
        job_or_404(job_id)
        job = await ctx.jobs.cancel(job_id)
        return JSONResponse(job.to_dict())

    @router.delete("/api/v1/jobs/{job_id}")
    async def delete_job(job_id: str, authorization: str | None = Header(None),
                         token: str | None = Query(None)) -> JSONResponse:
        """Cancel if needed, then delete the upload, outputs and job record."""
        ctx.authorize(authorization, token)
        job_or_404(job_id)
        await ctx.jobs.delete(job_id)
        return JSONResponse({"job_id": job_id, "deleted": True})

    @router.get("/api/v1/jobs/{job_id}/outputs")
    async def outputs(job_id: str, authorization: str | None = Header(None),
                      token: str | None = Query(None)) -> JSONResponse:
        ctx.authorize(authorization, token)
        job = job_or_404(job_id)
        return JSONResponse({"job_id": job_id, "outputs": [
            {"name": name, "url": f"/api/v1/jobs/{job_id}/outputs/{name}"}
            for name in sorted(job.outputs)]})

    @router.get("/api/v1/jobs/{job_id}/outputs/{name}")
    async def download(job_id: str, name: str, authorization: str | None = Header(None),
                       token: str | None = Query(None)) -> FileResponse:
        ctx.authorize(authorization, token)
        job = job_or_404(job_id)
        if name not in job.outputs:  # names come from the job record, never the path
            raise HTTPException(404, "unknown output")
        path = await ctx.store.get(job.outputs[name])
        return FileResponse(path, filename=name)

    @router.get("/api/v1/jobs/{job_id}/events")
    async def events(job_id: str, authorization: str | None = Header(None),
                     token: str | None = Query(None)) -> StreamingResponse:
        """Server-Sent Events: job.progress plus pipeline events."""
        ctx.authorize(authorization, token)
        job_or_404(job_id)

        async def stream():
            async for event in ctx.jobs.events(job_id):
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @router.websocket("/api/v1/jobs/{job_id}/ws")
    async def events_ws(websocket: WebSocket, job_id: str) -> None:
        supplied = extract_token(websocket.headers.get("authorization"),
                                 websocket.query_params.get("token"))
        if not token_valid(ctx.config.security, supplied):
            await websocket.close(code=4401)
            return
        if ctx.jobs.get(job_id) is None:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        try:
            async for event in ctx.jobs.events(job_id):
                await websocket.send_text(json.dumps(event, ensure_ascii=False, default=str))
            await websocket.close()
        except (WebSocketDisconnect, RuntimeError):
            pass  # client went away

    return router
