from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    model_loaded = hasattr(request.app.state, "detector")
    return {"status": "ok", "model_loaded": model_loaded}
