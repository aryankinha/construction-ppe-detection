from __future__ import annotations

import asyncio
import io
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.detector import Detection, PPEDetector

router = APIRouter(tags=["detect"])


def _detections_to_json(detections: list[Detection]) -> list[dict]:
    return [
        {
            "class_id": d.class_id,
            "class_name": d.class_name,
            "confidence": round(d.confidence, 4),
            "x1": d.x1,
            "y1": d.y1,
            "x2": d.x2,
            "y2": d.y2,
            "color": list(d.color),
        }
        for d in detections
    ]


def _counts(detections: list[Detection]) -> dict:
    return {
        "hardhat": sum(1 for d in detections if d.class_name == "Hardhat"),
        "no_hardhat": sum(1 for d in detections if d.class_name == "NO-Hardhat"),
        "vest": sum(1 for d in detections if d.class_name == "Safety Vest"),
        "no_vest": sum(1 for d in detections if d.class_name == "NO-Safety Vest"),
        "person": sum(1 for d in detections if d.class_name == "Person"),
        "total": len(detections),
    }


def _annotate(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), d.color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_top = max(d.y1 - th - 6, 0)
        cv2.rectangle(out, (d.x1, y_top), (d.x1 + tw + 6, y_top + th + 6), d.color, -1)
        cv2.putText(
            out, label, (d.x1 + 3, y_top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


async def _decode_upload(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(data) > 4 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 4MB)")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    max_dim = 960
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _get_detector(request: Request) -> PPEDetector:
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return detector


@router.post("/detect-image")
async def detect_image(
    request: Request,
    file: UploadFile = File(...),
    format: Literal["jpeg", "json"] = Query("jpeg"),
):
    detector = _get_detector(request)
    img = await _decode_upload(file)

    loop = asyncio.get_event_loop()
    detections = await loop.run_in_executor(None, detector.detect, img)

    if format == "json":
        return JSONResponse(
            {"detections": _detections_to_json(detections), "counts": _counts(detections)}
        )

    annotated = _annotate(img, detections)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Encode failed")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


@router.post("/detect-frame")
async def detect_frame(request: Request, file: UploadFile = File(...)):
    detector = _get_detector(request)
    img = await _decode_upload(file)

    loop = asyncio.get_event_loop()
    detections = await loop.run_in_executor(None, detector.detect, img)

    h, w = img.shape[:2]
    return {
        "width": w,
        "height": h,
        "detections": _detections_to_json(detections),
        "counts": _counts(detections),
    }
