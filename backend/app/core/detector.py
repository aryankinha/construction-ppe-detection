from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

# BGR colours matching original webcam.py
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 0, 0),    # Hardhat
    1: (0, 255, 0),    # Mask
    2: (0, 0, 255),    # NO-Hardhat
    3: (255, 255, 0),  # NO-Mask
    4: (255, 0, 255),  # NO-Safety Vest
    5: (0, 255, 255),  # Person
    6: (128, 0, 128),  # Safety Cone
    7: (128, 128, 0),  # Safety Vest
    8: (0, 128, 128),  # Machinery
    9: (128, 128, 128),  # Vehicle
}


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    color: tuple[int, int, int]


class PPEDetector:
    def __init__(self, model_path: str, confidence: float = 0.5) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence = confidence
        logger.info("PPEDetector loaded: %s (conf=%.2f)", model_path, confidence)

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.names

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # iou=0.45 tightens NMS so overlapping boxes of the same class
        # (e.g. two "Person" boxes on the same body) collapse into one.
        results = self.model(
            frame, conf=self.confidence, iou=0.45, verbose=False,
        )
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0])
                detections.append(
                    Detection(
                        class_id=cls,
                        class_name=self.model.names[cls],
                        confidence=float(box.conf[0]),
                        x1=int(box.xyxy[0][0]),
                        y1=int(box.xyxy[0][1]),
                        x2=int(box.xyxy[0][2]),
                        y2=int(box.xyxy[0][3]),
                        color=CLASS_COLORS.get(cls, (200, 200, 200)),
                    )
                )

        return _dedupe_same_class(detections, iou_threshold=0.4)


def _iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dedupe_same_class(dets: list[Detection], iou_threshold: float) -> list[Detection]:
    """Drop lower-confidence boxes that overlap a higher-confidence box of the same class."""
    sorted_dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for d in sorted_dets:
        if any(k.class_id == d.class_id and _iou(k, d) >= iou_threshold for k in kept):
            continue
        kept.append(d)
    return kept
