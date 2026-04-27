from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from app.core.logging import get_logger

logger = get_logger(__name__)

# Class names in the order produced by the YOLOv8 PPE model
CLASS_NAMES: list[str] = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "Machinery",
    "Vehicle",
]

# BGR colours per class
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 0, 128),
    7: (128, 128, 0),
    8: (0, 128, 128),
    9: (128, 128, 128),
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
    def __init__(self, model_path: str, confidence: float = 0.5, imgsz: int = 640) -> None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.confidence = confidence
        self.iou_threshold = 0.45
        logger.info("PPEDetector loaded (ONNX): %s (conf=%.2f)", model_path, confidence)

    @property
    def class_names(self) -> dict[int, str]:
        return {i: n for i, n in enumerate(CLASS_NAMES)}

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = img.shape[:2]
        scale = self.imgsz / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top = (self.imgsz - nh) // 2
        left = (self.imgsz - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, scale, left, top

    def detect(self, frame: np.ndarray) -> list[Detection]:
        orig_h, orig_w = frame.shape[:2]
        padded, scale, pad_x, pad_y = self._letterbox(frame)

        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        outputs = self.session.run(None, {self.input_name: blob})[0]
        # YOLOv8 ONNX output shape: (1, 4 + nc, N)
        preds = np.squeeze(outputs, axis=0).T  # -> (N, 4 + nc)

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        mask = confidences >= self.confidence
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Undo letterbox to original image coordinates
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y2 = np.clip(y2, 0, orig_h - 1)

        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        keep = cv2.dnn.NMSBoxes(
            nms_boxes, confidences.tolist(), self.confidence, self.iou_threshold
        )
        if len(keep) == 0:
            return []
        keep = np.array(keep).flatten()

        detections: list[Detection] = []
        for i in keep:
            cls = int(class_ids[i])
            detections.append(
                Detection(
                    class_id=cls,
                    class_name=CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls),
                    confidence=float(confidences[i]),
                    x1=int(x1[i]),
                    y1=int(y1[i]),
                    x2=int(x2[i]),
                    y2=int(y2[i]),
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
    sorted_dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for d in sorted_dets:
        if any(k.class_id == d.class_id and _iou(k, d) >= iou_threshold for k in kept):
            continue
        kept.append(d)
    return kept
