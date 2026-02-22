"""Vehicle detection using Ultralytics YOLO (tiled inference + optional upscaling for aerial imagery)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO
from torchvision.ops import nms
import cv2  # opencv-python

# Project root (parent of parkpulse package) for resolving models/best.pt
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# VisDrone-style class names (common)
VEHICLE_CLASS_NAMES = {"car", "van", "truck", "bus", "motor"}

_model: YOLO | None = None


def _resolve_model_path(model_name: str) -> str:
    """If path is models/best.pt (or models\\best.pt), resolve to project root."""
    p = Path(model_name)
    if p.name == "best.pt" and (model_name.startswith("models/") or "models" in p.parts):
        return str(_PROJECT_ROOT / "models" / "best.pt")
    return model_name


def load_model(model_name: str = "models/best.pt") -> YOLO:
    """Load a YOLO model by name (downloads if not present)."""
    global _model
    path = _resolve_model_path(model_name)
    _model = YOLO(path)
    return _model


def _get_model() -> YOLO:
    """Return the globally loaded model, loading default if needed."""
    global _model
    if _model is None:
        path = _resolve_model_path("models/best.pt")
        _model = YOLO(path)
    return _model


def detect_cars(
    image_rgb: np.ndarray,
    conf: float = 0.05,          
    imgsz: int = 1536,
    tile: int = 1024,            
    overlap: int = 256,          
    iou: float = 0.5,
    upscale_small: bool = True,
    max_det: int = 5000,         # cap per tile; avoid 1000 default so large lots aren't truncated
) -> list[dict[str, Any]]:
    """
    Detect vehicles in aerial/satellite imagery using tiled inference.

    Returns:
        List of detections dicts: x1,y1,x2,y2,conf,cls_id,cls_name.
        Coordinates are in original image pixel space.
    """
    # Strip alpha if present
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
        image_rgb = image_rgb[:, :, :3]

    # Ensure uint8 for OpenCV/YOLO robustness
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    
    scale = 1.0
    if upscale_small and min(image_rgb.shape[0], image_rgb.shape[1]) < 2048:
        scale = 2.0
        image_rgb_up = cv2.resize(
            image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    else:
        image_rgb_up = image_rgb

    model = _get_model()
    H, W, _ = image_rgb_up.shape
    step = max(1, tile - overlap)

    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_clses: list[np.ndarray] = []

    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            patch = image_rgb_up[y0:y1, x0:x1]

            r = model.predict(
                patch,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                verbose=False,
            )[0]

            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            clses = r.boxes.cls.cpu().numpy()

            # shift patch coords -> global coords (upscaled image space)
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_clses.append(clses)

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    clses = np.concatenate(all_clses, axis=0)

    # Global NMS to dedupe overlapping-tile detections
    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep = nms(boxes_t, scores_t, iou_threshold=0.35).cpu().numpy()

    names = getattr(model, "names", {}) or {}

    out: list[dict[str, Any]] = []
    for i in keep:
        cls_id = int(clses[i])
        cls_name = str(names.get(cls_id, cls_id)).lower()

        # Filter to vehicle-like classes (VisDrone)
        if cls_name not in VEHICLE_CLASS_NAMES:
            continue

        x1, y1, x2, y2 = boxes[i]

        # Scale back to original image coords if we upscaled
        if scale != 1.0:
            x1 /= scale
            y1 /= scale
            x2 /= scale
            y2 /= scale

        # ---- size/aspect filters to remove aerial false positives ----
        w = float(x2 - x1)
        h = float(y2 - y1)
        area = w * h
        aspect = w / max(1.0, h)

        if w < 6 or h < 6:
            continue
        if w > 160 or h > 160:
            continue
        if area > 160 * 160:
            continue
        if aspect < 0.25 or aspect > 4.0:
            continue

        out.append(
            {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "conf": float(scores[i]),
                "cls_id": cls_id,
                "cls_name": cls_name,
            }
        )

    return out