"""Visualization helpers for detections."""

from __future__ import annotations

import numpy as np
import cv2


def draw_detections(
    image_rgb: np.ndarray,
    detections: list[dict],
) -> np.ndarray:
    """Draw detection bounding boxes and labels on an RGB image.

    Uses OpenCV to draw rectangles and text. Modifies a copy of the
    image; the original is unchanged.

    Args:
        image_rgb: RGB image as numpy array (H, W, 3), uint8.
        detections: List of dicts with keys x1, y1, x2, y2, conf, cls_name
                    (e.g. as returned by detect_cars).

    Returns:
        New RGB image (uint8) with boxes and labels drawn.
    """
    out = np.asarray(image_rgb, dtype=np.uint8).copy()
    if not detections:
        return out

    for d in detections:
        x1 = int(d["x1"])
        y1 = int(d["y1"])
        x2 = int(d["x2"])
        y2 = int(d["y2"])
        conf = d.get("conf", 0)
        cls_name = d.get("cls_name", "")
        label = f"{cls_name} {conf:.2f}"

        color = (0, 255, 0)  # green in BGR
        thickness = 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1, y1 - 2),
            font,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out
