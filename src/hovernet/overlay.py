"""Render the segmentation overlay used everywhere in the UI and the reports."""

from __future__ import annotations

import numpy as np

from .checkpoint import TYPE_COLORS

BOUNDARY = (255, 255, 255)


def colorize_types(
    image: np.ndarray,
    nuclei,
    alpha: float = 0.45,
    boundaries: bool = True,
) -> np.ndarray:
    """Blend a type-coloured instance map over the original image."""
    base = _as_rgb(image)
    overlay = base.copy()
    h, w = base.shape[:2]

    import cv2

    masks: dict[int, list[np.ndarray]] = {}
    for n in nuclei:
        if n.contour.size >= 3:
            masks.setdefault(int(n.type_id), []).append(n.contour.astype(np.int32))

    for type_id, contours in masks.items():
        color = np.array(TYPE_COLORS.get(type_id, (200, 200, 200)), np.uint8)
        layer = overlay.copy()
        cv2.fillPoly(layer, contours, color.tolist())
        # keep the tint inside the image only
        overlay = layer

    blended = (alpha * overlay + (1 - alpha) * base).astype(np.uint8)

    if boundaries:
        edge = np.zeros((h, w), np.uint8)
        for type_id, contours in masks.items():
            cv2.polylines(edge, contours, isClosed=True, color=1, thickness=1)
        # 1px dilation gives a crisp outline without fattening small nuclei
        edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)
        blended[edge > 0] = BOUNDARY
    return blended


def type_legend(result) -> list[tuple[str, tuple[int, int, int], int]]:
    """(name, colour, count) rows in a stable order, for legends and reports."""
    counts: dict[str, int] = {}
    for n in result.nuclei:
        counts[n.type_name] = counts.get(n.type_name, 0) + 1
    rows = []
    for type_id, name in sorted(result.type_names.items()):
        if type_id == 0:
            continue
        rows.append((name, TYPE_COLORS.get(type_id, (200, 200, 200)), counts.get(name, 0)))
    return rows


def side_by_side(original: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Concatenate the original and the overlay with a separator."""
    base = _as_rgb(original)
    ov = _as_rgb(overlay)
    gap = np.full((base.shape[0], 4, 3), 240, np.uint8)
    return np.concatenate([base, gap, ov], axis=1)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, -1)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image