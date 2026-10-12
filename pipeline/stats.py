"""Quantitative analysis of a segmentation result.

Everything reported here is derived from the detected instances: counts,
type distribution, size statistics and (when the slide reports microns per
pixel) an area-normalised nuclear density.
"""

from __future__ import annotations

import numpy as np

UM_PER_PX_DEFAULT = 0.5  # 20x objective fallback, stated explicitly in reports


def compute(result) -> dict:
    """Return the full statistics block for an ``AnalysisResult``."""
    nuclei = result.nuclei
    n = len(nuclei)
    height, width = result.height, result.width
    mpp = result.mpp or UM_PER_PX_DEFAULT
    mpp_is_measured = result.mpp is not None

    field_area_mm2 = (height * mpp / 1000.0) * (width * mpp / 1000.0)
    area_mm2 = float(n) / field_area_mm2 if field_area_mm2 > 0 else 0.0

    areas = np.array([x.area for x in nuclei], dtype=np.float64)
    per_type: dict[str, dict] = {}
    for type_id, name in sorted(result.type_names.items()):
        if type_id == 0:
            # CoNSeP class 0 is "background"; nuclei that end up there are
            # reported separately instead of being mixed into a cell type.
            continue
        subset = [x for x in nuclei if x.type_id == type_id]
        if not subset:
            per_type[name] = {"count": 0, "fraction": 0.0, "mean_area_px": 0.0, "median_area_px": 0.0}
            continue
        a = np.array([x.area for x in subset], dtype=np.float64)
        per_type[name] = {
            "count": len(subset),
            "fraction": round(len(subset) / n, 4) if n else 0.0,
            "mean_area_px": round(float(a.mean()), 1),
            "median_area_px": round(float(np.median(a)), 1),
            "mean_type_confidence": round(float(np.mean([x.type_prob for x in subset])), 3),
        }

    probs = np.array([x.type_prob for x in nuclei], dtype=np.float64)
    unclassified = sum(1 for x in nuclei if x.type_id == 0)
    stats = {
        "nuclei_total": n,
        "unclassified": unclassified,
        "nuclei_per_mm2": round(area_mm2, 1),
        "field_area_mm2": round(field_area_mm2, 4),
        "microns_per_pixel": round(mpp, 4),
        "microns_per_pixel_source": "slide metadata" if mpp_is_measured else "assumed default",
        "by_type": per_type,
        "area_px": _describe(areas),
        "type_confidence": {
            "mean": round(float(probs.mean()), 3) if n else 0.0,
            "median": round(float(np.median(probs)), 3) if n else 0.0,
            "fraction_above_0.5": round(float((probs > 0.5).mean()), 3) if n else 0.0,
        },
        "coverage": {
            "nuclei_pixel_fraction": round(
                float(areas.sum()) / (height * width), 4
            ),
            "image_pixels": int(height * width),
        },
    }
    return stats


def _describe(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0, "max": 0, "p90": 0}
    return {
        "count": int(values.size),
        "mean": round(float(values.mean()), 1),
        "median": round(float(np.median(values)), 1),
        "std": round(float(values.std()), 1),
        "min": int(values.min()),
        "max": int(values.max()),
        "p90": int(np.percentile(values, 90)),
    }


def largest_nuclei(result, k: int = 10) -> list[dict]:
    return [
        {
            "id": n.id,
            "type": n.type_name,
            "area": n.area,
            "centroid": [round(n.centroid_y, 1), round(n.centroid_x, 1)],
            "confidence": round(n.type_prob, 3),
        }
        for n in sorted(result.nuclei, key=lambda x: -x.area)[:k]
    ]