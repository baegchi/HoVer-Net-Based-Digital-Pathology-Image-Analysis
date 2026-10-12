"""Serialise an analysis to JSON, CSV, Markdown and PNG overlay."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hovernet.overlay import colorize_types, side_by_side, type_legend

CSV_COLUMNS = [
    "id",
    "type_id",
    "type_name",
    "confidence",
    "area_px",
    "centroid_y",
    "centroid_x",
    "bbox_y0",
    "bbox_x0",
    "bbox_y1",
    "bbox_x1",
]

DISCLAIMER = (
    "Research and educational prototype. Not a medical device and not validated "
    "for clinical or diagnostic use."
)


def _meta(result) -> dict:
    return {
        "tool": "AI-Powered Digital Pathology Image Analyzer",
        "model": "HoVer-Net (vqdang/hover_net), CoNSeP pretrained checkpoint",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": result.source,
        "device": result.device,
        "disclaimer": DISCLAIMER,
    }


def to_json(result, stats: dict) -> dict:
    return {
        "meta": _meta(result),
        "summary": result.summary(),
        "statistics": stats,
        "nuclei": [n.to_dict() for n in result.nuclei],
    }


def write_json(result, stats: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(result, stats), indent=2))
    return path


def write_csv(result, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for n in result.nuclei:
            writer.writerow(
                [
                    n.id,
                    n.type_id,
                    n.type_name,
                    round(n.type_prob, 4),
                    n.area,
                    round(n.centroid_y, 1),
                    round(n.centroid_x, 1),
                    n.bbox_y0,
                    n.bbox_x0,
                    n.bbox_y1,
                    n.bbox_x1,
                ]
            )
    return path


def write_markdown(result, stats: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_report(result, stats))
    return path


def markdown_report(result, stats: dict) -> str:
    legend = type_legend(result)
    lines = [
        "# Digital Pathology Analysis Report",
        "",
        f"- **Source:** `{result.source}`",
        f"- **Image size:** {result.width} x {result.height} px",
        f"- **Model:** HoVer-Net (CoNSeP checkpoint), running on `{result.device}`",
        f"- **Analysed:** {datetime.now(timezone.utc).isoformat(timespec='seconds')} (UTC)",
        "",
        "## Headline",
        "",
        f"- Nuclei detected: **{stats['nuclei_total']}**",
        f"- Nuclear density: **{stats['nuclei_per_mm2']:,} nuclei/mm2**"
        f" (field {stats['field_area_mm2']} mm2)",
        f"- Scale: {stats['microns_per_pixel']} um/px ({stats['microns_per_pixel_source']})",
        f"- Runtime: {result.inference_seconds + result.postproc_seconds:.1f} s "
        f"over {result.tiles_total} tiles ({result.tiles_done} processed)",
        "",
        "## Cell type distribution",
        "",
        "| Type | Colour | Count | Share | Mean area (px) | Mean confidence |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, color, count in legend:
        row = stats["by_type"].get(name, {})
        hexcode = "#%02X%02X%02X" % color
        lines.append(
            f"| {name} | `{hexcode}` | {count} | {row.get('fraction', 0) * 100:.1f}% | "
            f"{row.get('mean_area_px', 0)} | {row.get('mean_type_confidence', 0)} |"
        )

    area = stats["area_px"]
    lines += [
        "",
        "## Nucleus size distribution",
        "",
        f"- Count: {area['count']}",
        f"- Mean: {area['mean']} px, median: {area['median']} px, sd: {area['std']} px",
        f"- Range: {area['min']} - {area['max']} px (p90: {area['p90']} px)",
        f"- Class-confidence: mean {stats['type_confidence']['mean']}, "
        f"{stats['type_confidence']['fraction_above_0.5'] * 100:.0f}% above 0.5",
        "",
        "## Largest nuclei",
        "",
        "| id | type | area (px) | centroid (y, x) | confidence |",
        "| ---: | --- | ---: | --- | ---: |",
    ]
    for n in sorted(result.nuclei, key=lambda x: -x.area)[:10]:
        lines.append(
            f"| {n.id} | {n.type_name} | {n.area} | "
            f"({n.centroid_y:.0f}, {n.centroid_x:.0f}) | {n.type_prob:.2f} |"
        )

    lines += [
        "",
        "## Method",
        "",
        f"Images are split into {result.stride}-px-stride windows. Each 270x270 input patch "
        "yields an 80x80 prediction that is averaged where windows overlap, then nuclei are "
        "extracted with the HoVer-Net marker-controlled watershed (nuclear probability "
        "threshold + HoVer gradient maps) and labelled from the type head.",
        "",
        f"Parameters: tile {result.summary()['parameters']['tile_size']} px, "
        f"output {result.summary()['parameters']['output_tile']} px, "
        f"batch {result.batch_size}, min area {result.min_area} px, device {result.device}.",
        "",
        "---",
        "",
        f"_{DISCLAIMER}_",
        "",
    ]
    return "\n".join(lines)


def save_overlay(image: np.ndarray, result, path: str | Path, alpha: float = 0.45) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = colorize_types(image, result.nuclei, alpha=alpha)
    Image.fromarray(overlay).save(path)
    return path


def save_comparison(image: np.ndarray, result, path: str | Path) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = colorize_types(image, result.nuclei)
    Image.fromarray(side_by_side(image, overlay)).save(path)
    return path


def export_all(
    image: np.ndarray,
    result,
    stats: dict,
    out_dir: str | Path,
    stem: str = "analysis",
    with_images: bool = True,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "json": str(write_json(result, stats, out_dir / f"{stem}.json")),
        "csv": str(write_csv(result, out_dir / f"{stem}.csv")),
        "report_md": str(write_markdown(result, stats, out_dir / f"{stem}_report.md")),
    }
    if with_images:
        written["overlay_png"] = str(save_overlay(image, result, out_dir / f"{stem}_overlay.png"))
        written["comparison_png"] = str(
            save_comparison(image, result, out_dir / f"{stem}_comparison.png")
        )
    return written