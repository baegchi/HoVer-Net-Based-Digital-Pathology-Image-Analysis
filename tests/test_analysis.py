"""Tests for statistics, overlay rendering and report serialisation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hovernet.engine import Nucleus, _contour_area  # noqa: E402
from hovernet.overlay import colorize_types, type_legend  # noqa: E402
from pipeline import report as report_mod  # noqa: E402
from pipeline import stats as stats_mod  # noqa: E402

TYPE_NAMES = {
    0: "Unclassified",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Dead-karyotic",
}


def make_nucleus(nid, type_id, area=250, y=10.0, x=20.0, prob=0.9):
    side = int(np.sqrt(area))
    contour = np.array(
        [[x, y], [x + side, y], [x + side, y + side], [x, y + side]], np.int32
    )
    return Nucleus(
        id=nid, type_id=type_id, type_name=TYPE_NAMES[type_id], type_prob=prob,
        area=area, centroid_y=y + side / 2, centroid_x=x + side / 2,
        bbox_y0=int(y), bbox_x0=int(x), bbox_y1=int(y + side), bbox_x1=int(x + side),
        contour=contour,
    )


class FakeResult:
    source = "unit-test.png"
    width = 1024
    height = 1024
    stride = 64
    batch_size = 8
    device = "cpu"
    nr_types = 6
    min_area = 20
    tiles_total = tiles_done = 64
    inference_seconds = 1.5
    postproc_seconds = 0.25
    read_seconds = 0.75
    mpp = 0.5
    model_info = {"checkpoint": "x.pth", "nr_types": 6, "mode": "fast", "device": "cpu", "params": 54}
    type_names = TYPE_NAMES

    def __init__(self, nuclei):
        self.nuclei = nuclei

    def summary(self):
        return {
            "image_size": {"width": self.width, "height": self.height},
            "tiles": {"total": self.tiles_total, "processed": self.tiles_done, "stride": self.stride},
            "parameters": {"tile_size": 256, "output_tile": 164, "batch_size": 8, "min_area_px": 20},
            "model": {"checkpoint": "x.pth"},
        }


@pytest.fixture
def result():
    return FakeResult([make_nucleus(1, 1), make_nucleus(2, 1, y=100), make_nucleus(3, 3, y=200)])


def test_counts_and_distribution(result):
    s = stats_mod.compute(result)
    assert s["nuclei_total"] == 3
    assert s["by_type"]["Neoplastic"]["count"] == 2
    assert s["by_type"]["Neoplastic"]["fraction"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["by_type"]["Inflammatory"]["count"] == 0


def test_density_uses_microns_per_pixel(result):
    s = stats_mod.compute(result)
    # 1024 px * 0.5 um/px = 512 um = 0.512 mm per side
    assert s["field_area_mm2"] == pytest.approx(0.512**2, rel=1e-3)
    assert s["microns_per_pixel_source"] == "slide metadata"
    # density is reported rounded to 1 decimal place
    assert s["nuclei_per_mm2"] == pytest.approx(3 / (0.512**2), abs=0.05)


def test_density_falls_back_to_default_scale():
    r = FakeResult([make_nucleus(1, 1)])
    r.mpp = None
    s = stats_mod.compute(r)
    assert s["microns_per_pixel_source"] == "assumed default"
    assert s["microns_per_pixel"] == 0.5


def test_area_statistics(result):
    s = stats_mod.compute(result)
    a = s["area_px"]
    assert a["count"] == 3
    assert a["min"] <= a["median"] <= a["max"]
    assert s["type_confidence"]["mean"] == pytest.approx(0.9, abs=1e-6)


def test_contour_area():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.int32)
    assert _contour_area(square) == pytest.approx(100.0)


def test_overlay_changes_pixels_but_keeps_shape(result):
    image = np.full((256, 256, 3), 200, np.uint8)
    overlay = colorize_types(image, result.nuclei)
    assert overlay.shape == image.shape
    assert not np.array_equal(overlay, image), "overlay should tint the nuclei"


def test_legend_counts(result):
    legend = {name: count for name, _, count in type_legend(result)}
    assert legend["Neoplastic"] == 2
    assert legend["Connective"] == 1
    assert "Background" not in legend


def test_json_and_csv_roundtrip(result, tmp_path):
    s = stats_mod.compute(result)
    payload = report_mod.to_json(result, s)
    assert len(payload["nuclei"]) == 3
    assert "disclaimer" in payload["meta"]
    assert payload["nuclei"][0]["contour"]

    csv_path = report_mod.write_csv(result, tmp_path / "n.csv")
    lines = csv_path.read_text().strip().split("\n")
    assert lines[0].startswith("id,type_id,type_name")
    assert len(lines) == 4  # header + 3 nuclei


def test_markdown_report_contains_key_numbers(result):
    s = stats_mod.compute(result)
    md = report_mod.markdown_report(result, s)
    assert "Nuclei detected: **3**" in md
    assert "Neoplastic" in md
    assert "not a medical device" in md.lower()


def test_export_all_writes_every_artifact(result, tmp_path):
    s = stats_mod.compute(result)
    image = np.full((256, 256, 3), 180, np.uint8)
    written = report_mod.export_all(image, result, s, tmp_path, stem="t")
    for key in ("json", "csv", "report_md", "overlay_png", "comparison_png"):
        assert Path(written[key]).exists(), key
        assert Path(written[key]).stat().st_size > 0