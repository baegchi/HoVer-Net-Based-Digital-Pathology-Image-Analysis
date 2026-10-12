"""Tests for the tiling geometry - the part most likely to silently misalign."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hovernet import tiling  # noqa: E402
from hovernet.readers import ArrayReader  # noqa: E402


@pytest.fixture(autouse=True)
def reset_original():
    tiling.configure("original")
    yield
    tiling.configure("original")


def coverage_map(height, width, stride):
    cov = np.zeros((height, width), np.int32)
    for t in tiling.plan_tiles(height, width, stride):
        cov[t.out_y : t.out_y + t.valid_h, t.out_x : t.out_x + t.valid_w] += 1
    return cov


@pytest.mark.parametrize(
    "h,w,stride",
    [(256, 256, 64), (100, 700, 64), (700, 100, 64), (1536, 1536, 64), (300, 300, 128), (64, 64, 64)],
)
def test_every_pixel_is_covered(h, w, stride):
    cov = coverage_map(h, w, stride)
    assert cov.min() >= 1, "some pixels were never predicted"


@pytest.mark.parametrize("h,w,stride", [(256, 256, 64), (1000, 800, 48), (1536, 1536, 64)])
def test_overlap_is_bounded(h, w, stride):
    """A pixel is covered by (y-windows x x-windows); keep it modest.

    Note the bound is the product of the per-axis overlap, since a tile
    contributes to a pixel only if its window covers it in both directions.
    """
    tiles = tiling.plan_tiles(h, w, stride)
    cov = coverage_map(h, w, stride)

    def per_axis_overlap(total):
        starts = sorted({t.out_y for t in tiles})
        return max(
            sum(1 for s in starts if s <= p < s + tiling.NET_OUTPUT)
            for p in (total // 2, total - tiling.NET_OUTPUT // 2)
        )

    per_axis = max(per_axis_overlap(h), per_axis_overlap(w))
    assert per_axis <= int(np.ceil(tiling.NET_OUTPUT / stride)) + 1
    assert cov.max() <= per_axis**2, f"{cov.max()} contributions at one pixel"


def test_tiles_do_not_exceed_image():
    for t in tiling.plan_tiles(200, 300, 64):
        assert t.out_y >= 0 and t.out_x >= 0
        assert t.out_y + t.valid_h <= 200
        assert t.out_x + t.valid_w <= 300


@pytest.mark.parametrize("h,w", [(256, 256), (100, 250), (40, 40)])
def test_patch_extraction_returns_network_input(h, w):
    reader = ArrayReader(np.zeros((h, w, 3), np.uint8))
    for t in tiling.plan_tiles(h, w, 64):
        patch = tiling.extract_patch(reader, t)
        assert patch.shape == (tiling.NET_INPUT, tiling.NET_INPUT, 3)
        assert patch.dtype == np.uint8


def test_patch_alignment_is_exact():
    """The output window must land at PAD_OUT inside the patch it came from.

    This is the invariant that keeps nuclei from drifting as they cross tiles:
    patch origin + PAD_OUT must equal the window origin.
    """
    reader = ArrayReader(np.zeros((400, 400, 3), np.uint8))
    tiles = tiling.plan_tiles(400, 400, 64)
    for tile in tiles[::7]:
        assert tile.patch_y + tiling.PAD_OUT == tile.out_y
        assert tile.patch_x + tiling.PAD_OUT == tile.out_x
        patch = tiling.extract_patch(reader, tile)
        assert patch.shape == (tiling.NET_INPUT, tiling.NET_INPUT, 3)
        # the kept output block must exist inside the patch
        oy, ox = tile.out_slice
        assert oy.stop <= tiling.NET_OUTPUT and ox.stop <= tiling.NET_OUTPUT


def test_interior_tiles_need_no_padding():
    """A tile fully inside the image must read exactly the real pixels."""
    img = np.zeros((400, 400, 3), np.uint8)
    img[:, :] = 7
    reader = ArrayReader(img)
    for tile in tiling.plan_tiles(400, 400, 64):
        if tile.patch_y >= 0 and tile.patch_x >= 0:
            patch = tiling.extract_patch(reader, tile)
            assert (patch == 7).all(), "interior tile was altered"


def test_fast_variant_changes_geometry():
    tiling.configure("fast")
    assert tiling.NET_INPUT == 256 and tiling.NET_OUTPUT == 164
    cov = coverage_map(300, 300, 64)
    assert cov.min() >= 1
    tiling.configure("original")
    assert tiling.NET_INPUT == 270 and tiling.NET_OUTPUT == 80