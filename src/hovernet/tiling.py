"""Sliding-window tiling for HoVer-Net.

HoVer-Net in ``original`` mode consumes a 270x270 patch but only emits an 80x80
prediction for the *central* part of that patch (the border is dropped because
the official training used a masked border loss). Every other implementation
treats 270/80 as fixed and tiles accordingly.

This module keeps that contract explicit: it enumerates 80x80 output windows and
returns, for each, the 270x270 patch that must be fed to the network. Windows
are laid out on a stride smaller than 80 so that neighbouring predictions
overlap and can be averaged, which removes the tiling seams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

# Fixed by the architecture - see src/hovernet/net_desc.py (`crop_op` calls).
NET_INPUT = 270
NET_OUTPUT = 80
# Distance from the patch border to the first valid output pixel.
PAD_OUT = (NET_INPUT - NET_OUTPUT) // 2  # 95


@dataclass(frozen=True)
class Tile:
    """One network input and the region of its output that we keep."""

    patch_y: int  # top-left of the network patch (can be negative -> padded)
    patch_x: int
    out_y: int  # top-left of the kept 80x80 block, in image coordinates
    out_x: int
    valid_h: int  # how many rows/cols of the output are inside the image
    valid_w: int

    @property
    def patch(self) -> tuple[int, int]:
        return self.patch_y, self.patch_x

    @property
    def out_slice(self) -> tuple[slice, slice]:
        # The network returns the central window already cropped, so the kept
        # block starts at index 0 of its output.
        return slice(0, self.valid_h), slice(0, self.valid_w)


def configure(mode: str) -> None:
    """Point the module at the input/output geometry of a model variant.

    ``original`` takes 270x270 and emits the central 80x80; ``fast`` takes
    256x256 and emits 164x164. Both follow the official HoVer-Net configs.
    """
    global NET_INPUT, NET_OUTPUT, PAD_OUT
    if mode == "fast":
        NET_INPUT, NET_OUTPUT = 256, 164
    else:
        NET_INPUT, NET_OUTPUT = 270, 80
    PAD_OUT = (NET_INPUT - NET_OUTPUT) // 2


def plan_tiles(
    height: int,
    width: int,
    stride: int = 64,
    out_size: int | None = None,
    pad_out: int | None = None,
) -> list[Tile]:
    """Enumerate output windows covering an ``height x width`` image.

    The window grid is anchored so that the first window starts at 0 and the
    last one ends flush with the image edge.
    """
    out_size = NET_OUTPUT if out_size is None else out_size
    pad_out = PAD_OUT if pad_out is None else pad_out
    if height <= 0 or width <= 0:
        return []
    stride = max(1, min(stride, out_size))

    def starts(total: int) -> list[int]:
        """Evenly spaced window origins covering [0, total).

        Spacing the windows evenly (rather than stepping by a fixed stride and
        clamping the last one) keeps the overlap - and therefore the blending
        weight - uniform right up to the image edge.
        """
        span = max(0, total - out_size)
        n = max(1, int(np.ceil(span / stride)) + 1)
        if n == 1 or span == 0:
            return [0]
        return [int(round(i * span / (n - 1))) for i in range(n)]

    tiles: list[Tile] = []
    for oy in starts(height):
        for ox in starts(width):
            valid_h = min(out_size, height - oy)
            valid_w = min(out_size, width - ox)
            # The patch must be positioned so the output window lands exactly
            # `pad_out` pixels inside it. A negative origin is fine: the reader
            # mirrors the image edge to fill the overhang.
            tiles.append(
                Tile(
                    patch_y=oy - pad_out,
                    patch_x=ox - pad_out,
                    out_y=oy,
                    out_x=ox,
                    valid_h=valid_h,
                    valid_w=valid_w,
                )
            )
    return tiles


def iter_tiles(
    height: int,
    width: int,
    stride: int = 64,
    limit: int | None = None,
) -> Iterator[Tile]:
    tiles = plan_tiles(height, width, stride)
    for i, tile in enumerate(tiles):
        if limit is not None and i >= limit:
            return
        yield tile


def extract_patch(reader, tile: Tile, input_size: int | None = None) -> np.ndarray:
    """Read the network-input patch for ``tile`` and pad it if it hangs off the edge.

    ``reader`` must expose ``read(y0, x0, h, w) -> np.ndarray`` (uint8 HxWx3);
    out-of-bounds rows/cols are filled by mirroring the edge, which is what the
    network saw during training for patch borders.
    """
    input_size = NET_INPUT if input_size is None else input_size
    h = reader.height
    w = reader.width
    py, px = tile.patch_y, tile.patch_x

    top = max(0, -py)
    left = max(0, -px)
    bottom = max(0, py + input_size - h)
    right = max(0, px + input_size - w)

    y0 = max(0, py)
    x0 = max(0, px)
    read_h = min(input_size - top - bottom, h - y0)
    read_w = min(input_size - left - right, w - x0)
    if read_h <= 0 or read_w <= 0:
        raise ValueError(
            f"image is smaller than the network input: {h}x{w} < {input_size}x{input_size}"
        )

    block = reader.read(y0, x0, read_h, read_w)
    if block.shape[0] != read_h or block.shape[1] != read_w:
        raise ValueError(
            f"reader returned {block.shape[:2]} for a ({read_h}, {read_w}) request"
        )
    if top or left or bottom or right:
        block = np.pad(
            block,
            ((top, bottom), (left, right), (0, 0)),
            mode="reflect" if min(block.shape[0], block.shape[1]) > 1 else "edge",
        )
    if block.shape[0] != input_size or block.shape[1] != input_size:
        raise ValueError(
            f"patch {block.shape[:2]} is not {input_size}x{input_size}; "
            "image smaller than the network input?"
        )
    return block


def count_tiles(height: int, width: int, stride: int = 64) -> int:
    return len(plan_tiles(height, width, stride))


def est_grid(height: int, width: int, stride: int = 64) -> tuple[int, int]:
    """Rows x cols of the tile grid (for progress reporting)."""
    tiles = plan_tiles(height, width, stride)
    if not tiles:
        return (0, 0)
    ys = {t.out_y for t in tiles}
    xs = {t.out_x for t in tiles}
    return (len(ys), len(xs))