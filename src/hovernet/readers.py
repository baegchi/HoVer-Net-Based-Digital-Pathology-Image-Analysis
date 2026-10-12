"""Random-access image readers and slide tissue detection.

The analyzer only ever asks a reader for the tile it is about to process, so a
whole-slide image costs the same memory as a thumbnail. Every reader exposes
the same tiny interface::

    height, width          # size at the analysis resolution
    read(y, x, h, w)       -> uint8 HxWx3
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # pathology images are huge by design


class ArrayReader:
    """Serve tiles from an in-memory array."""

    def __init__(self, array: np.ndarray):
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.shape[2] == 4:
            array = array[:, :, :3]
        self.array = np.ascontiguousarray(array, dtype=np.uint8)
        self.height, self.width = self.array.shape[:2]
        self.name = "array"
        self.mpp = None

    def read(self, y: int, x: int, h: int, w: int) -> np.ndarray:
        return self.array[y : y + h, x : x + w]


class FileReader(ArrayReader):
    """Serve tiles from a regular image file, decoded once up front."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with Image.open(self.path) as img:
            self.array = np.asarray(img.convert("RGB"), dtype=np.uint8)
        super().__init__(self.array)
        self.name = self.path.name


class SlideReader:
    """Serve tiles from a whole-slide image through OpenSlide."""

    def __init__(self, path: str | Path, level: int | None = None):
        import openslide

        self.path = Path(path)
        self.slide = openslide.OpenSlide(str(self.path))
        self.name = self.path.name

        full_w, full_h = self.slide.level_dimensions[0]
        requested = 0 if level is None else int(level)
        self.level = requested if requested < self.slide.level_count else 0
        self.width, self.height = self.slide.level_dimensions[self.level]
        # Level 0 is full resolution; higher levels are downsampled by this.
        self.downsample = (full_w / self.width, full_h / self.height)
        self.mpp = _read_mpp(self.slide, self.downsample)

    def read(self, y: int, x: int, h: int, w: int) -> np.ndarray:
        region = self.slide.read_region(
            (int(x * self.downsample[0]), int(y * self.downsample[1])),
            self.level,
            (int(w), int(h)),
        )
        return np.asarray(region.convert("RGB"), dtype=np.uint8)

    def thumbnail(self, max_side: int = 2048) -> np.ndarray:
        """Downsample view of the whole slide, read from the pyramid.

        Reading a small window out of level 0 would only show one corner of the
        slide (usually bare glass), so read the pyramid level closest to
        `max_side` in full and resize.
        """
        target_w = max(1, int(self.width * min(max_side / self.width, max_side / self.height, 1.0)))
        best = min(
            range(self.slide.level_count),
            key=lambda lv: abs(self.slide.level_dimensions[lv][0] - target_w),
        )
        lw, lh = self.slide.level_dimensions[best]
        region = self.slide.read_region((0, 0), best, (lw, lh))
        thumb = np.asarray(region.convert("RGB"), dtype=np.uint8)
        if max(lw, lh) > max_side:
            scale = max_side / max(lw, lh)
            thumb = cv2.resize(
                thumb,
                (max(1, int(lw * scale)), max(1, int(lh * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return thumb

    def find_tissue(self, thumb_side: int = 1024) -> "TissueMask":
        """Locate tissue from a low-resolution saturation scan.

        A whole-slide image is mostly bare glass; running the network over that
        wastes nearly all the compute. Scanning once at low resolution costs a
        few megabytes and lets the pipeline skip empty background entirely.
        """
        thumb = self.thumbnail(max_side=thumb_side)
        return TissueMask(tissue_mask(thumb), self.width, self.height)

    def close(self) -> None:
        self.slide.close()


def _read_mpp(slide, downsample) -> float | None:
    """Microns-per-pixel at the analysed level, from the slide metadata."""
    try:
        import openslide

        mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        if mpp:
            return float(mpp) * float(downsample[0])
    except Exception:  # noqa: BLE001 - vendor metadata is best-effort
        pass
    return None


class _RoiView:
    """Present a sub-rectangle of another reader as a standalone image."""

    def __init__(self, parent, y: int, x: int, height: int, width: int):
        self.parent = parent
        self.y, self.x = y, x
        self.height, self.width = height, width
        self.name = getattr(parent, "name", "array")
        self.mpp = getattr(parent, "mpp", None)

    @property
    def origin(self) -> tuple[int, int]:
        return self.y, self.x

    def read(self, y: int, x: int, h: int, w: int) -> np.ndarray:
        return self.parent.read(self.y + y, self.x + x, h, w)


def tissue_mask(thumbnail: np.ndarray, max_brightness: int = 225) -> np.ndarray:
    """Boolean tissue mask for a thumbnail.

    Bare scanner glass is near-white and near-grey; stained tissue is markedly
    darker in every channel. Brightness is used rather than HSV saturation
    because averaging a downsampled tile with the surrounding glass dilutes
    saturation, which makes a saturation threshold behave differently at every
    pyramid level.

    Brightness is scale-stable, so one threshold works on both a 1024 px
    thumbnail and a full-resolution crop.
    """
    if thumbnail.ndim == 2:
        thumbnail = np.stack([thumbnail] * 3, -1)
    gray = cv2.cvtColor(thumbnail[:, :, :3], cv2.COLOR_RGB2GRAY)
    mask = (gray < max_brightness).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # drop specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # close pinholes
    return mask.astype(bool)


class TissueMask:
    """Low-resolution tissue mask with tile lookup and ROI suggestion."""

    def __init__(self, mask: np.ndarray, width: int, height: int):
        self.mask = mask
        self.thumb_h, self.thumb_w = mask.shape[:2]
        self.sx = self.thumb_w / float(width)
        self.sy = self.thumb_h / float(height)
        self.fraction = float(mask.mean())

    def tile_fraction(self, y: int, x: int, h: int, w: int) -> float:
        """Tissue coverage of a tile, in analysis-resolution coordinates."""
        y0 = max(0, int(y * self.sy))
        x0 = max(0, int(x * self.sx))
        y1 = min(self.thumb_h, max(y0 + 1, int((y + h) * self.sy)))
        x1 = min(self.thumb_w, max(x0 + 1, int((x + w) * self.sx)))
        if y1 <= y0 or x1 <= x0:
            return 0.0
        return float(self.mask[y0:y1, x0:x1].mean())

    def best_roi(self, size: int, min_tissue: float = 0.5) -> tuple[int, int, int, int]:
        """The `(y, x, h, w)` window of `size` containing the most tissue.

        Window sums come from an integral image, so the search costs the same
        whatever the window size. Returns `(0, 0, 0, 0)` when no window is
        tissue-rich, which the caller should treat as "no tissue on this slide".
        """
        th, tw = self.thumb_h, self.thumb_w
        win_y = max(1, int(size * self.sy))
        win_x = max(1, int(size * self.sx))
        if win_y >= th or win_x >= tw:
            return (0, 0, size, size)

        integral = np.pad(self.mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        totals = (
            integral[win_y:, win_x:]
            - integral[:-win_y, win_x:]
            - integral[win_y:, :-win_x]
            + integral[:-win_y, :-win_x]
        )
        best_y, best_x = np.unravel_index(int(totals.argmax()), totals.shape)
        roi = (int(best_y / self.sy), int(best_x / self.sx), size, size)
        if self.tile_fraction(roi[0], roi[1], size, size) < min_tissue:
            return (0, 0, 0, 0)
        return roi


def open_reader(source: str | Path | np.ndarray, level: int | None = None):
    """Dispatch to the right reader for a path or an in-memory array."""
    if isinstance(source, np.ndarray):
        return ArrayReader(source)
    suffix = Path(source).suffix.lower()
    slide_exts = {".svs", ".ndpi", ".scn", ".vms", ".bif", ".mrxs", ".tif", ".tiff", ".ptif"}
    if suffix in slide_exts:
        import openslide

        try:
            return SlideReader(source, level=level)
        except openslide.OpenSlideUnsupportedFormatError:
            pass  # a plain (non-pyramidal) TIFF - fall back to PIL
    return FileReader(source)