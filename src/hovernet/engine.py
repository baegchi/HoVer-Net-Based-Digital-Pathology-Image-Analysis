"""The analyzer engine: run HoVer-Net over an image or a slide and return nuclei.

Memory discipline is the point of this module. Tiles are extracted, pushed
through the network in fixed-size batches and immediately folded into the
output accumulators, so peak memory is governed by ``batch_size`` and the tile
size, not by the size of the input.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import torch

from . import tiling
from .checkpoint import TYPE_COLORS, build_model, type_names
from .post_proc import process
from .readers import ArrayReader, SlideReader, _RoiView

Progress = Callable[[int, int, float], None]


@dataclass
class Nucleus:
    """One segmented nucleus in global image coordinates."""

    id: int
    type_id: int
    type_name: str
    type_prob: float
    area: int
    centroid_y: float
    centroid_x: float
    bbox_y0: int
    bbox_x0: int
    bbox_y1: int
    bbox_x1: int
    contour: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2), np.int32))
    tile: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type_id": self.type_id,
            "type_name": self.type_name,
            "type_prob": round(float(self.type_prob), 4),
            "area": int(self.area),
            "centroid": [round(float(self.centroid_y), 1), round(float(self.centroid_x), 1)],
            "bbox": [self.bbox_y0, self.bbox_x0, self.bbox_y1, self.bbox_x1],
            "tile": self.tile,
            "contour": self.contour.astype(int).tolist(),
        }


@dataclass
class AnalysisResult:
    source: str
    width: int
    height: int
    stride: int
    batch_size: int
    device: str
    nr_types: int
    type_names: dict
    mpp: float | None
    min_area: int
    tiles_total: int
    tiles_done: int
    inference_seconds: float
    postproc_seconds: float
    model_info: dict
    read_seconds: float = 0.0
    tissue_fraction: float = 1.0
    nuclei: list[Nucleus] = field(default_factory=list)

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for n in self.nuclei:
            counts[n.type_name] = counts.get(n.type_name, 0) + 1
        return {
            "source": self.source,
            "image_size": {"width": self.width, "height": self.height},
            "tiles": {
                "total": self.tiles_total,
                "processed": self.tiles_done,
                "stride": self.stride,
                "tissue_fraction": round(self.tissue_fraction, 4),
            },
            "nuclei_total": len(self.nuclei),
            "counts_by_type": counts,
            "microns_per_pixel": self.mpp,
            "device": self.device,
            "timing_seconds": {
                "inference": round(self.inference_seconds, 3),
                "post_processing": round(self.postproc_seconds, 3),
                "total": round(self.inference_seconds + self.postproc_seconds, 3),
            },
            "parameters": {
                "min_area_px": self.min_area,
                "tile_size": tiling.NET_INPUT,
                "output_tile": tiling.NET_OUTPUT,
                "batch_size": self.batch_size,
            },
            "model": {k: v for k, v in self.model_info.items() if k != "missing_keys"},
        }


class Analyzer:
    """Holds the model and runs tiled inference."""

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "auto",
        batch_size: int = 8,
        min_area: int = 20,
        stride: int = 64,
    ):
        self.net, self.model_info = build_model(checkpoint=checkpoint, device=device)
        self.device = self.model_info["device"]
        self.nr_types = self.model_info["nr_types"]
        raw = self.model_info.get("type_names", {})
        self.type_names = {int(k): v for k, v in raw.items()} or type_names(self.nr_types)
        self.batch_size = max(1, batch_size)
        self.min_area = max(1, min_area)
        self.stride = stride
        # Geometry depends on which variant the checkpoint was trained with.
        tiling.configure(self.model_info["mode"])
        self.patch = tiling.NET_INPUT
        self.out = tiling.NET_OUTPUT
        self.read_seconds = 0.0
        self.forward_seconds = 0.0

    # -- low level ---------------------------------------------------------
    def _forward(self, batch: np.ndarray) -> dict:
        """batch: HWC uint8 -> [tp, np, hv] channels, matching `post_proc.process`."""
        tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
        tensor = tensor.permute(0, 3, 1, 2).float()
        with torch.inference_mode():
            pred = self.net(tensor)
            # Outputs are still NCHW here, so softmax over dim=1 (the official
            # code permutes to NHWC first and uses dim=-1 - same thing).
            np_prob = torch.softmax(pred["np"], dim=1)[:, 1]
            hv_map = pred["hv"].permute(0, 2, 3, 1)
            tp_map = pred["tp"] if "tp" in pred else None
            if tp_map is not None:
                tp_map = torch.argmax(torch.softmax(tp_map, dim=1), dim=1).unsqueeze(-1)
            np_prob = np_prob.unsqueeze(-1).cpu().numpy()
            hv_map = hv_map.cpu().numpy()
            tp_map = tp_map.cpu().numpy() if tp_map is not None else None
        parts = [np_prob, hv_map] if tp_map is None else [tp_map, np_prob, hv_map]
        return {"inst": np.concatenate(parts, axis=-1).astype(np.float32)}

    # -- public API --------------------------------------------------------
    def analyze(
        self,
        source,
        stride: int | None = None,
        limit_tiles: int | None = None,
        progress: Progress | None = None,
        level: int | None = None,
        roi: tuple[int, int, int, int] | None = None,
        band_budget_mb: int = 400,
        tissue_only: bool = True,
        min_tissue: float = 0.10,
    ) -> AnalysisResult:
        """Segment and classify every nucleus in ``source``.

        ``source`` may be a file path (PNG/TIFF/SVS) or an HWC uint8 array.
        ``roi`` restricts the analysis to ``(y, x, height, width)``.
        ``tissue_only`` skips windows that fall on bare slide glass.

        The image is walked in horizontal bands whose height is chosen so the
        running accumulators stay inside ``band_budget_mb``. Nothing
        proportional to the full image size is ever allocated, which is what
        lets a 18000 x 20000 slide run in a normal amount of RAM.
        """
        from .readers import open_reader

        source_reader = open_reader(source, level=level)
        reader = source_reader
        stride = stride or self.stride

        if roi:
            # Present the ROI as its own virtual image.
            y0, x0, rh, rw = (int(v) for v in roi)
            rh, rw = min(rh, reader.height - y0), min(rw, reader.width - x0)
            if rh <= 0 or rw <= 0:
                raise ValueError(f"ROI {roi} lies outside the {reader.height}x{reader.width} image")
            reader = _RoiView(reader, y0, x0, rh, rw)
        height, width = reader.height, reader.width

        # Optionally skip background windows: on a whole-slide image most of the
        # area is bare glass, and running the network over it wastes the compute.
        tissue = None
        if tissue_only and isinstance(source_reader, SlideReader) and not roi:
            tissue = source_reader.find_tissue()

        tiles = tiling.plan_tiles(height, width, stride)
        if tissue is not None:
            tiles = [
                t
                for t in tiles
                if tissue.tile_fraction(t.out_y, t.out_x, t.valid_h, t.valid_w) >= min_tissue
            ]
        self.tissue_fraction = tissue.fraction if tissue is not None else 1.0
        if limit_tiles:
            tiles = tiles[:limit_tiles]
        if not tiles:
            raise ValueError(
                "no analysable tissue found. Lower `min_tissue`, set an explicit --roi, "
                "or pass tissue_only=False."
            )

        # 5 float32 channels of accumulator per pixel.
        bytes_per_row = width * 5 * 4
        band_rows = max(256, min(height, int(band_budget_mb * 1024 * 1024 // max(1, bytes_per_row))))
        bands = [
            (y0, min(y0 + band_rows, height))
            for y0 in range(0, height, band_rows)
        ]

        t0 = time.time()
        self.read_seconds = 0.0
        self.forward_seconds = 0.0
        nuclei: list[Nucleus] = []
        done = 0
        post_t0 = time.time()
        total_planned = len(tiles)
        for by0, by1 in bands:
            band_tiles = [t for t in tiles if by0 <= t.out_y < by1]
            if limit_tiles:
                band_tiles = band_tiles[: max(0, limit_tiles - done)]
            if not band_tiles:
                continue
            tp_acc, np_acc, hv_acc, _ = self._run_band(
                reader, band_tiles, stride, progress, base_done=done, base_total=total_planned
            )
            done += len(band_tiles)
            nuclei.extend(self._postprocess(np_acc, hv_acc, tp_acc))
            del tp_acc, np_acc, hv_acc  # free the band before the next one
        post_s = time.time() - post_t0 - self.forward_seconds
        infer_s = self.forward_seconds

        for i, n in enumerate(nuclei, start=1):
            n.id = i

        result = AnalysisResult(
            source=getattr(source_reader, "name", "array"),
            width=width,
            height=height,
            stride=stride,
            batch_size=self.batch_size,
            device=self.device,
            nr_types=self.nr_types,
            type_names=self.type_names,
            mpp=getattr(reader, "mpp", None),
            min_area=self.min_area,
            tiles_total=len(tiles),
            tiles_done=done,
            inference_seconds=infer_s,
            postproc_seconds=post_s,
            model_info=self.model_info,
            read_seconds=self.read_seconds,
            tissue_fraction=getattr(self, "tissue_fraction", 1.0),
            nuclei=nuclei,
        )
        close = getattr(source_reader, "close", None)
        if close:
            close()
        return result

# -- streaming bands ---------------------------------------------------
    def _run_band(self, reader, band_tiles, stride, progress, base_done=0, base_total=0) -> tuple:
        """Accumulate predictions for one horizontal band, then normalise.

        Returns the average prediction maps plus the per-pixel weight map (which
        tells us how many tiles contributed, and which pixels were never seen).
        """
        height, width = reader.height, reader.width
        tp_acc = np.zeros((height, width), np.float32)
        np_acc = np.zeros((height, width), np.float32)
        hv_acc = np.zeros((height, width, 2), np.float32)
        weight = np.zeros((height, width), np.float32)

        for start in range(0, len(band_tiles), self.batch_size):
            group = band_tiles[start : start + self.batch_size]
            t_read = time.time()
            patches = np.stack([tiling.extract_patch(reader, t) for t in group])
            self.read_seconds += time.time() - t_read
            t_fwd = time.time()
            inst = self._forward(patches)["inst"]
            self.forward_seconds += time.time() - t_fwd
            for k, tile in enumerate(group):
                y, x = tile.out_y, tile.out_x
                h, w = tile.valid_h, tile.valid_w
                # The network already emits exactly the central 80x80 window, so
                # it maps straight onto (out_y, out_x) - no extra cropping.
                pred = inst[k][:h, :w]
                tp_acc[y : y + h, x : x + w] += pred[..., 0]
                np_acc[y : y + h, x : x + w] += pred[..., 1]
                hv_acc[y : y + h, x : x + w, 0] += pred[..., 2]
                hv_acc[y : y + h, x : x + w, 1] += pred[..., 3]
                weight[y : y + h, x : x + w] += 1.0
            if progress:
                done = base_done + min(start + len(group), len(band_tiles))
                progress(done, base_total, self.read_seconds + self.forward_seconds)

        # Undo the overlap averaging; never-visited pixels keep a zero map.
        seen = weight > 0
        safe = np.where(seen, weight, 1.0)
        tp_acc /= safe
        np_acc /= safe
        hv_acc /= safe[..., None]
        return tp_acc, np_acc, hv_acc, weight

    # -- post processing ---------------------------------------------------
    def _postprocess(self, np_acc, hv_acc, tp_acc) -> list[Nucleus]:
        """Segment one band's blended maps into individual nuclei.

        The band is processed in overlapping blocks so the watershed never
        allocates more than a block at a time.
        """
        nuclei: list[Nucleus] = []
        h, w = np_acc.shape
        block = 1024  # working window for the watershed step
        margin = 32  # keeps nuclei whole across block seams

        ys = list(range(0, max(1, h), block))
        xs = list(range(0, max(1, w), block))
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(y0 + block, h), min(x0 + block, w)
                sy0, sx0 = max(0, y0 - margin), max(0, x0 - margin)
                sy1, sx1 = min(h, y1 + margin), min(w, x1 + margin)
                np_patch = np_acc[sy0:sy1, sx0:sx1]
                hv_patch = hv_acc[sy0:sy1, sx0:sx1]
                tp_patch = np.clip(tp_acc[sy0:sy1, sx0:sx1], 0, self.nr_types - 1)
                if np_patch.size == 0 or np_patch.max() < 0.5:
                    continue
                pred_map = np.concatenate(
                    [tp_patch[..., None], np_patch[..., None], hv_patch], axis=-1
                )
                inst, info = process(pred_map, self.nr_types, return_centroids=True)
                nuclei.extend(self._collect(info, (sy0, sx0, y1, x1)))
        return nuclei

    def _collect(self, info: dict, region) -> list[Nucleus]:
        sy0, sx0, clip_y, clip_x = region
        margin = 32
        out: list[Nucleus] = []
        for inst_id, rec in info.items():
            if rec["type"] is None:
                continue
            cy, cx = rec["centroid"]
            # Drop nuclei that belong to the neighbouring block (they are picked
            # up there instead, so nothing is double counted).
            if cy + sy0 >= clip_y + margin or cx + sx0 >= clip_x + margin:
                continue
            contour = rec["contour"].copy()
            if contour.size:
                contour[:, 0] += sx0  # X -> global
                contour[:, 1] += sy0  # Y -> global
            bbox = rec["bbox"].flatten()
            ry0, rx0, ry1, rx1 = bbox[0] + sy0, bbox[1] + sx0, bbox[2] + sy0, bbox[3] + sx0
            # Area measured from the polygon, consistent for every nucleus.
            area = int(_contour_area(contour)) if contour.size else 0
            if area < self.min_area:
                continue
            out.append(
                Nucleus(
                    id=0,  # renumbered after all blocks are merged
                    # Type 0 is the CoNSeP "background" class. The upstream
                    # post-processor can still assign it to an instance when no
                    # other class is present, so surface that honestly rather
                    # than hiding it.
                    type_id=int(rec["type"]),
                    type_name=self.type_names.get(int(rec["type"]), "Unclassified"),
                    type_prob=float(rec["type_prob"]),
                    area=area,
                    centroid_y=float(cy + sy0),
                    centroid_x=float(cx + sx0),
                    bbox_y0=int(ry0),
                    bbox_x0=int(rx0),
                    bbox_y1=int(ry1),
                    bbox_x1=int(rx1),
                    contour=contour,
                )
            )
        return out

    # -- overlay helpers ---------------------------------------------------
    @staticmethod
    def instance_mask(nuclei: Iterable[Nucleus], height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), np.int32)
        for n in nuclei:
            if n.contour.size:
                cv2_fill(mask, n)
        return mask

    @staticmethod
    def type_colors() -> dict:
        return dict(TYPE_COLORS)


def cv2_fill(mask: np.ndarray, nucleus: Nucleus) -> None:
    import cv2

    pts = nucleus.contour.astype(np.int32)
    if pts.ndim == 2 and pts.shape[0] >= 3:
        cv2.fillPoly(mask, [pts], color=int(nucleus.id))

def _contour_area(contour: np.ndarray) -> float:
    """Shoelace area of an (N, 2) contour given as (x, y) pairs."""
    x = contour[:, 0].astype(np.float64)
    y = contour[:, 1].astype(np.float64)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
