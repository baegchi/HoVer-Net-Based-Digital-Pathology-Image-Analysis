"""Command-line entry point: analyse an image or a whole-slide image.

    python scripts/run_analysis.py samples/images/pannuke_00000_256x256.png
    python scripts/run_analysis.py slides/tcga.svs --level 1 --max-tiles 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from hovernet.engine import Analyzer  # noqa: E402
from hovernet.readers import ArrayReader, SlideReader, open_reader  # noqa: E402
from pipeline import report, stats as stats_mod  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="PNG/TIFF image or whole-slide image (.svs/.ndpi/.tif)")
    ap.add_argument("-o", "--out", default=str(ROOT / "results"), help="output directory")
    ap.add_argument("--stem", default=None, help="base name for output files")
    ap.add_argument("--checkpoint", default=None, help="override checkpoint path")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--stride", type=int, default=64, help="tile stride in output pixels")
    ap.add_argument("--min-area", type=int, default=20, help="drop nuclei smaller than this")
    ap.add_argument("--level", type=int, default=None, help="pyramid level for whole-slide images")
    ap.add_argument("--max-tiles", type=int, default=None, help="stop after N tiles (debugging)")
    ap.add_argument(
        "--roi",
        default=None,
        metavar="Y,X,H,W",
        help="restrict analysis to a region of interest, in level pixels",
    )
    ap.add_argument("--band-mb", type=int, default=400, help="RAM budget per analysis band")
    ap.add_argument(
        "--no-tissue-filter",
        action="store_true",
        help="analyse every window, including bare slide glass",
    )
    ap.add_argument("--no-images", action="store_true", help="skip PNG output")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    source = Path(args.image)
    if not source.exists():
        print(f"error: no such file: {source}", file=sys.stderr)
        return 2

    analyzer = Analyzer(
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        min_area=args.min_area,
        stride=args.stride,
    )
    if not args.quiet:
        print(
            f"[model ] {Path(analyzer.model_info['checkpoint']).name} "
            f"| {analyzer.nr_types} classes | {analyzer.device} "
            f"| {analyzer.model_info['params'] / 1e6:.1f}M params",
            flush=True,
        )

    started = time.time()
    last = [0.0]

    def progress(done: int, total: int, elapsed: float) -> None:
        if args.quiet or time.time() - last[0] < 0.5 and done < total:
            return
        last[0] = time.time()
        rate = done / max(elapsed, 1e-6)
        eta = (total - done) / max(rate, 1e-6)
        print(f"\r[tiles] {done}/{total}  {rate:5.1f}/s  eta {eta:5.1f}s", end="", flush=True)

    roi = None
    if args.roi:
        roi = tuple(int(v) for v in args.roi.replace("x", ",").split(","))
        if len(roi) != 4:
            print("error: --roi expects Y,X,H,W", file=sys.stderr)
            return 2

    result = analyzer.analyze(
        source,
        stride=args.stride,
        limit_tiles=args.max_tiles,
        progress=progress,
        level=args.level,
        roi=roi,
        band_budget_mb=args.band_mb,
        tissue_only=not args.no_tissue_filter,
    )
    if not args.quiet:
        print()

    # Reload the pixels for the overlay. Only the analysed region is drawn, so the
    # coordinates in `result` line up with `image` exactly (no rescaling).
    from hovernet.readers import _RoiView

    reader = open_reader(source, level=args.level)
    close_reader = getattr(reader, "close", None)
    scaled = False
    if roi:
        view = _RoiView(reader, *roi)
        image = view.read(0, 0, view.height, view.width)
        if close_reader:
            close_reader()
    elif isinstance(reader, SlideReader):
        # Whole-slide run: draw on a downsampled view and scale the nuclei.
        image = reader.thumbnail(max_side=min(result.width, 4096))
        if close_reader:
            close_reader()
        scaled = True
    else:
        image = reader.array
    if scaled:
        # Map nuclei back into thumbnail coordinates.
        sy = image.shape[0] / result.height
        sx = image.shape[1] / result.width
        result.nuclei = [_scale_nucleus(n, sy, sx) for n in result.nuclei]

    statistics = stats_mod.compute(result)
    stem = args.stem or source.stem
    written = report.export_all(
        image, result, statistics, out_dir=args.out, stem=stem, with_images=not args.no_images
    )

    summary = result.summary()
    print(json.dumps(summary, indent=2))
    print("\n[type distribution]")
    for name, row in statistics["by_type"].items():
        print(f"  {name:<14} {row['count']:>6}  ({row['fraction'] * 100:5.1f}%)")
    print(f"\n[write ] {written}")
    print(f"[done  ] {time.time() - started:.1f}s total")
    return 0


def _scale_nucleus(n, sy, sx):
    import copy

    n.centroid_y *= sy
    n.centroid_x *= sx
    n.bbox_y0 *= int(round(sy))
    n.bbox_x0 *= int(round(sx))
    n.bbox_y1 *= int(round(sy))
    n.bbox_x1 *= int(round(sx))
    n.contour = np.round(n.contour * np.array([sx, sy])).astype(np.int32)
    return n


if __name__ == "__main__":
    sys.exit(main())