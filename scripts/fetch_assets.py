"""Download every external asset required to run the analyzer.

All downloads are resumable and idempotent: re-running skips files that are
already present and complete. Nothing here needs an API token.

    python scripts/fetch_assets.py            # everything
    python scripts/fetch_assets.py --only weights
    python scripts/fetch_assets.py --list
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
SAMPLES = ROOT / "samples"
CACHE = Path(os.environ.get("PATHOLOGY_CACHE", ROOT.parent / ".cache"))

UA = "Mozilla/5.0 (X11; Linux x86_64) pathology-analyzer/1.0"

# --- Official HoVer-Net artefacts (see README for provenance) -------------
CONSEP_CKPT_ID = "1FtoTDDnuZShZmQujjaFSLVJLD5sAh2_P"  # hovernet_original_consep_type_tf2pytorch.tar (209 MB)
PREACT_RESNET50_ID = "1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5"  # official ImageNet encoder (94 MB)

# --- Public data ----------------------------------------------------------
PANNUKE_PARQUET = (
    "https://huggingface.co/datasets/RationAI/PanNuke/resolve/main/data/fold1-00000-of-00001.parquet"
)
GDC_SVS_FILE_ID = "18081e12-b7d5-43b4-ab26-7574b94b98f0"  # TCGA-05-4245-01A-01-BS1.svs (85 MB, open access)

GROUPS = ("weights", "samples", "slide")


def log(msg: str) -> None:
    print(f"[fetch] {msg}", flush=True)


def rel(path: Path) -> str:
    """Project-relative path when possible, absolute otherwise (cache lives outside)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def _request(url: str, headers: dict | None = None) -> urllib.request.Request:
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    return urllib.request.Request(url, headers=h)


def download(url: str, dest: Path, expected_bytes: int | None = None) -> Path:
    """Resumable HTTP download with progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_bytes is None or dest.stat().st_size == expected_bytes):
        log(f"skip (present) {rel(dest)}")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}

    for attempt in range(4):
        try:
            with urllib.request.urlopen(_request(url, headers), timeout=90) as resp:
                total = expected_bytes or int(resp.headers.get("content-length", 0)) + start
                mode = "ab" if start and resp.status == 206 else "wb"
                if mode == "wb":
                    start = 0
                last = time.time()
                with open(part, mode) as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        start += len(chunk)
                        now = time.time()
                        if now - last > 2:
                            pct = f"{100 * start / total:5.1f}%" if total else "  ?  "
                            log(f"  {dest.name} {pct} {human(start)}")
                            last = now
            break
        except Exception as exc:  # noqa: BLE001 - network resilience
            wait = 2**attempt
            log(f"  retry {attempt + 1}/4 after {exc!r} (sleep {wait}s)")
            time.sleep(wait)
    else:
        raise RuntimeError(f"failed to download {url}")

    part.replace(dest)
    log(f"saved {rel(dest)} ({human(dest.stat().st_size)})")
    return dest


# --- Google Drive (large files sit behind a "can't scan for viruses" page) -
def gdrive_download(file_id: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1024:
        log(f"skip (present) {rel(dest)}")
        return dest
    base = "https://drive.usercontent.google.com/download"
    url = f"{base}?id={file_id}&export=download"
    resp = urllib.request.urlopen(_request(url), timeout=90)
    body = resp.read()
    if resp.headers.get("content-type", "").startswith("text/html"):
        text = body.decode("utf-8", "replace")
        uuid = re.search(r'name="uuid" value="([^"]+)"', text)
        confirm = re.search(r'name="confirm" value="([^"]+)"', text)
        if not (uuid and confirm):
            raise RuntimeError("Google Drive confirmation form not found - link may have changed")
        url = (
            f"{base}?id={file_id}&export=download"
            f"&confirm={urllib.parse.quote(confirm.group(1))}&uuid={urllib.parse.quote(uuid.group(1))}"
        )
        resp = urllib.request.urlopen(_request(url), timeout=90)
        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        start = part.stat().st_size if part.exists() else 0
        remaining, done, last = total - start, start, time.time()
        mode = "ab" if start and resp.status == 206 else "wb"
        done = start if mode == "ab" else 0
        with open(part, mode) as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 2:
                    log(f"  {dest.name} {100 * done / total:5.1f}% {human(done)}")
                    last = now
        part.replace(dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
    log(f"saved {rel(dest)} ({human(dest.stat().st_size)})")
    return dest


def fetch_weights() -> None:
    ckpt_dir = ASSETS / "checkpoints"
    # Preact-ResNet50 encoder
    resnet = ASSETS / "pretrained_models" / "S1-80x80"
    resnet.mkdir(parents=True, exist_ok=True)
    target = resnet / "latest.pth.tar"
    if not target.exists() or target.stat().st_size < 10_000_000:
        gdrive_download(PREACT_RESNET50_ID, CACHE / "preact_resnet50.pth.tar")
        shutil.copy(CACHE / "preact_resnet50.pth.tar", target)
        log(f"encoder -> {rel(target)}")

    # HoVer-Net CoNSeP checkpoint (tar of .pt files)
    have = sorted(ckpt_dir.glob("*.pth"))
    if not have:
        tar_path = CACHE / "hovernet_original_consep_type_tf2pytorch.tar"
        gdrive_download(CONSEP_CKPT_ID, tar_path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Despite the ".tar" name the upstream artefact is a *legacy pickle*
        # written by torch.save (pre-1.6 container format), so the reliable
        # way in is torch.load rather than tarfile/zipfile.
        import torch

        log(f"unpacking {tar_path.name} ...")
        blob = torch.load(tar_path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and set(blob) == {"desc"}:
            blob = blob["desc"]
        if not isinstance(blob, dict):
            raise RuntimeError(f"unexpected checkpoint structure: {type(blob)}")
        out = ckpt_dir / "hovernet_original_consep_type.pth"
        torch.save(blob, out)
        log(f"extracted {out.name} ({human(out.stat().st_size)}, {len(blob)} tensors)")
    else:
        log(f"skip checkpoint(s) present: {[p.name for p in have]}")

    info = {
        "consep_checkpoint": sorted(q.name for q in ckpt_dir.glob("*.pth")),
        "encoder": str(target),
        "source": "https://github.com/vqdang/hover_net#model-weights",
    }
    (ckpt_dir / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")


def _sample_pngs() -> list[Path]:
    """Pull a handful of H&E patches out of the PanNuke parquet shards."""
    out_dir = SAMPLES / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.png"))
    if len(existing) >= 6:
        log(f"skip (present) {len(existing)} sample images")
        return existing

    parquet = CACHE / "pannuke_fold1.parquet"
    download(PANNUKE_PARQUET, parquet)

    import io as _io

    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(parquet)
    cols = table.column_names
    log(f"PanNuke columns: {cols}")

    img_col = next((c for c in ("image", "images", "img") if c in cols), cols[0])
    written = []
    for i in range(table.num_rows):
        if len(written) >= 6:
            break
        value = table.column(img_col)[i].as_py()
        if isinstance(value, dict):
            raw = value.get("bytes") or value.get("path")
        elif isinstance(value, (bytes, bytearray)):
            raw = bytes(value)
        else:
            raw = str(value)
        try:
            img = Image.open(_io.BytesIO(raw) if isinstance(raw, (bytes, bytearray)) else raw)
        except Exception:  # noqa: BLE001
            continue
        img = img.convert("RGB")
        arr = np.asarray(img)
        if arr.shape[0] < 256:
            continue
        name = f"pannuke_{i:05d}_{arr.shape[1]}x{arr.shape[0]}.png"
        img.save(out_dir / name)
        written.append(out_dir / name)
        log(f"sample -> {name}")
    return written


def fetch_samples() -> None:
    _sample_pngs()
    # Small synthetic-free sanity image so the app always has something to load.
    import numpy as np
    from PIL import Image

    demo = SAMPLES / "images" / "demo_placeholder.png"
    if not demo.exists():
        rng = np.random.default_rng(0)
        arr = (rng.integers(200, 240, (256, 256, 3))).astype("uint8")
        Image.fromarray(arr).save(demo)
        log("placeholder -> demo_placeholder.png")


def fetch_slide() -> None:
    slide_dir = SAMPLES / "slides"
    slide_dir.mkdir(parents=True, exist_ok=True)
    dest = slide_dir / "TCGA-05-4245-01A-01-BS1.svs"
    if dest.exists() and dest.stat().st_size > 10_000_000:
        log(f"skip (present) {rel(dest)}")
        return
    url = f"https://api.gdc.cancer.gov/data/{GDC_SVS_FILE_ID}"
    download(url, dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=GROUPS, action="append", help="fetch only these groups")
    ap.add_argument("--list", action="store_true", help="show what would be downloaded")
    args = ap.parse_args()

    groups = args.only or list(GROUPS)
    if args.list:
        print("weights  : CoNSeP HoVer-Net checkpoint (209 MB) + Preact-ResNet50 encoder (94 MB)")
        print("samples  : PanNuke H&E patches via HuggingFace parquet")
        print("slide    : TCGA-05-4245 .svs from the NCI GDC open-access API (85 MB)")
        return 0

    CACHE.mkdir(parents=True, exist_ok=True)
    for group in groups:
        log(f"--- {group} ---")
        {"weights": fetch_weights, "samples": fetch_samples, "slide": fetch_slide}[group]()
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())