"""Locate and load the pretrained HoVer-Net weights.

The upstream CoNSeP artefact is a flat ``state_dict`` for the three-head
network (``tp``/``np``/``hv``) produced by ``create_model(mode="original",
nr_types=4)``. This module resolves the checkpoint path, validates that the
tensors actually match the architecture, and hands back a ready model.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch

from .net_desc import create_model

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT_DIR = ROOT / "assets" / "checkpoints"
DEFAULT_CKPT_NAME = "hovernet_original_consep_type.pth"

# CoNSeP taxonomy as used by `type_info.json` in the official repo.
CONSEP_TYPES = {
    0: "Unclassified",  # background class; only assigned when no other fits
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
}

# PanNuke adds a "dead karyotic" class and ships the `fast` architecture.
PANNUKE_TYPES = {
    0: "Unclassified",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Dead-karyotic",
}

TYPE_COLORS = {
    1: (255, 0, 0),
    2: (0, 200, 80),
    3: (0, 90, 255),
    4: (255, 200, 0),
    5: (200, 0, 200),
}


class CheckpointError(RuntimeError):
    pass


def find_checkpoint(path: str | Path | None = None) -> Path:
    """Return a usable checkpoint path or explain why none is available."""
    if path:
        p = Path(path).expanduser()
        if not p.exists():
            raise CheckpointError(
                f"checkpoint not found: {p}\nRun `python scripts/fetch_assets.py --only weights`."
            )
        return p

    candidates = sorted(DEFAULT_CKPT_DIR.glob("*.pth"))
    # Prefer the PanNuke checkpoint when present: the bundled sample images are
    # PanNuke, so it is the in-domain default. Both work; pick explicitly with
    # --checkpoint to analyse CoNSeP-style material instead.
    for candidate in candidates:
        if "pannuke" in candidate.name:
            return candidate
    return candidates[0] if candidates else None
    raise CheckpointError(
        "No HoVer-Net checkpoint found in "
        f"{DEFAULT_CKPT_DIR}.\nRun `python scripts/fetch_assets.py --only weights` first."
    )


def load_state_dict(path: str | Path | None = None) -> OrderedDict:
    """Load a checkpoint that may or may not be wrapped in a 'desc' key."""
    ckpt_path = find_checkpoint(path)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        if "desc" in blob and isinstance(blob["desc"], dict):
            blob = blob["desc"]
        elif "model_state_dict" in blob:
            blob = blob["model_state_dict"]
        elif "state_dict" in blob:
            blob = blob["state_dict"]
    if not isinstance(blob, dict):
        raise CheckpointError(f"{ckpt_path} does not contain a state dict (got {type(blob)})")
    return OrderedDict((k, v) for k, v in blob.items())


def detect_mode(state_dict: dict) -> str:
    """Recover which architecture variant a checkpoint was trained with.

    The `fast` model uses 3x3 dense-block kernels and has no TF-style padding
    layer before `u1`; `original` uses 5x5 and does. Probing the conv kernel
    size is unambiguous and avoids the user having to know which is which.
    """
    probes = [
        (k, v)
        for k, v in state_dict.items()
        if "decoder.np.u3.dense.units." in k and k.endswith("conv2.weight")
    ]
    if probes:
        return "fast" if probes[0][1].shape[-1] == 3 else "original"
    return "original"


def detect_type_names(path: str | Path, nr_types: int) -> dict[int, str]:
    """Pick the taxonomy that matches the checkpoint."""
    if nr_types >= 6:
        table = PANNUKE_TYPES
    else:
        table = CONSEP_TYPES
    names = {}
    for k, v in table.items():
        if k < nr_types:
            names[k] = v
    # Any class beyond the known table keeps a generic label.
    for k in range(nr_types):
        names.setdefault(k, f"Class {k}")
    return names


def detect_nr_types(state_dict: dict) -> int:
    """Infer the number of cell classes from the `tp` head's output width."""
    for key, value in state_dict.items():
        if key == "decoder.tp.u0.conv.weight" and len(value.shape) == 4:
            return int(value.shape[0])
    return 0


def build_model(
    ckpt: str | Path | None = None,
    checkpoint: str | Path | None = None,
    nr_types: int | None = None,
    device: str = "auto",
    mode: str = "auto",
) -> tuple[torch.nn.Module, dict]:
    """Instantiate HoVer-Net and load pretrained weights.

    Returns the model in eval mode plus an info dict describing what was loaded.
    """
    ckpt = checkpoint or ckpt
    state = load_state_dict(ckpt)
    detected = detect_nr_types(state)
    if mode == "auto":
        mode = detect_mode(state)
    if nr_types is None:
        nr_types = detected
    if nr_types == 0:
        raise CheckpointError(
            "Checkpoint has no `tp` (type) head; this analyzer needs a "
            "type-enabled checkpoint such as the CoNSeP one."
        )

    net = create_model(input_ch=3, nr_types=nr_types, freeze=True, mode=mode)

    # Strict load, but translate the handful of legacy key spellings that the
    # TF->PyTorch conversion introduced.
    own = net.state_dict()
    remapped, renamed = {}, 0
    for key, value in state.items():
        target = own.get(key)
        if target is None and key.startswith("decoder."):
            # e.g. decoder.tp.* -> decoder.tp.*  (unchanged) but handle the
            # "_256x256"-suffixed names used by some conversions.
            alt = key.replace(".tp_256x256.", ".tp.")
            target = own.get(alt)
            if target is not None:
                remapped[alt] = value
                renamed += 1
                continue
        if target is not None and tuple(target.shape) == tuple(value.shape):
            remapped[key] = value
        else:
            raise CheckpointError(
                f"Checkpoint tensor {key!r} with shape {tuple(value.shape)} does not fit the "
                f"model slot of shape {tuple(target.shape) if target is not None else None}. "
                "This checkpoint is probably for a different HoVer-Net variant."
            )

    missing, unexpected = net.load_state_dict(remapped, strict=False)
    # `num_batches_tracked` is absent from the TF-converted checkpoint, and
    # `upsample2x.unpool_mat` is a constant buffer - neither matters.
    ignorable = ("upsample2x", "num_batches_tracked")
    critical = [k for k in missing if not any(t in k for t in ignorable)]
    if critical:
        raise CheckpointError(
            f"{len(critical)} weights missing from the checkpoint, e.g. {critical[:3]}"
        )
    if renamed:
        print(f"[checkpoint] remapped {renamed} legacy keys")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    net.eval()

    info = {
        "checkpoint": str(Path(find_checkpoint(ckpt))),
        "type_names": {str(k): v for k, v in detect_type_names(find_checkpoint(ckpt), nr_types).items()},
        "nr_types": nr_types,
        "mode": mode,
        "device": device,
        "params": sum(p.numel() for p in net.parameters()),
        "missing_keys": list(missing),
    }
    return net, info


def type_names(nr_types: int) -> dict[int, str]:
    """Backwards-compatible helper for the CoNSeP taxonomy."""
    return detect_type_names(find_checkpoint(), nr_types)