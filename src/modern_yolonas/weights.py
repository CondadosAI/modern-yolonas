"""Download and load pretrained super-gradients checkpoints.

Downloads to ``~/.cache/modern_yolonas/`` and remaps state_dict keys
from the super-gradients module hierarchy to ours.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from urllib.request import urlopen

import torch
from torch import nn

logger = logging.getLogger(__name__)

WEIGHT_URLS = {
    "yolo_nas_s": "https://sg-hub-nv.s3.amazonaws.com/models/yolo_nas_s_coco.pth",
    "yolo_nas_m": "https://sg-hub-nv.s3.amazonaws.com/models/yolo_nas_m_coco.pth",
    "yolo_nas_l": "https://sg-hub-nv.s3.amazonaws.com/models/yolo_nas_l_coco.pth",
}

WEIGHT_CHECKSUMS: dict[str, str] = {
    # SHA256 checksums for official super-gradients pretrained weights
    # Populated after first verified download — empty means skip verification
}

CACHE_DIR = Path(os.environ.get("YOLONAS_CACHE_DIR", Path.home() / ".cache" / "modern_yolonas"))


_LICENSE_WARNING = (
    "The pretrained weights are from Deci AI's super-gradients and are licensed "
    "under the Super Gradients Model EULA (non-commercial use only). "
    "See https://docs.deci.ai/super-gradients/latest/LICENSE.YOLONAS.html for details."
)


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a Rich progress bar."""
    from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

    response = urlopen(url)  # noqa: S310
    total = int(response.headers.get("Content-Length", 0))

    with (
        Progress(BarColumn(), DownloadColumn(), TransferSpeedColumn()) as progress,
        open(dest, "wb") as f,
    ):
        task = progress.add_task("Downloading", total=total or None)
        while chunk := response.read(1024 * 64):
            f.write(chunk)
            progress.advance(task, len(chunk))


def _download(variant: str) -> Path:
    if variant not in WEIGHT_URLS:
        raise ValueError(f"Unknown variant: {variant!r}. Must be one of {list(WEIGHT_URLS)}")
    url = WEIGHT_URLS[variant]
    filename = url.rsplit("/", 1)[-1]
    dest = CACHE_DIR / filename
    logger.warning(_LICENSE_WARNING)
    if dest.exists():
        return dest
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s weights from %s ...", variant, url)
    try:
        _download_with_progress(url, dest)
    except Exception as exc:
        # Clean up partial download
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Failed to download {variant} weights from {url}") from exc

    # Verify checksum if available
    expected = WEIGHT_CHECKSUMS.get(variant)
    if expected:
        actual = _sha256(dest)
        if actual != expected:
            dest.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {variant}: expected {expected[:16]}..., got {actual[:16]}... "
                f"The file has been deleted. Please retry the download."
            )
    return dest


def _sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 64):
            h.update(chunk)
    return h.hexdigest()


def _strip_prefix(key: str) -> str:
    """Remove common DDP / checkpoint wrapper prefixes."""
    for prefix in ("net.", "module.", "ema_model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def remap_state_dict(raw_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap super-gradients state_dict keys to our module hierarchy.

    Super-gradients wraps the model in ``CustomizableDetector`` with::

        backbone  → backbone.stem, backbone.stage1 … backbone.stage4, backbone.context_module
        neck      → neck.neck1 … neck.neck4
        heads     → heads.head1 … heads.head3

    Our ``YoloNAS`` uses the same attribute names, so the only work is
    stripping DDP/EMA prefixes.
    """
    remapped = {}
    for key, value in raw_sd.items():
        new_key = _strip_prefix(key)
        remapped[new_key] = value
    return remapped


def load_pretrained(
    model: nn.Module,
    variant: str,
    strict: bool = True,
    prefer_ema: bool = False,
) -> nn.Module:
    """Download checkpoint and load into model.

    Args:
        model: A ``YoloNAS`` instance (or any nn.Module with matching keys).
        variant: One of ``"yolo_nas_s"``, ``"yolo_nas_m"``, ``"yolo_nas_l"``.
        strict: Whether to require exact key matching (default True).
        prefer_ema: If True, load EMA weights when available (often better quality).

    Returns:
        The model with loaded weights.
    """
    path = _download(variant)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    # super-gradients checkpoints may wrap the state_dict
    if prefer_ema and "ema_net" in checkpoint:
        raw_sd = checkpoint["ema_net"]
    elif "net" in checkpoint:
        raw_sd = checkpoint["net"]
    elif "state_dict" in checkpoint:
        raw_sd = checkpoint["state_dict"]
    elif "ema_net" in checkpoint:
        raw_sd = checkpoint["ema_net"]
    else:
        raw_sd = checkpoint

    sd = remap_state_dict(raw_sd)

    # Filter out keys that don't belong to the model (optimizer state, etc.)
    model_keys = set(model.state_dict().keys())
    sd = {k: v for k, v in sd.items() if k in model_keys}

    model.load_state_dict(sd, strict=strict)
    return model
