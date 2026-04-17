"""Download and load pretrained YOLO-NAS checkpoints from Hugging Face Hub.

The distributed safetensors were converted from Deci AI's super-gradients
releases and remain under the Super Gradients Model EULA (non-commercial use
only). For commercial deployments, train from scratch — no open-licensed COCO
pretrain is currently provided.
"""

from __future__ import annotations

import logging
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

logger = logging.getLogger(__name__)

# Override with YOLONAS_HF_REPO env var to pull from a fork / private mirror.
HF_REPO_ID = os.environ.get("YOLONAS_HF_REPO", "CondadosAI/detectors")

WEIGHT_FILES = {
    "yolo_nas_s": "yolo-nas-s.safetensors",
    "yolo_nas_m": "yolo-nas-m.safetensors",
    "yolo_nas_l": "yolo-nas-l.safetensors",
}


_LICENSE_WARNING = (
    "Pretrained YOLO-NAS COCO weights are derived from Deci AI's super-gradients "
    "releases and remain under the Super Gradients Model EULA (non-commercial use only). "
    "For commercial deployments, train from scratch — no open-licensed COCO pretrain "
    "is currently distributed. "
    "See https://docs.deci.ai/super-gradients/latest/LICENSE.YOLONAS.html."
)


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
    return {_strip_prefix(k): v for k, v in raw_sd.items()}


def _download(variant: str, repo_id: str, revision: str | None) -> str:
    if variant not in WEIGHT_FILES:
        raise ValueError(f"Unknown variant: {variant!r}. Must be one of {list(WEIGHT_FILES)}")
    logger.warning(_LICENSE_WARNING)
    return hf_hub_download(repo_id=repo_id, filename=WEIGHT_FILES[variant], revision=revision)


def load_pretrained(
    model: nn.Module,
    variant: str,
    strict: bool = True,
    repo_id: str = HF_REPO_ID,
    revision: str | None = None,
) -> nn.Module:
    """Download safetensors checkpoint from HF Hub and load into model.

    Args:
        model: A ``YoloNAS`` instance (or any nn.Module with matching keys).
        variant: One of ``"yolo_nas_s"``, ``"yolo_nas_m"``, ``"yolo_nas_l"``.
        strict: Require exact key matching.
        repo_id: HF Hub repo (default :data:`HF_REPO_ID`, overridable via the
            ``YOLONAS_HF_REPO`` env var or this arg).
        revision: Git ref (branch / tag / commit) of the repo.

    Returns:
        The model with loaded weights.
    """
    path = _download(variant, repo_id=repo_id, revision=revision)
    raw_sd = load_file(path)
    sd = remap_state_dict(raw_sd)

    model_keys = set(model.state_dict().keys())
    sd = {k: v for k, v in sd.items() if k in model_keys}

    model.load_state_dict(sd, strict=strict)
    return model
