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

from modern_yolonas.configs import CONFIGS
from modern_yolonas.head.dfl import NDFLHeads
from modern_yolonas.model import YoloNAS

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


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = True,
    map_location: str = "cpu",
) -> nn.Module:
    """Load a custom-trained ``.pt`` checkpoint into *model*.

    Handles both plain state-dicts and the richer checkpoint dicts produced by
    :class:`~modern_yolonas.training.trainer.Trainer` (which store the state
    under the ``"model_state_dict"`` key).

    Args:
        model: A ``YoloNAS`` instance whose architecture matches the checkpoint.
        checkpoint_path: Path to the ``.pt`` file.
        strict: Require exact key matching (default ``True``).
        map_location: Device string passed to :func:`torch.load`.

    Returns:
        The model with loaded weights.
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd, strict=strict)
    return model


def transfer_to(
    variant: str,
    num_classes: int,
    repo_id: str = HF_REPO_ID,
    revision: str | None = None,
) -> YoloNAS:
    """Build a model for transfer learning by loading pretrained weights then
    replacing the detection heads with freshly initialised ones.

    The strategy is:

    1. Build a full 80-class model and load the pretrained checkpoint **strictly**
       — backbone and neck weights are guaranteed to be fully and correctly loaded.
    2. Swap ``model.heads`` for a new :class:`~modern_yolonas.head.dfl.NDFLHeads`
       instance initialised for *num_classes*.  The new heads start from their
       default random initialisation (biases set via ``prior_prob``).

    This is the correct way to do transfer learning onto a different class set:
    no weight filtering, no shape mismatch surprises.

    Args:
        variant: One of ``"yolo_nas_s"``, ``"yolo_nas_m"``, ``"yolo_nas_l"``.
        num_classes: Number of classes in your dataset.
        repo_id: HF Hub repo (overridable via the ``YOLONAS_HF_REPO`` env var).
        revision: Git ref of the repo.

    Returns:
        A :class:`~modern_yolonas.model.YoloNAS` instance with pretrained
        backbone+neck and freshly initialised detection heads.
    """
    cfg = CONFIGS[variant]

    # Step 1 — load pretrained 80-class model with full strict matching
    model = YoloNAS.from_config(variant, num_classes=80)
    load_pretrained(model, variant, strict=True, repo_id=repo_id, revision=revision)

    # Step 2 — replace heads; backbone and neck are untouched
    model.heads = NDFLHeads(
        num_classes=num_classes,
        in_channels=tuple(model.neck.out_channels),
        **cfg["heads"],
    )
    logger.info(
        "Transferred pretrained %s backbone+neck; heads re-initialised for %d classes.",
        variant,
        num_classes,
    )
    return model
