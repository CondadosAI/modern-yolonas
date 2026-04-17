"""Letterbox resize and normalization for inference."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def letterbox(
    image: np.ndarray,
    target_size: int = 640,
    rescale_size: int = 636,
    pad_value: int = 114,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image with letterbox padding (matches super-gradients pipeline).

    Rescales so the longest side fits within ``rescale_size``, then center-pads
    to ``target_size``. This leaves a small guaranteed padding border.

    Args:
        image: HWC uint8 BGR image.
        target_size: Final square output size.
        rescale_size: Rescale longest side to this (default 636, matching SG).
        pad_value: Fill value for padding.

    Returns:
        (padded_image, scale, (pad_left, pad_top))
    """
    h, w = image.shape[:2]
    # Clamp so the resized longest side fits the target canvas (overflow crashes the paste below).
    rescale_size = min(rescale_size, target_size)
    scale = rescale_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    import cv2

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    left = pad_w // 2

    padded = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
    padded[top : top + new_h, left : left + new_w] = resized

    return padded, scale, (left, top)


def preprocess(image: np.ndarray, target_size: int = 640) -> tuple[Tensor, float, tuple[int, int]]:
    """Letterbox + normalize + to NCHW tensor.

    Args:
        image: HWC uint8 BGR image.
        target_size: Target square size.

    Returns:
        (tensor [1,3,H,W] float32, scale, (pad_left, pad_top))
    """
    padded, scale, pad = letterbox(image, target_size)
    # HWC → CHW, [0,255] → [0,1] (keep BGR — model was trained on BGR)
    img = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor, scale, pad
