"""Input validation for detection parameters."""

from __future__ import annotations

import torch


def validate_confidence(value: float, name: str = "Confidence threshold") -> float:
    """Validate that a confidence threshold is between 0 and 1."""
    if not 0 <= value <= 1:
        hint = ""
        if 1 < value <= 100:
            hint = f" Did you mean {value / 100}?"
        raise ValueError(f"{name} must be between 0 and 1, got {value}.{hint}")
    return value


def validate_iou_threshold(value: float) -> float:
    """Validate that an IoU threshold is between 0 and 1."""
    if not 0 <= value <= 1:
        hint = ""
        if 1 < value <= 100:
            hint = f" Did you mean {value / 100}?"
        raise ValueError(f"IoU threshold must be between 0 and 1, got {value}.{hint}")
    return value


def validate_device(device: str | torch.device) -> torch.device:
    """Validate and return a torch device."""
    try:
        dev = torch.device(device)
    except RuntimeError as e:
        raise ValueError(
            f"Invalid device: {device!r}. Use 'cpu', 'cuda', or 'cuda:N'."
        ) from e

    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"Device {device!r} requested but CUDA is not available. Use 'cpu' instead."
        )
    return dev


def validate_input_size(size: int) -> int:
    """Validate model input size."""
    if size < 32:
        raise ValueError(f"Input size must be at least 32, got {size}.")
    if size % 32 != 0:
        raise ValueError(f"Input size must be a multiple of 32, got {size}.")
    return size


def validate_model_name(name: str) -> str:
    """Validate model variant name."""
    valid = ("yolo_nas_s", "yolo_nas_m", "yolo_nas_l")
    if name not in valid:
        raise ValueError(f"Unknown model: {name!r}. Choose from {list(valid)}.")
    return name
