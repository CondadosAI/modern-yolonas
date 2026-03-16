"""Base classes for detection datasets and transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
from torch.utils.data import Dataset


class BaseDetectionDataset(Dataset, ABC):
    """Abstract base class for detection datasets.

    Subclasses must implement ``load_raw()`` and ``__len__()``.
    ``__getitem__()`` applies transforms to raw data automatically.
    """

    def __init__(self, transforms=None, input_size: int = 640):
        self.transforms = transforms
        self.input_size = input_size

    @abstractmethod
    def load_raw(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Load raw image and targets without transforms.

        Args:
            index: Sample index.

        Returns:
            Tuple of ``(image, targets)`` where image is HWC uint8 BGR
            and targets is ``[N, 5]`` with ``[class_id, x, y, w, h]`` normalized.
        """
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image, targets = self.load_raw(index)
        if self.transforms is not None:
            image, targets = self.transforms(image, targets)
        return image, targets


class DetectionTransform(Protocol):
    """Protocol for detection transforms.

    Any callable matching this signature can be used as a transform.
    """

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
