"""Train on several detection datasets as one."""

from __future__ import annotations

import bisect
from collections.abc import Sequence

import numpy as np
from torch.utils.data import Dataset


class ConcatDetectionDataset(Dataset):
    """Several detection datasets behind one index, sharing one transform pipeline.

    ``torch.utils.data.ConcatDataset`` is not enough here: Mosaic and Mixup draw
    their extra images through ``dataset.load_raw``, and ``CloseMosaicCallback``
    switches them off through ``dataset.transforms``. Both have to see the whole
    pool, so the pipeline lives on this wrapper and the children carry none.

    Every child must map category ids to the same label indices, which is checked
    here because a mismatch trains without error on the wrong classes. Build the
    extra datasets with ``cat_id_to_label=`` the first one's.

    Args:
        datasets: Children exposing ``load_raw(index)`` and ``__len__``. Class
            names and the category mapping are taken from the first.
        transforms: Pipeline applied to every sample, usually assigned after
            construction because Mosaic needs a reference to this dataset.
    """

    def __init__(self, datasets: Sequence[Dataset], transforms=None):
        if not datasets:
            raise ValueError("ConcatDetectionDataset needs at least one dataset")
        first = getattr(datasets[0], "cat_id_to_label", None)
        for index, child in enumerate(datasets[1:], start=1):
            mapping = getattr(child, "cat_id_to_label", None)
            if mapping != first:
                raise ValueError(
                    f"dataset {index} maps categories differently from dataset 0; "
                    f"build it with cat_id_to_label=datasets[0].cat_id_to_label"
                )
        self.datasets = list(datasets)
        self.transforms = transforms
        self.cat_id_to_label = first
        self.class_names = getattr(datasets[0], "class_names", None)
        self.input_size = getattr(datasets[0], "input_size", None)
        self.cumulative_sizes = np.cumsum([len(d) for d in self.datasets]).tolist()

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def _locate(self, index: int) -> tuple[Dataset, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} out of range for {len(self)} samples")
        which = bisect.bisect_right(self.cumulative_sizes, index)
        offset = self.cumulative_sizes[which - 1] if which else 0
        return self.datasets[which], index - offset

    def load_raw(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Load image and targets without transforms."""
        child, local = self._locate(index)
        return child.load_raw(local)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.transforms is not None and hasattr(self.transforms, 'apply'):
            return self.transforms.apply(index, self.load_raw)
        image, targets = self.load_raw(index)
        if self.transforms is not None:
            image, targets = self.transforms(image, targets)
        return image, targets
