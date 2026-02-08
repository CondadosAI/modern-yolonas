"""FiftyOne dataset wrapper for YOLO-NAS training.

Wraps a ``fiftyone.Dataset`` or ``fiftyone.DatasetView`` so it can be used
directly with the same training pipeline as :class:`YOLODetectionDataset`.

FiftyOne is an **optional** dependency — this module is importable without it
installed; the lazy import happens inside ``__init__`` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import fiftyone as fo


class FiftyOneDetectionDataset(Dataset):
    """Wraps a FiftyOne dataset/view for object detection training.

    Parameters
    ----------
    fo_dataset:
        A ``fiftyone.Dataset`` or ``fiftyone.DatasetView``.
    label_field:
        Name of the ``Detections`` field on each sample.
    class_names:
        Ordered list of class names.  If *None*, auto-discovered from the
        dataset's ``default_classes`` or by collecting ``distinct()`` labels.
    transforms:
        Optional ``(image, targets) -> (image, targets)`` callable.
    input_size:
        Target input size (not applied here, stored for downstream use).
    """

    def __init__(
        self,
        fo_dataset: fo.Dataset | fo.DatasetView,
        label_field: str = "ground_truth",
        class_names: list[str] | None = None,
        transforms=None,
        input_size: int = 640,
    ):
        import fiftyone  # noqa: F401 — lazy import, fail fast if missing

        self.fo_dataset = fo_dataset
        self.label_field = label_field
        self.transforms = transforms
        self.input_size = input_size

        # Materialize sample IDs for O(1) indexed access
        self.sample_ids: list[str] = [s.id for s in fo_dataset]

        # Resolve class names
        if class_names is not None:
            self.class_names = list(class_names)
        elif hasattr(fo_dataset, "default_classes") and fo_dataset.default_classes:
            self.class_names = list(fo_dataset.default_classes)
        else:
            self.class_names = sorted(
                fo_dataset.distinct(f"{label_field}.detections.label")
            )

        self._name_to_id = {name: i for i, name in enumerate(self.class_names)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def load_raw(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Load image and targets **without** transforms.

        Returns
        -------
        image : np.ndarray
            BGR image as ``(H, W, 3)`` uint8.
        targets : np.ndarray
            ``(N, 5)`` float32 array with rows ``[class_id, cx, cy, w, h]``
            (normalized 0-1).  Empty detections → ``(0, 5)``.
        """
        sample = self.fo_dataset[self.sample_ids[index]]
        image = cv2.imread(sample.filepath)

        detections = sample[self.label_field]
        if detections is None or len(detections.detections) == 0:
            return image, np.zeros((0, 5), dtype=np.float32)

        rows: list[np.ndarray] = []
        for det in detections.detections:
            cls_id = self._name_to_id.get(det.label)
            if cls_id is None:
                continue  # skip unknown labels
            # FiftyOne bbox: [x, y, w, h] top-left, normalized
            x, y, w, h = det.bounding_box
            cx = x + w / 2.0
            cy = y + h / 2.0
            rows.append(np.array([cls_id, cx, cy, w, h], dtype=np.float32))

        if len(rows) == 0:
            return image, np.zeros((0, 5), dtype=np.float32)

        return image, np.stack(rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.transforms is not None and hasattr(self.transforms, 'apply'):
            return self.transforms.apply(index, self.load_raw)
        image, targets = self.load_raw(index)
        if self.transforms is not None:
            image, targets = self.transforms(image, targets)
        return image, targets
