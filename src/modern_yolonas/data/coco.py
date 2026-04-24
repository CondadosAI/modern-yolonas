"""COCO format dataset using pycocotools."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset

Transform = Callable[[np.ndarray, dict[str, Any]], tuple[np.ndarray, dict[str, Any]]]


class COCODetectionDataset(Dataset):
    """COCO-format detection dataset.

    Args:
        root: Path to image directory (e.g., ``coco/images/train2017``).
        ann_file: Path to annotation JSON (e.g., ``coco/annotations/instances_train2017.json``).
        transforms: ``(image, targets) → (image, targets)`` callable.
        input_size: Target input size (used by transforms).
    """

    def __init__(
        self,
        root: str | Path,
        ann_file: str | Path,
        transforms: Transform | None = None,
        input_size: int = 640,
    ):
        from pycocotools.coco import COCO

        self.root = Path(root)
        self.transforms = transforms
        self.input_size = input_size

        self.coco = COCO(str(ann_file))
        self.ids = list(sorted(self.coco.imgs.keys()))

        # Build contiguous class mapping (COCO IDs are not 0-79)
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(cat_ids)}

        # Ordered list of class names aligned with label indices 0..N-1
        cats = self.coco.loadCats(cat_ids)
        self.class_names: list[str] = [c["name"] for c in cats]

    def __len__(self) -> int:
        return len(self.ids)

    def load_raw(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Load image and targets without transforms."""
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        image = cv2.imread(str(self.root / img_info["file_name"]))

        h, w = image.shape[:2]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        targets = []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, bw, bh = ann["bbox"]  # COCO: x,y,w,h (pixel, top-left)
            cls = self.cat_id_to_label[ann["category_id"]]
            # Convert to normalized x_center, y_center, w, h
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            targets.append([cls, xc, yc, nw, nh])

        targets = np.array(targets, dtype=np.float32).reshape(-1, 5) if targets else np.zeros((0, 5), dtype=np.float32)
        return image, targets

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image, targets = self.load_raw(index)
        if self.transforms is not None:
            image, targets = self.transforms(image, targets)
        return image, targets
