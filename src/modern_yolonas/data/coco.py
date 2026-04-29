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
        cache_annotations: bool = True,
        ignore_empty_annotations: bool = True,
    ):
        from pycocotools.coco import COCO

        self.root = Path(root)
        self.transforms = transforms
        self.input_size = input_size

        self.coco = COCO(str(ann_file))
        self.ids = list(sorted(self.coco.imgs.keys()))

        # Build contiguous class mapping using only categories present in annotations.
        # Excludes unused categories (e.g. a "background" id=0 that carries no annotations)
        # so that the resulting label indices are always 0-indexed and contiguous.
        ann_cat_ids = {ann["category_id"] for ann in self.coco.dataset.get("annotations", [])}
        cat_ids = sorted(c for c in self.coco.getCatIds() if c in ann_cat_ids)
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(cat_ids)}

        # Ordered list of class names aligned with label indices 0..N-1
        cats = self.coco.loadCats(cat_ids)
        self.class_names: list[str] = [c["name"] for c in cats]

        # Pre-load all annotation arrays into memory to avoid repeated pycocotools lookups
        self._label_cache: list[np.ndarray] | None = None
        if cache_annotations:
            self._label_cache = [self._parse_anns(img_id) for img_id in self.ids]

        # Drop images with no annotations (background-only samples waste training batches)
        if ignore_empty_annotations:
            labels = self._label_cache if self._label_cache is not None else [self._parse_anns(img_id) for img_id in self.ids]
            keep = [i for i, t in enumerate(labels) if len(t) > 0]
            self.ids = [self.ids[i] for i in keep]
            if self._label_cache is not None:
                self._label_cache = [self._label_cache[i] for i in keep]

    def __len__(self) -> int:
        return len(self.ids)

    def _parse_anns(self, img_id: int) -> np.ndarray:
        """Return ``[N, 5]`` float32 targets for *img_id* (no image needed)."""
        img_info = self.coco.loadImgs(img_id)[0]
        h_img, w_img = img_info["height"], img_info["width"]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        targets = []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, bw, bh = ann["bbox"]
            cls = self.cat_id_to_label[ann["category_id"]]
            xc = (x + bw / 2) / w_img
            yc = (y + bh / 2) / h_img
            nw = bw / w_img
            nh = bh / h_img
            targets.append([cls, xc, yc, nw, nh])
        return np.array(targets, dtype=np.float32).reshape(-1, 5) if targets else np.zeros((0, 5), dtype=np.float32)

    def load_raw(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Load image and targets without transforms."""
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        image = cv2.imread(str(self.root / img_info["file_name"]))

        if self._label_cache is not None:
            targets = self._label_cache[index]
        else:
            targets = self._parse_anns(img_id)

        return image, targets

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image, targets = self.load_raw(index)
        if self.transforms is not None:
            image, targets = self.transforms(image, targets)
        return image, targets
