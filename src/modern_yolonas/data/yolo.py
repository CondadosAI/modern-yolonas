"""YOLO txt-label format dataset.

Expected layout::

    root/
    ├── images/
    │   ├── train/
    │   │   ├── img001.jpg
    │   │   └── ...
    │   └── val/
    └── labels/
        ├── train/
        │   ├── img001.txt
        │   └── ...
        └── val/

Each label file has one row per object: ``class_id x_center y_center width height`` (normalized).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class YOLODetectionDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transforms=None,
        input_size: int = 640,
        cache_annotations: bool = True,
        ignore_empty_annotations: bool = True,
    ):
        self.root = Path(root)
        self.transforms = transforms
        self.input_size = input_size

        img_dir = self.root / "images" / split
        label_dir = self.root / "labels" / split

        self.images = sorted(img_dir.glob("*.*"))
        self.images = [p for p in self.images if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        self.label_dir = label_dir

        # Try to load class names from classes.txt or data.yaml at dataset root
        self.class_names: list[str] | None = self._load_class_names()

        # Pre-load all annotation arrays into memory (avoids repeated disk I/O during training)
        self._label_cache: list[np.ndarray] | None = None
        if cache_annotations:
            self._label_cache = [self._read_label(p) for p in self.images]

        # Drop images with no annotations (background-only samples waste training batches)
        if ignore_empty_annotations:
            labels = self._label_cache if self._label_cache is not None else [self._read_label(p) for p in self.images]
            keep = [i for i, t in enumerate(labels) if len(t) > 0]
            self.images = [self.images[i] for i in keep]
            if self._label_cache is not None:
                self._label_cache = [self._label_cache[i] for i in keep]

    def _load_class_names(self) -> list[str] | None:
        """Return class names from ``classes.txt`` or ``data.yaml`` if present."""
        classes_txt = self.root / "classes.txt"
        if classes_txt.exists():
            names = [label.strip() for label in classes_txt.read_text().splitlines() if label.strip()]
            return names if names else None

        for yaml_name in ("data.yaml", "dataset.yaml"):
            yaml_path = self.root / yaml_name
            if yaml_path.exists():
                import yaml
                data = yaml.safe_load(yaml_path.read_text())
                names = data.get("names")
                if isinstance(names, list) and names:
                    return [str(n) for n in names]
                if isinstance(names, dict):
                    return [str(names[k]) for k in sorted(names)]

        return None

    def __len__(self) -> int:
        return len(self.images)

    @property
    def num_classes(self) -> int:
        """Number of classes inferred from cached labels (or scanning all label files)."""
        labels = self._label_cache if self._label_cache is not None else [self._read_label(p) for p in self.images]
        all_cls = np.concatenate([t[:, 0] for t in labels if len(t)]) if any(len(t) for t in labels) else np.array([])
        return int(all_cls.max()) + 1 if len(all_cls) else 0

    def _label_path(self, img_path: Path) -> Path:
        return self.label_dir / (img_path.stem + ".txt")

    def _read_label(self, img_path: Path) -> np.ndarray:
        label_path = self._label_path(img_path)
        if label_path.exists():
            data = np.loadtxt(str(label_path), ndmin=2).reshape(-1, 5)
        else:
            data = np.zeros((0, 5))
        return data.astype(np.float32)

    def load_raw(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Load image and targets without transforms."""
        img_path = self.images[index]
        image = cv2.imread(str(img_path))
        targets = self._label_cache[index] if self._label_cache is not None else self._read_label(img_path)
        return image, targets

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image, targets = self.load_raw(index)
        if self.transforms is not None:
            image, targets = self.transforms(image, targets)
        return image, targets
