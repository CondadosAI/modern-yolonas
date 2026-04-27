"""Tests for dataset classes (base, COCO, YOLO) and remaining transforms."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from modern_yolonas.data.base import BaseDetectionDataset, DetectionTransform
from modern_yolonas.data.coco import COCODetectionDataset
from modern_yolonas.data.transforms import Mixup, Mosaic, RandomAffine
from modern_yolonas.data.yolo import YOLODetectionDataset


def _write_image(path: Path, size: int = 100, fill: int = 128) -> None:
    img = np.full((size, size, 3), fill, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_coco_annotations(path: Path, image_name: str = "a.jpg") -> None:
    ann = {
        "images": [{"id": 1, "file_name": image_name, "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 10, "image_id": 1, "category_id": 5,
                "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0,
            },
            {
                # Crowd annotation — should be filtered out by the dataset
                "id": 11, "image_id": 1, "category_id": 5,
                "bbox": [50, 50, 10, 10], "area": 100, "iscrowd": 1,
            },
            {
                # Second category used so cat_id_to_label includes both {5:0, 7:1}
                "id": 12, "image_id": 1, "category_id": 7,
                "bbox": [30, 30, 15, 15], "area": 225, "iscrowd": 0,
            },
        ],
        "categories": [{"id": 5, "name": "cat_a"}, {"id": 7, "name": "cat_b"}],
    }
    with path.open("w") as f:
        json.dump(ann, f)


class _MinimalDataset(BaseDetectionDataset):
    """Smallest possible concrete subclass for testing the ABC."""

    def __init__(self, transforms=None, input_size=64):
        super().__init__(transforms, input_size)
        self._samples = [
            (np.full((64, 64, 3), 200, dtype=np.uint8), np.zeros((0, 5), dtype=np.float32)),
            (np.full((64, 64, 3), 100, dtype=np.uint8),
             np.array([[0, 0.5, 0.5, 0.2, 0.2]], dtype=np.float32)),
        ]

    def load_raw(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)


class TestBaseDetectionDataset:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseDetectionDataset()  # type: ignore[abstract]

    def test_len_and_getitem_without_transforms(self):
        ds = _MinimalDataset()
        assert len(ds) == 2
        img, targets = ds[0]
        assert img.shape == (64, 64, 3)
        assert targets.shape == (0, 5)

    def test_getitem_applies_transforms(self):
        calls = []

        def t(image, targets):
            calls.append((image.shape, len(targets)))
            return image + 1, targets
        ds = _MinimalDataset(transforms=t)
        img, _ = ds[1]
        assert calls == [((64, 64, 3), 1)]
        # Transform bumped pixel values by 1
        assert img[0, 0, 0] == 101

    def test_detection_transform_protocol(self):
        """Any callable with the right signature satisfies the protocol."""

        def _t(image, targets):
            return image, targets

        tf: DetectionTransform = _t
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        targets = np.zeros((0, 5), dtype=np.float32)
        out_img, out_tgt = tf(img, targets)
        assert out_img.shape == img.shape


class TestCOCODetectionDataset:
    @pytest.fixture
    def tiny_coco(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _write_image(img_dir / "a.jpg", size=100)
        ann = tmp_path / "ann.json"
        _write_coco_annotations(ann)
        return img_dir, ann

    def test_len_matches_image_count(self, tiny_coco):
        img_dir, ann = tiny_coco
        ds = COCODetectionDataset(root=img_dir, ann_file=ann)
        assert len(ds) == 1

    def test_cat_id_remapping(self, tiny_coco):
        img_dir, ann = tiny_coco
        ds = COCODetectionDataset(root=img_dir, ann_file=ann)
        # Categories sorted: [5, 7] → labels {5:0, 7:1}
        assert ds.cat_id_to_label == {5: 0, 7: 1}

    def test_load_raw_filters_crowd(self, tiny_coco):
        img_dir, ann = tiny_coco
        ds = COCODetectionDataset(root=img_dir, ann_file=ann)
        image, targets = ds.load_raw(0)
        assert image.shape == (100, 100, 3)
        # 3 annotations total (ids 10, 11, 12); 1 is crowd (id 11) → 2 kept
        assert targets.shape == (2, 5)
        # First kept annotation: cat_id 5 → label 0, bbox [10,10,20,20]
        assert targets[0, 0] == 0
        # Normalized center of bbox [10, 10, 20, 20] → ((10+10)/100, (10+10)/100)
        assert targets[0, 1] == pytest.approx(0.2)
        assert targets[0, 2] == pytest.approx(0.2)
        assert targets[0, 3] == pytest.approx(0.2)  # w/100
        assert targets[0, 4] == pytest.approx(0.2)  # h/100

    def test_getitem_runs_transforms(self, tiny_coco):
        img_dir, ann = tiny_coco

        def _double_width(image, targets):
            return np.concatenate([image, image], axis=1), targets

        ds = COCODetectionDataset(root=img_dir, ann_file=ann, transforms=_double_width)
        image, _ = ds[0]
        assert image.shape == (100, 200, 3)

    def test_empty_annotations(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _write_image(img_dir / "a.jpg")
        ann = tmp_path / "ann.json"
        with ann.open("w") as f:
            json.dump({
                "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
                "annotations": [],
                "categories": [{"id": 5, "name": "cat_a"}],
            }, f)
        ds = COCODetectionDataset(root=img_dir, ann_file=ann, ignore_empty_annotations=False)
        _, targets = ds.load_raw(0)
        assert targets.shape == (0, 5)


class TestYOLODetectionDataset:
    @pytest.fixture
    def yolo_root(self, tmp_path):
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "labels" / "train").mkdir(parents=True)
        _write_image(tmp_path / "images" / "train" / "a.jpg")
        _write_image(tmp_path / "images" / "train" / "b.png")
        # Ignored non-image file
        (tmp_path / "images" / "train" / "readme.txt").write_text("ignored")
        # Labels: a has two rows, b has no label file
        (tmp_path / "labels" / "train" / "a.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n"
        )
        return tmp_path

    def test_discovers_images_only(self, yolo_root):
        ds = YOLODetectionDataset(root=yolo_root, split="train", ignore_empty_annotations=False)
        assert len(ds) == 2
        suffixes = {p.suffix for p in ds.images}
        assert ".txt" not in suffixes

    def test_load_raw_with_labels(self, yolo_root):
        ds = YOLODetectionDataset(root=yolo_root, split="train", ignore_empty_annotations=False)
        # First image is sorted by filename; find the one with labels
        image, targets = ds.load_raw(0)  # a.jpg
        assert image.shape == (100, 100, 3)
        assert targets.shape == (2, 5)
        assert targets.dtype == np.float32

    def test_load_raw_missing_label_file(self, yolo_root):
        ds = YOLODetectionDataset(root=yolo_root, split="train", ignore_empty_annotations=False)
        # b.png has no label file → empty targets
        image, targets = ds.load_raw(1)
        assert image.shape == (100, 100, 3)
        assert targets.shape == (0, 5)

    def test_getitem_applies_transforms(self, yolo_root):
        def _noop(image, targets):
            return image, targets + 0  # forces evaluation
        ds = YOLODetectionDataset(root=yolo_root, split="train", transforms=_noop, ignore_empty_annotations=False)
        image, targets = ds[0]
        assert image.shape == (100, 100, 3)
        assert targets.shape[1] == 5

    def test_ignore_empty_annotations_filters_unannotated(self, yolo_root):
        # Default: ignore_empty_annotations=True → only a.jpg (has labels) remains
        ds = YOLODetectionDataset(root=yolo_root, split="train")
        assert len(ds) == 1
        _, targets = ds.load_raw(0)
        assert len(targets) > 0


class TestRandomAffine:
    def test_shape_preserved_no_aug(self):
        # With all params at defaults of 0, only the random scale from (0.5, 1.5) fires
        tf = RandomAffine(degrees=0.0, translate=0.0, scale=(1.0, 1.0), shear=0.0)
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        targets = np.array([[0, 0.5, 0.5, 0.4, 0.4]], dtype=np.float32)
        out_img, out_targets = tf(img, targets)
        assert out_img.shape == img.shape
        # Identity scale + no rotation → targets roughly unchanged
        assert out_targets.shape == (1, 5)
        assert out_targets[0, 1] == pytest.approx(0.5, abs=0.01)

    def test_empty_targets(self):
        tf = RandomAffine()
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        _, targets = tf(img, np.zeros((0, 5), dtype=np.float32))
        assert targets.shape == (0, 5)

    def test_tiny_boxes_filtered_after_large_scale_down(self):
        tf = RandomAffine(degrees=0.0, translate=0.0, scale=(0.05, 0.05), shear=0.0)
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        # Tiny box that will become < 2px after scale-down
        targets = np.array([[0, 0.5, 0.5, 0.03, 0.03]], dtype=np.float32)
        _, out_targets = tf(img, targets)
        assert len(out_targets) == 0


class _FakeRawDataset:
    """Minimal dataset with `load_raw` and `__getitem__` for Mosaic/Mixup."""

    def __init__(self, n: int = 4):
        self.n = n

    def __len__(self):
        return self.n

    def load_raw(self, idx):
        img = np.full((80, 80, 3), 20 + idx * 40, dtype=np.uint8)
        # Each sample has one target in the center
        targets = np.array([[idx % 2, 0.5, 0.5, 0.3, 0.3]], dtype=np.float32)
        return img, targets

    def __getitem__(self, idx):
        return self.load_raw(idx)


class TestMosaic:
    def test_output_shape(self):
        ds = _FakeRawDataset(n=4)
        mosaic = Mosaic(ds, input_size=64)
        img, targets = mosaic(0)
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8
        # Targets are normalized and inside [0,1)
        if len(targets):
            assert targets[:, 1:].min() >= 0
            assert targets[:, 1:].max() <= 1.0

    def test_empty_dataset_targets(self):
        class _EmptyTargetsDS(_FakeRawDataset):
            def load_raw(self, idx):
                img, _ = super().load_raw(idx)
                return img, np.zeros((0, 5), dtype=np.float32)

        mosaic = Mosaic(_EmptyTargetsDS(n=4), input_size=32)
        img, targets = mosaic(0)
        assert img.shape == (32, 32, 3)
        assert targets.shape == (0, 5)


class TestMixup:
    def test_output_shape_and_dtype(self):
        ds = _FakeRawDataset(n=3)
        mixup = Mixup(ds, p=1.0, alpha=1.0, beta=1.0)
        img1, tgt1 = ds.load_raw(0)
        mixed_img, mixed_tgt = mixup(img1, tgt1)
        assert mixed_img.shape == img1.shape
        assert mixed_img.dtype == np.uint8
        # Mixup concatenates targets
        assert len(mixed_tgt) >= len(tgt1)

    def test_preserves_targets_when_partner_empty(self):
        class _EmptyPartnerDS(_FakeRawDataset):
            def load_raw(self, idx):
                img, _ = super().load_raw(idx)
                return img, np.zeros((0, 5), dtype=np.float32)

        ds = _EmptyPartnerDS(n=3)
        mixup = Mixup(ds, p=1.0, alpha=1.0, beta=1.0)
        # Shape must match the dataset's own images (80×80) for the blend
        img = np.full((80, 80, 3), 128, dtype=np.uint8)
        tgt = np.array([[0, 0.5, 0.5, 0.3, 0.3]], dtype=np.float32)
        _, mixed_tgt = mixup(img, tgt)
        # Partner empty → original targets preserved
        assert len(mixed_tgt) == 1
