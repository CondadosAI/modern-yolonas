"""Tests for data pipeline: transforms, collation, dataset config."""

import numpy as np
import pytest

from modern_yolonas.data.transforms import (
    HSVAugment,
    HorizontalFlip,
    LetterboxResize,
    Normalize,
    Compose,
)
from modern_yolonas.data.collate import detection_collate_fn
from modern_yolonas.data.dataset_config import DatasetConfig, load_dataset_config


@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_targets():
    return np.array([[0, 0.5, 0.5, 0.1, 0.1], [1, 0.3, 0.7, 0.2, 0.15]], dtype=np.float32)


class TestTransforms:
    def test_hsv_augment(self, sample_image, sample_targets):
        t = HSVAugment()
        img, tgt = t(sample_image, sample_targets)
        assert img.shape == sample_image.shape
        assert np.array_equal(tgt, sample_targets)

    def test_horizontal_flip(self, sample_image, sample_targets):
        t = HorizontalFlip(p=1.0)
        img, tgt = t(sample_image, sample_targets)
        assert img.shape == sample_image.shape
        assert tgt[0, 1] == pytest.approx(0.5, abs=1e-6)  # center stays centered
        assert tgt[1, 1] == pytest.approx(0.7, abs=1e-6)

    def test_letterbox_resize(self, sample_image, sample_targets):
        t = LetterboxResize(target_size=640)
        img, tgt = t(sample_image, sample_targets)
        assert img.shape == (640, 640, 3)

    def test_normalize(self, sample_image, sample_targets):
        t = Normalize()
        img, tgt = t(sample_image, sample_targets)
        assert img.shape == (3, 480, 640)
        assert img.dtype == np.float32
        assert img.max() <= 1.0
        assert img.min() >= 0.0

    def test_compose(self, sample_image, sample_targets):
        t = Compose([
            LetterboxResize(target_size=640),
            Normalize(),
        ])
        img, tgt = t(sample_image, sample_targets)
        assert img.shape == (3, 640, 640)
        assert img.dtype == np.float32


class TestCollate:
    def test_detection_collate(self):
        # 2 samples, CHW float format
        batch = [
            (np.random.rand(3, 640, 640).astype(np.float32), np.array([[0, 0.5, 0.5, 0.1, 0.1]])),
            (np.random.rand(3, 640, 640).astype(np.float32), np.array([[1, 0.3, 0.7, 0.2, 0.15], [2, 0.1, 0.2, 0.05, 0.05]])),
        ]
        images, targets = detection_collate_fn(batch)
        assert images.shape == (2, 3, 640, 640)
        assert targets.shape == (3, 6)  # 1 + 2 targets
        assert targets[0, 0] == 0  # batch index
        assert targets[1, 0] == 1
        assert targets[2, 0] == 1

    def test_empty_targets(self):
        batch = [
            (np.random.rand(3, 640, 640).astype(np.float32), np.zeros((0, 5))),
        ]
        images, targets = detection_collate_fn(batch)
        assert images.shape == (1, 3, 640, 640)
        assert targets.shape == (0, 6)


class TestDatasetConfig:
    def test_load_dataset_config(self, tmp_path):
        yaml_content = (
            "nc: 3\n"
            'names: ["person", "head", "hardhat"]\n'
            "train: images/train\n"
            "val: images/val\n"
        )
        yaml_file = tmp_path / "data.yaml"
        yaml_file.write_text(yaml_content)

        cfg = load_dataset_config(yaml_file)
        assert isinstance(cfg, DatasetConfig)
        assert cfg.root == tmp_path
        assert cfg.num_classes == 3
        assert cfg.class_names == ["person", "head", "hardhat"]
        assert cfg.train_split == "train"
        assert cfg.val_split == "val"

    def test_load_dataset_config_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_dataset_config("/nonexistent/data.yaml")

    def test_load_dataset_config_defaults(self, tmp_path):
        """train/val keys are optional and default to images/train, images/val."""
        yaml_content = "nc: 1\nnames: ['cat']\n"
        yaml_file = tmp_path / "data.yaml"
        yaml_file.write_text(yaml_content)

        cfg = load_dataset_config(yaml_file)
        assert cfg.num_classes == 1
        assert cfg.class_names == ["cat"]
        assert cfg.train_split == "train"
        assert cfg.val_split == "val"
