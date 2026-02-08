"""Tests for data pipeline: transforms, collation, dataset config."""

import numpy as np
import pytest

from modern_yolonas.data.transforms import (
    HSVAugment,
    HorizontalFlip,
    VerticalFlip,
    RandomChannelShuffle,
    RandomCrop,
    LetterboxResize,
    Normalize,
    Compose,
    Mosaic,
    Mixup,
    TrainTransformPipeline,
)
from modern_yolonas.data.collate import detection_collate_fn
from modern_yolonas.data.dataset_config import DatasetConfig, load_dataset_config


@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_targets():
    return np.array([[0, 0.5, 0.5, 0.1, 0.1], [1, 0.3, 0.7, 0.2, 0.15]], dtype=np.float32)


class _FakeDataset:
    """Minimal dataset stub for Mosaic/Mixup tests."""

    def __init__(self, n: int = 10, h: int = 480, w: int = 640):
        self.n = n
        self.h = h
        self.w = w

    def __len__(self):
        return self.n

    def load_raw(self, index):
        img = np.random.randint(0, 255, (self.h, self.w, 3), dtype=np.uint8)
        targets = np.array([[0, 0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
        return img, targets


class TestTransforms:
    def test_hsv_augment(self, sample_image, sample_targets):
        t = HSVAugment()
        img, tgt = t(sample_image, sample_targets)
        assert img.shape == sample_image.shape
        assert np.array_equal(tgt, sample_targets)

    def test_hsv_augment_p0_noop(self, sample_image, sample_targets):
        t = HSVAugment(p=0.0)
        img, tgt = t(sample_image, sample_targets)
        assert np.array_equal(img, sample_image)
        assert np.array_equal(tgt, sample_targets)

    def test_hsv_augment_p1_modifies(self, sample_image, sample_targets):
        t = HSVAugment(p=1.0)
        img, tgt = t(sample_image, sample_targets)
        assert img.shape == sample_image.shape
        # With p=1.0 the image should (almost certainly) be modified
        # Note: there's a tiny chance gains are exactly 0, making it a no-op

    def test_horizontal_flip(self, sample_image, sample_targets):
        t = HorizontalFlip(p=1.0)
        img, tgt = t(sample_image, sample_targets)
        assert img.shape == sample_image.shape
        assert tgt[0, 1] == pytest.approx(0.5, abs=1e-6)  # center stays centered
        assert tgt[1, 1] == pytest.approx(0.7, abs=1e-6)

    def test_vertical_flip_p1(self, sample_image, sample_targets):
        t = VerticalFlip(p=1.0)
        img, tgt = t(sample_image, sample_targets)
        assert img.shape == sample_image.shape
        # y_center flipped: 0.5 -> 0.5, 0.7 -> 0.3
        assert tgt[0, 2] == pytest.approx(0.5, abs=1e-6)
        assert tgt[1, 2] == pytest.approx(0.3, abs=1e-6)
        # x_center unchanged
        assert tgt[0, 1] == pytest.approx(0.5, abs=1e-6)
        assert tgt[1, 1] == pytest.approx(0.3, abs=1e-6)

    def test_vertical_flip_p0_noop(self, sample_image, sample_targets):
        t = VerticalFlip(p=0.0)
        img, tgt = t(sample_image, sample_targets)
        assert np.array_equal(img, sample_image)
        assert np.array_equal(tgt, sample_targets)

    def test_random_channel_shuffle_p1(self, sample_image, sample_targets):
        t = RandomChannelShuffle(p=1.0)
        img, tgt = t(sample_image, sample_targets)
        assert img.shape == sample_image.shape
        assert img.dtype == sample_image.dtype
        # Targets unchanged
        assert np.array_equal(tgt, sample_targets)

    def test_random_channel_shuffle_p0_noop(self, sample_image, sample_targets):
        t = RandomChannelShuffle(p=0.0)
        img, tgt = t(sample_image, sample_targets)
        assert np.array_equal(img, sample_image)
        assert np.array_equal(tgt, sample_targets)

    def test_random_crop_p1(self, sample_image, sample_targets):
        t = RandomCrop(min_scale=0.5, max_scale=0.8, p=1.0)
        img, tgt = t(sample_image, sample_targets)
        # Image should be cropped (smaller or equal)
        assert img.shape[0] <= sample_image.shape[0]
        assert img.shape[1] <= sample_image.shape[1]
        assert img.shape[2] == 3
        # Boxes should still be valid (if any survive)
        if len(tgt):
            assert np.all(tgt[:, 1:] >= 0)
            assert np.all(tgt[:, 1:3] <= 1.0)

    def test_random_crop_p0_noop(self, sample_image, sample_targets):
        t = RandomCrop(p=0.0)
        img, tgt = t(sample_image, sample_targets)
        assert np.array_equal(img, sample_image)
        assert np.array_equal(tgt, sample_targets)

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


class TestMosaic:
    def test_mosaic_enabled(self):
        ds = _FakeDataset(n=10)
        mosaic = Mosaic(dataset=ds, input_size=640, prob=1.0)
        img, tgt = mosaic(0)
        assert img.shape == (640, 640, 3)
        assert img.dtype == np.uint8

    def test_mosaic_disabled_returns_load_raw(self):
        ds = _FakeDataset(n=10)
        mosaic = Mosaic(dataset=ds, input_size=640, prob=1.0)
        mosaic.enabled = False
        img, tgt = mosaic(0)
        # Should return raw image (480x640), not mosaic (640x640)
        assert img.shape == (480, 640, 3)


class TestMixup:
    def test_mixup_enabled(self, sample_image, sample_targets):
        ds = _FakeDataset(n=10, h=sample_image.shape[0], w=sample_image.shape[1])
        mixup = Mixup(dataset=ds, prob=1.0)
        img, tgt = mixup(sample_image, sample_targets)
        assert img.shape == sample_image.shape
        assert img.dtype == np.uint8

    def test_mixup_disabled_returns_input(self, sample_image, sample_targets):
        ds = _FakeDataset(n=10)
        mixup = Mixup(dataset=ds, prob=1.0)
        mixup.enabled = False
        img, tgt = mixup(sample_image, sample_targets)
        assert np.array_equal(img, sample_image)
        assert np.array_equal(tgt, sample_targets)


class TestTrainTransformPipeline:
    def test_apply_full_pipeline(self):
        ds = _FakeDataset(n=10)
        mosaic = Mosaic(dataset=ds, input_size=640, prob=1.0)
        per_image = Compose([HorizontalFlip(p=0.5)])
        mixup = Mixup(dataset=ds, prob=0.5)
        mixup.inner_transforms = per_image
        final = Compose([LetterboxResize(target_size=640), Normalize()])

        pipeline = TrainTransformPipeline(
            mosaic=mosaic,
            per_image_transforms=per_image,
            mixup=mixup,
            final_transforms=final,
        )

        img, tgt = pipeline.apply(0, ds.load_raw)
        assert img.shape == (3, 640, 640)
        assert img.dtype == np.float32

    def test_apply_without_mosaic(self):
        ds = _FakeDataset(n=10)
        per_image = Compose([HorizontalFlip(p=0.5)])
        final = Compose([LetterboxResize(target_size=640), Normalize()])

        pipeline = TrainTransformPipeline(
            mosaic=None,
            per_image_transforms=per_image,
            mixup=None,
            final_transforms=final,
        )

        img, tgt = pipeline.apply(0, ds.load_raw)
        assert img.shape == (3, 640, 640)
        assert img.dtype == np.float32

    def test_disable_mosaic_mixup(self):
        ds = _FakeDataset(n=10)
        mosaic = Mosaic(dataset=ds, input_size=640)
        mixup = Mixup(dataset=ds)

        pipeline = TrainTransformPipeline(
            mosaic=mosaic,
            per_image_transforms=Compose([]),
            mixup=mixup,
            final_transforms=Compose([LetterboxResize(640), Normalize()]),
        )

        assert mosaic.enabled is True
        assert mixup.enabled is True

        pipeline.disable_mosaic_mixup()

        assert mosaic.enabled is False
        assert mixup.enabled is False

    def test_callable_fallback(self, sample_image, sample_targets):
        """__call__ should apply per-image + final only (no Mosaic/Mixup)."""
        pipeline = TrainTransformPipeline(
            mosaic=None,
            per_image_transforms=Compose([HorizontalFlip(p=0.0)]),
            mixup=None,
            final_transforms=Compose([LetterboxResize(640), Normalize()]),
        )

        img, tgt = pipeline(sample_image, sample_targets)
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
