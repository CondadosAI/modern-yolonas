"""Tests for the recipe → transform pipeline builder used by training and benchmarks."""

from __future__ import annotations

import numpy as np
import pytest

from modern_yolonas.data.transforms import (
    Compose,
    HorizontalFlip,
    HSVAugment,
    Mixup,
    Mosaic,
    RandomChannelShuffle,
    RandomCrop,
    TrainTransformPipeline,
    VerticalFlip,
)
from modern_yolonas.training.recipes import COCO_RECIPE, RF100VL_RECIPE
from modern_yolonas.training.run import build_transforms


class _StubDataset:
    """Four identical samples — enough for Mosaic to draw its three extra tiles."""

    def __init__(self, n: int = 4, size: int = 64):
        self.n = n
        self.size = size

    def __len__(self):
        return self.n

    def load_raw(self, index: int):
        rng = np.random.default_rng(index)
        image = rng.integers(0, 255, (self.size, self.size, 3), dtype=np.uint8)
        targets = np.array([[0, 0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        return image, targets


def _types(pipeline_steps) -> set[type]:
    return {type(t) for t in pipeline_steps}


class TestBuildTransformsVal:
    def test_val_is_letterbox_and_normalize_only(self):
        val = build_transforms(COCO_RECIPE, train=False)
        assert isinstance(val, Compose)
        # No augmentation may leak into validation, whatever the recipe enables.
        assert not (_types(val.transforms) & {HSVAugment, HorizontalFlip, Mosaic, Mixup})

    def test_val_ignores_augmentation_block(self):
        recipe = {**COCO_RECIPE, "augmentations": {"mosaic": True, "mixup": True, "hsv": True}}
        val = build_transforms(recipe, train=False, dataset=_StubDataset())
        assert isinstance(val, Compose)
        assert not (_types(val.transforms) & {Mosaic, Mixup})


class TestBuildTransformsTrain:
    def test_without_dataset_no_dataset_aware_augs(self):
        # Mosaic and Mixup need to sample other images; with no dataset to sample
        # from, the builder has to fall back to a plain per-image pipeline.
        train = build_transforms(COCO_RECIPE, train=True)
        assert isinstance(train, Compose)
        assert not (_types(train.transforms) & {Mosaic, Mixup})

    def test_with_dataset_builds_full_pipeline(self):
        train = build_transforms(COCO_RECIPE, train=True, dataset=_StubDataset())
        assert isinstance(train, TrainTransformPipeline)
        assert isinstance(train.mosaic, Mosaic)
        assert isinstance(train.mixup, Mixup)
        # The mixed-in image gets the same per-image augmentations as the primary.
        assert train.mixup.inner_transforms is not None

    def test_augmentation_flags_are_honoured(self):
        recipe = {
            **COCO_RECIPE,
            "augmentations": {
                **COCO_RECIPE["augmentations"],
                "hsv": False,
                "flip": False,
                "channel_shuffle": False,
                "vertical_flip": True,
                "random_crop": True,
            },
        }
        train = build_transforms(recipe, train=True, dataset=_StubDataset())
        present = _types(train.per_image_transforms.transforms)
        assert VerticalFlip in present
        assert RandomCrop in present
        assert HSVAugment not in present
        assert HorizontalFlip not in present
        assert RandomChannelShuffle not in present

    def test_probabilities_come_from_the_recipe(self):
        recipe = {
            **COCO_RECIPE,
            "augmentations": {**COCO_RECIPE["augmentations"], "mosaic_prob": 0.25, "mixup_prob": 0.75},
        }
        train = build_transforms(recipe, train=True, dataset=_StubDataset())
        assert train.mosaic.prob == 0.25
        assert train.mixup.prob == 0.75

    def test_mosaic_only_when_mixup_disabled(self):
        recipe = {**COCO_RECIPE, "augmentations": {**COCO_RECIPE["augmentations"], "mixup": False}}
        train = build_transforms(recipe, train=True, dataset=_StubDataset())
        assert isinstance(train.mosaic, Mosaic)
        assert train.mixup is None

    @pytest.mark.parametrize("recipe", [COCO_RECIPE, RF100VL_RECIPE], ids=["coco", "rf100vl"])
    def test_shipped_recipes_produce_usable_output(self, recipe):
        size = recipe["input_size"]
        ds = _StubDataset(size=size)
        train = build_transforms(recipe, train=True, dataset=ds)
        if isinstance(train, TrainTransformPipeline):
            out_img, out_targets = train.apply(0, ds.load_raw)
        else:
            out_img, out_targets = train(*ds.load_raw(0))
        # Normalize outputs CHW float32, so the spatial dims are the trailing two.
        assert out_img.shape[-2:] == (size, size)
        assert out_targets.ndim == 2


class TestCloseMosaic:
    def test_disable_turns_both_off(self):
        train = build_transforms(COCO_RECIPE, train=True, dataset=_StubDataset())
        train.disable_mosaic_mixup()
        assert train.mosaic.enabled is False
        assert train.mixup.enabled is False

    def test_disable_is_idempotent(self):
        # CloseMosaicCallback fires on every epoch start once the threshold passes.
        train = build_transforms(COCO_RECIPE, train=True, dataset=_StubDataset())
        train.disable_mosaic_mixup()
        train.disable_mosaic_mixup()
        assert train.mosaic.enabled is False

    def test_compose_pipeline_also_responds(self):
        # `yolonas train` builds a plain Compose with Mixup in the list, so the
        # callback has to reach it there too.
        ds = _StubDataset()
        pipeline = Compose([HorizontalFlip(), Mixup(ds, prob=1.0)])
        pipeline.disable_mosaic_mixup()
        assert pipeline.transforms[-1].enabled is False
