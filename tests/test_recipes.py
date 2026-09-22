"""Named recipes, and what each one actually builds.

`yolonas train` used to hardcode a pipeline that differed from the repository's
own COCO_RECIPE in every way that matters for COCO -- optimiser, epochs, geometry
and, above all, mosaic. Nothing compared the two, so they drifted. These tests
compare them.
"""

from __future__ import annotations

import pytest

from modern_yolonas.data.transforms import (
    Compose,
    HorizontalFlip,
    LetterboxResize,
    Mosaic,
    Normalize,
    RandomAffine,
    RandomResizedCropFlipAffine,
    TrainTransformPipeline,
)
from modern_yolonas.training.recipes import COCO_RECIPE, LEGACY_RECIPE, RECIPES
from modern_yolonas.training.run import build_transforms


class FakeDataset:
    """Mosaic and Mixup only need a length and a loader to hold a reference to."""

    def __len__(self) -> int:
        return 16

    def load_raw(self, index):
        raise NotImplementedError


def built(name: str, train: bool = True):
    return build_transforms({**RECIPES[name], "input_size": 640},
                            train=train, dataset=FakeDataset() if train else None)


def step_types(pipeline) -> list[str]:
    steps = (pipeline.per_image_transforms.transforms
             if isinstance(pipeline, TrainTransformPipeline) else pipeline.transforms)
    return [type(s).__name__ for s in steps]


class TestRecipeRegistry:
    def test_the_three_recipes_are_registered(self):
        assert set(RECIPES) == {"legacy", "coco", "rf100vl"}

    @pytest.mark.parametrize("name", sorted(RECIPES))
    def test_every_recipe_carries_what_the_trainer_reads(self, name):
        recipe = RECIPES[name]
        for key in ("epochs", "optimizer", "lr", "weight_decay", "batch_size", "augmentations"):
            assert key in recipe, f"{name} is missing {key!r}"


class TestCocoRecipe:
    def test_it_builds_mosaic(self):
        """The whole point of the change: the COCO path trains with mosaic."""
        pipeline = built("coco")
        assert isinstance(pipeline, TrainTransformPipeline)
        assert isinstance(pipeline.mosaic, Mosaic)

    def test_it_letterboxes_rather_than_cropping(self):
        """Training and validation must agree on geometry, or mAP measures a
        different object-size prior than the one trained for."""
        pipeline = built("coco")
        assert any(isinstance(s, LetterboxResize) for s in pipeline.final_transforms.transforms)
        assert "RandomResizedCropFlipAffine" not in step_types(pipeline)

    def test_validation_letterboxes_too(self):
        assert any(isinstance(s, LetterboxResize) for s in built("coco", train=False).transforms)

    def test_mixups_second_image_is_augmented(self):
        """Otherwise half the samples blend an augmented image with a pristine one."""
        pipeline = built("coco")
        assert pipeline.mixup is not None
        assert pipeline.mixup.inner_transforms is pipeline.per_image_transforms

    def test_it_closes_mosaic_before_the_end(self):
        assert COCO_RECIPE["augmentations"]["close_mosaic_epochs"] > 0


class TestLegacyRecipe:
    def test_it_has_no_mosaic(self):
        """Recorded, not endorsed: this is what `yolonas train` did before."""
        assert built("legacy").mosaic is None

    def test_it_uses_one_fused_warp_and_no_letterbox(self):
        pipeline = built("legacy")
        assert "RandomResizedCropFlipAffine" in step_types(pipeline)
        assert not any(isinstance(s, LetterboxResize)
                       for s in pipeline.final_transforms.transforms)

    def test_the_flip_lives_inside_the_fused_transform(self):
        """A separate HorizontalFlip would be a second warp of the whole image."""
        pipeline = built("legacy")
        assert "HorizontalFlip" not in step_types(pipeline)
        fused = next(s for s in pipeline.per_image_transforms.transforms
                     if isinstance(s, RandomResizedCropFlipAffine))
        assert fused.flip_prob == LEGACY_RECIPE["augmentations"]["flip_prob"]

    def test_it_still_normalizes_to_uint8(self):
        assert any(isinstance(s, Normalize)
                   for s in built("legacy").final_transforms.transforms)

    def test_it_keeps_the_optimiser_it_always_had(self):
        assert LEGACY_RECIPE["optimizer"] == "adamw"
        assert LEGACY_RECIPE["epochs"] == 300


class TestRf100vlRecipe:
    def test_it_is_a_plain_compose_with_no_dataset_aware_steps(self):
        pipeline = built("rf100vl")
        assert isinstance(pipeline, Compose)
        assert "RandomAffine" in step_types(pipeline)


class TestTheRecipesActuallyDiffer:
    def test_coco_and_legacy_disagree_on_everything_that_matters(self):
        """If these ever converge, the bug this change fixed has come back."""
        differences = {
            key for key in ("optimizer", "epochs", "lr")
            if COCO_RECIPE[key] != LEGACY_RECIPE[key]
        }
        assert differences == {"optimizer", "epochs", "lr"}
        assert COCO_RECIPE["augmentations"]["mosaic"]
        assert not LEGACY_RECIPE["augmentations"]["mosaic"]

    def test_only_legacy_uses_the_fused_geometry(self):
        assert LEGACY_RECIPE["augmentations"].get("fused_geometry")
        assert not COCO_RECIPE["augmentations"].get("fused_geometry")


def test_a_recipe_without_flip_produces_no_flip_anywhere():
    recipe = {**RECIPES["coco"], "input_size": 640}
    recipe["augmentations"] = {**recipe["augmentations"], "flip": False}
    pipeline = build_transforms(recipe, train=True, dataset=FakeDataset())
    assert not any(isinstance(s, HorizontalFlip) for s in pipeline.per_image_transforms.transforms)


def test_affine_parameters_reach_the_transform():
    recipe = {**RECIPES["rf100vl"], "input_size": 640}
    recipe["augmentations"] = {**recipe["augmentations"], "affine_translate": 0.33}
    pipeline = build_transforms(recipe, train=True, dataset=None)
    assert any(isinstance(s, RandomAffine) for s in pipeline.transforms)
