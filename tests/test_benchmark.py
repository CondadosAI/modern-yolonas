"""Tests for benchmark infrastructure: recipes, mAP validation, discover_datasets, run_training."""

import json
from unittest.mock import MagicMock

import pytest
import torch

from modern_yolonas import yolo_nas_s
from modern_yolonas.training.lightning_module import YoloNASLightningModule
from modern_yolonas.training.recipes import COCO_RECIPE, RF100VL_RECIPE


@pytest.fixture
def small_model():
    return yolo_nas_s(pretrained=False)


class TestRecipes:
    def test_coco_recipe_has_required_keys(self):
        required = [
            "epochs", "optimizer", "lr", "weight_decay",
            "cosine_final_lr_ratio", "warmup_epochs", "ema_decay",
            "precision", "input_size", "batch_size", "workers",
            "conf_threshold", "iou_threshold", "augmentations",
        ]
        for key in required:
            assert key in COCO_RECIPE, f"Missing key: {key}"

    def test_rf100vl_recipe_has_required_keys(self):
        required = [
            "epochs", "optimizer", "lr", "weight_decay",
            "cosine_final_lr_ratio", "warmup_epochs", "ema_decay",
            "precision", "input_size", "batch_size", "workers",
            "conf_threshold", "iou_threshold", "augmentations",
        ]
        for key in required:
            assert key in RF100VL_RECIPE, f"Missing key: {key}"

    def test_coco_recipe_values(self):
        assert COCO_RECIPE["epochs"] == 100
        assert COCO_RECIPE["optimizer"] == "sgd"
        assert COCO_RECIPE["batch_size"] == 32

    def test_rf100vl_recipe_values(self):
        assert RF100VL_RECIPE["epochs"] == 75
        assert RF100VL_RECIPE["optimizer"] == "adamw"
        assert RF100VL_RECIPE["batch_size"] == 16


class TestValidationWithMAPEval:
    def test_validation_step_without_evaluator(self, small_model):
        """Without val_ann_file, validation computes loss via train-mode forward."""
        lit = YoloNASLightningModule(model=small_model, num_classes=80, warmup_steps=10)

        images = torch.randn(2, 3, 640, 640)
        targets = torch.tensor([
            [0, 5, 0.5, 0.5, 0.1, 0.1],
            [1, 0, 0.6, 0.4, 0.3, 0.3],
        ])

        # Should not raise — uses train-mode fallback for loss
        lit.validation_step((images, targets), 0)

    def test_validation_step_with_evaluator(self, small_model):
        """With val_ann_file and a mock evaluator, mAP path runs."""
        lit = YoloNASLightningModule(
            model=small_model,
            num_classes=80,
            warmup_steps=10,
            val_ann_file="/fake/ann.json",
        )

        # Mock the evaluator to avoid needing a real COCO ann file
        mock_evaluator = MagicMock()
        lit._evaluator = mock_evaluator
        lit.val_dataset_ids = [1, 2]

        images = torch.randn(2, 3, 640, 640)
        targets = torch.tensor([
            [0, 5, 0.5, 0.5, 0.1, 0.1],
            [1, 0, 0.6, 0.4, 0.3, 0.3],
        ])

        # Model in eval mode (as Lightning would set it)
        lit.model.eval()
        lit.validation_step((images, targets), 0)

        # Evaluator should have been called with update
        mock_evaluator.update.assert_called_once()
        call_args = mock_evaluator.update.call_args
        image_ids = call_args[0][0]
        assert image_ids == [1, 2]

    def test_on_validation_epoch_end_logs_map(self, small_model):
        """on_validation_epoch_end should log mAP metrics."""
        lit = YoloNASLightningModule(
            model=small_model,
            num_classes=80,
            warmup_steps=10,
            val_ann_file="/fake/ann.json",
        )

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {
            "mAP": 0.35, "mAP_50": 0.55, "mAP_75": 0.38,
            "mAP_small": 0.15, "mAP_medium": 0.35, "mAP_large": 0.50,
        }
        lit._evaluator = mock_evaluator
        lit.log = MagicMock()

        lit.on_validation_epoch_end()

        # Check mAP was logged
        log_calls = {c[0][0]: c[0][1] for c in lit.log.call_args_list}
        assert log_calls["val/mAP"] == 0.35
        assert log_calls["val/mAP_50"] == 0.55
        assert log_calls["val/mAP_75"] == 0.38

        # Evaluator should be cleared
        assert lit._evaluator is None


class TestDiscoverDatasets:
    def test_discover_datasets(self, tmp_path):
        from modern_yolonas.benchmarks.rf100vl import discover_datasets

        # Create two mock RF100-VL datasets
        for name, n_classes in [("dogs-vs-cats", 2), ("vehicles", 5)]:
            ds_dir = tmp_path / name
            (ds_dir / "train").mkdir(parents=True)
            (ds_dir / "valid").mkdir(parents=True)

            ann = {"categories": [{"id": i} for i in range(n_classes)], "images": [], "annotations": []}
            for split in ("train", "valid"):
                with open(ds_dir / split / "_annotations.coco.json", "w") as f:
                    json.dump(ann, f)

        # Create a non-dataset directory (no annotations)
        (tmp_path / "readme.txt").touch()

        datasets = discover_datasets(tmp_path)
        assert len(datasets) == 2

        names = {d["name"] for d in datasets}
        assert "dogs-vs-cats" in names
        assert "vehicles" in names

        dogs = next(d for d in datasets if d["name"] == "dogs-vs-cats")
        assert dogs["num_classes"] == 2

        vehicles = next(d for d in datasets if d["name"] == "vehicles")
        assert vehicles["num_classes"] == 5

    def test_discover_datasets_empty(self, tmp_path):
        from modern_yolonas.benchmarks.rf100vl import discover_datasets

        datasets = discover_datasets(tmp_path)
        assert datasets == []

    def test_discover_datasets_incomplete(self, tmp_path):
        """Dataset with only train annotations should be skipped."""
        from modern_yolonas.benchmarks.rf100vl import discover_datasets

        ds_dir = tmp_path / "incomplete"
        (ds_dir / "train").mkdir(parents=True)
        ann = {"categories": [], "images": [], "annotations": []}
        with open(ds_dir / "train" / "_annotations.coco.json", "w") as f:
            json.dump(ann, f)

        datasets = discover_datasets(tmp_path)
        assert datasets == []


class TestBuildTransforms:
    def test_train_transforms(self):
        from modern_yolonas.training.run import build_transforms

        transforms = build_transforms(COCO_RECIPE, train=True)
        # Without dataset: Affine, ChannelShuffle, HSV, Flip, Resize, Normalize
        assert len(transforms.transforms) == 6

    def test_val_transforms(self):
        from modern_yolonas.training.run import build_transforms

        transforms = build_transforms(COCO_RECIPE, train=False)
        # Should have only Resize + Normalize
        assert len(transforms.transforms) == 2
