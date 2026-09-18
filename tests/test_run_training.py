"""Tests for how `run_training` configures Lightning.

The actual fit is stubbed out — what matters here is that the recipe, the logger
choice and the presence of a COCO annotation file end up as the right Trainer
configuration, since getting `monitor` wrong silently keeps the worst checkpoint.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from modern_yolonas.training import run as run_module
from modern_yolonas.training.recipes import COCO_RECIPE


class _TinyDataset(Dataset):
    def __init__(self, n: int = 4, size: int = 64):
        self.n = n
        self.size = size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = np.zeros((3, self.size, self.size), dtype=np.float32)
        targets = np.array([[0, 0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        return img, targets


class _FakeTrainer:
    """Stands in for ``L.Trainer``, recording how it was built and fitted."""

    last: _FakeTrainer | None = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls: list[dict] = []
        _FakeTrainer.last = self

    def fit(self, model, datamodule=None, ckpt_path=None):
        self.fit_calls.append({"model": model, "datamodule": datamodule, "ckpt_path": ckpt_path})


@pytest.fixture
def patched(monkeypatch, tmp_path):
    monkeypatch.setattr(run_module.L, "Trainer", _FakeTrainer)
    # A real builder would download pretrained weights.
    monkeypatch.setitem(
        run_module.MODEL_BUILDERS,
        "yolo_nas_s",
        lambda pretrained=True, num_classes=80: torch.nn.Conv2d(3, 8, 3),
    )
    return tmp_path


def _run(tmp_path, **overrides):
    recipe = {**COCO_RECIPE, "epochs": 1, "batch_size": 2, "input_size": 64, "workers": 0}
    kwargs = dict(
        model_name="yolo_nas_s",
        recipe=recipe,
        train_dataset=_TinyDataset(),
        val_dataset=_TinyDataset(),
        output_dir=tmp_path / "run",
        pretrained=False,
    )
    kwargs.update(overrides)
    return run_module.run_training(**kwargs)


class TestTrainerConfiguration:
    def test_epochs_and_precision_come_from_recipe(self, patched):
        _run(patched)
        kwargs = _FakeTrainer.last.kwargs
        assert kwargs["max_epochs"] == 1
        assert kwargs["precision"] == COCO_RECIPE["precision"]

    def test_epochs_override_wins_over_recipe(self, patched):
        _run(patched, epochs=7)
        assert _FakeTrainer.last.kwargs["max_epochs"] == 7

    def test_devices_are_passed_through(self, patched):
        _run(patched, devices=[0, 1])
        assert _FakeTrainer.last.kwargs["devices"] == [0, 1]

    @pytest.mark.parametrize(
        "logger,expected",
        [
            ("csv", "CSVLogger"),
            pytest.param(
                "tensorboard",
                "TensorBoardLogger",
                marks=pytest.mark.skipif(
                    importlib.util.find_spec("tensorboard") is None,
                    reason="tensorboard is an optional extra",
                ),
            ),
        ],
    )
    def test_logger_backend_selection(self, patched, logger, expected):
        _run(patched, logger=logger)
        assert type(_FakeTrainer.last.kwargs["logger"]).__name__ == expected


class TestCheckpointMonitor:
    def test_without_annotations_monitors_val_loss(self, patched):
        # No COCO annotations means validation reports loss, and lower is better.
        _run(patched)
        ckpt = [c for c in _FakeTrainer.last.kwargs["callbacks"] if hasattr(c, "monitor")][0]
        assert ckpt.monitor == "val/loss"
        assert ckpt.mode == "min"

    def test_with_annotations_monitors_map(self, patched):
        _run(patched, val_ann_file=str(patched / "instances_val.json"))
        ckpt = [c for c in _FakeTrainer.last.kwargs["callbacks"] if hasattr(c, "monitor")][0]
        assert ckpt.monitor == "val/mAP"
        assert ckpt.mode == "max"


class TestCallbacks:
    def test_ema_is_always_attached(self, patched):
        _run(patched)
        names = [type(c).__name__ for c in _FakeTrainer.last.kwargs["callbacks"]]
        assert "EMACallback" in names

    def test_close_mosaic_attached_when_recipe_asks_for_it(self, patched):
        _run(patched)
        names = [type(c).__name__ for c in _FakeTrainer.last.kwargs["callbacks"]]
        assert "CloseMosaicCallback" in names

    def test_close_mosaic_skipped_without_mosaic_or_mixup(self, patched):
        recipe = {
            **COCO_RECIPE,
            "epochs": 1,
            "batch_size": 2,
            "input_size": 64,
            "workers": 0,
            "augmentations": {**COCO_RECIPE["augmentations"], "mosaic": False, "mixup": False},
        }
        _run(patched, recipe=recipe)
        names = [type(c).__name__ for c in _FakeTrainer.last.kwargs["callbacks"]]
        assert "CloseMosaicCallback" not in names


class TestResume:
    def test_resume_path_reaches_fit(self, patched):
        _run(patched, resume_path=patched / "last.ckpt")
        assert _FakeTrainer.last.fit_calls[0]["ckpt_path"] == str(patched / "last.ckpt")

    def test_no_resume_path_means_fresh_fit(self, patched):
        _run(patched)
        assert _FakeTrainer.last.fit_calls[0]["ckpt_path"] is None

    def test_returns_last_checkpoint_when_no_best(self, patched):
        # The fake trainer never runs, so ModelCheckpoint has no best_model_path.
        result = _run(patched)
        assert result.name == "last.ckpt"
