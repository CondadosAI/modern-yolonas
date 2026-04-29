"""Tests for training module: optimizer, scheduler, EMA, callbacks, metrics."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset

from modern_yolonas import yolo_nas_s
from modern_yolonas.data.collate import detection_collate_fn
from modern_yolonas.training.callbacks import (
    Callback,
    CSVLoggerCallback,
    EarlyStoppingCallback,
    RichProgressCallback,
)
from modern_yolonas.training.ema import ModelEMA
from modern_yolonas.training.metrics import DetectionMetrics
from modern_yolonas.training.optimizer import create_optimizer
from modern_yolonas.training.scheduler import cosine_with_warmup
from modern_yolonas.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Shared helpers for Trainer tests
# ---------------------------------------------------------------------------

class _FakeDetDataset(TorchDataset):
    """Minimal dataset returning pre-normalised CHW float32 images."""

    def __init__(self, n: int = 4, img_size: int = 128):
        self.n = n
        self.img_size = img_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)
        img = rng.random((3, self.img_size, self.img_size)).astype(np.float32)
        targets = np.array([[0, 0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        return img, targets


def _make_det_loader(n: int = 4, batch_size: int = 2, img_size: int = 128) -> DataLoader:
    return DataLoader(
        _FakeDetDataset(n=n, img_size=img_size),
        batch_size=batch_size,
        collate_fn=detection_collate_fn,
    )


class _TinyModel(nn.Module):
    """Small CNN with a Conv + BN + bias to exercise optimizer param groups."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(8)
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.head(x.mean(dim=(2, 3)))


class TestCreateOptimizer:
    def test_adamw_default(self):
        model = _TinyModel()
        opt = create_optimizer(model, name="adamw", lr=1e-3, weight_decay=1e-4)
        assert isinstance(opt, torch.optim.AdamW)
        # Two param groups: decay and no_decay
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["weight_decay"] == 1e-4
        assert opt.param_groups[1]["weight_decay"] == 0.0

    def test_sgd(self):
        model = _TinyModel()
        opt = create_optimizer(model, name="SGD", lr=0.01, momentum=0.95)
        assert isinstance(opt, torch.optim.SGD)
        assert opt.param_groups[0]["momentum"] == 0.95
        assert opt.param_groups[0]["nesterov"]

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(_TinyModel(), name="rmsprop")

    def test_bn_and_bias_get_no_weight_decay(self):
        model = _TinyModel()
        opt = create_optimizer(model, weight_decay=0.5)
        decay_group, no_decay_group = opt.param_groups
        # BN weight, BN bias, conv bias, linear bias → 4 no-decay params
        assert len(no_decay_group["params"]) == 4
        # Conv weight + linear weight → 2 decay params
        assert len(decay_group["params"]) == 2

    def test_frozen_params_excluded(self):
        model = _TinyModel()
        for p in model.conv.parameters():
            p.requires_grad_(False)
        opt = create_optimizer(model)
        total = sum(len(g["params"]) for g in opt.param_groups)
        # 2 conv params frozen, 4 remaining params kept
        assert total == 4


class TestCosineWithWarmup:
    def _make_opt(self, lr: float = 1.0):
        model = _TinyModel()
        return torch.optim.SGD(model.parameters(), lr=lr)

    def test_warmup_ramps_linearly(self):
        opt = self._make_opt(lr=1.0)
        sched = cosine_with_warmup(opt, warmup_steps=10, warmup_lr=0.0, total_steps=100)
        # LambdaLR runs lr_lambda(0) at init → LR starts at ~0
        lr_start = opt.param_groups[0]["lr"]
        sched.step()
        lr_after_one = opt.param_groups[0]["lr"]
        sched.step()
        lr_after_two = opt.param_groups[0]["lr"]
        # Warmup linearly ramps up
        assert lr_start < lr_after_one < lr_after_two

    def test_post_warmup_follows_cosine(self):
        opt = self._make_opt(lr=1.0)
        sched = cosine_with_warmup(
            opt, warmup_steps=5, warmup_lr=0.1, total_steps=100, cosine_final_lr_ratio=0.1
        )
        for _ in range(5):
            sched.step()
        peak_lr = opt.param_groups[0]["lr"]
        for _ in range(50):
            sched.step()
        mid_lr = opt.param_groups[0]["lr"]
        assert mid_lr < peak_lr  # cosine is decaying

    def test_final_lr_approaches_ratio(self):
        opt = self._make_opt(lr=1.0)
        sched = cosine_with_warmup(
            opt, warmup_steps=0, warmup_lr=0.0, total_steps=10, cosine_final_lr_ratio=0.2
        )
        for _ in range(10):
            sched.step()
        # At end of schedule, LR should be near ratio * base
        assert opt.param_groups[0]["lr"] == pytest.approx(0.2, abs=0.01)


class TestModelEMA:
    def test_init_copies_and_freezes(self):
        model = _TinyModel()
        ema = ModelEMA(model, decay=0.999, warmup_steps=100)
        # EMA params match and are frozen
        for p_ema, p_model in zip(ema.ema.parameters(), model.parameters()):
            assert torch.allclose(p_ema, p_model)
            assert not p_ema.requires_grad
        assert ema.updates == 0

    def test_update_tracks_drift(self):
        torch.manual_seed(0)
        model = _TinyModel()
        ema = ModelEMA(model, decay=0.5, warmup_steps=1)

        # Perturb the model substantially
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        ema.update(model)
        assert ema.updates == 1

        # EMA should have moved toward the new model but not all the way
        any_diff = False
        for p_ema, p_model in zip(ema.ema.parameters(), model.parameters()):
            if not p_ema.dtype.is_floating_point:
                continue
            if not torch.allclose(p_ema, p_model):
                any_diff = True
                break
        assert any_diff

    def test_state_dict_roundtrip(self):
        model = _TinyModel()
        ema = ModelEMA(model, decay=0.9, warmup_steps=5)
        ema.update(model)
        ema.update(model)
        state = ema.state_dict()
        assert state["updates"] == 2
        assert state["decay"] == 0.9

        ema2 = ModelEMA(_TinyModel(), decay=0.1, warmup_steps=5)
        ema2.load_state_dict(state)
        assert ema2.updates == 2
        assert ema2.decay == 0.9


class TestCallbackBase:
    def test_default_hooks_are_noops(self):
        cb = Callback()
        # Should not raise
        cb.on_train_start(trainer=None)
        cb.on_epoch_start(trainer=None, epoch=0)
        cb.on_batch_end(trainer=None, epoch=0, batch_idx=0, loss=0.1, loss_dict={})
        cb.on_epoch_end(trainer=None, epoch=0, avg_loss=0.1)
        cb.on_validation_end(trainer=None, epoch=0, metrics={})
        cb.on_train_end(trainer=None)


class TestCSVLoggerCallback:
    def test_logs_rows(self, tmp_path):
        path = tmp_path / "sub" / "log.csv"
        cb = CSVLoggerCallback(path=path)

        class _FakeTrainer:
            class _Opt:
                param_groups = [{"lr": 1e-3}]

            optimizer = _Opt()

        trainer = _FakeTrainer()
        cb.on_train_start(trainer)
        cb.on_epoch_start(trainer, epoch=0)
        cb.on_batch_end(
            trainer, epoch=0, batch_idx=0, loss=0.5,
            loss_dict={"cls_loss": 0.2, "iou_loss": 0.2, "dfl_loss": 0.1},
        )
        cb.on_batch_end(
            trainer, epoch=0, batch_idx=1, loss=0.4,
            loss_dict={"cls_loss": 0.1, "iou_loss": 0.2, "dfl_loss": 0.1},
        )
        cb.on_epoch_end(trainer, epoch=0, avg_loss=0.45)
        cb.on_train_end(trainer)

        assert path.exists()
        with path.open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["epoch", "loss", "cls_loss", "iou_loss", "dfl_loss", "lr"]
        assert len(rows) == 2  # header + 1 epoch row
        assert rows[1][0] == "1"  # epoch is 1-indexed
        assert float(rows[1][1]) == pytest.approx(0.45, abs=1e-3)

    def test_on_train_end_without_start_is_safe(self, tmp_path):
        cb = CSVLoggerCallback(path=tmp_path / "x.csv")
        # Should not crash if train never started
        cb.on_train_end(trainer=None)


class TestEarlyStoppingCallback:
    def test_no_stop_on_improvement(self):
        cb = EarlyStoppingCallback(patience=3, min_delta=1e-3)

        class _T:
            _stop_training = False

        t = _T()
        for epoch, map_val in enumerate([0.5, 0.6, 0.7, 0.8]):
            cb.on_validation_end(t, epoch=epoch, metrics={"val_metrics/mAP": map_val})
        assert not t._stop_training

    def test_stops_after_patience(self):
        cb = EarlyStoppingCallback(patience=2, min_delta=1e-3)

        class _T:
            _stop_training = False

        t = _T()
        cb.on_validation_end(t, epoch=0, metrics={"val_metrics/mAP": 0.5})  # best
        cb.on_validation_end(t, epoch=1, metrics={"val_metrics/mAP": 0.5})  # no improvement, wait=1
        assert not t._stop_training
        cb.on_validation_end(t, epoch=2, metrics={"val_metrics/mAP": 0.5})  # wait=2, triggers stop
        assert t._stop_training

    def test_min_delta_requires_real_improvement(self):
        cb = EarlyStoppingCallback(patience=1, min_delta=0.1)

        class _T:
            _stop_training = False

        t = _T()
        cb.on_validation_end(t, epoch=0, metrics={"val_metrics/mAP": 0.5})
        # Improves but not by min_delta → counts as no improvement
        cb.on_validation_end(t, epoch=1, metrics={"val_metrics/mAP": 0.55})
        assert t._stop_training


class TestDetectionMetrics:
    """Tests for DetectionMetrics (torchmetrics-backed COCO-style mAP)."""

    def _make_perfect_pred(self, boxes: torch.Tensor, labels: torch.Tensor) -> dict:
        return {
            "boxes":  boxes.float(),
            "scores": torch.ones(len(boxes)),
            "labels": labels.int(),
        }

    def _make_target(self, boxes: torch.Tensor, labels: torch.Tensor) -> dict:
        return {"boxes": boxes.float(), "labels": labels.int()}

    def test_compute_returns_expected_keys(self):
        metrics = DetectionMetrics()
        boxes   = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        labels  = torch.tensor([0])
        metrics.update([self._make_perfect_pred(boxes, labels)], [self._make_target(boxes, labels)])
        result = metrics.compute()
        assert set(result.keys()) == {"mAP", "mAP_50", "mAR_100"}
        assert all(isinstance(v, float) for v in result.values())

    def test_perfect_predictions_give_map50_one(self):
        metrics = DetectionMetrics()
        # Two images, each with one GT box; predictions match exactly.
        boxes_a = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        boxes_b = torch.tensor([[50.0, 50.0, 80.0, 80.0]])
        metrics.update(
            preds=[
                self._make_perfect_pred(boxes_a, torch.tensor([0])),
                self._make_perfect_pred(boxes_b, torch.tensor([1])),
            ],
            targets=[
                self._make_target(boxes_a, torch.tensor([0])),
                self._make_target(boxes_b, torch.tensor([1])),
            ],
        )
        result = metrics.compute()
        assert result["mAP_50"] == pytest.approx(1.0)

    def test_no_predictions_gives_zero_map(self):
        metrics = DetectionMetrics()
        boxes = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        # Empty predictions against a real target → zero recall, zero mAP
        metrics.update(
            preds=[{"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int)}],
            targets=[self._make_target(boxes, torch.tensor([0]))],
        )
        result = metrics.compute()
        assert result["mAP"] == pytest.approx(0.0)
        assert result["mAP_50"] == pytest.approx(0.0)

    def test_reset_clears_state(self):
        metrics = DetectionMetrics()
        boxes  = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        labels = torch.tensor([0])
        metrics.update([self._make_perfect_pred(boxes, labels)], [self._make_target(boxes, labels)])
        # After perfect update mAP_50 is 1.0 — confirm state was accumulated
        assert metrics.compute()["mAP_50"] == pytest.approx(1.0)

        metrics.reset()

        # After reset, a completely wrong prediction (box far from GT) → mAP_50 = 0
        wrong_pred = {
            "boxes":  torch.tensor([[200.0, 200.0, 250.0, 250.0]]),  # no overlap with GT
            "scores": torch.tensor([0.99]),
            "labels": torch.tensor([0], dtype=torch.int),
        }
        metrics.update([wrong_pred], [self._make_target(boxes, labels)])
        result = metrics.compute()
        assert result["mAP_50"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

class TestTrainer:
    def test_init_no_ema(self, tmp_path):
        model = yolo_nas_s(pretrained=False)
        trainer = Trainer(
            model, _make_det_loader(),
            device="cpu", use_amp=False, use_ema=False,
            epochs=1, output_dir=tmp_path,
        )
        assert trainer.epochs == 1
        assert trainer.device == torch.device("cpu")
        assert trainer.ema is None
        assert trainer.scaler is None
        assert trainer.gradient_accum == 1

    def test_init_with_ema(self, tmp_path):
        model = yolo_nas_s(pretrained=False)
        trainer = Trainer(
            model, _make_det_loader(),
            device="cpu", use_amp=False, use_ema=True,
            epochs=1, output_dir=tmp_path,
        )
        assert trainer.ema is not None

    def test_save_checkpoint_last(self, tmp_path):
        trainer = Trainer(
            yolo_nas_s(pretrained=False), _make_det_loader(),
            device="cpu", use_amp=False, use_ema=True,
            epochs=2, output_dir=tmp_path,
        )
        trainer._save_checkpoint(epoch=0)
        assert (tmp_path / "last.pt").exists()

    def test_save_checkpoint_best(self, tmp_path):
        trainer = Trainer(
            yolo_nas_s(pretrained=False), _make_det_loader(),
            device="cpu", use_amp=False, use_ema=True,
            epochs=2, output_dir=tmp_path,
        )
        trainer._save_checkpoint(epoch=0, is_best=True)
        assert (tmp_path / "best.pt").exists()

    def test_save_checkpoint_milestone(self, tmp_path):
        trainer = Trainer(
            yolo_nas_s(pretrained=False), _make_det_loader(),
            device="cpu", use_amp=False, use_ema=False,
            epochs=55, output_dir=tmp_path,
        )
        trainer._save_checkpoint(epoch=49)  # epoch 50
        assert (tmp_path / "epoch_50.pt").exists()

    def test_resume(self, tmp_path):
        loader = _make_det_loader()
        trainer = Trainer(
            yolo_nas_s(pretrained=False), loader,
            device="cpu", use_amp=False, use_ema=True,
            epochs=3, output_dir=tmp_path,
        )
        trainer.global_step = 10
        trainer.best_map = 0.75
        trainer._save_checkpoint(epoch=1)

        trainer2 = Trainer(
            yolo_nas_s(pretrained=False), loader,
            device="cpu", use_amp=False, use_ema=True,
            epochs=3, output_dir=tmp_path,
        )
        trainer2.resume(tmp_path / "last.pt")
        assert trainer2.start_epoch == 2
        assert trainer2.best_map == pytest.approx(0.75)
        assert trainer2.global_step == 10

    def test_train_one_epoch(self, tmp_path):
        trainer = Trainer(
            yolo_nas_s(pretrained=False), _make_det_loader(),
            device="cpu", use_amp=False, use_ema=False,
            epochs=1, output_dir=tmp_path, warmup_steps=0,
        )
        trainer.train()
        assert (tmp_path / "last.pt").exists()
        assert trainer.global_step == 2  # 4 samples / batch_size=2

    def test_train_fires_callbacks(self, tmp_path):
        events = []

        class _Recorder(Callback):
            def on_train_start(self, trainer): events.append("train_start")
            def on_epoch_start(self, trainer, epoch): events.append(f"epoch_start_{epoch}")
            def on_batch_end(self, trainer, epoch, batch_idx, loss, loss_dict): events.append("batch_end")
            def on_epoch_end(self, trainer, epoch, avg_loss): events.append("epoch_end")
            def on_train_end(self, trainer): events.append("train_end")

        trainer = Trainer(
            yolo_nas_s(pretrained=False), _make_det_loader(),
            device="cpu", use_amp=False, use_ema=False,
            epochs=1, output_dir=tmp_path, warmup_steps=0,
            callbacks=[_Recorder()],
        )
        trainer.train()
        assert events == ["train_start", "epoch_start_0", "batch_end", "batch_end", "epoch_end", "train_end"]

    def test_train_with_ema(self, tmp_path):
        """Cover the EMA update path inside the training loop."""
        trainer = Trainer(
            yolo_nas_s(pretrained=False), _make_det_loader(),
            device="cpu", use_amp=False, use_ema=True,
            epochs=1, output_dir=tmp_path, warmup_steps=0,
        )
        trainer.train()
        assert trainer.ema.updates > 0

    def test_train_with_validation(self, tmp_path):
        """Cover the _validate() path including vis_images."""
        trainer = Trainer(
            yolo_nas_s(pretrained=False),
            _make_det_loader(),
            val_loader=_make_det_loader(n=2, batch_size=2),
            device="cpu", use_amp=False, use_ema=True,
            epochs=1, output_dir=tmp_path, warmup_steps=0,
            val_freq=1, val_vis_images=2,
        )
        trainer.train()
        assert (tmp_path / "last.pt").exists()

    def test_train_gradient_accum(self, tmp_path):
        """Cover gradient_accum > 1 code path."""
        trainer = Trainer(
            yolo_nas_s(pretrained=False), _make_det_loader(n=4, batch_size=2),
            device="cpu", use_amp=False, use_ema=False,
            epochs=1, output_dir=tmp_path, warmup_steps=0,
            gradient_accum=2,
        )
        trainer.train()
        assert trainer.global_step == 2

    def test_stop_training_flag(self, tmp_path):
        """Cover the _stop_training early exit in train()."""
        class _Stopper(Callback):
            def on_epoch_end(self, trainer, epoch, avg_loss):
                trainer._stop_training = True

        trainer = Trainer(
            yolo_nas_s(pretrained=False), _make_det_loader(),
            device="cpu", use_amp=False, use_ema=False,
            epochs=5, output_dir=tmp_path, warmup_steps=0,
            callbacks=[_Stopper()],
        )
        trainer.train()
        # Stopped after epoch 0 → only 2 batches ran
        assert trainer.global_step == 2


# ---------------------------------------------------------------------------
# RichProgressCallback
# ---------------------------------------------------------------------------

class TestRichProgressCallback:
    def test_full_lifecycle(self):
        class _FakeTrainer:
            epochs = 2
            class _Loader:
                def __len__(self): return 4
            train_loader = _Loader()

        cb = RichProgressCallback()
        t = _FakeTrainer()
        cb.on_train_start(t)
        cb.on_epoch_start(t, epoch=0)
        cb.on_batch_end(t, epoch=0, batch_idx=0, loss=0.5, loss_dict={})
        cb.on_epoch_start(t, epoch=1)  # replaces previous task
        cb.on_batch_end(t, epoch=1, batch_idx=0, loss=0.4, loss_dict={})
        cb.on_train_end(t)
