"""Tests for training module: optimizer, scheduler, EMA, callbacks, metrics."""

from __future__ import annotations

import csv

import pytest
import torch
from torch import nn

from modern_yolonas.training.callbacks import (
    Callback,
    CSVLoggerCallback,
    EarlyStoppingCallback,
)
from modern_yolonas.training.ema import ModelEMA
from modern_yolonas.training.metrics import DetectionMetrics
from modern_yolonas.training.optimizer import create_optimizer
from modern_yolonas.training.scheduler import cosine_with_warmup


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
        cb.on_batch_end(
            trainer, epoch=0, batch_idx=0, loss=0.5,
            loss_dict={"cls_loss": 0.2, "iou_loss": 0.2, "dfl_loss": 0.1},
        )
        cb.on_batch_end(
            trainer, epoch=0, batch_idx=1, loss=0.4,
            loss_dict={"cls_loss": 0.1, "iou_loss": 0.2, "dfl_loss": 0.1},
        )
        cb.on_train_end(trainer)

        assert path.exists()
        with path.open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["epoch", "batch", "loss", "cls_loss", "iou_loss", "dfl_loss", "lr"]
        assert len(rows) == 3  # header + 2 data rows
        assert rows[1][0] == "0"
        assert rows[1][2].startswith("0.5")

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
        for epoch, loss in enumerate([1.0, 0.9, 0.8, 0.7]):
            cb.on_epoch_end(t, epoch=epoch, avg_loss=loss)
        assert not t._stop_training

    def test_stops_after_patience(self):
        cb = EarlyStoppingCallback(patience=2, min_delta=1e-3)

        class _T:
            _stop_training = False

        t = _T()
        cb.on_epoch_end(t, epoch=0, avg_loss=1.0)  # best
        cb.on_epoch_end(t, epoch=1, avg_loss=1.0)  # no improvement, wait=1
        assert not t._stop_training
        cb.on_epoch_end(t, epoch=2, avg_loss=1.0)  # wait=2, triggers stop
        assert t._stop_training

    def test_min_delta_requires_real_improvement(self):
        cb = EarlyStoppingCallback(patience=1, min_delta=0.1)

        class _T:
            _stop_training = False

        t = _T()
        cb.on_epoch_end(t, epoch=0, avg_loss=1.0)
        # Drop below best but not by min_delta → counts as no improvement
        cb.on_epoch_end(t, epoch=1, avg_loss=0.95)
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
