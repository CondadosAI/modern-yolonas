"""Tests for the training pieces Lightning does not own: optimizer, scheduler, metrics.

The Trainer/EMA/callback suites that lived here went with the manual trainer; their
Lightning replacements are in ``test_lightning.py``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

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

