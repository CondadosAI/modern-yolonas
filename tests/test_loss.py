"""Tests for PPYoloE loss: forward/backward, assigner."""

import pytest
import torch

from modern_yolonas.training.loss import PPYoloELoss, bbox_iou, GIoULoss, VarifocalLoss, DFLLoss


class TestBboxIoU:
    def test_perfect_overlap(self):
        box = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        iou = bbox_iou(box, box)
        assert iou.item() == pytest.approx(1.0)

    def test_no_overlap(self):
        box1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        box2 = torch.tensor([[20, 20, 30, 30]], dtype=torch.float32)
        iou = bbox_iou(box1, box2)
        assert iou.item() == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        box1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        box2 = torch.tensor([[5, 5, 15, 15]], dtype=torch.float32)
        iou = bbox_iou(box1, box2)
        # intersection = 5*5=25, union = 100+100-25=175
        assert iou.item() == pytest.approx(25.0 / 175.0, abs=1e-4)


class TestLossComponents:
    def test_giou_loss(self):
        loss_fn = GIoULoss()
        pred = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32, requires_grad=True)
        target = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)
        loss.backward()

    def test_vfl_loss(self):
        loss_fn = VarifocalLoss()
        pred = torch.randn(1, 100, 80, requires_grad=True)
        gt_score = torch.zeros(1, 100, 80)
        label = torch.zeros(1, 100, 80)
        loss = loss_fn(pred, gt_score, label)
        loss.backward()
        assert loss.item() >= 0

    def test_dfl_loss(self):
        loss_fn = DFLLoss()
        pred = torch.randn(10, 4 * 17, requires_grad=True)  # reg_max=16
        target = torch.rand(10, 4) * 15  # targets in [0, 15]
        loss = loss_fn(pred, target)  # [10] per-anchor
        assert loss.shape == (10,)
        loss.sum().backward()
        assert loss.min().item() >= 0


class TestPPYoloELoss:
    def test_forward_backward(self):
        from modern_yolonas import yolo_nas_s

        model = yolo_nas_s(pretrained=False)
        model.train()
        loss_fn = PPYoloELoss(num_classes=80)

        x = torch.randn(2, 3, 640, 640)
        predictions = model(x)

        # Create dummy targets: [batch_idx, class_id, x, y, w, h]
        targets = torch.tensor([
            [0, 5, 0.5, 0.5, 0.1, 0.1],
            [0, 10, 0.3, 0.7, 0.2, 0.15],
            [1, 0, 0.6, 0.4, 0.3, 0.3],
        ])

        loss, loss_dict = loss_fn(predictions, targets)
        assert loss.requires_grad
        loss.backward()
        assert "cls_loss" in loss_dict
        assert "iou_loss" in loss_dict
        assert "dfl_loss" in loss_dict

    def test_no_targets(self):
        from modern_yolonas import yolo_nas_s

        model = yolo_nas_s(pretrained=False)
        model.train()
        loss_fn = PPYoloELoss(num_classes=80)

        x = torch.randn(1, 3, 640, 640)
        predictions = model(x)
        targets = torch.zeros(0, 6)  # No targets

        loss, loss_dict = loss_fn(predictions, targets)
        assert loss.item() == 0.0
