"""Tests for inference/visualize, nn/drop_path, nn/repvgg, and validation helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from modern_yolonas.inference.visualize import (
    COCO_NAMES,
    _color_for_class,
    draw_detections,
)
from modern_yolonas.nn.drop_path import DropPath, drop_path
from modern_yolonas.nn.repvgg import QARepVGGBlock, Residual
from modern_yolonas.validation import (
    validate_confidence,
    validate_device,
    validate_input_size,
    validate_iou_threshold,
    validate_model_name,
)


class TestValidation:
    def test_confidence_in_range(self):
        assert validate_confidence(0.5) == 0.5
        assert validate_confidence(0.0) == 0.0
        assert validate_confidence(1.0) == 1.0

    def test_confidence_out_of_range(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_confidence(-0.1)

    def test_confidence_percent_hint(self):
        with pytest.raises(ValueError, match=r"Did you mean 0\.5"):
            validate_confidence(50.0)

    def test_iou_threshold_in_range(self):
        assert validate_iou_threshold(0.65) == 0.65

    def test_iou_threshold_percent_hint(self):
        with pytest.raises(ValueError, match=r"Did you mean 0\.65"):
            validate_iou_threshold(65.0)

    def test_iou_threshold_negative(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_iou_threshold(-1.0)

    def test_input_size_valid(self):
        assert validate_input_size(640) == 640
        assert validate_input_size(32) == 32

    def test_input_size_too_small(self):
        with pytest.raises(ValueError, match="at least 32"):
            validate_input_size(16)

    def test_input_size_not_multiple(self):
        with pytest.raises(ValueError, match="multiple of 32"):
            validate_input_size(100)

    def test_model_name_valid(self):
        for name in ("yolo_nas_s", "yolo_nas_m", "yolo_nas_l"):
            assert validate_model_name(name) == name

    def test_model_name_invalid(self):
        with pytest.raises(ValueError, match="Unknown model"):
            validate_model_name("yolo_nas_xl")

    def test_device_cpu(self):
        dev = validate_device("cpu")
        assert dev.type == "cpu"

    def test_device_invalid_string(self):
        with pytest.raises(ValueError, match="Invalid device"):
            validate_device("not-a-device")

    def test_device_cuda_when_unavailable(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA available; cannot test the unavailable path")
        with pytest.raises(ValueError, match="CUDA is not available"):
            validate_device("cuda")


class TestDropPath:
    def test_drop_prob_zero_returns_input_unchanged(self):
        x = torch.randn(2, 3, 4, 4)
        out = drop_path(x, drop_prob=0.0, training=True)
        assert torch.equal(out, x)

    def test_eval_mode_returns_input_unchanged(self):
        x = torch.randn(2, 3, 4, 4)
        out = drop_path(x, drop_prob=0.5, training=False)
        assert torch.equal(out, x)

    def test_drop_prob_one_zeros_everything(self):
        torch.manual_seed(0)
        x = torch.ones(4, 3, 2, 2)
        out = drop_path(x, drop_prob=1.0, training=True)
        assert torch.all(out == 0)

    def test_shape_preserved(self):
        x = torch.randn(4, 3, 8, 8)
        out = drop_path(x, drop_prob=0.3, training=True)
        assert out.shape == x.shape

    def test_module_forward(self):
        layer = DropPath(drop_prob=0.2)
        layer.eval()  # drop_prob applies only in training
        x = torch.randn(2, 4, 8, 8)
        out = layer(x)
        assert torch.equal(out, x)

    def test_module_training_mode_scales(self):
        torch.manual_seed(42)
        layer = DropPath(drop_prob=0.5)
        layer.train()
        x = torch.ones(16, 3, 2, 2)
        out = layer(x)
        # Each batch element is either 0 (dropped) or scaled by 1/0.5 = 2
        unique_vals = torch.unique(out)
        assert set(unique_vals.tolist()).issubset({0.0, 2.0})


class TestResidual:
    def test_returns_input(self):
        x = torch.randn(2, 4)
        assert torch.equal(Residual()(x), x)


class TestQARepVGGBlock:
    def test_training_forward_shape(self):
        block = QARepVGGBlock(in_channels=8, out_channels=8, use_residual_connection=True)
        block.eval()
        x = torch.randn(2, 8, 16, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_stride_2_halves_spatial(self):
        # Residual requires stride==1, so test stride=2 without residual
        block = QARepVGGBlock(
            in_channels=4, out_channels=8, stride=2, use_residual_connection=False
        )
        block.eval()
        x = torch.randn(1, 4, 16, 16)
        out = block(x)
        assert out.shape == (1, 8, 8, 8)

    def test_partial_fusion_preserves_output(self):
        torch.manual_seed(0)
        block = QARepVGGBlock(in_channels=4, out_channels=4, use_residual_connection=True)
        block.eval()
        x = torch.randn(1, 4, 8, 8)
        out_before = block(x).clone()
        block.partial_fusion()
        out_after = block(x)
        assert torch.allclose(out_before, out_after, atol=1e-5)
        assert block.partially_fused
        # Fused branches removed
        assert not hasattr(block, "branch_3x3")
        assert not hasattr(block, "branch_1x1")

    def test_partial_fusion_idempotent(self):
        block = QARepVGGBlock(in_channels=4, out_channels=4)
        block.eval()
        block.partial_fusion()
        block.partial_fusion()  # should no-op, not raise

    def test_full_fusion_preserves_output(self):
        torch.manual_seed(0)
        block = QARepVGGBlock(in_channels=4, out_channels=4, use_residual_connection=True)
        block.eval()
        x = torch.randn(1, 4, 8, 8)
        out_before = block(x).clone()
        block.full_fusion()
        out_after = block(x)
        assert torch.allclose(out_before, out_after, atol=1e-5)
        assert block.fully_fused
        assert not hasattr(block, "post_bn")

    def test_full_fusion_requires_eval(self):
        block = QARepVGGBlock(in_channels=4, out_channels=4)
        block.train()
        with pytest.raises(RuntimeError, match="must not be called"):
            block.full_fusion()

    def test_full_to_partial_is_rejected(self):
        block = QARepVGGBlock(in_channels=4, out_channels=4)
        block.eval()
        block.full_fusion()
        with pytest.raises(NotImplementedError):
            block.partial_fusion()

    def test_use_alpha_learnable(self):
        block = QARepVGGBlock(
            in_channels=4, out_channels=4, use_alpha=True, use_residual_connection=True
        )
        assert isinstance(block.alpha, torch.nn.Parameter)
        block.eval()
        x = torch.randn(1, 4, 8, 8)
        out_before = block(x).clone()
        block.full_fusion()
        out_after = block(x)
        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_build_residual_branches_false_fuses_at_init(self):
        block = QARepVGGBlock(
            in_channels=4, out_channels=4, build_residual_branches=False
        )
        assert block.partially_fused


class TestColorForClass:
    def test_returns_bgr_tuple(self):
        color = _color_for_class(0)
        assert len(color) == 3
        for c in color:
            assert 55 <= c <= 254

    def test_deterministic(self):
        assert _color_for_class(7) == _color_for_class(7)

    def test_different_classes_differ(self):
        # Not a strict requirement of the API, but a useful sanity check for the hash.
        colors = {_color_for_class(i) for i in range(80)}
        assert len(colors) > 50  # almost all distinct


class TestDrawDetections:
    def test_empty_detections_returns_copy(self):
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        out = draw_detections(
            img,
            boxes=np.zeros((0, 4)),
            scores=np.zeros(0),
            class_ids=np.zeros(0, dtype=np.int64),
        )
        assert out.shape == img.shape
        assert np.array_equal(out, img)
        # Returned array is a copy, not the original
        out[0, 0] = [255, 255, 255]
        assert img[0, 0, 0] != 255

    def test_draws_box_changes_pixels(self):
        img = np.full((64, 64, 3), 0, dtype=np.uint8)
        boxes = np.array([[10, 10, 40, 40]], dtype=np.float32)
        out = draw_detections(img, boxes, np.array([0.9]), np.array([0]))
        assert not np.array_equal(out, img)

    def test_class_id_beyond_names_falls_back_to_number(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        out = draw_detections(
            img,
            boxes=np.array([[5, 5, 25, 25]], dtype=np.float32),
            scores=np.array([0.5]),
            class_ids=np.array([999]),  # out of range
        )
        # Just confirm it doesn't crash
        assert out.shape == img.shape

    def test_custom_class_names(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        out = draw_detections(
            img,
            boxes=np.array([[5, 5, 25, 25]], dtype=np.float32),
            scores=np.array([0.5]),
            class_ids=np.array([0]),
            class_names=["widget"],
        )
        assert out.shape == img.shape

    def test_coco_names_has_80(self):
        assert len(COCO_NAMES) == 80
        assert COCO_NAMES[0] == "person"
