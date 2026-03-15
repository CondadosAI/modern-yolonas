"""Tests for inference pipeline: preprocess, postprocess, detector."""

import numpy as np
import pytest
import torch

from modern_yolonas.inference.preprocess import letterbox, preprocess
from modern_yolonas.inference.postprocess import postprocess, rescale_boxes


class TestPreprocess:
    def test_letterbox_square(self):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        padded, scale, pad = letterbox(img, 640)
        assert padded.shape == (640, 640, 3)
        # rescale_size=636 by default, so scale = 636/640
        assert scale == pytest.approx(636 / 640, abs=1e-4)

    def test_letterbox_landscape(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        padded, scale, pad = letterbox(img, 640)
        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(636 / 640, abs=1e-4)

    def test_letterbox_portrait(self):
        img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        padded, scale, pad = letterbox(img, 640)
        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(636 / 640, abs=1e-4)

    def test_letterbox_small(self):
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        padded, scale, pad = letterbox(img, 640)
        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(636 / 300, abs=0.01)

    def test_preprocess_output(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, scale, pad = preprocess(img, 640)
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0


class TestPostprocess:
    def test_basic_nms(self):
        pred_bboxes = torch.tensor([[[10, 10, 100, 100], [12, 12, 102, 102], [200, 200, 300, 300]]], dtype=torch.float32)
        pred_scores = torch.zeros(1, 3, 80)
        pred_scores[0, 0, 0] = 0.9
        pred_scores[0, 1, 0] = 0.8
        pred_scores[0, 2, 5] = 0.7

        results = postprocess(pred_bboxes, pred_scores, conf_threshold=0.5, iou_threshold=0.5)
        boxes, scores, class_ids = results[0]
        # First two boxes overlap heavily; NMS should keep one + the third
        assert len(boxes) == 2

    def test_empty_after_filter(self):
        pred_bboxes = torch.randn(1, 10, 4)
        pred_scores = torch.full((1, 10, 80), 0.01)
        results = postprocess(pred_bboxes, pred_scores, conf_threshold=0.5)
        boxes, scores, class_ids = results[0]
        assert len(boxes) == 0

    def test_rescale_boxes(self):
        boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
        rescaled = rescale_boxes(boxes, scale=2.0, pad=(10, 20), orig_shape=(320, 320))
        # (100-10)/2=45, (100-20)/2=40, (200-10)/2=95, (200-20)/2=90
        assert rescaled[0, 0].item() == pytest.approx(45.0)
        assert rescaled[0, 1].item() == pytest.approx(40.0)
