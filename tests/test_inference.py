"""Tests for inference pipeline: preprocess, postprocess, detector."""

import numpy as np
import pytest
import torch

from modern_yolonas.inference.preprocess import letterbox, preprocess
from modern_yolonas.inference.postprocess import postprocess, rescale_boxes


class TestPreprocessChannelOrder:
    """Regression: ``preprocess`` must hand the model RGB, not BGR.

    The weights expect RGB — training normalizes to RGB and the Frigate export swaps
    channels in-graph. Feeding BGR costs 3.4 mAP on COCO val2017 (0.4420 vs 0.4761
    for yolo_nas_s), and it degrades quietly rather than failing, so only a test
    catches it.
    """

    def test_swaps_bgr_input_to_rgb_tensor(self):
        # Distinct per-channel values in OpenCV's BGR order.
        blue, green, red = 10, 120, 240
        bgr = np.zeros((640, 640, 3), dtype=np.uint8)
        bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2] = blue, green, red

        tensor, _, _ = preprocess(bgr, 640)

        # Sample the image interior; letterbox pads the border with 114.
        channels = tensor[0, :, 320, 320]
        assert channels[0] == pytest.approx(red / 255.0, abs=1e-3), "channel 0 must be R"
        assert channels[1] == pytest.approx(green / 255.0, abs=1e-3), "channel 1 must be G"
        assert channels[2] == pytest.approx(blue / 255.0, abs=1e-3), "channel 2 must be B"

    def test_output_is_contiguous(self):
        # The channel swap introduces a negative stride, which torch.from_numpy rejects.
        tensor, _, _ = preprocess(np.zeros((480, 640, 3), dtype=np.uint8), 640)
        assert tensor.is_contiguous()


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

    def test_multi_label_keeps_every_class_over_threshold(self):
        # An anchor scoring high on two classes yields two detections in
        # multi-label mode, where single-label mode would keep only the best.
        pred_bboxes = torch.tensor([[[10, 10, 100, 100]]], dtype=torch.float32)
        pred_scores = torch.zeros(1, 1, 80)
        pred_scores[0, 0, 0] = 0.9
        pred_scores[0, 0, 7] = 0.8

        multi = postprocess(pred_bboxes, pred_scores, conf_threshold=0.5, iou_threshold=0.99, multi_label=True)
        single = postprocess(pred_bboxes, pred_scores, conf_threshold=0.5, iou_threshold=0.99, multi_label=False)

        assert set(multi[0][2].tolist()) == {0, 7}
        assert single[0][2].tolist() == [0]

    def test_multi_label_empty_after_filter(self):
        pred_bboxes = torch.randn(1, 10, 4)
        pred_scores = torch.full((1, 10, 80), 0.01)
        boxes, scores, class_ids = postprocess(
            pred_bboxes, pred_scores, conf_threshold=0.5, multi_label=True
        )[0]
        assert len(boxes) == 0
        assert class_ids.dtype == torch.long

    def test_top_k_caps_candidates_before_nms(self):
        # 2000 non-overlapping boxes, all above threshold: the 1024-candidate cap
        # has to kick in before NMS, or NMS runs on the full set.
        n = 2000
        xs = torch.arange(n, dtype=torch.float32).unsqueeze(1) * 10
        pred_bboxes = torch.cat([xs, xs, xs + 5, xs + 5], dim=1).unsqueeze(0)
        pred_scores = torch.zeros(1, n, 80)
        pred_scores[0, :, 0] = torch.linspace(0.51, 0.99, n)

        boxes, scores, class_ids = postprocess(
            pred_bboxes, pred_scores, conf_threshold=0.5, iou_threshold=0.5
        )[0]
        assert len(boxes) <= 1024

    def test_rescale_boxes(self):
        boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
        rescaled = rescale_boxes(boxes, scale=2.0, pad=(10, 20), orig_shape=(320, 320))
        # (100-10)/2=45, (100-20)/2=40, (200-10)/2=95, (200-20)/2=90
        assert rescaled[0, 0].item() == pytest.approx(45.0)
        assert rescaled[0, 1].item() == pytest.approx(40.0)


class TestDetectorRenameAlias:
    """`Detector` was renamed `YoloNASDetector` in 0.5.0; the old spelling still works.

    The alias is a module-level ``__getattr__`` rather than a subclass, so it warns on
    attribute access without putting an extra class in the MRO. It has to hold on every
    path people actually import from, including the one the CLI tests patch.
    """

    MODULES = [
        "modern_yolonas",
        "modern_yolonas.inference",
        "modern_yolonas.inference.detect",
    ]

    @pytest.mark.parametrize("module_name", MODULES)
    def test_alias_is_the_renamed_class_and_warns(self, module_name):
        import importlib

        from modern_yolonas import YoloNASDetector

        module = importlib.import_module(module_name)
        with pytest.warns(DeprecationWarning, match="use YoloNASDetector instead"):
            assert module.Detector is YoloNASDetector

    @pytest.mark.parametrize("module_name", MODULES)
    def test_unknown_attribute_still_raises(self, module_name):
        import importlib

        module = importlib.import_module(module_name)
        with pytest.raises(AttributeError):
            module.NoSuchThing

    def test_new_name_does_not_warn(self, recwarn):
        import importlib

        importlib.reload(importlib.import_module("modern_yolonas")).YoloNASDetector
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
