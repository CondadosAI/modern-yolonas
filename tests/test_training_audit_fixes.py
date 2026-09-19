"""Regression tests for the 2026-09-13 training audit.

Each of these defects is silent: the run completes, the loss curve looks
plausible, and only the final mAP is wrong. See
``docs/guides/training-review-2026-09-13.md``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from modern_yolonas import yolo_nas_s
from modern_yolonas.data.transforms import Mosaic
from modern_yolonas.training.lightning_module import YoloNASLightningModule


class _SquareDataset:
    """Every image carries one box covering a fixed fraction of the frame."""

    def __init__(self, n: int = 8, size: int = 64, box_frac: float = 0.5):
        self.n = n
        self.size = size
        self.box_frac = box_frac

    def __len__(self):
        return self.n

    def load_raw(self, index: int):
        image = np.full((self.size, self.size, 3), 128, dtype=np.uint8)
        # cx, cy at the centre; w, h a known fraction of the frame.
        targets = np.array([[0, 0.5, 0.5, self.box_frac, self.box_frac]], dtype=np.float32)
        return image, targets


class TestMosaicPreservesBoxSize:
    """Mosaic renormalises coordinates by 2 after cropping the 2s canvas to s.

    Widths and heights need that factor as much as the centres do. Without it every
    box is emitted at half its true size, so the model is trained to predict boxes
    half as large as the objects — on every sample, since COCO_RECIPE runs mosaic at
    probability 1.0.
    """

    def _box_areas(self, box_frac: float, trials: int = 40) -> list[float]:
        ds = _SquareDataset(box_frac=box_frac)
        mosaic = Mosaic(dataset=ds, input_size=64, prob=1.0)
        areas = []
        for i in range(trials):
            _, targets = mosaic(i % len(ds))
            for t in targets:
                areas.append(float(t[3] * t[4]))
        return areas

    def test_uncropped_box_keeps_its_size(self):
        # A tile is half the mosaic canvas per side, so a box covering `f` of its
        # tile covers f/2 per side of the canvas — and after the ×2 renormalisation
        # of the s×s crop, `f` again. Boxes clipped by the crop window come out
        # smaller, never larger.
        areas = self._box_areas(box_frac=0.5)
        assert areas, "mosaic produced no boxes at all"
        expected = 0.5 * 0.5
        assert max(areas) > expected * 0.9
        # Before the fix every area was a quarter of the truth (half per side).
        assert max(areas) > expected * 0.5

    def test_scales_with_the_source_box(self):
        small = max(self._box_areas(box_frac=0.25))
        large = max(self._box_areas(box_frac=0.75))
        assert large > small * 2

    def test_boxes_stay_inside_the_frame(self):
        ds = _SquareDataset(box_frac=0.9)
        mosaic = Mosaic(dataset=ds, input_size=64, prob=1.0)
        for i in range(40):
            _, targets = mosaic(i % len(ds))
            for cx, cy, w, h in targets[:, 1:5]:
                # Clipping to the crop window must leave every corner in [0, 1].
                assert cx - w / 2 >= -1e-6
                assert cy - h / 2 >= -1e-6
                assert cx + w / 2 <= 1 + 1e-6
                assert cy + h / 2 <= 1 + 1e-6


class TestValidationDoesNotPolluteBatchNorm:
    """`torch.no_grad` stops gradients, not buffer updates.

    Running the model in train mode to get raw predictions for the validation loss
    lets BatchNorm overwrite its running statistics with validation-set statistics,
    which then go into the checkpoint. The exported model is subtly wrong in a way
    no loss curve shows.
    """

    def _bn_state(self, model) -> list[torch.Tensor]:
        return [
            m.running_mean.clone()
            for m in model.modules()
            if isinstance(m, torch.nn.BatchNorm2d) and m.running_mean is not None
        ]

    def test_running_stats_survive_a_validation_step(self):
        model = yolo_nas_s(pretrained=False, num_classes=4)
        lit = YoloNASLightningModule(model=model, num_classes=4)
        lit.eval()

        before = self._bn_state(model)
        assert before, "model has no BatchNorm layers to check"

        images = torch.randn(2, 3, 128, 128)
        targets = torch.tensor([[0, 0, 0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
        with torch.no_grad():
            lit.validation_step((images, targets), 0)

        after = self._bn_state(model)
        for b, a in zip(before, after):
            assert torch.equal(b, a)

    def test_model_is_left_in_eval_mode(self):
        model = yolo_nas_s(pretrained=False, num_classes=4)
        lit = YoloNASLightningModule(model=model, num_classes=4)
        lit.eval()

        images = torch.randn(2, 3, 128, 128)
        targets = torch.tensor([[0, 0, 0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
        with torch.no_grad():
            lit.validation_step((images, targets), 0)

        assert not model.training
        assert model.heads.return_raw_outputs is False


class TestReturnRawOutputs:
    """The flag that makes the above possible."""

    def test_eval_returns_decoded_only_by_default(self):
        model = yolo_nas_s(pretrained=False, num_classes=4).eval()
        out = model(torch.randn(1, 3, 128, 128))
        pred_bboxes, pred_scores = out
        assert pred_bboxes.ndim == 3
        assert pred_scores.shape[-1] == 4

    def test_eval_returns_raw_when_asked(self):
        model = yolo_nas_s(pretrained=False, num_classes=4).eval()
        model.heads.return_raw_outputs = True
        decoded, raw = model(torch.randn(1, 3, 128, 128))
        assert len(decoded) == 2
        assert len(raw) == 6

    def test_flag_does_not_change_training_mode(self):
        model = yolo_nas_s(pretrained=False, num_classes=4).eval()
        model.heads.return_raw_outputs = True
        model(torch.randn(1, 3, 128, 128))
        assert not model.training


class TestEvaluatorUndoesLetterbox:
    """Predictions come back in letterboxed model space; COCO ground truth is in
    original image pixels. Comparing the two directly costs almost all of the mAP —
    measured at 0.008 vs 0.588 on the same overfit model, because a box scaled about
    the image origin overlaps its own ground truth by well under 0.5 IoU.
    """

    def _ann_file(self, tmp_path, width: int, height: int):
        import json

        payload = {
            "images": [{"id": 1, "file_name": "a.png", "width": width, "height": height}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "thing"}],
        }
        path = tmp_path / "ann.json"
        path.write_text(json.dumps(payload))
        return path

    def _evaluator(self, tmp_path, width, height, input_size):
        from modern_yolonas.training.metrics import COCOEvaluator

        return COCOEvaluator(self._ann_file(tmp_path, width, height), input_size=input_size)

    def test_square_image_is_only_scaled(self, tmp_path):
        # 256 -> 320 means scale 1.25 and no padding.
        ev = self._evaluator(tmp_path, 256, 256, 320)
        box = torch.tensor([[12.5, 12.5, 62.5, 62.5]])
        ev.update([1], [box], [torch.tensor([0.9])], [torch.tensor([0])])
        assert ev.results[0]["bbox"] == [10.0, 10.0, 40.0, 40.0]

    def test_landscape_image_padding_is_removed(self, tmp_path):
        # 200x100 at input 320: scale 1.6, so the image is 320x160 and the 160px of
        # leftover height is padded 80 above and 80 below.
        ev = self._evaluator(tmp_path, 200, 100, 320)
        box = torch.tensor([[16.0, 96.0, 80.0, 160.0]])
        ev.update([1], [box], [torch.tensor([0.9])], [torch.tensor([0])])
        x, y, w, h = ev.results[0]["bbox"]
        assert (x, y) == pytest.approx((10.0, 10.0))
        assert (w, h) == pytest.approx((40.0, 40.0))

    def test_boxes_are_clipped_to_the_image(self, tmp_path):
        ev = self._evaluator(tmp_path, 256, 256, 320)
        box = torch.tensor([[-50.0, -50.0, 5000.0, 5000.0]])
        ev.update([1], [box], [torch.tensor([0.9])], [torch.tensor([0])])
        x, y, w, h = ev.results[0]["bbox"]
        assert x == 0.0 and y == 0.0
        assert x + w <= 256.0 and y + h <= 256.0

    def test_without_input_size_boxes_pass_through(self, tmp_path):
        # A caller that already rescaled must not be rescaled twice.
        from modern_yolonas.training.metrics import COCOEvaluator

        ev = COCOEvaluator(self._ann_file(tmp_path, 256, 256))
        box = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        ev.update([1], [box], [torch.tensor([0.9])], [torch.tensor([0])])
        assert ev.results[0]["bbox"] == [10.0, 10.0, 40.0, 40.0]

    def test_perfect_predictions_score_a_perfect_map(self, tmp_path):
        # End to end through pycocotools: feeding back the ground truth, letterboxed,
        # has to come out as mAP 1.0. It read ~0 before the fix.
        ev = self._evaluator(tmp_path, 256, 256, 320)
        box = torch.tensor([[12.5, 12.5, 62.5, 62.5]])
        ev.update([1], [box], [torch.tensor([0.99])], [torch.tensor([0])])
        assert ev.evaluate()["mAP"] == pytest.approx(1.0)
