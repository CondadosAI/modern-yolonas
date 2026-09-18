"""Tests for the ground-truth and validation-sample annotators.

These draw the images logged during validation, so a silent failure here means
looking at a blank or wrongly-coloured comparison for a whole training run.
"""

from __future__ import annotations

import numpy as np

from modern_yolonas.inference.visualize import annotate_validation_sample, draw_ground_truth

GT_GREEN = (0, 255, 0)


def _blank(h: int = 128, w: int = 128) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestDrawGroundTruth:
    def test_leaves_the_original_untouched(self):
        image = _blank()
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        out = draw_ground_truth(image, boxes, np.array([0]))
        assert image.sum() == 0
        assert out.sum() > 0

    def test_draws_in_green(self):
        out = draw_ground_truth(_blank(), np.array([[10, 10, 50, 50]], dtype=np.float32), np.array([0]))
        drawn = {tuple(p) for p in out.reshape(-1, 3) if p.any()}
        assert drawn == {GT_GREEN}

    def test_no_boxes_is_a_no_op(self):
        out = draw_ground_truth(_blank(), np.zeros((0, 4), dtype=np.float32), np.array([], dtype=int))
        assert out.sum() == 0

    def test_custom_class_names_are_used(self):
        # Two different names must not produce identical pixels — the label text
        # is the only thing that differs.
        boxes = np.array([[10, 10, 60, 60]], dtype=np.float32)
        a = draw_ground_truth(_blank(), boxes, np.array([0]), class_names=["aaaa"])
        b = draw_ground_truth(_blank(), boxes, np.array([0]), class_names=["wwww"])
        assert not np.array_equal(a, b)

    def test_out_of_range_class_id_falls_back_to_the_number(self):
        boxes = np.array([[10, 10, 60, 60]], dtype=np.float32)
        out = draw_ground_truth(_blank(), boxes, np.array([99]), class_names=["only"])
        assert out.sum() > 0

    def test_multiple_boxes_all_drawn(self):
        one = draw_ground_truth(_blank(), np.array([[5, 5, 30, 30]], dtype=np.float32), np.array([0]))
        two = draw_ground_truth(
            _blank(),
            np.array([[5, 5, 30, 30], [60, 60, 100, 100]], dtype=np.float32),
            np.array([0, 1]),
        )
        assert np.count_nonzero(two) > np.count_nonzero(one)


class TestAnnotateValidationSample:
    def _chw(self, h: int = 128, w: int = 128) -> np.ndarray:
        return np.full((3, h, w), 0.5, dtype=np.float32)

    def test_returns_hwc_uint8(self):
        out = annotate_validation_sample(
            self._chw(),
            pred_boxes=np.array([[10, 10, 50, 50]], dtype=np.float32),
            pred_scores=np.array([0.9], dtype=np.float32),
            pred_labels=np.array([0]),
            gt_boxes=np.array([[12, 12, 52, 52]], dtype=np.float32),
            gt_labels=np.array([0]),
        )
        assert out.shape == (128, 128, 3)
        assert out.dtype == np.uint8

    def test_channel_order_is_flipped_to_bgr(self):
        # A pure-red RGB input has to come back as pure blue in BGR.
        chw = np.zeros((3, 32, 32), dtype=np.float32)
        chw[0] = 1.0
        out = annotate_validation_sample(
            chw,
            pred_boxes=np.zeros((0, 4), dtype=np.float32),
            pred_scores=np.array([], dtype=np.float32),
            pred_labels=np.array([], dtype=int),
            gt_boxes=np.zeros((0, 4), dtype=np.float32),
            gt_labels=np.array([], dtype=int),
        )
        assert tuple(out[0, 0]) == (0, 0, 255)

    def test_empty_predictions_and_gt_still_returns_the_image(self):
        out = annotate_validation_sample(
            self._chw(),
            pred_boxes=np.zeros((0, 4), dtype=np.float32),
            pred_scores=np.array([], dtype=np.float32),
            pred_labels=np.array([], dtype=int),
            gt_boxes=np.zeros((0, 4), dtype=np.float32),
            gt_labels=np.array([], dtype=int),
        )
        assert out.shape == (128, 128, 3)
        assert np.all(out == 127)

    def test_gt_only_and_pred_only_differ(self):
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        empty = np.zeros((0, 4), dtype=np.float32)
        gt_only = annotate_validation_sample(
            self._chw(), empty, np.array([], dtype=np.float32), np.array([], dtype=int), boxes, np.array([0])
        )
        pred_only = annotate_validation_sample(
            self._chw(), boxes, np.array([0.9], dtype=np.float32), np.array([0]), empty, np.array([], dtype=int)
        )
        assert not np.array_equal(gt_only, pred_only)

    def test_float_input_is_clipped(self):
        chw = np.full((3, 32, 32), 2.0, dtype=np.float32)
        out = annotate_validation_sample(
            chw,
            pred_boxes=np.zeros((0, 4), dtype=np.float32),
            pred_scores=np.array([], dtype=np.float32),
            pred_labels=np.array([], dtype=int),
            gt_boxes=np.zeros((0, 4), dtype=np.float32),
            gt_labels=np.array([], dtype=int),
        )
        assert out.max() == 255
