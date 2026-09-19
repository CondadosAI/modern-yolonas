"""Zero-area annotations must never leave the dataset.

COCO train2017 carries two of them. Albumentations rejects a box whose y_max
equals its y_min, so a single such annotation kills a training run -- but only
once the image happens to be drawn, which took about a thousand steps and looked
like a random crash.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from modern_yolonas.data.coco import COCODetectionDataset
from modern_yolonas.data.transforms import RandomAffine


def coco_with(tmp_path, boxes: list[list[float]]):
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    cv2.imwrite(str(images / "000000000001.jpg"), np.zeros((100, 100, 3), np.uint8))
    ann = tmp_path / "a.json"
    ann.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "000000000001.jpg", "height": 100, "width": 100}],
        "annotations": [
            {"id": i + 1, "image_id": 1, "category_id": 1, "bbox": b,
             "area": max(b[2] * b[3], 0), "iscrowd": 0}
            for i, b in enumerate(boxes)
        ],
        "categories": [{"id": 1, "name": "person"}],
    }))
    return COCODetectionDataset(images, ann, transforms=None)


class TestDegenerateAnnotationsAreDropped:
    def test_a_zero_height_box_never_reaches_the_targets(self, tmp_path):
        ds = coco_with(tmp_path, [[10, 10, 20, 20], [30, 30, 5, 0]])
        _, targets = ds.load_raw(0)
        assert len(targets) == 1

    def test_a_zero_width_box_never_reaches_the_targets(self, tmp_path):
        ds = coco_with(tmp_path, [[10, 10, 20, 20], [30, 30, 0, 5]])
        _, targets = ds.load_raw(0)
        assert len(targets) == 1

    def test_a_negative_dimension_is_dropped(self, tmp_path):
        ds = coco_with(tmp_path, [[10, 10, 20, 20], [30, 30, -4, 5]])
        _, targets = ds.load_raw(0)
        assert len(targets) == 1

    def test_ordinary_boxes_are_untouched(self, tmp_path):
        ds = coco_with(tmp_path, [[10, 10, 20, 20], [40, 40, 30, 10]])
        _, targets = ds.load_raw(0)
        assert len(targets) == 2

    def test_a_sub_pixel_but_non_zero_box_survives_the_dataset(self, tmp_path):
        """Only truly degenerate boxes are dropped here. Anything with real extent
        is left to `_BBOX_PARAMS`, which filters on size after the transforms."""
        ds = coco_with(tmp_path, [[10, 10, 0.5, 0.5]])
        _, targets = ds.load_raw(0)
        assert len(targets) == 1


def test_the_transform_that_used_to_crash_now_runs(tmp_path):
    """The end-to-end shape of the failure: a real image plus a zero-height box
    through the Albumentations-backed affine that the COCO recipe uses."""
    ds = coco_with(tmp_path, [[10, 10, 20, 20], [30, 30, 5, 0]])
    image, targets = ds.load_raw(0)
    out_image, out_targets = RandomAffine(degrees=0.0, translate=0.1, scale=(1.0, 1.0))(image, targets)
    assert out_image.shape == image.shape
    assert len(out_targets) <= 1


def test_albumentations_really_does_reject_such_a_box():
    """Pins the reason this filter exists. If a future Albumentations tolerated
    degenerate boxes, this test says so and the filter can be reconsidered."""
    image = np.zeros((100, 100, 3), np.uint8)
    degenerate = np.array([[0.0, 0.5, 0.5, 0.2, 0.0]], np.float32)  # h == 0
    with pytest.raises(ValueError, match="y_max is less than or equal to y_min"):
        RandomAffine(degrees=0.0, translate=0.0, scale=(1.0, 1.0))(image, degenerate)
