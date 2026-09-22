"""Crowd regions must be ignored, not learned and not treated as background.

COCO marks areas holding many un-separated instances with ``iscrowd``. This repo
used to drop them, which teaches the model that a street full of people is
background. They now travel as ordinary targets carrying CROWD_CLASS, and the loss
uses them only to exclude the anchors they cover.

The failure this guards against is silent: without the mask the loss still falls,
the model still trains, and the only symptom is worse recall in busy scenes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from modern_yolonas import CROWD_CLASS, yolo_nas_s
from modern_yolonas.training.loss import PPYoloELoss, VarifocalLoss

SIZE = 320


@pytest.fixture(scope="module")
def predictions():
    torch.manual_seed(0)
    model = yolo_nas_s(pretrained=False, num_classes=80).train()
    with torch.no_grad():
        return model(torch.randn(1, 3, SIZE, SIZE))


def loss_for(predictions, rows) -> float:
    criterion = PPYoloELoss(num_classes=80)
    targets = torch.tensor(rows, dtype=torch.float32)
    total, parts = criterion(predictions, targets, input_size=(SIZE, SIZE), epoch=99)
    return parts["cls_loss"]


# [batch, class, xc, yc, w, h] — normalised
REAL = [0, 5, 0.2, 0.2, 0.15, 0.15]
CROWD_FAR = [0, CROWD_CLASS, 0.8, 0.8, 0.3, 0.3]
CROWD_OVER_REAL = [0, CROWD_CLASS, 0.2, 0.2, 0.2, 0.2]


class TestTheIgnoreMaskActuallyFires:
    def test_a_crowd_region_lowers_the_classification_loss(self, predictions):
        """Anchors under a crowd box were negatives contributing loss; now they are
        excluded, so the sum must drop. Equal values would mean the mask never applied."""
        without = loss_for(predictions, [REAL])
        with_crowd = loss_for(predictions, [REAL, CROWD_FAR])
        assert with_crowd < without, f"{with_crowd} is not below {without}"

    def test_a_bigger_crowd_region_ignores_more(self, predictions):
        small = loss_for(predictions, [REAL, [0, CROWD_CLASS, 0.8, 0.8, 0.1, 0.1]])
        large = loss_for(predictions, [REAL, [0, CROWD_CLASS, 0.6, 0.6, 0.7, 0.7]])
        assert large < small

    def test_a_crowd_region_is_never_a_training_target(self, predictions):
        """Only the real object may produce a box regression loss."""
        criterion = PPYoloELoss(num_classes=80)
        only_crowd = torch.tensor([CROWD_FAR], dtype=torch.float32)
        _, parts = criterion(predictions, only_crowd, input_size=(SIZE, SIZE), epoch=99)
        assert parts["iou_loss"] == 0.0 and parts["dfl_loss"] == 0.0

    def test_a_crowd_overlapping_a_real_object_keeps_the_real_positives(self, predictions):
        """The mask is `inside crowd AND not already assigned`, so a real instance
        inside a crowd region still trains."""
        criterion = PPYoloELoss(num_classes=80)
        rows = torch.tensor([REAL, CROWD_OVER_REAL], dtype=torch.float32)
        _, parts = criterion(predictions, rows, input_size=(SIZE, SIZE), epoch=99)
        assert parts["iou_loss"] > 0.0, "the real object's positives were suppressed"

    def test_no_crowd_leaves_the_loss_bit_for_bit_unchanged(self, predictions):
        """The common path must add nothing at all."""
        a = loss_for(predictions, [REAL])
        b = loss_for(predictions, [REAL, [0, 7, 0.7, 0.7, 0.1, 0.1]])
        assert a != b  # a second real object does change it
        assert loss_for(predictions, [REAL]) == a


class TestVarifocalIgnoreMask:
    def test_masked_anchors_contribute_nothing(self):
        torch.manual_seed(0)
        vfl = VarifocalLoss()
        pred = torch.randn(2, 6, 3)
        gt = torch.zeros(2, 6, 3)
        label = torch.zeros(2, 6, 3)

        full = vfl(pred, gt, label)
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[0, :3] = True
        masked = vfl(pred, gt, label, mask)
        assert masked < full

        # And exactly the complement: masking everything gives zero.
        assert vfl(pred, gt, label, torch.ones(2, 6, dtype=torch.bool)).item() == 0.0

    def test_none_is_the_untouched_path(self):
        torch.manual_seed(0)
        vfl = VarifocalLoss()
        pred, gt, label = torch.randn(1, 4, 2), torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)
        assert vfl(pred, gt, label) == vfl(pred, gt, label, None)


class TestIgnoreMaskGeometry:
    def test_only_anchors_inside_the_box_are_masked(self):
        anchors = torch.tensor([[10.0, 10.0], [50.0, 50.0], [90.0, 90.0]])
        crowd = torch.tensor([[0, CROWD_CLASS, 0.5, 0.5, 0.4, 0.4]], dtype=torch.float32)
        mask = PPYoloELoss._crowd_ignore_mask(crowd, anchors, 1, 100.0, 100.0)
        # box spans 30..70 in both axes
        assert mask.tolist() == [[False, True, False]]

    def test_crowd_boxes_are_matched_to_their_own_image(self):
        anchors = torch.tensor([[50.0, 50.0]])
        crowd = torch.tensor([[1, CROWD_CLASS, 0.5, 0.5, 0.4, 0.4]], dtype=torch.float32)
        mask = PPYoloELoss._crowd_ignore_mask(crowd, anchors, 2, 100.0, 100.0)
        assert mask.tolist() == [[False], [True]]

    def test_no_crowd_returns_none_so_the_common_path_is_free(self):
        anchors = torch.tensor([[1.0, 1.0]])
        empty = torch.zeros(0, 6)
        assert PPYoloELoss._crowd_ignore_mask(empty, anchors, 1, 10.0, 10.0) is None


class TestDatasetKeepsCrowd:
    def test_crowd_annotations_are_kept_and_marked(self, tmp_path):
        import json

        import cv2

        from modern_yolonas.data.coco import COCODetectionDataset

        images = tmp_path / "images"
        images.mkdir()
        cv2.imwrite(str(images / "000000000001.jpg"), np.zeros((64, 64, 3), np.uint8))
        ann = tmp_path / "a.json"
        ann.write_text(json.dumps({
            "images": [{"id": 1, "file_name": "000000000001.jpg", "height": 64, "width": 64}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10],
                 "area": 100, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [20, 20, 30, 30],
                 "area": 900, "iscrowd": 1},
            ],
            "categories": [{"id": 1, "name": "person"}],
        }))
        dataset = COCODetectionDataset(images, ann, transforms=None)
        _, targets = dataset.load_raw(0)
        assert len(targets) == 2, "the crowd annotation was dropped"
        assert sorted(targets[:, 0].tolist()) == [CROWD_CLASS, 0]
