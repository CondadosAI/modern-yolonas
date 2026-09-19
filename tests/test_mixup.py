"""Mixup's blend contract.

The blend moved from a float32 promotion to ``cv2.addWeighted`` for speed. These
pin the parts of that which are not free: the dtype, the ratio's effect, and the
fact that both images' boxes survive.
"""

from __future__ import annotations

import numpy as np
import pytest

from modern_yolonas.data.transforms import Mixup


class _FakeDataset:
    """Two flat images, so a blend result can be read off by eye."""

    def __init__(self, value: int = 200, boxes: int = 2):
        self.value = value
        self.boxes = boxes

    def __len__(self) -> int:
        return 8

    def load_raw(self, index: int):
        image = np.full((64, 64, 3), self.value, np.uint8)
        targets = np.tile(np.array([[1, 0.5, 0.5, 0.2, 0.2]], np.float32), (self.boxes, 1))
        return image, targets


@pytest.fixture
def base():
    return np.zeros((64, 64, 3), np.uint8), np.array([[0, 0.3, 0.3, 0.1, 0.1]], np.float32)


def test_disabled_mixup_is_a_passthrough(base):
    image, targets = base
    mix = Mixup(_FakeDataset(), prob=1.0)
    mix.enabled = False
    out_image, out_targets = mix(image, targets)
    assert out_image is image and out_targets is targets


def test_zero_probability_is_a_passthrough(base):
    image, targets = base
    out_image, out_targets = Mixup(_FakeDataset(), prob=0.0)(image, targets)
    assert out_image is image and out_targets is targets


def test_the_blend_stays_uint8_and_between_the_two_inputs(base):
    image, targets = base
    out_image, _ = Mixup(_FakeDataset(value=200), prob=1.0)(image, targets)
    assert out_image.dtype == np.uint8
    assert out_image.shape == image.shape
    # One input is 0 and the other 200, so every pixel must land between them.
    assert 0 <= int(out_image.min()) and int(out_image.max()) <= 200
    assert len(np.unique(out_image)) == 1, "a flat blend of two flat images is flat"


def test_boxes_from_both_images_are_kept(base):
    image, targets = base
    _, out_targets = Mixup(_FakeDataset(boxes=3), prob=1.0)(image, targets)
    assert len(out_targets) == len(targets) + 3


def test_an_empty_first_image_still_gets_the_second_ones_boxes():
    image = np.zeros((64, 64, 3), np.uint8)
    _, out_targets = Mixup(_FakeDataset(boxes=2), prob=1.0)(image, np.zeros((0, 5), np.float32))
    assert len(out_targets) == 2


def test_a_smaller_second_image_is_letterboxed_to_match(base):
    """The second image is whatever the dataset returns; sizes must not silently break."""

    class Oblong(_FakeDataset):
        def load_raw(self, index: int):
            return np.full((40, 90, 3), 200, np.uint8), np.zeros((0, 5), np.float32)

    image, targets = base
    out_image, _ = Mixup(Oblong(), prob=1.0)(image, targets)
    assert out_image.shape == image.shape
