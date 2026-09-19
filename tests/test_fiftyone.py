"""Tests for FiftyOneDetectionDataset — mock-based, no fiftyone required."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modern_yolonas.data.fiftyone import FiftyOneDetectionDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(label: str, bbox: list[float]) -> SimpleNamespace:
    """Mimic a fiftyone Detection object."""
    return SimpleNamespace(label=label, bounding_box=bbox)


def _make_sample(filepath: str, detections: list | None, label_field: str = "ground_truth") -> MagicMock:
    """Mimic a fiftyone Sample with __getitem__ for field access."""
    sample = MagicMock()
    sample.filepath = filepath
    sample.id = filepath  # use filepath as id for simplicity

    dets_obj = None
    if detections is not None:
        dets_obj = SimpleNamespace(detections=detections)

    sample.__getitem__ = lambda self, key: dets_obj if key == label_field else None
    return sample


def _make_dataset(
    samples: list,
    class_names: list[str] | None = None,
    label_field: str = "ground_truth",
) -> FiftyOneDetectionDataset:
    """Create a FiftyOneDetectionDataset bypassing real fiftyone init."""
    fo_dataset = MagicMock()
    fo_dataset.__iter__ = lambda self: iter(samples)
    fo_dataset.__getitem__ = lambda self, sid: next(s for s in samples if s.id == sid)
    fo_dataset.default_classes = class_names or []

    if class_names is None:
        fo_dataset.distinct = MagicMock(return_value=sorted({
            det.label
            for s in samples
            for det in (s[label_field].detections if s[label_field] else [])
        }))

    with patch("modern_yolonas.data.fiftyone.FiftyOneDetectionDataset.__init__", lambda self, *a, **kw: None):
        ds = object.__new__(FiftyOneDetectionDataset)

    ds.fo_dataset = fo_dataset
    ds.label_field = label_field
    ds.transforms = None
    ds.input_size = 640
    ds.sample_ids = [s.id for s in samples]

    if class_names is not None:
        ds.class_names = list(class_names)
    else:
        ds.class_names = sorted({
            det.label
            for s in samples
            for det in (s[label_field].detections if s[label_field] else [])
        })
    ds._name_to_id = {name: i for i, name in enumerate(ds.class_names)}

    return ds


# Tiny 4x4 white image on disk
@pytest.fixture
def tmp_image(tmp_path):
    import cv2

    path = tmp_path / "test.jpg"
    img = np.ones((4, 4, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(path), img)
    return str(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFiftyOneDetectionDataset:
    def test_len(self, tmp_image):
        samples = [
            _make_sample(tmp_image, [_make_detection("cat", [0.1, 0.2, 0.3, 0.4])]),
            _make_sample(tmp_image, [_make_detection("dog", [0.5, 0.5, 0.2, 0.2])]),
        ]
        # Give distinct IDs
        samples[0].id = "s0"
        samples[1].id = "s1"
        ds = _make_dataset(samples, class_names=["cat", "dog"])
        assert len(ds) == 2

    def test_load_raw_shape_and_format(self, tmp_image):
        det = _make_detection("cat", [0.1, 0.2, 0.3, 0.4])
        sample = _make_sample(tmp_image, [det])
        ds = _make_dataset([sample], class_names=["cat"])

        image, targets = ds.load_raw(0)

        assert image.shape == (4, 4, 3)
        assert image.dtype == np.uint8
        assert targets.shape == (1, 5)
        assert targets.dtype == np.float32

    def test_bbox_conversion(self, tmp_image):
        """FiftyOne [x, y, w, h] top-left → [class_id, cx, cy, w, h] center."""
        det = _make_detection("cat", [0.1, 0.2, 0.3, 0.4])
        sample = _make_sample(tmp_image, [det])
        ds = _make_dataset([sample], class_names=["cat"])

        _, targets = ds.load_raw(0)

        cls_id, cx, cy, w, h = targets[0]
        assert cls_id == 0.0
        assert np.isclose(cx, 0.1 + 0.3 / 2)  # 0.25
        assert np.isclose(cy, 0.2 + 0.4 / 2)  # 0.40
        assert np.isclose(w, 0.3)
        assert np.isclose(h, 0.4)

    def test_empty_detections(self, tmp_image):
        sample = _make_sample(tmp_image, [])
        ds = _make_dataset([sample], class_names=["cat"])

        _, targets = ds.load_raw(0)

        assert targets.shape == (0, 5)
        assert targets.dtype == np.float32

    def test_none_detections(self, tmp_image):
        sample = _make_sample(tmp_image, None)
        ds = _make_dataset([sample], class_names=["cat"])

        _, targets = ds.load_raw(0)

        assert targets.shape == (0, 5)

    def test_getitem_applies_transforms(self, tmp_image):
        det = _make_detection("cat", [0.1, 0.2, 0.3, 0.4])
        sample = _make_sample(tmp_image, [det])
        ds = _make_dataset([sample], class_names=["cat"])

        called = {}

        def fake_transform(img, tgt):
            called["img_shape"] = img.shape
            called["tgt_shape"] = tgt.shape
            return img, tgt

        ds.transforms = fake_transform
        image, targets = ds[0]

        assert "img_shape" in called
        assert called["img_shape"] == (4, 4, 3)
        assert called["tgt_shape"] == (1, 5)

    def test_unknown_labels_skipped(self, tmp_image):
        dets = [
            _make_detection("cat", [0.1, 0.2, 0.3, 0.4]),
            _make_detection("unicorn", [0.5, 0.5, 0.1, 0.1]),  # not in class_names
        ]
        sample = _make_sample(tmp_image, dets)
        ds = _make_dataset([sample], class_names=["cat", "dog"])

        _, targets = ds.load_raw(0)

        assert targets.shape == (0,) or targets.shape[0] == 1  # only "cat" kept
        assert len(targets) == 1
        assert targets[0, 0] == 0.0  # cat = class 0

    def test_num_classes(self, tmp_image):
        sample = _make_sample(tmp_image, [_make_detection("cat", [0.1, 0.2, 0.3, 0.4])])
        ds = _make_dataset([sample], class_names=["cat", "dog", "bird"])
        assert ds.num_classes == 3

    def test_multiple_detections(self, tmp_image):
        dets = [
            _make_detection("cat", [0.1, 0.2, 0.3, 0.4]),
            _make_detection("dog", [0.5, 0.5, 0.2, 0.2]),
        ]
        sample = _make_sample(tmp_image, dets)
        ds = _make_dataset([sample], class_names=["cat", "dog"])

        _, targets = ds.load_raw(0)

        assert targets.shape == (2, 5)
        assert targets[0, 0] == 0.0  # cat
        assert targets[1, 0] == 1.0  # dog
