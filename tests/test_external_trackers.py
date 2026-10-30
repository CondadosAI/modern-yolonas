"""The roboflow/trackers wrappers: the contract, not the algorithms."""

import numpy as np
import pytest
import supervision as sv

pytest.importorskip("trackers", reason="ByteTrack and OC-SORT are the optional `tracking` extra")

from modern_yolonas.tracking import ByteTrack, OCSort  # noqa: E402


def detections(boxes, scores):
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32),
        confidence=np.asarray(scores, dtype=np.float32),
        class_id=np.zeros(len(boxes), dtype=int),
    )


@pytest.mark.parametrize("cls", [ByteTrack, OCSort])
def test_unconfirmed_detections_are_not_returned(cls):
    """A first sighting has id -1 inside the library; it must not reach the caller."""
    tracker = cls()
    out = tracker.update_with_detections(detections([[10, 10, 50, 90]], [0.9]))
    assert out.tracker_id is None or (out.tracker_id != -1).all()


def test_a_steady_object_keeps_one_id():
    tracker = ByteTrack()
    ids = set()
    for step in range(10):
        box = [10 + step, 10, 50 + step, 90]
        out = tracker.update_with_detections(detections([box], [0.9]))
        ids.update(int(i) for i in out.tracker_id)
    assert len(ids) == 1


def test_reset_starts_a_new_id_namespace():
    tracker = ByteTrack()
    for step in range(5):
        first = tracker.update_with_detections(detections([[10 + step, 10, 50 + step, 90]], [0.9]))
    tracker.reset()
    for step in range(5):
        second = tracker.update_with_detections(detections([[10 + step, 10, 50 + step, 90]], [0.9]))
    assert len(first) and len(second)
    assert set(first.tracker_id) == set(second.tracker_id)


def test_changing_the_frame_rate_rebuilds_the_inner_tracker():
    tracker = ByteTrack()
    tracker.update_with_detections(detections([[10, 10, 50, 90]], [0.9]))
    assert tracker._inner is not None
    tracker.frame_rate = 25.0
    assert tracker._inner is None
    tracker.frame_rate = 25.0
    tracker.update_with_detections(detections([[10, 10, 50, 90]], [0.9]))
    assert tracker._inner is not None


def test_empty_input_is_fine():
    tracker = ByteTrack()
    out = tracker.update_with_detections(sv.Detections.empty())
    assert len(out) == 0


def test_the_detector_floor_is_what_the_benchmark_ran():
    assert ByteTrack.track_low_threshold == 0.1
