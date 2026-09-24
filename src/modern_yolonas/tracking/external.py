"""ByteTrack and OC-SORT from roboflow/trackers, behind this package's tracker contract.

The contract is the one :class:`~modern_yolonas.tracking.DeepHMSort` follows and
:meth:`~modern_yolonas.YoloNASDetector.track_video` relies on: ``reset()``, a
settable ``frame_rate``, ``track_low_threshold`` for the detector's floor, and
``update_with_detections(sv.Detections) -> sv.Detections`` returning only boxes
that belong to a track.

roboflow/trackers is the optional ``tracking`` extra. It is not a core dependency
because it requires ``opencv-python``, and this project depends on
``opencv-python-headless``: both ship the ``cv2`` module, and the GUI build wants
libGL at import time on a bare server.
"""

from __future__ import annotations

import supervision as sv

_INSTALL_HINT = (
    "ByteTrack and OC-SORT come from roboflow/trackers, the optional `tracking` extra: "
    'pip install "modern-yolonas[tracking]"  (or: uv sync --extra tracking)'
)


def _import_trackers():
    try:
        import trackers
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return trackers


class _RoboflowTracker:
    """Adapt a roboflow/trackers tracker to the contract above.

    Those trackers take the frame rate in their constructor, where this package
    sets it after construction (``track_video`` reads it off the clip). So the
    inner tracker is built on the first update, and rebuilt whenever
    ``frame_rate`` changes or :meth:`reset` is called.
    """

    #: Name of the class in the ``trackers`` package; set by subclasses.
    _class_name: str

    #: The detector floor. Neither tracker has one of its own: everything below
    #: ``high_conf_det_threshold`` goes to the low-score round, so the detector's
    #: threshold is the only floor there is. 0.1 is what the tracking benchmark ran.
    track_low_threshold: float = 0.1

    def __init__(self, frame_rate: float = 30.0, **kwargs):
        # Fail at construction rather than at the first frame.
        self._cls = getattr(_import_trackers(), self._class_name)
        self._kwargs = kwargs
        self._frame_rate = float(frame_rate)
        self._inner = None

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @frame_rate.setter
    def frame_rate(self, value: float) -> None:
        if float(value) != self._frame_rate:
            self._frame_rate = float(value)
            self._inner = None

    def reset(self) -> None:
        self._inner = None

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        if self._inner is None:
            self._inner = self._cls(frame_rate=self._frame_rate, **self._kwargs)
        tracked = self._inner.update(detections)
        # Detections that did not join a confirmed track come back with id -1.
        # They are not tracks, and a repeated id within a frame is invalid MOT output.
        if len(tracked) and tracked.tracker_id is not None:
            tracked = tracked[tracked.tracker_id != -1]
        return tracked


class ByteTrack(_RoboflowTracker):
    """ByteTrack (arXiv:2110.06864) as implemented by roboflow/trackers.

    Motion only: a Kalman filter and IoU, with a second association round for
    low-score detections. Keyword arguments go to ``trackers.ByteTrackTracker``;
    the library defaults are what the tracking benchmark measured.
    """

    _class_name = "ByteTrackTracker"


class OCSort(_RoboflowTracker):
    """OC-SORT (arXiv:2203.14360) as implemented by roboflow/trackers.

    Kept for the tracking benchmark; see ``docs/benchmarks/tracking.md`` for why it
    is not the default. Keyword arguments go to ``trackers.OCSORTTracker``.
    """

    _class_name = "OCSORTTracker"
