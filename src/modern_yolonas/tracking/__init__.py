"""Multi-object tracking.

:class:`ByteTrack` is the default tracker, from roboflow/trackers behind this
package's tracker contract; it needs the optional ``tracking`` extra. On SportsMOT
it scores 53.7 HOTA against 47.6 for Deep HM-SORT's default configuration (see
``docs/benchmarks/tracking.md``). :class:`OCSort` is the same library's OC-SORT.

:class:`DeepHMSort` is Deep HM-SORT, the harmonic-mean association from
`arXiv:2406.12081 <https://arxiv.org/abs/2406.12081>`_ on top of Deep-EIoU's
expansion scale-up, associating on the per-object embeddings the detector produces
in the same pass. It needs no extra dependency, and wins when the boxes are clean.

The roboflow/trackers classes are imported lazily, so the package still imports
without the extra.
"""

from typing import Protocol

import supervision as sv

from modern_yolonas.tracking.deep_hm_sort import DeepHMSort
from modern_yolonas.tracking.track import Track, TrackState

__all__ = ["ByteTrack", "DeepHMSort", "OCSort", "Track", "TrackState", "Tracker"]


class Tracker(Protocol):
    """What :meth:`~modern_yolonas.YoloNASDetector.track_video` needs from a tracker."""

    frame_rate: float
    #: The detector runs at this threshold unless told otherwise.
    track_low_threshold: float

    def reset(self) -> None: ...

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections: ...


def __getattr__(name: str):
    if name in ("ByteTrack", "OCSort"):
        from modern_yolonas.tracking import external

        return getattr(external, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
