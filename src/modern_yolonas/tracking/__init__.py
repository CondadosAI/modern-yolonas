"""Multi-object tracking.

:class:`DeepHMSort` is the built-in tracker — Deep HM-SORT, the harmonic-mean
association from `arXiv:2406.12081 <https://arxiv.org/abs/2406.12081>`_ on top of
Deep-EIoU's expansion scale-up. It consumes the per-object embeddings the detector
already produces in the detection pass, so tracking with appearance costs one
forward pass per frame rather than two.

:class:`ByteTrack` and :class:`OCSort` wrap roboflow/trackers behind the same
contract. They need the optional ``tracking`` extra, and are imported lazily so
the package still imports without it.
"""

from modern_yolonas.tracking.deep_hm_sort import DeepHMSort
from modern_yolonas.tracking.track import Track, TrackState

__all__ = ["ByteTrack", "DeepHMSort", "OCSort", "Track", "TrackState"]


def __getattr__(name: str):
    if name in ("ByteTrack", "OCSort"):
        from modern_yolonas.tracking import external

        return getattr(external, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
