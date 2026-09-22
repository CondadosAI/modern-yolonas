"""Multi-object tracking.

:class:`DeepHMSort` is the built-in tracker — Deep HM-SORT, the harmonic-mean
association from `arXiv:2406.12081 <https://arxiv.org/abs/2406.12081>`_ on top of
Deep-EIoU's expansion scale-up. It consumes the per-object embeddings the detector
already produces in the detection pass, so tracking with appearance costs one
forward pass per frame rather than two.
"""

from modern_yolonas.tracking.deep_hm_sort import DeepHMSort
from modern_yolonas.tracking.track import Track, TrackState

__all__ = ["DeepHMSort", "Track", "TrackState"]
