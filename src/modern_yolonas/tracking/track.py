"""Per-object state for :class:`~modern_yolonas.tracking.DeepHMSort`."""

from __future__ import annotations

import enum

import numpy as np


class TrackState(enum.Enum):
    """Where a track stands this frame.

    Members:
        TRACKED: Matched in the current frame.
        LOST: Alive but unmatched — still in the association pool, and still
            eligible to be found again.
        REMOVED: Out of the pool for good.
    """

    TRACKED = enum.auto()
    LOST = enum.auto()
    REMOVED = enum.auto()


class Track:
    """One tracked object.

    There is no Kalman filter here, and that is deliberate rather than missing.
    Deep-EIoU — the tracker Deep HM-SORT extends — abandons it: a constant-velocity
    Gaussian model is a poor description of an athlete changing direction, and the
    prediction it contributes is worth less than the expanded box it would be
    compared against. So :attr:`xyxy` is the last box this track actually matched,
    not a prediction of where it will be.

    Attributes:
        track_id: Stable id, unique within the tracker instance that made it.
        xyxy: ``(4,)`` last matched box, in source-image pixels.
        score: Confidence of the detection that last matched.
        class_id: Class of the detection that last matched. Tracks are not pinned
            to a class unless the tracker was built ``class_aware``.
        feature: ``(D,)`` L2-normalized appearance vector, or ``None`` when the
            track has only ever been matched without one.
        state: See :class:`TrackState`.
        start_frame: Frame index the track was created on.
        frame_id: Frame index it was last matched on.
        hits: How many frames it has matched in total.
        source_index: Row of the current frame's detections it matched, or ``None``
            if it did not match this frame. Used to carry the caller's own columns
            (class names, embeddings) through to the output.
    """

    __slots__ = (
        "track_id",
        "xyxy",
        "score",
        "class_id",
        "feature",
        "state",
        "start_frame",
        "frame_id",
        "hits",
        "source_index",
    )

    def __init__(
        self,
        track_id: int,
        xyxy: np.ndarray,
        score: float,
        class_id: int,
        feature: np.ndarray | None,
        frame_id: int,
        source_index: int,
    ):
        self.track_id = track_id
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(4)
        self.score = float(score)
        self.class_id = int(class_id)
        self.feature: np.ndarray | None = None
        self.state = TrackState.TRACKED
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.hits = 1
        self.source_index: int | None = source_index

        if feature is not None:
            self.feature = _unit(feature)

    def update(
        self,
        xyxy: np.ndarray,
        score: float,
        class_id: int,
        feature: np.ndarray | None,
        frame_id: int,
        source_index: int,
        momentum: float,
    ) -> None:
        """Attach a detection to this track.

        The appearance vector is blended rather than replaced::

            v_k = momentum * v_{k-1} + (1 - momentum) * f_k

        and then re-normalized — without that last step the running vector drifts
        off the unit sphere and a dot product stops being a cosine.

        Args:
            xyxy: ``(4,)`` matched box.
            score: Its confidence.
            class_id: Its class.
            feature: ``(D,)`` appearance vector, or ``None`` to leave the running
                one untouched (a low-score match carries a crop too unreliable to
                learn an identity from).
            frame_id: Current frame index.
            source_index: Row of the current frame's detections.
            momentum: The ``alpha`` above; higher means slower to change.
        """
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(4)
        self.score = float(score)
        self.class_id = int(class_id)
        self.state = TrackState.TRACKED
        self.frame_id = frame_id
        self.source_index = source_index
        self.hits += 1

        if feature is not None:
            unit = _unit(feature)
            if self.feature is None:
                self.feature = unit
            else:
                self.feature = _unit(momentum * self.feature + (1.0 - momentum) * unit)

    def mark_lost(self) -> None:
        self.state = TrackState.LOST
        self.source_index = None

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED
        self.source_index = None

    def __repr__(self) -> str:
        return f"Track(id={self.track_id}, state={self.state.name}, frames={self.start_frame}-{self.frame_id})"


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)
