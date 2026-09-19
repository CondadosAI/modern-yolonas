"""Deep HM-SORT — the built-in multi-object tracker.

Implements *Deep HM-SORT: Enhancing Multi-Object Tracking in Sports with Deep
Features, Harmonic Mean, and Expansion IOU* (Gran-Henriksen, Lindgård, Kiss and
Lindseth, NTNU, `arXiv:2406.12081 <https://arxiv.org/abs/2406.12081>`_), which is
a two-change delta on Deep-EIoU (`arXiv:2306.13074
<https://arxiv.org/abs/2306.13074>`_):

1. the association cost is the **harmonic mean** of the expansion-IoU distance and
   the appearance distance, where Deep-EIoU takes their minimum;
2. lost tracklets are **never discarded**, so an object that leaves the frame and
   comes back is re-identified rather than renumbered.

Everything underneath — the two-round expansion scale-up, the high/low score split,
the unconfirmed stage, no Kalman filter — is Deep-EIoU's, and Deep-EIoU's is
BoT-SORT's before that.

Where the paper is silent or ambiguous the choice is named in the docstring that
makes it, so the interpretation is visible rather than buried.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from modern_yolonas.tracking.matching import (
    GATED,
    box_iou,
    cosine_distance,
    expansion_iou_distance,
    fuse_costs,
    linear_assignment,
)
from modern_yolonas.tracking.track import Track, TrackState


class DeepHMSort:
    """Online multi-object tracker: detections in, stable ids out.

    Usage mirrors ``sv.ByteTrack``, so it is a drop-in for anything already built
    around supervision::

        from modern_yolonas import Task, YoloNASDetector
        from modern_yolonas.tracking import DeepHMSort

        detector = YoloNASDetector("yolo_nas_s", conf_threshold=0.4)
        tracker = DeepHMSort()

        for _, frame, _ in detector.detect_video("match.mp4"):
            result = detector.predict(frame, Task.DETECT | Task.EMBED_OBJECTS)
            tracked = tracker.update_with_detections(result.detections)
            tracked.tracker_id  # (N,) stable ids

    Or let the detector drive the whole loop, which is the same thing with the
    thresholds already lined up — see
    :meth:`~modern_yolonas.inference.detect.YoloNASDetector.track_video`.

    The appearance vectors come from ``detections.data["embedding"]``. Anything
    ``(N, D)`` works, so a purpose-trained re-identification model can be dropped
    in where the YOLO-NAS features are; without the key the tracker degrades
    cleanly to motion-only association, which is HM-SORT without the "Deep".

    Args:
        track_high_threshold: Detections at or above this score go into the first
            association round. The paper's 0.6.
        track_low_threshold: Detections below this are dropped entirely; between
            this and ``track_high_threshold`` they are kept for the second round,
            where they can hold an existing track alive but cannot start one. The
            paper's 0.4 — which means the *detector* has to run at 0.4 or lower,
            or the low band never arrives.
        new_track_threshold: Lowest score an unmatched detection may have and
            still become a new track. The paper gives 0.5 as "the threshold below
            which we discard tracks"; that phrasing also fits
            ``proximity_threshold``, and since Deep-EIoU's default for that is
            also 0.5, the two readings agree on the shipped configuration.
        match_threshold: Association costs above this are refused in the first
            round. The paper's 0.8.
        proximity_threshold: IoU-distance above which the appearance cost is not
            trusted. See :func:`~modern_yolonas.tracking.matching.fuse_costs`.
        appearance_threshold: Cosine-distance above which the appearance cost is
            discarded. The paper's 0.3.
        expansion: Box growth for the first association round; see
            :func:`~modern_yolonas.tracking.matching.expand_boxes`. The paper's 0.3.
        expansion_step: Added to ``expansion`` on each further round. The paper's
            0.3, giving 0.3 then 0.6.
        expansion_rounds: How many scale-up rounds to run. Deep-EIoU's 2.
        low_score_expansion: Box growth for the low-score round, and for the
            unconfirmed round. Deep-EIoU hardcodes 0.5 in both places.
        low_score_match_threshold: Cost ceiling for the low-score round.
        unconfirmed_match_threshold: Cost ceiling for the round that gives
            first-frame tracks a second chance.
        feature_momentum: ``alpha`` in the paper's Equation 1, the exponential
            moving average over a track's appearance vectors. Deep-EIoU's 0.9.
        fusion: ``"harmonic"`` is the paper's contribution; ``"min"`` is
            Deep-EIoU's original, kept so the difference can be measured on your
            own footage rather than taken on faith.
        max_lost: Frames a track may stay unmatched before it is discarded.
            ``None``, the default, is the paper's second contribution: keep every
            tracklet for the whole sequence. That is a *sports* assumption — a
            fixed camera on a closed pitch, where a player who walks off returns
            to roughly where they left. On open-world footage (a street, a
            doorway) the pool instead grows with every object that has ever
            appeared, and both memory and the cost matrix grow with it; set a
            frame budget there.
        class_aware: Refuse to associate a detection with a track of a different
            class. The paper is single-class (athletes) and says nothing about
            this; off is the faithful reading, and a class-agnostic track simply
            carries whatever class its last match had.
    """

    def __init__(
        self,
        track_high_threshold: float = 0.6,
        track_low_threshold: float = 0.4,
        new_track_threshold: float = 0.5,
        match_threshold: float = 0.8,
        proximity_threshold: float = 0.5,
        appearance_threshold: float = 0.3,
        expansion: float = 0.3,
        expansion_step: float = 0.3,
        expansion_rounds: int = 2,
        low_score_expansion: float = 0.5,
        low_score_match_threshold: float = 0.5,
        unconfirmed_match_threshold: float = 0.7,
        feature_momentum: float = 0.9,
        fusion: str = "harmonic",
        max_lost: int | None = None,
        class_aware: bool = False,
    ):
        if not 0.0 <= track_low_threshold <= track_high_threshold <= 1.0:
            raise ValueError(
                "expected 0 <= track_low_threshold <= track_high_threshold <= 1, got "
                f"{track_low_threshold} and {track_high_threshold}"
            )
        if not 0.0 <= feature_momentum < 1.0:
            raise ValueError(f"feature_momentum must be in [0, 1), got {feature_momentum}")
        if expansion_rounds < 1:
            raise ValueError(f"expansion_rounds must be >= 1, got {expansion_rounds}")
        if fusion not in ("harmonic", "min"):
            raise ValueError(f"fusion must be 'harmonic' or 'min', got {fusion!r}")
        if max_lost is not None and max_lost < 0:
            raise ValueError(f"max_lost must be >= 0 or None, got {max_lost}")

        self.track_high_threshold = track_high_threshold
        self.track_low_threshold = track_low_threshold
        self.new_track_threshold = new_track_threshold
        self.match_threshold = match_threshold
        self.proximity_threshold = proximity_threshold
        self.appearance_threshold = appearance_threshold
        self.expansion = expansion
        self.expansion_step = expansion_step
        self.expansion_rounds = expansion_rounds
        self.low_score_expansion = low_score_expansion
        self.low_score_match_threshold = low_score_match_threshold
        self.unconfirmed_match_threshold = unconfirmed_match_threshold
        self.feature_momentum = feature_momentum
        self.fusion = fusion
        self.max_lost = max_lost
        self.class_aware = class_aware

        self.reset()

    def reset(self) -> None:
        """Forget every track and restart ids from 1.

        Call between videos — ids are unique within a tracker instance, not
        globally, so two videos tracked by the same instance share a namespace.
        """
        self._tracked: list[Track] = []
        self._lost: list[Track] = []
        self.frame_id = 0
        self._next_id = 0

    @property
    def tracks(self) -> list[Track]:
        """Every live track, matched this frame or not, for inspection."""
        return list(self._tracked) + list(self._lost)

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # ------------------------------------------------------------------ costs

    def _appearance_distance(self, tracks: list[Track], features: np.ndarray | None) -> np.ndarray | None:
        """``(M, N)`` cosine distance, ``NaN`` wherever either side has no vector.

        :func:`~modern_yolonas.tracking.matching.fuse_costs` turns those ``NaN``\\ s
        into a gated cost, so a track or a detection without an embedding still
        associates — on motion alone.
        """
        if features is None or not tracks or len(features) == 0:
            return None

        rows = [i for i, track in enumerate(tracks) if track.feature is not None]
        usable = np.isfinite(features).all(axis=1)
        if not rows or not usable.any():
            return None

        distance = np.full((len(tracks), len(features)), np.nan, dtype=np.float32)
        cols = np.flatnonzero(usable)
        distance[np.ix_(rows, cols)] = cosine_distance(
            np.stack([tracks[i].feature for i in rows]), features[cols]
        )
        return distance

    def _cost(
        self,
        tracks: list[Track],
        boxes: np.ndarray,
        class_ids: np.ndarray,
        features: np.ndarray | None,
        expansion: float,
        with_appearance: bool,
    ) -> np.ndarray:
        """Association cost between ``tracks`` and a set of detections."""
        if not tracks or len(boxes) == 0:
            return np.zeros((len(tracks), len(boxes)), dtype=np.float32)

        track_boxes = np.stack([track.xyxy for track in tracks])
        iou_distance = expansion_iou_distance(track_boxes, boxes, expansion)

        cost = fuse_costs(
            iou_distance,
            self._appearance_distance(tracks, features) if with_appearance else None,
            proximity_threshold=self.proximity_threshold,
            appearance_threshold=self.appearance_threshold,
            fusion=self.fusion,
        )

        if self.class_aware:
            track_classes = np.array([track.class_id for track in tracks])
            cost = np.where(track_classes[:, None] != class_ids[None, :], GATED, cost)

        return cost

    # ------------------------------------------------------------------ update

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        """Associate one frame's detections with the live tracks.

        Args:
            detections: This frame's detections, in source-image pixels. Per-object
                appearance vectors are read from ``detections.data["embedding"]``,
                an ``(N, D)`` float array — which is exactly what
                :meth:`YoloNASDetector.predict <modern_yolonas.inference.detect.YoloNASDetector.predict>`
                puts there for ``Task.EMBED_OBJECTS``.

        Returns:
            The subset of ``detections`` that matched a track, with
            ``tracker_id`` set and every other column — confidence, class,
            ``data["class_name"]``, ``data["embedding"]`` — carried through
            untouched. Rows come back in track-id order.
        """
        self.frame_id += 1

        boxes, scores, class_ids, features = _unpack(detections)

        keep = scores >= self.track_low_threshold
        high = np.flatnonzero(keep & (scores >= self.track_high_threshold))
        low = np.flatnonzero(keep & (scores < self.track_high_threshold))

        # A track created last frame and not yet seen twice is "unconfirmed": it
        # gets its own, stricter round at the end rather than competing with
        # established tracks for this frame's detections.
        # A track born on frame 1 counts as confirmed straight away — there was no
        # earlier frame it could have failed to appear in, and making the whole
        # first frame's worth of objects fight for the unconfirmed round's narrower
        # terms would cost ids on anything moving from the start.
        def is_confirmed(track):
            return track.hits > 1 or track.start_frame == 1

        confirmed = [track for track in self._tracked if is_confirmed(track)]
        unconfirmed = [track for track in self._tracked if not is_confirmed(track)]
        for track in self._tracked + self._lost:
            track.source_index = None

        activated: list[Track] = []
        # Lost tracks join the first round — retention buys nothing if they are
        # not actually candidates for re-association.
        pool = confirmed + self._lost
        remaining = high

        for round_index in range(self.expansion_rounds):
            if not pool or len(remaining) == 0:
                break
            expansion = self.expansion + self.expansion_step * round_index
            cost = self._cost(pool, boxes[remaining], class_ids[remaining], _rows(features, remaining), expansion, True)
            matches, unmatched_tracks, unmatched_dets = linear_assignment(cost, self.match_threshold)

            for track_index, det_index in matches:
                activated.append(self._attach(pool[track_index], remaining[det_index], boxes, scores, class_ids, features))

            # Deep-EIoU drops lost tracks from the pool after each round, so they
            # only ever compete at the smallest expansion. Kept as-is: a wider
            # search radius for a track last seen long ago is how ids wander.
            pool = [pool[i] for i in unmatched_tracks if pool[i].state == TrackState.TRACKED]
            remaining = remaining[unmatched_dets]

        # Only *tracked* tracks reach the low-score round. A lost track revived by a
        # 0.45-score box near where it used to be is an id resurrected on nothing,
        # and with every tracklet kept forever there are a lot of such boxes. The
        # loop above filters as it goes, but it breaks early on a frame with no
        # strong detections at all — which is exactly the frame this protects.
        pool = [track for track in pool if track.state == TrackState.TRACKED]

        # Second round: low-score detections, motion only. A crop this uncertain is
        # not worth learning an identity from, so no appearance cost and no feature
        # update — it exists to keep a track alive through a bad frame.
        if pool and len(low):
            cost = self._cost(pool, boxes[low], class_ids[low], None, self.low_score_expansion, False)
            matches, unmatched_tracks, _ = linear_assignment(cost, self.low_score_match_threshold)
            for track_index, det_index in matches:
                activated.append(
                    self._attach(pool[track_index], low[det_index], boxes, scores, class_ids, features, learn=False)
                )
            pool = [pool[i] for i in unmatched_tracks]

        newly_lost = []
        for track in pool:
            if track.state != TrackState.LOST:
                track.mark_lost()
                newly_lost.append(track)

        # Unconfirmed tracks against whatever high-score detections are left.
        if unconfirmed:
            cost = self._cost(
                unconfirmed, boxes[remaining], class_ids[remaining], _rows(features, remaining),
                self.low_score_expansion, True,
            )
            matches, unmatched_tracks, unmatched_dets = linear_assignment(cost, self.unconfirmed_match_threshold)
            for track_index, det_index in matches:
                activated.append(
                    self._attach(unconfirmed[track_index], remaining[det_index], boxes, scores, class_ids, features)
                )
            for i in unmatched_tracks:
                unconfirmed[i].mark_removed()
            remaining = remaining[unmatched_dets]

        for det_index in remaining:
            if scores[det_index] < self.new_track_threshold:
                continue
            activated.append(
                Track(
                    track_id=self._new_id(),
                    xyxy=boxes[det_index],
                    score=scores[det_index],
                    class_id=class_ids[det_index],
                    feature=features[det_index] if features is not None else None,
                    frame_id=self.frame_id,
                    source_index=int(det_index),
                )
            )

        self._commit(activated, newly_lost)
        return self._output(detections)

    def _attach(
        self,
        track: Track,
        det_index: int,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        features: np.ndarray | None,
        learn: bool = True,
    ) -> Track:
        feature = features[det_index] if learn and features is not None else None
        if feature is not None and not np.isfinite(feature).all():
            feature = None
        track.update(
            xyxy=boxes[det_index],
            score=scores[det_index],
            class_id=class_ids[det_index],
            feature=feature,
            frame_id=self.frame_id,
            source_index=int(det_index),
            momentum=self.feature_momentum,
        )
        return track

    def _commit(self, activated: list[Track], newly_lost: list[Track]) -> None:
        """Fold this frame's outcome back into the track lists.

        Unconfirmed tracks that failed their round are simply not carried over:
        they were in ``_tracked``, never in ``_lost``, and ``_tracked`` is rebuilt
        from ``activated`` alone.
        """
        matched_ids = {track.track_id for track in activated}

        self._tracked = activated
        self._lost = [
            track
            for track in self._lost + newly_lost
            if track.track_id not in matched_ids and track.state == TrackState.LOST
        ]

        if self.max_lost is not None:
            self._lost = [t for t in self._lost if self.frame_id - t.frame_id <= self.max_lost]

        self._drop_duplicates()

    def _drop_duplicates(self) -> None:
        """Merge a lost track that has drifted onto the same box as a live one.

        Two tracks sitting on one object is the failure that follows from keeping
        every tracklet: a lost id that re-attaches next to the live one and starts
        competing for its detections. Where a tracked and a lost box overlap almost
        exactly, the younger of the two goes.
        """
        if not self._tracked or not self._lost:
            return

        overlap = box_iou(
            np.stack([t.xyxy for t in self._tracked]), np.stack([t.xyxy for t in self._lost])
        )
        drop_tracked, drop_lost = set(), set()
        for i, j in zip(*np.nonzero(overlap > 0.85)):
            age_tracked = self._tracked[i].frame_id - self._tracked[i].start_frame
            age_lost = self._lost[j].frame_id - self._lost[j].start_frame
            if age_tracked > age_lost:
                drop_lost.add(int(j))
            else:
                drop_tracked.add(int(i))

        self._tracked = [t for i, t in enumerate(self._tracked) if i not in drop_tracked]
        self._lost = [t for j, t in enumerate(self._lost) if j not in drop_lost]

    def _output(self, detections: sv.Detections) -> sv.Detections:
        """This frame's matched detections, with ``tracker_id`` attached."""
        matched = sorted(
            (t for t in self._tracked if t.source_index is not None), key=lambda t: t.track_id
        )
        if not matched:
            return sv.Detections.empty()

        result = detections[np.array([t.source_index for t in matched], dtype=int)]
        result.tracker_id = np.array([t.track_id for t in matched], dtype=int)
        return result


def _unpack(detections: sv.Detections) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Pull the arrays the tracker needs out of an ``sv.Detections``."""
    count = len(detections)
    boxes = np.asarray(detections.xyxy, dtype=np.float32).reshape(count, 4)

    scores = (
        np.asarray(detections.confidence, dtype=np.float32)
        if detections.confidence is not None
        else np.ones(count, dtype=np.float32)
    )
    class_ids = (
        np.asarray(detections.class_id, dtype=int)
        if detections.class_id is not None
        else np.zeros(count, dtype=int)
    )

    features = detections.data.get("embedding")
    if features is not None:
        features = np.asarray(features, dtype=np.float32).reshape(count, -1)
        if features.shape[1] == 0:
            features = None

    return boxes, scores, class_ids, features


def _rows(features: np.ndarray | None, index: np.ndarray) -> np.ndarray | None:
    return None if features is None else features[index]
