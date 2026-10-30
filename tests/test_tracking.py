"""Tests for Deep HM-SORT and its matching primitives."""

import numpy as np
import pytest
import supervision as sv

from modern_yolonas.tracking import DeepHMSort, TrackState
from modern_yolonas.tracking.matching import (
    box_iou,
    cosine_distance,
    expand_boxes,
    expansion_iou_distance,
    fuse_costs,
    harmonic_mean,
    linear_assignment,
)

BOX_W, BOX_H = 40.0, 80.0


def box(x, y=100.0):
    return [x, y, x + BOX_W, y + BOX_H]


def frame(boxes, embeddings=None, scores=None, class_ids=None):
    """Build an ``sv.Detections`` the way the detector would."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    detections = sv.Detections(
        xyxy=boxes,
        confidence=np.asarray(
            scores if scores is not None else [0.9] * len(boxes), dtype=np.float32
        ),
        class_id=np.asarray(
            class_ids if class_ids is not None else [0] * len(boxes), dtype=int
        ),
    )
    if embeddings is not None:
        detections.data["embedding"] = np.asarray(embeddings, dtype=np.float32).reshape(len(boxes), -1)
    return detections


def ids_by_x(detections):
    """Map each output box's left edge to the id it was given."""
    return {round(float(b[0])): int(t) for b, t in zip(detections.xyxy, detections.tracker_id)}


# --------------------------------------------------------------------- geometry


def test_expand_boxes_grows_each_side_by_its_own_fraction():
    grown = expand_boxes(np.array([[100.0, 200.0, 140.0, 280.0]]), 0.5)
    # 40 wide, 80 tall: half a width out on each side, half a height up and down.
    assert grown.tolist() == [[80.0, 160.0, 160.0, 320.0]]


def test_expand_boxes_doubles_linear_size_at_half():
    original = np.array([[0.0, 0.0, 10.0, 20.0]])
    grown = expand_boxes(original, 0.5)
    assert grown[0, 2] - grown[0, 0] == pytest.approx(20.0)
    assert grown[0, 3] - grown[0, 1] == pytest.approx(40.0)


def test_expand_boxes_zero_is_identity_and_negative_is_refused():
    original = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert expand_boxes(original, 0.0).tolist() == original.tolist()
    with pytest.raises(ValueError, match="expansion"):
        expand_boxes(original, -0.1)


def test_expand_boxes_handles_no_boxes():
    assert expand_boxes(np.zeros((0, 4)), 0.3).shape == (0, 4)


def test_box_iou_known_values():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    iou = box_iou(a, b)[0]
    assert iou[0] == pytest.approx(1.0)
    assert iou[1] == pytest.approx(50 / 150)
    assert iou[2] == pytest.approx(0.0)


def test_box_iou_empty_sides():
    assert box_iou(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)
    assert box_iou(np.zeros((2, 4)), np.zeros((0, 4))).shape == (2, 0)


def test_expansion_rescues_a_fast_mover_with_no_plain_overlap():
    """The reason ExpansionIoU exists: consecutive boxes that do not touch."""
    before = np.array([box(100)], dtype=np.float32)
    after = np.array([box(145)], dtype=np.float32)

    assert box_iou(before, after)[0, 0] == pytest.approx(0.0)
    assert expansion_iou_distance(before, after, 0.0)[0, 0] == pytest.approx(1.0)
    assert expansion_iou_distance(before, after, 0.3)[0, 0] < 0.9


# ------------------------------------------------------------------ appearance


def test_cosine_distance_endpoints():
    same = np.array([[1.0, 0.0]])
    opposite = np.array([[-1.0, 0.0]])
    orthogonal = np.array([[0.0, 1.0]])

    assert cosine_distance(same, same)[0, 0] == pytest.approx(0.0)
    assert cosine_distance(same, opposite)[0, 0] == pytest.approx(1.0)
    assert cosine_distance(same, orthogonal)[0, 0] == pytest.approx(0.5)


def test_cosine_distance_normalizes_its_inputs():
    """A track vector that drifted off the unit sphere still gets a true cosine."""
    unit = np.array([[1.0, 0.0]])
    scaled = np.array([[7.0, 0.0]])
    assert cosine_distance(scaled, unit)[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_rejects_mismatched_widths():
    with pytest.raises(ValueError, match="width"):
        cosine_distance(np.zeros((2, 4)), np.zeros((3, 8)))


def test_cosine_distance_empty_sides():
    assert cosine_distance(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


# ---------------------------------------------------------------------- fusion


def test_harmonic_mean_matches_the_formula():
    assert harmonic_mean(np.array([0.2]), np.array([0.4]))[0] == pytest.approx(2 * 0.2 * 0.4 / 0.6)


def test_harmonic_mean_survives_zeros():
    assert harmonic_mean(np.array([0.0]), np.array([0.0]))[0] == 0.0
    assert harmonic_mean(np.array([0.0]), np.array([0.7]))[0] == 0.0


def test_harmonic_mean_sits_between_min_and_twice_min():
    rng = np.random.default_rng(0)
    d1, d2 = rng.random((2, 200))
    hm = harmonic_mean(d1, d2)
    smaller = np.minimum(d1, d2)
    assert np.all(hm >= smaller - 1e-6)
    assert np.all(hm <= 2 * smaller + 1e-6)


def test_fuse_costs_gates_appearance_on_distance_and_on_proximity():
    iou = np.array([[0.1, 0.9]], dtype=np.float32)
    # column 0: a good appearance match, but only column 0 is geometrically near.
    emb = np.array([[0.05, 0.05]], dtype=np.float32)

    fused = fuse_costs(iou, emb, proximity_threshold=0.5, appearance_threshold=0.3, fusion="min")
    assert fused[0, 0] == pytest.approx(0.05)
    # Far away, so the appearance cue is discarded and only motion is left.
    assert fused[0, 1] == pytest.approx(0.9)


    gated = fuse_costs(
        iou, np.array([[0.4, 0.05]], dtype=np.float32),
        proximity_threshold=0.5, appearance_threshold=0.3, fusion="min",
    )
    assert gated[0, 0] == pytest.approx(0.1)


def test_fuse_costs_falls_back_to_motion_without_embeddings():
    iou = np.array([[0.3, 0.7]], dtype=np.float32)
    assert fuse_costs(iou, None, proximity_threshold=0.5, appearance_threshold=0.3).tolist() == iou.tolist()


def test_fuse_costs_falls_back_to_motion_where_appearance_is_unavailable():
    """A gated or missing cue must not become a penalty.

    ``harmonic_mean(d, 1) = 2d / (d + 1) > d``, so fusing against a placeholder
    would charge a pair for evidence it never had — and would do it on exactly the
    distant pairs the expansion scale-up exists to reach.
    """
    iou = np.array([[0.2, 0.9]], dtype=np.float32)
    emb = np.array([[np.nan, 0.05]], dtype=np.float32)

    for fusion in ("harmonic", "min"):
        fused = fuse_costs(
            iou, emb, proximity_threshold=0.5, appearance_threshold=0.3, fusion=fusion
        )
        assert np.isfinite(fused).all()
        assert fused[0, 0] == pytest.approx(0.2)  # no embedding on this detection
        assert fused[0, 1] == pytest.approx(0.9)  # too far away to trust appearance


def test_fuse_costs_rejects_an_unknown_fusion():
    with pytest.raises(ValueError, match="fusion"):
        fuse_costs(np.zeros((1, 1)), None, proximity_threshold=0.5, appearance_threshold=0.3, fusion="mean")


# ------------------------------------------------------------------ assignment


def test_linear_assignment_finds_the_optimum():
    cost = np.array([[0.1, 0.9], [0.9, 0.1]])
    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, 0.8)
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert unmatched_rows == [] and unmatched_cols == []


def test_linear_assignment_refuses_pairs_above_the_threshold():
    cost = np.array([[0.1, 0.9], [0.9, 0.95]])
    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, 0.5)
    assert matches == [(0, 0)]
    assert unmatched_rows == [1] and unmatched_cols == [1]


def test_linear_assignment_does_not_buy_a_cheap_pair_with_an_impossible_one():
    """A gated pair must never be taken just to lower the total."""
    cost = np.array([[0.05, 0.99], [0.06, 0.99]])
    matches, _, _ = linear_assignment(cost, 0.5)
    assert len(matches) == 1
    assert matches[0][1] == 0


def test_linear_assignment_handles_empty_and_rectangular():
    assert linear_assignment(np.zeros((0, 3)), 0.5) == ([], [], [0, 1, 2])
    matches, unmatched_rows, unmatched_cols = linear_assignment(np.zeros((1, 3)), 0.5)
    assert len(matches) == 1 and unmatched_rows == [] and len(unmatched_cols) == 2


# --------------------------------------------------------------------- tracker


def test_two_objects_keep_their_ids_across_a_sequence():
    tracker = DeepHMSort()
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])

    seen = []
    for k in range(8):
        out = tracker.update_with_detections(frame([box(100 + 8 * k), box(300 - 8 * k)], [a, b]))
        seen.append(ids_by_x(out))

    assert all(len(f) == 2 for f in seen)
    assert {i for f in seen for i in f.values()} == {1, 2}
    assert seen[0][100] == seen[-1][100 + 8 * 7]


def test_harmonic_mean_prevents_an_id_swap_that_min_fusion_makes():
    """The paper's ablation, reduced to the smallest case that shows it.

    Two lookalikes 25px apart close to 5px apart. On the frame they meet, each
    one's appearance leans very slightly towards the *other* — the situation the
    paper describes for same-team players. The motion cue still points the right
    way. Taking the minimum of the two costs lets the appearance cue decide alone
    and the ids swap; the harmonic mean lets the motion cue veto it.
    """
    def at(degrees):
        rad = np.radians(degrees)
        return np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)

    track_a, track_b = at(0.0), at(-2.5)
    contaminated_a, contaminated_b = at(-44.8), at(42.3)

    def run(fusion):
        tracker = DeepHMSort(fusion=fusion)
        for _ in range(3):
            tracker.update_with_detections(frame([box(100), box(125)], [track_a, track_b]))
        return ids_by_x(
            tracker.update_with_detections(frame([box(105), box(120)], [contaminated_a, contaminated_b]))
        )

    # The regime this depends on: appearance mildly favours the wrong pairing,
    # motion clearly favours the right one, and nothing is gated away.
    appearance = cosine_distance(np.stack([track_a, track_b]), np.stack([contaminated_a, contaminated_b]))
    motion = expansion_iou_distance(
        np.array([box(100), box(125)], np.float32), np.array([box(105), box(120)], np.float32), 0.3
    )
    assert appearance[0, 1] < appearance[0, 0] < 0.3
    assert motion[0, 0] < motion[0, 1] < 0.5

    assert run("harmonic") == {105: 1, 120: 2}
    assert run("min") == {105: 2, 120: 1}


def test_a_track_that_leaves_and_returns_keeps_its_id():
    """Deep HM-SORT's second contribution: tracklets are never discarded.

    Opt-in here rather than the default — see
    :func:`test_the_default_memory_is_two_seconds_of_video`.
    """
    tracker = DeepHMSort(max_lost_seconds=None)
    vector = np.array([1.0, 0.0])

    for _ in range(3):
        first = tracker.update_with_detections(frame([box(100)], [vector]))
    original = int(first.tracker_id[0])

    for _ in range(40):
        assert len(tracker.update_with_detections(sv.Detections.empty())) == 0

    returned = tracker.update_with_detections(frame([box(105)], [vector]))
    assert int(returned.tracker_id[0]) == original


def test_a_frame_budget_gives_a_returning_object_a_new_id():
    tracker = DeepHMSort(max_lost_seconds=0.4, frame_rate=25.0)  # 10 frames
    vector = np.array([1.0, 0.0])

    for _ in range(3):
        first = tracker.update_with_detections(frame([box(100)], [vector]))
    original = int(first.tracker_id[0])

    for _ in range(40):
        tracker.update_with_detections(sv.Detections.empty())

    returned = tracker.update_with_detections(frame([box(105)], [vector]))
    assert int(returned.tracker_id[0]) != original
    assert tracker.tracks == [t for t in tracker.tracks if t.state == TrackState.TRACKED]


def test_the_scale_up_round_catches_what_the_first_round_misses():
    """A jump too big for expansion 0.3, inside reach at 0.6."""
    near, far = np.array([box(100)], np.float32), np.array([box(145)], np.float32)
    assert expansion_iou_distance(near, far, 0.3)[0, 0] > 0.8
    assert expansion_iou_distance(near, far, 0.6)[0, 0] < 0.8

    vector = np.array([1.0, 0.0])

    def run(rounds):
        tracker = DeepHMSort(expansion_rounds=rounds)
        for _ in range(3):
            tracker.update_with_detections(frame([box(100)], [vector]))
        return int(tracker.update_with_detections(frame([box(145)], [vector])).tracker_id[0])

    assert run(2) == 1
    assert run(1) == 2


def test_low_score_detections_hold_a_track_alive_but_never_start_one():
    tracker = DeepHMSort()
    vector = np.array([1.0, 0.0])

    # A lone weak detection is not enough to create a track.
    assert len(tracker.update_with_detections(frame([box(300)], [vector], scores=[0.45]))) == 0

    for _ in range(3):
        tracker.update_with_detections(frame([box(100)], [vector]))
    weak = tracker.update_with_detections(frame([box(105)], [vector], scores=[0.45]))
    assert int(weak.tracker_id[0]) == 1


def test_detections_below_the_low_threshold_are_ignored_entirely():
    tracker = DeepHMSort()
    assert len(tracker.update_with_detections(frame([box(100)], scores=[0.3]))) == 0
    assert tracker.tracks == []


def test_tracking_works_without_any_embeddings():
    """Motion-only association — HM-SORT without the "Deep"."""
    tracker = DeepHMSort()
    for k in range(5):
        out = tracker.update_with_detections(frame([box(100 + 6 * k), box(300)]))
    assert sorted(int(i) for i in out.tracker_id) == [1, 2]


def test_class_aware_refuses_to_associate_across_classes():
    vector = np.array([1.0, 0.0])

    def run(class_aware):
        tracker = DeepHMSort(class_aware=class_aware)
        for _ in range(3):
            tracker.update_with_detections(frame([box(100)], [vector], class_ids=[0]))
        return int(tracker.update_with_detections(frame([box(105)], [vector], class_ids=[7])).tracker_id[0])

    assert run(False) == 1
    assert run(True) == 2


def test_output_carries_every_column_through():
    tracker = DeepHMSort()
    detections = frame([box(100)], [np.array([1.0, 0.0])], scores=[0.77], class_ids=[3])
    detections.data["class_name"] = np.array(["kite"])

    out = tracker.update_with_detections(detections)
    assert out.confidence[0] == pytest.approx(0.77)
    assert out.class_id[0] == 3
    assert out.data["class_name"][0] == "kite"
    assert out.data["embedding"].shape == (1, 2)
    assert out.xyxy[0].tolist() == box(100)


def test_empty_input_is_handled_on_the_first_frame_and_later():
    tracker = DeepHMSort()
    assert len(tracker.update_with_detections(sv.Detections.empty())) == 0
    tracker.update_with_detections(frame([box(100)]))
    assert len(tracker.update_with_detections(sv.Detections.empty())) == 0


def test_reset_clears_state_and_restarts_ids():
    tracker = DeepHMSort()
    tracker.update_with_detections(frame([box(100)]))
    tracker.update_with_detections(frame([box(106)]))
    assert tracker.frame_id == 2

    tracker.reset()
    assert tracker.tracks == [] and tracker.frame_id == 0
    assert int(tracker.update_with_detections(frame([box(400)])).tracker_id[0]) == 1


def test_ids_are_not_reused_after_an_object_disappears():
    tracker = DeepHMSort()
    for _ in range(3):
        tracker.update_with_detections(frame([box(100)]))
    for _ in range(5):
        tracker.update_with_detections(sv.Detections.empty())
    out = tracker.update_with_detections(frame([box(500)]))
    assert int(out.tracker_id[0]) == 2


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"track_low_threshold": 0.8, "track_high_threshold": 0.5}, "track_low_threshold"),
        ({"feature_momentum": 1.0}, "feature_momentum"),
        ({"expansion_rounds": 0}, "expansion_rounds"),
        ({"fusion": "average"}, "fusion"),
        ({"max_lost_seconds": 0}, "max_lost_seconds"),
        ({"frame_rate": 0}, "frame_rate"),
    ],
)
def test_bad_configuration_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DeepHMSort(**kwargs)


def test_feature_ema_stays_on_the_unit_sphere():
    tracker = DeepHMSort()
    tracker.update_with_detections(frame([box(100)], [np.array([1.0, 0.0])]))
    tracker.update_with_detections(frame([box(104)], [np.array([0.0, 1.0])]))
    assert np.linalg.norm(tracker.tracks[0].feature) == pytest.approx(1.0, abs=1e-5)


# ----------------------------------------------------------------- integration


@pytest.fixture(scope="module")
def detector():
    from modern_yolonas import YoloNASDetector

    # Untrained, so the scores are meaningless; the threshold is 0 to make sure
    # there is something to track at all, and multi_label off to keep it small.
    return YoloNASDetector(
        "yolo_nas_s", device="cpu", pretrained=False, input_size=320,
        conf_threshold=0.0, multi_label=False,
    )


def test_the_detector_hands_the_tracker_what_it_expects(detector):
    """One forward pass produces both the boxes and the vectors they associate on."""
    from modern_yolonas import Task

    image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    prediction = detector.predict(image, Task.DETECT | Task.EMBED_OBJECTS)

    assert "embedding" in prediction.detections.data
    assert prediction.detections.data["embedding"].shape == (
        len(prediction.detections),
        detector.embedding_dim,
    )

    tracked = DeepHMSort().update_with_detections(prediction.detections)
    assert len(tracked) <= len(prediction.detections)
    if len(tracked):
        assert tracked.tracker_id is not None
        assert tracked.data["embedding"].shape[0] == len(tracked)


def test_track_video_runs_a_frame_at_a_time(detector, tmp_path):
    import cv2

    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened()
    for k in range(4):
        canvas = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (20 + 10 * k, 40), (60 + 10 * k, 90), (255, 255, 255), -1)
        writer.write(canvas)
    writer.release()

    tracker = DeepHMSort()
    frames = list(detector.track_video(path, tracker, conf_threshold=0.0))

    assert [index for index, _, _ in frames] == [0, 1, 2, 3]
    assert all(detections.tracker_id is not None or len(detections) == 0 for _, _, detections in frames)
    assert tracker.frame_id == 4


def test_annotate_tracks_draws_without_a_tracker_id(detector):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    annotated = detector.annotate_tracks(image, frame([box(10, 10)]))
    assert annotated.shape == image.shape


def test_a_weak_detection_alone_cannot_revive_a_lost_track():
    """The low-score round holds live tracks up; it must not resurrect dead ones.

    Regression: the scale-up loop filters lost tracks out of the pool as it goes,
    but it breaks early on a frame with no strong detections — and that is exactly
    the frame where a lone weak box sits near where something used to be. With
    every tracklet kept forever, there is no shortage of such places.
    """
    tracker = DeepHMSort()
    for _ in range(3):
        tracker.update_with_detections(frame([box(100)]))
    for _ in range(5):
        tracker.update_with_detections(sv.Detections.empty())

    assert len(tracker.update_with_detections(frame([box(105)], scores=[0.45]))) == 0
    # A strong detection in the same place is still allowed to find it again.
    assert int(tracker.update_with_detections(frame([box(105)])).tracker_id[0]) == 1


def test_first_frame_tracks_are_confirmed_immediately():
    """There is no earlier frame they could have failed to appear in."""
    tracker = DeepHMSort()
    first = tracker.update_with_detections(frame([box(100)]))
    assert int(first.tracker_id[0]) == 1

    # A jump the unconfirmed round's fixed 0.5 expansion would reach, but which
    # only stays with the same id if the track went through the main pool.
    moved = tracker.update_with_detections(frame([box(145)]))
    assert int(moved.tracker_id[0]) == 1
    assert tracker.tracks[0].hits == 2

def test_the_default_memory_is_two_seconds_of_video():
    """The budget is written in seconds, so it survives a change of frame rate.

    The paper keeps every tracklet forever, which assumes a fixed camera on a
    closed scene. The default here is finite because most footage is not that; the
    length is Deep-EIoU's own (``track_buffer`` 60 at 30 fps).
    """
    assert DeepHMSort().max_lost_frames == 60
    assert DeepHMSort(frame_rate=25.0).max_lost_frames == 50
    assert DeepHMSort(frame_rate=60.0).max_lost_frames == 120
    assert DeepHMSort(max_lost_seconds=None).max_lost_frames is None

    # A budget so short it rounds to nothing still means "one frame", not "none".
    assert DeepHMSort(max_lost_seconds=0.001).max_lost_frames == 1


def test_the_default_budget_expires_a_track_at_the_right_frame():
    """Two seconds at 10 fps is 20 frames, whatever the wall clock says."""
    vector = np.array([1.0, 0.0])

    def run(gap):
        tracker = DeepHMSort(frame_rate=10.0)
        for _ in range(3):
            first = tracker.update_with_detections(frame([box(100)], [vector]))
        for _ in range(gap):
            tracker.update_with_detections(sv.Detections.empty())
        returned = tracker.update_with_detections(frame([box(105)], [vector]))
        return int(first.tracker_id[0]) == int(returned.tracker_id[0])

    assert run(19) is True
    assert run(25) is False


def test_track_video_takes_the_frame_rate_from_the_video(detector, tmp_path):
    """A 10 fps clip must not be given a 30 fps tracker's memory."""
    import cv2

    path = tmp_path / "slow.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    tracker = DeepHMSort()
    assert tracker.max_lost_frames == 60
    list(detector.track_video(path, tracker, conf_threshold=0.9))
    assert tracker.frame_rate == pytest.approx(10.0)
    assert tracker.max_lost_frames == 20


def test_skipping_frames_shortens_the_effective_frame_rate(detector, tmp_path):
    """The tracker sees every third frame, so its seconds have to stretch too."""
    import cv2

    path = tmp_path / "skipped.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
    for _ in range(3):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    tracker = DeepHMSort()
    list(detector.track_video(path, tracker, conf_threshold=0.9, skip_frames=2))
    assert tracker.frame_rate == pytest.approx(10.0)


def _write_clip(path, frames=4):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened()
    for k in range(frames):
        canvas = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (20 + 10 * k, 40), (60 + 10 * k, 90), (255, 255, 255), -1)
        writer.write(canvas)
    writer.release()
    return path


def _spy_tasks(detector, monkeypatch):
    """Record the tasks each predict() call asked for."""
    seen = []
    original = detector.predict

    def spy(image, tasks, *args, **kwargs):
        seen.append(tasks)
        return original(image, tasks, *args, **kwargs)

    monkeypatch.setattr(detector, "predict", spy)
    return seen


def test_track_video_defaults_to_bytetrack(detector, tmp_path, monkeypatch):
    pytest.importorskip("trackers", reason="ByteTrack is the optional `tracking` extra")
    from modern_yolonas import Task

    path = _write_clip(tmp_path / "clip.mp4")
    seen = _spy_tasks(detector, monkeypatch)

    frames = list(detector.track_video(path, conf_threshold=0.0))

    assert [index for index, _, _ in frames] == [0, 1, 2, 3]
    # ByteTrack reads no appearance, so the default skips the ROI pooling.
    assert seen and all(not (tasks & Task.EMBED_OBJECTS) for tasks in seen)


def test_track_video_without_the_extra_says_how_to_install(detector, tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "trackers", None)
    path = _write_clip(tmp_path / "clip.mp4")
    with pytest.raises(ImportError, match=r"modern-yolonas\[tracking\]"):
        next(detector.track_video(path))


def test_deep_hm_sort_still_gets_appearance_by_default(detector, tmp_path, monkeypatch):
    from modern_yolonas import Task

    path = _write_clip(tmp_path / "clip.mp4")
    seen = _spy_tasks(detector, monkeypatch)

    list(detector.track_video(path, DeepHMSort(), conf_threshold=0.0))

    assert seen and all(tasks & Task.EMBED_OBJECTS for tasks in seen)
