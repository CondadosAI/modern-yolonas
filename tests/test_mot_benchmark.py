"""Tests for the MOT benchmark harness.

No dataset is needed: every test builds a small MOT-format tree in ``tmp_path``. The
end-to-end one needs TrackEval, which is an optional extra (it pulls the non-headless
``opencv-python``), so it skips when that is not installed.
"""

import numpy as np
import importlib.util

import pytest

from modern_yolonas.benchmarks.mot import (
    SWEEP,
    DetectionCache,
    discover_sequences,
    evaluate,
    prepare_trackeval_layout,
    read_gt_boxes,
    read_split_file,
    replay,
    write_mot_file,
)

SEQINFO = """[Sequence]
name={name}
imDir=img1
frameRate={fps}
seqLength={length}
imWidth=64
imHeight=48
imExt=.jpg
"""


def make_sequence(root, name, gt_rows, length=4, fps=25, with_images=False):
    """Write one MOT-format sequence directory."""
    directory = root / name
    (directory / "gt").mkdir(parents=True, exist_ok=True)
    (directory / "seqinfo.ini").write_text(SEQINFO.format(name=name, fps=fps, length=length))
    np.savetxt(directory / "gt" / "gt.txt", np.asarray(gt_rows, dtype=float), fmt="%g", delimiter=",")

    if with_images:
        import cv2

        (directory / "img1").mkdir(exist_ok=True)
        for frame in range(1, length + 1):
            canvas = np.full((48, 64, 3), frame * 10 % 255, dtype=np.uint8)
            cv2.imwrite(str(directory / "img1" / f"{frame:06d}.jpg"), canvas)
    return directory


def two_object_gt(length=4):
    """Two objects walking apart, one row per object per frame."""
    rows = []
    for frame in range(1, length + 1):
        rows.append([frame, 1, 5 + 2 * frame, 10, 8, 16, 1, 1, 1])
        rows.append([frame, 2, 40 - 2 * frame, 10, 8, 16, 1, 1, 1])
    return rows


# ----------------------------------------------------------------------- discovery


def test_discover_sequences_reads_seqinfo(tmp_path):
    make_sequence(tmp_path, "seq-a", two_object_gt(), length=4, fps=25)
    make_sequence(tmp_path, "seq-b", two_object_gt(), length=6, fps=30)

    sequences = discover_sequences(tmp_path)
    assert [s.name for s in sequences] == ["seq-a", "seq-b"]
    assert sequences[0].frame_rate == 25 and sequences[0].length == 4
    assert sequences[1].frame_rate == 30 and sequences[1].length == 6
    assert sequences[0].width == 64 and sequences[0].height == 48


def test_discover_sequences_ignores_directories_without_seqinfo(tmp_path):
    make_sequence(tmp_path, "real", two_object_gt())
    (tmp_path / "not-a-sequence").mkdir()
    assert [s.name for s in discover_sequences(tmp_path)] == ["real"]


def test_discover_sequences_honours_the_requested_order(tmp_path):
    for name in ("a", "b", "c"):
        make_sequence(tmp_path, name, two_object_gt())
    assert [s.name for s in discover_sequences(tmp_path, ["c", "a"])] == ["c", "a"]


def test_discover_sequences_names_what_is_missing(tmp_path):
    make_sequence(tmp_path, "a", two_object_gt())
    with pytest.raises(FileNotFoundError, match="ghost"):
        discover_sequences(tmp_path, ["a", "ghost"])


def test_discover_sequences_refuses_a_non_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_sequences(tmp_path / "nope")


def test_read_split_file_drops_the_header_and_blanks(tmp_path):
    path = tmp_path / "val.txt"
    path.write_text("name\nseq-a\n\nseq-b\n")
    assert read_split_file(path) == ["seq-a", "seq-b"]


def test_frame_path_is_one_indexed_and_zero_padded(tmp_path):
    make_sequence(tmp_path, "seq", two_object_gt())
    sequence = discover_sequences(tmp_path)[0]
    assert sequence.frame_path(1).name == "000001.jpg"
    assert sequence.frame_path(1234).name == "001234.jpg"


# -------------------------------------------------------------------- ground truth


def test_read_gt_boxes_converts_xywh_to_xyxy(tmp_path):
    make_sequence(tmp_path, "seq", [[1, 1, 10, 20, 30, 40, 1, 1, 1]])
    boxes = read_gt_boxes(discover_sequences(tmp_path)[0])
    assert boxes[1].tolist() == [[10.0, 20.0, 40.0, 60.0]]


def test_read_gt_boxes_drops_ignored_and_non_pedestrian_rows(tmp_path):
    """Boxes the evaluator will not score must not be handed to the tracker either."""
    make_sequence(
        tmp_path,
        "seq",
        [
            [1, 1, 10, 20, 30, 40, 1, 1, 1],  # kept
            [1, 2, 10, 20, 30, 40, 0, 1, 1],  # conf 0: marked to be ignored
            [1, 3, 10, 20, 30, 40, 1, 7, 1],  # class 7: a MOT17 distractor
        ],
    )
    boxes = read_gt_boxes(discover_sequences(tmp_path)[0])
    assert len(boxes[1]) == 1


def test_read_gt_boxes_handles_a_frame_with_nothing(tmp_path):
    make_sequence(tmp_path, "seq", [[2, 1, 0, 0, 4, 4, 1, 1, 1]], length=3)
    boxes = read_gt_boxes(discover_sequences(tmp_path)[0])
    assert set(boxes) == {2}


# --------------------------------------------------------------------------- cache


def make_cache(length=4, dim=8):
    rows = np.array(two_object_gt(length))
    xywh = rows[:, 2:6]
    return DetectionCache(
        frame=rows[:, 0].astype(np.int32),
        xyxy=np.column_stack(
            [xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]]
        ).astype(np.float32),
        score=np.ones(len(rows), dtype=np.float32),
        embedding=np.tile(np.eye(2, dim, dtype=np.float32), (length, 1)),
        frame_rate=25.0,
        length=length,
    )


def test_cache_round_trips_through_disk(tmp_path):
    cache = make_cache()
    cache.save(tmp_path / "seq.npz")
    loaded = DetectionCache.load(tmp_path / "seq.npz")

    assert loaded.frame.tolist() == cache.frame.tolist()
    assert loaded.xyxy.tolist() == cache.xyxy.tolist()
    assert loaded.frame_rate == cache.frame_rate
    assert loaded.length == cache.length
    # Embeddings are unit vectors, so half precision is close enough to be a cosine.
    assert np.allclose(loaded.embedding.astype(np.float32), cache.embedding, atol=1e-3)


def test_per_frame_covers_every_frame_including_empty_ones():
    """A tracker that never hears about an empty frame never ages its tracks."""
    cache = make_cache(length=4)
    cache.frame = cache.frame.copy()
    keep = cache.frame != 3
    cache.frame, cache.xyxy = cache.frame[keep], cache.xyxy[keep]
    cache.score, cache.embedding = cache.score[keep], cache.embedding[keep]

    frames = cache.per_frame()
    assert [f for f, _, _, _ in frames] == [1, 2, 3, 4]
    assert len(frames[2][1]) == 0
    assert all(len(f[1]) == 2 for f in (frames[0], frames[1], frames[3]))


def test_per_frame_keeps_boxes_with_their_own_embeddings():
    cache = make_cache(length=2, dim=8)
    first = cache.per_frame()[0]
    assert first[3].shape == (2, 8)
    assert first[3][0][0] == pytest.approx(1.0)
    assert first[3][1][1] == pytest.approx(1.0)


# -------------------------------------------------------------------------- replay


def test_replay_writes_mot_columns():
    from modern_yolonas.tracking import DeepHMSort

    rows = replay(make_cache(), DeepHMSort())
    assert rows.shape[1] == 10
    assert set(np.unique(rows[:, 0])).issubset({1.0, 2.0, 3.0, 4.0})
    assert (rows[:, 7:] == -1).all()
    # x, y, w, h — not x1, y1, x2, y2.
    assert (rows[:, 4] > 0).all() and (rows[:, 5] > 0).all()


def test_replay_takes_the_frame_rate_from_the_cache():
    from modern_yolonas.tracking import DeepHMSort

    tracker = DeepHMSort()
    assert tracker.max_lost_frames == 60
    replay(make_cache(), tracker)
    assert tracker.frame_rate == 25.0
    assert tracker.max_lost_frames == 50


def test_replay_resets_between_sequences():
    """Two sequences replayed through one tracker must not share an id namespace."""
    from modern_yolonas.tracking import DeepHMSort

    tracker = DeepHMSort()
    first = replay(make_cache(), tracker)
    second = replay(make_cache(), tracker)
    assert sorted(np.unique(first[:, 1])) == sorted(np.unique(second[:, 1]))


def test_replay_can_withhold_the_embeddings():
    from modern_yolonas.tracking import DeepHMSort

    cache = make_cache()
    with_appearance = replay(cache, DeepHMSort(), use_embeddings=True)
    motion_only = replay(cache, DeepHMSort(), use_embeddings=False)
    # Two well-separated objects: both settle it, so the point is that it runs and
    # produces the same tracks, not that the numbers differ.
    assert len(with_appearance) == len(motion_only)


def test_write_mot_file_emits_nothing_for_an_empty_result(tmp_path):
    write_mot_file(tmp_path / "seq.txt", np.zeros((0, 10)))
    assert (tmp_path / "seq.txt").read_text().strip() == ""


def test_write_mot_file_is_readable_back(tmp_path):
    rows = np.array([[1, 7, 10.5, 20.25, 30, 40, 0.9, -1, -1, -1]])
    write_mot_file(tmp_path / "seq.txt", rows)
    back = np.loadtxt(tmp_path / "seq.txt", delimiter=",", ndmin=2)
    assert back[0, 0] == 1 and back[0, 1] == 7
    assert back[0, 2] == pytest.approx(10.5)


# ---------------------------------------------------------------------- the sweep


def test_every_sweep_entry_builds_a_tracker():
    from modern_yolonas.benchmarks.mot import build_tracker

    has_trackers = importlib.util.find_spec("trackers") is not None
    for name, settings in SWEEP.items():
        if settings.get("tracker") and not has_trackers:
            with pytest.raises(ImportError, match="tracking"):
                build_tracker(name)
            continue
        tracker, use_embeddings = build_tracker(name)
        assert hasattr(tracker, "update_with_detections"), name
        assert isinstance(use_embeddings, bool), name


def test_the_baselines_are_motion_only():
    assert SWEEP["bytetrack"]["use_embeddings"] is False
    assert SWEEP["ocsort"]["use_embeddings"] is False
    assert SWEEP["motion-matched"]["track_low_threshold"] == 0.1


def test_the_sweep_covers_all_three_questions():
    assert {"harmonic", "min"} <= set(SWEEP)
    assert SWEEP["motion"]["use_embeddings"] is False
    assert SWEEP["harmonic-keepall"]["max_lost_seconds"] is None


# ------------------------------------------------------------------- end-to-end


@pytest.fixture
def trackeval():
    return pytest.importorskip(
        "trackeval", reason="TrackEval is the optional `mot` extra: uv sync --extra mot"
    )


def test_ground_truth_replayed_as_a_result_scores_a_perfect_hundred(tmp_path, trackeval):
    """The instrument check. If this is not 100, no number below it means anything."""
    make_sequence(tmp_path / "data", "seq", two_object_gt(length=6), length=6)
    sequences = discover_sequences(tmp_path / "data")

    gt = np.loadtxt(sequences[0].gt_path, delimiter=",", ndmin=2)
    perfect = np.column_stack([gt[:, :6], np.ones(len(gt)), np.full((len(gt), 3), -1.0)])

    prepare_trackeval_layout(sequences, {"seq": perfect}, tmp_path / "work", "unit", "test")
    scores = evaluate(tmp_path / "work", "unit", "test")

    assert scores["HOTA"] == pytest.approx(100.0)
    assert scores["MOTA"] == pytest.approx(100.0)
    assert scores["IDSW"] == 0


def test_a_swapped_id_is_punished(tmp_path, trackeval):
    make_sequence(tmp_path / "data", "seq", two_object_gt(length=6), length=6)
    sequences = discover_sequences(tmp_path / "data")

    gt = np.loadtxt(sequences[0].gt_path, delimiter=",", ndmin=2)
    swapped = np.column_stack([gt[:, :6], np.ones(len(gt)), np.full((len(gt), 3), -1.0)])
    # Halfway through, the two objects exchange ids.
    late = swapped[:, 0] > 3
    swapped[late, 1] = 3 - swapped[late, 1]

    prepare_trackeval_layout(sequences, {"seq": swapped}, tmp_path / "work", "unit", "test")
    scores = evaluate(tmp_path / "work", "unit", "test")

    assert scores["HOTA"] < 100.0
    assert scores["IDSW"] == 2
    # Detection is still perfect; only the association went wrong.
    assert scores["DetA"] == pytest.approx(100.0)


def test_the_layout_links_the_ground_truth_rather_than_copying_it(tmp_path, trackeval):
    """These datasets are not redistributable; a copy is the accident to avoid."""
    make_sequence(tmp_path / "data", "seq", two_object_gt(), length=4)
    sequences = discover_sequences(tmp_path / "data")

    work = prepare_trackeval_layout(
        sequences, {"seq": np.zeros((0, 10))}, tmp_path / "work", "unit", "test"
    )
    linked = work / "gt" / "unit-test" / "seq" / "gt" / "gt.txt"
    assert linked.is_symlink()
    assert linked.resolve() == sequences[0].gt_path.resolve()

    seqmap = (work / "gt" / "seqmaps" / "unit-test.txt").read_text()
    assert seqmap.splitlines() == ["name", "seq"]


def test_a_missing_tracking_extra_names_the_install_command(monkeypatch):
    import sys

    from modern_yolonas.tracking import external

    monkeypatch.setitem(sys.modules, "trackers", None)
    with pytest.raises(ImportError, match=r"modern-yolonas\[tracking\]"):
        external.ByteTrack()
