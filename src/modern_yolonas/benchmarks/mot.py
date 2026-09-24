"""Multi-object tracking benchmarks in the MOT Challenge format.

Measures :class:`~modern_yolonas.tracking.DeepHMSort` on SportsMOT and MOT17, which
is what turns the tracking guide's "plausible, not measured" into a number.

The work is split in two so an ablation is cheap and, more importantly, *fair*:

1. :func:`build_detector_cache` (or :func:`build_oracle_cache`) runs over each
   sequence once and writes the boxes **and their embeddings** to disk.
2. :func:`replay` runs a tracker configuration over that cache.

Every configuration therefore sees byte-identical detections, so a difference in the
metrics can only come from the association. Re-running the detector per configuration
would leave a much less certain claim, and take hours instead of seconds.

Two detection sources, and the distinction matters when reading the results:

``detector``
    What a user actually gets: YOLO-NAS COCO weights, person class. On SportsMOT this
    also boxes referees, coaches and the crowd, none of which are in the ground truth,
    so the false-positive rate is high and the absolute scores are low.

``oracle``
    Ground-truth boxes fed in as detections, embedded with YOLO-NAS. Detection is then
    perfect by construction and the only remaining error is association — which is the
    thing Deep HM-SORT changes. This is the experiment that can answer whether the
    harmonic mean and the appearance cue help; the other one answers what you get out
    of the box.
"""

from __future__ import annotations

import configparser

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Columns of a MOT Challenge ``gt.txt`` / result file.
#: ``frame, id, x, y, w, h, conf, class, visibility``.
MOT_COLUMNS = 9


@dataclass(frozen=True)
class Sequence:
    """One video in a MOT-format dataset.

    Attributes:
        name: Directory name, which is also the sequence id in the seqmap.
        root: The sequence directory, holding ``img1/``, ``gt/`` and ``seqinfo.ini``.
        frame_rate: From ``seqinfo.ini``. Handed to the tracker so its memory, which
            is written in seconds, means the same span on a 25 fps clip and a 30 fps
            one.
        length: Number of frames, from ``seqinfo.ini``.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    name: str
    root: Path
    frame_rate: float
    length: int
    width: int
    height: int

    @property
    def image_dir(self) -> Path:
        return self.root / "img1"

    @property
    def gt_path(self) -> Path:
        return self.root / "gt" / "gt.txt"

    def frame_path(self, frame: int) -> Path:
        """Path of a 1-indexed frame, the way MOT numbers them."""
        return self.image_dir / f"{frame:06d}.jpg"


def discover_sequences(split_root: str | Path, names: list[str] | None = None) -> list[Sequence]:
    """Read every sequence under a MOT split directory.

    Args:
        split_root: Directory holding one subdirectory per sequence — SportsMOT's
            ``dataset/val`` or MOT17's ``train``.
        names: Restrict to these sequence names, in this order. ``None`` takes every
            subdirectory with a ``seqinfo.ini``, sorted.

    Returns:
        One :class:`Sequence` per video.
    """
    split_root = Path(split_root)
    if not split_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {split_root}")

    found = {}
    for directory in sorted(split_root.iterdir()):
        info = directory / "seqinfo.ini"
        if not info.is_file():
            continue
        parser = configparser.ConfigParser()
        parser.read(info)
        section = parser["Sequence"]
        found[directory.name] = Sequence(
            name=directory.name,
            root=directory,
            frame_rate=float(section.get("frameRate", 30)),
            length=int(section.get("seqLength", 0)),
            width=int(section.get("imWidth", 0)),
            height=int(section.get("imHeight", 0)),
        )

    if names is None:
        return list(found.values())

    missing = [name for name in names if name not in found]
    if missing:
        raise FileNotFoundError(f"{len(missing)} sequence(s) not under {split_root}: {missing[:5]}")
    return [found[name] for name in names]


def read_split_file(path: str | Path) -> list[str]:
    """Sequence names from a one-per-line split list, ignoring a header and blanks."""
    lines = [line.strip() for line in Path(path).read_text().splitlines()]
    return [line for line in lines if line and line.lower() != "name"]


# --------------------------------------------------------------------------- cache


@dataclass
class DetectionCache:
    """Per-frame boxes and embeddings for one sequence.

    Flat arrays rather than a list per frame, because that is what ``npz`` stores
    without pickling and what slicing a frame out of stays cheap on.

    Attributes:
        frame: ``(N,)`` 1-indexed frame number of each detection.
        xyxy: ``(N, 4)`` boxes in image pixels.
        score: ``(N,)`` confidence.
        embedding: ``(N, D)`` appearance vectors, float16 on disk — these are
            L2-normalized unit vectors, so half precision costs about 5e-4 of cosine
            and a third of the disk.
        frame_rate: Of the source video.
        length: Frames in the sequence, including any with no detections.
    """

    frame: np.ndarray
    xyxy: np.ndarray
    score: np.ndarray
    embedding: np.ndarray
    frame_rate: float
    length: int

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frame=self.frame.astype(np.int32),
            xyxy=self.xyxy.astype(np.float32),
            score=self.score.astype(np.float32),
            embedding=self.embedding.astype(np.float16),
            frame_rate=np.float32(self.frame_rate),
            length=np.int32(self.length),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DetectionCache":
        with np.load(path) as data:
            return cls(
                frame=data["frame"],
                xyxy=data["xyxy"],
                score=data["score"],
                embedding=data["embedding"],
                frame_rate=float(data["frame_rate"]),
                length=int(data["length"]),
            )

    def per_frame(self) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        """Split into ``(frame, xyxy, score, embedding)``, one entry per frame.

        Every frame of the sequence appears, including those with no detections —
        a tracker that never hears about an empty frame never ages its tracks, and
        the metrics would quietly improve for the wrong reason.
        """
        order = np.argsort(self.frame, kind="stable")
        frame = self.frame[order]
        boundaries = np.searchsorted(frame, np.arange(1, self.length + 2))

        out = []
        for index in range(self.length):
            start, stop = boundaries[index], boundaries[index + 1]
            rows = order[start:stop]
            out.append(
                (index + 1, self.xyxy[rows], self.score[rows], self.embedding[rows].astype(np.float32))
            )
        return out


def read_gt_boxes(sequence: Sequence) -> dict[int, np.ndarray]:
    """Ground-truth boxes per frame, as ``xyxy``.

    Only rows the benchmark actually evaluates: ``conf`` (column 7) of zero marks a
    box to be ignored, and on MOT17 anything outside the pedestrian class is a
    distractor that TrackEval handles during its own preprocessing. Feeding those in
    as oracle detections would be handing the tracker boxes that the evaluator is
    about to say do not exist.

    Args:
        sequence: The sequence to read.

    Returns:
        ``{frame: (K, 4)}``, 1-indexed frames, frames with no boxes omitted.
    """
    rows = np.loadtxt(sequence.gt_path, delimiter=",", ndmin=2)
    if rows.size == 0:
        return {}

    keep = rows[:, 6] > 0
    if rows.shape[1] > 7:
        keep &= rows[:, 7] == 1
    rows = rows[keep]

    boxes: dict[int, np.ndarray] = {}
    for frame in np.unique(rows[:, 0]).astype(int):
        block = rows[rows[:, 0] == frame]
        xywh = block[:, 2:6]
        boxes[int(frame)] = np.column_stack(
            [xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]]
        ).astype(np.float32)
    return boxes


def build_detector_cache(
    sequence: Sequence,
    detector,
    conf_threshold: float = 0.1,
    class_id: int = 0,
) -> DetectionCache:
    """Detect and embed every frame of one sequence, in one forward pass per frame.

    Args:
        sequence: What to run over.
        detector: A :class:`~modern_yolonas.inference.detect.YoloNASDetector`.
        conf_threshold: Deliberately low — the tracker's own ``track_low_threshold``
            filters on replay, so caching low leaves the score thresholds free to
            sweep without detecting again.
        class_id: Which class to keep; 0 is COCO's person.

    Returns:
        A :class:`DetectionCache`.
    """

    from modern_yolonas.inference.embed import Task

    tasks = Task.DETECT | Task.EMBED_OBJECTS
    collected = []

    for index in range(1, sequence.length + 1):
        image = _read_frame(sequence, index)
        detections = detector.predict(image, tasks, conf_threshold=conf_threshold).detections
        if detections.class_id is not None and len(detections):
            detections = detections[detections.class_id == class_id]
        if len(detections) == 0:
            continue
        collected.append(
            (
                index,
                detections.xyxy.astype(np.float32),
                detections.confidence.astype(np.float32),
                np.asarray(detections.data["embedding"], dtype=np.float32),
            )
        )

    return _assemble(collected, sequence, detector.embedding_dim)


def build_oracle_cache(sequence: Sequence, embedder) -> DetectionCache:
    """Embed the ground-truth boxes, so association is the only thing left to get wrong.

    No detector runs. Detection is perfect by construction, which is the point: what
    remains is the part Deep HM-SORT actually changes.

    Args:
        sequence: What to run over.
        embedder: A :class:`~modern_yolonas.inference.embed.YoloNASEmbedder`. Its
            ``embed_boxes`` clips to the frame, which matters here — ground-truth
            boxes on SportsMOT run off the edge during a camera pan.

    Returns:
        A :class:`DetectionCache` whose scores are all 1.
    """
    boxes_by_frame = read_gt_boxes(sequence)
    collected = []

    for index in sorted(boxes_by_frame):
        xyxy = boxes_by_frame[index]
        if len(xyxy) == 0:
            continue
        image = _read_frame(sequence, index)
        collected.append(
            (
                index,
                xyxy,
                np.ones(len(xyxy), dtype=np.float32),
                np.asarray(embedder.embed_boxes(image, xyxy), dtype=np.float32),
            )
        )

    return _assemble(collected, sequence, embedder.embedding_dim)


def _read_frame(sequence: Sequence, index: int) -> np.ndarray:
    import cv2

    path = sequence.frame_path(index)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read frame: {path}")
    return image


def _assemble(collected, sequence: Sequence, width: int) -> DetectionCache:
    """Flatten per-frame arrays into one cache."""
    if not collected:
        return DetectionCache(
            frame=np.zeros(0, dtype=np.int32),
            xyxy=np.zeros((0, 4), dtype=np.float32),
            score=np.zeros(0, dtype=np.float32),
            embedding=np.zeros((0, width), dtype=np.float32),
            frame_rate=sequence.frame_rate,
            length=sequence.length,
        )

    return DetectionCache(
        frame=np.concatenate([np.full(len(box), index, dtype=np.int32) for index, box, _, _ in collected]),
        xyxy=np.concatenate([box for _, box, _, _ in collected]),
        score=np.concatenate([score for _, _, score, _ in collected]),
        embedding=np.concatenate([vector for _, _, _, vector in collected]),
        frame_rate=sequence.frame_rate,
        length=sequence.length,
    )


#: Tracker configurations the benchmark compares, keyed by the name the CLI takes.
#:
#: Each entry is keyword arguments for :class:`~modern_yolonas.tracking.DeepHMSort`,
#: plus ``use_embeddings`` which is handled by :func:`replay` rather than the tracker.
#: Three questions, and nothing else, so the table stays readable:
#:
#: * ``harmonic`` against ``min`` — Deep HM-SORT's first contribution, the fusion.
#: * either against ``motion`` — whether the appearance cue earns its keep *at all*
#:   with YOLO-NAS features, which is the claim the tracking guide would not make.
#: * ``*-keepall`` against the rest — Deep HM-SORT's second contribution, and the
#:   default this project changed to two seconds.
SWEEP: dict[str, dict] = {
    "harmonic": {"fusion": "harmonic"},
    "min": {"fusion": "min"},
    "motion": {"use_embeddings": False},
    "harmonic-keepall": {"fusion": "harmonic", "max_lost_seconds": None},
    "motion-keepall": {"use_embeddings": False, "max_lost_seconds": None},
}


# -------------------------------------------------------------------------- replay


def replay(cache: DetectionCache, tracker, use_embeddings: bool = True) -> np.ndarray:
    """Run one tracker configuration over a cached sequence.

    Args:
        cache: From :func:`build_detector_cache` or :func:`build_oracle_cache`.
        tracker: A :class:`~modern_yolonas.tracking.DeepHMSort`. It is reset first
            and its ``frame_rate`` is taken from the cache, so the caller can reuse
            one instance across sequences without ids leaking between them.
        use_embeddings: ``False`` withholds the appearance vectors, which is the
            motion-only control.

    Returns:
        ``(N, 10)`` MOT rows: ``frame, id, x, y, w, h, conf, -1, -1, -1``.
    """
    import supervision as sv

    tracker.reset()
    tracker.frame_rate = cache.frame_rate

    rows = []
    for frame, xyxy, score, embedding in cache.per_frame():
        if len(xyxy):
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=score,
                class_id=np.zeros(len(xyxy), dtype=int),
            )
            if use_embeddings:
                detections.data["embedding"] = embedding
        else:
            detections = sv.Detections.empty()

        tracked = tracker.update_with_detections(detections)
        if len(tracked) == 0:
            continue

        box = tracked.xyxy
        rows.append(
            np.column_stack(
                [
                    np.full(len(tracked), frame, dtype=np.float64),
                    tracked.tracker_id.astype(np.float64),
                    box[:, 0],
                    box[:, 1],
                    box[:, 2] - box[:, 0],
                    box[:, 3] - box[:, 1],
                    tracked.confidence if tracked.confidence is not None else np.ones(len(tracked)),
                    np.full(len(tracked), -1.0),
                    np.full(len(tracked), -1.0),
                    np.full(len(tracked), -1.0),
                ]
            )
        )

    return np.concatenate(rows) if rows else np.zeros((0, 10))


def write_mot_file(path: str | Path, rows: np.ndarray) -> None:
    """Write tracker output in the MOT Challenge text format.

    Frames with no tracked object contribute no line at all, which is what the
    format means by an empty frame — a placeholder row would be scored as a false
    positive against nothing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d,%d,%d")


# ---------------------------------------------------------------------- evaluation


def prepare_trackeval_layout(
    sequences: list[Sequence],
    results: dict[str, np.ndarray],
    work_dir: str | Path,
    benchmark: str,
    split: str,
    tracker_name: str = "modern-yolonas",
) -> Path:
    """Lay out ground truth and results the way TrackEval insists on finding them.

    ``{work}/gt/{benchmark}-{split}/{seq}/gt/gt.txt`` plus ``seqinfo.ini``, a seqmap
    at ``{work}/gt/seqmaps/{benchmark}-{split}.txt``, and results under
    ``{work}/trackers/{benchmark}-{split}/{tracker}/data/{seq}.txt``. The ground
    truth is symlinked rather than copied — these datasets are non-redistributable
    and a copy inside the repo tree is exactly the accident to avoid.

    Args:
        sequences: The evaluated sequences, in seqmap order.
        results: ``{sequence name: rows}`` from :func:`replay`.
        work_dir: Where to build the tree.
        benchmark: TrackEval's ``BENCHMARK``, e.g. ``"sportsmot"``.
        split: TrackEval's ``SPLIT_TO_EVAL``, e.g. ``"val"``.
        tracker_name: Subdirectory name for this run's results.

    Returns:
        The work directory.
    """
    work_dir = Path(work_dir)
    gt_root = work_dir / "gt" / f"{benchmark}-{split}"
    seqmap_dir = work_dir / "gt" / "seqmaps"
    tracker_dir = work_dir / "trackers" / f"{benchmark}-{split}" / tracker_name / "data"

    gt_root.mkdir(parents=True, exist_ok=True)
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    tracker_dir.mkdir(parents=True, exist_ok=True)

    for sequence in sequences:
        target = gt_root / sequence.name
        (target / "gt").mkdir(parents=True, exist_ok=True)
        _link(sequence.gt_path, target / "gt" / "gt.txt")
        _link(sequence.root / "seqinfo.ini", target / "seqinfo.ini")
        write_mot_file(tracker_dir / f"{sequence.name}.txt", results[sequence.name])

    seqmap = seqmap_dir / f"{benchmark}-{split}.txt"
    seqmap.write_text("name\n" + "\n".join(s.name for s in sequences) + "\n")
    return work_dir


def _link(source: Path, target: Path) -> None:
    """Symlink ``source`` at ``target``, replacing whatever was there."""
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source.resolve())


def evaluate(
    work_dir: str | Path,
    benchmark: str,
    split: str,
    tracker_name: str = "modern-yolonas",
    do_preproc: bool = False,
) -> dict[str, float]:
    """Run TrackEval over a prepared layout and return the headline metrics.

    Args:
        work_dir: From :func:`prepare_trackeval_layout`.
        benchmark: TrackEval's ``BENCHMARK``.
        split: TrackEval's ``SPLIT_TO_EVAL``.
        tracker_name: Which results directory to score.
        do_preproc: TrackEval's MOT17 preprocessing — it drops ground truth marked
            to be ignored and forgives tracker boxes that land on a distractor class.
            MOT17 needs it. SportsMOT annotates only players, all with ``conf`` 1 and
            one class, so it has nothing to preprocess; it is off by default and
            :func:`preproc_changes_nothing` checks that claim rather than trusting it.

    Returns:
        ``{metric: value}`` averaged over sequences — HOTA, DetA, AssA, MOTA, IDF1
        and the raw ID-switch count.
    """
    import trackeval

    work_dir = Path(work_dir)

    eval_config = {
        **trackeval.Evaluator.get_default_eval_config(),
        "USE_PARALLEL": False,
        "PRINT_RESULTS": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_EMPTY_CLASSES": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
    }
    dataset_config = {
        **trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(),
        "GT_FOLDER": str(work_dir / "gt"),
        "TRACKERS_FOLDER": str(work_dir / "trackers"),
        "BENCHMARK": benchmark,
        "SPLIT_TO_EVAL": split,
        "TRACKERS_TO_EVAL": [tracker_name],
        "DO_PREPROC": do_preproc,
        "SEQMAP_FILE": str(work_dir / "gt" / "seqmaps" / f"{benchmark}-{split}.txt"),
        "PRINT_CONFIG": False,
        "SKIP_SPLIT_FOL": False,
    }

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [
        trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
        trackeval.metrics.CLEAR({"PRINT_CONFIG": False, "THRESHOLD": 0.5}),
        trackeval.metrics.Identity({"PRINT_CONFIG": False, "THRESHOLD": 0.5}),
    ]

    output, _ = evaluator.evaluate([dataset], metrics)
    combined = output["MotChallenge2DBox"][tracker_name]["COMBINED_SEQ"]["pedestrian"]

    return {
        "HOTA": float(np.mean(combined["HOTA"]["HOTA"])) * 100,
        "DetA": float(np.mean(combined["HOTA"]["DetA"])) * 100,
        "AssA": float(np.mean(combined["HOTA"]["AssA"])) * 100,
        "MOTA": float(combined["CLEAR"]["MOTA"]) * 100,
        "IDF1": float(combined["Identity"]["IDF1"]) * 100,
        "IDSW": float(combined["CLEAR"]["IDSW"]),
    }


def preproc_changes_nothing(work_dir: str | Path, benchmark: str, split: str, tracker_name: str) -> bool:
    """Whether TrackEval's preprocessing alters the result on this dataset.

    SportsMOT's ground truth has nothing for preprocessing to act on — every row is
    ``conf 1, class 1, visibility 1``. That is a claim about the data, so it is
    checked rather than asserted: the benchmark runs both ways once and stops if they
    disagree.
    """
    on = evaluate(work_dir, benchmark, split, tracker_name, do_preproc=True)
    off = evaluate(work_dir, benchmark, split, tracker_name, do_preproc=False)
    return all(abs(on[key] - off[key]) < 1e-6 for key in on)
