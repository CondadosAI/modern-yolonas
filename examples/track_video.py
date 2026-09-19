"""Track objects across a video with Deep HM-SORT, and report what happened.

    uv run examples/track_video.py --source match.mp4 --classes 0

The interesting output is not the annotated video — it is the last two lines. A
tracker that has found 40 distinct people in a clip containing 11 is fragmenting
ids, and the fix is a wider expansion or (if you set one) a longer memory. Running
it twice, once with ``--fusion min``, is the paper's ablation on your own footage.
"""

from __future__ import annotations

import argparse

from collections import Counter
from pathlib import Path

from modern_yolonas import YoloNASDetector
from modern_yolonas.tracking import DeepHMSort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video file path.")
    parser.add_argument("--output", default="results/tracked.mp4", help="Annotated video to write.")
    parser.add_argument("--model", default="yolo_nas_s", choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classes", default=None, help="Comma-separated class ids, e.g. '0' for person.")
    parser.add_argument("--fusion", default="harmonic", choices=["harmonic", "min"],
                        help="'harmonic' is Deep HM-SORT; 'min' is Deep-EIoU's original.")
    parser.add_argument("--max-lost", type=int, default=None,
                        help="Frames a track may go unmatched. Unset keeps every tracklet.")
    parser.add_argument("--no-appearance", action="store_true", help="Associate on motion alone.")
    parser.add_argument("--show-fps", action="store_true", help="Burn the per-frame time into the output.")
    parser.add_argument("--text-scale", type=float, default=None,
                        help="Shrink the id labels. Worth setting when a frame carries more than "
                             "a handful of boxes and the labels start covering the objects.")
    args = parser.parse_args()

    keep = {int(c) for c in args.classes.split(",")} if args.classes else None

    detector = YoloNASDetector(args.model, device=args.device)
    tracker = DeepHMSort(fusion=args.fusion, max_lost=args.max_lost)

    if args.text_scale is not None:
        import supervision as sv

        detector._track_label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.TRACK,
            text_scale=args.text_scale,
            text_padding=3,
            text_thickness=1,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np

    from modern_yolonas import Task

    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    tasks = Task.DETECT if args.no_appearance else Task.DETECT | Task.EMBED_OBJECTS
    lifetimes: Counter[int] = Counter()
    frames = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections = detector.predict(frame, tasks, conf_threshold=tracker.track_low_threshold).detections
            if keep is not None and len(detections):
                detections = detections[np.isin(detections.class_id, list(keep))]

            tracked = tracker.update_with_detections(detections)
            writer.write(detector.annotate_tracks(frame, tracked, show_fps=args.show_fps))

            frames += 1
            if tracked.tracker_id is not None:
                lifetimes.update(int(i) for i in tracked.tracker_id)
    finally:
        capture.release()
        writer.release()

    # A healthy run has a few long-lived ids. A long tail of ids seen for one or two
    # frames each is the signature of fragmentation, not of a crowded scene.
    fleeting = sum(1 for count in lifetimes.values() if count <= 2)
    print(f"\n{frames} frames -> {args.output}")
    print(f"{len(lifetimes)} distinct ids, {fleeting} of them seen in 2 frames or fewer")
    if lifetimes:
        longest = lifetimes.most_common(5)
        print("longest-lived:", ", ".join(f"#{i} ({n} frames)" for i, n in longest))


if __name__ == "__main__":
    main()
