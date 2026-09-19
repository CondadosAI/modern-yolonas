"""CLI: yolonas track"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


class ModelName(str, Enum):
    yolo_nas_s = "yolo_nas_s"
    yolo_nas_m = "yolo_nas_m"
    yolo_nas_l = "yolo_nas_l"


class Fusion(str, Enum):
    harmonic = "harmonic"
    min = "min"


def track(
    source: Annotated[str, typer.Option(help="Video file path.")],
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    weights: Annotated[str | None, typer.Option(help="Path to a custom checkpoint (.pt). When set, --model selects the architecture.")] = None,
    num_classes: Annotated[int, typer.Option(help="Number of classes in the custom checkpoint (ignored with pretrained weights).")] = 80,
    conf: Annotated[float | None, typer.Option(help="Detection confidence threshold. Defaults to --track-low, so the low-score association round has detections to work with.")] = None,
    iou: Annotated[float, typer.Option(help="NMS IoU threshold.")] = 0.7,
    device: Annotated[str, typer.Option(help="Device (cuda or cpu).")] = "cuda",
    output: Annotated[str, typer.Option(help="Output directory.")] = "results",
    input_size: Annotated[int, typer.Option(help="Model input size.")] = 640,
    codec: Annotated[str, typer.Option(help="Video output codec (e.g. mp4v, XVID, avc1).")] = "mp4v",
    classes: Annotated[str | None, typer.Option(help="Comma-separated class ids to track, e.g. '0' for person. Default: every class.")] = None,
    appearance: Annotated[bool, typer.Option(help="Associate on per-object embeddings as well as motion.")] = True,
    fusion: Annotated[Fusion, typer.Option(help="How motion and appearance costs combine. 'harmonic' is Deep HM-SORT; 'min' is Deep-EIoU's original, for comparison.")] = Fusion.harmonic,
    track_high: Annotated[float, typer.Option(help="Score at or above which a detection enters the first association round.")] = 0.6,
    track_low: Annotated[float, typer.Option(help="Score below which a detection is ignored entirely.")] = 0.4,
    new_track: Annotated[float, typer.Option(help="Lowest score an unmatched detection may have and still start a track.")] = 0.5,
    expansion: Annotated[float, typer.Option(help="Box growth for the first association round.")] = 0.3,
    max_lost: Annotated[int | None, typer.Option(help="Frames a track may go unmatched before it is dropped. Unset keeps every tracklet for the whole video, which is the paper's default and a closed-environment assumption.")] = None,
    class_aware: Annotated[bool, typer.Option(help="Refuse to associate a detection with a track of a different class.")] = False,
    show_fps: Annotated[bool, typer.Option(help="Burn the per-frame time and frame rate into the output.")] = False,
):
    """Track objects across a video with Deep HM-SORT."""
    from rich.console import Console

    from modern_yolonas.inference.detect import YoloNASDetector
    from modern_yolonas.tracking import DeepHMSort

    console = Console()

    source_path = Path(source)
    if source_path.suffix.lower() not in VIDEO_EXTENSIONS:
        console.print(f"[red]Tracking needs a video; got {source_path.suffix or 'no extension'}[/red]")
        raise typer.Abort()

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / source_path.name

    if weights:
        console.print(f"Loading {model.value} from checkpoint {weights!r} ({num_classes} classes)...")
    else:
        console.print(f"Loading pretrained {model.value}...")

    detector = YoloNASDetector(
        model.value,
        device=device,
        conf_threshold=conf if conf is not None else track_low,
        iou_threshold=iou,
        input_size=input_size,
        weights=weights,
        num_classes=num_classes,
    )

    tracker = DeepHMSort(
        track_high_threshold=track_high,
        track_low_threshold=track_low,
        new_track_threshold=new_track,
        expansion=expansion,
        fusion=fusion.value,
        max_lost=max_lost,
        class_aware=class_aware,
    )

    keep = {int(c) for c in classes.split(",") if c.strip()} if classes else None

    console.print(
        f"Tracking {source_path.name} — {fusion.value} fusion, "
        f"{'with' if appearance else 'without'} appearance, "
        f"{'all tracklets kept' if max_lost is None else f'{max_lost}-frame memory'}..."
    )

    stats = _track_video(detector, tracker, source_path, out_path, codec, appearance, keep, show_fps)

    console.print(
        f"  {stats['total_frames']} frames, "
        f"{stats['total_detections']} tracked detections, "
        f"{stats['unique_ids']} distinct ids"
    )
    console.print(f"[green]Done! Video saved to {out_path}[/green]")


def _track_video(detector, tracker, source_path, out_path, codec, appearance, keep, show_fps):
    """Track to file, optionally restricted to a set of classes.

    Filtering happens *before* the tracker rather than after: an id spent on a
    class nobody asked about is an id that competes for associations.
    """
    if keep is None:
        return detector.track_video_to_file(
            source=str(source_path),
            output=str(out_path),
            tracker=tracker,
            appearance=appearance,
            codec=codec,
            show_fps=show_fps,
        )

    import cv2
    import numpy as np

    from modern_yolonas.inference.embed import Task

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, size)
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to create video writer for {out_path} with codec {codec!r}")

    tasks = Task.DETECT | Task.EMBED_OBJECTS if appearance else Task.DETECT
    seen: set[int] = set()
    frames = total = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections = detector.predict(frame, tasks).detections
            if detections.class_id is not None and len(detections):
                detections = detections[np.isin(detections.class_id, list(keep))]

            tracked = tracker.update_with_detections(detections)
            writer.write(detector.annotate_tracks(frame, tracked, show_fps=show_fps))

            frames += 1
            total += len(tracked)
            if tracked.tracker_id is not None:
                seen.update(int(i) for i in tracked.tracker_id)
    finally:
        capture.release()
        writer.release()

    return {"total_frames": frames, "total_detections": total, "unique_ids": len(seen), "fps": fps}
