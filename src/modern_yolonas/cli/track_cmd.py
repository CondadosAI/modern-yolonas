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


class TrackerName(str, Enum):
    bytetrack = "bytetrack"
    deep_hm_sort = "deep-hm-sort"


#: Deep HM-SORT's own options, with their defaults. They default to ``None`` on the
#: command line so that passing one alongside ``--tracker bytetrack`` can be refused
#: rather than silently ignored.
DEEP_HM_SORT_DEFAULTS = {
    "appearance": True,
    "fusion": Fusion.harmonic,
    "track_high": 0.6,
    "track_low": 0.4,
    "new_track": 0.5,
    "expansion": 0.3,
    "max_lost_seconds": 2.0,
    "keep_all_tracks": False,
    "class_aware": False,
}


def track(
    source: Annotated[str, typer.Option(help="Video file path.")],
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    weights: Annotated[str | None, typer.Option(help="Path to a custom checkpoint (.pt). When set, --model selects the architecture.")] = None,
    num_classes: Annotated[int, typer.Option(help="Number of classes in the custom checkpoint (ignored with pretrained weights).")] = 80,
    tracker_name: Annotated[TrackerName, typer.Option("--tracker", help="ByteTrack (the default; needs the `tracking` extra) or Deep HM-SORT. See docs/benchmarks/tracking.md.")] = TrackerName.bytetrack,
    conf: Annotated[float | None, typer.Option(help="Detection confidence threshold. Defaults to the tracker's floor: 0.1 for ByteTrack, --track-low for Deep HM-SORT.")] = None,
    iou: Annotated[float, typer.Option(help="NMS IoU threshold.")] = 0.7,
    device: Annotated[str, typer.Option(help="Device (cuda or cpu).")] = "cuda",
    output: Annotated[str, typer.Option(help="Output directory.")] = "results",
    input_size: Annotated[int, typer.Option(help="Model input size.")] = 640,
    codec: Annotated[str, typer.Option(help="Video output codec (e.g. mp4v, XVID, avc1).")] = "mp4v",
    classes: Annotated[str | None, typer.Option(help="Comma-separated class ids to track, e.g. '0' for person. Default: every class. ByteTrack is not class-aware, so filter here when classes should not share ids.")] = None,
    appearance: Annotated[bool | None, typer.Option(help="[deep-hm-sort] Associate on per-object embeddings as well as motion. Default: on.")] = None,
    fusion: Annotated[Fusion | None, typer.Option(help="[deep-hm-sort] How motion and appearance costs combine. 'harmonic' is Deep HM-SORT; 'min' is Deep-EIoU's original. Default: harmonic.")] = None,
    track_high: Annotated[float | None, typer.Option(help="[deep-hm-sort] Score at or above which a detection enters the first association round. Default: 0.6.")] = None,
    track_low: Annotated[float | None, typer.Option(help="[deep-hm-sort] Score below which a detection is ignored entirely. Default: 0.4.")] = None,
    new_track: Annotated[float | None, typer.Option(help="[deep-hm-sort] Lowest score an unmatched detection may have and still start a track. Default: 0.5.")] = None,
    expansion: Annotated[float | None, typer.Option(help="[deep-hm-sort] Box growth for the first association round. Default: 0.3.")] = None,
    max_lost_seconds: Annotated[float | None, typer.Option(help="[deep-hm-sort] How long a track may go unmatched before it is dropped, in seconds of video. Default: 2.")] = None,
    keep_all_tracks: Annotated[bool | None, typer.Option(help="[deep-hm-sort] Never drop a track — the paper's setting. Right for a fixed camera on a closed scene.")] = None,
    class_aware: Annotated[bool | None, typer.Option(help="[deep-hm-sort] Refuse to associate a detection with a track of a different class.")] = None,
    show_fps: Annotated[bool, typer.Option(help="Burn the per-frame time and frame rate into the output.")] = False,
):
    """Track objects across a video with ByteTrack or Deep HM-SORT."""
    from rich.console import Console

    from modern_yolonas.inference.detect import YoloNASDetector

    console = Console()

    given = {
        "appearance": appearance, "fusion": fusion, "track_high": track_high, "track_low": track_low,
        "new_track": new_track, "expansion": expansion, "max_lost_seconds": max_lost_seconds,
        "keep_all_tracks": keep_all_tracks, "class_aware": class_aware,
    }
    if tracker_name is TrackerName.bytetrack:
        stray = sorted("--" + name.replace("_", "-") for name, value in given.items() if value is not None)
        if stray:
            raise typer.BadParameter(
                f"{', '.join(stray)} only apply to --tracker deep-hm-sort; ByteTrack uses its library defaults."
            )
        try:
            from modern_yolonas.tracking.external import ByteTrack

            tracker = ByteTrack()
        except ImportError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        use_appearance = False
        description = "ByteTrack"
    else:
        from modern_yolonas.tracking import DeepHMSort

        opts = {name: DEEP_HM_SORT_DEFAULTS[name] if value is None else value for name, value in given.items()}
        tracker = DeepHMSort(
            track_high_threshold=opts["track_high"],
            track_low_threshold=opts["track_low"],
            new_track_threshold=opts["new_track"],
            expansion=opts["expansion"],
            fusion=opts["fusion"].value,
            max_lost_seconds=None if opts["keep_all_tracks"] else opts["max_lost_seconds"],
            class_aware=opts["class_aware"],
        )
        use_appearance = opts["appearance"]
        memory = "all tracklets kept" if opts["keep_all_tracks"] else f"{opts['max_lost_seconds']:g}s memory"
        description = (
            f"Deep HM-SORT, {opts['fusion'].value} fusion, "
            f"{'with' if use_appearance else 'without'} appearance, {memory}"
        )

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
        conf_threshold=conf if conf is not None else tracker.track_low_threshold,
        iou_threshold=iou,
        input_size=input_size,
        weights=weights,
        num_classes=num_classes,
    )

    keep = {int(c) for c in classes.split(",") if c.strip()} if classes else None

    console.print(f"Tracking {source_path.name} — {description}...")

    stats = _track_video(detector, tracker, source_path, out_path, codec, use_appearance, keep, show_fps)

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

    # `track_video` does this for us; this hand-rolled loop has to do it itself, or
    # the tracker's memory would be measured in 30ths of a second on a clip that is
    # not 30 fps.
    if 1.0 <= fps <= 240.0:
        tracker.frame_rate = fps

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
