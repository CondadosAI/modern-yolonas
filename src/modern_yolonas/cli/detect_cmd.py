"""CLI: yolonas detect"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


class ModelName(str, Enum):
    yolo_nas_s = "yolo_nas_s"
    yolo_nas_m = "yolo_nas_m"
    yolo_nas_l = "yolo_nas_l"


def detect(
    source: Annotated[str, typer.Option(help="Image file, directory, or video path.")],
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    weights: Annotated[str | None, typer.Option(help="Path to a custom checkpoint (.pt) produced by the trainer. When set, --model selects the architecture.")] = None,
    num_classes: Annotated[int, typer.Option(help="Number of classes in the custom checkpoint (ignored when using pretrained weights).")] = 80,
    conf: Annotated[float, typer.Option(help="Confidence threshold.")] = 0.25,
    iou: Annotated[float, typer.Option(help="NMS IoU threshold.")] = 0.7,
    device: Annotated[str, typer.Option(help="Device (cuda or cpu).")] = "cuda",
    output: Annotated[str, typer.Option(help="Output directory.")] = "results",
    input_size: Annotated[int, typer.Option(help="Model input size.")] = 640,
    skip_frames: Annotated[int, typer.Option(help="Process every N-th frame for video (0 = every frame).")] = 0,
    codec: Annotated[str, typer.Option(help="Video output codec (e.g. mp4v, XVID, avc1).")] = "mp4v",
):
    """Run object detection on images or video."""
    from rich.console import Console

    from modern_yolonas.inference.detect import Detector

    console = Console()
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if weights:
        console.print(f"Loading {model.value} from checkpoint {weights!r} ({num_classes} classes)...")
    else:
        console.print(f"Loading pretrained {model.value}...")

    det = Detector(
        model.value,
        device=device,
        conf_threshold=conf,
        iou_threshold=iou,
        input_size=input_size,
        weights=weights,
        num_classes=num_classes,
    )

    source_path = Path(source)

    if source_path.is_dir():
        files = sorted(source_path.glob("*.*"))
        files = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
        _detect_images(det, files, out_dir, console)

    elif source_path.suffix.lower() in VIDEO_EXTENSIONS:
        _detect_video(det, source_path, out_dir, console, skip_frames, codec)

    elif source_path.suffix.lower() in IMAGE_EXTENSIONS:
        _detect_images(det, [source_path], out_dir, console)

    else:
        console.print(f"[red]Unknown source type: {source_path.suffix}[/red]")
        raise typer.Abort()


def _detect_images(det, files: list[Path], out_dir: Path, console):
    """Run detection on a list of image files."""
    for f in files:
        console.print(f"Processing {f.name}...")
        result = det(str(f))
        out_path = out_dir / f.name
        result.save(out_path)
        console.print(f"  {len(result.boxes)} detections -> {out_path}")

    console.print(f"[green]Done! {len(files)} images saved to {out_dir}[/green]")


def _detect_video(det, source_path: Path, out_dir: Path, console, skip_frames: int, codec: str):
    """Run detection on a video file."""
    out_path = out_dir / source_path.name
    console.print(f"Processing video {source_path.name}...")

    stats = det.detect_video_to_file(
        source=str(source_path),
        output=str(out_path),
        codec=codec,
        skip_frames=skip_frames,
    )

    console.print(
        f"  {stats['total_frames']} frames, "
        f"{stats['processed_frames']} processed, "
        f"{stats['total_detections']} total detections"
    )
    console.print(f"[green]Done! Video saved to {out_path}[/green]")
