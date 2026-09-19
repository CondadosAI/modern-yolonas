"""Example: Run YOLO-NAS inference on a video file.

Usage:
    uv run examples/detect_video.py path/to/video.mp4
    uv run examples/detect_video.py path/to/video.mp4 --output output.mp4 --model yolo_nas_l
"""

import argparse
from pathlib import Path

from modern_yolonas.inference.detect import YoloNASDetector


def main():
    parser = argparse.ArgumentParser(description="YOLO-NAS video detection")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--model", default="yolo_nas_s", choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--output", default=None, help="Output video path (default: <input>_detect.<ext>)")
    parser.add_argument("--skip-frames", type=int, default=0, help="Process every N-th frame (0 = all)")
    parser.add_argument("--codec", default="mp4v", help="Output video codec")
    parser.add_argument("--debug", action="store_true", help="Show FPS and inference speed overlay")
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        src = Path(args.video)
        args.output = str(src.parent / f"{src.stem}_detect{src.suffix}")

    # Create detector
    det = YoloNASDetector(args.model, device=args.device, conf_threshold=args.conf, iou_threshold=args.iou)

    # --- Option 1: Write annotated video directly ---
    print(f"Processing {args.video} ...")
    stats = det.detect_video_to_file(
        source=args.video,
        output=args.output,
        codec=args.codec,
        skip_frames=args.skip_frames,
        show_fps=args.debug,
    )
    print(
        f"Done! {stats['total_frames']} frames, "
        f"{stats['processed_frames']} processed, "
        f"{stats['total_detections']} total detections"
    )
    print(f"Saved to {args.output}")

    # --- Option 2: Iterate frames with a generator (commented out) ---
    # This is useful when you need custom per-frame logic:
    #
    # for frame_idx, frame, detections in det.detect_video(args.video):
    #     print(f"Frame {frame_idx}: {len(detections)} detections")
    #     # detections.xyxy / .confidence / .class_id, and filtering:
    #     #   people = detections[detections.class_id == COCOClass.PERSON]
    #     # Or get an annotated frame: det.annotate(frame, detections)


if __name__ == "__main__":
    main()
