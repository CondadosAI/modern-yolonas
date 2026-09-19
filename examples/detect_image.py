"""Example: Run YOLO-NAS inference on a single image.

Usage:
    uv run examples/detect_image.py path/to/image.jpg
    uv run examples/detect_image.py path/to/image.jpg --model yolo_nas_l --device cpu
"""

import argparse

import cv2

from modern_yolonas.inference.detect import YoloNASDetector


def main():
    parser = argparse.ArgumentParser(description="YOLO-NAS image detection")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", default="yolo_nas_s", choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--output", default="output.jpg", help="Output image path")
    args = parser.parse_args()

    # Create detector (downloads pretrained weights on first run)
    det = YoloNASDetector(args.model, device=args.device, conf_threshold=args.conf, iou_threshold=args.iou)

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    # Run detection -> an sv.Detections
    detections = det(image)

    print(f"Found {len(detections)} objects:")
    for box, score, name in zip(detections.xyxy, detections.confidence, detections.data["class_name"]):
        x1, y1, x2, y2 = box
        print(f"  {name}: {score:.2f} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    # Save annotated image
    cv2.imwrite(args.output, det.annotate(image, detections))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
