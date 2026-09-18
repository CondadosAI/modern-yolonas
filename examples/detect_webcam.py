# uv extra dependence to force use opencv-python instead of opencv-python-headless, which is incompatible with cv2.imshow

"""Example: Run YOLO-NAS live on webcam feed.

Usage:
    uv run examples/detect_webcam.py
    uv run examples/detect_webcam.py --model yolo_nas_m --device cuda
"""

import argparse

import cv2

from modern_yolonas.inference.detect import Detector


def main():
    parser = argparse.ArgumentParser(description="YOLO-NAS webcam detection")
    parser.add_argument("--model", default="yolo_nas_s", choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--debug", action="store_true", help="Show FPS and inference speed overlay")
    args = parser.parse_args()

    det = Detector(args.model, device=args.device, conf_threshold=args.conf)

    print("Press 'q' to quit")
    for _frame_idx, frame, detections in det.detect_video(source=args.camera):
        cv2.imshow("YOLO-NAS", det.annotate(frame, detections, show_fps=args.debug))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
