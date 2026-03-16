"""High-level detection API for images and video."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import numpy as np
import torch

from modern_yolonas.inference.preprocess import preprocess
from modern_yolonas.inference.postprocess import postprocess, rescale_boxes
from modern_yolonas.inference.visualize import draw_detections
from modern_yolonas.validation import validate_confidence, validate_device, validate_input_size, validate_iou_threshold, validate_model_name

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


@dataclass
class Detection:
    """Detection results for a single image/frame."""

    boxes: np.ndarray  # [D, 4] x1y1x2y2
    scores: np.ndarray  # [D]
    class_ids: np.ndarray  # [D]
    image: np.ndarray | None = field(default=None, repr=False)

    def visualize(self, class_names: list[str] | None = None) -> np.ndarray:
        """Draw detections on the original image."""
        if self.image is None:
            raise ValueError("Original image not stored; pass retain_image=True to Detector")
        return draw_detections(self.image, self.boxes, self.scores, self.class_ids, class_names)

    def save(self, path: str | Path, class_names: list[str] | None = None):
        """Visualize and save to file."""
        import cv2

        img = self.visualize(class_names)
        if not cv2.imwrite(str(path), img):
            raise IOError(f"Failed to save image to {path}. Check the path and directory exist.")


class Detector:
    """High-level detector: load model → preprocess → forward → postprocess.

    Usage::

        det = Detector("yolo_nas_s", device="cuda")

        # Single image
        result = det("image.jpg")
        result.save("output.jpg")

        # Video (yields per-frame results)
        for frame_idx, result in det.detect_video("video.mp4"):
            print(f"Frame {frame_idx}: {len(result.boxes)} detections")
    """

    def __init__(
        self,
        model: str = "yolo_nas_s",
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        input_size: int = 640,
        pretrained: bool = True,
        multi_label: bool = True,
        precision: str = "fp32",
    ):
        from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l

        validate_model_name(model)
        validate_confidence(conf_threshold)
        validate_iou_threshold(iou_threshold)
        validate_input_size(input_size)

        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision must be 'fp32' or 'fp16', got {precision!r}")

        builders = {
            "yolo_nas_s": yolo_nas_s,
            "yolo_nas_m": yolo_nas_m,
            "yolo_nas_l": yolo_nas_l,
        }

        self.device = validate_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.multi_label = multi_label
        self.precision = precision

        self.model = builders[model](pretrained=pretrained).to(self.device)
        if precision == "fp16":
            self.model = self.model.half()
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        source: str | Path | np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        retain_image: bool = True,
    ) -> Detection:
        """Run detection on a single image.

        Args:
            source: File path or BGR numpy array.
            conf_threshold: Override instance default.
            iou_threshold: Override instance default.
            retain_image: Store original image in result for visualization.
        """
        import cv2

        if isinstance(source, (str, Path)):
            image = cv2.imread(str(source))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
        else:
            image = source

        tensor, scale, pad = preprocess(image, self.input_size)
        tensor = tensor.to(self.device)
        if self.precision == "fp16":
            tensor = tensor.half()

        with torch.amp.autocast("cuda", enabled=self.precision == "fp16"):
            pred_bboxes, pred_scores = self.model(tensor)

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold
        results = postprocess(pred_bboxes, pred_scores, conf, iou, multi_label=self.multi_label)

        boxes, scores, class_ids = results[0]
        boxes = rescale_boxes(boxes, scale, pad, image.shape[:2])

        return Detection(
            boxes=boxes.cpu().numpy(),
            scores=scores.cpu().numpy(),
            class_ids=class_ids.cpu().numpy(),
            image=image if retain_image else None,
        )

    @torch.no_grad()
    def detect_batch(
        self,
        sources: list[str | Path | np.ndarray],
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        retain_image: bool = True,
    ) -> list[Detection]:
        """Run detection on a batch of images in a single forward pass.

        Args:
            sources: List of file paths or BGR numpy arrays.
            conf_threshold: Override instance default.
            iou_threshold: Override instance default.
            retain_image: Store original image in result for visualization.

        Returns:
            List of Detection results, one per input image.
        """
        import cv2

        images = []
        tensors = []
        scales = []
        pads = []

        for source in sources:
            if isinstance(source, (str, Path)):
                image = cv2.imread(str(source))
                if image is None:
                    raise FileNotFoundError(f"Cannot read image: {source}")
            else:
                image = source
            images.append(image)

            tensor, scale, pad = preprocess(image, self.input_size)
            tensors.append(tensor)
            scales.append(scale)
            pads.append(pad)

        batch_tensor = torch.cat(tensors, dim=0).to(self.device)
        pred_bboxes, pred_scores = self.model(batch_tensor)

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold
        results = postprocess(pred_bboxes, pred_scores, conf, iou, multi_label=self.multi_label)

        detections = []
        for i, (boxes, scores, class_ids) in enumerate(results):
            boxes = rescale_boxes(boxes, scales[i], pads[i], images[i].shape[:2])
            detections.append(Detection(
                boxes=boxes.cpu().numpy(),
                scores=scores.cpu().numpy(),
                class_ids=class_ids.cpu().numpy(),
                image=images[i] if retain_image else None,
            ))
        return detections

    def detect_video(
        self,
        source: str | Path | int,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        retain_image: bool = True,
        skip_frames: int = 0,
    ) -> Generator[tuple[int, Detection], None, None]:
        """Run detection on each frame of a video.

        Args:
            source: Video file path or camera index (0 for webcam).
            conf_threshold: Override instance default.
            iou_threshold: Override instance default.
            retain_image: Store frame in each Detection result.
            skip_frames: Process every N-th frame (0 = every frame).

        Yields:
            ``(frame_index, Detection)`` for each processed frame.
        """
        import cv2

        cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {source}")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                result = self(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold, retain_image=retain_image)
                yield frame_idx, result
                frame_idx += 1
        finally:
            cap.release()

    def detect_video_to_file(
        self,
        source: str | Path,
        output: str | Path,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        class_names: list[str] | None = None,
        codec: str = "mp4v",
        skip_frames: int = 0,
    ) -> dict[str, int | float]:
        """Run detection on a video and write annotated output.

        Args:
            source: Input video path.
            output: Output video path.
            conf_threshold: Override instance default.
            iou_threshold: Override instance default.
            class_names: Class names for labels (defaults to COCO).
            codec: FourCC codec string.
            skip_frames: Process every N-th frame (0 = every frame).
                Skipped frames are written without annotations.

        Returns:
            Dict with ``total_frames``, ``processed_frames``, ``total_detections``.
        """
        import cv2

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {output} with codec '{codec}'")

        frame_idx = 0
        processed = 0
        total_detections = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                should_process = skip_frames == 0 or frame_idx % (skip_frames + 1) == 0

                if should_process:
                    result = self(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
                    annotated = draw_detections(frame, result.boxes, result.scores, result.class_ids, class_names)
                    writer.write(annotated)
                    processed += 1
                    total_detections += len(result.boxes)
                else:
                    writer.write(frame)

                frame_idx += 1
        finally:
            cap.release()
            writer.release()

        return {
            "total_frames": frame_idx,
            "processed_frames": processed,
            "total_detections": total_detections,
            "fps": fps,
        }
