"""High-level detection API for images and video.

Results are returned as :class:`supervision.Detections`, the interchange format used
across the computer vision ecosystem. That buys filtering (``detections[detections.confidence > 0.5]``),
merging, and every supervision annotator, tracker and zone for free, instead of this
project growing its own versions of them.
"""

from __future__ import annotations

import time

from pathlib import Path
from typing import Generator

import numpy as np
import supervision as sv
import torch

from modern_yolonas.inference.preprocess import preprocess
from modern_yolonas.inference.postprocess import postprocess, rescale_boxes
from modern_yolonas.inference.visualize import COCO_NAMES
from modern_yolonas.validation import validate_confidence, validate_device, validate_input_size, validate_iou_threshold, validate_model_name

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


class Detector:
    """High-level detector: load model → preprocess → forward → postprocess.

    Usage::

        det = Detector("yolo_nas_s", device="cuda")

        # Single image
        image = cv2.imread("image.jpg")
        detections = det(image)
        cv2.imwrite("output.jpg", det.annotate(image, detections))

        # Filter like any supervision result
        people = detections[detections.class_id == 0]
        confident = detections[detections.confidence > 0.5]

        # From a custom checkpoint trained with --num-classes 3
        det = Detector("yolo_nas_s", weights="runs/train/best.pt", num_classes=3,
                       class_names=["cat", "dog", "bird"])

        # Video (yields per-frame results)
        for frame_idx, frame, detections in det.detect_video("video.mp4"):
            print(f"Frame {frame_idx}: {len(detections)} detections")
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
        weights: str | Path | None = None,
        num_classes: int = 80,
        class_names: list[str] | None = None,
    ):
        from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l

        validate_model_name(model)
        validate_confidence(conf_threshold)
        validate_iou_threshold(iou_threshold)
        validate_input_size(input_size)

        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision must be 'fp32' or 'fp16', got {precision!r}")

        if class_names is not None and len(class_names) != num_classes:
            raise ValueError(
                f"class_names has {len(class_names)} entries but num_classes is {num_classes}"
            )

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
        # Only the 80-class default can be named without being told; a fine-tuned
        # model with a different head gets unlabelled ids unless the caller says.
        if class_names is not None:
            self.class_names: list[str] | None = class_names
        else:
            self.class_names = COCO_NAMES if num_classes == len(COCO_NAMES) else None

        # Wall-clock of the last single-image detect call, for the FPS overlay and
        # for callers that want to report throughput without timing it themselves.
        self.last_inference_ms: float | None = None

        self._box_annotator = sv.BoxAnnotator()
        self._label_annotator = sv.LabelAnnotator()

        if weights is not None:
            # Load a custom checkpoint saved by Trainer._save_checkpoint.
            # Build the architecture for the requested num_classes (no pretrained
            # weights), then overwrite with the checkpoint's model_state_dict.
            self.model = builders[model](pretrained=False, num_classes=num_classes).to(self.device)
            ckpt = torch.load(weights, map_location=self.device, weights_only=True)
            state_dict = ckpt.get("ema", {}).get("ema") or ckpt.get("model_state_dict")
            if state_dict is None:
                raise KeyError(
                    f"Checkpoint {weights!r} does not contain 'model_state_dict' or 'ema'. "
                    "Make sure it was saved by the Trainer."
                )
            self.model.load_state_dict(state_dict)
        else:
            self.model = builders[model](pretrained=pretrained, num_classes=num_classes).to(self.device)

        if precision == "fp16":
            self.model = self.model.half()
        self.model.eval()

    def _to_detections(self, boxes: torch.Tensor, scores: torch.Tensor, class_ids: torch.Tensor) -> sv.Detections:
        """Wrap raw postprocessed tensors as an ``sv.Detections``."""
        if len(boxes) == 0:
            return sv.Detections.empty()

        class_id = class_ids.cpu().numpy().astype(int)
        detections = sv.Detections(
            xyxy=boxes.cpu().numpy().astype(np.float32),
            confidence=scores.cpu().numpy().astype(np.float32),
            class_id=class_id,
        )
        if self.class_names is not None:
            # sv.LabelAnnotator reads this key, so labels need no lookup table.
            detections.data["class_name"] = np.array(
                [self.class_names[i] if i < len(self.class_names) else f"class_{i}" for i in class_id]
            )
        return detections

    def annotate(self, image: np.ndarray, detections: sv.Detections, show_fps: bool = False) -> np.ndarray:
        """Draw boxes and labels on a copy of ``image``.

        A convenience wrapper over ``sv.BoxAnnotator`` and ``sv.LabelAnnotator``. Build
        your own annotators when you want different styling.

        Args:
            image: BGR frame to draw on.
            detections: What to draw.
            show_fps: Overlay the last call's inference time and the frame rate it
                implies. Reads `last_inference_ms`, so it reflects the most recent
                detect call, not the frame passed here — they are the same frame in a
                normal capture loop.
        """
        annotated = self._box_annotator.annotate(image.copy(), detections)
        annotated = self._label_annotator.annotate(annotated, detections)
        if show_fps and self.last_inference_ms is not None:
            import cv2

            fps = 1000.0 / self.last_inference_ms if self.last_inference_ms > 0 else 0.0
            text = f"{self.last_inference_ms:.1f}ms ({fps:.1f} FPS)"
            cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return annotated

    @torch.no_grad()
    def __call__(
        self,
        source: str | Path | np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> sv.Detections:
        """Run detection on a single image.

        Args:
            source: File path or BGR numpy array.
            conf_threshold: Override instance default.
            iou_threshold: Override instance default.
        """
        import cv2

        if isinstance(source, (str, Path)):
            image = cv2.imread(str(source))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
        else:
            image = source

        t0 = time.perf_counter()

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

        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

        return self._to_detections(boxes, scores, class_ids)

    @torch.no_grad()
    def detect_batch(
        self,
        sources: list[str | Path | np.ndarray],
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[sv.Detections]:
        """Run detection on a batch of images in a single forward pass.

        Args:
            sources: List of file paths or BGR numpy arrays.
            conf_threshold: Override instance default.
            iou_threshold: Override instance default.

        Returns:
            One ``sv.Detections`` per input image, in input order.
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
            detections.append(self._to_detections(boxes, scores, class_ids))
        return detections

    def detect_video(
        self,
        source: str | Path | int,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        skip_frames: int = 0,
    ) -> Generator[tuple[int, np.ndarray, sv.Detections], None, None]:
        """Run detection on each frame of a video.

        Reads with OpenCV rather than ``sv.get_video_frames_generator`` because that
        one takes a file path only, and this accepts a camera index too.

        Args:
            source: Video file path or camera index (0 for webcam).
            conf_threshold: Override instance default.
            iou_threshold: Override instance default.
            skip_frames: Process every N-th frame (0 = every frame).

        Yields:
            ``(frame_index, frame, detections)`` for each processed frame. The frame is
            the BGR array as read, so it can be annotated or written directly.
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

                detections = self(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
                yield frame_idx, frame, detections
                frame_idx += 1
        finally:
            cap.release()

    def detect_video_to_file(
        self,
        source: str | Path,
        output: str | Path,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        codec: str = "mp4v",
        skip_frames: int = 0,
        show_fps: bool = False,
    ) -> dict[str, int | float]:
        """Run detection on a video and write annotated output.

        Args:
            source: Input video path.
            output: Output video path.
            conf_threshold: Override instance default.
            iou_threshold: Override instance default.
            codec: FourCC codec string.
            skip_frames: Process every N-th frame (0 = every frame).
                Skipped frames are written without annotations.
            show_fps: Burn the per-frame inference time and frame rate into the output.

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
                    detections = self(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
                    writer.write(self.annotate(frame, detections, show_fps=show_fps))
                    processed += 1
                    total_detections += len(detections)
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
