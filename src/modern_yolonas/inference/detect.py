"""High-level detection API for images and video.

Results are returned as :class:`supervision.Detections`, the interchange format used
across the computer vision ecosystem. That buys filtering (``detections[detections.confidence > 0.5]``),
merging, and every supervision annotator, tracker and zone for free, instead of this
project growing its own versions of them.
"""

from __future__ import annotations

import time

from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np
import supervision as sv
import torch

from modern_yolonas.inference.embed import (
    FeaturePooler,
    Prediction,
    Task,
    boxes_to_rois,
    valid_region,
)
from modern_yolonas.inference.preprocess import preprocess
from modern_yolonas.inference.postprocess import postprocess, rescale_boxes
from modern_yolonas.inference.visualize import COCO_NAMES
from modern_yolonas.validation import validate_confidence, validate_device, validate_input_size, validate_iou_threshold, validate_model_name

if TYPE_CHECKING:
    from modern_yolonas.tracking import DeepHMSort

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


class YoloNASDetector:
    """High-level detector: load model → preprocess → forward → postprocess.

    Usage::

        from modern_yolonas import COCOClass, YoloNASDetector

        det = YoloNASDetector("yolo_nas_s", device="cuda")

        # Single image
        image = cv2.imread("image.jpg")
        detections = det(image)
        cv2.imwrite("output.jpg", det.annotate(image, detections))

        # Filter like any supervision result
        people = detections[detections.class_id == COCOClass.PERSON]
        confident = detections[detections.confidence > 0.5]

        # From a custom checkpoint trained with --num-classes 3
        det = YoloNASDetector("yolo_nas_s", weights="runs/train/best.pt", num_classes=3,
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
        embedding: FeaturePooler | None = None,
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

        # How feature maps become vectors, when `predict` is asked for embeddings.
        # Defaults to the same c5 / average / L2-normalized choice YoloNASEmbedder
        # makes; pass a FeaturePooler to pick different layers or pooling.
        self.pooler = embedding if embedding is not None else FeaturePooler()
        self._embedding_dim: int | None = None

        self._box_annotator = sv.BoxAnnotator()
        self._label_annotator = sv.LabelAnnotator()
        # Tracked output is coloured by id, not by class, so a swap shows up as a
        # colour change instead of hiding inside a row of same-class boxes.
        self._track_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self._track_label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)

        if weights is not None:
            # Build the architecture for the requested num_classes (no pretrained
            # weights), then overwrite it with whatever the checkpoint holds —
            # Lightning .ckpt, a legacy trainer dict, or a plain state_dict.
            from modern_yolonas.weights import extract_model_state_dict

            self.model = builders[model](pretrained=False, num_classes=num_classes).to(self.device)
            self.model.load_state_dict(extract_model_state_dict(weights, map_location=str(self.device)))
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

    @property
    def embedding_dim(self) -> int:
        """Width of the vectors :meth:`predict` produces, for the configured pooler."""
        if self._embedding_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.input_size, self.input_size, device=self.device)
                if self.precision == "fp16":
                    dummy = dummy.half()
                self._embedding_dim = self.pooler.dim(self.model.forward_features(dummy))
        return self._embedding_dim

    def _read(self, source: str | Path | np.ndarray) -> np.ndarray:
        import cv2

        if isinstance(source, (str, Path)):
            image = cv2.imread(str(source))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
            return image
        return source

    @torch.no_grad()
    def predict_batch(
        self,
        sources: list[str | Path | np.ndarray],
        tasks: Task = Task.DETECT,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[Prediction]:
        """Run every requested task over a batch in **one** forward pass.

        Args:
            sources: File paths or BGR numpy arrays.
            tasks: Any combination of :class:`Task` flags, combined with ``|``.
            conf_threshold: Override instance default.
            iou_threshold: Override instance default.

        Returns:
            One :class:`Prediction` per input, in input order.
        """
        if Task.EMBED_OBJECTS in tasks:
            # The boxes are what gets embedded, so detection is not optional here.
            tasks = tasks | Task.DETECT
        if not tasks:
            raise ValueError("tasks must name at least one Task flag")
        if not sources:
            return []

        images, tensors, scales, pads = [], [], [], []
        for source in sources:
            image = self._read(source)
            tensor, scale, pad = preprocess(image, self.input_size)
            images.append(image)
            tensors.append(tensor)
            scales.append(scale)
            pads.append(pad)

        batch = torch.cat(tensors, dim=0).to(self.device)
        if self.precision == "fp16":
            batch = batch.half()

        # Backbone and neck run exactly once; the head is a thin extra on top of
        # p3/p4/p5, which is why both outputs cost roughly one detection pass.
        with torch.amp.autocast("cuda", enabled=self.precision == "fp16"):
            features = self.model.forward_features(batch)
            if Task.DETECT in tasks:
                pred_bboxes, pred_scores = self.model.heads(
                    (features["p3"], features["p4"], features["p5"])
                )

        predictions = [Prediction() for _ in sources]

        if Task.EMBED in tasks:
            regions = [valid_region(img, sc, pd) for img, sc, pd in zip(images, scales, pads)]
            vectors = self.pooler.finalize(self.pooler.pool_images(features, regions, self.input_size))
            for prediction, vector in zip(predictions, vectors):
                prediction.embedding = vector

        if Task.DETECT in tasks:
            conf = conf_threshold if conf_threshold is not None else self.conf_threshold
            iou = iou_threshold if iou_threshold is not None else self.iou_threshold
            results = postprocess(pred_bboxes, pred_scores, conf, iou, multi_label=self.multi_label)

            rescaled = [
                rescale_boxes(boxes, scales[i], pads[i], images[i].shape[:2])
                for i, (boxes, _, _) in enumerate(results)
            ]

            # Embed the boxes *after* rescaling, then map them back onto the canvas.
            # `rescale_boxes` clips to the frame, and a detection that runs off the
            # edge should be described by the part of it that is actually visible —
            # the rest of its extent is letterbox padding. Going through the clipped
            # boxes is also what makes these vectors identical to the ones
            # `YoloNASEmbedder.embed_boxes` produces from the same detections.
            object_vectors = (
                self._embed_detected(features, rescaled, scales, pads)
                if Task.EMBED_OBJECTS in tasks
                else None
            )

            for i, (_, scores, class_ids) in enumerate(results):
                detections = self._to_detections(rescaled[i], scores, class_ids)
                if object_vectors is not None and len(detections):
                    detections.data["embedding"] = object_vectors[i]
                predictions[i].detections = detections

        return predictions

    def _embed_detected(
        self,
        features: dict[str, torch.Tensor],
        rescaled: list[torch.Tensor],
        scales: list[float],
        pads: list[tuple[int, int]],
    ) -> list[np.ndarray]:
        """Per-detection vectors for a whole batch, in one ``roi_align`` call.

        Args:
            features: Feature maps from the shared forward pass.
            rescaled: Per image, the detected boxes in source-image coordinates.
            scales: Per image letterbox scale.
            pads: Per image letterbox ``(left, top)``.
        """
        rois = [
            boxes_to_rois(boxes.cpu().numpy(), scales[i], pads[i], batch_index=i)
            for i, boxes in enumerate(rescaled)
            if len(boxes)
        ]
        if not rois:
            return [np.zeros((0, self.embedding_dim), dtype=np.float32) for _ in rescaled]

        pooled = self.pooler.finalize(
            self.pooler.pool_rois(features, torch.cat(rois).to(self.device), self.input_size)
        )

        # Split the flat (K, D) block back into one array per image.
        per_image, offset = [], 0
        for boxes in rescaled:
            per_image.append(pooled[offset : offset + len(boxes)])
            offset += len(boxes)
        return per_image

    def predict(
        self,
        source: str | Path | np.ndarray,
        tasks: Task = Task.DETECT,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> Prediction:
        """Run every requested task on one image in a single forward pass.

        Detection and embedding share the whole network up to the head, so asking
        for both costs one pass rather than two::

            from modern_yolonas import Task, YoloNASDetector

            detector = YoloNASDetector("yolo_nas_s")
            result = detector.predict(image, Task.DETECT | Task.EMBED | Task.EMBED_OBJECTS)

            result.detections                       # sv.Detections
            result.embedding                        # (768,) whole-image vector
            result.detections.data["embedding"]     # (N, 768), one row per detection

        Because the per-object vectors live in ``detections.data``, they follow the
        boxes through supervision's slicing::

            people = result.detections[result.detections.class_id == COCOClass.PERSON]
            people.data["embedding"]                # rows still aligned

        Args:
            source: File path or BGR numpy array.
            tasks: Any combination of :class:`Task` flags, combined with ``|``.
                :attr:`Task.EMBED_OBJECTS` implies :attr:`Task.DETECT`.
            conf_threshold: Override instance default.
            iou_threshold: Override instance default.

        Returns:
            A :class:`Prediction`; fields not asked for are ``None``.
        """
        t0 = time.perf_counter()
        prediction = self.predict_batch([source], tasks, conf_threshold, iou_threshold)[0]
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return prediction

    def __call__(
        self,
        source: str | Path | np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> sv.Detections:
        """Run detection on a single image.

        Shorthand for ``predict(source, Task.DETECT).detections``. Use
        :meth:`predict` when you want embeddings from the same pass.

        Args:
            source: File path or BGR numpy array.
            conf_threshold: Override instance default.
            iou_threshold: Override instance default.
        """
        return self.predict(source, Task.DETECT, conf_threshold, iou_threshold).detections

    def detect_batch(
        self,
        sources: list[str | Path | np.ndarray],
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[sv.Detections]:
        """Run detection on a batch of images in a single forward pass.

        Shorthand for the ``detections`` of
        ``predict_batch(sources, Task.DETECT)``.

        Args:
            sources: List of file paths or BGR numpy arrays.
            conf_threshold: Override instance default.
            iou_threshold: Override instance default.

        Returns:
            One ``sv.Detections`` per input image, in input order.
        """
        predictions = self.predict_batch(sources, Task.DETECT, conf_threshold, iou_threshold)
        return [prediction.detections for prediction in predictions]

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


    def track_video(
        self,
        source: str | Path | int,
        tracker: DeepHMSort | None = None,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        appearance: bool = True,
        skip_frames: int = 0,
    ) -> Generator[tuple[int, np.ndarray, sv.Detections], None, None]:
        """Detect and track each frame of a video, in one forward pass per frame.

        The appearance vectors Deep HM-SORT associates on are the same per-object
        embeddings :meth:`predict` produces, read off the features the detection
        pass already computed — so tracking with appearance costs one pass per
        frame, not the two a bolted-on re-identification model would need.

        The detector runs at the tracker's ``track_low_threshold`` unless told
        otherwise. That is deliberate: Deep HM-SORT's second association round
        exists to hold a track through a frame where the detector wavers, and it
        can only do that if those weak detections reach it. Raising
        ``conf_threshold`` to the tracker's ``track_high_threshold`` turns that
        round off.

        Args:
            source: Video file path or camera index (0 for webcam).
            tracker: A configured :class:`~modern_yolonas.tracking.DeepHMSort`, or
                ``None`` for a default one. Pass your own to tune it, to keep
                inspecting ``tracker.tracks``, or to reuse it across calls — it is
                stateful, so :meth:`~modern_yolonas.tracking.DeepHMSort.reset`
                between unrelated videos. Its ``frame_rate`` is set from this
                video, so ``max_lost_seconds`` means the same span of time
                whatever the clip was shot at.
            conf_threshold: Override the tracker-derived detection threshold.
            iou_threshold: Override the instance default.
            appearance: Compute per-object embeddings and associate on them.
                ``False`` drops to motion-only association, which is faster by the
                ROI pooling and markedly worse through occlusions.
            skip_frames: Process every N-th frame (0 = every frame). Note the
                tracker sees only the processed frames, so objects move further
                between them and the expansion has more work to do.

        Yields:
            ``(frame_index, frame, detections)`` for each processed frame, where
            ``detections`` carries ``tracker_id`` and holds only the detections
            that matched a track.
        """
        from modern_yolonas.tracking import DeepHMSort

        if tracker is None:
            tracker = DeepHMSort()
        if conf_threshold is None:
            conf_threshold = tracker.track_low_threshold

        tasks = Task.DETECT | Task.EMBED_OBJECTS if appearance else Task.DETECT

        import cv2

        cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {source}")

        # The tracker's memory is written in seconds; tell it how long a frame is.
        # Files usually report an honest rate, cameras often report 0 or something
        # absurd, so an implausible value is left alone rather than believed. With
        # `skip_frames` the tracker sees a slower video than the file claims.
        reported = cap.get(cv2.CAP_PROP_FPS)
        if 1.0 <= reported <= 240.0:
            tracker.frame_rate = reported / (skip_frames + 1)

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                t0 = time.perf_counter()
                prediction = self.predict(frame, tasks, conf_threshold, iou_threshold)
                tracked = tracker.update_with_detections(prediction.detections)
                self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

                yield frame_idx, frame, tracked
                frame_idx += 1
        finally:
            cap.release()

    def track_video_to_file(
        self,
        source: str | Path,
        output: str | Path,
        tracker: DeepHMSort | None = None,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        appearance: bool = True,
        codec: str = "mp4v",
        show_fps: bool = False,
    ) -> dict[str, int | float]:
        """Track a video and write it out with ids drawn on.

        Args:
            source: Input video path.
            output: Output video path.
            tracker: See :meth:`track_video`.
            conf_threshold: See :meth:`track_video`.
            iou_threshold: Override the instance default.
            appearance: See :meth:`track_video`.
            codec: FourCC codec string.
            show_fps: Burn the per-frame time and frame rate into the output.

        Returns:
            Dict with ``total_frames``, ``total_detections`` and ``unique_ids`` —
            the last being how many distinct objects the tracker believes it saw,
            which is the number to watch when tuning: far above the truth means
            ids are fragmenting.
        """
        import cv2

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {output} with codec '{codec}'")

        seen: set[int] = set()
        total_detections = 0
        frames = 0

        try:
            for frame_idx, frame, detections in self.track_video(
                source, tracker, conf_threshold, iou_threshold, appearance
            ):
                writer.write(self.annotate_tracks(frame, detections, show_fps=show_fps))
                frames = frame_idx + 1
                total_detections += len(detections)
                if detections.tracker_id is not None:
                    seen.update(int(i) for i in detections.tracker_id)
        finally:
            writer.release()

        return {
            "total_frames": frames,
            "total_detections": total_detections,
            "unique_ids": len(seen),
            "fps": fps,
        }

    def annotate_tracks(
        self, image: np.ndarray, detections: sv.Detections, show_fps: bool = False
    ) -> np.ndarray:
        """Draw tracked boxes, coloured and labelled by ``tracker_id``.

        Colouring by id rather than by class is what makes an ID-swap visible: the
        box changes colour the instant the tracker changes its mind.

        Args:
            image: BGR frame to draw on.
            detections: Output of :meth:`track_video`.
            show_fps: Overlay the last frame's time and frame rate.
        """
        if detections.tracker_id is None:
            return self.annotate(image, detections, show_fps=show_fps)

        labels = []
        for i, track_id in enumerate(detections.tracker_id):
            name = detections.data.get("class_name", [None] * len(detections))[i]
            labels.append(f"#{int(track_id)} {name}" if name is not None else f"#{int(track_id)}")

        annotated = self._track_box_annotator.annotate(image.copy(), detections)
        annotated = self._track_label_annotator.annotate(annotated, detections, labels=labels)
        if show_fps and self.last_inference_ms is not None:
            import cv2

            rate = 1000.0 / self.last_inference_ms if self.last_inference_ms > 0 else 0.0
            cv2.putText(
                annotated, f"{self.last_inference_ms:.1f}ms ({rate:.1f} FPS)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
        return annotated


def _warn_detector_alias(module_name: str) -> type[YoloNASDetector]:
    """Back the deprecated `Detector` spelling, kept through one minor release.

    Renamed in 0.5.0 so the ergonomic classes can grow task siblings
    (`YoloNASSegmenter`, `YoloNASPoseEstimator`) without `YoloNAS*` colliding
    with the `nn.Module` names those tasks will want.
    """
    import warnings

    warnings.warn(
        f"{module_name}.Detector is deprecated and will be removed in 0.7.0; "
        "use YoloNASDetector instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return YoloNASDetector


def __getattr__(name: str) -> type[YoloNASDetector]:
    if name == "Detector":
        return _warn_detector_alias(__name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
