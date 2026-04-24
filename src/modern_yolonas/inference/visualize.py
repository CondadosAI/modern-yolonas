"""Draw detection boxes on images."""

from __future__ import annotations

import numpy as np

# COCO class names
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    """Deterministic color per class (BGR)."""
    # Simple hash-based palette
    r = ((class_id * 47) % 200) + 55
    g = ((class_id * 97) % 200) + 55
    b = ((class_id * 157) % 200) + 55
    return (int(b), int(g), int(r))


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str] | None = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Args:
        image: HWC uint8 BGR image (modified in-place and returned).
        boxes: ``[D, 4]`` x1y1x2y2.
        scores: ``[D]`` confidence scores.
        class_ids: ``[D]`` integer class IDs.
        class_names: List of class names (defaults to COCO).
        thickness: Box line thickness.
        font_scale: Label font scale.

    Returns:
        Annotated image.
    """
    import cv2

    if class_names is None:
        class_names = COCO_NAMES

    image = image.copy()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = map(int, box)
        color = _color_for_class(cls_id)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return image


def draw_ground_truth(
    image: np.ndarray,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str] | None = None,
    thickness: int = 2,
    font_scale: float = 0.4,
) -> np.ndarray:
    """Draw ground-truth boxes on *image* in a visually distinct style.

    GT boxes use a bright green outline with a small class label, making them
    easy to distinguish from prediction boxes drawn by :func:`draw_detections`.

    Args:
        image: HWC uint8 BGR image (a copy is made; the original is unchanged).
        boxes: ``[M, 4]`` x1y1x2y2 ground-truth boxes.
        class_ids: ``[M]`` integer class IDs.
        class_names: Optional list of class names.
        thickness: Box border thickness.
        font_scale: Label font scale.

    Returns:
        Annotated copy of the image.
    """
    import cv2

    if class_names is None:
        class_names = COCO_NAMES

    GT_COLOR = (0, 255, 0)  # bright green (BGR)

    image = image.copy()
    for box, cls_id in zip(boxes, class_ids):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), GT_COLOR, thickness)

        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(image, (x1, y2), (x1 + tw, y2 + th + 4), GT_COLOR, -1)
        cv2.putText(image, name, (x1, y2 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

    return image


def annotate_validation_sample(
    image_chw_float: "np.ndarray",
    pred_boxes: "np.ndarray",
    pred_scores: "np.ndarray",
    pred_labels: "np.ndarray",
    gt_boxes: "np.ndarray",
    gt_labels: "np.ndarray",
    class_names: list[str] | None = None,
) -> np.ndarray:
    """Convert a normalised CHW float tensor to a uint8 BGR image and overlay
    both predictions and ground-truth boxes.

    Conventions:

    - **Predictions** are drawn with class-coloured filled-background labels
      and confidence scores.
    - **Ground-truth** boxes are drawn in bright green underneath the
      predictions to make comparisons easy at a glance.

    Args:
        image_chw_float: ``[3, H, W]`` float32 array in ``[0, 1]`` RGB (as
            produced by :class:`~modern_yolonas.data.transforms.Normalize`).
        pred_boxes: ``[D, 4]`` x1y1x2y2 prediction boxes (pixel coords).
        pred_scores: ``[D]`` confidence scores.
        pred_labels: ``[D]`` integer predicted class IDs.
        gt_boxes: ``[M, 4]`` x1y1x2y2 ground-truth boxes (pixel coords).
        gt_labels: ``[M]`` integer ground-truth class IDs.
        class_names: Optional list of class names for both sets.

    Returns:
        HWC uint8 BGR annotated image ready for logging.
    """
    # [3,H,W] float [0,1] RGB → HWC uint8 BGR
    img_hwc = (image_chw_float.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    img_bgr = img_hwc[:, :, ::-1].copy()

    # GT first (underneath), then predictions on top
    if len(gt_boxes):
        img_bgr = draw_ground_truth(img_bgr, gt_boxes, gt_labels, class_names)
    if len(pred_boxes):
        img_bgr = draw_detections(img_bgr, pred_boxes, pred_scores, pred_labels, class_names)

    return img_bgr
