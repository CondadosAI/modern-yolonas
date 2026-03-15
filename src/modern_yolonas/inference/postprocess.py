"""Post-processing: NMS, confidence filtering, box rescaling."""

from __future__ import annotations

import torch
from torch import Tensor
import torchvision


def postprocess(
    pred_bboxes: Tensor,
    pred_scores: Tensor,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    max_detections: int = 300,
    multi_label: bool = True,
) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Apply confidence filtering + NMS to batched predictions.

    Args:
        pred_bboxes: ``[B, N, 4]`` in x1y1x2y2 format (pixel coords).
        pred_scores: ``[B, N, C]`` class probabilities.
        conf_threshold: Minimum confidence.
        iou_threshold: NMS IoU threshold.
        max_detections: Maximum detections per image.
        multi_label: Allow multiple class detections per anchor.

    Returns:
        List of (boxes [D,4], scores [D], class_ids [D]) per batch element.
    """
    batch_size = pred_bboxes.shape[0]
    results = []

    for i in range(batch_size):
        boxes = pred_bboxes[i]  # [N, 4]
        scores = pred_scores[i]  # [N, C]

        if multi_label:
            # Multi-label: each anchor can produce detections for multiple classes
            anchor_idx, class_ids = (scores > conf_threshold).nonzero(as_tuple=True)
            if anchor_idx.shape[0] == 0:
                results.append((
                    torch.empty(0, 4, device=boxes.device),
                    torch.empty(0, device=boxes.device),
                    torch.empty(0, dtype=torch.long, device=boxes.device),
                ))
                continue
            boxes_filtered = boxes[anchor_idx]
            scores_filtered = scores[anchor_idx, class_ids]
        else:
            # Single-label: max score per anchor
            max_scores, class_ids = scores.max(dim=-1)
            mask = max_scores > conf_threshold
            boxes_filtered = boxes[mask]
            scores_filtered = max_scores[mask]
            class_ids = class_ids[mask]

            if boxes_filtered.shape[0] == 0:
                results.append((
                    torch.empty(0, 4, device=boxes.device),
                    torch.empty(0, device=boxes.device),
                    torch.empty(0, dtype=torch.long, device=boxes.device),
                ))
                continue

        # Top-K before NMS
        if boxes_filtered.shape[0] > max_detections * 3:
            topk = scores_filtered.topk(max_detections * 3).indices
            boxes_filtered = boxes_filtered[topk]
            scores_filtered = scores_filtered[topk]
            class_ids = class_ids[topk]

        # Batched NMS (per-class)
        keep = torchvision.ops.batched_nms(boxes_filtered, scores_filtered, class_ids, iou_threshold)
        keep = keep[:max_detections]

        results.append((boxes_filtered[keep], scores_filtered[keep], class_ids[keep]))

    return results


def rescale_boxes(
    boxes: Tensor,
    scale: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
) -> Tensor:
    """Rescale boxes from letterboxed coords to original image coords.

    Args:
        boxes: ``[D, 4]`` in x1y1x2y2 format.
        scale: Scale factor used in letterbox.
        pad: ``(pad_left, pad_top)``.
        orig_shape: ``(orig_h, orig_w)``.
    """
    boxes = boxes.clone()
    pad_left, pad_top = pad
    boxes[:, 0] -= pad_left
    boxes[:, 1] -= pad_top
    boxes[:, 2] -= pad_left
    boxes[:, 3] -= pad_top
    boxes /= scale
    # Clip to image
    orig_h, orig_w = orig_shape
    boxes[:, 0].clamp_(0, orig_w)
    boxes[:, 1].clamp_(0, orig_h)
    boxes[:, 2].clamp_(0, orig_w)
    boxes[:, 3].clamp_(0, orig_h)
    return boxes
