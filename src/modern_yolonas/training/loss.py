"""PPYoloE loss for YOLO-NAS training.

Components:
- TaskAlignedAssigner: dynamic label assignment
- ATSSAssigner: static label assignment for warmup epochs
- VarifocalLoss: classification loss
- GIoULoss: box regression loss (per-element)
- DFLLoss: distribution focal loss (per-anchor)
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn, Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def bbox_iou(box1: Tensor, box2: Tensor, eps: float = 1e-9) -> Tensor:
    """Compute IoU between two sets of boxes (x1y1x2y2 format).

    Args:
        box1: ``[N, 4]``
        box2: ``[M, 4]``

    Returns:
        ``[N, M]`` IoU matrix.
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + eps)


def batch_bbox_iou(box1: Tensor, box2: Tensor, eps: float = 1e-9) -> Tensor:
    """Compute IoU between two batched sets of boxes (x1y1x2y2 format).

    Args:
        box1: ``[B, N, 4]``
        box2: ``[B, M, 4]``

    Returns:
        ``[B, N, M]`` IoU matrix.
    """
    area1 = (box1[:, :, 2] - box1[:, :, 0]) * (box1[:, :, 3] - box1[:, :, 1])
    area2 = (box2[:, :, 2] - box2[:, :, 0]) * (box2[:, :, 3] - box2[:, :, 1])

    inter_x1 = torch.max(box1[:, :, None, 0], box2[:, None, :, 0])
    inter_y1 = torch.max(box1[:, :, None, 1], box2[:, None, :, 1])
    inter_x2 = torch.min(box1[:, :, None, 2], box2[:, None, :, 2])
    inter_y2 = torch.min(box1[:, :, None, 3], box2[:, None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, :, None] + area2[:, None, :] - inter

    return inter / (union + eps)


def batch_distance2bbox(points: Tensor, distance: Tensor) -> Tensor:
    """Convert distance predictions (l, t, r, b) to bounding boxes (x1, y1, x2, y2).

    Args:
        points: ``[N, 2]`` anchor points (x, y).
        distance: ``[B, N, 4]`` distances.

    Returns:
        ``[B, N, 4]`` bounding boxes.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ---------------------------------------------------------------------------
# Task-Aligned Assigner
# ---------------------------------------------------------------------------


class TaskAlignedAssigner:
    """Dynamic label assignment based on task alignment metric.

    Computes alignment metric = score^alpha * iou^beta and selects
    top-K anchors per GT as positive samples.
    """

    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def assign(
        self,
        pred_scores: Tensor,
        pred_bboxes: Tensor,
        anchor_points: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Assign ground truths to anchors.

        Args:
            pred_scores: ``[B, N, C]`` predicted class scores (sigmoid).
            pred_bboxes: ``[B, N, 4]`` predicted boxes (x1y1x2y2).
            anchor_points: ``[N, 2]``.
            gt_labels: ``[B, max_gt, 1]`` class labels.
            gt_bboxes: ``[B, max_gt, 4]`` ground truth boxes (x1y1x2y2).
            mask_gt: ``[B, max_gt, 1]`` valid GT mask.

        Returns:
            assigned_labels, assigned_bboxes, assigned_scores, fg_mask
        """
        batch_size = pred_scores.shape[0]
        num_anchors = pred_scores.shape[1]
        num_classes = pred_scores.shape[2]
        num_max_boxes = gt_bboxes.shape[1]
        device = pred_scores.device

        if num_max_boxes == 0:
            return (
                torch.full([batch_size, num_anchors], 0, dtype=torch.long, device=device),
                torch.zeros([batch_size, num_anchors, 4], device=device),
                torch.zeros([batch_size, num_anchors, num_classes], device=device),
                torch.zeros([batch_size, num_anchors], dtype=torch.bool, device=device),
            )

        # Check which anchors are inside GT boxes: [B, N, M]
        lt = anchor_points[None, :, None, :] - gt_bboxes[:, None, :, :2]
        rb = gt_bboxes[:, None, :, 2:] - anchor_points[None, :, None, :]
        bbox_deltas = torch.cat([lt, rb], dim=-1)
        mask_in_gts = bbox_deltas.amin(dim=-1) > self.eps

        # GT class labels: [B, M]
        gt_labels_expanded = gt_labels.squeeze(-1).long()

        # Gather predicted scores for GT classes: [B, N, M]
        gt_labels_for_gather = gt_labels_expanded[:, None, :].expand(-1, num_anchors, -1)
        gt_labels_for_gather = gt_labels_for_gather.clamp(0, num_classes - 1)
        pred_scores_for_gt = pred_scores.gather(2, gt_labels_for_gather)

        # Batched IoU: [B, N, M]
        pair_wise_ious = batch_bbox_iou(pred_bboxes, gt_bboxes)

        # Alignment metric: [B, N, M]
        alignment_metric = pred_scores_for_gt.pow(self.alpha) * pair_wise_ious.pow(self.beta)
        alignment_metric *= mask_in_gts.float()
        alignment_metric *= mask_gt.permute(0, 2, 1).float()

        # Vectorized top-K selection
        alignment_t = alignment_metric.permute(0, 2, 1)  # [B, M, N]
        k = min(self.topk, num_anchors)
        _, topk_idx = alignment_t.topk(k, dim=-1)  # [B, M, k]

        topk_mask_t = torch.zeros_like(alignment_t, dtype=torch.bool)
        topk_mask_t.scatter_(2, topk_idx, True)
        topk_mask = topk_mask_t.permute(0, 2, 1)  # [B, N, M]

        topk_mask &= alignment_metric > 0
        topk_metrics = torch.where(topk_mask, alignment_metric, torch.zeros_like(alignment_metric))

        # Resolve conflicts: pick GT with highest IoU per anchor (matches super-gradients)
        mask_pos = topk_mask  # [B, N, M]
        fg_mask = mask_pos.any(dim=-1)  # [B, N]
        masked_ious_for_conflict = torch.where(mask_pos, pair_wise_ious, torch.zeros_like(pair_wise_ious))
        _, max_gt_idx = masked_ious_for_conflict.max(dim=-1)  # [B, N]

        # Vectorized assignment using gather
        assigned_labels = torch.gather(gt_labels_expanded, 1, max_gt_idx)
        assigned_labels *= fg_mask.long()

        gt_idx_for_bboxes = max_gt_idx.unsqueeze(-1).expand(-1, -1, 4)
        assigned_bboxes = torch.gather(gt_bboxes, 1, gt_idx_for_bboxes)
        assigned_bboxes *= fg_mask.unsqueeze(-1).float()

        # Score normalization — vectorized over batch, loop over M only
        masked_ious = torch.where(mask_pos, pair_wise_ious, torch.zeros_like(pair_wise_ious))
        max_iou_per_gt = masked_ious.amax(dim=1)  # [B, M]
        max_metric_per_gt = topk_metrics.amax(dim=1)  # [B, M]
        norm_metrics = topk_metrics / (max_metric_per_gt.unsqueeze(1) + self.eps) * max_iou_per_gt.unsqueeze(1)

        assigned_scores = torch.zeros([batch_size, num_anchors, num_classes], device=device)
        for m in range(num_max_boxes):
            pos_m = mask_pos[:, :, m]  # [B, N]
            if not pos_m.any():
                continue
            class_idx = gt_labels_expanded[:, m].unsqueeze(1).expand(-1, num_anchors).unsqueeze(-1)  # [B, N, 1]
            current = assigned_scores.gather(2, class_idx).squeeze(-1)  # [B, N]
            updated = torch.where(pos_m, torch.max(current, norm_metrics[:, :, m]), current)
            assigned_scores.scatter_(2, class_idx, updated.unsqueeze(-1))

        return assigned_labels, assigned_bboxes, assigned_scores, fg_mask


# ---------------------------------------------------------------------------
# ATSS Assigner (static warmup)
# ---------------------------------------------------------------------------


class ATSSAssigner:
    """Adaptive Training Sample Selection assigner for warmup epochs.

    Selects positive anchors based on center distance and IoU statistics,
    providing more stable assignments than TAL when predictions are poor.
    """

    def __init__(self, topk: int = 9):
        self.topk = topk

    @torch.no_grad()
    def assign(
        self,
        anchor_bboxes: Tensor,
        num_anchors_list: list[int],
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
        num_classes: int,
        pred_bboxes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Assign ground truths to anchors using ATSS.

        Args:
            anchor_bboxes: ``[N, 4]`` anchor boxes (from grid cell generation).
            num_anchors_list: Number of anchors per stride level.
            gt_labels: ``[B, M, 1]`` class labels.
            gt_bboxes: ``[B, M, 4]`` ground truth boxes (x1y1x2y2).
            mask_gt: ``[B, M, 1]`` valid GT mask.
            num_classes: Number of object classes.
            pred_bboxes: ``[B, N, 4]`` predicted boxes (pixel coords). When provided,
                assigned scores are scaled by IoU(pred, GT) for prediction-quality weighting.

        Returns:
            assigned_labels, assigned_bboxes, assigned_scores, fg_mask
        """
        batch_size = gt_labels.shape[0]
        num_anchors = anchor_bboxes.shape[0]
        num_max_boxes = gt_bboxes.shape[1]
        device = gt_labels.device

        if num_max_boxes == 0:
            return (
                torch.zeros([batch_size, num_anchors], dtype=torch.long, device=device),
                torch.zeros([batch_size, num_anchors, 4], device=device),
                torch.zeros([batch_size, num_anchors, num_classes], device=device),
                torch.zeros([batch_size, num_anchors], dtype=torch.bool, device=device),
            )

        gt_labels_expanded = gt_labels.squeeze(-1).long()  # [B, M]

        # Anchor and GT centers
        anchor_centers = (anchor_bboxes[:, :2] + anchor_bboxes[:, 2:]) / 2  # [N, 2]
        gt_centers = (gt_bboxes[:, :, :2] + gt_bboxes[:, :, 2:]) / 2  # [B, M, 2]

        # L2 distances: [B, N, M]
        distances = torch.cdist(
            anchor_centers.unsqueeze(0).expand(batch_size, -1, -1), gt_centers
        )

        # Per-stride topk closest anchors for each GT
        candidate_mask = torch.zeros(
            [batch_size, num_anchors, num_max_boxes], dtype=torch.bool, device=device
        )
        start = 0
        for level_n in num_anchors_list:
            end = start + level_n
            level_dists = distances[:, start:end, :]  # [B, level_n, M]
            k = min(self.topk, level_n)
            _, topk_idx = level_dists.topk(k, dim=1, largest=False)  # [B, k, M]
            level_mask = torch.zeros(
                [batch_size, level_n, num_max_boxes], dtype=torch.bool, device=device
            )
            level_mask.scatter_(1, topk_idx, True)
            candidate_mask[:, start:end, :] = level_mask
            start = end

        # IoU between anchor boxes and GTs: [B, N, M]
        ious = batch_bbox_iou(
            anchor_bboxes.unsqueeze(0).expand(batch_size, -1, -1), gt_bboxes
        )

        # IoU-based adaptive threshold per GT: mean + std of candidate IoUs
        candidate_ious = ious * candidate_mask.float()
        candidate_count = candidate_mask.float().sum(dim=1).clamp(min=1)  # [B, M]
        candidate_mean = candidate_ious.sum(dim=1) / candidate_count  # [B, M]
        candidate_sq_mean = (candidate_ious.pow(2) * candidate_mask.float()).sum(dim=1) / candidate_count
        candidate_std = (candidate_sq_mean - candidate_mean.pow(2)).clamp(min=0).sqrt()
        iou_threshold = candidate_mean + candidate_std  # [B, M]

        # Positive: candidate AND IoU >= threshold AND center inside GT
        is_pos = candidate_mask & (ious >= iou_threshold.unsqueeze(1))

        ac = anchor_centers  # [N, 2]
        center_in_gt = (
            (ac[None, :, None, 0] >= gt_bboxes[:, None, :, 0])
            & (ac[None, :, None, 0] <= gt_bboxes[:, None, :, 2])
            & (ac[None, :, None, 1] >= gt_bboxes[:, None, :, 1])
            & (ac[None, :, None, 1] <= gt_bboxes[:, None, :, 3])
        )
        is_pos &= center_in_gt
        is_pos &= mask_gt.permute(0, 2, 1).bool()

        # Resolve conflicts: pick GT with highest IoU
        fg_mask = is_pos.any(dim=-1)  # [B, N]
        pos_ious = torch.where(is_pos, ious, torch.zeros_like(ious))
        _, max_gt_idx = pos_ious.max(dim=-1)  # [B, N]

        # Gather assignments
        assigned_labels = torch.gather(gt_labels_expanded, 1, max_gt_idx)
        assigned_labels *= fg_mask.long()

        gt_idx_for_bboxes = max_gt_idx.unsqueeze(-1).expand(-1, -1, 4)
        assigned_bboxes = torch.gather(gt_bboxes, 1, gt_idx_for_bboxes)
        assigned_bboxes *= fg_mask.unsqueeze(-1).float()

        # Assigned scores: IoU-based soft labels
        assigned_ious = torch.gather(ious, 2, max_gt_idx.unsqueeze(-1)).squeeze(-1)
        assigned_ious *= fg_mask.float()

        # Scale by prediction quality IoU(pred, GT) when available (matches SG)
        if pred_bboxes is not None:
            pred_ious = batch_bbox_iou(pred_bboxes, gt_bboxes)  # [B, N, M]
            pred_assigned_ious = torch.gather(pred_ious, 2, max_gt_idx.unsqueeze(-1)).squeeze(-1)
            pred_assigned_ious *= fg_mask.float()
            assigned_ious = assigned_ious * pred_assigned_ious

        assigned_scores = torch.zeros([batch_size, num_anchors, num_classes], device=device)
        class_idx = assigned_labels.unsqueeze(-1)  # [B, N, 1]
        assigned_scores.scatter_(2, class_idx, assigned_ious.unsqueeze(-1))
        assigned_scores *= fg_mask.unsqueeze(-1).float()

        return assigned_labels, assigned_bboxes, assigned_scores, fg_mask


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------


class VarifocalLoss(nn.Module):
    """Varifocal loss from VarifocalNet."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_score: Tensor, gt_score: Tensor, label: Tensor) -> Tensor:
        """
        Args:
            pred_score: ``[B, N, C]`` predicted logits (pre-sigmoid).
            gt_score: ``[B, N, C]`` soft target scores.
            label: ``[B, N, C]`` binary labels (1 for positive).
        """
        pred_sigmoid = pred_score.sigmoid()
        weight = self.alpha * pred_sigmoid.pow(self.gamma) * (1 - label) + gt_score * label
        bce = F.binary_cross_entropy_with_logits(pred_score, gt_score, reduction="none")
        return (weight * bce).sum()


class GIoULoss(nn.Module):
    """Generalized IoU loss. Returns per-element loss (no reduction)."""

    def forward(self, pred_bboxes: Tensor, target_bboxes: Tensor) -> Tensor:
        """
        Args:
            pred_bboxes: ``[N, 4]`` x1y1x2y2.
            target_bboxes: ``[N, 4]`` x1y1x2y2.

        Returns:
            ``[N]`` per-element GIoU loss.
        """
        # Intersection
        inter_x1 = torch.max(pred_bboxes[:, 0], target_bboxes[:, 0])
        inter_y1 = torch.max(pred_bboxes[:, 1], target_bboxes[:, 1])
        inter_x2 = torch.min(pred_bboxes[:, 2], target_bboxes[:, 2])
        inter_y2 = torch.min(pred_bboxes[:, 3], target_bboxes[:, 3])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # Union
        area1 = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
        area2 = (target_bboxes[:, 2] - target_bboxes[:, 0]) * (target_bboxes[:, 3] - target_bboxes[:, 1])
        union = area1 + area2 - inter

        iou = inter / (union + 1e-9)

        # Enclosing box
        enc_x1 = torch.min(pred_bboxes[:, 0], target_bboxes[:, 0])
        enc_y1 = torch.min(pred_bboxes[:, 1], target_bboxes[:, 1])
        enc_x2 = torch.max(pred_bboxes[:, 2], target_bboxes[:, 2])
        enc_y2 = torch.max(pred_bboxes[:, 3], target_bboxes[:, 3])
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

        giou = iou - (enc_area - union) / (enc_area + 1e-9)
        return 1.0 - giou


class DFLLoss(nn.Module):
    """Distribution Focal Loss for fine-grained box regression.

    Returns per-anchor loss (averaged over 4 coordinates).
    """

    def forward(self, pred_dist: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred_dist: ``[N, 4*(reg_max+1)]`` raw distribution predictions.
            target: ``[N, 4]`` target distances (float, in [0, reg_max] range).

        Returns:
            ``[N]`` per-anchor DFL loss (averaged over 4 coords).
        """
        num_anchors = pred_dist.shape[0]
        reg_max = pred_dist.shape[-1] // 4 - 1
        pred_dist = pred_dist.reshape(-1, reg_max + 1)
        target = target.reshape(-1)

        target_left = target.long().clamp(0, reg_max)
        target_right = (target_left + 1).clamp(0, reg_max)
        weight_right = target - target_left.float()
        weight_left = 1.0 - weight_right

        loss = (
            F.cross_entropy(pred_dist, target_left, reduction="none") * weight_left
            + F.cross_entropy(pred_dist, target_right, reduction="none") * weight_right
        )
        return loss.reshape(num_anchors, 4).mean(dim=-1)


# ---------------------------------------------------------------------------
# Combined PPYoloE loss
# ---------------------------------------------------------------------------


class PPYoloELoss(nn.Module):
    """Combined loss for YOLO-NAS training.

    Components:
    - VarifocalLoss (classification)
    - GIoULoss (box regression, weighted by assigned scores)
    - DFLLoss (distribution focal loss, weighted by assigned scores)

    Weighted sum: cls_weight * vfl + iou_weight * giou + dfl_weight * dfl
    All terms normalized by ``assigned_scores_sum`` (matching super-gradients).

    Args:
        num_classes: Number of object classes.
        reg_max: Distribution regression maximum.
        cls_weight: Classification loss weight.
        iou_weight: Box regression loss weight.
        dfl_weight: Distribution focal loss weight.
        static_assigner_epochs: Use ATSS for the first N epochs (0 to disable).
    """

    def __init__(
        self,
        num_classes: int = 80,
        reg_max: int = 16,
        cls_weight: float = 1.0,
        iou_weight: float = 2.5,
        dfl_weight: float = 0.5,
        static_assigner_epochs: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.cls_weight = cls_weight
        self.iou_weight = iou_weight
        self.dfl_weight = dfl_weight
        self.static_assigner_epochs = static_assigner_epochs

        self.assigner = TaskAlignedAssigner()
        self.static_assigner = ATSSAssigner(topk=9) if static_assigner_epochs > 0 else None
        self.vfl = VarifocalLoss()
        self.giou_loss = GIoULoss()
        self.dfl_loss = DFLLoss()

    def _bbox2dist(self, anchor_points: Tensor, gt_bboxes: Tensor) -> Tensor:
        """Convert bounding boxes to distances from anchor points."""
        x1y1 = anchor_points - gt_bboxes[..., :2]
        x2y2 = gt_bboxes[..., 2:] - anchor_points
        dist = torch.cat([x1y1, x2y2], dim=-1)
        return dist.clamp(0, self.reg_max - 0.01)

    def forward(
        self,
        predictions: tuple,
        targets: Tensor,
        input_size: tuple[int, int] | None = None,
        epoch: int | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute loss.

        Args:
            predictions: ``(decoded_predictions, raw_predictions)`` from NDFLHeads in training mode.
                decoded_predictions: ``(pred_bboxes [B,N,4], pred_scores [B,N,C])``
                raw_predictions: ``(cls_logits [B,N,C], reg_distri [B,N,4*(reg_max+1)],
                                   anchors, anchor_points, num_anchors_list, stride_tensor)``
            targets: ``[sum(N_i), 6]`` with ``[batch_idx, class_id, x, y, w, h]`` (normalized xywh).
            input_size: ``(H, W)`` of the input image. If None, inferred from anchor grid.
            epoch: Current training epoch (used for ATSS warmup).

        Returns:
            (total_loss, loss_dict)
        """
        (pred_bboxes_decoded, pred_scores_decoded), (
            cls_logits,
            reg_distri,
            anchors,
            anchor_points,
            num_anchors_list,
            stride_tensor,
        ) = predictions

        batch_size = cls_logits.shape[0]
        device = cls_logits.device

        # Determine input image size for scaling normalized GT to pixel coords
        if input_size is not None:
            img_h, img_w = input_size[0], input_size[1]
        else:
            inferred = (anchor_points.max(dim=0).values + stride_tensor.min() / 2).clamp(min=1)
            img_w, img_h = inferred[0], inferred[1]

        # Validate GT class labels
        if targets.numel() > 0:
            class_ids = targets[:, 1]
            if (class_ids < 0).any() or (class_ids >= self.num_classes).any():
                logger.warning(
                    "GT class labels out of range [0, %d): min=%d, max=%d. "
                    "Check your dataset labels.",
                    self.num_classes,
                    int(class_ids.min().item()),
                    int(class_ids.max().item()),
                )

        # Prepare GT in format expected by assigner
        gt_labels_list = []
        gt_bboxes_list = []
        for b in range(batch_size):
            mask = targets[:, 0] == b
            if mask.any():
                t = targets[mask]
                gt_labels_list.append(t[:, 1:2])
                xc, yc, w, h = t[:, 2], t[:, 3], t[:, 4], t[:, 5]
                xc, w = xc * img_w, w * img_w
                yc, h = yc * img_h, h * img_h
                gt_bboxes_list.append(torch.stack([
                    xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
                ], dim=-1))
            else:
                gt_labels_list.append(torch.zeros(0, 1, device=device))
                gt_bboxes_list.append(torch.zeros(0, 4, device=device))

        max_gt = max(len(g) for g in gt_labels_list)
        if max_gt == 0:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, {"cls_loss": 0.0, "iou_loss": 0.0, "dfl_loss": 0.0, "total_loss": 0.0}

        gt_labels = torch.zeros(batch_size, max_gt, 1, device=device)
        gt_bboxes = torch.zeros(batch_size, max_gt, 4, device=device)
        mask_gt = torch.zeros(batch_size, max_gt, 1, device=device)

        for b in range(batch_size):
            n = len(gt_labels_list[b])
            if n > 0:
                gt_labels[b, :n] = gt_labels_list[b]
                gt_bboxes[b, :n] = gt_bboxes_list[b]
                mask_gt[b, :n] = 1.0

        # Select assigner: ATSS for warmup, TAL after
        use_static = (
            self.static_assigner is not None
            and epoch is not None
            and epoch < self.static_assigner_epochs
        )

        if use_static:
            assigned_labels, assigned_bboxes, assigned_scores, fg_mask = self.static_assigner.assign(
                anchors, num_anchors_list, gt_labels, gt_bboxes, mask_gt, self.num_classes,
                pred_bboxes=pred_bboxes_decoded.detach(),
            )
        else:
            assigned_labels, assigned_bboxes, assigned_scores, fg_mask = self.assigner.assign(
                pred_scores_decoded, pred_bboxes_decoded, anchor_points,
                gt_labels, gt_bboxes, mask_gt,
            )

        # Normalization: sum of assigned soft scores (matches super-gradients)
        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        # Classification loss (VFL)
        cls_loss = self.vfl(cls_logits, assigned_scores, (assigned_scores > 0).float()) / assigned_scores_sum

        # Box regression loss (GIoU) — weighted by per-anchor assigned scores
        if fg_mask.any():
            pos_pred_bboxes = pred_bboxes_decoded[fg_mask]
            pos_target_bboxes = assigned_bboxes[fg_mask]
            bbox_weight = assigned_scores[fg_mask].sum(-1)  # [num_pos]
            iou_loss_per_anchor = self.giou_loss(pos_pred_bboxes, pos_target_bboxes)  # [num_pos]
            iou_loss = (iou_loss_per_anchor * bbox_weight).sum() / assigned_scores_sum
        else:
            iou_loss = torch.tensor(0.0, device=device)

        # DFL loss — weighted by per-anchor assigned scores
        if fg_mask.any():
            pos_reg_distri = reg_distri[fg_mask]
            pos_anchor_points = anchor_points.unsqueeze(0).expand(batch_size, -1, -1)[fg_mask]
            pos_stride = stride_tensor.unsqueeze(0).expand(batch_size, -1, -1)[fg_mask]
            pos_target_bboxes = assigned_bboxes[fg_mask] / pos_stride
            pos_anchor_points_scaled = pos_anchor_points / pos_stride
            target_dist = self._bbox2dist(pos_anchor_points_scaled, pos_target_bboxes)
            dfl_loss_per_anchor = self.dfl_loss(pos_reg_distri, target_dist)  # [num_pos]
            dfl_loss = (dfl_loss_per_anchor * bbox_weight).sum() / assigned_scores_sum
        else:
            dfl_loss = torch.tensor(0.0, device=device)

        total_loss = self.cls_weight * cls_loss + self.iou_weight * iou_loss + self.dfl_weight * dfl_loss

        loss_dict = {
            "cls_loss": cls_loss.item(),
            "iou_loss": iou_loss.item(),
            "dfl_loss": dfl_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict
