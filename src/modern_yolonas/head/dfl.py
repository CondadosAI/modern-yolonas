"""YoloNAS DFL (Distribution Focal Loss) detection heads.

Attribute names ``stem``, ``cls_convs``, ``reg_convs``, ``cls_pred``, ``reg_pred``
and ``head1`` … ``headN`` mirror super-gradients for state_dict key compatibility.
"""

from __future__ import annotations

import math

import torch
from torch import nn, Tensor

from modern_yolonas.nn.blocks import ConvBNReLU, width_multiplier


class YoloNASDFLHead(nn.Module):
    """Per-scale detection head: stem → cls branch + reg branch.

    Attributes: ``stem``, ``cls_convs``, ``reg_convs``, ``cls_pred``, ``reg_pred``,
    ``cls_dropout_rate``, ``reg_dropout_rate``.
    """

    def __init__(
        self,
        in_channels: int,
        inter_channels: int,
        width_mult: float,
        first_conv_group_size: int,
        num_classes: int,
        stride: int,
        reg_max: int,
        cls_dropout_rate: float = 0.0,
        reg_dropout_rate: float = 0.0,
    ):
        super().__init__()

        inter_channels = width_multiplier(inter_channels, width_mult, 8)
        if first_conv_group_size == 0:
            groups = 0
        elif first_conv_group_size == -1:
            groups = 1
        else:
            groups = inter_channels // first_conv_group_size

        self.num_classes = num_classes
        self.stride = stride

        self.stem = ConvBNReLU(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False)

        first_cls_conv = (
            [ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)]
            if groups
            else []
        )
        self.cls_convs = nn.Sequential(
            *first_cls_conv,
            ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        first_reg_conv = (
            [ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)]
            if groups
            else []
        )
        self.reg_convs = nn.Sequential(
            *first_reg_conv,
            ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.cls_pred = nn.Conv2d(inter_channels, num_classes, 1, 1, 0)
        self.reg_pred = nn.Conv2d(inter_channels, 4 * (reg_max + 1), 1, 1, 0)

        self.cls_dropout_rate = nn.Dropout2d(cls_dropout_rate) if cls_dropout_rate > 0 else nn.Identity()
        self.reg_dropout_rate = nn.Dropout2d(reg_dropout_rate) if reg_dropout_rate > 0 else nn.Identity()

        self.prior_prob = 1e-2
        self._initialize_biases()

    def _initialize_biases(self):
        prior_bias = -math.log((1 - self.prior_prob) / self.prior_prob)
        nn.init.constant_(self.cls_pred.bias, prior_bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.stem(x)

        cls_feat = self.cls_convs(x)
        cls_feat = self.cls_dropout_rate(cls_feat)
        cls_output = self.cls_pred(cls_feat)

        reg_feat = self.reg_convs(x)
        reg_feat = self.reg_dropout_rate(reg_feat)
        reg_output = self.reg_pred(reg_feat)

        return reg_output, cls_output


def _batch_distance2bbox(points: Tensor, distance: Tensor) -> Tensor:
    """Convert distance predictions (l, t, r, b) to bounding boxes (x1, y1, x2, y2)."""
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


@torch.no_grad()
def _generate_anchors_for_grid_cell(
    feats: tuple[Tensor, ...],
    fpn_strides: tuple[int, ...],
    grid_cell_scale: float = 5.0,
    grid_cell_offset: float = 0.5,
    dtype: torch.dtype = torch.float,
) -> tuple[Tensor, Tensor, list[int], Tensor]:
    """Generate ATSS-style anchors for grid cells."""
    device = feats[0].device
    anchors = []
    anchor_points = []
    num_anchors_list = []
    stride_tensor = []

    for feat, stride in zip(feats, fpn_strides):
        _, _, h, w = feat.shape
        cell_half_size = grid_cell_scale * stride * 0.5
        shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
        shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")

        anchor = torch.stack(
            [shift_x - cell_half_size, shift_y - cell_half_size, shift_x + cell_half_size, shift_y + cell_half_size],
            dim=-1,
        ).to(dtype=dtype)
        anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype)

        anchors.append(anchor.reshape(-1, 4))
        anchor_points.append(anchor_point.reshape(-1, 2))
        num_anchors_list.append(anchors[-1].shape[0])
        stride_tensor.append(torch.full([num_anchors_list[-1], 1], stride, dtype=dtype, device=device))

    return torch.cat(anchors), torch.cat(anchor_points), num_anchors_list, torch.cat(stride_tensor)


class NDFLHeads(nn.Module):
    """Multi-scale DFL heads wrapper.

    Attributes: ``head1``, ``head2``, ``head3``, ``proj_conv`` (buffer).
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: tuple[int, int, int],
        heads_list: list[dict],
        grid_cell_scale: float = 5.0,
        grid_cell_offset: float = 0.5,
        reg_max: int = 16,
        eval_size: tuple[int, int] | None = None,
        width_mult: float = 1.0,
    ):
        super().__init__()

        in_channels = tuple(max(round(c * width_mult), 1) for c in in_channels)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.eval_size = eval_size

        # When True, ``forward`` also returns the raw (undecoded) predictions in eval
        # mode. Validation needs them to compute the loss *without* putting BatchNorm
        # into training mode, which would update running statistics from the validation
        # set. Tracing always ignores this and returns decoded outputs only.
        self.return_raw_outputs = False

        proj = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape(1, self.reg_max + 1, 1, 1)
        self.register_buffer("proj_conv", proj, persistent=False)

        self.num_heads = len(heads_list)
        fpn_strides: list[int] = []
        for i, head_cfg in enumerate(heads_list):
            head = YoloNASDFLHead(
                in_channels=in_channels[i],
                num_classes=num_classes,
                reg_max=reg_max,
                **head_cfg,
            )
            fpn_strides.append(head.stride)
            setattr(self, f"head{i + 1}", head)

        self.fpn_strides = tuple(fpn_strides)

        # Cache anchors for eval
        if self.eval_size:
            self._init_anchors()

    def _init_anchors(self):
        dtype = torch.float32
        anchor_points, stride_tensor = self._generate_anchors(dtype=dtype, device=torch.device("cpu"))
        self.register_buffer("anchor_points", anchor_points, persistent=False)
        self.register_buffer("stride_tensor", stride_tensor, persistent=False)

    def _generate_anchors(
        self,
        feats: tuple[Tensor, ...] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        anchor_points = []
        stride_tensor = []

        if feats is not None:
            dtype = dtype or feats[0].dtype
            device = device or feats[0].device
        else:
            dtype = dtype or torch.float32
            device = device or torch.device("cpu")

        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)

            shift_x = torch.arange(end=w, dtype=torch.float32, device=device) + self.grid_cell_offset
            shift_y = torch.arange(end=h, dtype=torch.float32, device=device) + self.grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")

            anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype)
            anchor_points.append(anchor_point.reshape(-1, 2))
            stride_tensor.append(torch.full([h * w, 1], stride, dtype=dtype, device=device))

        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def forward(self, feats: tuple[Tensor, ...]) -> tuple[tuple[Tensor, Tensor], ...]:
        feats = feats[: self.num_heads]
        cls_score_list, reg_distri_list, reg_dist_reduced_list = [], [], []

        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            height_mul_width = h * w
            reg_distri, cls_logit = getattr(self, f"head{i + 1}")(feat)
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1]))

            reg_dist_reduced = torch.permute(
                reg_distri.reshape(-1, 4, self.reg_max + 1, height_mul_width),
                [0, 2, 3, 1],
            )
            reg_dist_reduced = torch.nn.functional.softmax(reg_dist_reduced, dim=1) * self.proj_conv
            reg_dist_reduced = reg_dist_reduced.sum(dim=1, keepdim=False)

            cls_score_list.append(cls_logit.reshape(b, self.num_classes, height_mul_width))
            reg_dist_reduced_list.append(reg_dist_reduced)

        cls_score_list = torch.cat(cls_score_list, dim=-1)
        cls_score_list = torch.permute(cls_score_list, [0, 2, 1])

        reg_distri_list = torch.cat(reg_distri_list, dim=1)
        reg_dist_reduced_list = torch.cat(reg_dist_reduced_list, dim=1)

        # Generate or use cached anchors
        if self.eval_size and hasattr(self, "anchor_points"):
            anchor_points_inference = self.anchor_points
            stride_tensor = self.stride_tensor
        else:
            anchor_points_inference, stride_tensor = self._generate_anchors(feats)

        pred_scores = cls_score_list.sigmoid()
        pred_bboxes = _batch_distance2bbox(anchor_points_inference, reg_dist_reduced_list) * stride_tensor
        decoded_predictions = pred_bboxes, pred_scores

        if torch.jit.is_tracing() or not (self.training or self.return_raw_outputs):
            return decoded_predictions

        # Training (or eval with ``return_raw_outputs``): also return raw predictions
        # for loss computation.
        anchors, anchor_points, num_anchors_list, stride_tensor_raw = _generate_anchors_for_grid_cell(
            feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset
        )
        raw_predictions = cls_score_list, reg_distri_list, anchors, anchor_points, num_anchors_list, stride_tensor_raw
        return decoded_predictions, raw_predictions
