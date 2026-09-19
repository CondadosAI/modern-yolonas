"""Model preparation for quantization (PTQ and QAT).

Strategy: quantize backbone+neck only (95 % of FLOPs). The detection
heads stay in float32 because they contain dynamic control flow
(anchor generation, train/eval branching) that FX cannot trace.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn, Tensor
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx

from modern_yolonas.model import YoloNAS
from modern_yolonas.nn.repvgg import QARepVGGBlock
from modern_yolonas.quantization.config import default_qconfig_mapping, qat_qconfig_mapping


class QuantizableBackboneNeck(nn.Module):
    """FX-traceable wrapper around backbone + neck.

    After RepVGG partial fusion the backbone and neck are purely
    sequential Conv+BN+ReLU chains that FX can symbolically trace.
    """

    def __init__(self, backbone: nn.Module, neck: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.backbone(x)
        p3, p4, p5 = self.neck(features)
        return p3, p4, p5


class QuantizedYoloNAS(nn.Module):
    """YoloNAS with quantized backbone+neck and float32 heads."""

    def __init__(
        self,
        quantized_backbone_neck: nn.Module,
        heads: nn.Module,
    ):
        super().__init__()
        self.backbone_neck = quantized_backbone_neck
        self.heads = heads

    def forward(self, x: Tensor):
        p3, p4, p5 = self.backbone_neck(x)
        return self.heads((p3, p4, p5))


def fuse_repvgg_for_quantization(model: nn.Module) -> nn.Module:
    """Partially fuse all QARepVGGBlocks in *model* (in-place).

    After partial fusion each block becomes::

        rbr_reparam (Conv2d) -> post_bn (BN) -> nonlinearity (ReLU)

    This Conv+BN+ReLU pattern is what FX quantization recognises and
    fuses automatically during conversion.
    """
    for module in model.modules():
        if isinstance(module, QARepVGGBlock) and not module.partially_fused:
            module.partial_fusion()
    return model


def prepare_model_ptq(
    model: YoloNAS,
    qconfig_mapping: QConfigMapping | None = None,
    backend: str = "x86",
    example_input: Tensor | None = None,
) -> QuantizedYoloNAS:
    """Prepare a YoloNAS model for Post-Training Quantization.

    The original *model* is **not** modified (deep-copied internally).

    Args:
        model: YoloNAS model.
        qconfig_mapping: Override quantization config.
        backend: ``'x86'``, ``'qnnpack'``, or ``'onednn'``.
        example_input: Example tensor for FX tracing (default ``[1,3,640,640]``).

    Returns:
        :class:`QuantizedYoloNAS` with observer nodes in backbone+neck.
    """
    model = deepcopy(model)
    model.eval()

    fuse_repvgg_for_quantization(model)

    if qconfig_mapping is None:
        qconfig_mapping = default_qconfig_mapping(backend)
    if example_input is None:
        example_input = torch.randn(1, 3, 640, 640)

    backbone_neck = QuantizableBackboneNeck(model.backbone, model.neck)
    backbone_neck.eval()

    torch.backends.quantized.engine = backend
    prepared = prepare_fx(
        backbone_neck,
        qconfig_mapping,
        example_inputs=(example_input,),
    )

    return QuantizedYoloNAS(prepared, model.heads)


def prepare_model_qat(
    model: YoloNAS,
    qconfig_mapping: QConfigMapping | None = None,
    backend: str = "x86",
    example_input: Tensor | None = None,
) -> QuantizedYoloNAS:
    """Prepare a YoloNAS model for Quantization-Aware Training.

    The original *model* is **not** modified (deep-copied internally).

    .. warning::
        AMP must be **disabled** during QAT fine-tuning because
        fake-quant nodes conflict with autocast.

    Args:
        model: YoloNAS model.
        qconfig_mapping: Override quantization config.
        backend: ``'x86'``, ``'qnnpack'``, or ``'onednn'``.
        example_input: Example tensor for FX tracing (default ``[1,3,640,640]``).

    Returns:
        :class:`QuantizedYoloNAS` with fake-quant nodes, in train mode.
    """
    model = deepcopy(model)

    fuse_repvgg_for_quantization(model)

    if qconfig_mapping is None:
        qconfig_mapping = qat_qconfig_mapping(backend)
    if example_input is None:
        example_input = torch.randn(1, 3, 640, 640)

    backbone_neck = QuantizableBackboneNeck(model.backbone, model.neck)
    backbone_neck.train()

    torch.backends.quantized.engine = backend
    prepared = prepare_qat_fx(
        backbone_neck,
        qconfig_mapping,
        example_inputs=(example_input,),
    )

    qat_model = QuantizedYoloNAS(prepared, model.heads)
    qat_model.train()
    return qat_model
