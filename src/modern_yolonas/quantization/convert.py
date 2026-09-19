"""Convert prepared/QAT models to quantized form and export."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch.ao.quantization.quantize_fx import convert_fx

from modern_yolonas.quantization.prepare import QuantizedYoloNAS


def convert_quantized(
    model: QuantizedYoloNAS,
    inplace: bool = False,
) -> QuantizedYoloNAS:
    """Convert a prepared model to final quantized integer operations.

    For PTQ: observers become quantize/dequantize nodes with fixed scales.
    For QAT: fake-quant nodes become real quantized operations.

    Args:
        model: :class:`QuantizedYoloNAS` after calibration (PTQ) or
            fine-tuning (QAT).
        inplace: Modify in place instead of deep-copying.

    Returns:
        :class:`QuantizedYoloNAS` with int8 quantized backbone+neck.
    """
    if not inplace:
        model = deepcopy(model)

    model.eval()
    quantized_backbone_neck = convert_fx(model.backbone_neck)
    return QuantizedYoloNAS(quantized_backbone_neck, model.heads)


def export_quantized_onnx(
    model: QuantizedYoloNAS,
    output_path: str,
    input_size: int = 640,
    opset: int = 17,
) -> None:
    """Export a quantized model to ONNX with QDQ nodes.

    For best deployment compatibility, export the **prepared** model
    (before :func:`convert_quantized`) — the fake-quant / observer
    nodes become ``QuantizeLinear`` / ``DequantizeLinear`` operators
    that TensorRT and OpenVINO natively optimise.

    Args:
        model: :class:`QuantizedYoloNAS` (prepared or converted).
        output_path: ``.onnx`` file path.
        input_size: Spatial input size.
        opset: ONNX opset (17+ recommended for quant ops).
    """
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["images"],
        output_names=["pred_bboxes", "pred_scores"],
        dynamic_axes={
            "images": {0: "batch"},
            "pred_bboxes": {0: "batch"},
            "pred_scores": {0: "batch"},
        },
        opset_version=opset,
    )
