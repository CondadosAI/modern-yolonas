"""Quantization configuration presets for YOLO-NAS."""

from __future__ import annotations

import torch
from torch import nn
from torch.ao.quantization import QConfig, QConfigMapping
from torch.ao.quantization.observer import (
    HistogramObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    default_per_channel_weight_observer,
    default_weight_observer,
)
from torch.ao.quantization.fake_quantize import FakeQuantize


def default_qconfig_mapping(backend: str = "x86") -> QConfigMapping:
    """QConfigMapping for PTQ.

    Uses HistogramObserver for activations (better range estimation)
    and per-channel weight observers for weights.  ConvTranspose2d gets
    per-tensor weights because PyTorch does not support per-channel
    quantization for transposed convolutions.

    Args:
        backend: Quantization backend (``'x86'``, ``'qnnpack'``, ``'onednn'``).
    """
    activation_observer = HistogramObserver.with_args(
        dtype=torch.quint8, qscheme=torch.per_tensor_affine
    )

    if backend == "qnnpack":
        weight_observer = default_weight_observer
    else:
        weight_observer = default_per_channel_weight_observer

    qconfig = QConfig(activation=activation_observer, weight=weight_observer)

    # ConvTranspose2d needs per-tensor weights (per-channel not supported)
    conv_transpose_qconfig = QConfig(
        activation=activation_observer, weight=default_weight_observer
    )

    return (
        QConfigMapping()
        .set_global(qconfig)
        .set_object_type(nn.ConvTranspose2d, conv_transpose_qconfig)
    )


def _per_tensor_weight_fq():
    """FakeQuantize for per-tensor symmetric int8 weights."""
    return FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
    )


def qat_qconfig_mapping(backend: str = "x86") -> QConfigMapping:
    """QConfigMapping for QAT.

    Uses ``FakeQuantize`` with per-tensor activations and per-channel
    weights.  ConvTranspose2d uses per-tensor weights.

    Args:
        backend: Quantization backend (``'x86'``, ``'qnnpack'``, ``'onednn'``).
    """
    activation_fq = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    )

    per_tensor_weight_fq = _per_tensor_weight_fq()

    if backend == "qnnpack":
        weight_fq = per_tensor_weight_fq
    else:
        weight_fq = FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        )

    qconfig = QConfig(activation=activation_fq, weight=weight_fq)

    # ConvTranspose2d needs per-tensor weights (per-channel not supported)
    conv_transpose_qconfig = QConfig(
        activation=activation_fq, weight=per_tensor_weight_fq
    )

    return (
        QConfigMapping()
        .set_global(qconfig)
        .set_object_type(nn.ConvTranspose2d, conv_transpose_qconfig)
    )
