"""ONNX export.

One function, so that the CLI, the export zoo script and the OpenVINO and TensorRT
paths all produce the same graph rather than three graphs that happen to agree.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

__all__ = ["fuse_for_inference", "export_onnx", "onnx_session"]


def fuse_for_inference(model: nn.Module) -> nn.Module:
    """Collapse the RepVGG training branches into a single conv, in place.

    Every export path needs this: an unfused graph exports and runs, it is just two
    to three times the convolutions for identical outputs.
    """
    for module in model.modules():
        if hasattr(module, "fuse_block_residual_branches"):
            module.fuse_block_residual_branches()
    return model.eval()


def export_onnx(
    model: nn.Module,
    path: str | Path,
    input_size: int = 640,
    opset: int = 18,
    dynamic_batch: bool = True,
    half: bool = False,
    external_data: bool = False,
    fuse: bool = True,
) -> Path:
    """Export to ONNX at a fixed spatial size.

    Args:
        model: A YOLO-NAS model. Exported as-is, so a fine-tuned head exports with
            its own class count.
        path: Destination ``.onnx`` file.
        input_size: Square input side. The spatial dimensions are baked in — the head
            builds its anchor grid from them, so one file serves one resolution.
        opset: ONNX opset. 18 is the lowest the current PyTorch exporter emits
            natively; asking for less makes it convert down afterwards.
        dynamic_batch: Leave the batch dimension symbolic. Set ``False`` for runtimes
            that specialise on a static shape — TensorRT otherwise needs an
            optimisation profile, and OpenVINO's INT8 calibration is simpler.
        half: Export FP16 weights and an FP16 input. TensorRT 11 is always strongly
            typed — its FP16 builder flag is gone — so an FP16 engine can only come
            from an FP16 graph. OpenVINO instead compresses at conversion time, and
            ONNX Runtime's CPU provider has no use for it, so leave this off there.
        external_data: Write the weights to a ``.onnx.data`` sidecar instead of into
            the file. PyTorch's exporter defaults this on; this function defaults it
            off, because a published artifact that silently needs a second file beside
            it is a support burden, and TensorRT's ``parser.parse(bytes)`` cannot
            resolve the sidecar at all. Only turn it on above the 2 GB protobuf
            limit, which no YOLO-NAS variant approaches.
        fuse: Run :func:`fuse_for_inference` first. Only turn off to export a graph
            that is meant to keep its training-time branches.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fuse:
        fuse_for_inference(model)
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)
    if half:
        model = model.half()
        dummy = dummy.half()
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch"} for name in ("images", "pred_bboxes", "pred_scores")}

    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["images"],
        output_names=["pred_bboxes", "pred_scores"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        external_data=external_data,
    )
    return path


def onnx_session(path: str | Path, provider: str = "cpu"):
    """An ONNX Runtime session that fails loudly instead of falling back to the CPU.

    ``ort.get_available_providers()`` reports what the *build* supports; a session can
    still resolve to the CPU because a CUDA shared library did not load, and ONNX
    Runtime reports that as a warning on stderr and nowhere else. Asserting on
    ``session.get_providers()`` is what turns that into an error.

    Args:
        path: ONNX file.
        provider: ``"cpu"``, ``"cuda"`` or ``"tensorrt"``.

    Returns:
        An ``onnxruntime.InferenceSession``.

    Raises:
        ValueError: On an unknown provider name.
        RuntimeError: If the requested provider is not the one the session resolved to.
    """
    # torch first, deliberately: it dlopens the CUDA libraries shipped in the venv's
    # nvidia-* wheels, which onnxruntime does not know to look for.
    import torch  # noqa: F401
    import onnxruntime as ort

    wanted = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }
    if provider not in wanted:
        raise ValueError(f"unknown provider {provider!r}; expected one of {sorted(wanted)}")

    order = [wanted[provider]]
    if provider != "cpu":
        order.append("CPUExecutionProvider")  # ORT requires a fallback in the list

    session = ort.InferenceSession(str(path), providers=order)
    resolved = session.get_providers()
    if wanted[provider] not in resolved:
        raise RuntimeError(
            f"asked for {wanted[provider]}, session resolved to {resolved}. "
            "Check stderr above for the library that failed to load, and that exactly one of "
            "onnxruntime / onnxruntime-gpu is installed."
        )
    return session
