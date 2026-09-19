"""OpenVINO IR export, optionally quantised to INT8 with NNCF."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

__all__ = ["export_openvino", "calibration_batches"]


def calibration_batches(
    image_dir: str | Path, input_size: int, limit: int = 128
) -> list[np.ndarray]:
    """Real images, preprocessed exactly as inference does, for PTQ calibration.

    Calibrating on random noise puts the activation ranges somewhere the model never
    goes, which costs several AP for no reason. Any directory of representative
    images works; COCO ``val2017`` is the obvious one for the pretrained weights.
    """
    import cv2

    from modern_yolonas.inference.preprocess import preprocess

    paths = sorted(
        p for p in Path(image_dir).expanduser().iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise ValueError(f"no images found in {image_dir}")

    # Spread the sample over the directory rather than taking the first N, which on a
    # sorted COCO listing is a contiguous slice of image ids.
    step = max(1, len(paths) // limit)
    batches = []
    for path in paths[::step][:limit]:
        image = cv2.imread(str(path))
        if image is None:
            continue
        tensor, _, _ = preprocess(image, input_size)
        batches.append(tensor.numpy().astype(np.float32))
    return batches


# The head's decode tail — DFL softmax, the anchor add/subtract, the stack back into
# boxes — is elementwise arithmetic on a few thousand values. Quantizing it buys
# nothing measurable, and OpenVINO's low-precision transformations throw on the
# dequantization it produces there:
#
#   [ReshapeTransformation] ... opset1::Multiply .../DequantizationMultiply
#   [0]:f32[1,2100,2100] -> (f32[1,2100,1]) CALLBACK HAS THROWN
#
# (that [1, 2100, 2100] intermediate exists only after quantization — the FP32 graph's
# largest activation is 1.2M elements). Leaving these op types alone keeps INT8 where
# the compute is, in the convolutions.
_ARITHMETIC_TAIL = ["Add", "Subtract", "Multiply", "Concat", "ReduceSum", "Softmax", "Reshape"]


def _quantize(model, batches: list[np.ndarray]):
    import nncf

    dataset = nncf.Dataset(batches, lambda batch: {0: batch})
    return nncf.quantize(
        model,
        dataset,
        subset_size=len(batches),
        ignored_scope=nncf.IgnoredScope(types=_ARITHMETIC_TAIL),
    )


def export_openvino(
    onnx_path: str | Path,
    path: str | Path,
    precision: str = "fp32",
    calibration_dir: str | Path | None = None,
    input_size: int = 640,
    calibration_images: int = 128,
) -> Path:
    """Convert an ONNX file to OpenVINO IR.

    Args:
        onnx_path: Source ONNX, as written by :func:`~modern_yolonas.export.onnx.export_onnx`.
        path: Destination ``.xml``; the ``.bin`` lands beside it.
        precision: ``"fp32"``, ``"fp16"`` (weights compressed, the runtime still picks
            its own execution precision per device) or ``"int8"``.
        calibration_dir: Directory of images for INT8 calibration. Required for
            ``"int8"``.
        input_size: Input side used to preprocess the calibration images. Must match
            the ONNX file's.
        calibration_images: How many images to calibrate on.

    Returns:
        The ``.xml`` path written.

    Raises:
        ValueError: On an unknown precision, or INT8 without a calibration directory.
    """
    import openvino as ov

    precision = precision.lower()
    if precision not in {"fp32", "fp16", "int8"}:
        raise ValueError(f"unknown precision {precision!r}; expected fp32, fp16 or int8")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model = ov.convert_model(str(onnx_path))

    if precision == "int8":
        if calibration_dir is None:
            raise ValueError("int8 export needs --calibration-dir: a directory of representative images")
        model = _quantize(model, calibration_batches(calibration_dir, input_size, calibration_images))

    ov.save_model(model, str(path), compress_to_fp16=(precision == "fp16"))
    return path


def available_devices() -> Iterator[str]:
    """Device names this OpenVINO install can compile for (``CPU``, ``GPU.0``, ...)."""
    import openvino as ov

    yield from ov.Core().available_devices
