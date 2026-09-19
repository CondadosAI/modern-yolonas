"""Export a trained model to the runtimes it will actually be deployed on.

Every path starts from the same ONNX graph, so a difference between two runtimes is
a difference in the runtime rather than in how each one was exported.
"""

from modern_yolonas.export.onnx import export_onnx, fuse_for_inference, onnx_session

__all__ = [
    "export_onnx",
    "fuse_for_inference",
    "onnx_session",
    "export_openvino",
    "build_engine",
    "EngineRunner",
]


def __getattr__(name: str):
    # OpenVINO and TensorRT are optional extras. Importing them eagerly would make
    # `from modern_yolonas.export import export_onnx` fail on a base install.
    if name == "export_openvino":
        from modern_yolonas.export.openvino import export_openvino

        return export_openvino
    if name in {"build_engine", "EngineRunner"}:
        from modern_yolonas.export import tensorrt

        return getattr(tensorrt, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
