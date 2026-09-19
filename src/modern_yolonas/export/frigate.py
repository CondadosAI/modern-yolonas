"""ONNX graph surgery to produce Frigate-compatible models.

Frigate hands the detector a fixed-size uint8 BGR frame and expects detections back,
so this target is the generic end-to-end graph (:mod:`modern_yolonas.export.nms`)
with one extra phase in front: uint8 BGR → float32 RGB / 255.

Frigate does its own resize, which is why baking the colour conversion in is safe
here and baking a letterbox in is not — see the module docstring in ``nms.py``.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from modern_yolonas.export.nms import append_nms, make_constant


def _preproc_nodes(input_name: str) -> list[onnx.NodeProto]:
    """uint8 BGR ``images_uint8`` → the float32 RGB tensor the base graph expects."""
    return [
        helper.make_node("Cast", ["images_uint8"], ["images_float"], to=TensorProto.FLOAT),
        make_constant("div_const", np.array(255.0, dtype=np.float32)),
        helper.make_node("Div", ["images_float", "div_const"], ["images_norm"]),
        # BGR → RGB: gather channels in reverse along axis 1.
        make_constant("bgr_indices", np.array([2, 1, 0], dtype=np.int64)),
        helper.make_node("Gather", ["images_norm", "bgr_indices"], [input_name], axis=1),
    ]


def make_frigate_onnx(
    base_onnx_path: str,
    output_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 20,
) -> None:
    """Rewrite a base YOLO-NAS ONNX into a Frigate-compatible graph.

    The resulting model accepts ``uint8 [1, 3, H, W]`` BGR input and returns a
    single ``float32 [D, 7]`` tensor with columns
    ``[batch_index, x_min, y_min, x_max, y_max, confidence, class_id]``.
    """
    model = onnx.load(base_onnx_path)
    input_name = model.graph.input[0].name
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]

    new_model = append_nms(
        model,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        extra_inputs=[helper.make_tensor_value_info("images_uint8", TensorProto.UINT8, input_shape)],
        prefix_nodes=_preproc_nodes(input_name),
    )
    onnx.save(new_model, output_path)
