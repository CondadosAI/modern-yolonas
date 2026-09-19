"""ONNX graph surgery to produce Frigate-compatible models.

Transforms a base YOLO-NAS ONNX graph by inserting:
  1. Preprocessing — uint8 BGR → float32 RGB /255
  2. NMS — ONNX NonMaxSuppression op
  3. Output formatting — flat [D, 7] tensor: [batch, x1, y1, x2, y2, conf, class]
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_constant(name: str, value: np.ndarray) -> onnx.NodeProto:
    """Create a Constant node that produces *value* as a tensor."""
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=numpy_helper.from_array(value, name=name),
    )


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
    graph = model.graph

    # Identify the original model input / outputs
    orig_input_name = graph.input[0].name  # "images"
    orig_input_shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]

    bbox_output_name = graph.output[0].name  # "pred_bboxes"
    score_output_name = graph.output[1].name  # "pred_scores"

    # ------------------------------------------------------------------
    # Phase 1 — Preprocessing: uint8 BGR → float32 RGB /255
    # ------------------------------------------------------------------
    new_input = helper.make_tensor_value_info(
        "images_uint8", TensorProto.UINT8, orig_input_shape
    )

    preproc_nodes = [
        # Cast uint8 → float32
        helper.make_node("Cast", ["images_uint8"], ["images_float"], to=TensorProto.FLOAT),
        # Divide by 255
        make_constant("div_const", np.array(255.0, dtype=np.float32)),
        helper.make_node("Div", ["images_float", "div_const"], ["images_norm"]),
        # BGR → RGB channel swap: Gather along axis=1 with indices [2, 1, 0]
        make_constant("bgr_indices", np.array([2, 1, 0], dtype=np.int64)),
        helper.make_node("Gather", ["images_norm", "bgr_indices"], [orig_input_name], axis=1),
    ]

    # ------------------------------------------------------------------
    # Phase 2 — NMS
    # ------------------------------------------------------------------
    nms_nodes = [
        # Transpose scores from [B, N, C] → [B, C, N] for the NMS op
        make_constant("score_perm", np.array([0, 2, 1], dtype=np.int64)),
        helper.make_node("Transpose", [score_output_name], ["scores_nms"], perm=[0, 2, 1]),
        # NMS constants
        make_constant("max_det", np.array([max_detections], dtype=np.int64)),
        make_constant("iou_thr", np.array([iou_threshold], dtype=np.float32)),
        make_constant("conf_thr", np.array([conf_threshold], dtype=np.float32)),
        # NonMaxSuppression → selected_indices [D, 3] with [batch_idx, class_idx, box_idx]
        helper.make_node(
            "NonMaxSuppression",
            [bbox_output_name, "scores_nms", "max_det", "iou_thr", "conf_thr"],
            ["selected_indices"],
        ),
    ]

    # ------------------------------------------------------------------
    # Phase 3 — Output formatting to [D, 7]
    # ------------------------------------------------------------------
    fmt_nodes = [
        # Extract columns: batch_idx (col 0), class_idx (col 1), box_idx (col 2)
        make_constant("idx_0", np.array(0, dtype=np.int64)),
        make_constant("idx_1", np.array(1, dtype=np.int64)),
        make_constant("idx_2", np.array(2, dtype=np.int64)),
        helper.make_node("Gather", ["selected_indices", "idx_0"], ["batch_col"], axis=1),
        helper.make_node("Gather", ["selected_indices", "idx_1"], ["class_col"], axis=1),
        helper.make_node("Gather", ["selected_indices", "idx_2"], ["box_col"], axis=1),
        # Unsqueeze each [D] → [D, 1]
        make_constant("unsq_axis", np.array([1], dtype=np.int64)),
        helper.make_node("Unsqueeze", ["batch_col", "unsq_axis"], ["batch_2d"]),
        helper.make_node("Unsqueeze", ["class_col", "unsq_axis"], ["class_2d"]),
        helper.make_node("Unsqueeze", ["box_col", "unsq_axis"], ["box_2d"]),
        # --- Gather selected boxes via GatherND ---
        # Build [D, 2] indices: [batch_idx, box_idx]
        helper.make_node("Concat", ["batch_2d", "box_2d"], ["bbox_gather_idx"], axis=1),
        helper.make_node("GatherND", [bbox_output_name, "bbox_gather_idx"], ["selected_boxes"]),
        # --- Gather selected scores via GatherND ---
        # Build [D, 3] indices: [batch_idx, box_idx, class_idx]
        helper.make_node(
            "Concat", ["batch_2d", "box_2d", "class_2d"], ["score_gather_idx"], axis=1
        ),
        helper.make_node("GatherND", [score_output_name, "score_gather_idx"], ["selected_scores"]),
        # Unsqueeze scores [D] → [D, 1]
        helper.make_node("Unsqueeze", ["selected_scores", "unsq_axis"], ["scores_2d"]),
        # Cast batch and class indices to float32
        helper.make_node("Cast", ["batch_2d"], ["batch_float"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["class_2d"], ["class_float"], to=TensorProto.FLOAT),
        # Final concat: [batch, x1, y1, x2, y2, conf, class] → [D, 7]
        helper.make_node(
            "Concat",
            ["batch_float", "selected_boxes", "scores_2d", "class_float"],
            ["detections"],
            axis=1,
        ),
    ]

    # ------------------------------------------------------------------
    # Rebuild the graph
    # ------------------------------------------------------------------
    all_nodes = preproc_nodes + list(graph.node) + nms_nodes + fmt_nodes

    new_output = helper.make_tensor_value_info("detections", TensorProto.FLOAT, [None, 7])

    new_graph = helper.make_graph(
        all_nodes,
        graph.name,
        [new_input],
        [new_output],
        initializer=list(graph.initializer),
    )

    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version

    onnx.checker.check_model(new_model)
    onnx.save(new_model, output_path)
