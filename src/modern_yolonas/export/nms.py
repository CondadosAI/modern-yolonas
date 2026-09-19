"""Bake NMS into the ONNX graph, so one file is the whole detector.

What this buys is not fewer FLOPs — NMS is cheap — but one round trip fewer. With
NMS outside the graph, every frame ships ``[N, 4] + [N, 80]`` back to the host (a
megabyte at 640) so a Python loop can throw almost all of it away. With NMS inside,
only the surviving detections cross that boundary.

What it costs is a data-dependent output shape, which some runtimes and most
statically-allocated callers dislike. Both forms are measured side by side in
``docs/benchmarks/runtime_matrix.md``.

Pre-processing deliberately stays outside. Letterboxing has a per-image scale and
padding that the caller needs anyway to map boxes back to original pixels, so baking
it in would hide the numbers the caller has to have. The one exception is the Frigate
target, where Frigate itself does the resize and hands over a fixed-size uint8 frame.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

__all__ = ["make_constant", "nms_nodes", "append_nms", "make_end2end_onnx"]


def make_constant(name: str, value: np.ndarray) -> onnx.NodeProto:
    """A Constant node producing *value*."""
    return helper.make_node("Constant", inputs=[], outputs=[name], value=numpy_helper.from_array(value, name=name))


def nms_nodes(
    bbox_output: str,
    score_output: str,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> list[onnx.NodeProto]:
    """Nodes taking ``[B, N, 4]`` boxes and ``[B, N, C]`` scores to ``[D, 7]``.

    The seven columns are ``[batch_index, x1, y1, x2, y2, confidence, class_id]``, in
    the letterboxed coordinate space the model was fed.
    """
    return [
        # NonMaxSuppression wants scores as [B, C, N].
        helper.make_node("Transpose", [score_output], ["scores_nms"], perm=[0, 2, 1]),
        make_constant("max_det", np.array([max_detections], dtype=np.int64)),
        make_constant("iou_thr", np.array([iou_threshold], dtype=np.float32)),
        make_constant("conf_thr", np.array([conf_threshold], dtype=np.float32)),
        helper.make_node(
            "NonMaxSuppression",
            [bbox_output, "scores_nms", "max_det", "iou_thr", "conf_thr"],
            ["selected_indices"],  # [D, 3]: batch, class, box
        ),
        make_constant("idx_0", np.array(0, dtype=np.int64)),
        make_constant("idx_1", np.array(1, dtype=np.int64)),
        make_constant("idx_2", np.array(2, dtype=np.int64)),
        helper.make_node("Gather", ["selected_indices", "idx_0"], ["batch_col"], axis=1),
        helper.make_node("Gather", ["selected_indices", "idx_1"], ["class_col"], axis=1),
        helper.make_node("Gather", ["selected_indices", "idx_2"], ["box_col"], axis=1),
        make_constant("unsq_axis", np.array([1], dtype=np.int64)),
        helper.make_node("Unsqueeze", ["batch_col", "unsq_axis"], ["batch_2d"]),
        helper.make_node("Unsqueeze", ["class_col", "unsq_axis"], ["class_2d"]),
        helper.make_node("Unsqueeze", ["box_col", "unsq_axis"], ["box_2d"]),
        helper.make_node("Concat", ["batch_2d", "box_2d"], ["bbox_gather_idx"], axis=1),
        helper.make_node("GatherND", [bbox_output, "bbox_gather_idx"], ["selected_boxes"]),
        helper.make_node("Concat", ["batch_2d", "box_2d", "class_2d"], ["score_gather_idx"], axis=1),
        helper.make_node("GatherND", [score_output, "score_gather_idx"], ["selected_scores"]),
        helper.make_node("Unsqueeze", ["selected_scores", "unsq_axis"], ["scores_2d"]),
        helper.make_node("Cast", ["batch_2d"], ["batch_float"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["class_2d"], ["class_float"], to=TensorProto.FLOAT),
        helper.make_node(
            "Concat",
            ["batch_float", "selected_boxes", "scores_2d", "class_float"],
            ["detections"],
            axis=1,
        ),
    ]


def append_nms(
    model: onnx.ModelProto,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
    extra_inputs: list | None = None,
    prefix_nodes: list | None = None,
) -> onnx.ModelProto:
    """Return *model* with NMS appended, its output replaced by ``detections [D, 7]``.

    Args:
        model: A base YOLO-NAS graph, outputs ``pred_bboxes`` then ``pred_scores``.
        conf_threshold: Score below which a detection is dropped, inside the graph.
        iou_threshold: NMS IoU.
        max_detections: Per class per image, as ONNX ``NonMaxSuppression`` defines it.
        extra_inputs: Replaces the graph inputs — used by the Frigate target, which
            prepends its own uint8 input.
        prefix_nodes: Nodes to run before the model's, for the same reason.

    Returns:
        A new checked model. The input model is not modified.
    """
    graph = model.graph
    bbox_output, score_output = graph.output[0].name, graph.output[1].name

    nodes = list(prefix_nodes or []) + list(graph.node)
    nodes += nms_nodes(bbox_output, score_output, conf_threshold, iou_threshold, max_detections)

    new_graph = helper.make_graph(
        nodes,
        graph.name,
        list(extra_inputs) if extra_inputs is not None else list(graph.input),
        # [None, 7]: the detection count is decided at run time by the thresholds.
        [helper.make_tensor_value_info("detections", TensorProto.FLOAT, [None, 7])],
        initializer=list(graph.initializer),
    )
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version
    onnx.checker.check_model(new_model)
    return new_model


def make_end2end_onnx(
    base_onnx_path: str,
    output_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> None:
    """Rewrite a base graph into one that returns detections instead of raw tensors.

    The input is unchanged — still the letterboxed ``float32 [B, 3, S, S]`` the base
    model takes — so preprocessing and the box rescale stay in the caller's hands.
    """
    onnx.save(
        append_nms(onnx.load(base_onnx_path), conf_threshold, iou_threshold, max_detections),
        output_path,
    )
