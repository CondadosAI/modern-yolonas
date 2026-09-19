"""Export paths that run without downloading pretrained weights."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from modern_yolonas import yolo_nas_s
from modern_yolonas.export import export_onnx, fuse_for_inference, onnx_session

onnx = pytest.importorskip("onnx")


@pytest.fixture(scope="module")
def small_model():
    return yolo_nas_s(pretrained=False, num_classes=4).eval()


def test_fuse_is_numerically_transparent(small_model):
    """Fusing RepVGG branches must not change the output, only the op count."""
    import copy

    model = copy.deepcopy(small_model)
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        before = model(x)
        after = fuse_for_inference(model)(x)

    for b, a in zip(before, after):
        assert torch.allclose(b, a, atol=1e-4)


@pytest.mark.parametrize("input_size,anchors", [(320, 2100), (640, 8400)])
def test_onnx_export_runs_and_matches_torch(small_model, tmp_path, input_size, anchors):
    """The anchor count follows the input size, and ONNX Runtime agrees with torch."""
    import copy

    model = fuse_for_inference(copy.deepcopy(small_model))
    path = export_onnx(model, tmp_path / "m.onnx", input_size=input_size)

    # external_data defaults off, so the artifact is one file.
    assert path.exists()
    assert not (tmp_path / "m.onnx.data").exists()

    x = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        torch_boxes, torch_scores = model(x)

    session = onnx_session(path, "cpu")
    boxes, scores = session.run(None, {"images": x.numpy()})

    assert boxes.shape == (1, anchors, 4)
    assert scores.shape == (1, anchors, 4)
    assert np.abs(torch_boxes.numpy() - boxes).max() < 1e-2
    assert np.abs(torch_scores.numpy() - scores).max() < 1e-4


def test_onnx_session_rejects_unknown_provider(small_model, tmp_path):
    path = export_onnx(fuse_for_inference(small_model), tmp_path / "m.onnx", input_size=128)
    with pytest.raises(ValueError, match="unknown provider"):
        onnx_session(path, "metal")


def test_half_export_carries_fp16_into_the_graph(small_model, tmp_path):
    """TensorRT 11 is strongly typed, so fp16 has to be visible in the ONNX itself."""
    import copy

    path = export_onnx(
        fuse_for_inference(copy.deepcopy(small_model)), tmp_path / "half.onnx", input_size=128, half=True
    )
    model = onnx.load(str(path))
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16


def test_end2end_graph_matches_torch_nms(small_model, tmp_path):
    """NMS baked into the graph must return what `postprocess` returns."""
    import copy

    from modern_yolonas.export.nms import make_end2end_onnx
    from modern_yolonas.inference.postprocess import postprocess

    torch.manual_seed(0)
    model = fuse_for_inference(copy.deepcopy(small_model))
    base = export_onnx(model, tmp_path / "base.onnx", input_size=128, dynamic_batch=False)

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        boxes, scores = model(x)

    # An untrained head is not confident about anything, so a fixed threshold would
    # leave both sides empty and the comparison would pass on nothing. Pick one from
    # the model's own distribution instead.
    conf = float(scores.quantile(0.995))
    make_end2end_onnx(str(base), str(tmp_path / "e2e.onnx"), conf_threshold=conf, iou_threshold=0.45)
    expected = postprocess(boxes, scores, conf_threshold=conf, iou_threshold=0.45, max_detections=300)[0]
    assert expected[0].shape[0] > 0

    (detections,) = onnx_session(tmp_path / "e2e.onnx", "cpu").run(None, {"images": x.numpy()})
    assert detections.shape[1] == 7  # batch, x1, y1, x2, y2, conf, class
    assert detections.shape[0] == expected[0].shape[0]

    # NMS may order ties differently, so compare the score multisets rather than rows.
    got = np.sort(detections[:, 5])[::-1]
    want = np.sort(expected[1].numpy())[::-1]
    assert np.abs(got - want).max() < 1e-4


def test_end2end_rejects_int8(tmp_path):
    """NNCF cannot calibrate through a baked-in NMS, so the combination is refused."""
    from typer.testing import CliRunner

    from modern_yolonas.cli import app

    result = CliRunner().invoke(
        app, ["export", "--target", "end2end", "--format", "openvino", "--precision", "int8"]
    )
    assert result.exit_code != 0
    assert "NMS" in result.output or "nms" in result.output
