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
