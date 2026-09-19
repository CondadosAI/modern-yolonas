"""The export API and the runtimes it targets, without downloading any weights.

The embedding and per-object graphs have their own file, `test_export.py`; this one
covers `modern_yolonas.export` itself — the ONNX writer, the NMS surgery and the
OpenVINO conversion. TensorRT is absent on purpose: it needs an NVIDIA GPU, so it is
covered by `examples/runtime_accuracy.py` instead, and `pyproject.toml` records that
as the reason its module is omitted from coverage.
"""

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


def _plain(output: str) -> str:
    """CLI output with rich's markup and wrapping removed.

    Asserting on a raw `result.output` is a trap: rich colours a `--flag` by
    inserting escape sequences *inside* it, and wraps at the terminal width, so
    `"--target generic"` is present locally and absent on a narrower CI runner.
    """
    import re

    return re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", output))


@pytest.mark.parametrize("target", ["end2end", "frigate", "objects"])
def test_int8_is_refused_for_every_rewritten_graph(tmp_path, target):
    """NNCF cannot calibrate through a rewritten graph, so the combination is refused."""
    from typer.testing import CliRunner

    from modern_yolonas.cli import app

    result = CliRunner().invoke(
        app, ["export", "--target", target, "--format", "openvino", "--precision", "int8"]
    )
    assert result.exit_code != 0
    assert "--target generic" in _plain(result.output)


# --- OpenVINO -------------------------------------------------------------------
# Runs on CPU from a pip wheel, so CI covers it. TensorRT cannot be tested here: it
# needs an NVIDIA GPU, and `pyproject.toml` omits that module from coverage saying so.

ov = pytest.importorskip("openvino")


@pytest.fixture(scope="module")
def tiny_onnx(small_model, tmp_path_factory):
    import copy

    path = tmp_path_factory.mktemp("ov") / "m.onnx"
    return export_onnx(fuse_for_inference(copy.deepcopy(small_model)), path, input_size=128, dynamic_batch=False)


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_openvino_export_runs_and_matches_onnx(tiny_onnx, tmp_path, precision):
    from modern_yolonas.export.openvino import export_openvino

    xml = export_openvino(tiny_onnx, tmp_path / f"m_{precision}.xml", precision=precision)
    assert xml.exists() and xml.with_suffix(".bin").exists()

    x = np.random.randn(1, 3, 128, 128).astype(np.float32)
    reference = onnx_session(tiny_onnx, "cpu").run(None, {"images": x})

    core = ov.Core()
    request = core.compile_model(core.read_model(xml), "CPU").create_infer_request()
    result = request.infer({0: x})
    boxes, scores = (result[o] for o in request.model_outputs)

    assert boxes.shape == reference[0].shape
    assert scores.shape == reference[1].shape
    # fp16 compresses the weights; OpenVINO still executes in fp32 on a CPU, so the
    # tolerance is the weight rounding rather than a half-precision forward pass.
    tolerance = 1e-4 if precision == "fp32" else 5e-2
    assert np.abs(reference[1] - scores).max() < tolerance


def test_openvino_int8_requires_calibration_images(tiny_onnx, tmp_path):
    from modern_yolonas.export.openvino import export_openvino

    with pytest.raises(ValueError, match="calibration-dir"):
        export_openvino(tiny_onnx, tmp_path / "m.xml", precision="int8")


def test_openvino_rejects_unknown_precision(tiny_onnx, tmp_path):
    from modern_yolonas.export.openvino import export_openvino

    with pytest.raises(ValueError, match="unknown precision"):
        export_openvino(tiny_onnx, tmp_path / "m.xml", precision="fp8")


def test_calibration_batches_spread_over_the_directory(tmp_path):
    """Sampling the first N of a sorted COCO listing is a contiguous slice of ids."""
    import cv2

    from modern_yolonas.export.openvino import calibration_batches

    for i in range(20):
        cv2.imwrite(str(tmp_path / f"{i:04d}.jpg"), np.full((40, 60, 3), i * 10, dtype=np.uint8))

    batches = calibration_batches(tmp_path, input_size=64, limit=5)
    assert len(batches) == 5
    assert all(b.shape == (1, 3, 64, 64) for b in batches)
    # Distinct fill values, so a contiguous first-five slice would collapse the spread.
    assert len({float(b.mean()) for b in batches}) == 5


def test_calibration_batches_needs_images(tmp_path):
    from modern_yolonas.export.openvino import calibration_batches

    with pytest.raises(ValueError, match="no images found"):
        calibration_batches(tmp_path, input_size=64)
