"""Tests for the ONNX export graphs.

The embedding equivalence that actually matters —
``FeaturePooler.pool_images_masked`` against ``pool_images`` — is pinned in
``test_embed.py``, which runs whether or not onnx is installed. These tests check
the rest: that the graphs export, that ONNX Runtime agrees with PyTorch, and that
the batch axis really is dynamic.
"""

import warnings

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from modern_yolonas import yolo_nas_s  # noqa: E402
from modern_yolonas.export.embedding import DetectAndEmbedGraph, EmbeddingGraph  # noqa: E402
from modern_yolonas.inference.embed import FeaturePooler, valid_region  # noqa: E402
from modern_yolonas.inference.preprocess import preprocess  # noqa: E402

CANVAS = 320


@pytest.fixture(scope="module")
def fused_model():
    """Fusion is what export runs on, so it is what ORT must be compared against."""
    model = yolo_nas_s(pretrained=False).eval()
    for module in model.modules():
        if hasattr(module, "fuse_block_residual_branches"):
            module.fuse_block_residual_branches()
    return model


def _export(graph, path, output_names):
    dummy = (
        torch.randn(1, 3, CANVAS, CANVAS),
        torch.tensor([[0, 0, CANVAS, CANVAS]], dtype=torch.long),
    )
    input_names = ["images", "valid_region"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            graph,
            dummy,
            str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={name: {0: "batch"} for name in input_names + output_names},
            opset_version=18,
            external_data=False,
        )
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _letterboxed(shapes, seed=0):
    """Preprocess a few differently-shaped images into one batch plus its regions."""
    rng = np.random.default_rng(seed)
    tensors, regions = [], []
    for height, width in shapes:
        image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        tensor, scale, pad = preprocess(image, CANVAS)
        tensors.append(tensor)
        regions.append(valid_region(image, scale, pad))
    return torch.cat(tensors), torch.tensor(regions, dtype=torch.long)


@pytest.fixture(scope="module")
def session(fused_model, tmp_path_factory):
    """The embedding-only graph, exported once for the whole module."""
    graph = EmbeddingGraph(fused_model, FeaturePooler(), CANVAS).eval()
    return _export(graph, tmp_path_factory.mktemp("onnx") / "embedding.onnx", ["embedding"])


@pytest.fixture(scope="module")
def combined_session(fused_model, tmp_path_factory):
    graph = DetectAndEmbedGraph(fused_model, FeaturePooler(), CANVAS).eval()
    return _export(
        graph, tmp_path_factory.mktemp("onnx") / "combined.onnx",
        ["pred_bboxes", "pred_scores", "embedding"],
    )


class TestEmbeddingGraph:
    def test_signature(self, session):
        assert [i.name for i in session.get_inputs()] == ["images", "valid_region"]
        assert [o.name for o in session.get_outputs()] == ["embedding"]

    def test_matches_pytorch(self, session, fused_model):
        images, regions = _letterboxed([(360, 640), (640, 480)])
        graph = EmbeddingGraph(fused_model, FeaturePooler(), CANVAS).eval()
        with torch.no_grad():
            expected = graph(images, regions)

        actual = session.run(None, {"images": images.numpy(), "valid_region": regions.numpy()})[0]
        assert np.allclose(actual, expected.numpy(), atol=1e-4)

    def test_batch_axis_is_dynamic(self, session):
        """Exported at batch 1; it has to run at other batch sizes."""
        for batch in (1, 3):
            images, regions = _letterboxed([(360, 640)] * batch)
            out = session.run(None, {"images": images.numpy(), "valid_region": regions.numpy()})[0]
            assert out.shape == (batch, 768)

    def test_output_is_normalized(self, session):
        images, regions = _letterboxed([(360, 640), (200, 1600)])
        out = session.run(None, {"images": images.numpy(), "valid_region": regions.numpy()})[0]
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_region_changes_the_embedding(self, session):
        """The second input has to actually reach the pooling.

        A graph that silently ignored `valid_region` would pass every other test
        here, and would be the exact bug the input exists to prevent.
        """
        images, regions = _letterboxed([(200, 1600)])
        cropped = session.run(None, {"images": images.numpy(), "valid_region": regions.numpy()})[0]
        whole = session.run(
            None,
            {"images": images.numpy(), "valid_region": np.array([[0, 0, CANVAS, CANVAS]], dtype=np.int64)},
        )[0]
        assert not np.allclose(cropped, whole, atol=1e-3)


class TestDetectAndEmbedGraph:
    def test_signature(self, combined_session):
        assert [o.name for o in combined_session.get_outputs()] == ["pred_bboxes", "pred_scores", "embedding"]

    def test_detections_match_the_plain_model(self, combined_session, fused_model):
        """The detection half must be untouched by sharing the pass with pooling."""
        images, regions = _letterboxed([(360, 640), (640, 480)])
        with torch.no_grad():
            expected_bboxes, expected_scores = fused_model(images)

        bboxes, scores, _ = combined_session.run(
            None, {"images": images.numpy(), "valid_region": regions.numpy()}
        )
        assert bboxes.shape == expected_bboxes.shape
        assert np.allclose(bboxes, expected_bboxes.numpy(), atol=1e-3)
        assert np.allclose(scores, expected_scores.numpy(), atol=1e-4)

    def test_embedding_matches_the_embedding_only_graph(self, combined_session, fused_model, tmp_path):
        """Combining must not change the vector you would have got on its own."""
        images, regions = _letterboxed([(360, 640)])
        alone = _export(
            EmbeddingGraph(fused_model, FeaturePooler(), CANVAS).eval(),
            tmp_path / "embedding.onnx",
            ["embedding"],
        )
        feeds = {"images": images.numpy(), "valid_region": regions.numpy()}
        assert np.allclose(combined_session.run(None, feeds)[2], alone.run(None, feeds)[0], atol=1e-5)


class TestCustomPooler:
    def test_layers_and_pooling_reach_the_graph(self, fused_model, tmp_path):
        pooler = FeaturePooler(layers=("c4", "c5"), pooling="max", normalize=False)
        session = _export(
            EmbeddingGraph(fused_model, pooler, CANVAS).eval(), tmp_path / "custom.onnx", ["embedding"]
        )
        images, regions = _letterboxed([(360, 640)])
        out = session.run(None, {"images": images.numpy(), "valid_region": regions.numpy()})[0]

        assert out.shape == (1, 384 + 768)
        assert not np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-3)


class TestExportedFileIsSelfContained:
    def test_no_external_data_sidecar(self, fused_model, tmp_path):
        """`--output model.onnx` must produce a model, not half of one.

        Torch's dynamo exporter defaults to writing weights into a sibling
        `<name>.onnx.data`; an .onnx shipped without it loads and then fails.
        """
        path = tmp_path / "selfcontained.onnx"
        _export(EmbeddingGraph(fused_model, FeaturePooler(), CANVAS).eval(), path, ["embedding"])

        assert not list(tmp_path.glob("*.onnx.data"))
        model = onnx.load(str(path), load_external_data=False)
        external = [i for i in model.graph.initializer if i.data_location == onnx.TensorProto.EXTERNAL]
        assert external == []


class TestFusionDoesNotMoveTheEmbedding:
    """The exported graph runs on a fused model; users compare it to an unfused one.

    `YoloNASEmbedder` never fuses, so if fusion moved the vector, `embed_onnx.py`
    and `embed_image.py` would quietly rank the same gallery differently. Folding
    BatchNorm into the preceding convolution is exact in principle and only to
    float precision in fact, so the size of the gap belongs on record.
    """

    def test_fused_matches_unfused(self):
        import copy

        unfused = yolo_nas_s(pretrained=False).eval()
        fused = copy.deepcopy(unfused)
        for module in fused.modules():
            if hasattr(module, "fuse_block_residual_branches"):
                module.fuse_block_residual_branches()

        images, regions = _letterboxed([(360, 640), (200, 1600)])
        pooler = FeaturePooler()
        with torch.no_grad():
            before = EmbeddingGraph(unfused, pooler, CANVAS).eval()(images, regions)
            after = EmbeddingGraph(fused, pooler, CANVAS).eval()(images, regions)

        cosine = (before * after).sum(dim=1)
        assert torch.all(cosine > 0.9999), f"fusion moved the embedding: cosine {cosine.tolist()}"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A random-weight checkpoint, so the CLI tests never reach the network."""
    path = tmp_path_factory.mktemp("ckpt") / "weights.pt"
    torch.save(yolo_nas_s(pretrained=False).state_dict(), path)
    return str(path)


class TestCliProducesAWorkingGraph:
    """The graphs working is not the same as the command working."""

    def test_embedding_target(self, tmp_path, checkpoint):
        from typer.testing import CliRunner

        from modern_yolonas.cli import app

        output = tmp_path / "embedding.onnx"
        result = CliRunner().invoke(
            app,
            [
                "export", "--model", "yolo_nas_s", "--target", "embedding",
                "--input-size", "320", "--embed-layers", "c4,c5",
                "--output", str(output), "--checkpoint", checkpoint,
            ],
        )
        assert result.exit_code == 0, result.output
        assert output.exists()
        assert not list(tmp_path.glob("*.onnx.data"))

        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        assert [i.name for i in session.get_inputs()] == ["images", "valid_region"]

        images, regions = _letterboxed([(360, 640)])
        out = session.run(None, {"images": images.numpy(), "valid_region": regions.numpy()})[0]
        assert out.shape == (1, 384 + 768), "--embed-layers c4,c5 must reach the graph"

    def test_combined_target(self, tmp_path, checkpoint):
        from typer.testing import CliRunner

        from modern_yolonas.cli import app

        output = tmp_path / "combined.onnx"
        result = CliRunner().invoke(
            app,
            ["export", "--model", "yolo_nas_s", "--target", "combined",
             "--input-size", "320", "--output", str(output), "--checkpoint", checkpoint],
        )
        assert result.exit_code == 0, result.output

        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        assert [o.name for o in session.get_outputs()] == ["pred_bboxes", "pred_scores", "embedding"]

    def test_default_opset_needs_no_downconversion(self, tmp_path, checkpoint):
        """Asking for the default must not trip torch's failed 17-conversion path."""
        from typer.testing import CliRunner

        from modern_yolonas.cli import app

        output = tmp_path / "default.onnx"
        result = CliRunner().invoke(
            app,
            ["export", "--model", "yolo_nas_s", "--target", "embedding",
             "--input-size", "320", "--output", str(output), "--checkpoint", checkpoint],
        )
        assert result.exit_code == 0, result.output

        model = onnx.load(str(output), load_external_data=False)
        exported = {i.domain or "ai.onnx": i.version for i in model.opset_import}
        assert exported["ai.onnx"] >= 18
