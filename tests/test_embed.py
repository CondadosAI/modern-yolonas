"""Tests for feature extraction and the embedding API."""

import numpy as np
import pytest
import torch

from modern_yolonas import FEATURE_LAYERS, YoloNASEmbedder, yolo_nas_s
from modern_yolonas.inference.preprocess import preprocess


@pytest.fixture(scope="module")
def model():
    return yolo_nas_s(pretrained=False).eval()


@pytest.fixture(scope="module")
def embedder():
    return YoloNASEmbedder("yolo_nas_s", device="cpu", pretrained=False)


def _image(height=360, width=640, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (height, width, 3), dtype=np.uint8)


class TestForwardFeatures:
    def test_returns_every_named_layer(self, model):
        with torch.no_grad():
            features = model.forward_features(torch.zeros(1, 3, 640, 640))
        assert set(features) == set(FEATURE_LAYERS)

    @pytest.mark.parametrize(
        "name,channels,stride",
        [
            ("c2", 96, 4),
            ("c3", 192, 8),
            ("c4", 384, 16),
            ("c5", 768, 32),
            ("p3", 96, 8),
            ("p4", 192, 16),
            ("p5", 384, 32),
        ],
    )
    def test_shapes(self, model, name, channels, stride):
        with torch.no_grad():
            features = model.forward_features(torch.zeros(1, 3, 640, 640))
        assert features[name].shape == (1, channels, 640 // stride, 640 // stride)

    def test_forward_is_unchanged(self, model):
        """Regression guard: ``forward_features`` must not disturb ``forward``.

        The detection path is what ONNX export, the Frigate graph and the parity
        tests depend on, so the feature API stays strictly additive.
        """
        x = torch.zeros(1, 3, 640, 640)
        with torch.no_grad():
            boxes, scores = model(x)
        assert boxes.shape == (1, 8400, 4)
        assert scores.shape == (1, 8400, 80)


class TestEmbedder:
    def test_single_image_shape_and_norm(self, embedder):
        vector = embedder(_image())
        assert vector.shape == (768,)
        assert vector.dtype == np.float32
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)

    def test_embedding_dim_matches_output(self, embedder):
        assert embedder.embedding_dim == embedder(_image()).shape[0]

    def test_batch_matches_single(self, embedder):
        """Pooling is per sample, so batching must not change a single result."""
        a, b = _image(360, 640, seed=1), _image(640, 480, seed=2)
        batch = embedder.embed_batch([a, b])
        assert batch.shape == (2, 768)
        assert np.allclose(batch[0], embedder(a), atol=1e-5)
        assert np.allclose(batch[1], embedder(b), atol=1e-5)

    def test_concatenates_multiple_layers(self):
        embedder = YoloNASEmbedder("yolo_nas_s", device="cpu", pretrained=False, layers=("c4", "p3"))
        assert embedder.embedding_dim == 384 + 96
        assert embedder(_image()).shape == (480,)

    def test_max_pooling_differs_from_avg(self):
        image = _image()
        avg = YoloNASEmbedder("yolo_nas_s", device="cpu", pretrained=False, pooling="avg")(image)
        mx = YoloNASEmbedder("yolo_nas_s", device="cpu", pretrained=False, pooling="max")(image)
        assert not np.allclose(avg, mx)

    def test_normalize_off_leaves_magnitude(self):
        embedder = YoloNASEmbedder("yolo_nas_s", device="cpu", pretrained=False, normalize=False)
        assert np.linalg.norm(embedder(_image())) != pytest.approx(1.0, abs=1e-3)

    def test_empty_batch(self, embedder):
        assert embedder.embed_batch([]).shape == (0, 768)

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"layers": ()}, "at least one"),
            ({"layers": ("c9",)}, "unknown layer"),
            ({"pooling": "median"}, "pooling must be"),
            ({"precision": "int8"}, "precision must be"),
        ],
    )
    def test_rejects_bad_arguments(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            YoloNASEmbedder("yolo_nas_s", device="cpu", pretrained=False, **kwargs)


class TestPaddingIsExcluded:
    """``preprocess`` letterboxes onto a gray canvas; pooling must ignore the gray.

    Averaging the padding in makes embeddings cluster by aspect ratio rather than
    by content. Measured with the COCO weights on a 1600x300 strip: naive
    full-canvas pooling scores an unrelated noise image against a street photo at
    0.958 cosine — *higher* than that same street photo against a second real
    photo — because both carry the same padding. Cropping to the valid region
    puts the pair at 0.396.
    """

    def _naive(self, embedder, image):
        """What pooling the whole canvas would give, for comparison."""
        tensor, _, _ = preprocess(image, embedder.input_size)
        with torch.no_grad():
            feature = embedder.model.forward_features(tensor)["c5"][0]
        vector = feature.reshape(feature.shape[0], -1).mean(dim=1)
        return torch.nn.functional.normalize(vector, dim=0).numpy()

    def test_pools_only_the_valid_region(self, embedder):
        """A wide image and a square one must not be pooled over the same extent."""
        wide = _image(200, 640, seed=3)
        assert not np.allclose(embedder(wide), self._naive(embedder, wide), atol=1e-4)

    def test_square_image_has_almost_no_padding_to_exclude(self, embedder):
        """At 640x640 the letterbox border is only 2px, so both agree closely."""
        square = _image(640, 640, seed=4)
        cosine = float(embedder(square) @ self._naive(embedder, square))
        assert cosine > 0.99

    def test_valid_region_maps_through_the_letterbox(self, embedder):
        image = _image(300, 900, seed=5)
        _, scale, pad = preprocess(image, 640)
        left, top, right, bottom = embedder._valid_region(image, scale, pad)
        assert (left, top) == pad
        assert right - left == pytest.approx(round(900 * scale))
        assert bottom - top == pytest.approx(round(300 * scale))


class TestEmbedBoxes:
    def test_one_vector_per_box(self, embedder):
        boxes = np.array([[10, 10, 200, 200], [300, 50, 600, 350]], dtype=np.float32)
        assert embedder.embed_boxes(_image(), boxes).shape == (2, 768)

    def test_normalized(self, embedder):
        vectors = embedder.embed_boxes(_image(), np.array([[0, 0, 640, 360]]))
        assert np.linalg.norm(vectors[0]) == pytest.approx(1.0, abs=1e-5)

    def test_no_boxes(self, embedder):
        assert embedder.embed_boxes(_image(), np.zeros((0, 4))).shape == (0, 768)

    def test_distinct_regions_give_distinct_vectors(self, embedder):
        image = _image()
        vectors = embedder.embed_boxes(image, np.array([[0, 0, 200, 200], [440, 160, 640, 360]]))
        assert not np.allclose(vectors[0], vectors[1], atol=1e-3)

    def test_accepts_supervision_xyxy(self, embedder):
        """``sv.Detections.xyxy`` is float32 (K, 4) — it must pass through as is."""
        import supervision as sv

        detections = sv.Detections(
            xyxy=np.array([[10.0, 10.0, 100.0, 100.0]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([0]),
        )
        assert embedder.embed_boxes(_image(), detections.xyxy).shape == (1, 768)
