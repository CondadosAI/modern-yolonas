"""Tests for feature extraction and the embedding API."""

import numpy as np
import pytest
import torch

from modern_yolonas import FEATURE_LAYERS, YoloNASEmbedder, yolo_nas_s
from modern_yolonas.inference.embed import valid_region
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

    def test_widths_are_identical_across_variants(self):
        """S, M and L differ in depth, not in what any stage outputs.

        This is what lets `embedding_dim` stay 768 when you change variant, and it
        is the claim the docs make, so it is pinned rather than assumed.
        """
        from modern_yolonas import yolo_nas_l, yolo_nas_m

        x = torch.zeros(1, 3, 640, 640)
        with torch.no_grad():
            widths = [
                {name: feature.shape[1] for name, feature in build(pretrained=False).eval().forward_features(x).items()}
                for build in (yolo_nas_s, yolo_nas_m, yolo_nas_l)
            ]
        assert widths[0] == widths[1] == widths[2]
        assert widths[0]["c5"] == 768

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
        left, top, right, bottom = valid_region(image, scale, pad)
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

    def test_clips_boxes_to_the_frame(self, embedder):
        """A caller-supplied box past the edge must be embedded from what is visible.

        Outside the frame there is only letterbox padding, so an unclipped box would
        be described partly by gray — and would disagree with `predict`, which gets
        its boxes already clipped by `rescale_boxes`.
        """
        image = _image(360, 640)
        inside = np.array([[100.0, 50.0, 640.0, 360.0]], dtype=np.float32)
        overhanging = np.array([[100.0, 50.0, 5000.0, 4000.0]], dtype=np.float32)
        assert np.allclose(embedder.embed_boxes(image, overhanging), embedder.embed_boxes(image, inside))

    def test_fully_outside_box_is_zero_not_nan(self, embedder):
        vectors = embedder.embed_boxes(_image(), np.array([[900.0, 900.0, 1000.0, 1000.0]]))
        assert np.isfinite(vectors).all()
        assert not vectors.any()

    def test_accepts_supervision_xyxy(self, embedder):
        """``sv.Detections.xyxy`` is float32 (K, 4) — it must pass through as is."""
        import supervision as sv

        detections = sv.Detections(
            xyxy=np.array([[10.0, 10.0, 100.0, 100.0]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([0]),
        )
        assert embedder.embed_boxes(_image(), detections.xyxy).shape == (1, 768)


@pytest.fixture(scope="module")
def detector():
    from modern_yolonas import YoloNASDetector

    # conf 0 / single-label so an untrained model still yields boxes to embed.
    return YoloNASDetector(
        "yolo_nas_s", device="cpu", pretrained=False, conf_threshold=0.0, multi_label=False
    )


@pytest.fixture(scope="module")
def weight_shared_pair(detector, embedder):
    """A detector and an embedder holding the *same* weights.

    Each class builds its own model, so two ``pretrained=False`` instances start
    from different random initializations and would compare nothing.
    """
    detector.model.load_state_dict(embedder.model.state_dict())
    detector.model.eval()
    return detector, embedder


class TestSinglePassPredict:
    """``predict`` must get detections and embeddings out of one forward pass.

    Running the detector and the embedder separately costs two passes through the
    backbone, which is the expensive part. Detection and embedding share every
    layer up to the head, so the combined call runs the backbone once and reads
    both off it.
    """

    def test_backbone_runs_once_for_every_task(self, detector):
        from modern_yolonas import Task

        calls = []
        handle = detector.model.backbone.register_forward_hook(lambda *_: calls.append(1))
        try:
            detector.predict(_image(), Task.DETECT | Task.EMBED | Task.EMBED_OBJECTS)
        finally:
            handle.remove()
        assert len(calls) == 1

    def test_detect_only_matches_the_call_shorthand(self, detector):
        from modern_yolonas import Task

        image = _image()
        result = detector.predict(image, Task.DETECT)
        direct = detector(image)

        assert result.embedding is None
        assert "embedding" not in result.detections.data
        assert np.allclose(result.detections.xyxy, direct.xyxy)
        assert np.allclose(result.detections.confidence, direct.confidence)
        assert np.array_equal(result.detections.class_id, direct.class_id)

    def test_embed_only_returns_no_detections(self, detector):
        from modern_yolonas import Task

        result = detector.predict(_image(), Task.EMBED)
        assert result.detections is None
        assert result.embedding.shape == (768,)
        assert np.linalg.norm(result.embedding) == pytest.approx(1.0, abs=1e-5)

    def test_embed_objects_implies_detect(self, detector):
        from modern_yolonas import Task

        result = detector.predict(_image(), Task.EMBED_OBJECTS)
        assert result.detections is not None
        assert result.detections.data["embedding"].shape == (len(result.detections), 768)

    def test_object_vectors_slice_with_the_detections(self, detector):
        """The whole point of ``detections.data``: filtering keeps rows aligned."""
        from modern_yolonas import Task

        detections = detector.predict(_image(), Task.EMBED_OBJECTS).detections
        assert len(detections) > 1

        keep = np.zeros(len(detections), dtype=bool)
        keep[[0, 2]] = True
        subset = detections[keep]

        assert subset.data["embedding"].shape == (2, 768)
        assert np.allclose(subset.data["embedding"], detections.data["embedding"][[0, 2]])

    def test_rejects_empty_task_set(self, detector):
        from modern_yolonas import Task

        with pytest.raises(ValueError, match="at least one Task"):
            detector.predict(_image(), Task.DETECT & Task.EMBED)

    def test_batch_matches_single(self, detector):
        from modern_yolonas import Task

        a, b = _image(360, 640, seed=7), _image(480, 480, seed=8)
        batch = detector.predict_batch([a, b], Task.DETECT | Task.EMBED)
        assert len(batch) == 2
        assert np.allclose(batch[0].embedding, detector.predict(a, Task.EMBED).embedding, atol=1e-5)
        assert np.allclose(batch[1].embedding, detector.predict(b, Task.EMBED).embedding, atol=1e-5)

    def test_empty_batch(self, detector):
        from modern_yolonas import Task

        assert detector.predict_batch([], Task.DETECT) == []


class TestPredictMatchesStandaloneEmbedder:
    """One pass must produce the *same* vectors as the two-pass route."""

    def test_image_embedding_matches(self, weight_shared_pair):
        from modern_yolonas import Task

        detector, embedder = weight_shared_pair
        image = _image(360, 640, seed=9)
        assert np.allclose(detector.predict(image, Task.EMBED).embedding, embedder(image), atol=1e-5)

    def test_object_embeddings_match(self, weight_shared_pair):
        """``predict`` embeds canvas boxes; ``embed_boxes`` maps source boxes back.

        They travel different coordinate routes to the same ROIs, so this pins the
        rescale round-trip as well as the pooling.
        """
        from modern_yolonas import Task

        detector, embedder = weight_shared_pair
        image = _image(360, 640, seed=10)

        detections = detector.predict(image, Task.EMBED_OBJECTS).detections
        assert len(detections) > 0

        standalone = embedder.embed_boxes(image, detections.xyxy)
        assert np.allclose(detections.data["embedding"], standalone, atol=1e-4)

    def test_custom_pooler_is_honoured(self):
        from modern_yolonas import FeaturePooler, Task, YoloNASDetector

        detector = YoloNASDetector(
            "yolo_nas_s",
            device="cpu",
            pretrained=False,
            embedding=FeaturePooler(layers=("c4", "c5"), pooling="max", normalize=False),
        )
        vector = detector.predict(_image(), Task.EMBED).embedding
        assert vector.shape == (384 + 768,)
        assert np.linalg.norm(vector) != pytest.approx(1.0, abs=1e-3)

    def test_off_frame_boxes_are_embedded_from_the_visible_part(self, weight_shared_pair):
        """Regression: object vectors must come from the *clipped* boxes.

        ``rescale_boxes`` clips detections to the frame. Embedding the unclipped
        canvas box instead would describe a region that is partly letterbox
        padding, and would silently disagree with ``embed_boxes`` on exactly the
        detections that touch an edge.
        """
        from modern_yolonas import Task

        detector, embedder = weight_shared_pair
        image = _image(360, 640, seed=11)

        detections = detector.predict(image, Task.EMBED_OBJECTS).detections
        touches_edge = (
            (detections.xyxy[:, 0] <= 0)
            | (detections.xyxy[:, 1] <= 0)
            | (detections.xyxy[:, 2] >= image.shape[1])
            | (detections.xyxy[:, 3] >= image.shape[0])
        )
        assert touches_edge.any(), "fixture no longer produces edge detections"

        edge = detections[touches_edge]
        assert np.allclose(edge.data["embedding"], embedder.embed_boxes(image, edge.xyxy), atol=1e-4)


class TestMaskedPoolingMatchesSlicing:
    """``pool_images_masked`` must equal ``pool_images`` exactly.

    The sliced version is the reference, but its bounds are Python ``int``s, which
    ``torch.export`` cannot trace — so the export graphs use the masked version
    instead. These two implementations are only useful if they agree, and this is
    the test that says so; the ONNX tests then only have to show that ORT matches
    PyTorch. It runs whether or not onnx is installed, which is the point.
    """

    SHAPES = [(360, 640), (640, 360), (640, 640), (200, 1600), (1600, 200), (103, 997), (1, 1), (7, 13)]

    def _features_and_regions(self, model, canvas):
        tensors, regions = [], []
        for index, (height, width) in enumerate(self.SHAPES):
            image = _image(height, width, seed=index)
            tensor, scale, pad = preprocess(image, canvas)
            tensors.append(tensor)
            regions.append(valid_region(image, scale, pad))
        with torch.no_grad():
            return model.forward_features(torch.cat(tensors)), regions

    @pytest.mark.parametrize("canvas", [320, 640])
    @pytest.mark.parametrize("layers", [("c5",), FEATURE_LAYERS])
    @pytest.mark.parametrize("pooling", ["avg", "max"])
    def test_equivalent(self, model, canvas, layers, pooling):
        from modern_yolonas import FeaturePooler

        features, regions = self._features_and_regions(model, canvas)
        pooler = FeaturePooler(layers=layers, pooling=pooling)

        sliced = pooler.pool_images(features, regions, canvas)
        masked = pooler.pool_images_masked(features, torch.tensor(regions), canvas)

        assert masked.shape == sliced.shape
        assert torch.allclose(masked, sliced, atol=1e-5)

    def test_mask_covers_the_same_extent_as_the_slice(self, model):
        """Not just the pooled value — the exact rows and columns each one covers.

        A mask off by one row would still average to something close on smooth
        features, so the extents are compared directly. The reference extent is
        read out of ``pool_images`` itself rather than recomputed here: a feature
        map whose values *are* the row and column indices, max-pooled, reports the
        last index the slice touched, and the same map negated reports the first.
        Reimplementing the clamps in the test would only prove the test agrees
        with itself.
        """
        from modern_yolonas import FeaturePooler

        canvas = 640
        image = _image(200, 1600, seed=42)
        _, scale, pad = preprocess(image, canvas)
        region = valid_region(image, scale, pad)

        for size in (160, 80, 40, 20):
            cols = torch.arange(size, dtype=torch.float32).expand(size, size)
            rows = cols.t().contiguous()
            # Channels: +col, +row, -col, -row — so a max-pool reads out
            # (last_col, last_row, -first_col, -first_row).
            probe = {"c5": torch.stack([cols, rows, -cols, -rows]).unsqueeze(0)}

            pooler = FeaturePooler(layers=("c5",), pooling="max", normalize=False)
            sliced = pooler.pool_images(probe, [region], canvas)[0]
            last_col, last_row, first_col, first_row = (
                int(sliced[0]), int(sliced[1]), int(-sliced[2]), int(-sliced[3])
            )

            mask = FeaturePooler._region_mask(
                torch.tensor([region]), size, size, canvas, torch.device("cpu")
            )[0]
            mask_cols = mask.any(dim=0).nonzero().flatten()
            mask_rows = mask.any(dim=1).nonzero().flatten()

            assert (int(mask_cols[0]), int(mask_cols[-1])) == (first_col, last_col), f"cols at {size}x{size}"
            assert (int(mask_rows[0]), int(mask_rows[-1])) == (first_row, last_row), f"rows at {size}x{size}"
