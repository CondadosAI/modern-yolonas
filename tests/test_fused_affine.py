"""RandomResizedCropFlipAffine must equal the three-transform chain it replaces.

Every test here drives the fused path and the chain from the *same* sampled
parameters, so a difference is the composition maths and never the random draw.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest

from modern_yolonas.data.transforms import RandomResizedCropFlipAffine

SIZE = 640
FILL = 114


def smooth_image(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """A gradient with a few hard blocks.

    Noise would be the wrong fixture: resampling once versus three times differs most
    where neighbouring pixels differ most, so pure noise makes a correct
    implementation look broken.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([xx / w * 255, yy / h * 255, (xx + yy) / (w + h) * 255], -1)
    for _ in range(6):
        x0, y0 = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        img[y0:y0 + 50, x0:x0 + 50] = rng.integers(0, 255, 3)
    return img.astype(np.uint8)


def apply_chain(image: np.ndarray, params: dict) -> np.ndarray:
    """What RandomResizedCrop → HorizontalFlip → RandomAffine does to an image."""
    x1, y1, x2, y2 = params["crop"]
    out = cv2.resize(image[y1:y2, x1:x2], (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    if params["flip"]:
        out = cv2.flip(out, 1)
    return cv2.warpAffine(
        out, params["matrix"][:2], (SIZE, SIZE), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(FILL,) * 3,
    )


def diff_against_chain(fused_image: np.ndarray, chain_image: np.ndarray):
    """Compare only where the chain has real content, not the padded border."""
    inside = (chain_image != FILL).any(-1)
    d = np.abs(fused_image.astype(np.int16) - chain_image.astype(np.int16))[inside]
    return d.mean(), np.percentile(d, 99)


def seeded(seed: int, **kwargs) -> RandomResizedCropFlipAffine:
    """A transform whose sampling is reproducible.

    ``np.random.seed`` does not reach it: the crop and affine parameters come from
    Albumentations' own per-transform RNG, and the flip from the ``random`` module.
    Seeding only numpy leaves these tests flaky — they passed, then failed on the
    identical tree.
    """
    tf = RandomResizedCropFlipAffine(size=SIZE, pad_value=FILL, **kwargs)
    tf._crop.set_random_seed(seed)
    tf._affine.set_random_seed(seed)
    random.seed(seed)
    return tf


@pytest.fixture
def tf() -> RandomResizedCropFlipAffine:
    return seeded(0)


class TestMatchesTheChain:
    @pytest.mark.parametrize("seed", range(6))
    def test_image_has_no_registration_error_against_the_chain(self, tf, seed):
        """Mean difference only. The 99th percentile is deliberately not asserted.

        One bilinear resample and three do not agree pixel for pixel — three blur more
        — so the tail difference is real, expected, and the reason to fuse in the first
        place. It reaches 2-3 grey levels on hard edges. A threshold tuned to admit that
        would be a threshold tuned to the implementation.

        The mean is the registration signal: a sub-pixel shift moves *every* pixel and
        lifts the mean, while blur moves only the edges and barely touches it.
        Content-to-label registration is pinned separately, by the marker test below.
        """
        tf = seeded(seed)
        image = smooth_image(seed=seed)
        params = tf.sample_params(*image.shape[:2])

        fused, _ = tf.apply(image, np.zeros((0, 5), np.float32), params)
        mean, _ = diff_against_chain(fused, apply_chain(image, params))
        assert mean < 0.5, f"mean difference {mean:.3f} suggests a registration error"

    @pytest.mark.parametrize("seed", range(8))
    def test_the_picture_lands_where_the_box_says_it_does(self, seed):
        """The invariant that actually matters, and the one a half-pixel bug breaks.

        A bright square is the only content; its bounding box is the target. After the
        warp, the box the transform reports must still enclose the bright pixels. If the
        image matrix and the box matrix drift apart, the labels slide off the objects and
        training degrades with nothing raising.
        """
        tf = seeded(seed, flip_prob=0.5)
        h, w = 480, 640
        image = np.zeros((h, w, 3), np.uint8)
        x1, y1, x2, y2 = 200, 150, 360, 310
        image[y1:y2, x1:x2] = 255
        targets = np.array([[0.0,
                             (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                             (x2 - x1) / w, (y2 - y1) / h]], np.float32)

        # Sample until the square survives the crop; an empty result proves nothing.
        for _ in range(60):
            params = tf.sample_params(h, w)
            out, boxes = tf.apply(image, targets, params)
            if len(boxes) and (out > 128).any():
                break
        else:
            pytest.skip("no draw kept the marker in frame")

        ys, xs = np.nonzero((out > 128).any(-1))
        bx1 = (boxes[0, 1] - boxes[0, 3] / 2) * SIZE
        by1 = (boxes[0, 2] - boxes[0, 4] / 2) * SIZE
        bx2 = (boxes[0, 1] + boxes[0, 3] / 2) * SIZE
        by2 = (boxes[0, 2] + boxes[0, 4] / 2) * SIZE

        # One pixel of slack for the bilinear ramp at the marker's own edge.
        assert bx1 - 1 <= xs.min() and xs.max() <= bx2 + 1, f"x: box [{bx1:.1f},{bx2:.1f}] vs pixels [{xs.min()},{xs.max()}]"
        assert by1 - 1 <= ys.min() and ys.max() <= by2 + 1, f"y: box [{by1:.1f},{by2:.1f}] vs pixels [{ys.min()},{ys.max()}]"

    def test_the_crop_matrix_uses_pixel_centres(self, tf):
        """The half-pixel correction, asserted in closed form.

        cv2 maps pixel *centres*: the source coordinate behind output pixel ``d`` is
        ``(d + 0.5) * s - 0.5``. Writing the obvious-looking ``dst = (src - x1) * S/crop``
        instead shifts the picture by ``0.5 * (s - 1)`` px — half a pixel when the crop
        is upscaled 2x — while the boxes, which use exactly that geometric form, stay
        where they were. Nothing raises; the labels stop matching the pixels.

        Asserted on the matrix rather than on rendered images because the image
        difference vanishes into rounding for gentle crops, and a guard that passes
        for free on half its inputs is not a guard.
        """
        crop = (10, 20, 330, 340)          # 320x320 source into 640x640 out: exactly 2x
        centre = tf._crop_matrix(crop, pixel_centre=True)
        geometric = tf._crop_matrix(crop, pixel_centre=False)

        sx = SIZE / (crop[2] - crop[0])
        assert sx == 2.0
        assert centre[0, 0] == geometric[0, 0] == sx
        assert centre[0, 2] - geometric[0, 2] == pytest.approx(0.5 * (sx - 1))
        assert centre[1, 2] - geometric[1, 2] == pytest.approx(0.5 * (sx - 1))
        assert 0.5 * (sx - 1) == 0.5      # the shift a 2x crop would suffer

    @pytest.mark.parametrize("seed", range(6))
    def test_the_geometric_matrix_is_never_better_on_real_images(self, tf, seed):
        """The empirical companion to the closed-form test above.

        It cannot be strict: for a crop close to 1x the shift falls below one grey
        level and both matrices tie. It still catches a correction applied backwards.
        """
        tf = seeded(seed)
        image = smooth_image(seed=seed)
        params = tf.sample_params(*image.shape[:2])
        chain = apply_chain(image, params)

        eye = np.eye(3)
        wrong = (
            params["matrix"]
            @ (tf._flip_matrix(pixel_centre=True) if params["flip"] else eye)
            @ tf._crop_matrix(params["crop"], pixel_centre=False)
        )
        shifted = cv2.warpAffine(
            image, wrong[:2], (SIZE, SIZE), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(FILL,) * 3,
        )

        mean_wrong, _ = diff_against_chain(shifted, chain)
        fused, _ = tf.apply(image, np.zeros((0, 5), np.float32), params)
        mean_right, _ = diff_against_chain(fused, chain)
        assert mean_right <= mean_wrong

    @pytest.mark.parametrize("seed", range(6))
    def test_boxes_match_the_chain_exactly(self, tf, seed):
        """Box geometry has no resampling in it, so it must agree to float precision."""
        tf = seeded(seed)
        h, w = 480, 640
        rng = np.random.default_rng(seed)
        targets = np.concatenate([
            rng.integers(0, 80, (12, 1)),
            rng.uniform(0.2, 0.8, (12, 2)),
            rng.uniform(0.05, 0.3, (12, 2)),
        ], axis=1).astype(np.float32)

        params = tf.sample_params(h, w)
        _, fused = tf.apply(smooth_image(h, w, seed), targets, params)

        # The chain, in pixel space, with the same matrix.
        m = tf.box_matrix(params)
        cx, cy = targets[:, 1] * w, targets[:, 2] * h
        bw, bh = targets[:, 3] * w, targets[:, 4] * h
        corners_x = np.stack([cx - bw / 2, cx + bw / 2])
        corners_y = np.stack([cy - bh / 2, cy + bh / 2])
        px, py = m[0, 0] * corners_x + m[0, 2], m[1, 1] * corners_y + m[1, 2]
        x1, x2 = np.clip(px.min(0), 0, SIZE), np.clip(px.max(0), 0, SIZE)
        y1, y2 = np.clip(py.min(0), 0, SIZE), np.clip(py.max(0), 0, SIZE)
        keep = ((x2 - x1) >= 2.0) & ((y2 - y1) >= 2.0)

        assert len(fused) == int(keep.sum())
        if len(fused):
            np.testing.assert_allclose(fused[:, 1], (x1[keep] + x2[keep]) / 2 / SIZE, atol=1e-6)
            np.testing.assert_allclose(fused[:, 3], (x2[keep] - x1[keep]) / SIZE, atol=1e-6)


class TestEdges:
    def test_identity_parameters_return_the_input(self):
        """Whole-image crop, no flip, identity affine: the warp must be a copy."""
        tf = RandomResizedCropFlipAffine(size=64, pad_value=FILL)
        image = smooth_image(64, 64, seed=3)
        params = {"crop": (0, 0, 64, 64), "flip": False,
                  "matrix": np.eye(3), "bbox_matrix": np.eye(3)}
        out, _ = tf.apply(image, np.zeros((0, 5), np.float32), params)
        assert np.array_equal(out, image)

    def test_a_box_over_the_crop_edge_lands_on_the_border(self):
        tf = RandomResizedCropFlipAffine(size=64, pad_value=FILL)
        image = smooth_image(64, 64, seed=1)
        # Crop the right half; a box centred on the left lands partly outside.
        params = {"crop": (32, 0, 64, 64), "flip": False,
                  "matrix": np.eye(3), "bbox_matrix": np.eye(3)}
        targets = np.array([[0, 0.5, 0.5, 0.4, 0.4]], np.float32)
        _, out = tf.apply(image, targets, params)
        assert len(out) == 1
        assert out[0, 1] - out[0, 3] / 2 == pytest.approx(0.0, abs=1e-6)

    def test_boxes_that_vanish_are_dropped_not_kept_at_zero_size(self):
        tf = RandomResizedCropFlipAffine(size=64, pad_value=FILL)
        image = smooth_image(64, 64, seed=1)
        params = {"crop": (32, 0, 64, 64), "flip": False,
                  "matrix": np.eye(3), "bbox_matrix": np.eye(3)}
        targets = np.array([[0, 0.1, 0.5, 0.05, 0.4]], np.float32)  # entirely left of the crop
        _, out = tf.apply(image, targets, params)
        assert len(out) == 0

    def test_empty_targets_survive_the_round_trip(self, tf):
        image = smooth_image(seed=0)
        out, t = tf(image, np.zeros((0, 5), np.float32))
        assert out.shape == (SIZE, SIZE, 3) and t.shape == (0, 5)

    def test_flip_actually_mirrors(self):
        tf = RandomResizedCropFlipAffine(size=64, pad_value=FILL)
        image = smooth_image(64, 64, seed=2)
        params = {"crop": (0, 0, 64, 64), "flip": True,
                  "matrix": np.eye(3), "bbox_matrix": np.eye(3)}
        out, _ = tf.apply(image, np.zeros((0, 5), np.float32), params)
        assert np.array_equal(out, cv2.flip(image, 1))


class TestRefusesWhatItCannotFuse:
    @pytest.mark.parametrize("kwargs", [{"degrees": 5.0}, {"shear": 3.0}, {"degrees": 1.0, "shear": 1.0}])
    def test_rotation_and_shear_raise(self, kwargs):
        with pytest.raises(ValueError, match="axis-aligned"):
            RandomResizedCropFlipAffine(**kwargs)

    def test_the_axis_aligned_default_is_accepted(self):
        RandomResizedCropFlipAffine(degrees=0.0, shear=0.0)
