"""Contracts behind the 2026-09-19 loader work.

These pin behaviour that is invisible at a glance and would fail silently:
a uint8 batch that nothing converts trains on 0-255 inputs and merely converges
badly, and an HSV shift with the wrong wrap-around darkens reds without raising.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from modern_yolonas.data.transforms import HSVAugment, Normalize
from modern_yolonas.training.lightning_module import YoloNASLightningModule


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)


@pytest.fixture
def targets() -> np.ndarray:
    return np.array([[0, 0.5, 0.5, 0.2, 0.2]], dtype=np.float32)


class TestNormalizeDtype:
    def test_float32_is_the_default(self, image, targets):
        out, _ = Normalize()(image, targets)
        assert out.dtype == np.float32
        assert out.shape == (3, 32, 48)
        assert 0.0 <= out.min() and out.max() <= 1.0

    def test_uint8_reconstructs_float32_exactly(self, image, targets):
        """The whole point of deferring the divide: it must cost nothing numerically."""
        f32, _ = Normalize(dtype="float32")(image, targets)
        u8, _ = Normalize(dtype="uint8")(image, targets)
        assert u8.dtype == np.uint8
        assert np.array_equal(f32, u8.astype(np.float32) / 255.0)

    def test_uint8_is_a_quarter_of_the_bytes(self, image, targets):
        f32, _ = Normalize(dtype="float32")(image, targets)
        u8, _ = Normalize(dtype="uint8")(image, targets)
        assert u8.nbytes * 4 == f32.nbytes

    def test_output_is_contiguous(self, image, targets):
        """A non-contiguous array would be copied again by torch.from_numpy."""
        for dtype in ("float32", "uint8"):
            out, _ = Normalize(dtype=dtype)(image, targets)
            assert out.flags["C_CONTIGUOUS"]

    def test_channel_order_is_rgb(self, targets):
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[..., 2] = 255  # red, in BGR
        out, _ = Normalize(dtype="uint8")(bgr, targets)
        assert out[0].max() == 255 and out[1].max() == 0 and out[2].max() == 0

    def test_rejects_an_unknown_dtype(self):
        with pytest.raises(ValueError, match="float32"):
            Normalize(dtype="float16")


class TestOnAfterBatchTransfer:
    """The hook is what makes `Normalize(dtype="uint8")` safe."""

    @staticmethod
    def _module(**kw) -> YoloNASLightningModule:
        return YoloNASLightningModule(model=torch.nn.Identity(), **kw)

    def test_uint8_batches_are_scaled_to_unit_range(self):
        images = torch.full((2, 3, 8, 8), 255, dtype=torch.uint8)
        out, _ = self._module().on_after_batch_transfer((images, torch.zeros(0, 6)), 0)
        assert out.dtype == torch.float32
        assert torch.allclose(out, torch.ones_like(out))

    def test_float32_batches_pass_through_untouched(self):
        """A float32 dataloader must still work against the same module."""
        images = torch.rand(2, 3, 8, 8)
        out, _ = self._module().on_after_batch_transfer((images, torch.zeros(0, 6)), 0)
        assert torch.equal(out, images)

    def test_it_is_idempotent(self):
        """Running twice must not divide by 255 twice."""
        m = self._module()
        images = torch.full((1, 3, 4, 4), 255, dtype=torch.uint8)
        once, _ = m.on_after_batch_transfer((images, torch.zeros(0, 6)), 0)
        twice, _ = m.on_after_batch_transfer((once, torch.zeros(0, 6)), 0)
        assert torch.equal(once, twice)

    def test_channels_last_changes_layout_not_values(self):
        m = self._module(channels_last=True)
        images = torch.rand(2, 3, 8, 8)
        out, _ = m.on_after_batch_transfer((images, torch.zeros(0, 6)), 0)
        assert out.is_contiguous(memory_format=torch.channels_last)
        assert torch.equal(out, images)

    def test_channels_last_is_off_by_default(self):
        assert self._module().channels_last is False


class TestHSVAugment:
    def test_p_zero_returns_the_image_untouched(self, image, targets):
        out, _ = HSVAugment(p=0.0)(image, targets)
        assert out is image

    def test_zero_gain_adds_nothing_to_the_colour_round_trip(self, image, targets):
        """With every gain at zero the three LUTs are the identity, so the result must
        equal a bare BGR→HSV→BGR conversion exactly.

        Comparing against the input instead would need a tolerance for uint8 HSV
        quantisation, which reaches 6 on random pixels — a threshold loose enough to
        hide a genuinely wrong LUT.
        """
        out, _ = HSVAugment(hgain=0, sgain=0, vgain=0, p=1.0)(image, targets)
        expected = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2BGR)
        assert np.array_equal(out, expected)

    def test_shape_dtype_and_targets_are_preserved(self, image, targets):
        out, t = HSVAugment(p=1.0)(image, targets)
        assert out.shape == image.shape and out.dtype == np.uint8
        assert np.array_equal(t, targets)

    def test_hue_wraps_instead_of_clipping(self, targets):
        """Hue is an angle. Clipping it at 179 would turn magenta into red."""
        hsv = np.zeros((4, 4, 3), dtype=np.uint8)
        hsv[..., 0] = 175          # near the top of OpenCV's hue range
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        aug = HSVAugment(hgain=10, sgain=0, vgain=0, p=1.0)
        seen = set()
        for _ in range(200):
            out, _ = aug(bgr, targets)
            seen.add(int(cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[0, 0, 0]))

        # A +10 shift from 175 must land near 5, not pile up at 179.
        assert any(h < 15 for h in seen), f"hue never wrapped past 180: {sorted(seen)}"

    def test_shift_stays_inside_the_configured_range(self, targets):
        """vgain bounds the brightness shift; a LUT that overflowed would break this."""
        flat = np.full((8, 8, 3), 100, dtype=np.uint8)
        aug = HSVAugment(hgain=0, sgain=0, vgain=20, p=1.0)
        for _ in range(100):
            out, _ = aug(flat, targets)
            v = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[..., 2]
            assert 100 - 20 - 2 <= int(v.min()) and int(v.max()) <= 100 + 20 + 2
