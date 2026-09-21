"""Tests for the domain-distance measurement.

These check the statistics against cases whose answer is known in advance: two
samples of one distribution must read near zero, and two clearly separated ones
must read near the maximum. A metric that cannot do both is not measuring
distance, and both failures produce plausible-looking numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from modern_yolonas.analysis.domain_distance import (
    compare,
    find_images,
    kernel_distance,
    proxy_a_distance,
)


def _blobs(n: int, dim: int, shift: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(n, dim, generator=generator) + shift).double()


def test_identical_distributions_read_near_zero():
    """Two samples of one Gaussian: a probe cannot beat chance, so d_A ~ 0."""
    a = _blobs(400, 16, shift=0.0, seed=0)
    b = _blobs(400, 16, shift=0.0, seed=1)
    d_a, epsilon = proxy_a_distance(a, b, seed=0)
    assert d_a < 0.35, f"d_A={d_a} on identical distributions"
    assert 0.4 < epsilon < 0.6


def test_separated_distributions_read_near_two():
    """Gaussians five standard deviations apart are trivially separable."""
    a = _blobs(400, 16, shift=0.0, seed=0)
    b = _blobs(400, 16, shift=5.0, seed=1)
    d_a, epsilon = proxy_a_distance(a, b, seed=0)
    assert d_a > 1.8, f"d_A={d_a} on well-separated distributions"
    assert epsilon < 0.05


def test_distance_grows_with_separation():
    """Monotonic in the shift -- the property that makes it a distance at all."""
    a = _blobs(400, 16, shift=0.0, seed=0)
    distances = [
        proxy_a_distance(a, _blobs(400, 16, shift=s, seed=1), seed=0)[0]
        for s in (0.0, 0.5, 1.5, 4.0)
    ]
    assert distances == sorted(distances), distances


def test_proxy_a_distance_uses_held_out_error():
    """In-sample error would report 2.0 for identical distributions.

    A linear probe on 16 dimensions and 800 points can memorise noise. This is the
    single assumption the formula rests on, and getting it wrong makes every
    dataset look maximally distant -- which reads like a working tool.
    """
    a = _blobs(60, 256, shift=0.0, seed=0)
    b = _blobs(60, 256, shift=0.0, seed=1)
    d_a, epsilon = proxy_a_distance(a, b, seed=0)
    # Fitted on the data it is scored on, this configuration separates perfectly.
    assert epsilon > 0.2, f"error {epsilon} is too low to be held out"
    assert d_a < 1.2


def test_proxy_a_distance_is_bounded():
    a = _blobs(200, 8, shift=0.0, seed=0)
    b = _blobs(200, 8, shift=0.0, seed=1)
    d_a, _ = proxy_a_distance(a, b, seed=3)
    assert 0.0 <= d_a <= 2.0


def test_kid_is_near_zero_for_one_distribution():
    a = _blobs(300, 16, shift=0.0, seed=0)
    b = _blobs(300, 16, shift=0.0, seed=1)
    kid, std = kernel_distance(a, b, subsets=5, seed=0)
    assert abs(kid) < 10 * (std + 1e-6) or abs(kid) < 1e-2, f"kid={kid} std={std}"


def test_kid_grows_with_separation():
    a = _blobs(300, 16, shift=0.0, seed=0)
    near = kernel_distance(a, _blobs(300, 16, shift=0.5, seed=1), subsets=5, seed=0)[0]
    far = kernel_distance(a, _blobs(300, 16, shift=3.0, seed=1), subsets=5, seed=0)[0]
    assert far > near


def test_kid_unbiased_estimator_excludes_the_diagonal():
    """The biased estimator cannot return a negative value; the unbiased one can.

    Two samples of one distribution give an MMD^2 estimate that scatters around
    zero, so some seeds must come out below it. If none ever do, the diagonal is
    still in the sum.
    """
    values = [
        kernel_distance(
            _blobs(200, 8, shift=0.0, seed=s),
            _blobs(200, 8, shift=0.0, seed=s + 100),
            subsets=3,
            seed=s,
        )[0]
        for s in range(8)
    ]
    assert min(values) < 0, values


def test_compare_reports_both_statistics():
    a = _blobs(200, 16, shift=0.0, seed=0)
    b = _blobs(200, 16, shift=4.0, seed=1)
    result = compare(a, b, encoder_name="test")
    assert result.d_a > 1.5
    assert result.n_query == 200
    assert "proxy A-distance" in result.summary()
    assert "KID" in result.summary()


@pytest.fixture
def image_dir(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(20):
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(
            tmp_path / f"{i:03d}.jpg"
        )
    (tmp_path / "notes.txt").write_text("not an image")
    return tmp_path


def test_find_images_skips_non_images(image_dir):
    assert len(find_images(image_dir)) == 20


def test_find_images_samples_rather_than_truncates(image_dir):
    """Directories are often ordered by time or class, so the first N are biased."""
    chosen = find_images(image_dir, limit=5, seed=0)
    assert len(chosen) == 5
    assert chosen != sorted(image_dir.glob("*.jpg"))[:5]


def test_find_images_is_reproducible(image_dir):
    assert find_images(image_dir, limit=5, seed=1) == find_images(image_dir, limit=5, seed=1)


def test_find_images_raises_on_an_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="No images under"):
        find_images(tmp_path)
