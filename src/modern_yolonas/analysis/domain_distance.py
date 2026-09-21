"""How far is a dataset from COCO?

The question behind this module is a spending decision. Self-supervised
pretraining and distillation both cost GPU-days, and both are worth it in
proportion to how badly a COCO-trained representation already fits the images you
have. Nobody can eyeball that, and the honest alternative -- fine-tune twice and
compare -- costs exactly what the measurement is supposed to save.

Two numbers, because they fail differently.

**Proxy A-distance** trains a classifier to tell the two datasets apart and reports
``d_A = 2(1 - 2*eps)``, where ``eps`` is its held-out error. It is bounded in
``[0, 2]`` and it has a sentence attached: ``d_A = 0`` means a linear probe cannot
separate the datasets at all; ``d_A = 2`` means it separates them perfectly. That
sentence is why it is the headline.

**KID** is the squared maximum mean discrepancy under a polynomial kernel. Unlike
FID it is unbiased, which matters here because we sample a couple of thousand
images rather than fifty thousand -- a biased estimator would move with the sample
size and invite comparisons between runs that used different ``--samples``.

Both are computed in the feature space of the encoder whose weights would actually
be transferred, which is the space the question is about. A distance measured in
some third representation would answer a different question.

**There is no calibrated threshold here, and this module does not invent one.**
What it reports instead is the measured floor: the same statistic computed between
two samples of COCO itself. A reader can see how far their number sits above
"indistinguishable". Mapping a given distance to an expected AP gain would take an
experiment nobody has run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(root: str | Path, limit: int | None = None, seed: int = 0) -> list[Path]:
    """A reproducible random sample of the images under *root*.

    Sampled rather than truncated: image directories are very often ordered by
    capture time or by class, so the first N files are a biased slice of the
    dataset and would make two runs over the same data disagree.
    """
    root = Path(root)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No images under {root}.")
    if limit is not None and len(paths) > limit:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(paths), size=limit, replace=False)
        paths = [paths[i] for i in sorted(chosen)]
    return paths


class _ImageSet(torch.utils.data.Dataset):
    """Images resized to a square, scaled to [0, 1]. No augmentation."""

    def __init__(self, paths: list[Path], size: int):
        self.paths = paths
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        import cv2

        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {self.paths[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)


class BackboneEncoder(nn.Module):
    """Global-average-pooled SPP features from a YOLO-NAS backbone.

    The default encoder, because it is the representation a fine-tune would
    actually start from -- so the distance it reports is the distance that matters
    to the decision -- and because it needs no download beyond the detection
    weights the user already has.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        *_, c5 = self.backbone(images)
        return c5.mean(dim=(2, 3))


@torch.no_grad()
def embed(
    paths: list[Path],
    encoder: nn.Module,
    device: str = "cuda",
    size: int = 640,
    batch_size: int = 32,
    workers: int = 8,
) -> Tensor:
    """One feature vector per image, on the CPU as float64.

    float64 because the MMD estimator below subtracts sums of similar magnitude,
    and in float32 that cancellation is a real source of error at n = 2000.
    """
    encoder = encoder.to(device).eval()
    loader = torch.utils.data.DataLoader(
        _ImageSet(paths, size), batch_size=batch_size, num_workers=workers, pin_memory=True
    )
    chunks = [encoder(batch.to(device, non_blocking=True)).cpu() for batch in loader]
    return torch.cat(chunks).double()


def proxy_a_distance(
    features_a: Tensor,
    features_b: Tensor,
    folds: int = 5,
    epochs: int = 200,
    lr: float = 0.1,
    seed: int = 0,
) -> tuple[float, float]:
    """``d_A = 2(1 - 2*eps)`` from a cross-validated linear domain discriminator.

    The error has to be held out. A linear probe on a few thousand high-dimensional
    vectors will separate almost any two sets perfectly on the data it was fitted
    to, which would report ``d_A = 2`` for every dataset ever passed in.

    The classes are balanced by truncating the larger set, so ``eps = 0.5`` really
    is the chance level the formula assumes.

    Returns:
        ``(d_A, eps)``. ``d_A`` is clamped to ``[0, 2]``: a discriminator can land
        below chance on a fold, which the formula would otherwise turn into a
        negative distance.
    """
    generator = torch.Generator().manual_seed(seed)
    n = min(len(features_a), len(features_b))
    features_a = features_a[torch.randperm(len(features_a), generator=generator)[:n]]
    features_b = features_b[torch.randperm(len(features_b), generator=generator)[:n]]

    x = torch.cat([features_a, features_b]).float()
    y = torch.cat([torch.zeros(n), torch.ones(n)])
    # Standardise: the features are unnormalised activations, and without this the
    # optimiser's single learning rate has to serve wildly different scales.
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)

    order = torch.randperm(len(x), generator=generator)
    x, y = x[order], y[order]
    fold_of = torch.arange(len(x)) % folds

    errors = []
    for fold in range(folds):
        train, test = fold_of != fold, fold_of == fold
        model = nn.Linear(x.shape[1], 1)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        for _ in range(epochs):
            optimiser.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(x[train]).squeeze(1), y[train]
            )
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            predicted = (model(x[test]).squeeze(1) > 0).float()
        errors.append((predicted != y[test]).float().mean().item())

    epsilon = float(np.mean(errors))
    return float(np.clip(2 * (1 - 2 * epsilon), 0.0, 2.0)), epsilon


def _polynomial_kernel(x: Tensor, y: Tensor) -> Tensor:
    dim = x.shape[1]
    return (x @ y.T / dim + 1.0) ** 3


def kernel_distance(features_a: Tensor, features_b: Tensor, subsets: int = 10, seed: int = 0) -> tuple[float, float]:
    """KID: the unbiased MMD^2 under a degree-3 polynomial kernel.

    Unbiased means the diagonal is excluded from the within-set terms. FID's
    estimator is biased by the sample size, so an FID computed on 2000 images is
    not comparable to one computed on 5000 -- which is exactly the mistake a CLI
    flag invites.

    Averaged over random subsets, as the KID paper does, so the spread across them
    can be reported alongside the value. A KID whose standard deviation is the size
    of the value itself is not a measurement.

    Returns:
        ``(mean, std)`` over the subsets.
    """
    generator = torch.Generator().manual_seed(seed)
    n = min(len(features_a), len(features_b), 1000)

    values = []
    for _ in range(subsets):
        x = features_a[torch.randperm(len(features_a), generator=generator)[:n]]
        y = features_b[torch.randperm(len(features_b), generator=generator)[:n]]

        k_xx = _polynomial_kernel(x, x)
        k_yy = _polynomial_kernel(y, y)
        k_xy = _polynomial_kernel(x, y)

        # Drop the diagonals: k(x_i, x_i) is what makes the biased estimator biased.
        sum_xx = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1))
        sum_yy = (k_yy.sum() - k_yy.diag().sum()) / (n * (n - 1))
        values.append((sum_xx + sum_yy - 2 * k_xy.mean()).item())

    return float(np.mean(values)), float(np.std(values))


@dataclass
class DomainDistance:
    """The measurement, plus everything needed to reproduce or distrust it."""

    d_a: float
    discriminator_error: float
    kid: float
    kid_std: float
    n_query: int
    n_reference: int
    encoder: str

    def summary(self) -> str:
        separable = (1 - self.discriminator_error) * 100
        return (
            f"proxy A-distance : {self.d_a:.3f}  (of a possible 2.0)\n"
            f"  a linear probe tells the two sets apart {separable:.1f}% of the time\n"
            # Scientific notation, not fixed decimals: a same-domain KID lands
            # around 1e-5, which four decimal places renders as a bare "0.0000"
            # -- indistinguishable from an encoder that emitted nothing.
            f"KID              : {self.kid:.3e} +/- {self.kid_std:.3e}\n"
            f"encoder          : {self.encoder}\n"
            f"images           : {self.n_query} query / {self.n_reference} reference"
        )


def compare(
    features_query: Tensor,
    features_reference: Tensor,
    encoder_name: str,
    seed: int = 0,
) -> DomainDistance:
    """Both statistics over one pair of feature sets."""
    d_a, epsilon = proxy_a_distance(features_query, features_reference, seed=seed)
    kid, kid_std = kernel_distance(features_query, features_reference, seed=seed)
    return DomainDistance(
        d_a=d_a,
        discriminator_error=epsilon,
        kid=kid,
        kid_std=kid_std,
        n_query=len(features_query),
        n_reference=len(features_reference),
        encoder=encoder_name,
    )
