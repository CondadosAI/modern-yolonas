"""Shared pytest configuration.

Augmentations fire on a coin flip, so an unseeded run takes a different set of
branches each time — which makes both flaky failures and a coverage percentage
that drifts across runs. Seeding every test keeps both reproducible.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_rng():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
