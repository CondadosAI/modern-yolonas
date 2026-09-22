"""Tests for self-supervised pretraining.

The load-bearing one is `test_checkpoint_loads_as_a_distilled_backbone`: pretraining
is only useful if detection training can read what it writes, and that contract is a
string prefix nothing else enforces.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from modern_yolonas import yolo_nas_s
from modern_yolonas.weights import load_distilled_backbone

lightly = pytest.importorskip("lightly", reason="pretraining needs the `ssl` extra")

from modern_yolonas.training.pretrain import (  # noqa: E402
    DenseCLPretrainModule,
    UnlabelledImageFolder,
    build_transform,
)


@pytest.fixture
def image_dir(tmp_path):
    """Four synthetic images, in a nested directory to exercise the recursion."""
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(2):
        Image.fromarray(rng.integers(0, 255, (128, 160, 3), dtype=np.uint8)).save(
            tmp_path / f"{i}.jpg"
        )
        Image.fromarray(rng.integers(0, 255, (128, 160, 3), dtype=np.uint8)).save(
            nested / f"{i}.png"
        )
    return tmp_path


def test_folder_finds_images_recursively(image_dir):
    dataset = UnlabelledImageFolder(image_dir, transform=build_transform(input_size=64))
    assert len(dataset) == 4
    view0, view1 = dataset[0]
    assert view0.shape == (3, 64, 64)
    assert view1.shape == (3, 64, 64)
    # Two different crops of the same image, not the same tensor twice.
    assert not torch.allclose(view0, view1)


def test_folder_raises_on_an_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="No images under"):
        UnlabelledImageFolder(tmp_path, transform=build_transform())


def test_views_are_not_imagenet_normalised(image_dir):
    """The detection path feeds uint8/255, so SSL must too.

    An ImageNet-normalised view has negative values; a /255 one does not. Getting
    this wrong shifts the input distribution between pretraining and fine-tuning,
    and nothing downstream would raise.
    """
    dataset = UnlabelledImageFolder(image_dir, transform=build_transform(input_size=64))
    view0, _ = dataset[0]
    assert view0.min() >= 0.0
    assert view0.max() <= 1.0


def test_match_finds_the_identical_token():
    """A query token matched against a key set containing its exact copy picks it."""
    torch.manual_seed(0)
    key = torch.randn(2, 5, 8)
    # Query pixel i is a scaled copy of key pixel (4 - i): cosine ignores the scale,
    # so the match must invert the order regardless of magnitude.
    query = key.flip(1) * 3.0
    indices = DenseCLPretrainModule.match(query, key)
    expected = torch.tensor([[4, 3, 2, 1, 0], [4, 3, 2, 1, 0]])
    assert torch.equal(indices, expected)


@pytest.fixture
def module():
    net = yolo_nas_s(pretrained=False, num_classes=80)
    return DenseCLPretrainModule(model=net, memory_bank_size=64, max_steps=10, warmup_steps=1)


def test_momentum_branch_takes_no_gradient(module):
    assert all(not p.requires_grad for p in module.backbone_momentum.parameters())
    assert all(not p.requires_grad for p in module.global_head_momentum.parameters())
    assert all(not p.requires_grad for p in module.dense_head_momentum.parameters())


def test_optimiser_leaves_the_momentum_branch_alone(module):
    optimised = {id(p) for group in module.configure_optimizers()["optimizer"].param_groups
                 for p in group["params"]}
    assert not any(id(p) in optimised for p in module.backbone_momentum.parameters())
    # And it does include what should be trained.
    assert all(id(p) in optimised for p in module.model.backbone.parameters())


def test_momentum_update_moves_the_key_encoder_by_one_minus_m(module):
    """`update_momentum` must produce k <- m*k + (1-m)*q, on the backbone and both heads.

    Asserted in closed form rather than as "it changed": a no-op update and a
    wrong-direction update both leave the loss falling.
    """
    query = next(module.model.backbone.parameters())
    key = next(module.backbone_momentum.parameters())
    with torch.no_grad():
        query.add_(1.0)  # force a gap between the two branches
    before = key.detach().clone()
    m = module.momentum

    module._update_momentum(module.model.backbone, module.backbone_momentum, m=m)

    expected = m * before + (1 - m) * query.detach()
    assert torch.allclose(key, expected, atol=1e-7)


def test_a_training_step_produces_a_finite_loss(module, image_dir):
    dataset = UnlabelledImageFolder(image_dir, transform=build_transform(input_size=64))
    batch = torch.utils.data.default_collate([dataset[i] for i in range(4)])
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_checkpoint_loads_as_a_distilled_backbone(tmp_path, image_dir):
    """The whole point: `yolonas train --init-backbone` must read what this writes.

    The contract is the `model.backbone.` key prefix, which holds only because the
    module keeps the detection model at `self.model`. Nothing else enforces it, and
    breaking it would surface as a ValueError days into a training run.
    """
    import lightning as L

    dataset = UnlabelledImageFolder(image_dir, transform=build_transform(input_size=64))
    net = yolo_nas_s(pretrained=False, num_classes=80)
    module = DenseCLPretrainModule(model=net, memory_bank_size=8, max_steps=2, warmup_steps=1)

    trainer = L.Trainer(
        max_steps=2, logger=False, enable_checkpointing=False, enable_progress_bar=False,
        accelerator="cpu", default_root_dir=str(tmp_path),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    trainer.fit(module, loader)

    checkpoint = tmp_path / "pretrained.ckpt"
    trainer.save_checkpoint(checkpoint)

    fresh = yolo_nas_s(pretrained=False, num_classes=80)
    loaded = load_distilled_backbone(fresh, checkpoint)
    assert loaded == len(fresh.backbone.state_dict())

    # And the weights are the *query* encoder's, not the momentum copy's. The two
    # branches have to be distinct first, or comparing against the query encoder
    # says nothing -- an implementation that saved the momentum copy would satisfy
    # the loop below by aliasing.
    assert module.model.backbone is not module.backbone_momentum
    query_state = module.model.backbone.state_dict()
    momentum_state = module.backbone_momentum.state_dict()
    assert any(
        not torch.allclose(query_state[k], momentum_state[k]) for k in query_state
    ), "query and momentum encoders are identical; the momentum update did not run"

    for key, value in query_state.items():
        assert torch.allclose(fresh.backbone.state_dict()[key], value)


def test_lambda_dense_zero_really_skips_the_dense_branch(image_dir):
    """The CLI documents lambda_dense=0 as MoCo-v2, so it must not run DenseCL.

    Weighting the dense term by zero would still compute the [B, HW, HW]
    correspondence and still push into the dense memory bank -- the loss value
    would be right and everything downstream would be wrong about what was run.
    """
    net = yolo_nas_s(pretrained=False, num_classes=80)
    module = DenseCLPretrainModule(
        model=net, memory_bank_size=8, max_steps=2, warmup_steps=1, lambda_dense=0.0
    )
    dataset = UnlabelledImageFolder(image_dir, transform=build_transform(input_size=64))
    batch = torch.utils.data.default_collate([dataset[i] for i in range(4)])

    calls = []
    original = DenseCLPretrainModule.match
    DenseCLPretrainModule.match = staticmethod(
        lambda q, k: calls.append(1) or original(q, k)
    )
    try:
        loss = module.training_step(batch, 0)
    finally:
        DenseCLPretrainModule.match = staticmethod(original)

    assert calls == [], "the dense correspondence ran with lambda_dense=0"
    assert torch.isfinite(loss)
