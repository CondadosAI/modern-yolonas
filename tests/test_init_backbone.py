"""Initialising detection training from a distilled backbone.

This is the hand-off between stage 1 and stage 3, and the failure it guards is
silent: a backbone half-loaded from a checkpoint trains, converges, and is simply
worse than one loaded properly. Nothing raises, and the only evidence is a lower
final AP -- which is exactly the number the stage 1 gate compares.
"""

from __future__ import annotations

import pytest
import torch

from modern_yolonas import yolo_nas_s
from modern_yolonas.weights import load_distilled_backbone


@pytest.fixture(scope="module")
def donor():
    """A model whose backbone carries recognisable weights."""
    model = yolo_nas_s(pretrained=False, num_classes=80)
    with torch.no_grad():
        for tensor in model.backbone.parameters():
            tensor.fill_(0.25)
    return model


def write_checkpoint(tmp_path, model, prefix: str = "model.backbone."):
    state = {prefix + k: v for k, v in model.backbone.state_dict().items()}
    path = tmp_path / "distill.ckpt"
    torch.save({"state_dict": state, "epoch": 19}, path)
    return path


class TestLoading:
    def test_the_backbone_actually_changes(self, tmp_path, donor):
        path = write_checkpoint(tmp_path, donor)
        target = yolo_nas_s(pretrained=False, num_classes=80)
        before = next(target.backbone.parameters()).clone()

        count = load_distilled_backbone(target, path)

        assert count == len(target.backbone.state_dict())
        assert not torch.equal(before, next(target.backbone.parameters()))
        assert torch.allclose(next(target.backbone.parameters()),
                              torch.full_like(before, 0.25))

    def test_every_backbone_tensor_matches_the_donor(self, tmp_path, donor):
        path = write_checkpoint(tmp_path, donor)
        target = yolo_nas_s(pretrained=False, num_classes=80)
        load_distilled_backbone(target, path)
        for key, value in donor.backbone.state_dict().items():
            assert torch.equal(target.backbone.state_dict()[key], value), key

    def test_the_neck_and_heads_are_left_alone(self, tmp_path, donor):
        """Only the backbone was distilled; touching the rest would be a lie."""
        path = write_checkpoint(tmp_path, donor)
        target = yolo_nas_s(pretrained=False, num_classes=80)
        neck_before = {k: v.clone() for k, v in target.neck.state_dict().items()}
        head_before = {k: v.clone() for k, v in target.heads.state_dict().items()}

        load_distilled_backbone(target, path)

        for key, value in neck_before.items():
            assert torch.equal(target.neck.state_dict()[key], value)
        for key, value in head_before.items():
            assert torch.equal(target.heads.state_dict()[key], value)

    def test_the_model_still_runs_after_loading(self, tmp_path, donor):
        path = write_checkpoint(tmp_path, donor)
        target = yolo_nas_s(pretrained=False, num_classes=80).eval()
        load_distilled_backbone(target, path)
        with torch.no_grad():
            boxes, scores = target(torch.randn(1, 3, 128, 128))
        assert boxes.shape[-1] == 4 and scores.shape[-1] == 80


class TestRefusals:
    def test_a_checkpoint_without_backbone_tensors_is_refused(self, tmp_path):
        path = tmp_path / "wrong.ckpt"
        torch.save({"state_dict": {"model.heads.cls_pred.weight": torch.zeros(3)}}, path)
        target = yolo_nas_s(pretrained=False, num_classes=80)
        with pytest.raises(ValueError, match="no 'model.backbone."):
            load_distilled_backbone(target, path)

    def test_a_partial_checkpoint_is_refused_rather_than_warned_about(self, tmp_path, donor):
        """The load-bearing test. A half-loaded backbone trains and converges and is
        merely worse, so this must fail loudly rather than log and continue."""
        state = {f"model.backbone.{k}": v
                 for k, v in donor.backbone.state_dict().items()}
        dropped = sorted(state)[0]
        del state[dropped]
        path = tmp_path / "partial.ckpt"
        torch.save({"state_dict": state}, path)

        target = yolo_nas_s(pretrained=False, num_classes=80)
        with pytest.raises(ValueError, match="absent from"):
            load_distilled_backbone(target, path)

    def test_a_bare_state_dict_without_the_lightning_wrapper_works(self, tmp_path, donor):
        """Not every checkpoint carries a 'state_dict' key."""
        state = {f"model.backbone.{k}": v
                 for k, v in donor.backbone.state_dict().items()}
        path = tmp_path / "bare.ckpt"
        torch.save(state, path)
        target = yolo_nas_s(pretrained=False, num_classes=80)
        assert load_distilled_backbone(target, path) == len(state)
