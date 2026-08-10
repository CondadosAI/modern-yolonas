"""Tests for weight loading and state_dict remapping."""

import torch

from modern_yolonas.model import YoloNAS
from modern_yolonas.weights import (
    HF_REPO_ID,
    WEIGHT_FILES,
    _strip_prefix,
    filter_loadable,
    remap_state_dict,
)


class TestStripPrefix:
    def test_no_prefix(self):
        assert _strip_prefix("backbone.stem.conv.weight") == "backbone.stem.conv.weight"

    def test_net_prefix(self):
        assert _strip_prefix("net.backbone.stem.conv.weight") == "backbone.stem.conv.weight"

    def test_module_prefix(self):
        assert _strip_prefix("module.backbone.stem.conv.weight") == "backbone.stem.conv.weight"

    def test_ema_prefix(self):
        assert _strip_prefix("ema_model.backbone.stem.conv.weight") == "backbone.stem.conv.weight"

    def test_double_prefix(self):
        # All matching prefixes are stripped
        result = _strip_prefix("net.module.backbone.stem.conv.weight")
        assert result == "backbone.stem.conv.weight"

    def test_empty_key(self):
        assert _strip_prefix("") == ""


class TestRemapStateDict:
    def test_basic_remap(self):
        raw = {
            "net.backbone.stem.conv.weight": torch.randn(48, 3, 3, 3),
            "net.backbone.stem.bn.weight": torch.randn(48),
            "net.neck.neck1.conv.weight": torch.randn(96, 48, 1, 1),
        }
        remapped = remap_state_dict(raw)
        assert "backbone.stem.conv.weight" in remapped
        assert "backbone.stem.bn.weight" in remapped
        assert "neck.neck1.conv.weight" in remapped
        assert len(remapped) == 3

    def test_preserves_tensor_values(self):
        t = torch.randn(10, 10)
        raw = {"net.layer.weight": t}
        remapped = remap_state_dict(raw)
        assert torch.equal(remapped["layer.weight"], t)

    def test_empty_dict(self):
        assert remap_state_dict({}) == {}

    def test_no_prefix_keys_unchanged(self):
        raw = {
            "backbone.stem.conv.weight": torch.randn(48, 3, 3, 3),
            "heads.head1.cls.weight": torch.randn(80, 192),
        }
        remapped = remap_state_dict(raw)
        assert "backbone.stem.conv.weight" in remapped
        assert "heads.head1.cls.weight" in remapped


class TestWeightFiles:
    def test_all_variants_have_files(self):
        assert "yolo_nas_s" in WEIGHT_FILES
        assert "yolo_nas_m" in WEIGHT_FILES
        assert "yolo_nas_l" in WEIGHT_FILES

    def test_files_are_safetensors(self):
        for filename in WEIGHT_FILES.values():
            assert isinstance(filename, str)
            assert filename.endswith(".safetensors")

    def test_default_repo_id(self):
        assert "/" in HF_REPO_ID


class TestFilterLoadable:
    def test_keeps_matching_entries(self):
        model = YoloNAS.from_config("yolo_nas_s", num_classes=80)
        kept, mismatched = filter_loadable(model.state_dict(), model)
        assert kept.keys() == model.state_dict().keys()
        assert mismatched == []

    def test_drops_unknown_keys(self):
        model = YoloNAS.from_config("yolo_nas_s", num_classes=80)
        sd = {"not.a.real.key": torch.randn(4)}
        kept, mismatched = filter_loadable(sd, model)
        assert kept == {}
        # An unknown key is not a shape mismatch — it is simply absent.
        assert mismatched == []

    def test_reports_shape_mismatches(self):
        model = YoloNAS.from_config("yolo_nas_s", num_classes=80)
        key = "heads.head1.cls_pred.weight"
        sd = {key: torch.randn(3, 64, 1, 1)}
        kept, mismatched = filter_loadable(sd, model)
        assert kept == {}
        assert mismatched == [key]


class TestPartialLoadAcrossNumClasses:
    """Regression: a checkpoint with a different ``num_classes`` must load partially.

    ``load_state_dict(strict=False)`` tolerates missing/unexpected keys but still
    raises on a shape mismatch, so the reshaped classification heads have to be
    dropped before loading. This is the documented fine-tuning path.
    """

    def test_partial_load_succeeds_and_transfers_backbone(self):
        pretrained_sd = YoloNAS.from_config("yolo_nas_s", num_classes=80).state_dict()
        model = YoloNAS.from_config("yolo_nas_s", num_classes=3)

        stem = "backbone.stem.conv.branch_3x3.conv.weight"
        before = model.state_dict()[stem].clone()

        kept, mismatched = filter_loadable(pretrained_sd, model)
        model.load_state_dict(kept, strict=False)

        # The reshaped heads were skipped, the backbone was not.
        assert any("cls_pred" in k for k in mismatched)
        assert stem not in mismatched
        assert torch.equal(model.state_dict()[stem], pretrained_sd[stem])
        assert not torch.equal(model.state_dict()[stem], before)

        # Head keeps its new class count, still at initialization.
        assert model.state_dict()["heads.head1.cls_pred.weight"].shape[0] == 3
