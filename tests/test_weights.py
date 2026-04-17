"""Tests for weight loading and state_dict remapping."""

import torch

from modern_yolonas.weights import (
    HF_REPO_ID,
    WEIGHT_FILES,
    _strip_prefix,
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
