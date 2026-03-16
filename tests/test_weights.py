"""Tests for weight loading and state_dict remapping."""

import os
from unittest.mock import patch

import torch

from modern_yolonas.weights import (
    CACHE_DIR,
    WEIGHT_URLS,
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


class TestWeightURLs:
    def test_all_variants_have_urls(self):
        assert "yolo_nas_s" in WEIGHT_URLS
        assert "yolo_nas_m" in WEIGHT_URLS
        assert "yolo_nas_l" in WEIGHT_URLS

    def test_urls_are_strings(self):
        for url in WEIGHT_URLS.values():
            assert isinstance(url, str)
            assert url.startswith("https://")


class TestCacheDir:
    def test_default_cache_dir(self):
        assert "modern_yolonas" in str(CACHE_DIR)

    def test_custom_cache_dir(self):
        with patch.dict(os.environ, {"YOLONAS_CACHE_DIR": "/tmp/custom_cache"}):
            # Re-import to pick up the env var
            from importlib import reload
            import modern_yolonas.weights as w

            reload(w)
            assert str(w.CACHE_DIR) == "/tmp/custom_cache"

            # Restore
            reload(w)
