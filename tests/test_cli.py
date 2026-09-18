"""Tests for CLI commands."""

import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import supervision as sv
from typer.testing import CliRunner

from modern_yolonas.cli import app

runner = CliRunner()

# Typer injects Rich ANSI color codes *inside* option names (e.g. `--source` becomes
# `\x1b[1;36m-\x1b[0m\x1b[1;36m-source\x1b[0m`), which breaks substring matches in CI
# where FORCE_COLOR / a non-tty but colored-by-default environment is set.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    return _ANSI.sub("", output)


class TestVersionAndHelp:
    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "modern-yolonas" in _plain(result.output)

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        out = _plain(result.output)
        assert "detect" in out
        assert "train" in out
        assert "export" in out
        assert "eval" in out

    def test_detect_help(self):
        result = runner.invoke(app, ["detect", "--help"])
        assert result.exit_code == 0
        out = _plain(result.output)
        assert "--source" in out
        assert "--model" in out
        assert "--conf" in out

    def test_train_help(self):
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        out = _plain(result.output)
        assert "--data" in out
        assert "--epochs" in out
        assert "--format" in out

    def test_export_help(self):
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        out = _plain(result.output)
        assert "--format" in out
        assert "--output" in out

    def test_eval_help(self):
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        out = _plain(result.output)
        assert "--data" in out
        assert "--split" in out

    def test_quantize_help(self):
        result = runner.invoke(app, ["quantize", "--help"])
        assert result.exit_code == 0
        out = _plain(result.output)
        assert "--data" in out
        assert "--backend" in out
        assert "--num-batches" in out

    def test_qat_help(self):
        result = runner.invoke(app, ["qat", "--help"])
        assert result.exit_code == 0
        out = _plain(result.output)
        assert "--data" in out
        assert "--backend" in out
        assert "--epochs" in out

    def test_benchmark_dataset_lists_subcommands(self):
        result = runner.invoke(app, ["benchmark-dataset", "--help"])
        assert result.exit_code == 0
        out = _plain(result.output)
        assert "coco" in out
        assert "rf100vl" in out

    @pytest.mark.parametrize("subcommand", ["coco", "rf100vl"])
    def test_benchmark_dataset_subcommand_help(self, subcommand):
        result = runner.invoke(app, ["benchmark-dataset", subcommand, "--help"])
        assert result.exit_code == 0
        out = _plain(result.output)
        assert "--download" in out
        assert "--logger" in out

    def test_benchmark_is_still_the_latency_one(self):
        # `benchmark` shipped in 0.4.0 measuring latency; the accuracy benchmarks
        # live under `benchmark-dataset` rather than taking its name.
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "latency" in _plain(result.output).lower()


class TestDetectValidation:
    def test_invalid_source_type(self):
        result = runner.invoke(app, ["detect", "--source", "file.xyz"])
        assert result.exit_code != 0

    def test_nonexistent_source(self):
        result = runner.invoke(app, ["detect", "--source", "/nonexistent/image.jpg"])
        assert result.exit_code != 0

    def test_invalid_model(self):
        result = runner.invoke(app, ["detect", "--source", "test.jpg", "--model", "invalid_model"])
        assert result.exit_code != 0


class TestDetectIntegration:
    @pytest.mark.slow
    def test_detect_synthetic_image(self):
        """Detect on a small synthetic image with pretrained=False."""
        import cv2

        with tempfile.TemporaryDirectory() as tmpdir:
            img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            img_path = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(img_path), img)

            out_dir = Path(tmpdir) / "output"

            with patch("modern_yolonas.inference.detect.Detector") as mock_cls:
                # Mock detector to avoid weight download
                mock_det = mock_cls.return_value
                mock_det.return_value = sv.Detections.empty()
                mock_det.annotate.side_effect = lambda image, detections: image

                result = runner.invoke(app, [
                    "detect",
                    "--source", str(img_path),
                    "--output", str(out_dir),
                    "--device", "cpu",
                ])
                assert result.exit_code == 0

    @pytest.mark.slow
    def test_detect_directory(self):
        """Detect on a directory of synthetic images."""
        import cv2

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                cv2.imwrite(str(Path(tmpdir) / f"img_{i}.jpg"), img)

            out_dir = Path(tmpdir) / "output"

            with patch("modern_yolonas.inference.detect.Detector") as mock_cls:
                mock_det = mock_cls.return_value
                mock_det.return_value = sv.Detections.empty()
                mock_det.annotate.side_effect = lambda image, detections: image

                result = runner.invoke(app, [
                    "detect",
                    "--source", tmpdir,
                    "--output", str(out_dir),
                    "--device", "cpu",
                ])
                assert result.exit_code == 0


class TestCLIConfig:
    def test_load_config_basic(self, tmp_path):
        from modern_yolonas.cli.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("epochs: 100\nbatch-size: 32\n")
        result = load_config(cfg_file)
        assert result["epochs"] == 100
        assert result["batch_size"] == 32  # hyphen → underscore

    def test_load_config_empty(self, tmp_path):
        from modern_yolonas.cli.config import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("")
        assert load_config(cfg_file) == {}

    def test_load_config_not_found(self):
        from modern_yolonas.cli.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_merge_config_cli_wins(self):
        from modern_yolonas.cli.config import merge_config

        config = {"epochs": 100, "batch_size": 32, "device": "cuda"}
        cli = {"epochs": 50, "batch_size": None}  # None = not explicitly set by user
        result = merge_config(config, cli)
        assert result["epochs"] == 50      # CLI wins
        assert result["batch_size"] == 32  # None skipped, config value kept
        assert result["device"] == "cuda"  # from config only

    def test_merge_config_all_none(self):
        from modern_yolonas.cli.config import merge_config

        config = {"epochs": 100}
        result = merge_config(config, {"epochs": None})
        assert result["epochs"] == 100
