"""Tests for CLI commands."""

import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
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
                mock_result = type("Detection", (), {
                    "boxes": np.zeros((0, 4)),
                    "scores": np.zeros(0),
                    "class_ids": np.zeros(0),
                    "image": img,
                    "save": lambda self, path, **kw: cv2.imwrite(str(path), img),
                })()
                mock_det.return_value = mock_result

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
                mock_result = type("Detection", (), {
                    "boxes": np.zeros((0, 4)),
                    "scores": np.zeros(0),
                    "class_ids": np.zeros(0),
                    "image": np.zeros((32, 32, 3), dtype=np.uint8),
                    "save": lambda self, path, **kw: cv2.imwrite(str(path), np.zeros((32, 32, 3), dtype=np.uint8)),
                })()
                mock_det.return_value = mock_result

                result = runner.invoke(app, [
                    "detect",
                    "--source", tmpdir,
                    "--output", str(out_dir),
                    "--device", "cpu",
                ])
                assert result.exit_code == 0
