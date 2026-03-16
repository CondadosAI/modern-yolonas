"""Tests for CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from modern_yolonas.cli import app

runner = CliRunner()


class TestVersionAndHelp:
    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "modern-yolonas" in result.output

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "detect" in result.output
        assert "train" in result.output
        assert "export" in result.output
        assert "eval" in result.output

    def test_detect_help(self):
        result = runner.invoke(app, ["detect", "--help"])
        assert result.exit_code == 0
        assert "--source" in result.output
        assert "--model" in result.output
        assert "--conf" in result.output

    def test_train_help(self):
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--data" in result.output
        assert "--epochs" in result.output
        assert "--format" in result.output

    def test_export_help(self):
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "--output" in result.output

    def test_eval_help(self):
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "--data" in result.output
        assert "--split" in result.output


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
