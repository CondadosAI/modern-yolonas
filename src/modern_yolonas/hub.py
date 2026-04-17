"""Hugging Face Hub integration for loading and pushing YOLO-NAS models."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file
from torch import nn


def from_hub(
    repo_id: str,
    variant: str = "yolo_nas_s",
    num_classes: int = 80,
    filename: str = "model.safetensors",
    revision: str | None = None,
) -> nn.Module:
    """Load a YOLO-NAS model from Hugging Face Hub.

    Args:
        repo_id: HF Hub repository ID (e.g. ``"user/yolonas-custom"``).
        variant: Model architecture variant.
        num_classes: Number of classes in the checkpoint.
        filename: Checkpoint filename in the repository.
        revision: Git revision (branch, tag, or commit hash).

    Returns:
        Loaded YoloNAS model.
    """
    from modern_yolonas.model import YoloNAS

    path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    model = YoloNAS.from_config(variant, num_classes=num_classes)
    model.load_state_dict(load_file(path), strict=False)
    return model


def push_to_hub(
    model: nn.Module,
    repo_id: str,
    filename: str = "model.safetensors",
    commit_message: str = "Upload YOLO-NAS model",
    private: bool = False,
) -> str:
    """Push a YOLO-NAS model to Hugging Face Hub as safetensors.

    Args:
        model: The model to upload.
        repo_id: HF Hub repository ID.
        filename: Filename for the checkpoint.
        commit_message: Git commit message.
        private: Whether to create a private repository.

    Returns:
        URL of the uploaded file.
    """
    import tempfile

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, private=private)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / filename
        save_file(model.state_dict(), str(path))
        return api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=filename,
            repo_id=repo_id,
            commit_message=commit_message,
        )
