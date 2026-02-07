"""Parse YOLO / Roboflow data.yaml files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    """Parsed dataset configuration from a YOLO ``data.yaml``."""

    root: Path
    num_classes: int
    class_names: list[str]
    train_split: str
    val_split: str


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Parse a YOLO / Roboflow ``data.yaml`` and return a :class:`DatasetConfig`.

    Expected YAML format::

        nc: 3
        names: ["person", "head", "hardhat"]
        train: images/train
        val: images/val

    The *root* is set to the parent directory of the YAML file, and
    *train_split* / *val_split* are the basenames of the train/val
    image directories (e.g. ``"train"`` from ``images/train``).

    Args:
        path: Path to the ``data.yaml`` file.

    Returns:
        A :class:`DatasetConfig` with the parsed fields.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If required keys (``nc``, ``names``) are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    nc = cfg["nc"]
    names = cfg["names"]

    train_raw = cfg.get("train", "images/train")
    val_raw = cfg.get("val", "images/val")

    # Extract the split name (last component of the path)
    train_split = Path(train_raw).name
    val_split = Path(val_raw).name

    return DatasetConfig(
        root=path.parent,
        num_classes=nc,
        class_names=list(names),
        train_split=train_split,
        val_split=val_split,
    )
