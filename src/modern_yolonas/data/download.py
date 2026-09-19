"""Dataset download utilities for COCO and RF100-VL."""

from __future__ import annotations

from pathlib import Path


def download_coco(dest: str | Path) -> Path:
    """Download COCO 2017 train+val using FiftyOne.

    Creates the standard layout::

        dest/
            images/train2017/
            images/val2017/
            annotations/instances_train2017.json
            annotations/instances_val2017.json

    Args:
        dest: Destination directory.

    Returns:
        Path to the dataset root.
    """
    try:
        import fiftyone.zoo as foz  # noqa: F401 (fiftyone must be installed)
    except ImportError:
        raise ImportError(
            "FiftyOne is required for COCO download. "
            "Install with: pip install fiftyone"
        )

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation"):
        foz.load_zoo_dataset(
            "coco-2017",
            split=split,
            dataset_dir=str(dest / "fiftyone"),
        )

    # Symlink to standard COCO layout
    img_dir = dest / "images"
    img_dir.mkdir(exist_ok=True)
    ann_dir = dest / "annotations"
    ann_dir.mkdir(exist_ok=True)

    fo_root = dest / "fiftyone" / "coco-2017"

    # Train
    train_src = fo_root / "train" / "data"
    train_dst = img_dir / "train2017"
    if not train_dst.exists():
        train_dst.symlink_to(train_src.resolve())

    train_ann_src = fo_root / "train" / "labels.json"
    train_ann_dst = ann_dir / "instances_train2017.json"
    if not train_ann_dst.exists():
        train_ann_dst.symlink_to(train_ann_src.resolve())

    # Validation
    val_src = fo_root / "validation" / "data"
    val_dst = img_dir / "val2017"
    if not val_dst.exists():
        val_dst.symlink_to(val_src.resolve())

    val_ann_src = fo_root / "validation" / "labels.json"
    val_ann_dst = ann_dir / "instances_val2017.json"
    if not val_ann_dst.exists():
        val_ann_dst.symlink_to(val_ann_src.resolve())

    return dest


def download_rf100vl(dest: str | Path) -> Path:
    """Download RF100-VL datasets.

    Args:
        dest: Destination directory.

    Returns:
        Path to the RF100-VL root directory.
    """
    try:
        from rf100vl import download_rf100vl as _download
    except ImportError:
        raise ImportError(
            "rf100vl package is required for RF100-VL download. "
            "Install with: pip install rf100vl"
        )

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _download(str(dest))
    return dest
