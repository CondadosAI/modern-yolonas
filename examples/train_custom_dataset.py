"""Train YOLO-NAS on a custom YOLO-format dataset.

Usage:
    python examples/train_custom_dataset.py --data /path/to/dataset --epochs 50
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from modern_yolonas import yolo_nas_s
from modern_yolonas.data.collate import detection_collate_fn
from modern_yolonas.data.transforms import (
    Compose,
    HorizontalFlip,
    HSVAugment,
    LetterboxResize,
    Normalize,
    RandomAffine,
)
from modern_yolonas.data.yolo import YOLODetectionDataset
from modern_yolonas.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train YOLO-NAS on a custom dataset")
    parser.add_argument("--data", required=True, help="Path to YOLO-format dataset root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="runs/custom_train")
    args = parser.parse_args()

    # Build model with pretrained COCO backbone
    model = yolo_nas_s(pretrained=True)

    # Training augmentations
    train_transforms = Compose([
        HSVAugment(),
        HorizontalFlip(),
        RandomAffine(degrees=0.0, translate=0.1, scale=(0.5, 1.5)),
        LetterboxResize(target_size=args.input_size),
        Normalize(),
    ])

    # Validation: just resize + normalize
    val_transforms = Compose([
        LetterboxResize(target_size=args.input_size),
        Normalize(),
    ])

    train_ds = YOLODetectionDataset(args.data, split="train", transforms=train_transforms, input_size=args.input_size)
    val_ds = YOLODetectionDataset(args.data, split="val", transforms=val_transforms, input_size=args.input_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=detection_collate_fn, num_workers=8, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=detection_collate_fn, num_workers=8, pin_memory=True,
    )

    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output,
        device=args.device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
