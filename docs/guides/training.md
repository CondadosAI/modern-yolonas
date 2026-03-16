# Training Guide

## Dataset format

modern-yolonas supports two dataset formats:

### YOLO format

```
dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/
│       ├── img_001.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img_001.txt    # one line per object
    │   └── ...
    └── val/
        ├── img_001.txt
        └── ...
```

Each label file has one line per object: `class_id x_center y_center width height` (normalized 0-1).

### COCO format

Standard COCO directory layout with `annotations/instances_train2017.json`.

## Training with the CLI

```bash
# Fine-tune from COCO pretrained weights
yolonas train \
    --model yolo_nas_s \
    --data /path/to/dataset \
    --format yolo \
    --epochs 100 \
    --batch-size 16 \
    --lr 2e-4 \
    --pretrained

# Train from scratch
yolonas train \
    --model yolo_nas_m \
    --data /path/to/dataset \
    --format coco \
    --epochs 300 \
    --no-pretrained
```

## Training with Python

```python
from modern_yolonas import yolo_nas_s
from modern_yolonas.data.yolo import YOLODetectionDataset
from modern_yolonas.data.transforms import (
    Compose, HSVAugment, HorizontalFlip, RandomAffine,
    LetterboxResize, Normalize,
)
from modern_yolonas.data.collate import detection_collate_fn
from modern_yolonas.training.trainer import Trainer
from torch.utils.data import DataLoader

model = yolo_nas_s(pretrained=True)

transforms = Compose([
    HSVAugment(),
    HorizontalFlip(),
    RandomAffine(degrees=0.0, translate=0.1, scale=(0.5, 1.5)),
    LetterboxResize(target_size=640),
    Normalize(),
])

train_ds = YOLODetectionDataset(
    "/path/to/data", split="train",
    transforms=transforms, input_size=640,
)
val_ds = YOLODetectionDataset(
    "/path/to/data", split="val",
    transforms=Compose([LetterboxResize(640), Normalize()]),
    input_size=640,
)

train_loader = DataLoader(
    train_ds, batch_size=16, shuffle=True,
    collate_fn=detection_collate_fn, num_workers=8,
)
val_loader = DataLoader(
    val_ds, batch_size=16, shuffle=False,
    collate_fn=detection_collate_fn, num_workers=8,
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    lr=2e-4,
    output_dir="runs/my_experiment",
)
trainer.train()
```

## Checkpoints

Checkpoints are saved to the output directory:

- `last.pt` — latest checkpoint (every epoch)
- `epoch_50.pt`, `epoch_100.pt`, ... — periodic saves

Resume training:

```bash
yolonas train --data /path --resume runs/train/last.pt
```
