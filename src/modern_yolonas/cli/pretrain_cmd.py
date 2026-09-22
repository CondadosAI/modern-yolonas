"""CLI: yolonas pretrain"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer


class ModelName(str, Enum):
    yolo_nas_s = "yolo_nas_s"
    yolo_nas_m = "yolo_nas_m"
    yolo_nas_l = "yolo_nas_l"


def pretrain(
    images: Annotated[Path, typer.Option(help="Directory of unlabelled images, searched recursively.")],
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 100,
    batch_size: Annotated[int, typer.Option(help="Batch size per GPU. Contrastive learning wants this large.")] = 64,
    lr: Annotated[float, typer.Option(help="Peak learning rate.")] = 0.03,
    weight_decay: Annotated[float, typer.Option(help="Weight decay.")] = 1e-4,
    input_size: Annotated[int, typer.Option(help="Crop size. Smaller than detection's 640 because two views per image doubles the cost.")] = 448,
    min_scale: Annotated[float, typer.Option(help="Smallest random-resized-crop scale.")] = 0.2,
    momentum: Annotated[float, typer.Option(help="EMA factor for the key encoder.")] = 0.999,
    temperature: Annotated[float, typer.Option(help="Contrastive temperature.")] = 0.2,
    memory_bank_size: Annotated[int, typer.Option(help="Negatives for the global term.")] = 4096,
    lambda_dense: Annotated[float, typer.Option(help="Weight of the dense term. 0 reduces this to MoCo-v2.")] = 0.5,
    workers: Annotated[int, typer.Option(help="DataLoader workers.")] = 8,
    output: Annotated[Path, typer.Option(help="Output directory.")] = Path("runs/pretrain"),
    amp: Annotated[bool, typer.Option("--amp/--no-amp", help="Mixed precision.")] = True,
    channels_last: Annotated[bool, typer.Option("--channels-last/--no-channels-last", help="NHWC activations.")] = True,
    val_fraction: Annotated[float, typer.Option(help="Fraction held out to watch the loss on.")] = 0.02,
    wandb: Annotated[bool, typer.Option("--wandb/--no-wandb", help="Enable Weights & Biases logging.")] = False,
    wandb_project: Annotated[str, typer.Option(help="W&B project name.")] = "yolo-nas-pretrain",
    wandb_name: Annotated[str | None, typer.Option(help="W&B run name.")] = None,
    tensorboard: Annotated[bool, typer.Option("--tensorboard/--no-tensorboard", help="Enable TensorBoard logging.")] = False,
    tensorboard_dir: Annotated[str, typer.Option(help="TensorBoard root log directory.")] = "runs/tensorboard",
    tensorboard_name: Annotated[str, typer.Option(help="TensorBoard experiment name.")] = "pretrain",
):
    """Pretrain the backbone on unlabelled images, with DenseCL.

    Needs no annotations at all -- only a directory of images. That is the point:
    the common real situation is a large pile of unlabelled frames from your own
    domain and a COCO checkpoint trained on something else.

    The checkpoint it writes is the same shape the distillation stage writes, so
    detection training reads it the same way::

        yolonas pretrain --images ~/frames --model yolo_nas_s --epochs 100
        yolonas train --data ~/mydata --format coco --model yolo_nas_s \\
            --no-pretrained --init-backbone runs/pretrain/last.ckpt

    Whether this is worth the GPU time depends on how far your images are from
    COCO. ``yolonas domain-distance`` measures that before you spend the days.

    Requires the ``ssl`` extra::

        uv pip install 'modern-yolonas[ssl]'
    """
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from rich.console import Console
    from torch.utils.data import DataLoader, Subset

    from modern_yolonas import yolo_nas_l, yolo_nas_m, yolo_nas_s
    from modern_yolonas.training.pretrain import (
        DenseCLPretrainModule,
        UnlabelledImageFolder,
        build_transform,
    )

    console = Console()

    dataset = UnlabelledImageFolder(images, transform=build_transform(input_size, min_scale))
    console.print(f"{len(dataset)} unlabelled images under {images} | views at {input_size}px")

    held_out = max(1, int(len(dataset) * val_fraction))
    permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))
    train_set = Subset(dataset, permutation[held_out:].tolist())
    val_set = Subset(dataset, permutation[:held_out].tolist())

    def loader(subset, shuffle: bool) -> DataLoader:
        return DataLoader(
            subset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
            pin_memory=True, drop_last=True, persistent_workers=workers > 0,
        )

    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    net = builders[model.value](pretrained=False, num_classes=80)
    if channels_last:
        net = net.to(memory_format=torch.channels_last)

    steps_per_epoch = max(1, len(train_set) // batch_size)
    module = DenseCLPretrainModule(
        model=net,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        temperature=temperature,
        memory_bank_size=memory_bank_size,
        lambda_dense=lambda_dense,
        warmup_steps=min(1000, steps_per_epoch),
        max_steps=steps_per_epoch * epochs,
    )

    loggers: list[Any] = []
    if wandb:
        loggers.append(L.pytorch.loggers.WandbLogger(
            project=wandb_project, name=wandb_name, save_dir=str(output)))
        console.print(f"[green]W&B logging enabled → project={wandb_project!r}[/green]")
    if tensorboard:
        loggers.append(L.pytorch.loggers.TensorBoardLogger(tensorboard_dir, name=tensorboard_name))
        console.print(f"[green]TensorBoard → {tensorboard_dir}/{tensorboard_name}[/green]")

    trainer = L.Trainer(
        max_epochs=epochs,
        precision="16-mixed" if amp else "32-true",
        default_root_dir=str(output),
        logger=loggers,
        callbacks=[
            ModelCheckpoint(dirpath=str(output), filename="backbone-{epoch:03d}",
                            monitor="val/loss", save_top_k=2, save_last=True),
            LearningRateMonitor(logging_interval="step"),
        ],
        log_every_n_steps=50,
    )
    trainer.fit(module, loader(train_set, True), loader(val_set, False))
    console.print(
        f"[green]Pretrained backbone written to {output}[/green]\n"
        f"Use it with: yolonas train --no-pretrained --init-backbone {output}/last.ckpt"
    )
