"""CLI: yolonas distill"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer


class ModelName(str, Enum):
    yolo_nas_s = "yolo_nas_s"
    yolo_nas_m = "yolo_nas_m"
    yolo_nas_l = "yolo_nas_l"


def distill(
    cache: Annotated[Path, typer.Option(help="Feature cache from tools/cache_teacher_features.py.")],
    images: Annotated[list[Path], typer.Option(help="Image directories named in the cache index.")],
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 20,
    batch_size: Annotated[int, typer.Option(help="Batch size per GPU.")] = 24,
    lr: Annotated[float, typer.Option(help="Peak learning rate.")] = 1e-3,
    weight_decay: Annotated[float, typer.Option(help="Weight decay.")] = 0.05,
    workers: Annotated[int, typer.Option(help="DataLoader workers.")] = 8,
    flip_prob: Annotated[float, typer.Option(help="Horizontal flip probability. Needs a cache built with --with-flip.")] = 0.5,
    hsv_prob: Annotated[float, typer.Option(help="Photometric jitter probability.")] = 0.5,
    output: Annotated[Path, typer.Option(help="Output directory.")] = Path("runs/distill"),
    amp: Annotated[bool, typer.Option("--amp/--no-amp", help="Mixed precision.")] = True,
    channels_last: Annotated[bool, typer.Option("--channels-last/--no-channels-last", help="NHWC activations.")] = True,
    compile_model: Annotated[bool, typer.Option("--compile/--no-compile", help="torch.compile the model.")] = False,
    val_fraction: Annotated[float, typer.Option(help="Fraction held out to watch the loss on.")] = 0.02,
    val_every: Annotated[int, typer.Option(help="Run validation every N epochs.")] = 1,
    wandb: Annotated[bool, typer.Option("--wandb/--no-wandb", help="Enable Weights & Biases logging.")] = False,
    wandb_project: Annotated[str, typer.Option(help="W&B project name.")] = "yolo-nas-distill",
    wandb_name: Annotated[str | None, typer.Option(help="W&B run name (auto-generated when omitted).")] = None,
    tensorboard: Annotated[bool, typer.Option("--tensorboard/--no-tensorboard", help="Enable TensorBoard logging. Worth it on a run this long: the CSV logger writes a file nobody can watch.")] = False,
    tensorboard_dir: Annotated[str, typer.Option(help="TensorBoard root log directory.")] = "runs/tensorboard",
    tensorboard_name: Annotated[str, typer.Option(help="TensorBoard experiment name (sub-directory under tensorboard-dir).")] = "distill",
):
    """Distil a teacher's patch features into the backbone.

    Stage 1 of the Apache-2.0 weights plan: this is what replaces the Objects365
    pretraining whose licence excludes us. It trains the *backbone only* against
    cached teacher features, changes no architecture, and writes a checkpoint the
    detection stage loads without surgery.

    Build the cache first::

        uv run tools/cache_teacher_features.py --images DIR --out CACHE --with-flip

    Then::

        yolonas distill --cache CACHE --images DIR --epochs 20 --tensorboard

    This runs for hours. Without ``--tensorboard`` or ``--wandb`` the only record is
    a CSV the Lightning logger flushes in batches, which is awkward to watch while
    the run is in progress -- so a long run should turn one of them on.
    """
    from typing import Any

    import lightning as L
    import torch
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from rich.console import Console
    from torch.utils.data import DataLoader, random_split

    from modern_yolonas import yolo_nas_l, yolo_nas_m, yolo_nas_s
    from modern_yolonas.training.distill import (
        BackboneDistillModule,
        CachedFeatureDataset,
        distill_collate_fn,
    )

    console = Console()
    dataset = CachedFeatureDataset(
        images=images, cache=cache, flip_prob=flip_prob, hsv_prob=hsv_prob
    )
    console.print(
        f"{len(dataset)} images | teacher grid {dataset.grid}x{dataset.grid}x{dataset.dim} "
        f"| flip cache: {dataset.has_flip}"
    )

    held_out = max(1, int(len(dataset) * val_fraction))
    train_set, val_set = random_split(
        dataset, [len(dataset) - held_out, held_out],
        generator=torch.Generator().manual_seed(0),
    )

    def loader(subset, shuffle: bool) -> DataLoader:
        return DataLoader(
            subset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
            collate_fn=distill_collate_fn, pin_memory=True, drop_last=shuffle,
            persistent_workers=workers > 0,
        )

    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    net = builders[model.value](pretrained=False, num_classes=80)
    if channels_last:
        net = net.to(memory_format=torch.channels_last)
    if compile_model:
        net = torch.compile(net)

    steps_per_epoch = max(1, len(train_set) // batch_size)
    module = BackboneDistillModule(
        model=net,
        teacher_dim=dataset.dim,
        lr=lr,
        weight_decay=weight_decay,
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
    if not loggers:
        loggers.append(L.pytorch.loggers.CSVLogger(str(output)))
        console.print(
            "[yellow]Logging to CSV only. A run of this length is easier to follow with "
            "--tensorboard.[/yellow]"
        )

    trainer = L.Trainer(
        max_epochs=epochs,
        precision="16-mixed" if amp else "32-true",
        default_root_dir=str(output),
        logger=loggers,
        check_val_every_n_epoch=val_every,
        callbacks=[
            ModelCheckpoint(dirpath=str(output), filename="backbone-{epoch:03d}",
                            monitor="val/loss", save_top_k=2, save_last=True),
            LearningRateMonitor(logging_interval="step"),
        ],
        log_every_n_steps=50,
    )
    trainer.fit(module, loader(train_set, True), loader(val_set, False))
    console.print(f"[green]Distilled backbone written to {output}[/green]")
