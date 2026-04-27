"""CLI: yolonas train"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml


class ModelName(str, Enum):
    yolo_nas_s = "yolo_nas_s"
    yolo_nas_m = "yolo_nas_m"
    yolo_nas_l = "yolo_nas_l"


class DataFormat(str, Enum):
    yolo = "yolo"
    coco = "coco"


def train(
    data: Annotated[str, typer.Option(help="Path to dataset root.")],
    config: Annotated[str | None, typer.Option("--config", "-c", help="Path to YAML config file. CLI flags override config values.")] = None,
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    data_format: Annotated[DataFormat, typer.Option("--format", help="Dataset format.")] = DataFormat.yolo,
    epochs: Annotated[int, typer.Option(help="Number of training epochs.")] = 300,
    batch_size: Annotated[int, typer.Option(help="Batch size per GPU.")] = 32,
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 2e-4,
    device: Annotated[str, typer.Option(help="Device.")] = "cuda",
    output: Annotated[str, typer.Option(help="Output directory.")] = "runs/train",
    resume_path: Annotated[str | None, typer.Option("--resume", help="Checkpoint path to resume from.")] = None,
    input_size: Annotated[int, typer.Option(help="Model input size.")] = 640,
    workers: Annotated[int, typer.Option(help="DataLoader workers.")] = 8,
    pretrained: Annotated[bool, typer.Option("--pretrained/--no-pretrained", help="Use pretrained COCO weights.")] = True,
    num_classes: Annotated[int, typer.Option(help="Number of classes (auto-detected from dataset when 0).")] = 0,
    # Logging
    wandb: Annotated[bool, typer.Option("--wandb/--no-wandb", help="Enable Weights & Biases logging.")] = False,
    wandb_project: Annotated[str, typer.Option(help="W&B project name.")] = "yolo-nas",
    wandb_name: Annotated[str | None, typer.Option(help="W&B run name (auto-generated when omitted).")] = None,
    tensorboard: Annotated[bool, typer.Option("--tensorboard/--no-tensorboard", help="Enable TensorBoard logging.")] = False,
    tensorboard_dir: Annotated[str, typer.Option(help="TensorBoard root log directory.")] = "runs/tensorboard",
    tensorboard_name: Annotated[str, typer.Option(help="TensorBoard experiment name (sub-directory under tensorboard-dir).")] = "default",
    val_freq: Annotated[int, typer.Option(help="Run validation every N epochs.")] = 10,
    compile_model: Annotated[bool, typer.Option("--compile/--no-compile", help="torch.compile the model for faster training (PyTorch 2+).")] = False,
    gradient_accum: Annotated[int, typer.Option(help="Gradient accumulation steps (1 = disabled).")] = 1,
    early_stopping_patience: Annotated[int, typer.Option(help="Stop training if train loss doesn't improve for N epochs (0 = disabled).")] = 0,
    early_stopping_min_delta: Annotated[float, typer.Option(help="Minimum improvement in train loss to count as progress.")] = 1e-4,
    amp: Annotated[bool, typer.Option("--amp/--no-amp", help="Automatic mixed precision training (fp16). Reduces VRAM and speeds up training on Ampere+ GPUs.")] = True,
    num_gpus: Annotated[int, typer.Option(help="Number of GPUs for DDP training. 1 = single GPU. Values >1 spawn child processes via torchrun.")] = 1,
    ignore_empty: Annotated[bool, typer.Option("--ignore-empty/--no-ignore-empty", help="Skip images with zero annotations (background-only samples).")] = True,
    # COCO-specific path overrides
    train_images: Annotated[str | None, typer.Option(help="[COCO] Training images directory. Defaults to <data>/images/train.")] = None,
    val_images: Annotated[str | None, typer.Option(help="[COCO] Validation images directory. Defaults to <data>/images/val.")] = None,
    train_ann: Annotated[str | None, typer.Option(help="[COCO] Training annotation JSON. Defaults to <data>/annotations/train.json.")] = None,
    val_ann: Annotated[str | None, typer.Option(help="[COCO] Validation annotation JSON. Defaults to <data>/annotations/val.json.")] = None,
):
    """Train a YOLO-NAS model."""
    from rich.console import Console

    # --- Config file -------------------------------------------------------
    # Load YAML first; any flag that still holds its default value is replaced
    # by the config value, so explicit CLI flags always win.
    # Supports a nested layout with top-level "model" and "train" sections:
    #
    #   model:
    #     type: yolo_nas_s
    #     num_classes: 3
    #     input-size: 320
    #   train:
    #     epochs: 100
    #     batch-size: 64
    #     ...
    #
    # as well as the original flat layout.
    if config is not None:
        cfg: dict[str, Any] = yaml.safe_load(Path(config).read_text()) or {}

        # Normalise: if nested sections exist, merge them into a flat dict
        cfg_model: dict[str, Any] = {}
        cfg_train: dict[str, Any] = {}
        if isinstance(cfg.get("model"), dict):
            cfg_model = cfg.pop("model")
            # "type" maps to the "model" CLI flag
            if "type" in cfg_model:
                cfg_model["model"] = cfg_model.pop("type")
        if isinstance(cfg.get("train"), dict):
            cfg_train = cfg.pop("train")
        flat: dict[str, Any] = {**cfg_train, **cfg_model, **cfg}

        def _pick(val: Any, key: str, default: Any) -> Any:
            """Return val if it differs from default, otherwise fall back to flat cfg."""
            return val if val != default else flat.get(key, val)

        model_str = _pick(model.value, "model", ModelName.yolo_nas_s.value)
        model      = ModelName(model_str)
        epochs     = _pick(epochs,      "epochs",      300)
        batch_size = _pick(batch_size,  "batch-size",  32)
        lr         = float(_pick(lr,          "lr",          2e-4))
        device     = _pick(device,      "device",      "cuda")
        output     = _pick(output,      "output",      "runs/train")
        input_size = _pick(input_size,  "input-size",  640)
        workers    = _pick(workers,     "workers",     8)
        pretrained = _pick(pretrained,  "pretrained",  True)
        if num_classes == 0:
            num_classes = int(flat.get("num_classes", flat.get("num-classes", 0)))
        compile_model   = _pick(compile_model,   "compile",        False)
        val_freq        = _pick(val_freq,         "val-freq",       10)
        gradient_accum  = int(_pick(gradient_accum, "gradient_accum", 1))
        early_stopping_patience  = int(_pick(early_stopping_patience,  "early-stopping-patience",  0))
        early_stopping_min_delta = float(_pick(early_stopping_min_delta, "early-stopping-min-delta", 1e-4))
        amp      = bool(_pick(amp,      "amp",       True))
        num_gpus = int(_pick(num_gpus,  "num-gpus",  1))
        ignore_empty = bool(_pick(ignore_empty, "ignore-empty", True))

    # -----------------------------------------------------------------------
    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l
    from modern_yolonas.data.transforms import Compose, HSVAugment, HorizontalFlip, RandomAffine, RandomChannelSwap, LetterboxResize, Normalize, Mixup
    from modern_yolonas.data.collate import detection_collate_fn
    from modern_yolonas.training.trainer import Trainer
    from torch.utils.data import DataLoader

    console = Console()
    data_path = Path(data)

    train_transforms = Compose([
        HSVAugment(p=0.5),
        HorizontalFlip(),
        RandomAffine(degrees=0.0, translate=0.25, scale=(0.5, 1.5)),
        RandomChannelSwap(p=0.5),
        LetterboxResize(target_size=input_size),
        # Mixup is appended here after the dataset is created (needs dataset reference)
        Normalize(),
    ])
    val_transforms = Compose([LetterboxResize(target_size=input_size), Normalize()])

    if data_format == DataFormat.yolo:
        from modern_yolonas.data.yolo import YOLODetectionDataset

        train_dataset = YOLODetectionDataset(data, split="train", transforms=train_transforms, input_size=input_size, ignore_empty_annotations=ignore_empty)
        val_dataset = YOLODetectionDataset(data, split="val", transforms=val_transforms, input_size=input_size)
        if num_classes == 0:
            num_classes = train_dataset.num_classes

    else:  # COCO format
        from modern_yolonas.data.coco import COCODetectionDataset

        # Resolve paths with sensible defaults matching the most common layouts
        _train_images = Path(train_images) if train_images else data_path / "train"
        _val_images   = Path(val_images)   if val_images   else data_path / "val"
        _train_ann    = Path(train_ann)    if train_ann    else data_path / "annotations" / "train.json"
        _val_ann      = Path(val_ann)      if val_ann      else data_path / "annotations" / "val.json"

        # Fallback to the standard COCO 2017 layout if the defaults don't exist
        if not _train_images.exists() and (data_path / "images" / "train2017").exists():
            _train_images = data_path / "images" / "train2017"
        if not _val_images.exists() and (data_path / "images" / "val2017").exists():
            _val_images = data_path / "images" / "val2017"
        if not _train_ann.exists() and (data_path / "annotations" / "instances_train2017.json").exists():
            _train_ann = data_path / "annotations" / "instances_train2017.json"
        if not _val_ann.exists() and (data_path / "annotations" / "instances_val2017.json").exists():
            _val_ann = data_path / "annotations" / "instances_val2017.json"

        train_dataset = COCODetectionDataset(_train_images, _train_ann, transforms=train_transforms, input_size=input_size, ignore_empty_annotations=ignore_empty)
        val_dataset   = COCODetectionDataset(_val_images,   _val_ann,   transforms=val_transforms,   input_size=input_size)

        if num_classes == 0:
            num_classes = len(train_dataset.cat_id_to_label)

    # Wire Mixup now that we have a dataset (placed just before Normalize)
    train_transforms.transforms.insert(-1, Mixup(train_dataset, p=0.5))

    # Resolve class names from dataset (used for annotation in val image logging)
    class_names: list[str] | None = getattr(train_dataset, "class_names", None)
    if class_names:
        console.print(f"Class names: {class_names}")

    console.print(f"Classes: {num_classes} | Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    # Build model -------------------------------------------------------
    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    if pretrained and num_classes != 80:
        # Transfer learning: load pretrained backbone+neck with strict=True,
        # then swap heads for a freshly initialised num_classes version.
        from modern_yolonas.weights import transfer_to

        console.print(
            f"[yellow]Transfer learning: loading pretrained {model.value} backbone+neck, "
            f"re-initialising heads for {num_classes} classes.[/yellow]"
        )
        yolo_model = transfer_to(model.value, num_classes=num_classes)
    else:
        console.print(f"Building {model.value} (pretrained={pretrained}, num_classes={num_classes})...")
        yolo_model = builders[model.value](pretrained=pretrained, num_classes=num_classes)

    if compile_model:
        import torch
        yolo_model = torch.compile(yolo_model)
        console.print("[green]torch.compile enabled[/green]")

    # DataLoaders --------------------------------------------------------
    _persistent = workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        collate_fn=detection_collate_fn, pin_memory=True, drop_last=True,
        persistent_workers=_persistent, prefetch_factor=2 if _persistent else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=detection_collate_fn, pin_memory=True,
        persistent_workers=_persistent, prefetch_factor=2 if _persistent else None,
    )

    # Callbacks ----------------------------------------------------------
    callbacks = []
    if wandb:
        from modern_yolonas.training.callbacks import WandbCallback
        callbacks.append(WandbCallback(project=wandb_project, name=wandb_name))
        console.print(f"[green]W&B logging enabled → project={wandb_project!r}[/green]")
    if tensorboard:
        from modern_yolonas.training.callbacks import TensorBoardCallback
        callbacks.append(TensorBoardCallback(log_dir=tensorboard_dir, experiment_name=tensorboard_name))
        console.print(f"[green]TensorBoard logging enabled → {tensorboard_dir}/{tensorboard_name}[/green]")
    if early_stopping_patience > 0:
        from modern_yolonas.training.callbacks import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(patience=early_stopping_patience, min_delta=early_stopping_min_delta))
        console.print(f"[green]Early stopping enabled → patience={early_stopping_patience}, min_delta={early_stopping_min_delta}[/green]")

    trainer = Trainer(
        model=yolo_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        output_dir=output,
        device=device,
        callbacks=callbacks,
        class_names=class_names,
        val_freq=val_freq,
        gradient_accum=gradient_accum,
        use_amp=amp,
    )

    if resume_path:
        trainer.resume(resume_path)

    if num_gpus > 1:
        import os
        import torch.multiprocessing as mp

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        console.print(f"[green]DDP training on {num_gpus} GPUs[/green]")
        mp.spawn(
            _ddp_worker,
            args=(num_gpus, yolo_model, train_loader, val_loader, num_classes,
                  epochs, lr, output, device, callbacks, class_names, val_freq,
                  gradient_accum, amp, resume_path),
            nprocs=num_gpus,
            join=True,
        )
    else:
        trainer.train()


def _ddp_worker(
    local_rank: int,
    world_size: int,
    model,
    train_loader,
    val_loader,
    num_classes: int,
    epochs: int,
    lr: float,
    output_dir: str,
    device: str,
    callbacks: list,
    class_names,
    val_freq: int,
    gradient_accum: int,
    use_amp: bool,
    resume_path: str | None,
):
    """DDP worker spawned by ``torch.multiprocessing.spawn`` for multi-GPU training."""
    import os
    import torch.distributed as dist
    from modern_yolonas.training.trainer import Trainer

    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"]       = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=local_rank)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        output_dir=output_dir,
        device=f"cuda:{local_rank}",
        callbacks=callbacks,
        class_names=class_names,
        val_freq=val_freq,
        gradient_accum=gradient_accum,
        use_amp=use_amp,
        local_rank=local_rank,
    )
    if resume_path:
        trainer.resume(resume_path)
    trainer.train()

    dist.destroy_process_group()

