"""CLI: yolonas benchmark (coco / rf100vl)"""

from __future__ import annotations

from pathlib import Path

import click


def parse_devices(devices: str) -> str | int | list[int]:
    """Parse devices string to Lightning-compatible format."""
    if devices == "auto":
        return devices
    if "," in devices:
        return [int(d) for d in devices.split(",")]
    return int(devices)


@click.group()
def benchmark():
    """Run standard benchmarks."""


@benchmark.command()
@click.option("--model", default="yolo_nas_s", type=click.Choice(["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]))
@click.option("--data", default=None, help="Path to COCO dataset root.")
@click.option("--download/--no-download", default=False, help="Download COCO via FiftyOne.")
@click.option("--output", default="runs/benchmark/coco", help="Output directory.")
@click.option("--devices", default="auto", help="Devices (e.g. 'auto', '1', '0,1').")
@click.option("--logger", default="csv", type=click.Choice(["csv", "tensorboard", "wandb"]))
@click.option("--epochs", default=None, type=int, help="Override recipe epochs (default: 100).")
@click.option("--batch-size", default=None, type=int, help="Override recipe batch size (default: 32).")
@click.option("--resume", "resume_path", default=None, help="Resume from checkpoint.")
@click.option("--input-size", default=640, help="Model input size.")
@click.option("--workers", default=8, help="DataLoader workers.")
def coco(
    model: str,
    data: str | None,
    download: bool,
    output: str,
    devices: str,
    logger: str,
    epochs: int | None,
    batch_size: int | None,
    resume_path: str | None,
    input_size: int,
    workers: int,
):
    """Train YOLO-NAS from scratch on COCO and evaluate mAP."""
    from rich.console import Console

    from modern_yolonas.data.coco import COCODetectionDataset
    from modern_yolonas.training.recipes import COCO_RECIPE
    from modern_yolonas.training.run import build_transforms, run_training

    console = Console()
    recipe = {**COCO_RECIPE, "input_size": input_size, "workers": workers}

    # Download if requested
    if download:
        from modern_yolonas.data.download import download_coco

        data = str(download_coco(data or "datasets/coco"))
        console.print(f"COCO downloaded to {data}")

    if data is None:
        raise click.UsageError("--data is required (or use --download)")

    data_path = Path(data)

    # Build datasets
    train_transforms = build_transforms(recipe, train=True)
    val_transforms = build_transforms(recipe, train=False)

    train_dataset = COCODetectionDataset(
        data_path / "images" / "train2017",
        data_path / "annotations" / "instances_train2017.json",
        transforms=train_transforms,
        input_size=input_size,
    )
    val_dataset = COCODetectionDataset(
        data_path / "images" / "val2017",
        data_path / "annotations" / "instances_val2017.json",
        transforms=val_transforms,
        input_size=input_size,
    )
    val_ann_file = data_path / "annotations" / "instances_val2017.json"

    console.print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # Train from scratch
    best_ckpt = run_training(
        model_name=model,
        recipe=recipe,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        val_ann_file=str(val_ann_file),
        output_dir=output,
        num_classes=80,
        pretrained=False,
        devices=parse_devices(devices),
        logger=logger,
        resume_path=resume_path,
        epochs=epochs,
        batch_size=batch_size,
    )

    console.print(f"\n[bold green]Training complete![/bold green] Best checkpoint: {best_ckpt}")

    # Final evaluation with best checkpoint
    console.print("\nRunning final evaluation...")
    import torch

    from modern_yolonas.data.collate import detection_collate_fn
    from modern_yolonas.inference.postprocess import postprocess
    from modern_yolonas.training.lightning_module import extract_model_state_dict
    from modern_yolonas.training.metrics import COCOEvaluator
    from modern_yolonas.training.run import MODEL_BUILDERS

    builder = MODEL_BUILDERS[model]
    eval_model = builder(pretrained=False, num_classes=80)
    sd = extract_model_state_dict(best_ckpt)
    eval_model.load_state_dict(sd)
    eval_model = eval_model.cuda().eval()

    from torch.utils.data import DataLoader

    eval_bs = batch_size or recipe["batch_size"]
    eval_loader = DataLoader(
        val_dataset, batch_size=eval_bs, shuffle=False,
        num_workers=workers, collate_fn=detection_collate_fn, pin_memory=True,
    )

    evaluator = COCOEvaluator(val_ann_file)
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(eval_loader):
            images = images.cuda(non_blocking=True)
            pred_bboxes, pred_scores = eval_model(images)
            results = postprocess(pred_bboxes, pred_scores, 0.001, 0.65)

            start_idx = batch_idx * eval_bs
            end_idx = min(start_idx + eval_bs, len(val_dataset))
            image_ids = [val_dataset.ids[i] for i in range(start_idx, end_idx)]

            evaluator.update(
                image_ids,
                [r[0] for r in results],
                [r[1] for r in results],
                [r[2] for r in results],
            )

            if (batch_idx + 1) % 20 == 0:
                console.print(f"  [{batch_idx + 1}/{len(eval_loader)}]")

    metrics = evaluator.evaluate()
    console.print("\n[bold]Final Results:[/bold]")
    for k, v in metrics.items():
        console.print(f"  {k}: {v:.4f}")


@benchmark.command()
@click.option("--model", default="yolo_nas_s", type=click.Choice(["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]))
@click.option("--data", default=None, help="Path to RF100-VL root directory.")
@click.option("--download/--no-download", default=False, help="Download RF100-VL.")
@click.option("--output", default="runs/benchmark/rf100vl", help="Output directory.")
@click.option("--devices", default="auto", help="Devices (e.g. 'auto', '1', '0,1').")
@click.option("--logger", default="csv", type=click.Choice(["csv", "tensorboard", "wandb"]))
@click.option("--epochs", default=None, type=int, help="Override recipe epochs (default: 75).")
@click.option("--batch-size", default=None, type=int, help="Override recipe batch size (default: 16).")
@click.option("--datasets", "dataset_filter", default=None, help="Comma-separated dataset names to include.")
def rf100vl(
    model: str,
    data: str | None,
    download: bool,
    output: str,
    devices: str,
    logger: str,
    epochs: int | None,
    batch_size: int | None,
    dataset_filter: str | None,
):
    """Train on RF100-VL datasets and report aggregate mAP."""
    from rich.console import Console

    from modern_yolonas.benchmarks.rf100vl import run_rf100vl_benchmark

    console = Console()

    # Download if requested
    if download:
        from modern_yolonas.data.download import download_rf100vl

        data = str(download_rf100vl(data or "datasets/rf100vl"))
        console.print(f"RF100-VL downloaded to {data}")

    if data is None:
        raise click.UsageError("--data is required (or use --download)")

    ds_filter = dataset_filter.split(",") if dataset_filter else None

    run_rf100vl_benchmark(
        data_root=data,
        model_name=model,
        output_dir=output,
        devices=parse_devices(devices),
        logger=logger,
        epochs=epochs,
        batch_size=batch_size,
        dataset_filter=ds_filter,
    )
