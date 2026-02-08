"""RF100-VL benchmark: train on all 100 datasets, aggregate mAP."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table


def discover_datasets(root: str | Path) -> list[dict]:
    """Scan RF100-VL directory and find all valid datasets.

    Each dataset must have:
    - ``train/_annotations.coco.json``
    - ``valid/_annotations.coco.json``

    Args:
        root: Path to RF100-VL root directory.

    Returns:
        List of dicts with keys: name, train_images, train_ann, val_images, val_ann, num_classes.
    """
    root = Path(root)
    datasets = []

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        train_ann = dataset_dir / "train" / "_annotations.coco.json"
        val_ann = dataset_dir / "valid" / "_annotations.coco.json"

        if not train_ann.exists() or not val_ann.exists():
            continue

        # Read num_classes from the annotation file
        with open(val_ann) as f:
            ann_data = json.load(f)
        num_classes = len(ann_data.get("categories", []))

        datasets.append({
            "name": dataset_dir.name,
            "train_images": str(dataset_dir / "train"),
            "train_ann": str(train_ann),
            "val_images": str(dataset_dir / "valid"),
            "val_ann": str(val_ann),
            "num_classes": num_classes,
        })

    return datasets


def run_rf100vl_benchmark(
    data_root: str | Path,
    model_name: str = "yolo_nas_s",
    recipe: dict | None = None,
    output_dir: str | Path = "runs/benchmark/rf100vl",
    devices: str | int | list[int] = "auto",
    logger: str = "csv",
    epochs: int | None = None,
    batch_size: int | None = None,
    dataset_filter: list[str] | None = None,
) -> dict:
    """Run the RF100-VL benchmark across all discovered datasets.

    Args:
        data_root: Path to RF100-VL root directory.
        model_name: Model variant.
        recipe: Training recipe dict. Defaults to RF100VL_RECIPE.
        output_dir: Output directory for results.
        devices: Lightning devices spec.
        logger: Logger backend.
        epochs: Override recipe epochs.
        batch_size: Override recipe batch_size.
        dataset_filter: Optional list of dataset names to include.

    Returns:
        Dict with per-dataset mAP and mean mAP.
    """
    import torch
    from rich.progress import Progress

    from modern_yolonas.data.coco import COCODetectionDataset
    from modern_yolonas.inference.postprocess import postprocess
    from modern_yolonas.training.lightning_module import extract_model_state_dict
    from modern_yolonas.training.recipes import RF100VL_RECIPE
    from modern_yolonas.training.run import MODEL_BUILDERS, build_transforms, run_training
    from modern_yolonas.training.metrics import COCOEvaluator

    if recipe is None:
        recipe = RF100VL_RECIPE

    console = Console()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(data_root)
    if dataset_filter:
        datasets = [d for d in datasets if d["name"] in dataset_filter]

    console.print(f"Found {len(datasets)} datasets in {data_root}")

    results = {}

    with Progress(console=console) as progress:
        task = progress.add_task("RF100-VL Benchmark", total=len(datasets))

        for ds_info in datasets:
            name = ds_info["name"]
            num_classes = ds_info["num_classes"]
            ds_output = output_dir / name

            progress.update(task, description=f"Training {name}")
            console.print(f"\n[bold]Dataset: {name}[/bold] ({num_classes} classes)")

            # Build datasets without transforms first
            train_dataset = COCODetectionDataset(
                ds_info["train_images"],
                ds_info["train_ann"],
                transforms=None,
                input_size=recipe["input_size"],
            )
            val_dataset = COCODetectionDataset(
                ds_info["val_images"],
                ds_info["val_ann"],
                transforms=None,
                input_size=recipe["input_size"],
            )

            # Assign transforms (train pipeline needs dataset for Mosaic/Mixup)
            train_dataset.transforms = build_transforms(recipe, train=True, dataset=train_dataset)
            val_dataset.transforms = build_transforms(recipe, train=False)

            # Train
            try:
                best_ckpt = run_training(
                    model_name=model_name,
                    recipe=recipe,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    val_ann_file=ds_info["val_ann"],
                    output_dir=ds_output,
                    num_classes=num_classes,
                    pretrained=True,
                    devices=devices,
                    logger=logger,
                    epochs=epochs,
                    batch_size=batch_size,
                )

                # Evaluate best checkpoint
                builder = MODEL_BUILDERS[model_name]
                eval_model = builder(pretrained=False, num_classes=num_classes)
                sd = extract_model_state_dict(best_ckpt)
                eval_model.load_state_dict(sd)
                eval_model = eval_model.cuda().eval()

                evaluator = COCOEvaluator(ds_info["val_ann"])
                from torch.utils.data import DataLoader
                from modern_yolonas.data.collate import detection_collate_fn

                eval_bs = batch_size or recipe["batch_size"]
                eval_loader = DataLoader(
                    val_dataset, batch_size=eval_bs, shuffle=False,
                    num_workers=4, collate_fn=detection_collate_fn, pin_memory=True,
                )

                with torch.no_grad():
                    for batch_idx, (images, targets) in enumerate(eval_loader):
                        images = images.cuda(non_blocking=True)
                        pred_bboxes, pred_scores = eval_model(images)
                        post_results = postprocess(pred_bboxes, pred_scores, 0.001, 0.65)

                        start_idx = batch_idx * eval_bs
                        end_idx = min(start_idx + eval_bs, len(val_dataset))
                        image_ids = [val_dataset.ids[i] for i in range(start_idx, end_idx)]

                        evaluator.update(
                            image_ids,
                            [r[0] for r in post_results],
                            [r[1] for r in post_results],
                            [r[2] for r in post_results],
                        )

                metrics = evaluator.evaluate()
                results[name] = metrics
                console.print(f"  mAP: {metrics['mAP']:.4f}, mAP_50: {metrics['mAP_50']:.4f}")

            except Exception as e:
                console.print(f"  [red]FAILED: {e}[/red]")
                results[name] = {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0, "error": str(e)}

            progress.advance(task)

    # Compute aggregate metrics
    valid = [r for r in results.values() if "error" not in r]
    mean_map = sum(r["mAP"] for r in valid) / len(valid) if valid else 0.0
    mean_map_50 = sum(r["mAP_50"] for r in valid) / len(valid) if valid else 0.0

    summary = {
        "model": model_name,
        "num_datasets": len(datasets),
        "num_successful": len(valid),
        "mean_mAP": mean_map,
        "mean_mAP_50": mean_map_50,
        "per_dataset": results,
    }

    # Save results
    results_path = output_dir / "rf100vl_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    table = Table(title="RF100-VL Benchmark Results")
    table.add_column("Dataset", style="cyan")
    table.add_column("mAP", justify="right")
    table.add_column("mAP_50", justify="right")
    for name, metrics in sorted(results.items()):
        if "error" in metrics:
            table.add_row(name, "[red]FAIL[/red]", "[red]FAIL[/red]")
        else:
            table.add_row(name, f"{metrics['mAP']:.4f}", f"{metrics['mAP_50']:.4f}")
    table.add_row("[bold]Mean[/bold]", f"[bold]{mean_map:.4f}[/bold]", f"[bold]{mean_map_50:.4f}[/bold]")
    console.print(table)

    console.print(f"\nResults saved to {results_path}")
    return summary
