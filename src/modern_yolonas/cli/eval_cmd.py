"""CLI: yolonas eval"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer


class ModelName(str, Enum):
    yolo_nas_s = "yolo_nas_s"
    yolo_nas_m = "yolo_nas_m"
    yolo_nas_l = "yolo_nas_l"


def eval_cmd(
    data: Annotated[str, typer.Option(help="Path to COCO dataset root.")],
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    split: Annotated[str, typer.Option(help="Split name.")] = "val2017",
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 32,
    device: Annotated[str, typer.Option(help="Device.")] = "cuda",
    input_size: Annotated[int, typer.Option(help="Model input size.")] = 640,
    conf: Annotated[float, typer.Option(help="Confidence threshold for eval.")] = 0.001,
    iou: Annotated[float, typer.Option(help="NMS IoU threshold for eval.")] = 0.65,
    checkpoint: Annotated[str | None, typer.Option(help="Custom checkpoint path.")] = None,
):
    """Evaluate model on COCO dataset."""
    import torch
    from rich.console import Console
    from torch.utils.data import DataLoader

    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l
    from modern_yolonas.data.coco import COCODetectionDataset
    from modern_yolonas.data.transforms import Compose, LetterboxResize, Normalize
    from modern_yolonas.data.collate import detection_collate_fn
    from modern_yolonas.inference.postprocess import postprocess
    from modern_yolonas.training.metrics import COCOEvaluator

    console = Console()
    data_path = Path(data)

    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}

    if checkpoint:
        yolo_model = builders[model.value](pretrained=False)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        yolo_model.load_state_dict(sd)
    else:
        yolo_model = builders[model.value](pretrained=True)

    yolo_model = yolo_model.to(device).eval()

    ann_file = data_path / "annotations" / f"instances_{split}.json"
    img_dir = data_path / "images" / split

    dataset = COCODetectionDataset(
        img_dir, ann_file,
        transforms=Compose([LetterboxResize(target_size=input_size), Normalize()]),
        input_size=input_size,
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8,
        collate_fn=detection_collate_fn, pin_memory=True,
    )

    evaluator = COCOEvaluator(ann_file)

    console.print(f"Evaluating {model.value} on {split} ({len(dataset)} images)...")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            pred_bboxes, pred_scores = yolo_model(images)

            results = postprocess(pred_bboxes, pred_scores, conf, iou)

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            image_ids = [dataset.ids[i] for i in range(start_idx, end_idx)]

            boxes_list = [r[0] for r in results]
            scores_list = [r[1] for r in results]
            class_ids_list = [r[2] for r in results]

            evaluator.update(image_ids, boxes_list, scores_list, class_ids_list)

            if (batch_idx + 1) % 20 == 0:
                console.print(f"  [{batch_idx + 1}/{len(loader)}]")

    metrics = evaluator.evaluate()
    console.print("\n[bold]Results:[/bold]")
    for k, v in metrics.items():
        console.print(f"  {k}: {v:.4f}")
