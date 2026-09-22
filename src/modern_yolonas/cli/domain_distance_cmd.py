"""CLI: yolonas domain-distance"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer


class ModelName(str, Enum):
    yolo_nas_s = "yolo_nas_s"
    yolo_nas_m = "yolo_nas_m"
    yolo_nas_l = "yolo_nas_l"


def domain_distance(
    images: Annotated[Path, typer.Option(help="Your dataset's images, searched recursively.")],
    reference: Annotated[Path, typer.Option(help="The reference images, normally COCO train2017.")],
    baseline: Annotated[Path | None, typer.Option(help="A second sample of the reference domain (e.g. COCO val2017). Measures the floor, so your number has a scale.")] = None,
    model: Annotated[ModelName, typer.Option(help="Backbone whose feature space the distance is measured in.")] = ModelName.yolo_nas_s,
    samples: Annotated[int, typer.Option(help="Images sampled per dataset.")] = 2000,
    size: Annotated[int, typer.Option(help="Square resize before encoding.")] = 640,
    batch_size: Annotated[int, typer.Option(help="Encoding batch size.")] = 32,
    workers: Annotated[int, typer.Option(help="DataLoader workers.")] = 8,
    device: Annotated[str, typer.Option(help="Device for the encoder.")] = "cuda",
    seed: Annotated[int, typer.Option(help="Seed for sampling and cross-validation.")] = 0,
):
    """Measure how far your dataset is from COCO.

    Self-supervised pretraining and distillation both cost GPU-days, and both pay
    off in proportion to how badly a COCO-trained representation already fits your
    images. This measures that in minutes instead::

        yolonas domain-distance --images ~/mydata/images \\
            --reference ~/datasets/coco/train2017 \\
            --baseline ~/datasets/coco/val2017

    Two statistics are reported. The proxy A-distance is the headline: it trains a
    linear probe to tell your images from the reference and reports
    ``2(1 - 2*error)``, so 0 means indistinguishable and 2 means trivially
    separable. KID is the unbiased kernel distance, for comparing runs.

    Pass ``--baseline`` a second sample of the reference domain. It measures the
    floor -- what this statistic reads when two samples really are the same domain
    -- so your own number has something to sit against. Without it the numbers have
    no scale, and the command says so rather than guessing a threshold.
    """
    import torch
    from rich.console import Console

    from modern_yolonas import yolo_nas_l, yolo_nas_m, yolo_nas_s
    from modern_yolonas.analysis.domain_distance import (
        BackboneEncoder,
        compare,
        embed,
        find_images,
    )

    console = Console()

    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    net = builders[model.value](pretrained=True, num_classes=80)
    encoder = BackboneEncoder(net)

    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]No CUDA device; falling back to CPU.[/yellow]")
        device = "cpu"

    def features(path: Path, label: str):
        paths = find_images(path, limit=samples, seed=seed)
        console.print(f"encoding {len(paths)} images from {label} ...")
        return embed(
            paths, encoder, device=device, size=size, batch_size=batch_size, workers=workers
        )

    query = features(images, str(images))
    reference_features = features(reference, str(reference))

    encoder_name = f"{model.value} backbone (SPP, global average pool)"
    result = compare(query, reference_features, encoder_name, seed=seed)

    console.print()
    console.rule("your dataset vs the reference")
    console.print(result.summary())

    if baseline is not None:
        floor_features = features(baseline, str(baseline))
        floor = compare(floor_features, reference_features, encoder_name, seed=seed)
        console.rule("the floor: reference vs itself")
        console.print(floor.summary())
        console.print()
        console.print(
            f"[bold]Your dataset reads {result.d_a:.3f} where two samples of the same "
            f"domain read {floor.d_a:.3f}.[/bold]"
        )
        console.print(
            "What that is worth in AP is not calibrated -- no experiment here maps a "
            "distance to a gain. A number near the floor does mean a COCO backbone "
            "already represents your images about as well as it represents COCO, and "
            "pretraining on them has correspondingly less to add."
        )
    else:
        console.print()
        console.print(
            "[yellow]No --baseline given, so this number has no scale.[/yellow] Pass a "
            "second sample of the reference domain (COCO val2017 against COCO "
            "train2017) to see what 'same domain' reads on your own images and encoder."
        )
