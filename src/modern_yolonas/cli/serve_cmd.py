"""CLI: yolonas serve"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

import typer


class ModelName(str, Enum):
    yolo_nas_s = "yolo_nas_s"
    yolo_nas_m = "yolo_nas_m"
    yolo_nas_l = "yolo_nas_l"


def serve(
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    device: Annotated[str, typer.Option(help="Device.")] = "cuda",
    port: Annotated[int, typer.Option(help="Server port.")] = 8000,
    host: Annotated[str, typer.Option(help="Server host.")] = "0.0.0.0",
    conf: Annotated[float, typer.Option(help="Default confidence threshold.")] = 0.25,
    iou: Annotated[float, typer.Option(help="Default IoU threshold.")] = 0.7,
):
    """Start a REST API server for object detection."""
    try:
        import uvicorn
    except ImportError:
        typer.echo("Install serve dependencies: pip install modern-yolonas[serve]")
        raise typer.Exit(1)

    from modern_yolonas.serve.app import create_app

    app = create_app(model=model.value, device=device, conf_threshold=conf, iou_threshold=iou)
    uvicorn.run(app, host=host, port=port)
