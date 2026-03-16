"""CLI: yolonas demo"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

import typer


class ModelName(str, Enum):
    yolo_nas_s = "yolo_nas_s"
    yolo_nas_m = "yolo_nas_m"
    yolo_nas_l = "yolo_nas_l"


def demo(
    model: Annotated[ModelName, typer.Option(help="Default model variant.")] = ModelName.yolo_nas_s,
    device: Annotated[str, typer.Option(help="Device.")] = "cuda",
    port: Annotated[int, typer.Option(help="Gradio server port.")] = 7860,
    share: Annotated[bool, typer.Option("--share", help="Create a public link.")] = False,
):
    """Launch a Gradio web demo for object detection."""
    try:
        import gradio as gr
    except ImportError:
        typer.echo("Install demo dependencies: pip install modern-yolonas[demo]")
        raise typer.Exit(1)

    import cv2
    import numpy as np

    from modern_yolonas.inference.detect import Detector
    from modern_yolonas.inference.visualize import COCO_NAMES

    detectors: dict[str, Detector] = {}

    def get_detector(model_name: str) -> Detector:
        if model_name not in detectors:
            detectors[model_name] = Detector(model_name, device=device)
        return detectors[model_name]

    # Pre-load default model
    get_detector(model.value)

    def detect_fn(
        image: np.ndarray | None,
        model_name: str,
        conf_threshold: float,
        iou_threshold: float,
    ) -> tuple[np.ndarray | None, str]:
        if image is None:
            return None, "No image provided"

        # Gradio provides RGB, convert to BGR
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        det = get_detector(model_name)
        result = det(bgr, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

        lines = []
        for box, score, cls_id in zip(result.boxes, result.scores, result.class_ids):
            name = COCO_NAMES[int(cls_id)] if int(cls_id) < len(COCO_NAMES) else f"class_{int(cls_id)}"
            x1, y1, x2, y2 = box
            lines.append(f"{name}: {score:.2f} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

        annotated = result.visualize()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        summary = f"Found {len(result.boxes)} objects\n" + "\n".join(lines)
        return annotated_rgb, summary

    interface = gr.Interface(
        fn=detect_fn,
        inputs=[
            gr.Image(type="numpy", label="Input Image"),
            gr.Dropdown(
                choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"],
                value=model.value,
                label="Model",
            ),
            gr.Slider(0, 1, value=0.25, step=0.05, label="Confidence Threshold"),
            gr.Slider(0, 1, value=0.7, step=0.05, label="IoU Threshold"),
        ],
        outputs=[
            gr.Image(type="numpy", label="Detections"),
            gr.Textbox(label="Results", lines=10),
        ],
        title="YOLO-NAS Object Detection",
        description="Upload an image to detect objects using YOLO-NAS.",
    )

    interface.launch(server_port=port, share=share)
