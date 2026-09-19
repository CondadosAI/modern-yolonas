"""Gradio demo for modern-yolonas, sized for a CPU-only Hugging Face Space.

Runs against the `modern-yolonas` release pinned in requirements.txt, not the
working tree, so this file uses only API that release actually ships. Inference
is timed here rather than read off the detector, which keeps this working across
releases that change what the detector records.
"""

from __future__ import annotations

import time

from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from modern_yolonas import YoloNASDetector

EXAMPLES_DIR = Path(__file__).parent / "examples"

# Wall-clock for one 640x640 forward pass, measured on two cores to approximate the
# free tier. Shown in the UI because nothing else on the page tells you that the
# accuracy you gain from L costs you roughly triple the wait.
MODELS = {
    "yolo_nas_s": "YOLO-NAS-S — fastest, ~0.5s/image on CPU",
    "yolo_nas_m": "YOLO-NAS-M — balanced, ~1s/image on CPU",
    "yolo_nas_l": "YOLO-NAS-L — most accurate, ~1.5s/image on CPU",
}
DEFAULT_MODEL = "yolo_nas_s"

_detectors: dict[str, YoloNASDetector] = {}


def get_detector(model_name: str) -> YoloNASDetector:
    """Return a cached CPU detector, downloading weights on first use.

    Only the default variant is loaded at boot; M and L cost 205 MB and 268 MB of
    download plus their load time, which would stall startup for visitors who
    never switch models.
    """
    if model_name not in _detectors:
        _detectors[model_name] = YoloNASDetector(model_name, device="cpu")
    return _detectors[model_name]


def detect(
    image: np.ndarray | None,
    model_name: str,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[np.ndarray | None, str, list[list[str | float]]]:
    """Annotate `image` and return it alongside a status line and a detection table."""
    if image is None:
        return None, "Upload an image to run detection.", []

    # Gradio hands over RGB; every modern-yolonas entry point speaks OpenCV BGR.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detector = get_detector(model_name)

    start = time.perf_counter()
    detections = detector(bgr, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    annotated = cv2.cvtColor(detector.annotate(bgr, detections), cv2.COLOR_BGR2RGB)

    names = detections.data.get("class_name")
    if names is None:
        names = detections.class_id.astype(str)

    rows: list[list[str | float]] = []
    for name, score, box in zip(names, detections.confidence, detections.xyxy):
        x1, y1, x2, y2 = (round(float(v)) for v in box)
        rows.append([str(name), round(float(score), 3), f"{x1}, {y1}, {x2}, {y2}"])
    rows.sort(key=lambda row: row[1], reverse=True)

    status = f"**{len(detections)} objects** · {model_name} · {elapsed_ms:.0f} ms on CPU"
    return annotated, status, rows


DESCRIPTION = """
# modern-yolonas — YOLO-NAS object detection

A clean, minimal reimplementation of YOLO-NAS: a model variant is a function, a config is a
dataclass, and `state_dict` keys match [super-gradients](https://github.com/Deci-AI/super-gradients)
exactly. Detections come back as [supervision](https://github.com/roboflow/supervision) `Detections`.

[GitHub](https://github.com/CondadosAI/modern-yolonas) ·
[Docs](https://condadosai.github.io/modern-yolonas/) ·
[PyPI](https://pypi.org/project/modern-yolonas/)

This Space runs on free CPU hardware, so expect about half a second per image for S and around
a second and a half for L. On a GPU the same code runs in milliseconds.
"""

NOTICE = """
**Weights licence** — the pretrained COCO checkpoints served here were converted from Deci AI's
super-gradients releases and remain under the
[Super Gradients Model EULA](https://docs.deci.ai/super-gradients/latest/LICENSE.YOLONAS.html)
(**non-commercial use only**). The modern-yolonas source is Apache-2.0; the weights are not.
For commercial deployment, train from scratch.

Sample photo by [Wilfredor](https://commons.wikimedia.org/wiki/User:Wilfredor),
[CC0](https://creativecommons.org/publicdomain/zero/1.0/).
"""

with gr.Blocks(title="modern-yolonas") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Input image")
            model_name = gr.Dropdown(
                choices=[(label, name) for name, label in MODELS.items()],
                value=DEFAULT_MODEL,
                label="Model",
            )
            conf_threshold = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="Confidence threshold")
            iou_threshold = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="NMS IoU threshold")
            run_button = gr.Button("Detect", variant="primary")

        with gr.Column():
            output_image = gr.Image(type="numpy", label="Detections")
            status = gr.Markdown()
            table = gr.Dataframe(
                headers=["class", "confidence", "box (x1, y1, x2, y2)"],
                label="Detections",
                wrap=True,
            )

    gr.Examples(
        examples=[[str(EXAMPLES_DIR / "street.jpg"), DEFAULT_MODEL, 0.4, 0.7]],
        inputs=[input_image, model_name, conf_threshold, iou_threshold],
        outputs=[output_image, status, table],
        fn=detect,
        cache_examples=False,
    )

    gr.Markdown(NOTICE)

    run_button.click(
        fn=detect,
        inputs=[input_image, model_name, conf_threshold, iou_threshold],
        outputs=[output_image, status, table],
    )

if __name__ == "__main__":
    # Warm the default model so the first visitor does not pay for the download.
    get_detector(DEFAULT_MODEL)
    # Two vCPUs: a second concurrent forward pass would only slow both down.
    demo.queue(default_concurrency_limit=1).launch()
