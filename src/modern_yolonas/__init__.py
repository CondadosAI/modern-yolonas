"""modern-yolonas — Modern YOLO-NAS object detection."""

from modern_yolonas._version import __version__
from modern_yolonas.model import YoloNAS
from modern_yolonas.weights import load_pretrained, load_checkpoint, transfer_to
from modern_yolonas.inference.detect import Detector, Detection


def yolo_nas_s(pretrained: bool = False, num_classes: int = 80) -> YoloNAS:
    """Create a YOLO-NAS S model."""
    model = YoloNAS.from_config("yolo_nas_s", num_classes=num_classes)
    if pretrained:
        load_pretrained(model, "yolo_nas_s")
    return model


def yolo_nas_m(pretrained: bool = False, num_classes: int = 80) -> YoloNAS:
    """Create a YOLO-NAS M model."""
    model = YoloNAS.from_config("yolo_nas_m", num_classes=num_classes)
    if pretrained:
        load_pretrained(model, "yolo_nas_m")
    return model


def yolo_nas_l(pretrained: bool = False, num_classes: int = 80) -> YoloNAS:
    """Create a YOLO-NAS L model."""
    model = YoloNAS.from_config("yolo_nas_l", num_classes=num_classes)
    if pretrained:
        load_pretrained(model, "yolo_nas_l")
    return model


__all__ = [
    "__version__",
    "YoloNAS",
    "yolo_nas_s",
    "yolo_nas_m",
    "yolo_nas_l",
    "load_pretrained",
    "load_checkpoint",
    "transfer_to",
    "Detector",
    "Detection",
]
