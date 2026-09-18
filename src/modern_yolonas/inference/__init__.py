from modern_yolonas.inference.detect import YoloNASDetector

__all__ = ["YoloNASDetector"]


def __getattr__(name: str) -> type[YoloNASDetector]:
    if name == "Detector":
        from modern_yolonas.inference.detect import _warn_detector_alias

        return _warn_detector_alias(__name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
