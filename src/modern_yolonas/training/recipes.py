"""Standard training recipes for YOLO-NAS benchmarks."""

COCO_RECIPE = {
    "epochs": 100,
    "optimizer": "sgd",
    "lr": 4e-4,
    "weight_decay": 5e-4,
    "cosine_final_lr_ratio": 0.1,
    "warmup_epochs": 3,
    "ema_decay": 0.9997,
    "precision": "16-mixed",
    "input_size": 640,
    "batch_size": 32,
    "workers": 8,
    "conf_threshold": 0.001,
    "iou_threshold": 0.65,
    "augmentations": {
        "hsv": True,
        "flip": True,
        "affine_degrees": 0.0,
        "affine_translate": 0.1,
        "affine_scale": (0.5, 1.5),
    },
}

RF100VL_RECIPE = {
    "epochs": 75,
    "optimizer": "adamw",
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "cosine_final_lr_ratio": 0.1,
    "warmup_epochs": 3,
    "ema_decay": 0.9997,
    "precision": "16-mixed",
    "input_size": 640,
    "batch_size": 16,
    "workers": 8,
    "conf_threshold": 0.001,
    "iou_threshold": 0.65,
    "augmentations": {
        "hsv": True,
        "flip": True,
        "affine_degrees": 0.0,
        "affine_translate": 0.1,
        "affine_scale": (0.5, 1.5),
    },
}
