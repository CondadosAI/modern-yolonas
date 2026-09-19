"""Named training recipes.

A recipe is the whole shape of a run: optimiser, schedule and augmentation. They
live here rather than inside a CLI command so that two commands cannot drift into
two different ideas of how to train on the same dataset -- which is exactly what
happened, and what LEGACY_RECIPE now records.
"""

COCO_RECIPE = {
    "epochs": 100,
    "optimizer": "sgd",
    # Measured 2026-09-19, not inherited. This was 4e-4, an AdamW-scale value that
    # survived the switch to SGD in a scaffolding commit and was never validated.
    # Eight epochs on 5k images, batch 16, averaging train/loss over the final epoch:
    #
    #   4e-4   4.739 -> 4.109     5e-3   4.312 -> 3.314
    #   2e-2   4.220 -> 3.100     5e-2   4.133 -> 2.999
    #
    # Monotonic, saturating between 2e-2 and 5e-2, which is the range published YOLO
    # recipes use with SGD. 2e-2 sits mid-plateau rather than at the edge, where a
    # 300-epoch run is likelier to destabilise.
    #
    # The probe measures convergence *speed* over 8 epochs, not final quality over
    # 300. What it establishes firmly is that 4e-4 converges at half the rate of
    # anything else -- enough to stop a week of GPU time going into it.
    "lr": 2e-2,
    "weight_decay": 5e-4,
    "cosine_final_lr_ratio": 0.1,
    "warmup_epochs": 3,
    "ema_decay": 0.9997,
    "precision": "16-mixed",
    "grad_clip": 10.0,
    "input_size": 640,
    "batch_size": 32,
    "workers": 8,
    "conf_threshold": 0.001,
    "iou_threshold": 0.65,
    "augmentations": {
        "mosaic": True,
        "mosaic_prob": 1.0,
        "mixup": True,
        "mixup_prob": 0.5,
        "close_mosaic_epochs": 15,
        "hsv": True,
        "hsv_prob": 0.5,
        "channel_shuffle": True,
        "channel_shuffle_prob": 0.5,
        "flip": True,
        "flip_prob": 0.5,
        "random_crop": False,
        "affine_degrees": 0.0,
        "affine_translate": 0.25,
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
    "grad_clip": 10.0,
    "input_size": 640,
    "batch_size": 16,
    "workers": 8,
    "conf_threshold": 0.001,
    "iou_threshold": 0.65,
    "augmentations": {
        "mosaic": False,
        "mixup": False,
        "close_mosaic_epochs": 0,
        "hsv": True,
        "hsv_prob": 0.5,
        "channel_shuffle": True,
        "channel_shuffle_prob": 0.5,
        "flip": True,
        "flip_prob": 0.5,
        "random_crop": False,
        "affine_degrees": 0.0,
        "affine_translate": 0.1,
        "affine_scale": (0.5, 1.5),
    },
}


# What `yolonas train` did before recipes existed, written down rather than
# hardcoded in the command. Kept because changing anyone's defaults silently
# would be worse than carrying a name.
#
# It differs from COCO_RECIPE in every way that matters for COCO: AdamW instead
# of SGD, 300 epochs instead of 100, **no mosaic at all**, and a random resized
# crop where COCO_RECIPE letterboxes. That last one is the geometry mismatch the
# roadmap has listed as a blocker since 2026-09-13: training saw crops, while
# validation and inference letterbox.
#
# One deliberate difference from the original: Mixup's second image now goes
# through the same augmentations as the first. It used to be pristine and
# letterboxed, so half of all samples blended an aggressively cropped image with
# an untouched one. Preserving that faithfully would mean keeping a defect for the
# sake of a name.
LEGACY_RECIPE = {
    "epochs": 300,
    "optimizer": "adamw",
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "cosine_final_lr_ratio": 0.1,
    "warmup_epochs": 3,
    "ema_decay": 0.9997,
    "precision": "16-mixed",
    "grad_clip": 10.0,
    "input_size": 640,
    "batch_size": 32,
    "workers": 8,
    "conf_threshold": 0.001,
    "iou_threshold": 0.65,
    "augmentations": {
        "mosaic": False,
        "mixup": True,
        "mixup_prob": 0.5,
        "close_mosaic_epochs": 0,
        "hsv": True,
        "hsv_prob": 0.5,
        "channel_shuffle": True,
        "channel_shuffle_prob": 0.5,
        # One fused warp for crop + flip + affine, and no letterbox: the fused
        # transform already outputs input_size.
        "fused_geometry": True,
        "fused_scale": (0.05, 0.8),
        "fused_ratio": (0.75, 1.33),
        "flip": True,
        "flip_prob": 0.5,
        "affine_translate": 0.25,
        "affine_scale": (0.5, 1.5),
    },
}

RECIPES = {
    "legacy": LEGACY_RECIPE,
    "coco": COCO_RECIPE,
    "rf100vl": RF100VL_RECIPE,
}
