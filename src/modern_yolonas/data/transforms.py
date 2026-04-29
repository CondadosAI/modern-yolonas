"""Detection-aware data augmentations.

Each transform operates on ``(image, targets)`` where:
- image: HWC uint8 BGR numpy array
- targets: ``[N, 5]`` numpy array with ``[class_id, x_center, y_center, w, h]`` (normalized)

HSVAugment, HorizontalFlip, RandomAffine, RandomResizedCrop, and
RandomChannelSwap are backed by `Albumentations <https://albumentations.ai>`_
(MIT license, v2+).
Mosaic, Mixup, LetterboxResize, and Normalize use native implementations
because they have no direct Albumentations equivalent.
"""

from __future__ import annotations

import random

import albumentations as A
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers: convert between our [N, 5] format and Albumentations bbox API
# ---------------------------------------------------------------------------

def _to_albu(targets: np.ndarray) -> tuple[list, list]:
    """Split ``[N, 5]`` targets into ``(bboxes, class_labels)`` for Albumentations."""
    if len(targets) == 0:
        return [], []
    return targets[:, 1:].tolist(), targets[:, 0].astype(int).tolist()


def _from_albu(bboxes: list, labels: list, dtype) -> np.ndarray:
    """Re-assemble Albumentations ``(bboxes, labels)`` into ``[N, 5]`` targets."""
    if not bboxes:
        return np.zeros((0, 5), dtype=dtype)
    return np.concatenate(
        [np.array(labels, dtype=dtype).reshape(-1, 1), np.array(bboxes, dtype=dtype)],
        axis=1,
    )


_BBOX_PARAMS = A.BboxParams(
    format="yolo",
    label_fields=["class_labels"],
    # Discard boxes smaller than 2×2 pixels after spatial transforms (wh_thr=2 from SG recipe).
    # filter_invalid_bboxes=True activates pixel-unit min_width / min_height filtering.
    filter_invalid_bboxes=True,
    min_width=2,
    min_height=2,
    clip=True,
)


class Compose:
    """Chain multiple transforms sequentially.

    Args:
        transforms: List of callables, each accepting ``(image, targets)``
            and returning ``(image, targets)``.
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            image, targets = t(image, targets)
        return image, targets


class HSVAugment:
    """Randomly adjust hue, saturation, and value via Albumentations.

    Args:
        hgain: Max hue shift in degrees (Albumentations ``hue_shift_limit``).
              Matches the super-gradients ``hgain`` recipe param. Default: 18.
        sgain: Max saturation shift in absolute units (``sat_shift_limit``).
              Default: 30.
        vgain: Max value shift in absolute units (``val_shift_limit``).
              Default: 30.
        p: Probability of applying the transform.
    """

    def __init__(self, hgain: int = 18, sgain: int = 30, vgain: int = 30, p: float = 0.5):
        self._aug = A.HueSaturationValue(
            hue_shift_limit=hgain,
            sat_shift_limit=sgain,
            val_shift_limit=vgain,
            p=p,
        )

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Albumentations expects RGB; our pipeline carries BGR images
        rgb = image[:, :, ::-1].copy()
        result = self._aug(image=rgb)
        return result["image"][:, :, ::-1].copy(), targets


class HorizontalFlip:
    """Randomly flip the image and bounding boxes horizontally via Albumentations.

    Args:
        p: Probability of applying the flip.
    """

    def __init__(self, p: float = 0.5):
        self.p = p
        self._aug = A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=_BBOX_PARAMS)

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.p:
            return image, targets
        bboxes, labels = _to_albu(targets)
        r = self._aug(image=image, bboxes=bboxes, class_labels=labels)
        return r["image"], _from_albu(r["bboxes"], r["class_labels"], targets.dtype)


class RandomAffine:
    """Apply random rotation, scale, translation, and shear via Albumentations.

    Args:
        degrees: Maximum rotation in degrees.
        translate: Maximum translation as a fraction of image size.
        scale: Scale range ``(min, max)``.
        shear: Maximum shear in degrees.
    """

    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.25,
        scale: tuple[float, float] = (0.5, 1.5),
        shear: float = 0.0,
    ):
        self._aug = A.Compose(
            [
                A.Affine(
                    scale=scale,
                    translate_percent={"x": (-translate, translate), "y": (-translate, translate)},
                    rotate=(-degrees, degrees),
                    shear=(-shear, shear),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=114,
                    p=1.0,
                )
            ],
            bbox_params=_BBOX_PARAMS,
        )

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bboxes, labels = _to_albu(targets)
        r = self._aug(image=image, bboxes=bboxes, class_labels=labels)
        return r["image"], _from_albu(r["bboxes"], r["class_labels"], targets.dtype)


class RandomResizedCrop:
    """Randomly crop a region of the image and resize it to ``size`` via Albumentations.

    Mirrors ``torchvision.transforms.RandomResizedCrop`` but is bounding-box
    aware.  Boxes whose area falls below ``min_width`` / ``min_height`` pixels
    after cropping are automatically discarded by ``_BBOX_PARAMS``.

    Args:
        size: Output square side length in pixels.
        scale: Range of fraction of the original image area to crop.
              Default ``(0.08, 1.0)`` matches the torchvision default.
        ratio: Range of aspect ratio of the crop.
              Default ``(0.75, 1.333)`` matches the torchvision default.
        interpolation: OpenCV interpolation flag (default ``cv2.INTER_LINEAR``).
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        size: int = 640,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.333),
        interpolation: int = cv2.INTER_LINEAR,
        p: float = 1.0,
    ):
        self._aug = A.Compose(
            [
                A.RandomResizedCrop(
                    size=(size, size),
                    scale=scale,
                    ratio=ratio,
                    interpolation=interpolation,
                    p=1.0,
                )
            ],
            bbox_params=_BBOX_PARAMS,
        )
        self.p = p

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.p:
            return image, targets
        bboxes, labels = _to_albu(targets)
        r = self._aug(image=image, bboxes=bboxes, class_labels=labels)
        return r["image"], _from_albu(r["bboxes"], r["class_labels"], targets.dtype)


class RandomChannelSwap:
    """Randomly swap BGR channel order to RGB (and vice-versa) via Albumentations.

    Adds photometric variety without touching bounding boxes.

    Args:
        p: Probability of swapping channels.
    """

    def __init__(self, p: float = 0.5):
        self._aug = A.ChannelShuffle(p=p)

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._aug(image=image)["image"], targets


class CenterCrop:
    """Crop the center of the image to ``size`` × ``size`` pixels.

    Bounding boxes that fall outside the cropped region are discarded;
    those that overlap are clipped to the new canvas by ``_BBOX_PARAMS``.

    Args:
        size: Output square side length in pixels.
    """

    def __init__(self, size: int = 640):
        self._aug = A.Compose(
            [A.CenterCrop(height=size, width=size, p=1.0)],
            bbox_params=_BBOX_PARAMS,
        )

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bboxes, labels = _to_albu(targets)
        r = self._aug(image=image, bboxes=bboxes, class_labels=labels)
        return r["image"], _from_albu(r["bboxes"], r["class_labels"], targets.dtype)


class Mosaic:
    """4-image mosaic augmentation."""

    def __init__(self, dataset, input_size: int = 640):
        self.dataset = dataset
        self.input_size = input_size

    def __call__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.input_size
        yc, xc = (int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2))

        indices = [index] + [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        all_targets = []

        for i, idx in enumerate(indices):
            img, targets = self.dataset.load_raw(idx)
            h, w = img.shape[:2]

            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            pad_w = x1a - x1b
            pad_h = y1a - y1b

            if len(targets):
                targets = targets.copy()
                # Convert to pixel coords, offset, then back to normalized
                targets[:, 1] = (targets[:, 1] * w + pad_w) / (s * 2)
                targets[:, 2] = (targets[:, 2] * h + pad_h) / (s * 2)
                targets[:, 3] = targets[:, 3] * w / (s * 2)
                targets[:, 4] = targets[:, 4] * h / (s * 2)
                all_targets.append(targets)

        targets = np.concatenate(all_targets, 0) if all_targets else np.zeros((0, 5))

        # Crop to input_size
        crop_x = int(random.uniform(0, s))
        crop_y = int(random.uniform(0, s))
        mosaic_img = mosaic_img[crop_y : crop_y + s, crop_x : crop_x + s]

        if len(targets):
            targets = targets.copy()
            targets[:, 1] = targets[:, 1] * 2 - crop_x / s
            targets[:, 2] = targets[:, 2] * 2 - crop_y / s

            # Filter out-of-bounds
            valid = (
                (targets[:, 1] > 0) & (targets[:, 1] < 1)
                & (targets[:, 2] > 0) & (targets[:, 2] < 1)
                & (targets[:, 3] > 0.002) & (targets[:, 4] > 0.002)
            )
            targets = targets[valid]

        return mosaic_img, targets


class Mixup:
    """Mixup augmentation for detection.

    Should be placed in the pipeline **after** ``LetterboxResize`` and
    **before** ``Normalize``, so both images are already square uint8.

    Args:
        dataset: Dataset exposing a ``load_raw(index)`` method.
        p: Per-sample probability of applying mixup.
        alpha: Beta distribution ``alpha`` parameter.
        beta: Beta distribution ``beta`` parameter.
    """

    def __init__(self, dataset, p: float = 0.5, alpha: float = 1.5, beta: float = 1.5):
        self.dataset = dataset
        self.p = p
        self.alpha = alpha
        self.beta = beta

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.p:
            return image, targets

        idx2 = random.randint(0, len(self.dataset) - 1)
        img2, targets2 = self.dataset.load_raw(idx2)

        # Letterbox the second image to match the (already resized) first image
        target_h, target_w = image.shape[:2]
        h2, w2 = img2.shape[:2]
        if (h2, w2) != (target_h, target_w):
            scale = min(target_h / h2, target_w / w2)
            new_h, new_w = int(round(h2 * scale)), int(round(w2 * scale))
            img2 = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            top  = (target_h - new_h) // 2
            left = (target_w - new_w) // 2
            padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
            padded[top:top + new_h, left:left + new_w] = img2
            img2 = padded
            if len(targets2):
                targets2 = targets2.copy()
                targets2[:, 1] = (targets2[:, 1] * new_w + left) / target_w
                targets2[:, 2] = (targets2[:, 2] * new_h + top)  / target_h
                targets2[:, 3] = targets2[:, 3] * new_w / target_w
                targets2[:, 4] = targets2[:, 4] * new_h / target_h

        r = np.random.beta(self.alpha, self.beta)
        mixed = (image.astype(np.float32) * r + img2.astype(np.float32) * (1 - r)).astype(np.uint8)

        if len(targets) and len(targets2):
            combined = np.concatenate([targets, targets2], 0)
        elif len(targets):
            combined = targets
        else:
            combined = targets2

        return mixed, combined


class LetterboxResize:
    """Resize with aspect ratio preservation and center padding.

    Args:
        target_size: Output square dimension.
        pad_value: Pixel value for padding (default 114, matching YOLO convention).
    """

    def __init__(self, target_size: int = 640, pad_value: int = 114):
        self.target_size = target_size
        self.pad_value = pad_value

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        top = pad_h // 2
        left = pad_w // 2

        padded = np.full((self.target_size, self.target_size, 3), self.pad_value, dtype=np.uint8)
        padded[top : top + new_h, left : left + new_w] = image

        if len(targets):
            targets = targets.copy()
            # Adjust for padding (targets are normalized)
            targets[:, 1] = (targets[:, 1] * new_w + left) / self.target_size
            targets[:, 2] = (targets[:, 2] * new_h + top) / self.target_size
            targets[:, 3] = targets[:, 3] * new_w / self.target_size
            targets[:, 4] = targets[:, 4] * new_h / self.target_size

        return padded, targets


class Normalize:
    """Convert HWC uint8 to CHW float32 [0,1] tensor."""

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = image[:, :, ::-1].copy()  # BGR → RGB
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        return image, targets
