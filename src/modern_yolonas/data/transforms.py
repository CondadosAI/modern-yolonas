"""Detection-aware data augmentations.

Each transform operates on ``(image, targets)`` where:
- image: HWC uint8 BGR numpy array
- targets: ``[N, 5]`` numpy array with ``[class_id, x_center, y_center, w, h]`` (normalized)
"""

from __future__ import annotations

import random

import cv2
import numpy as np


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
    """Randomly adjust hue, saturation, and value channels.

    Args:
        h_gain: Maximum hue shift factor.
        s_gain: Maximum saturation shift factor.
        v_gain: Maximum value shift factor.
    """

    def __init__(self, h_gain: float = 0.015, s_gain: float = 0.7, v_gain: float = 0.4):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
        hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return image, targets


class HorizontalFlip:
    """Randomly flip the image and targets horizontally.

    Args:
        p: Probability of applying the flip.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            image = np.fliplr(image).copy()
            if len(targets):
                targets = targets.copy()
                targets[:, 1] = 1.0 - targets[:, 1]  # flip x_center
        return image, targets


class RandomAffine:
    """Apply random rotation, scaling, translation, and shear.

    Transforms both the image and bounding box targets, filtering out
    boxes that become too small after the transform.

    Args:
        degrees: Maximum rotation in degrees.
        translate: Maximum translation as a fraction of image size.
        scale: Scale range ``(min, max)``.
        shear: Maximum shear in degrees.
    """

    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: tuple[float, float] = (0.5, 1.5),
        shear: float = 0.0,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]

        # Rotation + scale
        angle = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(*self.scale)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, s)

        # Translation
        M[0, 2] += random.uniform(-self.translate, self.translate) * w
        M[1, 2] += random.uniform(-self.translate, self.translate) * h

        image = cv2.warpAffine(image, M, (w, h), borderValue=(114, 114, 114))

        if len(targets):
            targets = targets.copy()
            # Convert normalized xywh to pixel xyxy
            boxes = np.zeros((len(targets), 4))
            boxes[:, 0] = (targets[:, 1] - targets[:, 3] / 2) * w
            boxes[:, 1] = (targets[:, 2] - targets[:, 4] / 2) * h
            boxes[:, 2] = (targets[:, 1] + targets[:, 3] / 2) * w
            boxes[:, 3] = (targets[:, 2] + targets[:, 4] / 2) * h

            # Transform corners
            corners = np.array([
                [boxes[:, 0], boxes[:, 1]],
                [boxes[:, 2], boxes[:, 1]],
                [boxes[:, 2], boxes[:, 3]],
                [boxes[:, 0], boxes[:, 3]],
            ]).transpose(2, 0, 1).reshape(-1, 2)

            ones = np.ones((corners.shape[0], 1))
            corners = np.hstack([corners, ones])
            transformed = (M @ corners.T).T.reshape(-1, 4, 2)

            new_boxes = np.zeros((len(targets), 4))
            new_boxes[:, 0] = transformed[:, :, 0].min(axis=1)
            new_boxes[:, 1] = transformed[:, :, 1].min(axis=1)
            new_boxes[:, 2] = transformed[:, :, 0].max(axis=1)
            new_boxes[:, 3] = transformed[:, :, 1].max(axis=1)

            # Clip and convert back to normalized xywh
            new_boxes[:, 0] = np.clip(new_boxes[:, 0], 0, w)
            new_boxes[:, 1] = np.clip(new_boxes[:, 1], 0, h)
            new_boxes[:, 2] = np.clip(new_boxes[:, 2], 0, w)
            new_boxes[:, 3] = np.clip(new_boxes[:, 3], 0, h)

            box_w = new_boxes[:, 2] - new_boxes[:, 0]
            box_h = new_boxes[:, 3] - new_boxes[:, 1]
            valid = (box_w > 2) & (box_h > 2)

            targets = targets[valid]
            new_boxes = new_boxes[valid]
            if len(targets):
                targets[:, 1] = ((new_boxes[:, 0] + new_boxes[:, 2]) / 2) / w
                targets[:, 2] = ((new_boxes[:, 1] + new_boxes[:, 3]) / 2) / h
                targets[:, 3] = (new_boxes[:, 2] - new_boxes[:, 0]) / w
                targets[:, 4] = (new_boxes[:, 3] - new_boxes[:, 1]) / h

        return image, targets


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
    """Mixup augmentation for detection."""

    def __init__(self, dataset, alpha: float = 1.5, beta: float = 1.5):
        self.dataset = dataset
        self.alpha = alpha
        self.beta = beta

    def __call__(self, image: np.ndarray, targets: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
        idx2 = random.randint(0, len(self.dataset) - 1)
        img2, targets2 = self.dataset[idx2]

        r = np.random.beta(self.alpha, self.beta)
        image = (image * r + img2 * (1 - r)).astype(np.uint8)
        targets = np.concatenate([targets, targets2], 0) if len(targets2) else targets
        return image, targets


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
