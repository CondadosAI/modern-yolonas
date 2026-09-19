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

from modern_yolonas.data.base import BaseDetectionDataset


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

    def disable_mosaic_mixup(self):
        """Turn off any Mosaic or Mixup in the chain (called by CloseMosaicCallback).

        ``TrainTransformPipeline`` holds them in named slots; a plain Compose just has
        them somewhere in the list, so find them by type. Idempotent.
        """
        for t in self.transforms:
            if isinstance(t, (Mosaic, Mixup)):
                t.enabled = False


class HSVAugment:
    """Randomly shift hue, saturation and value.

    Implemented with three 256-entry lookup tables rather than through
    Albumentations. The shift is constant across the image, so it is exactly a
    per-channel table lookup, and ``cv2.LUT`` applies one in a single pass. The
    Albumentations route cost two whole-image copies to swap BGR/RGB around a call
    that converted colour space again internally — 5.32 ms against 2.32 ms per
    640x640 image, measured 2026-09-19.

    The shift ranges are unchanged: OpenCV's uint8 HSV encoding puts hue in
    ``[0, 179]`` and saturation and value in ``[0, 255]``, which is the same
    convention ``HueSaturationValue`` used, so recipes keep their meaning.

    Args:
        hgain: Max hue shift, in OpenCV hue units. Matches the super-gradients
              ``hgain`` recipe param. Default: 18.
        sgain: Max saturation shift in absolute units. Default: 30.
        vgain: Max value shift in absolute units. Default: 30.
        p: Probability of applying the transform.
    """

    def __init__(self, hgain: int = 18, sgain: int = 30, vgain: int = 30, p: float = 0.5):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.p = p
        self._ramp = np.arange(256, dtype=np.int16)

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.p:
            return image, targets

        dh, ds, dv = (
            random.uniform(-1, 1) * self.hgain,
            random.uniform(-1, 1) * self.sgain,
            random.uniform(-1, 1) * self.vgain,
        )

        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        x = self._ramp
        # Hue is an angle, so it wraps; saturation and value saturate at the ends.
        lut_h = ((x + dh) % 180).astype(np.uint8)
        lut_s = np.clip(x + ds, 0, 255).astype(np.uint8)
        lut_v = np.clip(x + dv, 0, 255).astype(np.uint8)

        merged = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
        return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR), targets


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


class VerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            image = np.flipud(image).copy()
            if len(targets):
                targets = targets.copy()
                targets[:, 2] = 1.0 - targets[:, 2]  # flip y_center
        return image, targets


class RandomChannelShuffle:
    """Randomly permute BGR channels (equivalent to super-gradients DetectionRGB2BGR)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            perm = np.random.permutation(3)
            image = image[:, :, perm].copy()
        return image, targets


class RandomCrop:
    """Random crop with bbox clipping and re-normalization."""

    def __init__(self, min_scale: float = 0.3, max_scale: float = 1.0, p: float = 1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.p:
            return image, targets

        h, w = image.shape[:2]

        # Sample crop dimensions
        crop_h = int(h * random.uniform(self.min_scale, self.max_scale))
        crop_w = int(w * random.uniform(self.min_scale, self.max_scale))
        crop_h = max(1, crop_h)
        crop_w = max(1, crop_w)

        # Sample crop position
        y0 = random.randint(0, h - crop_h)
        x0 = random.randint(0, w - crop_w)

        image = image[y0 : y0 + crop_h, x0 : x0 + crop_w].copy()

        if len(targets):
            targets = targets.copy()
            # Convert normalized xywh to pixel xyxy
            x1 = (targets[:, 1] - targets[:, 3] / 2) * w
            y1 = (targets[:, 2] - targets[:, 4] / 2) * h
            x2 = (targets[:, 1] + targets[:, 3] / 2) * w
            y2 = (targets[:, 2] + targets[:, 4] / 2) * h

            # Shift to crop coordinates and clip
            x1 = np.clip(x1 - x0, 0, crop_w)
            y1 = np.clip(y1 - y0, 0, crop_h)
            x2 = np.clip(x2 - x0, 0, crop_w)
            y2 = np.clip(y2 - y0, 0, crop_h)

            # Filter out boxes that are too small
            box_w = x2 - x1
            box_h = y2 - y1
            valid = (box_w > 2) & (box_h > 2)

            targets = targets[valid]
            x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]

            if len(targets):
                targets[:, 1] = ((x1 + x2) / 2) / crop_w
                targets[:, 2] = ((y1 + y2) / 2) / crop_h
                targets[:, 3] = (x2 - x1) / crop_w
                targets[:, 4] = (y2 - y1) / crop_h

        return image, targets


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


class RandomResizedCropFlipAffine:
    """``RandomResizedCrop`` → ``HorizontalFlip`` → ``RandomAffine``, as one warp.

    The three transforms it replaces are each affine, so their matrices multiply and
    a single ``cv2.warpAffine`` produces the same output. That is worth doing twice
    over: it resamples the image once instead of three times, and it drops two full
    passes over a 640x640 array plus two Albumentations bbox round-trips.

    The parameters are drawn by calling Albumentations' own samplers, so the
    augmentation distribution is identical by construction rather than by
    reimplementation — ``A.Affine`` samples x and y scale *independently*, for one,
    which a hand-rolled version would be unlikely to reproduce.

    **Two matrices, deliberately.** OpenCV's ``resize`` and ``warpAffine`` map pixel
    *centres*: the source coordinate for output pixel ``d`` is ``(d + 0.5) * s - 0.5``.
    Albumentations' bbox crop is geometric — ``(src - x1) * S / crop`` with no half-pixel
    term — which is also why ``A.Affine`` returns a separate ``bbox_matrix``. Composing
    the image chain with the geometric convention shifts the picture by
    ``0.5 * (1 - S / crop)`` pixels, half a pixel when the crop is upscaled 2x, while
    the boxes stay put. Nothing raises; the labels simply stop matching the pixels.
    Measured: with the geometric matrix the 99th percentile of the difference against
    the chain is up to 14 grey levels, with the pixel-centre matrix it is 1.

    So ``_image_matrix`` carries the half-pixel terms and ``_box_matrix`` does not.
    They are not meant to agree.

    Only axis-aligned chains are supported. With rotation or shear, clipping a box once
    at the end is not the same as clipping it after each step, and this class would
    silently produce different boxes; it raises instead.

    Args:
        size: Output square side.
        scale: Crop area fraction range, as ``A.RandomResizedCrop``.
        ratio: Crop aspect ratio range, as ``A.RandomResizedCrop``.
        flip_prob: Probability of the horizontal flip.
        translate: Affine translation as a fraction of output size.
        affine_scale: Affine scale range.
        degrees: Rotation. Must be 0.
        shear: Shear. Must be 0.
        pad_value: Fill for pixels pulled in from outside the source.
        min_box_size: Drop boxes thinner than this, in output pixels. Matches the
            ``min_width`` / ``min_height`` of ``_BBOX_PARAMS``.
    """

    def __init__(
        self,
        size: int = 640,
        scale: tuple[float, float] = (0.05, 0.8),
        ratio: tuple[float, float] = (0.75, 1.33),
        flip_prob: float = 0.5,
        translate: float = 0.25,
        affine_scale: tuple[float, float] = (0.5, 1.5),
        degrees: float = 0.0,
        shear: float = 0.0,
        pad_value: int = 114,
        min_box_size: float = 2.0,
    ):
        if degrees != 0.0 or shear != 0.0:
            raise ValueError(
                "RandomResizedCropFlipAffine fuses an axis-aligned chain only. "
                f"Got degrees={degrees}, shear={shear}. Clipping boxes once at the end "
                "is equivalent to clipping them at each step only while the transform "
                "stays axis-aligned; under rotation or shear it is not, and the fused "
                "boxes would differ from the chain's without any error being raised. "
                "Use RandomResizedCrop + HorizontalFlip + RandomAffine for those."
            )

        self.size = size
        self.flip_prob = flip_prob
        self.pad_value = pad_value
        self.min_box_size = min_box_size

        # Sampling is delegated to Albumentations so the distributions cannot drift.
        self._crop = A.RandomResizedCrop(size=(size, size), scale=scale, ratio=ratio, p=1.0)
        self._affine = A.Affine(
            scale=affine_scale,
            translate_percent={"x": (-translate, translate), "y": (-translate, translate)},
            rotate=(0, 0),
            shear=(0, 0),
            border_mode=cv2.BORDER_CONSTANT,
            fill=pad_value,
            p=1.0,
        )

    # -- matrices -----------------------------------------------------------

    def _crop_matrix(self, crop: tuple[int, int, int, int], *, pixel_centre: bool) -> np.ndarray:
        x1, y1, x2, y2 = crop
        sx, sy = self.size / (x2 - x1), self.size / (y2 - y1)
        if pixel_centre:
            tx, ty = (0.5 - x1) * sx - 0.5, (0.5 - y1) * sy - 0.5
        else:
            tx, ty = -sx * x1, -sy * y1
        return np.array([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]])

    def _flip_matrix(self, *, pixel_centre: bool) -> np.ndarray:
        # cv2.flip sends column x to W-1-x; a normalised box coordinate goes to 1-x,
        # which in pixels is W-x. One pixel apart, and each is right for its target.
        offset = self.size - 1 if pixel_centre else self.size
        return np.array([[-1.0, 0.0, offset], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # -- api ----------------------------------------------------------------

    def sample_params(self, height: int, width: int) -> dict:
        """Draw one set of parameters. Split out so tests can drive both paths alike."""
        crop = self._crop.get_params_dependent_on_data({"shape": (height, width, 3)}, {})["crop_coords"]
        affine = self._affine.get_params_dependent_on_data({"shape": (self.size, self.size, 3)}, {})
        return {
            "crop": tuple(crop),
            "flip": random.random() < self.flip_prob,
            "matrix": np.asarray(affine["matrix"], dtype=np.float64),
            "bbox_matrix": np.asarray(affine["bbox_matrix"], dtype=np.float64),
        }

    def image_matrix(self, params: dict) -> np.ndarray:
        eye = np.eye(3)
        return (
            params["matrix"]
            @ (self._flip_matrix(pixel_centre=True) if params["flip"] else eye)
            @ self._crop_matrix(params["crop"], pixel_centre=True)
        )

    def box_matrix(self, params: dict) -> np.ndarray:
        eye = np.eye(3)
        return (
            params["bbox_matrix"]
            @ (self._flip_matrix(pixel_centre=False) if params["flip"] else eye)
            @ self._crop_matrix(params["crop"], pixel_centre=False)
        )

    def apply(self, image: np.ndarray, targets: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        out = cv2.warpAffine(
            image,
            self.image_matrix(params)[:2],
            (self.size, self.size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(self.pad_value,) * image.shape[2] if image.ndim == 3 else self.pad_value,
        )

        if not len(targets):
            return out, targets

        m = self.box_matrix(params)
        cx, cy, bw, bh = targets[:, 1] * w, targets[:, 2] * h, targets[:, 3] * w, targets[:, 4] * h
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2

        # Map both corners. Axis-aligned, so the transformed corners are still the
        # extremes; min/max guards against a negative scale (the flip).
        px = m[0, 0] * np.stack([x1, x2]) + m[0, 2]
        py = m[1, 1] * np.stack([y1, y2]) + m[1, 2]
        nx1, nx2 = px.min(0), px.max(0)
        ny1, ny2 = py.min(0), py.max(0)

        np.clip(nx1, 0, self.size, out=nx1)
        np.clip(nx2, 0, self.size, out=nx2)
        np.clip(ny1, 0, self.size, out=ny1)
        np.clip(ny2, 0, self.size, out=ny2)

        keep = ((nx2 - nx1) >= self.min_box_size) & ((ny2 - ny1) >= self.min_box_size)
        if not keep.any():
            return out, np.zeros((0, 5), dtype=targets.dtype)

        kept = np.empty((int(keep.sum()), 5), dtype=targets.dtype)
        kept[:, 0] = targets[keep, 0]
        kept[:, 1] = (nx1[keep] + nx2[keep]) / 2 / self.size
        kept[:, 2] = (ny1[keep] + ny2[keep]) / 2 / self.size
        kept[:, 3] = (nx2[keep] - nx1[keep]) / self.size
        kept[:, 4] = (ny2[keep] - ny1[keep]) / self.size
        return out, kept

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.apply(image, targets, self.sample_params(*image.shape[:2]))


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

    def __init__(self, dataset: BaseDetectionDataset, input_size: int = 640, prob: float = 1.0):
        self.dataset = dataset
        self.input_size = input_size
        self.prob = prob
        self.enabled = True

    def __call__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.enabled or random.random() >= self.prob:
            return self.dataset.load_raw(index)

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
            # Targets are normalized to the 2s x 2s canvas; the crop is s x s, so every
            # coordinate is renormalized by 2. Widths and heights need that factor just
            # as much as the centres do — omitting it emits every box at half size.
            targets[:, 1] = targets[:, 1] * 2 - crop_x / s
            targets[:, 2] = targets[:, 2] * 2 - crop_y / s
            targets[:, 3] = targets[:, 3] * 2
            targets[:, 4] = targets[:, 4] * 2

            # Clip to the crop window, then drop boxes whose centre left it or that
            # survive only as a sliver.
            x1 = np.clip(targets[:, 1] - targets[:, 3] / 2, 0.0, 1.0)
            y1 = np.clip(targets[:, 2] - targets[:, 4] / 2, 0.0, 1.0)
            x2 = np.clip(targets[:, 1] + targets[:, 3] / 2, 0.0, 1.0)
            y2 = np.clip(targets[:, 2] + targets[:, 4] / 2, 0.0, 1.0)
            centre_inside = (
                (targets[:, 1] > 0) & (targets[:, 1] < 1)
                & (targets[:, 2] > 0) & (targets[:, 2] < 1)
            )
            targets[:, 1] = (x1 + x2) / 2
            targets[:, 2] = (y1 + y2) / 2
            targets[:, 3] = x2 - x1
            targets[:, 4] = y2 - y1

            valid = centre_inside & (targets[:, 3] > 0.002) & (targets[:, 4] > 0.002)
            targets = targets[valid]

        return mosaic_img, targets


class Mixup:
    """Mixup augmentation for detection.

    Should be placed in the pipeline **after** ``LetterboxResize`` and
    **before** ``Normalize``, so both images are already square uint8.

    Args:
        dataset: Dataset exposing a ``load_raw(index)`` method.
        alpha: Beta distribution ``alpha`` parameter.
        beta: Beta distribution ``beta`` parameter.
        prob: Per-sample probability of applying mixup.
    """

    def __init__(
        self,
        dataset: BaseDetectionDataset,
        alpha: float = 1.5,
        beta: float = 1.5,
        prob: float = 0.5,
    ):
        self.dataset = dataset
        self.alpha = alpha
        self.beta = beta
        self.prob = prob
        self.enabled = True
        self.inner_transforms: Compose | None = None

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.enabled or random.random() >= self.prob:
            return image, targets

        idx2 = random.randint(0, len(self.dataset) - 1)
        img2, targets2 = self.dataset.load_raw(idx2)

        # Per-image augmentations for the mixed-in image, so it is not a pristine
        # sample pasted onto an augmented one.
        if self.inner_transforms is not None:
            img2, targets2 = self.inner_transforms(img2, targets2)

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

        r = float(np.random.beta(self.alpha, self.beta))
        # cv2 blends in one SIMD pass over uint8. Promoting both images to float32
        # first allocated two 4.9 MB arrays per call and cost 1.47 ms against 0.30 ms,
        # measured 2026-09-19. The results differ by at most one grey level, and in
        # cv2's favour: it rounds, where the float32 cast truncated.
        mixed = cv2.addWeighted(image, r, img2, 1.0 - r, 0.0)

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
    """Convert HWC BGR uint8 to CHW RGB, optionally leaving the scaling to the GPU.

    With ``dtype="uint8"`` the array stays 1.23 MB per 640x640 sample instead of
    4.92 MB, which is what then crosses the collate function, the ``pin_memory``
    thread and the PCIe bus; ``YoloNASLightningModule.on_after_batch_transfer``
    divides by 255 on the device, where it is free. Measured 2026-09-19:
    3.23 ms to 0.28 ms per sample, and the reconstructed float32 is bit-identical.

    The default stays ``"float32"`` so that every consumer that reads batches
    directly — ``yolonas eval``, ``yolonas quantize``, the examples — is unaffected.

    Args:
        dtype: ``"float32"`` to divide by 255 here, ``"uint8"`` to defer it.
    """

    def __init__(self, dtype: str = "float32"):
        if dtype not in ("float32", "uint8"):
            raise ValueError(f"dtype must be 'float32' or 'uint8', got {dtype!r}")
        self.dtype = dtype

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # BGR → RGB and HWC → CHW in one go; ascontiguousarray materialises the
        # result once, where the old chain copied for the flip and again for the cast.
        image = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
        if self.dtype == "float32":
            image = image.astype(np.float32) / 255.0
        return image, targets


class TrainTransformPipeline:
    """Full training augmentation pipeline with Mosaic and Mixup support.

    Pipeline order (matching super-gradients):
    Mosaic → per_image_transforms (RandomAffine, RandomChannelShuffle, HSV, HorizontalFlip)
    → Mixup → final_transforms (LetterboxResize, Normalize)
    """

    def __init__(
        self,
        mosaic: Mosaic | None,
        per_image_transforms: Compose,
        mixup: Mixup | None,
        final_transforms: Compose,
    ):
        self.mosaic = mosaic
        self.per_image_transforms = per_image_transforms
        self.mixup = mixup
        self.final_transforms = final_transforms

    def apply(self, index: int, load_raw_fn) -> tuple[np.ndarray, np.ndarray]:
        """Dataset-aware pipeline entry point called from __getitem__."""
        # 1. Mosaic (or fallback to load_raw)
        if self.mosaic is not None:
            image, targets = self.mosaic(index)
        else:
            image, targets = load_raw_fn(index)

        # 2. Per-image transforms
        image, targets = self.per_image_transforms(image, targets)

        # 3. Mixup
        if self.mixup is not None:
            image, targets = self.mixup(image, targets)

        # 4. Final transforms (resize + normalize)
        image, targets = self.final_transforms(image, targets)

        return image, targets

    def __call__(self, image: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fallback: per-image + final only (no Mosaic/Mixup)."""
        image, targets = self.per_image_transforms(image, targets)
        image, targets = self.final_transforms(image, targets)
        return image, targets

    def disable_mosaic_mixup(self):
        """Disable Mosaic and Mixup (called by CloseMosaicCallback)."""
        if self.mosaic is not None:
            self.mosaic.enabled = False
        if self.mixup is not None:
            self.mixup.enabled = False
