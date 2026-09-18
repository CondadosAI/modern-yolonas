/**
 * Confidence filtering, class-aware NMS and box rescaling.
 *
 * A direct port of `modern_yolonas/inference/postprocess.py`, kept in its own module
 * so the same code that runs in the browser can be checked in Node against the
 * detections the Python pipeline produces for the same input.
 *
 * The model has already decoded its boxes: it emits `pred_bboxes` [1, 8400, 4] in
 * x1y1x2y2 pixel coordinates on the 640x640 letterboxed canvas, and `pred_scores`
 * [1, 8400, 80] of per-class probabilities. Nothing here touches DFL.
 */

export const COCO_NAMES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
  "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
  "toothbrush",
];

/** Matches super-gradients' `nms_top_k`: candidates kept before NMS runs. */
const NMS_TOP_K = 1024;

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (inter === 0) return 0;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

/**
 * Filter, suppress and rescale raw model output.
 *
 * @param {Float32Array} bboxes   Flat [numAnchors * 4] x1y1x2y2 on the letterboxed canvas.
 * @param {Float32Array} scores   Flat [numAnchors * numClasses] class probabilities.
 * @param {object} opts
 * @param {number} opts.numAnchors
 * @param {number} opts.numClasses
 * @param {number} opts.confThreshold
 * @param {number} opts.iouThreshold
 * @param {number} opts.scale      Letterbox scale factor.
 * @param {number} opts.padLeft
 * @param {number} opts.padTop
 * @param {number} opts.origW
 * @param {number} opts.origH
 * @param {number} [opts.maxDetections=300]
 * @returns {{box: number[], score: number, classId: number, name: string}[]} Sorted by score, descending.
 */
export function decodeDetections(bboxes, scores, opts) {
  const {
    numAnchors, numClasses, confThreshold, iouThreshold,
    scale, padLeft, padTop, origW, origH, maxDetections = 300,
  } = opts;

  // Multi-label, as the Python defaults to: one anchor may survive for several
  // classes, so candidates are (anchor, class) pairs rather than one per anchor.
  const candidates = [];
  for (let a = 0; a < numAnchors; a++) {
    const scoreBase = a * numClasses;
    for (let c = 0; c < numClasses; c++) {
      const score = scores[scoreBase + c];
      if (score > confThreshold) {
        const b = a * 4;
        candidates.push({
          box: [bboxes[b], bboxes[b + 1], bboxes[b + 2], bboxes[b + 3]],
          score,
          classId: c,
        });
      }
    }
  }
  if (candidates.length === 0) return [];

  candidates.sort((p, q) => q.score - p.score);
  const pool = candidates.length > NMS_TOP_K ? candidates.slice(0, NMS_TOP_K) : candidates;

  // Class-aware NMS. torchvision's batched_nms offsets boxes by class so that
  // different classes can never suppress each other; walking the globally
  // score-sorted pool and only comparing same-class boxes is equivalent.
  const kept = [];
  const suppressed = new Uint8Array(pool.length);
  for (let i = 0; i < pool.length && kept.length < maxDetections; i++) {
    if (suppressed[i]) continue;
    kept.push(pool[i]);
    for (let j = i + 1; j < pool.length; j++) {
      if (suppressed[j] || pool[j].classId !== pool[i].classId) continue;
      if (iou(pool[i].box, pool[j].box) > iouThreshold) suppressed[j] = 1;
    }
  }

  // Undo the letterbox, then clip to the source image.
  return kept.map(({ box, score, classId }) => {
    const x1 = Math.min(Math.max((box[0] - padLeft) / scale, 0), origW);
    const y1 = Math.min(Math.max((box[1] - padTop) / scale, 0), origH);
    const x2 = Math.min(Math.max((box[2] - padLeft) / scale, 0), origW);
    const y2 = Math.min(Math.max((box[3] - padTop) / scale, 0), origH);
    return { box: [x1, y1, x2, y2], score, classId, name: COCO_NAMES[classId] ?? `class_${classId}` };
  });
}

/**
 * Letterbox geometry, matching `preprocess.letterbox`.
 *
 * The longest side is resized to 636 rather than the full 640, which is what
 * super-gradients does; the leftover pixels become a guaranteed border.
 */
export function letterboxGeometry(srcW, srcH, targetSize = 640, rescaleSize = 636) {
  const rescale = Math.min(rescaleSize, targetSize);
  const scale = rescale / Math.max(srcH, srcW);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  return {
    scale,
    newW,
    newH,
    padLeft: Math.floor((targetSize - newW) / 2),
    padTop: Math.floor((targetSize - newH) / 2),
  };
}
