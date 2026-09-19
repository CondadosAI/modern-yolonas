/**
 * Parity check: postprocess.js must reproduce the Python pipeline's detections
 * when both are given the identical input tensor.
 *
 *   python tests/make_fixtures.py      # from an env with modern-yolonas
 *   npm install onnxruntime-web
 *   node tests/test_parity.mjs
 *
 * Runs the model under the same WASM execution provider the browser uses, single
 * threaded, because a static Space cannot send the COOP/COEP headers that
 * SharedArrayBuffer requires.
 */

import * as ort from "onnxruntime-web";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { decodeDetections } from "../postprocess.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEMO = path.join(HERE, "..");

// Tolerances: float accumulation order differs between torch and the WASM kernels,
// so exact equality is not the bar. Anything above these means a real port bug.
const MAX_SCORE_DELTA = 1e-3;
const MAX_BOX_DELTA = 1.0; // pixels, in original-image coordinates

ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

const meta = JSON.parse(fs.readFileSync(path.join(HERE, "meta.json")));
const reference = JSON.parse(fs.readFileSync(path.join(HERE, "reference.json")));
const raw = fs.readFileSync(path.join(HERE, "input.bin"));
const input = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);

const session = await ort.InferenceSession.create(path.join(DEMO, "models", "yolo-nas-s.onnx"), {
  executionProviders: ["wasm"],
  graphOptimizationLevel: "all",
});

const started = Date.now();
const output = await session.run({ images: new ort.Tensor("float32", input, [1, 3, 640, 640]) });
console.log(`inference: ${((Date.now() - started) / 1000).toFixed(2)}s (wasm, 1 thread)`);

const { pred_bboxes: bboxes, pred_scores: scores } = output;
const detections = decodeDetections(bboxes.data, scores.data, {
  numAnchors: bboxes.dims[1],
  numClasses: scores.dims[2],
  confThreshold: meta.conf,
  iouThreshold: meta.iou,
  scale: meta.scale,
  padLeft: meta.pad[0],
  padTop: meta.pad[1],
  origW: meta.orig[1],
  origH: meta.orig[0],
});

const failures = [];
if (detections.length !== reference.length) {
  failures.push(`count: JS ${detections.length} vs Python ${reference.length}`);
}

let worstScore = 0;
let worstBox = 0;
for (let i = 0; i < Math.min(detections.length, reference.length); i++) {
  const [name, score, box] = reference[i];
  if (detections[i].name !== name) {
    failures.push(`class at ${i}: JS ${detections[i].name} vs Python ${name}`);
    continue;
  }
  worstScore = Math.max(worstScore, Math.abs(detections[i].score - score));
  for (let k = 0; k < 4; k++) {
    worstBox = Math.max(worstBox, Math.abs(detections[i].box[k] - box[k]));
  }
}
if (worstScore > MAX_SCORE_DELTA) failures.push(`score delta ${worstScore.toExponential(2)}`);
if (worstBox > MAX_BOX_DELTA) failures.push(`box delta ${worstBox.toFixed(3)}px`);

console.log(`detections: ${detections.length} (Python: ${reference.length})`);
console.log(`max score delta: ${worstScore.toExponential(2)} | max box delta: ${worstBox.toFixed(3)}px`);

if (failures.length) {
  console.error("\nFAIL\n  " + failures.join("\n  "));
  process.exit(1);
}
console.log("\nPASS");
