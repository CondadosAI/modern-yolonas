/**
 * Browser-side YOLO-NAS: fetch the ONNX graph, preprocess on a canvas, run it under
 * onnxruntime-web, and draw the result. Nothing is uploaded — a static Space serves
 * files and runs no code of its own.
 */

import { decodeDetections, letterboxGeometry } from "./postprocess.js";

const ORT_VERSION = "1.30.0";
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const MODEL_URL = "models/yolo-nas-s.onnx";
const EXAMPLE_URL = "examples/street.jpg";
const INPUT_SIZE = 640;
const PAD_VALUE = 114;

const els = {
  file: document.getElementById("file"),
  conf: document.getElementById("conf"),
  iou: document.getElementById("iou"),
  confVal: document.getElementById("confVal"),
  iouVal: document.getElementById("iouVal"),
  run: document.getElementById("run"),
  example: document.getElementById("example"),
  status: document.getElementById("status"),
  bar: document.getElementById("bar"),
  barFill: document.querySelector("#bar > i"),
  canvas: document.getElementById("canvas"),
  table: document.getElementById("table"),
  tbody: document.querySelector("#table tbody"),
};

let session = null;
let currentImage = null;

const setStatus = (text) => { els.status.textContent = text; };

function showProgress(fraction) {
  els.bar.style.display = "block";
  els.barFill.style.width = `${Math.round(fraction * 100)}%`;
}

function hideProgress() {
  els.bar.style.display = "none";
  els.barFill.style.width = "0";
}

/** Fetch with a progress bar — 47 MB with no feedback reads as a broken page. */
async function fetchModel(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Model fetch failed: ${response.status}`);

  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) {
      showProgress(received / total);
      setStatus(`Downloading model… ${(received / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`);
    } else {
      setStatus(`Downloading model… ${(received / 1048576).toFixed(1)} MB`);
    }
  }

  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length; }
  return buffer;
}

/**
 * Letterbox onto a 640x640 canvas and return an NCHW float tensor.
 *
 * Mirrors `preprocess.py`: longest side to 636, centered on a 114-filled canvas,
 * RGB, scaled to [0,1]. Canvas resampling is not bit-identical to OpenCV's
 * INTER_LINEAR, so scores can differ in the third decimal.
 */
function preprocess(image) {
  const geom = letterboxGeometry(image.naturalWidth, image.naturalHeight, INPUT_SIZE);

  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = `rgb(${PAD_VALUE}, ${PAD_VALUE}, ${PAD_VALUE})`;
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(image, geom.padLeft, geom.padTop, geom.newW, geom.newH);

  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const plane = INPUT_SIZE * INPUT_SIZE;
  const tensor = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const p = i * 4;
    tensor[i] = data[p] / 255;                 // R
    tensor[plane + i] = data[p + 1] / 255;     // G
    tensor[2 * plane + i] = data[p + 2] / 255; // B
  }
  return { tensor, geom };
}

function draw(image, detections) {
  const canvas = els.canvas;
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0);

  // Scale the furniture with the image so a 4000px photo is not annotated in
  // hairlines, but cap it: a crowded street scene at full scale is all label and
  // no picture.
  const unit = Math.min(Math.max(Math.max(canvas.width, canvas.height) / 1000, 1), 2);
  const lineWidth = Math.max(2, Math.round(2 * unit));
  const fontSize = Math.max(12, Math.round(13 * unit));
  ctx.lineWidth = lineWidth;
  ctx.font = `600 ${fontSize}px system-ui, sans-serif`;
  ctx.textBaseline = "top";

  for (const det of detections) {
    const [x1, y1, x2, y2] = det.box;
    // Stable per-class hue: the same class keeps its colour across images.
    const hue = (det.classId * 47) % 360;
    const color = `hsl(${hue}, 85%, 55%)`;

    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `${det.name} ${det.score.toFixed(2)}`;
    const padX = Math.round(5 * unit);
    const width = ctx.measureText(label).width + padX * 2;
    const height = fontSize + Math.round(7 * unit);
    // Keep the label inside the frame: above the box unless that clips the top,
    // and pulled left when a box near the right edge would push it off-canvas.
    const labelY = y1 - height < 0 ? y1 : y1 - height;
    const labelX = Math.min(Math.max(x1 - lineWidth / 2, 0), Math.max(canvas.width - width, 0));

    ctx.fillStyle = color;
    ctx.fillRect(labelX, labelY, width, height);
    ctx.fillStyle = "#000";
    ctx.fillText(label, labelX + padX, labelY + Math.round(3 * unit));
  }
}

function fillTable(detections) {
  els.tbody.replaceChildren();
  for (const det of detections) {
    const row = document.createElement("tr");
    const box = det.box.map((v) => Math.round(v)).join(", ");
    for (const text of [det.name, det.score.toFixed(3), box]) {
      const cell = document.createElement("td");
      cell.textContent = text;
      row.appendChild(cell);
    }
    els.tbody.appendChild(row);
  }
  els.table.hidden = detections.length === 0;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    // The bundled example is stored in LFS, which Hugging Face serves by redirecting
    // to a CDN on another origin. Without this the canvas is tainted and getImageData
    // throws, so preprocessing fails on the deployed Space while working locally.
    // Object URLs for user-picked files are same-origin and unaffected.
    if (!src.startsWith("blob:") && !src.startsWith("data:")) {
      image.crossOrigin = "anonymous";
    }
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not decode that image"));
    image.src = src;
  });
}

async function detect() {
  if (!session || !currentImage) return;
  els.run.disabled = true;
  setStatus("Running inference…");
  // Yield once so the status paints before the main thread blocks on WASM.
  await new Promise((r) => setTimeout(r, 0));

  try {
    const { tensor, geom } = preprocess(currentImage);
    const feeds = { images: new ort.Tensor("float32", tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]) };

    const started = performance.now();
    const output = await session.run(feeds);
    const elapsed = performance.now() - started;

    const bboxes = output.pred_bboxes;
    const scores = output.pred_scores;
    const detections = decodeDetections(bboxes.data, scores.data, {
      numAnchors: bboxes.dims[1],
      numClasses: scores.dims[2],
      confThreshold: Number(els.conf.value),
      iouThreshold: Number(els.iou.value),
      scale: geom.scale,
      padLeft: geom.padLeft,
      padTop: geom.padTop,
      origW: currentImage.naturalWidth,
      origH: currentImage.naturalHeight,
    });

    draw(currentImage, detections);
    fillTable(detections);
    setStatus(`${detections.length} object${detections.length === 1 ? "" : "s"} · ${elapsed.toFixed(0)} ms in-browser`);
  } catch (error) {
    setStatus(`Inference failed: ${error.message}`);
    throw error;
  } finally {
    els.run.disabled = false;
  }
}

async function useImage(src) {
  currentImage = await loadImage(src);
  draw(currentImage, []);
  fillTable([]);
  els.run.disabled = false;
  await detect();
}

async function init() {
  els.conf.addEventListener("input", () => { els.confVal.textContent = Number(els.conf.value).toFixed(2); });
  els.iou.addEventListener("input", () => { els.iouVal.textContent = Number(els.iou.value).toFixed(2); });
  els.run.addEventListener("click", detect);
  els.example.addEventListener("click", () => useImage(EXAMPLE_URL).catch((e) => setStatus(e.message)));
  els.file.addEventListener("change", () => {
    const file = els.file.files?.[0];
    if (file) useImage(URL.createObjectURL(file)).catch((e) => setStatus(e.message));
  });

  try {
    // A static Space cannot send the COOP/COEP headers that SharedArrayBuffer needs,
    // so multi-threaded WASM is unavailable here by construction.
    ort.env.wasm.wasmPaths = ORT_DIST;
    ort.env.wasm.numThreads = 1;

    const modelBytes = await fetchModel(MODEL_URL);
    hideProgress();
    setStatus("Initialising runtime…");
    session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    els.example.disabled = false;
    setStatus("Model ready — load the example or pick an image.");
  } catch (error) {
    hideProgress();
    setStatus(`Could not load the model: ${error.message}`);
    throw error;
  }
}

init();
