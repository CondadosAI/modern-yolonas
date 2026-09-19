# Hugging Face Spaces

Source of truth for the public demos. Each subfolder maps 1:1 to a Space repository root —
what is in the folder is what gets uploaded, nothing more.

This file stays here and is **not** part of any Space upload.

| Folder | Space | Hardware | Status |
|---|---|---|---|
| `static-demo/` | [CondadosAI/modern-yolonas-demo](https://huggingface.co/spaces/CondadosAI/modern-yolonas-demo) | static (free) | **live** |
| `gradio-demo/` | not deployed | `cpu-basic` (**needs HF PRO**) | ready to push |

## Why there are two

Hugging Face now bills Gradio and Docker Spaces: hosting one on free `cpu-basic` requires a PRO
subscription, and creating one without it fails with `402 Payment Required`. Static Spaces remain
free for everyone. `static-demo/` is therefore the deployed demo, and `gradio-demo/` is kept
finished and tested for whenever a PRO account exists.

---

## `static-demo/` — browser inference, free

No server: the Space serves files, and YOLO-NAS runs in the visitor's browser via
`onnxruntime-web`. Nothing is uploaded, which also sidesteps the abuse surface of a public
inference endpoint.

- `models/yolo-nas-s.onnx` — opset 17, fp32, 46.5 MB, produced by `yolonas export`.
- `postprocess.js` — a port of `inference/postprocess.py` (multi-label filtering, class-aware NMS
  at IoU 0.7, top-1024 before NMS, letterbox rescale) plus the geometry from `preprocess.py`.
  Kept in its own module so Node can check it against the Python pipeline.
- `app.js` — canvas letterboxing, ORT session, rendering.

### Two facts worth not re-deriving

**fp32, not int8.** Dynamic int8 quantization gives 12.5 MB instead of 46.5 MB, but measured on the
sample image it returned 17 detections instead of 19 **and was not faster** (148 ms vs 146 ms under
native onnxruntime). It buys download size and nothing else.

**Single-threaded WASM.** A static Space cannot send the COOP/COEP headers `SharedArrayBuffer`
needs, so `ort.env.wasm.numThreads = 1` is a constraint, not a choice. One forward pass is ~0.75 s.

### Verifying before pushing

Parity against the Python pipeline, given an identical input tensor (19/19 detections, 0 class
mismatches, max score delta 6e-7, max box delta 0.000 px):

```bash
cd spaces/static-demo
uv run python tests/make_fixtures.py   # needs modern-yolonas + the onnx extra
npm install onnxruntime-web
node tests/test_parity.mjs             # prints PASS, exits non-zero on drift
```

In a browser, which also exercises canvas letterboxing:

```bash
cd spaces/static-demo && python3 -m http.server 8099
# then open http://127.0.0.1:8099/index.html
```

Expect 18 detections on the example at confidence 0.40 — one fewer than Python, because canvas
resampling is not OpenCV's `INTER_LINEAR` and one marginal box falls under the threshold. That is
preprocessing, not the port; `tests/test_parity.mjs` is what pins the port itself.

### Deploying

```bash
hf auth login
hf repos create CondadosAI/modern-yolonas-demo --type space --space-sdk static --exist-ok
```

Upload through the Python API rather than `hf upload`: the CLI calls `create_repo` on every
invocation without passing an SDK, which the Hub answers with `402 Payment Required` for a
non-PRO account even when the static Space already exists.

```python
from huggingface_hub import HfApi

HfApi().upload_folder(
    folder_path="spaces/static-demo",
    repo_id="CondadosAI/modern-yolonas-demo",
    repo_type="space",
    # Dev-only files: the fixtures and the parity test are not part of the page, and a
    # .gitignore written for this repo would mislead anyone reading the Space's.
    ignore_patterns=[".gitignore", "tests/*", "node_modules/*", "__pycache__/*", "*.pyc"],
)
```

Static Spaces have no build step, so the push is live in seconds.

### Regenerating the ONNX

```bash
uv run yolonas export --model yolo_nas_s --format onnx \
    --output spaces/static-demo/models/yolo-nas-s.onnx --opset 17
```

Needs the `onnx` extra. The export must keep the two-output signature
(`pred_bboxes` `[1,8400,4]`, `pred_scores` `[1,8400,80]`) that `app.js` reads by name — boxes arrive
already decoded, so there is no DFL logic in the JavaScript. Do **not** use `--target frigate`
here: it bakes NMS into the graph and changes the outputs.

---

## `gradio-demo/` — the PRO path

A Gradio app running `Detector` server-side on `cpu-basic`. Complete and locally verified
(including the full `gradio_client` request path); it only needs an account that may create one.

- `torch==2.9.1+cpu` / `torchvision==0.24.1+cpu` from
  `--extra-index-url https://download.pytorch.org/whl/cpu`. The default PyPI `torch` bundles
  roughly 2.5 GB of CUDA libraries that this hardware cannot use.
- `modern-yolonas` is pinned to a **released** version and the app uses only that version's API.
  `0.4.0` calls the class `Detector`; `0.5.0` renames it to `YoloNASDetector` and keeps `Detector`
  as a deprecated alias until `0.7.0`. Bump the pin and the class name together.
- Gradio is deliberately absent from `requirements.txt` — the Space installs the version given by
  `sdk_version` in the frontmatter, and listing it in both places invites a conflict.

Measured on two cores: 0.5 s for S, 1.0 s for M, 1.4 s for L per 640×640 image.

### Deploying (once the account is PRO)

```bash
hf repos create CondadosAI/<name> --type space --space-sdk gradio --exist-ok
hf upload CondadosAI/<name> spaces/gradio-demo . --type space --exclude "__pycache__/*"
```

### Verifying before pushing

The build loop on Spaces is ~10 minutes; the local one is one minute.

```bash
uv venv /tmp/spacetest --python 3.11
VIRTUAL_ENV=/tmp/spacetest uv pip install -r spaces/gradio-demo/requirements.txt \
    gradio==6.28.0 --index-strategy unsafe-best-match
cd spaces/gradio-demo && CUDA_VISIBLE_DEVICES="" /tmp/spacetest/bin/python app.py
```

`--index-strategy unsafe-best-match` is what lets uv see the `+cpu` wheels on the PyTorch index
alongside everything else on PyPI. Spaces uses plain pip, which already behaves this way.

### Optional: `HF_TOKEN` secret

Without one, the Space downloads weights anonymously and logs a rate-limit warning. A read-scoped
`HF_TOKEN` in **Settings → Variables and secrets** raises the limit. Not required — the weights
repo is public.
