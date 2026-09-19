# ARM support: a plan, not a table

No ARM machine has ever run this project. Everything below is either verified from
package metadata and vendor documentation — each such claim says how it was checked —
or written as a prediction with the experiment that would settle it. Nothing here is
a measurement, and no number in this file belongs in the README until an ARM board
produces it.

The short version: **packaging is not the problem, and one line of `pyproject.toml`
is.** Every binary dependency already publishes `linux-aarch64` and `macos-arm64`
wheels. What does not work is the CUDA index pin, which sends a Raspberry Pi to a
3 GB CUDA build it cannot use.

## What the dependency graph already allows

Checked 2026-09-19 against the PyPI JSON API, latest release of each:

| Package | linux-aarch64 | macos-arm64 | Note |
|---|:--:|:--:|---|
| `torch` 2.14 | ✅ | ✅ | PyPI's aarch64 wheel is CPU-only; MPS on macOS |
| `torchvision` 0.29 | ✅ | ✅ | |
| `onnxruntime` 1.30 | ✅ | ✅ | also `win-arm64` |
| `openvino` 2026.4 | ✅ | ✅ | ARM CPU plugin; no iGPU plugin on ARM |
| `opencv-python-headless` | ✅ | ✅ | |
| `numpy`, `pycocotools` | ✅ | ✅ | |
| `nncf`, `supervision`, `lightning` | pure Python | pure Python | |
| `tensorrt-cu13` | see below | ❌ | PyPI ships a stub that pulls from NVIDIA's index |

So `pip install modern-yolonas[onnx]` on an aarch64 Linux box should already resolve.
`uv sync` in this repo should not, for the reason below.

## The one concrete blocker

`pyproject.toml` pins torch to NVIDIA's CUDA 13 wheel index:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu130" }
```

That index **does** publish `manylinux_2_28_aarch64` wheels — they are the SBSA build
for Grace-Hopper — and the platform tag does not distinguish a GH200 from a Raspberry
Pi 5. A Pi would therefore resolve to a multi-gigabyte CUDA build with no CUDA under
it. Verified by listing the index: 89 aarch64 filenames under `whl/cu130/torch/`.

The fix is a marker, so only x86_64 (and an explicitly opted-in SBSA host) takes the
CUDA index and everything else takes `whl/cpu`. It is a small change and it should be
made with an ARM machine in hand to confirm, rather than blind.

## The four targets, in the order they are worth doing

### 1. Generic aarch64 Linux — Raspberry Pi 5, Graviton, Ampere Altra

The cheapest target and the one that proves the rest. No GPU path: ONNX Runtime's CPU
provider and OpenVINO's ARM CPU plugin, both from PyPI.

The interesting number is INT8. OpenVINO's INT8 on x86 CPU is the single largest
speedup this project measures, and on ARM it runs on different kernels (dotprod /
i8mm on a Cortex-A76 or Neoverse), so the ratio does not carry over. `320` is the
size that matters here — at `640` a Pi is not a real-time device under any runtime.

Expected first failure: none in installation; a long first-inference pause while
OpenVINO compiles.

### 2. NVIDIA Jetson Orin (Nano / NX / AGX)

The target people actually ask for, and the one with the most ways to go wrong.

- **TensorRT comes from JetPack**, preinstalled in `/usr/lib/python3*/dist-packages`.
  Installing the `tensorrt` extra from PyPI on a Jetson is the mistake to guard
  against: it pulls a desktop build against a CUDA that L4T does not have.
- **torch comes from NVIDIA's Jetson index**, not PyPI. The PyPI aarch64 wheel is
  CPU-only, so `torch.cuda.is_available()` returns `False` and every GPU leg silently
  becomes a CPU leg — exactly the failure `onnx_session`'s provider assertion exists
  to catch, and `bench_pytorch` currently has no equivalent guard for.
- A Jetson venv usually has to be created with `--system-site-packages` so it can see
  the JetPack-installed `tensorrt`.
- Orin is `sm_87`, which is Ampere, so **an `--hardware-compatible` engine built on any
  Ampere-or-newer desktop card should load on an Orin**. That is the prediction most
  worth testing, because if it holds, the published engines cover Jetson for free.
  If it does not, the fallback is what this project already recommends: publish the
  ONNX and build the engine on the device.

Power mode dominates Jetson numbers. `nvpmodel -q` and `jetson_clocks` have to be
recorded beside any measurement, the same way `power_source` already is — an Orin
Nano at 7 W and at 15 W are two different devices.

### 3. Apple Silicon

`torch` with the MPS backend, and ONNX Runtime's CoreML execution provider. Neither is
exercised anywhere in this codebase: `bench_pytorch` hardcodes `.cuda()`, and
`onnx_session` has no `"coreml"` entry. Both are small additions, and the CoreML one
needs the same "did the provider actually resolve" assertion the CUDA one has.

The model itself should be fine — it is convolutions and elementwise arithmetic, with
no custom op.

### 4. Android / NNAPI, Qualcomm QNN

Out of scope until someone asks. Listing it so the absence is deliberate.

## How this gets tested

**CI (free, today).** `ubuntu-24.04-arm` is generally available and free for public
repositories, 4 vCPU, which is enough for the part that actually needs guarding:
that the package installs and that the exported ONNX produces the same numbers on
aarch64 as on x86_64. That is a correctness job, not a latency job — a shared 4 vCPU
runner cannot produce a latency number worth publishing, and pretending otherwise
would put a meaningless row in the table.

```yaml
  arm64:
    runs-on: ubuntu-24.04-arm
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --dev --extra onnx
      - run: uv run pytest tests/test_export.py tests/test_model.py -v
```

This job is what would have caught the CUDA-index pin.

**Hardware, for latency.** A Pi 5 and an Orin Nano, benchmarked with the same
`examples/runtime_matrix.py` that produced the x86_64 table, writing to the same JSON
schema so the rows merge. The script already records `platform.machine()`, the power
source and the GPU clock under load, so an ARM run is a new set of rows rather than a
new script. Jetson needs `nvpmodel` added to `environment()` first.

**Emulation is not a substitute.** QEMU can tell us the package installs and the
outputs match. It cannot tell us anything about latency, and a number from it would
be worse than no number.

## What this does not commit to

Publishing prebuilt ARM artifacts. The ONNX files are architecture-independent and
already cover ARM; OpenVINO IR is too. A Jetson engine would be one more thing to
build, version and explain, and it is only worth it once the AMPERE_PLUS prediction
above has been tested one way or the other.
