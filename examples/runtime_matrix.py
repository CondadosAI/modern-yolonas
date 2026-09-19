"""Latency across runtime x device x precision x input size.

Replaces the one-off `bench_devices.py`, which read `ov_{size}.xml` files it did not
produce and calibrated against a video that was never in the repo. Everything here is
built from the model this package exports, so the table regenerates:

    uv run examples/runtime_matrix.py --models yolo_nas_s --sizes 320,640 \
        --calibration-dir ~/datasets/coco/images/val2017

Model inference only unless the leg says `nms=graph`: no letterboxing, no host-side
NMS, single stream, synchronous, median of 30 runs after 8 warmup runs. Latency, not
throughput — an async multi-stream pipeline reports higher FPS on the same hardware.

Legs that fail are recorded with the reason rather than dropped. A runtime missing
from the environment is a row that says so, which is more useful than a gap.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

RUNS, WARMUP = 30, 8

# Set by main() from --sample-image. Random noise would be a fine input for the
# convolutions and a misleading one for NMS: nothing in noise clears a 0.25 score, so
# every NMS leg would sort an empty list and report itself free.
SAMPLE: dict[int, "np.ndarray"] = {}


def sample_input(size: int):
    """A real preprocessed frame at *size*, or noise if none was given."""
    if size in SAMPLE:
        return SAMPLE[size].copy()
    return np.random.randn(1, 3, size, size).astype(np.float32)


# --------------------------------------------------------------------------------
# Conditions. A latency number without these is not reproducible: the same laptop GPU
# runs ~40% slower on battery, and an idle card that never ramped reports a clock it
# was not running at.
# --------------------------------------------------------------------------------
def _nvidia_smi(query: str) -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def power_source() -> str:
    for online in Path("/sys/class/power_supply").glob("*/online"):
        try:
            if online.read_text().strip() == "1":
                return "mains"
        except OSError:
            continue
    return "battery-or-unknown"


def gpu_clock_health() -> dict:
    """SM clock as a fraction of maximum. Sample this *after* a timed run."""
    reading = _nvidia_smi("clocks.sm,clocks.max.sm,temperature.gpu")
    try:
        sm, sm_max, temp = (int(p.strip().split()[0]) for p in reading.split(","))
    except Exception:
        return {"sm_clock": reading}
    return {"sm_mhz": sm, "sm_max_mhz": sm_max, "sm_fraction": round(sm / sm_max, 3), "gpu_temp_c": temp}


def environment() -> dict:
    env = {
        "cpu": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "power_source": power_source(),
        "input": "real frame" if SAMPLE else "random noise (NMS legs are not meaningful)",
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()
        env["driver"] = _nvidia_smi("driver_version")
    for name, module in (("onnxruntime", "onnxruntime"), ("openvino", "openvino"), ("tensorrt", "tensorrt")):
        try:
            env[name] = __import__(module).__version__
        except Exception as exc:
            env[name] = f"absent ({type(exc).__name__})"
    return env


# --------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------
def timeit(call) -> float:
    """Median milliseconds of one call, warmup discarded whole."""
    for _ in range(WARMUP):
        call()
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        call()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


class Matrix:
    def __init__(self, out: Path):
        self.out = out
        self.rows: dict[tuple, dict] = {}
        # Merge rather than overwrite: ONNX Runtime's CPU and CUDA builds cannot be
        # installed together, so the CUDA row comes from a second run.
        if out.exists():
            for row in json.loads(out.read_text())["rows"]:
                self.rows[self._key(row)] = row

    @staticmethod
    def _key(row) -> tuple:
        return tuple(row[k] for k in ("runtime", "device", "precision", "model", "input", "nms"))

    def record(self, runtime, device, precision, model, size, ms=None, error=None, nms="external", **extra):
        row = {
            "runtime": runtime, "device": device, "precision": precision, "model": model,
            "input": size, "nms": nms, "median_ms": ms, "fps": round(1000.0 / ms, 1) if ms else None,
            "error": error, **extra,
        }
        self.rows[self._key(row)] = row
        label = f"{runtime:<12}{device:<16}{precision:<6}{model:<12}{size:>5} {nms:<8}"
        print(f"{label} {'SKIP: ' + error if error else f'{ms:8.2f} ms  {1000 / ms:7.1f} FPS'}", flush=True)
        self.save()

    def save(self):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"environment": environment(), "rows": sorted(self.rows.values(), key=self._key)}
        self.out.write_text(json.dumps(payload, indent=2) + "\n")


# --------------------------------------------------------------------------------
# Legs
# --------------------------------------------------------------------------------
def fresh_model(name: str):
    """A new model per leg: fusing and `.half()` both mutate in place, so a shared
    object would make a later FP32 leg secretly measure FP16."""
    import modern_yolonas
    from modern_yolonas.export import fuse_for_inference

    return fuse_for_inference(getattr(modern_yolonas, name)(pretrained=True))


def with_torch_nms(forward):
    """Wrap a forward pass in the `postprocess` an `external` graph still owes.

    Without this the `external` and `graph` rows are not comparable: the first
    excludes NMS entirely while the second includes it. This is the leg that says
    whether baking NMS into the graph pays.

    Note what this does *not* include: `postprocess` runs on the tensors where they
    already are, so on CUDA legs nothing is copied to the host. A pipeline that does
    move detections host-side pays more than this row shows.
    """
    from modern_yolonas.inference.postprocess import postprocess

    def run():
        boxes, scores = forward()
        postprocess(boxes.float(), scores.float(), conf_threshold=0.25, iou_threshold=0.45, max_detections=300)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    return run


def bench_pytorch(matrix, name, size, cpu: bool):
    x = torch.from_numpy(sample_input(size))

    if cpu:
        model = fresh_model(name)
        with torch.no_grad():
            matrix.record("PyTorch", "CPU", "fp32", name, size, timeit(lambda: model(x)))
        model = None

    if not torch.cuda.is_available():
        return
    for precision in ("fp32", "fp16"):
        model = fresh_model(name).cuda()
        xc = x.cuda()
        if precision == "fp16":
            model, xc = model.half(), xc.half()

        def run():
            with torch.no_grad():
                model(xc)
            torch.cuda.synchronize()

        matrix.record("PyTorch", "dGPU", precision, name, size, timeit(run), **gpu_clock_health())

        def forward():
            with torch.no_grad():
                return model(xc)

        matrix.record("PyTorch", "dGPU", precision, name, size, timeit(with_torch_nms(forward)),
                      nms="torch", **gpu_clock_health())
        model = xc = None
        torch.cuda.empty_cache()


def bench_onnxruntime(matrix, onnx_path, name, size, nms):
    from modern_yolonas.export import onnx_session

    x = sample_input(size)
    for provider, device in (("cpu", "CPU"), ("cuda", "dGPU")):
        try:
            session = onnx_session(onnx_path, provider)
        except Exception as exc:
            matrix.record("ORT", device, "fp32", name, size, error=f"{type(exc).__name__}: {exc}"[:90], nms=nms)
            continue
        import onnxruntime as ort

        matrix.record(
            "ORT", device, "fp32", name, size,
            timeit(lambda: session.run(None, {"images": x})), nms=nms, onnxruntime=ort.__version__,
        )


def bench_openvino(matrix, onnx_path, name, size, devices, calibration_dir, nms):
    import openvino as ov

    from modern_yolonas.export.openvino import export_openvino

    core = ov.Core()
    x = sample_input(size)

    for precision in ("fp32", "fp16", "int8"):
        if precision == "int8" and (calibration_dir is None or nms != "external"):
            reason = "no --calibration-dir" if calibration_dir is None else "NNCF cannot calibrate through graph NMS"
            for _, label in devices:
                matrix.record("OpenVINO", label, precision, name, size, error=reason, nms=nms)
            continue
        try:
            ir = Path(f"build/ov/{name}_{size}_{precision}_{nms}.xml")
            if not ir.exists():
                export_openvino(
                    onnx_path, ir, precision=precision,
                    calibration_dir=calibration_dir, input_size=size,
                )
            model = core.read_model(ir)
        except Exception as exc:
            for _, label in devices:
                matrix.record("OpenVINO", label, precision, name, size,
                              error=f"convert: {type(exc).__name__}: {exc}"[:90], nms=nms)
            continue

        for device, label in devices:
            try:
                compiled = core.compile_model(model, device)
            except Exception as exc:
                matrix.record("OpenVINO", label, precision, name, size,
                              error=f"compile: {type(exc).__name__}: {exc}"[:90], nms=nms)
                continue
            request = compiled.create_infer_request()
            matrix.record("OpenVINO", label, precision, name, size,
                          timeit(lambda: request.infer({0: x})), nms=nms)


def bench_tensorrt(matrix, name, size, nms, half_onnx, fp32_onnx):
    from modern_yolonas.export import EngineRunner
    from modern_yolonas.export.tensorrt import build_engine

    legs = [
        ("fp16", half_onnx, False, "native"),
        # The engine that is actually publishable: AMPERE_PLUS loads on any sm_80+
        # card. Measured beside the native build so the portability has a price tag.
        ("fp16", half_onnx, True, "ampere_plus"),
        ("fp32", fp32_onnx, False, "native"),
    ]
    for precision, onnx_path, portable, variant in legs:
        engine = Path(f"build/trt/{name}_{size}_{precision}_{variant}_{nms}.engine")
        try:
            if not engine.exists():
                build_engine(onnx_path, engine, hardware_compatible=portable)
            runner = EngineRunner(engine)
        except Exception as exc:
            matrix.record("TensorRT", f"dGPU {variant}", precision, name, size,
                          error=f"{type(exc).__name__}: {exc}"[:90], nms=nms)
            continue

        dtype = torch.float16 if precision == "fp16" else torch.float32
        x = torch.from_numpy(sample_input(size)).to("cuda", dtype)
        matrix.record("TensorRT", f"dGPU {variant}", precision, name, size,
                      timeit(lambda: runner(x)), nms=nms, **gpu_clock_health())
        if nms == "external":
            matrix.record("TensorRT", f"dGPU {variant}", precision, name, size,
                          timeit(with_torch_nms(lambda: runner(x))), nms="torch", **gpu_clock_health())
        runner = None
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="yolo_nas_s", help="Comma-separated variants.")
    parser.add_argument("--sizes", default="320,640", help="Comma-separated square input sizes.")
    parser.add_argument("--nms", default="external,graph",
                        help="external (raw tensors), torch (forward + postprocess), graph (--target end2end).")
    parser.add_argument("--calibration-dir", default=None, help="Images for OpenVINO INT8 calibration.")
    parser.add_argument("--runtimes", default="pytorch,ort,openvino,tensorrt")
    parser.add_argument("--skip-cpu", action="store_true", help="Skip the slow PyTorch-on-CPU legs.")
    parser.add_argument("--sample-image", default=None,
                        help="A real frame to time on. Without it the NMS legs see noise, "
                             "find nothing above threshold and report NMS as free.")
    parser.add_argument("--output", default="docs/benchmarks/runtime_matrix.json")
    args = parser.parse_args()

    if args.sample_image:
        import cv2

        from modern_yolonas.inference.preprocess import preprocess

        frame = cv2.imread(args.sample_image)
        if frame is None:
            raise SystemExit(f"could not read {args.sample_image}")
        for size in (int(s) for s in args.sizes.split(",")):
            SAMPLE[size] = preprocess(frame, size)[0].numpy()

    from modern_yolonas.export import export_onnx
    from modern_yolonas.export.nms import make_end2end_onnx

    runtimes = set(args.runtimes.split(","))
    matrix = Matrix(Path(args.output))

    ov_devices = []
    if "openvino" in runtimes:
        try:
            import openvino as ov

            # "GPU.0"/"GPU.1" means nothing in a published table, and on this machine
            # GPU.1 is the NVIDIA card, not a second Intel one.
            core = ov.Core()
            ov_devices = [(d, core.get_property(d, "FULL_DEVICE_NAME")) for d in core.available_devices]
        except Exception as exc:
            print(f"OpenVINO absent: {type(exc).__name__}")

    for name in args.models.split(","):
        for size in (int(s) for s in args.sizes.split(",")):
            for nms in args.nms.split(","):
                print(f"\n== {name} @ {size} (nms={nms})", flush=True)

                base = Path(f"build/onnx/{name}_{size}.onnx")
                half = Path(f"build/onnx/{name}_{size}_fp16.onnx")
                if not base.exists():
                    export_onnx(fresh_model(name), base, input_size=size, dynamic_batch=False)
                if not half.exists():
                    export_onnx(fresh_model(name), half, input_size=size, dynamic_batch=False, half=True)

                if nms == "graph":
                    graph_base = Path(f"build/onnx/{name}_{size}_e2e.onnx")
                    graph_half = Path(f"build/onnx/{name}_{size}_fp16_e2e.onnx")
                    if not graph_base.exists():
                        make_end2end_onnx(str(base), str(graph_base), max_detections=300)
                    if not graph_half.exists():
                        make_end2end_onnx(str(half), str(graph_half), max_detections=300)
                    onnx_fp32, onnx_fp16 = graph_base, graph_half
                else:
                    onnx_fp32, onnx_fp16 = base, half

                # PyTorch has no graph-NMS form; its comparable leg is the external one.
                if "pytorch" in runtimes and nms == "external":
                    bench_pytorch(matrix, name, size, cpu=not args.skip_cpu)
                if "ort" in runtimes:
                    bench_onnxruntime(matrix, onnx_fp32, name, size, nms)
                if "openvino" in runtimes and ov_devices:
                    bench_openvino(matrix, onnx_fp32, name, size, ov_devices, args.calibration_dir, nms)
                if "tensorrt" in runtimes and torch.cuda.is_available():
                    bench_tensorrt(matrix, name, size, nms, onnx_fp16, onnx_fp32)

    matrix.save()
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
