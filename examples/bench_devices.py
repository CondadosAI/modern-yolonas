"""YOLO-NAS-S latency across runtime x device x precision.

Model inference only (no preprocessing, no NMS), single stream, synchronous, median of
N timed runs after warmup. Latency, not throughput: an async/multi-stream pipeline will
report higher FPS for the same hardware.

Legs that fail are recorded rather than silently dropped.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch

RUNS, WARM = 30, 8
rows = []


def record(runtime, device, precision, size, ms=None, error=None):
    rows.append({"runtime": runtime, "device": device, "precision": precision,
                 "input": size, "median_ms": ms, "fps": (1000.0 / ms) if ms else None,
                 "error": error})
    tag = f"{runtime:<14}{device:<18}{precision:<6}{size:>5}"
    print(f"{tag} {'FAIL: ' + error if error else f'{ms:8.2f} ms  {1000/ms:6.1f} FPS'}", flush=True)


def timeit(fn):
    for _ in range(WARM):
        fn()
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def calib_batches(video, size, n=48):
    """Real frames, preprocessed exactly as inference does, for PTQ calibration."""
    from modern_yolonas.inference.preprocess import preprocess
    cap = cv2.VideoCapture(video)
    out, i = [], 0
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        if i % 17 == 0:
            t, _, _ = preprocess(f, size)
            out.append(t.numpy().astype(np.float32))
        i += 1
    cap.release()
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="256,640")
ap.add_argument("--calib-video", default="worker-zone-detection.mp4")
ap.add_argument("--out", default="matrix.json")
args = ap.parse_args()

for size in [int(s) for s in args.sizes.split(",")]:
    x = np.random.randn(1, 3, size, size).astype(np.float32)
    xt = torch.from_numpy(x)
    calib = calib_batches(args.calib_video, size)

    # ---------------- PyTorch ----------------
    from modern_yolonas import yolo_nas_s
    m = yolo_nas_s(pretrained=True).eval()
    with torch.no_grad():
        record("PyTorch", "CPU i7-12700H", "FP32", size, timeit(lambda: m(xt)))

    if torch.cuda.is_available():
        mc, xc = m.cuda(), xt.cuda()
        def run_fp32():
            with torch.no_grad():
                mc(xc)
            torch.cuda.synchronize()
        record("PyTorch", "dGPU RTX 3060", "FP32", size, timeit(run_fp32))

        mh, xh = mc.half(), xc.half()
        def run_fp16():
            with torch.no_grad():
                mh(xh)
            torch.cuda.synchronize()
        record("PyTorch", "dGPU RTX 3060", "FP16", size, timeit(run_fp16))
        del mc, mh
        torch.cuda.empty_cache()

    # ---------------- OpenVINO ----------------
    import openvino as ov
    core = ov.Core()
    base = core.read_model(f"ov_{size}.xml")

    variants = {"FP32": base}
    try:
        import openvino.runtime.passes as _  # noqa: F401
    except Exception:
        pass
    try:
        fp16 = core.read_model(f"ov_{size}.xml")
        ov.save_model(fp16, f"ov_{size}_fp16.xml", compress_to_fp16=True)
        variants["FP16"] = core.read_model(f"ov_{size}_fp16.xml")
    except Exception as e:
        record("OpenVINO", "-", "FP16", size, error=f"convert: {type(e).__name__}")

    try:
        import nncf
        ds = nncf.Dataset(calib, lambda b: {0: b})
        variants["INT8"] = nncf.quantize(base, ds, subset_size=len(calib))
    except Exception as e:
        record("OpenVINO", "-", "INT8", size, error=f"quantize: {type(e).__name__}: {str(e)[:60]}")

    for prec, model in variants.items():
        for dev, label in (("CPU", "CPU i7-12700H"), ("GPU.0", "iGPU Iris Xe")):
            try:
                req = core.compile_model(model, dev).create_infer_request()
                record("OpenVINO", label, prec, size, timeit(lambda: req.infer({0: x})))
            except Exception as e:
                record("OpenVINO", label, prec, size, error=f"{type(e).__name__}: {str(e)[:60]}")

    # ---------------- TensorRT (TRT 11: precision comes from the ONNX) --------
    try:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.ERROR)
        for prec, path in (("FP32", f"onnx_{size}.onnx"), ("FP16", f"onnx_{size}_fp16.onnx")):
            try:
                if prec == "FP16":
                    import onnx
                    from onnxconverter_common import float16
                    mo = onnx.load(f"onnx_{size}.onnx")
                    onnx.save(float16.convert_float_to_float16(mo, keep_io_types=False), path,
                              save_as_external_data=True, all_tensors_to_one_file=True,
                              location=f"onnx_{size}_fp16.onnx.data")
                builder = trt.Builder(logger)
                net = builder.create_network()   # explicit batch is the only mode in TRT 11
                parser = trt.OnnxParser(net, logger)
                if not parser.parse_from_file(path):
                    raise RuntimeError(parser.get_error(0).desc()[:70])
                cfg = builder.create_builder_config()
                cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
                prof = builder.create_optimization_profile()
                shp = (1, 3, size, size)
                prof.set_shape(net.get_input(0).name, shp, shp, shp)
                cfg.add_optimization_profile(prof)
                plan = builder.build_serialized_network(net, cfg)
                if plan is None:
                    raise RuntimeError("build returned None")
                eng = trt.Runtime(logger).deserialize_cuda_engine(plan)
                ctx = eng.create_execution_context()
                names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
                ctx.set_input_shape(names[0], shp)
                bufs = {}
                for n in names:
                    td = torch.float16 if eng.get_tensor_dtype(n) == trt.DataType.HALF else torch.float32
                    t = torch.empty(tuple(ctx.get_tensor_shape(n)), dtype=td, device="cuda")
                    bufs[n] = t
                    ctx.set_tensor_address(n, int(t.data_ptr()))
                bufs[names[0]].copy_(torch.from_numpy(x).cuda().to(bufs[names[0]].dtype))
                stream = torch.cuda.Stream()

                def run_trt():
                    ctx.execute_async_v3(stream.cuda_stream)
                    stream.synchronize()

                record("TensorRT", "dGPU", prec, size, timeit(run_trt))
                del ctx, eng
            except Exception as e:
                record("TensorRT", "dGPU", prec, size, error=f"{type(e).__name__}: {str(e)[:70]}")
        record("TensorRT", "dGPU", "INT8", size,
               error="not measured: TRT 11 removed the INT8 calibrator API; needs a QDQ ONNX")
    except Exception as e:
        record("TensorRT", "dGPU", "-", size, error=f"import: {type(e).__name__}")

json.dump(rows, open(args.out, "w"), indent=2)
print("wrote " + args.out)
