# YOLO-NAS-S latency, three sessions

Three runs of `examples/runtime_matrix.py` for YOLO-NAS-S, back to back on 23 Sep 2026,
to measure how much a latency figure moves between sessions on the same machine. One
session's median of 30 runs says nothing about that.

```bash
uv run examples/runtime_matrix.py --models yolo_nas_s --sizes 320,640 \
    --calibration-dir ~/datasets/coco/images/val2017 \
    --sample-image ~/datasets/coco/images/val2017/000000435081.jpg \
    --output docs/benchmarks/sessions-2026-09-23/session1.json   # then session2, session3
```

**Conditions.** Code at `f505310`, whose benchmark path is unchanged from `dbd6619`, the
commit that produced `../runtime_matrix.json`. Same library versions as that file: torch
2.10.0+cu130, OpenVINO 2026.3.1, TensorRT 11.3.0.99, ONNX Runtime 1.24.3, NVIDIA driver
580.173.02. i7-12700H, Iris Xe, RTX 3060 Laptop GPU, on mains, performance power profile.
The machine was not idle: the load average was between 2 and 5.

**The three sessions share one set of exported files.** The first session built the ONNX
graphs, the OpenVINO IR (including INT8 calibration) and the TensorRT engines. The second
and third reused them from `build/`. So the spread between sessions is timing drift on
identical files, not variation between exports.

## Range across the sessions, YOLO-NAS-S, model only (`nms=external`), ms

| cell | session 1 | session 2 | session 3 | median |
|---|---:|---:|---:|---:|
| OpenVINO INT8, CPU, 320 | 4.77 | 4.98 | 5.58 | 4.98 |
| OpenVINO INT8, CPU, 640 | 19.46 | 20.34 | 20.65 | 20.34 |
| OpenVINO INT8, iGPU, 320 | 6.76 | 6.98 | 5.93 | 6.76 |
| OpenVINO INT8, iGPU, 640 | 12.62 | 12.48 | 12.55 | 12.55 |
| OpenVINO FP32, CPU, 640 | 62.18 | 65.63 | 76.49 | 65.63 |
| PyTorch FP32, CPU, 320 | 31.74 | 44.70 | 38.41 | 38.41 |
| TensorRT FP16, dGPU, 320 | 1.17 | 0.93 | 0.93 | 0.93 |
| TensorRT FP16, dGPU, 640 | 2.25 | 2.18 | 2.14 | 2.18 |

The same TensorRT engine file timed at 1.17 ms in one session and 0.93 ms in the next, a
20% swing with nothing about the model changing. The CPU legs moved most with the load on
the machine. Treat a difference between two cells smaller than these ranges as noise.
