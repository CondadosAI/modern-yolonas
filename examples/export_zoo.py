"""Pre-export every pretrained variant, at every size, to every runtime.

Produces the artifact set published at
https://huggingface.co/CondadosAI/modern-yolonas-export, plus a `manifest.json`
recording what each file is, what produced it and what it will load on.

    uv run examples/export_zoo.py --out build/zoo \
        --calibration-dir ~/datasets/coco/images/val2017

TensorRT engines are the exception to "pre-export and publish". An engine is
compiled for one GPU architecture and one TensorRT version, so the only kind worth
publishing is a hardware-compatible (AMPERE_PLUS) one, which loads on any sm_80+
card and is measurably slower than a native build. Both facts go in the manifest;
`--no-tensorrt` skips them entirely, and the ONNX is always the portable artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, root: Path, **fields) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        **fields,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="yolo_nas_s,yolo_nas_m,yolo_nas_l")
    parser.add_argument("--sizes", default="320,640")
    parser.add_argument("--out", default="build/zoo")
    parser.add_argument("--calibration-dir", default=None, help="Images for OpenVINO INT8. Skipped without it.")
    parser.add_argument("--no-tensorrt", action="store_true")
    parser.add_argument("--output", default=None, help="Manifest path (default: <out>/manifest.json).")
    args = parser.parse_args()

    import modern_yolonas
    from modern_yolonas.export import export_onnx, fuse_for_inference
    from modern_yolonas.export.nms import make_end2end_onnx

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    entries = []

    def fresh(name):
        # Both fusing and .half() mutate in place, so every artifact starts clean.
        return fuse_for_inference(getattr(modern_yolonas, name)(pretrained=True))

    for name in args.models.split(","):
        for size in (int(s) for s in args.sizes.split(",")):
            stem = f"{name}_{size}"
            print(f"\n== {stem}", flush=True)

            onnx_path = export_onnx(fresh(name), root / f"{stem}.onnx", input_size=size, dynamic_batch=True)
            entries.append(artifact(onnx_path, root, model=name, input_size=size, format="onnx",
                                    precision="fp32", nms="external", batch="dynamic"))
            print(f"   onnx {onnx_path.stat().st_size / 1e6:.0f} MB")

            static = export_onnx(fresh(name), root / f"{stem}_static.onnx", input_size=size, dynamic_batch=False)
            e2e = root / f"{stem}_end2end.onnx"
            make_end2end_onnx(str(static), str(e2e), conf_threshold=0.25, iou_threshold=0.45, max_detections=300)
            static.unlink()
            entries.append(artifact(e2e, root, model=name, input_size=size, format="onnx",
                                    precision="fp32", nms="graph", batch="1",
                                    note="NMS baked in; returns detections [D, 7] as "
                                         "[batch, x1, y1, x2, y2, conf, class] in letterboxed pixels"))
            print("   onnx end2end")

            half = export_onnx(fresh(name), root / f"{stem}_fp16.onnx", input_size=size,
                               dynamic_batch=False, half=True)
            entries.append(artifact(half, root, model=name, input_size=size, format="onnx",
                                    precision="fp16", nms="external", batch="1",
                                    note="the input for a TensorRT 11 FP16 engine, which is strongly "
                                         "typed and takes its precision from the graph"))

            from modern_yolonas.export.openvino import export_openvino

            for precision in ("fp16", "int8"):
                if precision == "int8" and args.calibration_dir is None:
                    print("   openvino int8 skipped (no --calibration-dir)")
                    continue
                xml = export_openvino(onnx_path, root / f"{stem}_{precision}.xml", precision=precision,
                                      calibration_dir=args.calibration_dir, input_size=size)
                for path in (xml, xml.with_suffix(".bin")):
                    entries.append(artifact(path, root, model=name, input_size=size, format="openvino",
                                            precision=precision, nms="external", batch="dynamic"))
                print(f"   openvino {precision}")

            if not args.no_tensorrt:
                from modern_yolonas.export.tensorrt import build_engine, engine_metadata

                engine = root / f"{stem}_fp16_ampere_plus.engine"
                build_engine(half, engine, hardware_compatible=True)
                entries.append(artifact(engine, root, model=name, input_size=size, format="tensorrt",
                                        precision="fp16", nms="external", batch="1",
                                        built_on=engine_metadata(),
                                        note="AMPERE_PLUS: loads on any sm_80+ GPU, but only under the "
                                             "exact TensorRT build recorded in built_on — the engine is "
                                             "not version-compatible. It is also slower than an engine "
                                             "built natively on the target card, so rebuilding from the "
                                             "fp16 ONNX beside it is the better option whenever possible."))
                print("   tensorrt fp16 (AMPERE_PLUS)")

    manifest = Path(args.output) if args.output else root / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": entries}, indent=2) + "\n")
    total = sum(e["bytes"] for e in entries)
    print(f"\nwrote {manifest}: {len(entries)} artifacts, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
