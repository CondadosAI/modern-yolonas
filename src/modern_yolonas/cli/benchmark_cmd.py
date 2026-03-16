"""CLI: yolonas benchmark"""

from __future__ import annotations

import json as json_module
import time
from enum import Enum
from typing import Annotated

import typer


class ModelName(str, Enum):
    yolo_nas_s = "yolo_nas_s"
    yolo_nas_m = "yolo_nas_m"
    yolo_nas_l = "yolo_nas_l"


class Precision(str, Enum):
    fp32 = "fp32"
    fp16 = "fp16"


def benchmark(
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_s,
    device: Annotated[str, typer.Option(help="Device.")] = "cuda",
    input_size: Annotated[int, typer.Option(help="Model input size.")] = 640,
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 1,
    iterations: Annotated[int, typer.Option(help="Number of inference iterations.")] = 100,
    warmup: Annotated[int, typer.Option(help="Warmup iterations.")] = 10,
    precision: Annotated[Precision, typer.Option(help="Inference precision.")] = Precision.fp32,
    json: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
):
    """Benchmark model inference latency and throughput."""
    import torch
    from rich.console import Console
    from rich.table import Table

    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l

    console = Console()

    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    yolo_model = builders[model.value](pretrained=False).to(device).eval()

    # Fuse RepVGG branches
    for m in yolo_model.modules():
        if hasattr(m, "fuse_block_residual_branches"):
            m.fuse_block_residual_branches()

    use_fp16 = precision == Precision.fp16
    if use_fp16:
        yolo_model = yolo_model.half()

    dtype = torch.float16 if use_fp16 else torch.float32
    dummy = torch.randn(batch_size, 3, input_size, input_size, device=device, dtype=dtype)

    # Count parameters
    param_count = sum(p.numel() for p in yolo_model.parameters())

    # Warmup
    if not json:
        console.print(f"Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            yolo_model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            yolo_model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies.sort()
    mean_ms = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    throughput = batch_size * 1000 / mean_ms  # img/s

    peak_memory_mb = 0.0
    if device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    results = {
        "model": model.value,
        "device": device,
        "precision": precision.value,
        "input_size": input_size,
        "batch_size": batch_size,
        "iterations": iterations,
        "params_millions": round(param_count / 1e6, 2),
        "latency_mean_ms": round(mean_ms, 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "throughput_img_per_sec": round(throughput, 1),
        "peak_gpu_memory_mb": round(peak_memory_mb, 1),
    }

    if json:
        typer.echo(json_module.dumps(results, indent=2))
    else:
        table = Table(title=f"Benchmark: {model.value}")
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Parameters", f"{results['params_millions']}M")
        table.add_row("Precision", results["precision"])
        table.add_row("Batch size", str(batch_size))
        table.add_row("Input size", f"{input_size}x{input_size}")
        table.add_row("Latency (mean)", f"{results['latency_mean_ms']:.2f} ms")
        table.add_row("Latency (p50)", f"{results['latency_p50_ms']:.2f} ms")
        table.add_row("Latency (p95)", f"{results['latency_p95_ms']:.2f} ms")
        table.add_row("Latency (p99)", f"{results['latency_p99_ms']:.2f} ms")
        table.add_row("Throughput", f"{results['throughput_img_per_sec']:.1f} img/s")
        if peak_memory_mb > 0:
            table.add_row("Peak GPU memory", f"{results['peak_gpu_memory_mb']:.1f} MB")
        console.print(table)
