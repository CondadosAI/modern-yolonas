"""CLI: yolonas benchmark-tracking (cache / evaluate).

Measures the tracker on MOT-format datasets. Two commands, because the detector pass
is minutes and the tracker pass is seconds, and an ablation should not pay for the
first one every time::

    yolonas benchmark-tracking cache --data ~/datasets/sportsmot/extracted/val
    yolonas benchmark-tracking evaluate --data ~/datasets/sportsmot/extracted/val
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

benchmark_tracking_app = typer.Typer(
    help="Measure the tracker on a MOT-format dataset (SportsMOT, MOT17).",
    no_args_is_help=True,
)


class ModelName(str, Enum):
    yolo_nas_s = "yolo_nas_s"
    yolo_nas_m = "yolo_nas_m"
    yolo_nas_l = "yolo_nas_l"


class Source(str, Enum):
    detector = "detector"
    oracle = "oracle"


@benchmark_tracking_app.command()
def cache(
    data: Annotated[str, typer.Option(help="Split directory, holding one subdirectory per sequence.")],
    output: Annotated[str, typer.Option(help="Where to write the per-sequence caches.")] = "runs/mot-cache",
    model: Annotated[ModelName, typer.Option(help="Model variant.")] = ModelName.yolo_nas_l,
    source: Annotated[Source, typer.Option(help="'detector' runs detection; 'oracle' embeds the ground-truth boxes, leaving association as the only source of error.")] = Source.detector,
    device: Annotated[str, typer.Option(help="Device (cuda or cpu).")] = "cuda",
    conf: Annotated[float, typer.Option(help="Detector threshold. Low on purpose — the tracker filters on replay, so score sweeps need no re-detection.")] = 0.1,
    input_size: Annotated[int, typer.Option(help="Model input size.")] = 640,
    sequences: Annotated[str | None, typer.Option(help="Split file listing sequence names, or a comma-separated list. Default: every sequence found.")] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N sequences, for a smoke run.")] = None,
    overwrite: Annotated[bool, typer.Option(help="Rebuild caches that already exist.")] = False,
):
    """Detect (or read) and embed every frame, once, to disk."""
    import time

    from rich.console import Console
    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

    from modern_yolonas.benchmarks.mot import (
        build_detector_cache,
        build_oracle_cache,
    )

    console = Console()
    picked = _sequences(data, sequences, limit)
    out_dir = Path(output) / source.value
    out_dir.mkdir(parents=True, exist_ok=True)

    total_frames = sum(s.length for s in picked)
    console.print(
        f"{len(picked)} sequences, {total_frames} frames — {source.value} source, {model.value}"
    )

    if source == Source.oracle:
        from modern_yolonas import YoloNASEmbedder

        engine = YoloNASEmbedder(model.value, device=device, input_size=input_size)
        build = build_oracle_cache
    else:
        from modern_yolonas import YoloNASDetector

        engine = YoloNASDetector(model.value, device=device, input_size=input_size)

        def build(sequence, detector):
            return build_detector_cache(sequence, detector, conf_threshold=conf)

    started = time.perf_counter()
    done_frames = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("caching", total=len(picked))
        for sequence in picked:
            path = out_dir / f"{sequence.name}.npz"
            if path.exists() and not overwrite:
                progress.advance(task)
                continue
            progress.update(task, description=sequence.name[:28])
            build(sequence, engine).save(path)
            done_frames += sequence.length
            progress.advance(task)

    elapsed = time.perf_counter() - started
    rate = done_frames / elapsed if elapsed > 0 else 0.0
    console.print(f"[green]Cached to {out_dir}[/green] — {done_frames} frames in {elapsed:.0f}s ({rate:.1f} fps)")


@benchmark_tracking_app.command()
def evaluate(
    data: Annotated[str, typer.Option(help="Split directory, the same one passed to `cache`.")],
    cache_dir: Annotated[str, typer.Option("--cache", help="Where `cache` wrote its output.")] = "runs/mot-cache",
    source: Annotated[Source, typer.Option(help="Which cache to replay.")] = Source.detector,
    output: Annotated[str, typer.Option(help="Where to write the results table and TrackEval tree.")] = "runs/mot-eval",
    benchmark: Annotated[str, typer.Option(help="TrackEval benchmark name; also names the output folder.")] = "sportsmot",
    split: Annotated[str, typer.Option(help="TrackEval split name.")] = "val",
    preproc: Annotated[bool, typer.Option(help="TrackEval's MOT17 preprocessing. Needed for MOT17, a no-op on SportsMOT.")] = False,
    sequences: Annotated[str | None, typer.Option(help="Split file or comma-separated names. Default: every cached sequence.")] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N sequences.")] = None,
    configs: Annotated[str | None, typer.Option(help="Comma-separated configuration names to run. Default: all of them.")] = None,
):
    """Replay every tracker configuration over the cache and score it."""
    import json

    from rich.console import Console
    from rich.table import Table

    from modern_yolonas.benchmarks.mot import (
        DetectionCache,
        SWEEP,
        evaluate as score,
        prepare_trackeval_layout,
        replay,
    )
    from modern_yolonas.tracking import DeepHMSort

    console = Console()
    picked = _sequences(data, sequences, limit)
    cache_root = Path(cache_dir) / source.value

    missing = [s.name for s in picked if not (cache_root / f"{s.name}.npz").exists()]
    if missing:
        raise typer.BadParameter(
            f"{len(missing)} sequence(s) have no cache under {cache_root} "
            f"(first: {missing[0]}). Run `yolonas benchmark-tracking cache` first."
        )

    chosen = list(SWEEP) if configs is None else [c.strip() for c in configs.split(",") if c.strip()]
    unknown = [name for name in chosen if name not in SWEEP]
    if unknown:
        raise typer.BadParameter(f"unknown configuration(s) {unknown}; choose from {list(SWEEP)}")

    console.print(f"Loading {len(picked)} caches from {cache_root}...")
    caches = {s.name: DetectionCache.load(cache_root / f"{s.name}.npz") for s in picked}

    work_dir = Path(output) / f"{benchmark}-{split}-{source.value}"
    rows = {}

    for name in chosen:
        settings = dict(SWEEP[name])
        use_embeddings = settings.pop("use_embeddings", True)
        tracker = DeepHMSort(**settings)

        results = {s.name: replay(caches[s.name], tracker, use_embeddings) for s in picked}
        prepare_trackeval_layout(picked, results, work_dir, benchmark, split)
        rows[name] = score(work_dir, benchmark, split, do_preproc=preproc)
        console.print(f"  {name}: HOTA {rows[name]['HOTA']:.2f}")

    table = Table(title=f"{benchmark}-{split} — {source.value} detections, {len(picked)} sequences")
    table.add_column("configuration")
    for column in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"):
        table.add_column(column, justify="right")
    for name, metrics in rows.items():
        table.add_row(name, *[f"{metrics[c]:.2f}" if c != "IDSW" else f"{metrics[c]:.0f}" for c in
                              ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW")])
    console.print(table)

    summary = work_dir / "results.json"
    summary.write_text(json.dumps({"sequences": [s.name for s in picked], "results": rows}, indent=2))
    console.print(f"[green]Wrote {summary}[/green]")


def _sequences(data: str, sequences: str | None, limit: int | None):
    """Resolve the sequence list from a split file, a comma-separated list, or the directory."""
    from modern_yolonas.benchmarks.mot import discover_sequences, read_split_file

    names = None
    if sequences is not None:
        path = Path(sequences)
        names = read_split_file(path) if path.is_file() else [n.strip() for n in sequences.split(",") if n.strip()]

    picked = discover_sequences(data, names)
    return picked[:limit] if limit else picked
