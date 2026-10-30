import logging
from typing import Annotated

import typer

from modern_yolonas.cli.benchmark_cmd import benchmark
from modern_yolonas.cli.benchmark_tracking_cmd import benchmark_tracking_app
from modern_yolonas.cli.dataset_benchmark_cmd import benchmark_dataset_app
from modern_yolonas.cli.demo_cmd import demo
from modern_yolonas.cli.detect_cmd import detect
from modern_yolonas.cli.distill_cmd import distill
from modern_yolonas.cli.domain_distance_cmd import domain_distance
from modern_yolonas.cli.eval_cmd import eval_cmd
from modern_yolonas.cli.export_cmd import export
from modern_yolonas.cli.pretrain_cmd import pretrain
from modern_yolonas.cli.qat_cmd import qat
from modern_yolonas.cli.quantize_cmd import quantize
from modern_yolonas.cli.serve_cmd import serve
from modern_yolonas.cli.track_cmd import track
from modern_yolonas.cli.train_cmd import train

app = typer.Typer(help="YOLO-NAS object detection.", no_args_is_help=True)
app.command()(detect)
app.command()(track)
app.command()(train)
app.command(name="export")(export)
app.command(name="eval")(eval_cmd)
app.command()(serve)
app.command()(benchmark)
app.command()(demo)
app.command()(distill)
app.command()(pretrain)
app.command(name="domain-distance")(domain_distance)
app.command()(quantize)
app.command()(qat)
# `benchmark` measures latency; this one trains and reports mAP, so it gets its own
# name rather than becoming a subcommand of a released command.
app.add_typer(benchmark_dataset_app, name="benchmark-dataset")
# Tracking is measured, not trained, and reports HOTA rather than mAP — so it is a
# third command rather than a subcommand of either existing one.
app.add_typer(benchmark_tracking_app, name="benchmark-tracking")


def _version_callback(value: bool):
    if value:
        from modern_yolonas._version import __version__

        typer.echo(f"modern-yolonas {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable debug logging.")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress all logging output.")
    ] = False,
):
    """YOLO-NAS object detection."""
    level = logging.WARNING
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(name)s: %(message)s")


def main():
    app()
