import logging
from typing import Annotated

import typer

from modern_yolonas.cli.benchmark_cmd import benchmark
from modern_yolonas.cli.demo_cmd import demo
from modern_yolonas.cli.detect_cmd import detect
from modern_yolonas.cli.eval_cmd import eval_cmd
from modern_yolonas.cli.export_cmd import export
from modern_yolonas.cli.serve_cmd import serve
from modern_yolonas.cli.train_cmd import train

app = typer.Typer(help="YOLO-NAS object detection.", no_args_is_help=True)
app.command()(detect)
app.command()(train)
app.command(name="export")(export)
app.command(name="eval")(eval_cmd)
app.command()(serve)
app.command()(benchmark)
app.command()(demo)


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
