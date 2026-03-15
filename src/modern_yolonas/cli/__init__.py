from typing import Annotated

import typer

from modern_yolonas.cli.detect_cmd import detect
from modern_yolonas.cli.train_cmd import train
from modern_yolonas.cli.export_cmd import export
from modern_yolonas.cli.eval_cmd import eval_cmd

app = typer.Typer(help="YOLO-NAS object detection.", no_args_is_help=True)
app.command()(detect)
app.command()(train)
app.command(name="export")(export)
app.command(name="eval")(eval_cmd)


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
):
    """YOLO-NAS object detection."""


def main():
    app()
