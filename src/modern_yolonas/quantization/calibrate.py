"""PTQ calibration — forward passes to collect activation statistics."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from rich.console import Console

console = Console()


@torch.no_grad()
def run_calibration(
    model: nn.Module,
    dataloader: DataLoader,
    num_batches: int = 100,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Run calibration for PTQ.

    Observer nodes collect activation statistics (min/max, histogram)
    during forward passes over representative data.

    Args:
        model: :class:`QuantizedYoloNAS` from :func:`prepare_model_ptq`.
        dataloader: Calibration dataset loader.
        num_batches: Batches to calibrate on (100-200 is typical).
        device: Device to run calibration on.

    Returns:
        The same model with calibrated observer statistics.
    """
    model = model.to(device)
    model.eval()

    console.print(f"[bold]Running PTQ calibration ({num_batches} batches)...[/bold]")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images = images.to(device, non_blocking=True)
        model(images)

        if (batch_idx + 1) % 20 == 0:
            console.print(f"  Calibrated {batch_idx + 1}/{num_batches} batches")

    console.print("[green]Calibration complete.[/green]")
    return model
