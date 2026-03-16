"""YAML config file loading for CLI commands."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return as a flat dict.

    Args:
        path: Path to YAML config file.

    Returns:
        Dict of option names to values. Keys use underscores (e.g. ``input_size``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    # Flatten: replace hyphens with underscores to match CLI option names
    return {k.replace("-", "_"): v for k, v in data.items()}


def merge_config(config: dict, cli_kwargs: dict) -> dict:
    """Merge config file values with CLI arguments.

    CLI arguments take precedence over config file values.
    Only non-None CLI values override config values.

    Args:
        config: Values loaded from YAML config file.
        cli_kwargs: CLI argument dict (may contain None for unset values).

    Returns:
        Merged dict with CLI values overriding config values.
    """
    merged = dict(config)
    for key, value in cli_kwargs.items():
        if value is not None:
            merged[key] = value
    return merged
