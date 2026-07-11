"""
Config loader — reads default.yaml or custom path.
"""
from pathlib import Path
from typing import Any, Dict

import yaml


_DEFAULT_CFG = Path(__file__).resolve().parent / "default.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML config, falling back to default.yaml."""
    path = Path(path) if path else _DEFAULT_CFG
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
