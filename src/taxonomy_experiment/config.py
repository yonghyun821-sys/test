from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)(?::([^}]*))?\}$")
_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_project_dotenv(root: Path) -> None:
    """Load simple KEY=VALUE entries without overriding the caller's environment."""
    path = root / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not _ENV_KEY_PATTERN.fullmatch(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value
    match = _ENV_PATTERN.match(value)
    if not match:
        return value
    name, default = match.groups()
    return os.getenv(name, default)


def load_config(path: str | Path = "config/experiment.yaml") -> dict[str, Any]:
    root = project_root()
    _load_project_dotenv(root)
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = _expand_env(yaml.safe_load(handle))
    config["_root"] = root
    config["_config_path"] = config_path
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    path = Path(config["paths"][key])
    return path if path.is_absolute() else config["_root"] / path


def experiment_datasets(config: dict[str, Any]) -> tuple[str, str]:
    """Return the two ordered datasets participating in the configured experiment."""
    datasets = (
        str(config["experiment"]["dataset_a"]),
        str(config["experiment"]["dataset_b"]),
    )
    if len(set(datasets)) != 2:
        raise ValueError("dataset_a and dataset_b must be different")
    return datasets
