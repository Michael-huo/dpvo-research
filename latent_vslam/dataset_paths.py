"""Resolve machine-local dataset locations without adding paths to science config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


LOCAL_PATHS_CONFIG = Path(__file__).resolve().parents[1] / "configs/paths.local.yaml"
DATASET_KEYS = ("euroc", "euroc_groundtruth")


def load_dataset_paths(path: str | Path = LOCAL_PATHS_CONFIG) -> dict[str, Path]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"machine-local dataset paths are missing: {config_path}; "
            "copy configs/paths.example.yaml to configs/paths.local.yaml "
            "and set absolute dataset directories"
        )
    payload: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    if not isinstance(datasets, dict):
        raise ValueError(f"invalid machine-local dataset paths in {config_path}: expected datasets mapping")
    resolved: dict[str, Path] = {}
    for key in DATASET_KEYS:
        raw = datasets.get(key)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"missing datasets.{key} in {config_path}")
        location = Path(raw).expanduser()
        if not location.is_absolute():
            raise ValueError(f"datasets.{key} must be an absolute path in {config_path}: {raw}")
        location = location.resolve()
        if not location.is_dir():
            raise FileNotFoundError(f"datasets.{key} directory does not exist: {location}")
        resolved[key] = location
    return resolved


def dataset_root(key: str, *, config_path: str | Path = LOCAL_PATHS_CONFIG) -> Path:
    if key not in DATASET_KEYS:
        raise ValueError(f"unknown dataset path key: {key}")
    return load_dataset_paths(config_path)[key]


def groundtruth_path(sequence: str, *, config_path: str | Path = LOCAL_PATHS_CONFIG) -> Path:
    path = dataset_root("euroc_groundtruth", config_path=config_path) / f"{sequence}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"EuRoC groundtruth is missing for {sequence}: {path}")
    return path
