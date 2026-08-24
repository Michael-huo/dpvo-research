"""Shared immutable sample schema and filesystem helpers for Exp4."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

import yaml


Split = Literal["train", "val", "test"]
VALID_SPLITS = frozenset(("train", "val", "test"))
VALID_CATEGORIES = frozenset(("easy", "medium", "difficult"))
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml"
REPO_ROOT = PACKAGE_DIR.parents[3]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).resolve()
    data = resolved.read_bytes()
    config = yaml.safe_load(data)
    if not isinstance(config, dict):
        raise TypeError(f"Exp4 config must be a mapping: {resolved}")
    return config, resolved, hashlib.sha256(data).hexdigest()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    data = (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    atomic_write_bytes(path, data)


@dataclass(frozen=True)
class Exp4Sample:
    dataset_type: str
    dataset_group: str
    sequence_category: str
    sequence: str
    frame_id: int
    timestamp_ns: int
    image_path: str
    split: Split
    jepa_feature_path: str
    dpvo_feature_path: str
    jepa_shape: tuple[int, int]
    fmap_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        for name, value in (("dataset_type", self.dataset_type), ("dataset_group", self.dataset_group), ("sequence", self.sequence)):
            if not value or SAFE_NAME.fullmatch(value) is None:
                raise ValueError(f"{name} must be a safe non-empty name")
        if self.sequence_category not in VALID_CATEGORIES:
            raise ValueError(f"invalid sequence_category: {self.sequence_category}")
        if not self.sequence.endswith(f"_{self.sequence_category}"):
            raise ValueError("sequence_category must match the sequence name suffix")
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be positive")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"invalid split: {self.split}")
        image = Path(self.image_path)
        if not image.is_absolute():
            raise ValueError("image_path must be absolute")
        if image.stem != str(self.timestamp_ns):
            raise ValueError("image filename stem must equal timestamp_ns")
        for name, value in (("jepa_feature_path", self.jepa_feature_path), ("dpvo_feature_path", self.dpvo_feature_path)):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{name} must be a safe dataset-relative path")
            kind = "jepa" if name == "jepa_feature_path" else "dpvo_fmap"
            expected_prefix = ("features", kind, self.dataset_type, self.dataset_group, self.sequence)
            expected_parts = (*expected_prefix, f"{self.timestamp_ns}.pt")
            if path.parts != expected_parts:
                raise ValueError(f"{name} does not match sample dataset identity")
        if len(self.jepa_shape) != 2 or any(int(value) <= 0 for value in self.jepa_shape):
            raise ValueError(f"invalid jepa_shape: {self.jepa_shape}")
        if len(self.fmap_shape) != 3 or any(int(value) <= 0 for value in self.fmap_shape):
            raise ValueError(f"invalid fmap_shape: {self.fmap_shape}")

    @property
    def identity(self) -> tuple[str, str, str, int, int]:
        return self.dataset_type, self.dataset_group, self.sequence, self.frame_id, self.timestamp_ns

    def feature_path(self, dataset_root: str | Path, kind: Literal["jepa", "dpvo_fmap"]) -> Path:
        relative = self.jepa_feature_path if kind == "jepa" else self.dpvo_feature_path
        return Path(dataset_root).resolve() / relative

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["jepa_shape"] = list(self.jepa_shape)
        payload["fmap_shape"] = list(self.fmap_shape)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Exp4Sample":
        expected = set(cls.__dataclass_fields__)
        missing, extra = expected - set(payload), set(payload) - expected
        if missing or extra:
            raise ValueError(f"sample schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        values = dict(payload)
        values["jepa_shape"] = tuple(int(value) for value in values["jepa_shape"])
        values["fmap_shape"] = tuple(int(value) for value in values["fmap_shape"])
        values["frame_id"] = int(values["frame_id"])
        values["timestamp_ns"] = int(values["timestamp_ns"])
        return cls(**values)


def encode_jsonl(samples: Iterable[Exp4Sample]) -> bytes:
    return b"".join((json.dumps(sample.to_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode() for sample in samples)


def read_jsonl(path: str | Path) -> list[Exp4Sample]:
    rows: list[Exp4Sample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(Exp4Sample.from_dict(json.loads(line)))
                except Exception as error:
                    raise ValueError(f"Invalid Exp4 sample at {path}:{line_number}: {error}") from error
    identities = [sample.identity for sample in rows]
    if len(identities) != len(set(identities)):
        raise ValueError(f"Duplicate sample identity in {path}")
    return rows


def iter_index_paths(dataset_root: str | Path, splits: str | Sequence[str]) -> Iterator[Path]:
    root = Path(dataset_root)
    requested = tuple(VALID_SPLITS) if splits == "all" else ((splits,) if isinstance(splits, str) else tuple(splits))
    invalid = set(requested) - VALID_SPLITS
    if invalid:
        raise ValueError(f"invalid split selection: {sorted(invalid)}")
    for name in ("train", "val", "test"):
        if name in requested:
            yield root / f"{name}.jsonl"


def select_samples(dataset_root: str | Path, splits: str | Sequence[str], limit: int | None = None) -> list[Exp4Sample]:
    rows: list[Exp4Sample] = []
    for path in iter_index_paths(dataset_root, splits):
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(read_jsonl(path))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise RuntimeError("selection contains no samples")
    return rows


def identity_metadata(sample: Exp4Sample) -> dict[str, Any]:
    return {
        "dataset_type": sample.dataset_type,
        "dataset_group": sample.dataset_group,
        "sequence_category": sample.sequence_category,
        "sequence": sample.sequence,
        "frame_id": sample.frame_id,
        "timestamp_ns": sample.timestamp_ns,
        "image_path": sample.image_path,
        "split": sample.split,
    }


def validate_metadata_identity(sample: Exp4Sample, metadata: dict[str, Any], source: str) -> None:
    observed = (
        metadata.get("dataset_type"), metadata.get("dataset_group"), metadata.get("sequence"),
        metadata.get("frame_id"), metadata.get("timestamp_ns"),
    )
    if observed != sample.identity:
        raise ValueError(f"{source} identity mismatch: index={sample.identity}, feature={observed}")
    if metadata.get("sequence_category") != sample.sequence_category:
        raise ValueError(f"{source} sequence_category mismatch for {sample.identity}")
    if metadata.get("image_path") != sample.image_path:
        raise ValueError(f"{source} image_path mismatch for {sample.identity}")
