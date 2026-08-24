"""Fixed-protocol EuRoC preparation and aligned JEPA/FMap loading."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .schema import (
    REPO_ROOT,
    Exp4Sample,
    atomic_write_bytes,
    atomic_write_json,
    encode_jsonl,
    load_config,
    read_jsonl,
    repo_path,
    validate_metadata_identity,
)


@dataclass(frozen=True)
class DiscoveredSequence:
    dataset_type: str
    dataset_group: str
    sequence: str
    sequence_category: str
    sequence_root: Path


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _sequence_category(sequence: str) -> str:
    category = sequence.rsplit("_", 1)[-1]
    if category not in {"easy", "medium", "difficult"}:
        raise ValueError(f"cannot derive EuRoC sequence category from {sequence!r}")
    return category


def discover_sequences(config: dict[str, Any]) -> dict[str, DiscoveredSequence]:
    dataset = config["dataset"]
    root = repo_path(dataset["root"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    allowed_groups = set(dataset["groups"])
    discovered: dict[str, DiscoveredSequence] = {}
    for data_csv in sorted(root.rglob("mav0/cam0/data.csv")):
        sequence_root = data_csv.parents[2]
        group, sequence = sequence_root.parent.name, sequence_root.name
        if group not in allowed_groups:
            continue
        if sequence in discovered:
            raise ValueError(f"duplicate discovered sequence: {sequence}")
        discovered[sequence] = DiscoveredSequence(
            dataset_type=str(dataset["dataset_type"]),
            dataset_group=group,
            sequence=sequence,
            sequence_category=_sequence_category(sequence),
            sequence_root=sequence_root.resolve(),
        )
    if not discovered:
        raise RuntimeError("dataset discovery found no supported EuRoC sequences")
    return discovered


def _camera_rows(data_csv: Path, image_dir: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with data_csv.open(encoding="utf-8") as handle:
        for row in csv.reader(line for line in handle if not line.startswith("#")):
            if not row:
                continue
            timestamp, filename = int(row[0]), row[1]
            if filename != f"{timestamp}.png":
                raise ValueError(f"timestamp/filename mismatch in {data_csv}: {row}")
            if not (image_dir / filename).is_file():
                raise FileNotFoundError(image_dir / filename)
            rows.append((timestamp, filename))
    timestamps = [timestamp for timestamp, _ in rows]
    if not rows or any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"timestamps must be non-empty, unique, and strictly increasing: {data_csv}")
    return rows


def _split_mapping(config: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for sequence in config["dataset"]["splits"][split]:
            if sequence in mapping:
                raise ValueError(f"sequence {sequence!r} occurs in more than one split")
            mapping[str(sequence)] = split
    if not all(config["dataset"]["splits"].get(split) for split in ("train", "val", "test")):
        raise ValueError("train/val/test must each contain at least one sequence")
    return mapping


def prepare_dataset(
    dataset_root: str | Path,
    *,
    config_path: str | Path,
    smoke: bool,
) -> dict[str, Any]:
    """Build the immutable fixed-protocol index without copying images."""
    config, resolved_config, config_sha256 = load_config(config_path)
    discovered = discover_sequences(config)
    split_map = _split_mapping(config)
    missing = set(split_map) - set(discovered)
    if missing:
        raise FileNotFoundError(f"configured EuRoC sequences were not discovered: {sorted(missing)}")
    stride = int(config["dataset"]["stride"])
    skip = int(config["dataset"]["skip"])
    smoke_cap = int(config["dataset"]["smoke_frames_per_sequence"])
    if stride <= 0 or skip < 0 or smoke_cap <= 0:
        raise ValueError("stride/smoke cap must be positive and skip non-negative")

    root = Path(dataset_root).resolve()
    jepa_shape = tuple(int(value) for value in config["jepa"]["expected_shape"])
    fmap_shape = tuple(int(value) for value in config["dpvo_fmap"]["reference_shape"])
    by_split: dict[str, list[Exp4Sample]] = {name: [] for name in ("train", "val", "test")}
    sequence_counts: dict[str, dict[str, Any]] = {}
    camera_name = str(config["dataset"]["camera"])

    for sequence, split in split_map.items():
        item = discovered[sequence]
        camera_root = item.sequence_root / "mav0" / camera_name
        rows = _camera_rows(camera_root / "data.csv", camera_root / "data")
        selected = list(enumerate(rows))[skip::stride]
        if smoke:
            selected = selected[:smoke_cap]
        for frame_id, (timestamp_ns, filename) in selected:
            prefix = f"{item.dataset_type}/{item.dataset_group}/{sequence}/{timestamp_ns}.pt"
            by_split[split].append(Exp4Sample(
                dataset_type=item.dataset_type,
                dataset_group=item.dataset_group,
                sequence_category=item.sequence_category,
                sequence=sequence,
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                image_path=str((camera_root / "data" / filename).resolve()),
                split=split,
                jepa_feature_path=f"features/jepa/{prefix}",
                dpvo_feature_path=f"features/dpvo_fmap/{prefix}",
                jepa_shape=jepa_shape,
                fmap_shape=fmap_shape,
            ))
        sequence_counts[sequence] = {
            "dataset_type": item.dataset_type,
            "dataset_group": item.dataset_group,
            "sequence_category": item.sequence_category,
            "source_root": str(item.sequence_root),
            "split": split,
            "raw_cam0_frames": len(rows),
            "indexed_frames": len(selected),
        }

    indices: dict[str, dict[str, Any]] = {}
    for split, samples in by_split.items():
        payload = encode_jsonl(samples)
        atomic_write_bytes(root / f"{split}.jsonl", payload)
        indices[split] = {
            "path": f"{split}.jsonl",
            "frames": len(samples),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "schema_version": int(config["schema_version"]),
        "experiment": config["experiment"]["name"],
        "repository_commit": _git_commit(REPO_ROOT),
        "config_path": str(resolved_config),
        "config_sha256": config_sha256,
        "scope": "smoke" if smoke else "formal",
        "protocol": {
            "dataset_type": config["dataset"]["dataset_type"],
            "camera": camera_name,
            "splits": config["dataset"]["splits"],
            "stride": stride,
            "skip": skip,
            "frame_id_definition": "zero-based cam0 data.csv row before stride",
            "images_copied": False,
        },
        "sequences": sequence_counts,
        "indices": indices,
        "total_indexed_frames": sum(len(samples) for samples in by_split.values()),
        "sample_schema": {
            "identity": ["dataset_type", "dataset_group", "sequence", "frame_id", "timestamp_ns"],
            "jepa_shape": list(jepa_shape),
            "fmap_shape_reference": list(fmap_shape),
        },
    }
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


def dataset_fingerprint(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    names = ("manifest.json", "train.jsonl", "val.jsonl", "test.jsonl")
    combined = hashlib.sha256()
    files: dict[str, str] = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        files[name] = digest
        combined.update(name.encode())
        combined.update(b"\0")
        combined.update(payload)
    return {"sha256": combined.hexdigest(), "files": files}


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(f"Cannot load feature file {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"Malformed feature payload: {path}")
    return payload


def _check_tensor(
    *, sample: Exp4Sample, payload: dict[str, Any], path: Path,
    kind: str, tensor_key: str, expected_shape: tuple[int, ...], config_sha256: str,
) -> torch.Tensor:
    metadata = payload["metadata"]
    validate_metadata_identity(sample, metadata, str(path))
    if metadata.get("feature_kind") != kind:
        raise ValueError(f"{path}: expected feature_kind={kind!r}, got {metadata.get('feature_kind')!r}")
    if metadata.get("config_sha256") != config_sha256:
        raise ValueError(f"{path}: config fingerprint differs from dataset manifest")
    tensor = payload.get(tensor_key)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{path}: missing tensor {tensor_key!r}")
    if tuple(tensor.shape) != expected_shape or list(tensor.shape) != metadata.get("shape"):
        raise ValueError(f"{path}: tensor shape does not match index/metadata")
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
        raise ValueError(f"{path}: feature must be finite floating point")
    return tensor


class Exp4Dataset(Dataset[dict[str, Any]]):
    """Load one aligned split without augmentation."""

    def __init__(
        self,
        index_path: str | Path,
        dataset_root: str | Path | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.index_path = Path(index_path).resolve()
        self.dataset_root = Path(dataset_root).resolve() if dataset_root else self.index_path.parent
        self.samples = read_jsonl(self.index_path)
        if not self.samples:
            raise ValueError(f"Exp4 index is empty: {self.index_path}")
        if not isinstance(output_dtype, torch.dtype) or not output_dtype.is_floating_point:
            raise TypeError("output_dtype must be a floating-point torch dtype")
        self.output_dtype = output_dtype
        manifest_path = self.dataset_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.config_sha256 = str(self.manifest["config_sha256"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        jepa_path = sample.feature_path(self.dataset_root, "jepa")
        fmap_path = sample.feature_path(self.dataset_root, "dpvo_fmap")
        jepa_payload = _load_payload(jepa_path)
        fmap_payload = _load_payload(fmap_path)
        tokens = _check_tensor(
            sample=sample, payload=jepa_payload, path=jepa_path,
            kind="jepa_tokens", tensor_key="tokens",
            expected_shape=sample.jepa_shape, config_sha256=self.config_sha256,
        )
        fmap = _check_tensor(
            sample=sample, payload=fmap_payload, path=fmap_path,
            kind="dpvo_fmap", tensor_key="fmap",
            expected_shape=sample.fmap_shape, config_sha256=self.config_sha256,
        )
        return {
            "jepa_tokens": tokens.to(dtype=self.output_dtype),
            "fmap_teacher": fmap.to(dtype=self.output_dtype),
            "metadata": {
                "sample": sample.to_dict(),
                "jepa": jepa_payload["metadata"],
                "dpvo_fmap": fmap_payload["metadata"],
            },
        }


class AdapterDataset(Dataset[dict[str, Any]]):
    """Thin wrapper retaining the adapter-facing dataset contract."""

    def __init__(self, index_path: str | Path, *, dataset_root: str | Path | None = None) -> None:
        self.base = Exp4Dataset(index_path, dataset_root=dataset_root)
        self.preflight()

    @property
    def dataset_root(self) -> Path:
        return self.base.dataset_root

    @property
    def samples(self) -> list[Exp4Sample]:
        return self.base.samples

    def preflight(self) -> None:
        missing: list[str] = []
        for sample in self.base.samples:
            for kind in ("jepa", "dpvo_fmap"):
                path = sample.feature_path(self.base.dataset_root, kind)
                if not path.is_file():
                    missing.append(str(path))
        if missing:
            preview = "\n".join(missing[:10])
            suffix = f"\n... and {len(missing) - 10} more" if len(missing) > 10 else ""
            raise FileNotFoundError(f"Missing {len(missing)} Exp4 feature files:\n{preview}{suffix}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.base[index]
