"""Shared Phase 1 identities, RGB boundary, EuRoC IO, and deterministic schedule."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
RATIO_SCHEDULE_METHOD = "exp6_post_bootstrap_ratio"
SUPPORTED_SEQUENCES = ("MH_01_easy", "MH_03_medium", "MH_05_difficult")


class FrameRole(str, Enum):
    ANCHOR = "anchor"
    HIDDEN = "hidden"


class RepresentationOrigin(str, Enum):
    ORACLE = "oracle"
    PREDICTED = "predicted"


class RGBConsumer(str, Enum):
    OFFLINE_ORACLE_EXTRACTOR = "offline_oracle_extractor"
    ONLINE_RUNTIME = "online_runtime"


def canonical_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_bytes(path, (json.dumps(
        payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False,
    ) + "\n").encode("utf-8"))


@dataclass(frozen=True)
class FrameIdentity:
    dataset: str
    dataset_group: str
    split: str
    sequence: str
    frame_id: int
    candidate_index: int
    timestamp_ns: int

    def __post_init__(self) -> None:
        if min(self.frame_id, self.candidate_index, self.timestamp_ns) < 0:
            raise ValueError("frame identity integer fields must be non-negative")
        if not all((self.dataset, self.dataset_group, self.split, self.sequence)):
            raise ValueError("frame identity string fields must be non-empty")

    @property
    def key(self) -> str:
        return (f"{self.dataset}/{self.dataset_group}/{self.sequence}/"
                f"{self.frame_id}/{self.timestamp_ns}")

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrivateFrameRecord:
    identity: FrameIdentity
    rgb_path: str
class HiddenRGBAccessError(PermissionError):
    pass


class RGBAccessGate:
    @staticmethod
    def require(role: FrameRole, consumer: RGBConsumer, rgb_path: str | None) -> str:
        if role is FrameRole.HIDDEN and consumer is RGBConsumer.ONLINE_RUNTIME:
            raise HiddenRGBAccessError("hidden online RGB access is forbidden")
        if not rgb_path:
            raise HiddenRGBAccessError("RGB access requested without a private path")
        return rgb_path




def post_bootstrap_ratio_roles(
    identities: Sequence[FrameIdentity], *, bootstrap_end_candidate_index: int,
    anchor_ratio: float = 0.2,
) -> dict[str, str]:
    """Build the deterministic canonical ratio schedule."""
    ratio = float(anchor_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("anchor_ratio must be in (0, 1]")
    if bootstrap_end_candidate_index < 0:
        raise ValueError("bootstrap_end_candidate_index must be non-negative")
    result: dict[str, str] = {}
    # Emit the first post-bootstrap candidate, then maintain the requested density.
    accumulator = 1.0 - ratio
    for identity in identities:
        if identity.candidate_index <= bootstrap_end_candidate_index:
            result[identity.key] = FrameRole.ANCHOR.value
            continue
        accumulator += ratio
        if accumulator >= 1.0 - 1e-12:
            result[identity.key] = FrameRole.ANCHOR.value
            accumulator -= 1.0
        else:
            result[identity.key] = FrameRole.HIDDEN.value
    return result


def ratio_schedule_payload(
    identities: Sequence[FrameIdentity], *, bootstrap_end_candidate_index: int,
    anchor_ratio: float,
) -> dict[str, Any]:
    roles = post_bootstrap_ratio_roles(
        identities,
        bootstrap_end_candidate_index=bootstrap_end_candidate_index,
        anchor_ratio=anchor_ratio,
    )
    anchor_keys = [item.key for item in identities if roles[item.key] == FrameRole.ANCHOR.value]
    hidden_keys = [item.key for item in identities if roles[item.key] == FrameRole.HIDDEN.value]
    payload = {
        "schedule_method": RATIO_SCHEDULE_METHOD,
        "requested_post_bootstrap_anchor_ratio": float(anchor_ratio),
        "actual_full_sequence_anchor_ratio": len(anchor_keys) / len(identities),
        "bootstrap_end_candidate_index": int(bootstrap_end_candidate_index),
        "candidate_count": len(identities),
        "anchor_count": len(anchor_keys),
        "hidden_count": len(hidden_keys),
        "anchor_identity_sha256": canonical_sha256(anchor_keys),
        "hidden_identity_sha256": canonical_sha256(hidden_keys),
    }
    payload["schedule_sha256"] = canonical_sha256(payload)
    return payload




def load_sequence_records(config: Mapping[str, Any], sequence: str, *,
                          limit: int | None = None) -> list[PrivateFrameRecord]:
    if sequence not in SUPPORTED_SEQUENCES:
        raise ValueError(f"unsupported EuRoC sequence: {sequence}")
    dataset = config["dataset"]
    root = repo_path(dataset.get("root", "research/assets/datasets/euroc"))
    groups = dataset.get("groups", ("machine_hall", "vicon_room1", "vicon_room2"))
    camera_name = str(dataset.get("camera", "cam0"))
    roots = [root / str(group) / sequence for group in groups
             if (root / str(group) / sequence / "mav0" / camera_name / "data.csv").is_file()]
    if len(roots) != 1:
        raise FileNotFoundError(f"expected one EuRoC sequence root for {sequence}")
    camera = roots[0] / "mav0" / camera_name
    rows: list[tuple[int, str]] = []
    with (camera / "data.csv").open(encoding="utf-8") as handle:
        for row in csv.reader(line for line in handle if not line.startswith("#")):
            if not row:
                continue
            timestamp, filename = int(row[0]), row[1]
            if filename != f"{timestamp}.png" or not (camera / "data" / filename).is_file():
                raise ValueError(f"invalid EuRoC frame identity: {row}")
            rows.append((timestamp, filename))
    if not rows or any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"non-monotonic EuRoC timestamps: {sequence}")
    selected = list(enumerate(rows))[int(dataset.get("skip", 0))::int(dataset.get("stride", 2))]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    return [PrivateFrameRecord(
        FrameIdentity("euroc", roots[0].parent.name, "sequence", sequence,
                      frame_id, candidate, timestamp),
        str((camera / "data" / filename).resolve()),
    ) for candidate, (frame_id, (timestamp, filename)) in enumerate(selected)]
