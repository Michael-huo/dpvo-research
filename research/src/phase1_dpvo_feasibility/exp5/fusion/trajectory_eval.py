"""Final-trajectory artifacts and Sim(3) ATE verification for Exp5-2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


POSE_FORMAT = "tx_ty_tz_qx_qy_qz_qw"
TIMESTAMP_UNIT = "nanoseconds"
ESTIMATE_KEYS = frozenset(
    (
        "schema_version", "kind", "sequence", "method", "timestamp_unit",
        "pose_format", "timestamps", "frame_ids", "ordinals", "poses", "num_poses",
    )
)
GROUND_TRUTH_KEYS = frozenset(
    (
        "schema_version", "kind", "sequence", "method", "timestamp_unit",
        "pose_format", "timestamps", "poses", "num_poses", "source_path",
        "source_sha256",
    )
)
METHOD_ALIASES = {
    "dpvo_baseline": "baseline",
    "random_dense_projection_fusion": "random",
    "jepa_dense_projection_fusion": "learned",
}
SUPPORTED_ESTIMATE_METHODS = frozenset(
    (*METHOD_ALIASES, "oracle_jepa_missing_rgb")
)
SEQUENCE_ROLES = {
    "MH_01_easy": "train_diagnostic",
    "MH_03_medium": "transfer_diagnostic",
    "MH_05_difficult": "decision",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_common(payload: dict[str, Any], *, expected_kind: str) -> tuple[np.ndarray, np.ndarray]:
    expected_keys = ESTIMATE_KEYS if expected_kind == "estimate" else GROUND_TRUTH_KEYS
    if set(payload) != expected_keys:
        raise ValueError(
            f"{expected_kind} trajectory schema mismatch: "
            f"missing={sorted(expected_keys - set(payload))}, "
            f"extra={sorted(set(payload) - expected_keys)}"
        )
    if payload["schema_version"] != 1 or payload["kind"] != expected_kind:
        raise ValueError(f"unsupported {expected_kind} trajectory schema")
    if payload["timestamp_unit"] != TIMESTAMP_UNIT or payload["pose_format"] != POSE_FORMAT:
        raise ValueError("trajectory timestamp/pose contract mismatch")
    count = int(payload["num_poses"])
    timestamps = np.asarray(payload["timestamps"], dtype=np.int64)
    poses = np.asarray(payload["poses"], dtype=np.float64)
    if count <= 0 or timestamps.shape != (count,) or poses.shape != (count, 7):
        raise ValueError("trajectory must contain a non-empty aligned timestamp/pose sequence")
    if not np.isfinite(poses).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("trajectory timestamps must be unique/increasing and poses finite")
    if not str(payload["sequence"]) or not str(payload["method"]):
        raise ValueError("trajectory sequence/method cannot be empty")
    return timestamps, poses


def validate_estimated_trajectory(payload: dict[str, Any]) -> dict[str, Any]:
    timestamps, _ = _validate_common(payload, expected_kind="estimate")
    count = len(timestamps)
    frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64)
    ordinals = np.asarray(payload["ordinals"], dtype=np.int64)
    if frame_ids.shape != (count,) or ordinals.shape != (count,):
        raise ValueError("estimated trajectory identity arrays must match pose count")
    if not np.array_equal(ordinals, np.arange(count, dtype=np.int64)):
        raise ValueError("estimated trajectory ordinals must be contiguous and ordered")
    if len(np.unique(frame_ids)) != count or np.any(np.diff(frame_ids) <= 0):
        raise ValueError("estimated trajectory frame_ids must be unique and increasing")
    if payload["method"] not in SUPPORTED_ESTIMATE_METHODS:
        raise ValueError(f"unsupported Exp5-2 trajectory method: {payload['method']}")
    return payload


def validate_ground_truth_trajectory(payload: dict[str, Any]) -> dict[str, Any]:
    _validate_common(payload, expected_kind="ground_truth")
    if payload["method"] != "ground_truth":
        raise ValueError("ground-truth trajectory method must be ground_truth")
    source = Path(payload["source_path"])
    if not source.is_absolute() or len(str(payload["source_sha256"])) != 64:
        raise ValueError("ground-truth trajectory provenance is invalid")
    return payload


def build_estimated_trajectory(
    *, sequence: str, method: str, records: list[dict[str, Any]],
    ordinals: np.ndarray, poses: np.ndarray,
) -> dict[str, Any]:
    ordinal_array = np.asarray(ordinals, dtype=np.float64)
    pose_array = np.asarray(poses, dtype=np.float64)
    if not np.isfinite(ordinal_array).all() or not np.equal(
        ordinal_array, np.floor(ordinal_array)
    ).all():
        raise ValueError("DPVO terminate ordinals must be finite integers")
    ordinal_int = ordinal_array.astype(np.int64)
    if not np.array_equal(ordinal_int, np.arange(len(records), dtype=np.int64)):
        raise ValueError("DPVO terminate ordinals do not match stream identity")
    ordered_records = [records[int(value)] for value in ordinal_int]
    payload = {
        "schema_version": 1,
        "kind": "estimate",
        "sequence": str(sequence),
        "method": str(method),
        "timestamp_unit": TIMESTAMP_UNIT,
        "pose_format": POSE_FORMAT,
        "timestamps": [int(record["timestamp"]) for record in ordered_records],
        "frame_ids": [int(record["frame_id"]) for record in ordered_records],
        "ordinals": ordinal_int.tolist(),
        "poses": pose_array.tolist(),
        "num_poses": int(len(pose_array)),
    }
    return validate_estimated_trajectory(payload)


def build_ground_truth_subset(
    *, sequence: str, source_path: str | Path, timestamp_min: int, timestamp_max: int,
) -> dict[str, Any]:
    from evo.tools import file_interface

    source = Path(source_path).resolve()
    reference = file_interface.read_tum_trajectory_file(str(source))
    source_timestamps = np.rint(reference.timestamps).astype(np.int64)
    keep = (source_timestamps >= int(timestamp_min)) & (source_timestamps <= int(timestamp_max))
    if not bool(keep.any()):
        raise ValueError(f"ground truth has no samples in estimate range: {sequence}")
    positions = reference.positions_xyz[keep]
    quaternions_wxyz = reference.orientations_quat_wxyz[keep]
    poses = np.concatenate(
        (positions, quaternions_wxyz[:, [1, 2, 3, 0]]), axis=1
    ).astype(np.float64)
    payload = {
        "schema_version": 1,
        "kind": "ground_truth",
        "sequence": str(sequence),
        "method": "ground_truth",
        "timestamp_unit": TIMESTAMP_UNIT,
        "pose_format": POSE_FORMAT,
        "timestamps": source_timestamps[keep].tolist(),
        "poses": poses.tolist(),
        "num_poses": int(len(poses)),
        "source_path": str(source),
        "source_sha256": _sha256(source),
    }
    return validate_ground_truth_trajectory(payload)


def write_trajectory(path: str | Path, payload: dict[str, Any]) -> None:
    if payload.get("kind") == "estimate":
        validate_estimated_trajectory(payload)
    elif payload.get("kind") == "ground_truth":
        validate_ground_truth_trajectory(payload)
    else:
        raise ValueError("unsupported trajectory kind")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_trajectory(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") == "estimate":
        return validate_estimated_trajectory(payload)
    if payload.get("kind") == "ground_truth":
        return validate_ground_truth_trajectory(payload)
    raise ValueError("unsupported trajectory artifact kind")


def to_evo_trajectory(payload: dict[str, Any]) -> Any:
    from evo.core.trajectory import PoseTrajectory3D

    timestamps, poses = _validate_common(payload, expected_kind=str(payload["kind"]))
    return PoseTrajectory3D(
        positions_xyz=poses[:, :3],
        orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
        timestamps=timestamps.astype(np.float64),
    )


def evaluate_trajectory(
    estimate_payload: dict[str, Any], ground_truth_payload: dict[str, Any],
) -> dict[str, Any]:
    import evo.main_ape as main_ape
    from evo.core import sync
    from evo.core.metrics import PoseRelation

    validate_estimated_trajectory(estimate_payload)
    validate_ground_truth_trajectory(ground_truth_payload)
    if estimate_payload["sequence"] != ground_truth_payload["sequence"]:
        raise ValueError("estimate and ground truth sequences differ")
    reference, estimate = sync.associate_trajectories(
        to_evo_trajectory(ground_truth_payload), to_evo_trajectory(estimate_payload)
    )
    if estimate.num_poses == 0:
        raise ValueError("trajectory timestamp association is empty")
    result = main_ape.ape(
        reference,
        estimate,
        est_name=str(estimate_payload["method"]),
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )
    return {
        "method": str(estimate_payload["method"]),
        "ate_rmse": float(result.stats["rmse"]),
        "associated_count": int(estimate.num_poses),
        "alignment": "Sim(3)",
    }


def aligned_positions_for_plot(
    estimate_payload: dict[str, Any], ground_truth_payload: dict[str, Any],
) -> np.ndarray:
    from evo.core import sync
    from evo.core.geometry import umeyama_alignment

    validate_estimated_trajectory(estimate_payload)
    validate_ground_truth_trajectory(ground_truth_payload)
    reference, estimate = sync.associate_trajectories(
        to_evo_trajectory(ground_truth_payload), to_evo_trajectory(estimate_payload)
    )
    if estimate.num_poses == 0:
        raise ValueError("trajectory timestamp association is empty")
    rotation, translation, scale = umeyama_alignment(
        estimate.positions_xyz.T, reference.positions_xyz.T, with_scale=True
    )
    raw_positions = np.asarray(estimate_payload["poses"], dtype=np.float64)[:, :3]
    return (scale * (rotation @ raw_positions.T) + translation[:, None]).T
