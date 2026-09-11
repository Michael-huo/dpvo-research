"""Small artifact and trajectory helpers for Phase 1 feasibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .evaluation import associated_groundtruth_poses, fit_sim3_trajectory
from .protocol import SUPPORTED_SEQUENCES


def load_yaml(path: str | Path, *, schema_version: int = 1) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != schema_version:
        raise ValueError(f"invalid Phase 1 config schema: {resolved}")
    return payload, resolved


def resolve_sequences(values: Sequence[str] | None,
                      default: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(values or default)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("--sequences must be non-empty and unique")
    unsupported = sorted(set(requested) - set(SUPPORTED_SEQUENCES))
    if unsupported:
        raise ValueError(f"unsupported sequences: {unsupported}")
    return requested


def trajectory_payload(
    records: Sequence[Any], roles: Mapping[str, str],
    arrays: Mapping[str, Mapping[str, np.ndarray]], population: Any,
    groundtruth: Path,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "input__frame_identity_keys": np.asarray([row.identity.key for row in records]),
        "input__candidate_indices": np.asarray([row.identity.candidate_index for row in records], np.int64),
        "input__timestamps_ns": np.asarray([row.identity.timestamp_ns for row in records], np.uint64),
        "input__roles": np.asarray([roles[row.identity.key] for row in records]),
        "canonical__timestamps_ns": np.asarray(population.common_anchor_timestamps_ns, np.uint64),
        "canonical__groundtruth_poses": associated_groundtruth_poses(
            groundtruth, population.common_anchor_timestamps_ns,
        ),
        "canonical__population_sha256": np.asarray(population.population_sha256),
        "canonical__rpe_pairs_ns": np.asarray(population.rpe_pairs_ns, np.uint64),
    }
    for condition, values in arrays.items():
        fitted = fit_sim3_trajectory(values, population, groundtruth)
        prefix = f"{condition}__"
        payload[prefix + "timestamps_ns"] = np.asarray(values["timestamps_ns"], np.uint64)
        payload[prefix + "poses_raw"] = np.asarray(values["poses"], np.float64)
        payload[prefix + "poses_sim3_aligned"] = np.asarray(fitted["aligned_poses"], np.float64)
        payload[prefix + "sim3_scale"] = np.asarray(fitted["scale"], np.float64)
        payload[prefix + "sim3_rotation"] = np.asarray(fitted["rotation"], np.float64)
        payload[prefix + "sim3_translation"] = np.asarray(fitted["translation"], np.float64)
        payload[prefix + "sim3_fitting_population_sha256"] = np.asarray(
            fitted["fitting_population_sha256"],
        )
    return payload


def validate_trajectory_payload(path: Path, conditions: Sequence[str]) -> None:
    with np.load(path, allow_pickle=False) as bundle:
        canonical_hash = str(bundle["canonical__population_sha256"].item())
        for condition in conditions:
            prefix = f"{condition}__"
            raw = np.asarray(bundle[prefix + "poses_raw"], dtype=np.float64)
            aligned = np.asarray(bundle[prefix + "poses_sim3_aligned"], dtype=np.float64)
            scale = float(bundle[prefix + "sim3_scale"].item())
            rotation = np.asarray(bundle[prefix + "sim3_rotation"], dtype=np.float64)
            translation = np.asarray(bundle[prefix + "sim3_translation"], dtype=np.float64)
            reconstructed = (scale * (rotation @ raw[:, :3].T) + translation[:, None]).T
            if not np.allclose(reconstructed, aligned[:, :3], atol=1e-10):
                raise RuntimeError(f"saved Sim3 does not reconstruct {condition}")
            reconstructed_q = Rotation.from_matrix(
                rotation[None] @ Rotation.from_quat(raw[:, 3:7]).as_matrix()
            ).as_quat()
            if not np.allclose(np.abs(np.sum(reconstructed_q * aligned[:, 3:7], axis=1)),
                               1.0, atol=1e-10):
                raise RuntimeError(f"saved rotation does not reconstruct {condition}")
            if str(bundle[prefix + "sim3_fitting_population_sha256"].item()) != canonical_hash:
                raise RuntimeError(f"population hash mismatch for {condition}")
