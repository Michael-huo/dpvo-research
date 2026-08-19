"""Prepare the fixed formal 4000-sample contract for Experiment 2."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "research/configs/phase1_exp2.yaml"
PROFILE_NAME = "formal"

REQUIRED_OBSERVATION_KEYS = {
    "factor_uid", "source_patch_uid", "source_dpvo_counter",
    "target_dpvo_counter", "update_dpvo_counter", "weight_mean",
    "delta_norm", "apparent_motion", "corr_peak_l0", "corr_margin_l0",
    "corr_entropy_l0",
}
REQUIRED_PATCH_KEYS = {"patch_uid", "x", "y"}
REQUIRED_FRAME_KEYS = {
    "dpvo_counter", "euroc_timestamp_ns", "texture_gradient",
    "difficulty_texture_quintile", "difficulty_model_motion_quintile",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _schema(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in arrays.items()
    }


def _require_keys(name: str, arrays: dict[str, np.ndarray], required: set[str]) -> None:
    missing = required - set(arrays)
    if missing:
        raise KeyError(f"{name} is missing required fields: {sorted(missing)}")


def crop_metadata(width: int, height: int, crop_size: int, short_side_scale: float) -> dict[str, Any]:
    short_side = int(crop_size * short_side_scale)
    if width < height:
        resized_width = short_side
        resized_height = int(round(height * short_side / width))
    else:
        resized_height = short_side
        resized_width = int(round(width * short_side / height))
    resized_width = max(resized_width, crop_size)
    resized_height = max(resized_height, crop_size)
    left = max(0, (resized_width - crop_size) // 2)
    top = max(0, (resized_height - crop_size) // 2)
    return {
        "original_size_hw": [height, width],
        "resized_size_hw": [resized_height, resized_width],
        "crop_box_xyxy": [left, top, left + crop_size, top + crop_size],
        "crop_size": crop_size,
    }


def camera_matrix(calibration: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = calibration[:4]
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def undistorted_to_raw(points_xy: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    matrix = camera_matrix(calibration)
    normalized = np.column_stack(
        ((points[:, 0] - matrix[0, 2]) / matrix[0, 0],
         (points[:, 1] - matrix[1, 2]) / matrix[1, 1],
         np.ones(len(points), dtype=np.float64))
    )
    distortion = np.asarray(calibration[4:], dtype=np.float64)
    projected, _ = cv2.projectPoints(
        normalized, np.zeros(3), np.zeros(3), matrix, distortion
    )
    return projected.reshape(-1, 2)


def raw_to_token(points_xy: np.ndarray, metadata: dict[str, Any], patch_size: int) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    original_h, original_w = metadata["original_size_hw"]
    resized_h, resized_w = metadata["resized_size_hw"]
    left, top, _, _ = metadata["crop_box_xyxy"]
    crop_x = (points[:, 0] + 0.5) * resized_w / original_w - 0.5 - left
    crop_y = (points[:, 1] + 0.5) * resized_h / original_h - 0.5 - top
    return np.column_stack(((crop_x + 0.5) / patch_size - 0.5,
                            (crop_y + 0.5) / patch_size - 0.5))


def _groundtruth_range(path: Path) -> tuple[int, int]:
    timestamps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            timestamps.append(int(round(float(line.split()[0]))))
    if not timestamps:
        raise RuntimeError(f"No ground-truth rows in {path}")
    return min(timestamps), max(timestamps)


def _join_patches(observations: dict[str, np.ndarray], patches: dict[str, np.ndarray]) -> np.ndarray:
    order = np.argsort(patches["patch_uid"])
    sorted_uid = patches["patch_uid"][order]
    positions = np.searchsorted(sorted_uid, observations["source_patch_uid"])
    valid = positions < len(sorted_uid)
    valid &= sorted_uid[np.minimum(positions, len(sorted_uid) - 1)] == observations["source_patch_uid"]
    if not valid.all():
        raise AssertionError(f"Unresolved source_patch_uid rows: {(~valid).sum()}")
    return order[positions]


def _select_groups(
    candidates: dict[str, np.ndarray], source_frames: np.ndarray, *,
    count: int, cap: int, rng: np.random.Generator,
) -> dict[str, list[int]]:
    queues = {name: list(rng.permutation(indices).astype(np.int64)) for name, indices in candidates.items()}
    positions = {name: 0 for name in queues}
    selected: dict[str, list[int]] = {"good": [], "bad": []}
    frame_counts: Counter[int] = Counter()
    while any(len(selected[name]) < count for name in ("good", "bad")):
        progressed = False
        for name in ("good", "bad"):
            if len(selected[name]) >= count:
                continue
            queue = queues[name]
            while positions[name] < len(queue):
                index = int(queue[positions[name]])
                positions[name] += 1
                frame = int(source_frames[index])
                if frame_counts[frame] >= cap:
                    continue
                selected[name].append(index)
                frame_counts[frame] += 1
                progressed = True
                break
        if not progressed:
            break
    if max(frame_counts.values(), default=0) > cap:
        raise AssertionError("Per-source-frame cap violated")
    return selected


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: None for key in ("mean", "median", "q25", "q75", "min", "max")} | {"count": 0}
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _sample_to_row(
    *, sequence: str, group: str, rank: int, observation_index: int,
    observations: dict[str, np.ndarray], patches: dict[str, np.ndarray],
    frames: dict[str, np.ndarray], patch_index: int, raw_xy: np.ndarray,
    token_xy: np.ndarray, euroc_root: Path,
) -> dict[str, Any]:
    source_frame = int(observations["source_dpvo_counter"][observation_index])
    target_frame = int(observations["target_dpvo_counter"][observation_index])
    source_ts = int(frames["euroc_timestamp_ns"][source_frame])
    target_ts = int(frames["euroc_timestamp_ns"][target_frame])
    factor_uid = int(observations["factor_uid"][observation_index])
    update = int(observations["update_dpvo_counter"][observation_index])
    image_dir = euroc_root / sequence / "mav0/cam0/data"
    return {
        "sample_id": f"{sequence}__{group}__f{factor_uid}__u{update}",
        "sequence": sequence,
        "group": group,
        "group_description": "MH_01 Q20 frozen-threshold bad group" if group == "bad" else "MH_01 Q80 frozen-threshold good group",
        "sample_rank": rank,
        "exp1_observation_index": observation_index,
        "factor_uid": factor_uid,
        "update_dpvo_counter": update,
        "source_frame_id": source_frame,
        "target_frame_id": target_frame,
        "source_timestamp_ns": source_ts,
        "target_timestamp_ns": target_ts,
        "source_timestamp": source_ts * 1e-9,
        "target_timestamp": target_ts * 1e-9,
        "delta_t": (target_ts - source_ts) * 1e-9,
        "source_image_path": str(image_dir / f"{source_ts}.png"),
        "target_image_path": str(image_dir / f"{target_ts}.png"),
        "patch_uid": int(patches["patch_uid"][patch_index]),
        "source_xy": [float(patches["x"][patch_index]), float(patches["y"][patch_index])],
        "source_undistorted_xy": [float(patches["x"][patch_index] * 4.0), float(patches["y"][patch_index] * 4.0)],
        "source_raw_xy": [float(raw_xy[0]), float(raw_xy[1])],
        "source_token_xy": [float(token_xy[0]), float(token_xy[1])],
        "source_image_hw": [480, 752],
        "dpvo_coordinate_system": "undistorted 1/4-resolution feature grid; (x,y); full-resolution pixel=(4x,4y)",
        "dpvo_corr_peak": float(observations["corr_peak_l0"][observation_index]),
        "dpvo_corr_margin": float(observations["corr_margin_l0"][observation_index]),
        "dpvo_corr_entropy": float(observations["corr_entropy_l0"][observation_index]),
        "dpvo_weight": float(observations["weight_mean"][observation_index]),
        "dpvo_delta_norm": float(observations["delta_norm"][observation_index]),
        "texture_gradient": float(frames["texture_gradient"][source_frame]),
        "texture_quintile": int(frames["difficulty_texture_quintile"][source_frame]),
        "low_texture": bool(frames["difficulty_texture_quintile"][source_frame] == 5),
        "apparent_motion": float(observations["apparent_motion"][observation_index]),
        "motion_quintile": int(frames["difficulty_model_motion_quintile"][source_frame]),
        "large_motion": bool(frames["difficulty_model_motion_quintile"][source_frame] == 5),
        "has_shuffled_control": False,
        "shuffled_target_frame_id": -1,
        "shuffled_target_timestamp_ns": -1,
        "shuffled_target_image_path": "",
    }


def _rows_to_npz(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = rows[0].keys()
    payload: dict[str, np.ndarray] = {}
    for key in keys:
        values = [row[key] for row in rows]
        if key in {"source_xy", "source_undistorted_xy", "source_raw_xy", "source_token_xy", "source_image_hw"}:
            payload[key] = np.asarray(values)
        elif isinstance(values[0], bool):
            payload[key] = np.asarray(values, dtype=bool)
        elif isinstance(values[0], str):
            payload[key] = np.asarray(values, dtype=np.str_)
        elif isinstance(values[0], int):
            payload[key] = np.asarray(values, dtype=np.int64)
        else:
            payload[key] = np.asarray(values, dtype=np.float64)
    return payload


def main() -> int:
    config_bytes = CONFIG_PATH.read_bytes()
    config = yaml.safe_load(config_bytes)
    profile = config["sampling_profiles"][PROFILE_NAME]
    paths = config["paths"]
    exp1_root = REPO_ROOT / paths["exp1_root"]
    euroc_root = REPO_ROOT / paths["euroc_root"]
    output_root = REPO_ROOT / paths["output_root"] / PROFILE_NAME
    artifacts_dir = output_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    calibration = np.loadtxt(REPO_ROOT / paths["euroc_calibration"], delimiter=" ")
    coord = config["coordinates"]
    raw_h, raw_w = map(int, coord["raw_image_hw"])
    metadata = crop_metadata(raw_w, raw_h, int(coord["vjepa_crop_size"]), float(coord["vjepa_short_side_scale"]))
    expected_metadata = {
        "original_size_hw": [480, 752], "resized_size_hw": [438, 686],
        "crop_box_xyxy": [151, 27, 535, 411], "crop_size": 384,
    }
    if metadata != expected_metadata:
        raise AssertionError(f"Unexpected frozen V-JEPA crop metadata: {metadata}")

    loaded: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    schemas: dict[str, Any] = {}
    for sequence in config["experiment"]["sequences"]:
        artifact_dir = exp1_root / sequence / "artifacts"
        arrays = {name: _load_npz(artifact_dir / f"{name}.npz") for name in ("observations", "patches", "frames", "windows")}
        _require_keys("observations", arrays["observations"], REQUIRED_OBSERVATION_KEYS)
        _require_keys("patches", arrays["patches"], REQUIRED_PATCH_KEYS)
        _require_keys("frames", arrays["frames"], REQUIRED_FRAME_KEYS)
        loaded[sequence] = arrays
        schemas[sequence] = {name: _schema(value) for name, value in arrays.items()}

    threshold_config = config["thresholds"]
    reference = loaded[threshold_config["source_sequence"]]["observations"]
    margin = reference[threshold_config["metric"]]
    finite_reference = margin[np.isfinite(margin)]
    q20, q80 = np.quantile(finite_reference, [threshold_config["q20_quantile"], threshold_config["q80_quantile"]])
    if not np.isclose(q20, threshold_config["expected_q20"], rtol=0.0, atol=1e-12):
        raise AssertionError(f"MH_01 Q20 changed: {q20}")
    if not np.isclose(q80, threshold_config["expected_q80"], rtol=0.0, atol=1e-12):
        raise AssertionError(f"MH_01 Q80 changed: {q80}")

    seed = int(config["experiment"]["seed"])
    sample_count = int(profile["samples_per_group"])
    cap = int(config["experiment"]["per_source_frame_cap"])
    all_rows: list[dict[str, Any]] = []
    population: dict[str, Any] = {}
    frame_tables: dict[str, dict[str, np.ndarray]] = {}

    for sequence_index, sequence in enumerate(config["experiment"]["sequences"]):
        arrays = loaded[sequence]
        observations, patches, frames = arrays["observations"], arrays["patches"], arrays["frames"]
        if not np.array_equal(frames["dpvo_counter"], np.arange(len(frames["dpvo_counter"]))):
            raise AssertionError(f"{sequence}: DPVO frame counters are not contiguous")
        patch_index = _join_patches(observations, patches)
        undistorted_xy = np.column_stack((patches["x"][patch_index], patches["y"][patch_index])) * float(coord["dpvo_resolution_scale"])
        raw_xy = undistorted_to_raw(undistorted_xy, calibration)
        token_xy = raw_to_token(raw_xy, metadata, int(coord["token_patch_size"]))
        grid_h, grid_w = map(int, coord["expected_token_grid_hw"])

        source = observations["source_dpvo_counter"].astype(np.int64)
        target = observations["target_dpvo_counter"].astype(np.int64)
        counter_valid = (source >= 0) & (source < len(frames["dpvo_counter"])) & (target >= 0) & (target < len(frames["dpvo_counter"]))
        if not counter_valid.all():
            raise AssertionError(f"{sequence}: observation frame counter out of bounds")
        source_ts = frames["euroc_timestamp_ns"][source].astype(np.int64)
        target_ts = frames["euroc_timestamp_ns"][target].astype(np.int64)
        gt_path = REPO_ROOT / paths["groundtruth_pattern"].format(sequence=sequence)
        gt_min, gt_max = _groundtruth_range(gt_path)
        gt_valid = (source_ts >= gt_min) & (source_ts <= gt_max) & (target_ts >= gt_min) & (target_ts <= gt_max)
        crop_valid = ((token_xy[:, 0] >= 0.0) & (token_xy[:, 0] <= grid_w - 1) &
                      (token_xy[:, 1] >= 0.0) & (token_xy[:, 1] <= grid_h - 1))
        finite = np.ones(len(source), dtype=bool)
        for key in ("corr_peak_l0", "corr_margin_l0", "corr_entropy_l0", "weight_mean", "delta_norm", "apparent_motion"):
            finite &= np.isfinite(observations[key])
        valid = (source != target) & gt_valid & crop_valid & finite

        threshold_masks = {"bad": observations["corr_margin_l0"] <= q20,
                           "good": observations["corr_margin_l0"] >= q80}
        eligible = {name: np.flatnonzero(mask & valid) for name, mask in threshold_masks.items()}
        rng = np.random.default_rng(np.random.SeedSequence([seed, sequence_index]))
        selected = _select_groups(eligible, source, count=sample_count, cap=cap, rng=rng)
        population[sequence] = {}
        for group in ("good", "bad"):
            group_delta_t = [
                (int(frames["euroc_timestamp_ns"][int(target[index])]) -
                 int(frames["euroc_timestamp_ns"][int(source[index])])) * 1e-9
                for index in selected[group]
            ]
            population[sequence][group] = {
                "observation_count_before_threshold": int(len(observations["corr_margin_l0"])),
                "finite_observation_count_before_threshold": int(finite.sum()),
                "threshold_candidate_count": int(threshold_masks[group].sum()),
                "eligible_after_gt_crop_validity_count": int(len(eligible[group])),
                "final_sampled_count": int(len(selected[group])),
                "source_frame_count": int(len(set(int(source[index]) for index in selected[group]))),
                "delta_t_seconds_signed": _distribution(group_delta_t),
            }
            for rank, observation_index in enumerate(selected[group]):
                pi = int(patch_index[observation_index])
                row = _sample_to_row(
                    sequence=sequence, group=group, rank=rank,
                    observation_index=observation_index, observations=observations,
                    patches=patches, frames=frames, patch_index=pi,
                    raw_xy=raw_xy[observation_index], token_xy=token_xy[observation_index],
                    euroc_root=euroc_root,
                )
                if not Path(row["source_image_path"]).is_file() or not Path(row["target_image_path"]).is_file():
                    raise FileNotFoundError(f"Missing sample image for {row['sample_id']}")
                all_rows.append(row)
        frame_tables[sequence] = {"timestamps": frames["euroc_timestamp_ns"].astype(np.int64), "gt_range": np.asarray([gt_min, gt_max])}

    # Canonical order makes JSONL and its digest stable across reruns.
    all_rows.sort(key=lambda row: (config["experiment"]["sequences"].index(row["sequence"]), 0 if row["group"] == "good" else 1, row["sample_rank"]))
    row_by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        row_by_group.setdefault((row["sequence"], row["group"]), []).append(row)
    shuffled_per_group = int(profile["shuffled_per_group"])
    minimum_separation = int(config["experiment"]["shuffled_min_frame_separation"])
    shuffle_rng = np.random.default_rng(np.random.SeedSequence([seed, 991]))
    for sequence in config["experiment"]["sequences"]:
        timestamps = frame_tables[sequence]["timestamps"]
        gt_min, gt_max = frame_tables[sequence]["gt_range"]
        covered_frames = np.flatnonzero((timestamps >= gt_min) & (timestamps <= gt_max))
        image_dir = euroc_root / sequence / "mav0/cam0/data"
        for group in ("good", "bad"):
            rows = row_by_group[(sequence, group)]
            control_count = min(shuffled_per_group, len(rows))
            control_indices = shuffle_rng.choice(len(rows), size=control_count, replace=False)
            for row_index in sorted(control_indices.tolist()):
                row = rows[row_index]
                source_frame = int(row["source_frame_id"])
                candidates = covered_frames[
                    (np.abs(covered_frames - source_frame) >= minimum_separation) &
                    (covered_frames != int(row["target_frame_id"]))
                ]
                if not len(candidates):
                    raise RuntimeError(f"No distant shuffled target for {row['sample_id']}")
                shuffled_frame = int(shuffle_rng.choice(candidates))
                shuffled_ts = int(timestamps[shuffled_frame])
                shuffled_path = image_dir / f"{shuffled_ts}.png"
                if not shuffled_path.is_file():
                    raise FileNotFoundError(shuffled_path)
                row["has_shuffled_control"] = True
                row["shuffled_target_frame_id"] = shuffled_frame
                row["shuffled_target_timestamp_ns"] = shuffled_ts
                row["shuffled_target_image_path"] = str(shuffled_path)

    expected_samples = sum(
        population[sequence][group]["final_sampled_count"]
        for sequence in config["experiment"]["sequences"] for group in ("good", "bad")
    )
    if len(all_rows) != expected_samples or len({row["sample_id"] for row in all_rows}) != expected_samples:
        raise AssertionError("Formal sample count or identity mismatch")
    source_caps = Counter((row["sequence"], row["source_frame_id"]) for row in all_rows)
    if max(source_caps.values()) > cap:
        raise AssertionError("Shared per-sequence source-frame cap violated")
    actual_shuffled = sum(bool(row["has_shuffled_control"]) for row in all_rows)
    expected_shuffled = sum(
        min(shuffled_per_group, population[sequence][group]["final_sampled_count"])
        for sequence in config["experiment"]["sequences"] for group in ("good", "bad")
    )
    if actual_shuffled != expected_shuffled:
        raise AssertionError("Formal shuffled-control count mismatch")

    jsonl_bytes = b"".join(
        (json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in all_rows
    )
    (artifacts_dir / "samples.jsonl").write_bytes(jsonl_bytes)
    np.savez_compressed(artifacts_dir / "samples.npz", **_rows_to_npz(all_rows))

    vjepa_repo = Path(config["vjepa"]["repo"])
    manifest = {
        "schema_version": 2,
        "experiment": config["experiment"]["name"],
        "sampling_profile": PROFILE_NAME,
        "dpvo_git_commit": _git_commit(REPO_ROOT),
        "vjepa_git_commit": _git_commit(vjepa_repo),
        "config_path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        "config_sha256": _sha256(config_bytes),
        "samples_jsonl_sha256": _sha256(jsonl_bytes),
        "thresholds": {
            "metric": "corr_margin_l0", "mh01_q20": float(q20), "mh01_q80": float(q80),
            "bad_definition": "corr_margin_l0 <= MH_01 Q20",
            "bad_group_description": threshold_config["bad_group_description"],
            "good_definition": "corr_margin_l0 >= MH_01 Q80",
            "q20_zero_tie_preserved": True,
        },
        "sampling": {
            "seed": seed, "per_source_frame_cap_shared_across_groups": cap,
            "sample_counts": population,
            "requested_samples_per_group": sample_count,
            "requested_shuffled_per_group": shuffled_per_group,
            "total_samples": len(all_rows), "shuffled_control_count": actual_shuffled,
            "algorithm": "per-sequence seeded permutations; alternate good/bad acceptance under a shared source-frame cap",
        },
        "validity_filter": {
            "cross_frame_only": True, "both_timestamps_inside_gt": True,
            "source_continuous_token_xy_range": [[0.0, 23.0], [0.0, 23.0]],
            "no_padding_bilinear_source_sampling": True,
        },
        "coordinate_contract": {"dpvo_scale": 4.0, "crop_metadata": metadata, "token_patch_size": 16, "token_grid_hw": [24, 24]},
        "exp1_source_artifacts": {
            sequence: str((exp1_root / sequence / "artifacts").relative_to(REPO_ROOT))
            for sequence in config["experiment"]["sequences"]
        },
        "exp1_schema": schemas,
        "sample_schema": _schema(_rows_to_npz(all_rows)),
        "secondary_tags": {
            "low_texture": "source difficulty_texture_quintile == 5",
            "large_motion": "source difficulty_model_motion_quintile == 5",
            "dpvo_delta": "only delta_norm is recoverable from Exp 1; no vector is claimed",
        },
    }
    _write_json(artifacts_dir / "manifest.json", manifest)
    print(json.dumps({"profile": PROFILE_NAME, "q20": float(q20), "q80": float(q80), "population": population, "samples": len(all_rows), "shuffled": actual_shuffled}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
