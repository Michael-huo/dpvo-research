"""Lossless scientific aggregation and publication of anchor-budget artifacts.

This module only reads completed results. It never trains, runs SLAM, fits an
alignment, or computes an evaluation metric. Worker/staging files stay in /tmp.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .protocol import REPO_ROOT, atomic_write_json, canonical_sha256, sha256_file
from .artifact_runtime import (
    staged_directory, publish_current_canonical, validate_lightweight_results,
    without_worker_log_paths,
)

CHECKPOINT_ROOT = REPO_ROOT / "research/checkpoints/anchor-budget"
FORMAL_FILES = {"SUMMARY.md", "results.json", "trajectories.npz",
                "figures/trajectories.png", "figures/tradeoffs.png"}


def inventory(root):
    return {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def repository_provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    return {"git_commit": git("rev-parse", "HEAD"), "worktree_dirty": bool(git("status", "--short"))}


def columnar(rows):
    columns = list(rows[0]) if rows else []
    if any(set(row) != set(columns) for row in rows):
        raise ValueError("scientific table columns differ")
    return {"columns": columns, "rows": [[row[key] for key in columns] for row in rows]}


def expand_columns(table):
    return [dict(zip(table["columns"], row)) for row in table["rows"]]


def _array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def add_trajectory(bundle, name, timestamps, poses):
    timestamps, poses = np.asarray(timestamps), np.asarray(poses)
    if poses.shape != (len(timestamps), 7) or len(timestamps) < 2:
        raise ValueError(f"invalid trajectory shape: {name}")
    if not np.all(np.diff(timestamps.astype(np.int64)) > 0) or not np.isfinite(poses).all():
        raise ValueError(f"invalid trajectory values: {name}")
    bundle[f"{name}__timestamps_ns"] = timestamps.copy()
    bundle[f"{name}__translation"] = poses[:, :3].copy()
    bundle[f"{name}__quaternion_xyzw"] = poses[:, 3:7].copy()
    return {"point_count": len(timestamps), "timestamps_dtype": str(timestamps.dtype),
            "poses_dtype": str(poses.dtype), "timestamps_sha256": _array_hash(timestamps),
            "poses_sha256": _array_hash(poses), "quaternion_order": "xyzw",
            "coordinate_frame": "groundtruth_world" if name == "GT" else "original_dpvo_unaligned"}


def reconstruct_trajectory(bundle, name):
    return {"timestamps_ns": bundle[f"{name}__timestamps_ns"].copy(),
            "poses": np.concatenate((bundle[f"{name}__translation"],
                                      bundle[f"{name}__quaternion_xyzw"]), axis=1)}


def reconstruct_schedule(bundle, stride):
    prefix = f"schedule_stride_{stride}__"
    return {"roles": dict(zip(bundle[prefix + "identities"].tolist(), bundle[prefix + "roles"].tolist())),
            "intervals": [json.loads(value) for value in bundle[prefix + "intervals_json"].tolist()]}


def _compact_condition(row, population_ref):
    result = {key: copy.deepcopy(value) for key, value in row.items()
              if key not in {"canonical_population", "provider_usage", "sequential_execution"}}
    result["evaluation_population_ref"] = population_ref
    # Aggregate timing/capability fields remain; per-frame worker telemetry does not.
    if "provider_usage" in row:
        result["deployment"] = {key: copy.deepcopy(value) for key, value in row["provider_usage"].items()
                                if key not in {"timeline", "waits", "jepa_worker_pid", "worker_provenance",
                                               "lifecycle_cleanup", "workers"}}
    result["execution"] = {key: copy.deepcopy(row["sequential_execution"][key]) for key in (
        "kind", "condition", "logical_device", "required_logical_devices", "elapsed_seconds")
        if key in row["sequential_execution"]}
    if "h2_stage_profile" in result:
        result["h2_stage_profile"].pop("stage_c_runtime", None)
        # Already retained verbatim in h2_stage_profile, not two copies.
        result["runtime"].pop("stage_c_timing", None)
    return without_worker_log_paths(result)


def _compact_training(training):
    result = copy.deepcopy(training)
    for key in ("lineage", "horizon_resolved_quality", "split", "anchor_stride"):
        result.pop(key)
    result["epoch_metrics"] = columnar(result["summary"].pop("history"))
    result["total_wall_seconds"] = result["training_and_diagnostics_wall_seconds"]
    result["predictor_wall_seconds"] = result["summary"]["elapsed_seconds"]
    result["total_wall_definition"] = "original_training_and_diagnostics_wall_seconds_includes_preparation"
    # Batch-level extraction timing is debugging detail, not a training target/result.
    samples = result["test_extraction"].pop("encoder_inference_ms", [])
    result["test_extraction"]["encoder_inference_summary"] = {
        "batch_count": len(samples), "total_ms": sum(samples),
        "source_samples_sha256": canonical_sha256(samples)}
    runtime = result["test_extraction"].get("jepa", {})
    runtime.pop("worker_pid", None)
    runtime.pop("runtime", None)
    return result


def _checkpoint_metadata(path, destination, training, expected_hash):
    import torch
    from .predictor import predictor_state_sha256
    from .anchor_budget_training import validate_stride_lineage
    if sha256_file(path) != expected_hash:
        raise RuntimeError("checkpoint file SHA256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stride = training["anchor_stride"]
    lineage = checkpoint["training_lineage"]
    validate_stride_lineage(lineage, training["lineage"], stride)
    if predictor_state_sha256(checkpoint["state_dict"]) != checkpoint["state_dict_sha256"]:
        raise RuntimeError("checkpoint tensor integrity mismatch")
    return {"relative_path": os.path.relpath(destination, REPO_ROOT), "path_base": "repository_root",
            "sha256": expected_hash, "stride": stride, "seed": lineage["seed"],
            "best_epoch": training["summary"]["best_epoch"], "bytes": path.stat().st_size,
            "state_dict_sha256": checkpoint["state_dict_sha256"],
            "scientific_lineage_sha256": lineage["training_lineage_sha256"],
            "scientific_lineage": lineage}


def build_compact(source, checkpoint_root=CHECKPOINT_ROOT):
    """Build compact values plus an exact source-data verification contract."""
    source, checkpoint_root = Path(source), Path(checkpoint_root)
    original = inventory(source)
    used = set()
    def read(name):
        used.add(str(name))
        return json.loads((source / name).read_text())
    index = read("INDEX.json")
    if index["status"] != "complete":
        raise RuntimeError("only a completed experiment can be consolidated")
    strides, sequence = index["anchor_strides"], index["sequence"]
    if sequence != "MH_01_easy" or not strides or len(set(strides)) != len(strides) or any(isinstance(s, bool) or not isinstance(s, int) or s < 2 for s in strides):
        raise ValueError("unsupported completed pilot")
    result = {"schema_version": 2, "metadata": {
        "status": "complete", "sequence": sequence, "anchor_strides": strides,
        "run_policy": "fresh_current_canonical_replace",
        "scientific_question": "How do Sparse RGB and Ours change as anchor upload budget and prediction horizon change?",
        "fresh_predictor_per_stride": True,
        "fixed_split": index["preparation"]["fixed_split"],
        "protocol": index["protocol"], "trajectory_comparison": index["trajectory_comparison"],
        "config_path": os.path.relpath(index["config"], REPO_ROOT), "config_sha256": index["config_sha256"],
        "repository": index["repository"],
        "execution": without_worker_log_paths(index["execution"]),
        "stride5_equivalence": index["preparation"]["stride5_equivalence"],
        "evaluation_populations": {}, "trajectories": {},
        "plotting": {"projection": "xy", "alignment": "apply_saved_sim3_no_refit",
                     "full_rgb_alignment": "one_saved_canonical_stride5_sim3_shared_across_panels",
                     "trend_samples": "measured_points_only_no_interpolation"}},
        "full_rgb": {}, "strides": {}}
    metadata = result["metadata"]
    bundle, proof, checkpoints = {}, {"trajectories": {}, "admission": {}, "schedules": {}, "scientific": {}}, []
    seqroot = Path("sequences") / sequence
    gt_ref = read(seqroot / "groundtruth.json")
    gt_path = Path(gt_ref["path"])
    if sha256_file(gt_path) != gt_ref["sha256"]:
        raise RuntimeError("groundtruth reference changed")
    gt = np.loadtxt(gt_path)
    # Identical to the feasibility evaluation reader, including float64 timestamp rounding.
    gt_ts = np.rint(gt[:, 0]).astype(np.int64)
    gt_poses = np.concatenate((gt[:, 1:4], gt[:, [5,6,7,4]]), axis=1)
    metadata["groundtruth"] = {"source_path": os.path.relpath(gt_path, REPO_ROOT),
                               "source_sha256": gt_ref["sha256"], "role": "reference_not_slam_run",
                               "timestamp_semantics": "research_np_loadtxt_float64_rint_int64",
                               "scope": "entire_original_gt_reference_included"}
    metadata["trajectories"]["GT"] = add_trajectory(bundle, "GT", gt_ts, gt_poses)
    proof["trajectories"]["GT"] = {"timestamps_ns": gt_ts, "poses": gt_poses}

    def trajectory(name, folder, row):
        path = folder / "trajectory.npz"
        used.add(str(path))
        if sha256_file(source / path) != row["trajectory_sha256"]:
            raise RuntimeError(f"source trajectory hash changed: {name}")
        with np.load(source / path, allow_pickle=False) as old:
            allowed = {"poses", "timestamps_ns", "admission_identities", "admission_insert"}
            if not {"poses", "timestamps_ns"} <= set(old.files) <= allowed:
                raise ValueError(f"unmapped scientific arrays in {path}")
            arrays = {key: old[key].copy() for key in old.files}
        metadata["trajectories"][name] = add_trajectory(bundle, name, arrays["timestamps_ns"], arrays["poses"])
        proof["trajectories"][name] = {key: arrays[key] for key in ("poses", "timestamps_ns")}
        admission_keys = {"admission_identities", "admission_insert"}
        if admission_keys & arrays.keys():
            if not admission_keys <= arrays.keys():
                raise ValueError(f"incomplete admission arrays: {name}")
            identities, mask = arrays["admission_identities"], arrays["admission_insert"]
            if (identities.ndim != 1 or identities.dtype.kind != "U"
                    or mask.shape != identities.shape or mask.dtype != np.bool_
                    or len(set(identities.tolist())) != len(identities)
                    or int(mask.sum()) != len(arrays["timestamps_ns"])):
                raise ValueError(f"invalid admission arrays: {name}")
            proof["admission"][name] = {key: arrays[key].copy() for key in admission_keys}
            metadata["trajectories"][name]["admission_arrays_sha256"] = {
                key: _array_hash(arrays[key]) for key in sorted(admission_keys)}
            for key in admission_keys:
                bundle[name + "__" + key] = arrays[key]

    full = read(seqroot / "full_rgb/results.json")
    metadata["evaluation_populations"]["stride_5"] = full["canonical_population"]
    result["full_rgb"] = _compact_condition(full, "stride_5")
    trajectory("Full_RGB", seqroot / "full_rgb", full)
    proof["scientific"]["full_rgb"] = copy.deepcopy(result["full_rgb"])
    comparisons, horizons = [], []
    for stride in strides:
        folder = seqroot / f"stride_{stride}"
        old = read(folder / "results.json")
        schedule = read(folder / "schedule.json")
        if {key: schedule[key] for key in old["population"]} != old["population"]:
            raise RuntimeError("schedule summaries disagree")
        sparse = read(folder / "sparse_rgb/results.json")
        ours = read(folder / "predicted_jepa/results.json")
        population_key = f"stride_{stride}"
        population = sparse["canonical_population"]
        if population != ours["canonical_population"] or (population_key in metadata["evaluation_populations"]
                and metadata["evaluation_populations"][population_key] != population):
            raise RuntimeError("stored evaluation populations disagree")
        metadata["evaluation_populations"][population_key] = population
        training = read(Path(old["training_ref"]))
        query_rows = read(Path(old["training_ref"]).parent / "horizon_queries.json")
        checkpoint_source = source / old["predictor_ref"]
        used.add(old["predictor_ref"])
        destination = checkpoint_root / f"predictor_stride_{stride}.pt"
        checkpoint = _checkpoint_metadata(checkpoint_source, destination, training, old["predictor_sha256"])
        checkpoints.append((checkpoint_source, destination, checkpoint["sha256"]))
        compact_schedule = {key: copy.deepcopy(value) for key, value in old["population"].items()
                            if key not in {"anchor_stride", "communication"}}
        compact_schedule["identity_arrays_prefix"] = f"schedule_stride_{stride}__"
        row = {"schedule": compact_schedule, "communication": old["population"]["communication"],
               "training": _compact_training(training),
               "sparse": _compact_condition(sparse, population_key),
               "ours": _compact_condition(ours, population_key),
               "horizon": {"by_relative_index": training["horizon_resolved_quality"], "queries": columnar(query_rows)},
               "comparison": old["comparison"],
               "provenance": {"checkpoint": checkpoint, "H1_bridge_sha256": training["h1_bridge"]["file_sha256"],
                              "scientific_config": training["lineage"]["scientific_training_contract"],
                              "training_cleanup": old["training_cleanup"]}}
        row = without_worker_log_paths(row)
        result["strides"][str(stride)] = row
        for name, condition, old_row in (("sparse", "sparse_rgb", sparse), ("ours", "predicted_jepa", ours)):
            trajectory(f"stride_{stride}_{name}", folder / condition, old_row)
        prefix = compact_schedule["identity_arrays_prefix"]
        bundle[prefix + "identities"] = np.asarray(list(schedule["roles"]))
        bundle[prefix + "roles"] = np.asarray(list(schedule["roles"].values()))
        bundle[prefix + "intervals_json"] = np.asarray([json.dumps(i, sort_keys=True, separators=(",", ":"))
                                                       for i in schedule["intervals"]])
        proof["schedules"][stride] = {"roles": schedule["roles"], "intervals": schedule["intervals"]}
        proof["scientific"][str(stride)] = copy.deepcopy(row)
        if expand_columns(row["horizon"]["queries"]) != query_rows or expand_columns(row["training"]["epoch_metrics"]) != training["summary"]["history"]:
            raise RuntimeError("scientific table conversion was not exact")
        comparisons.append(old["comparison"])
        horizons.extend(training["horizon_resolved_quality"])
    if comparisons != index["accuracy_vs_communication"] or horizons != index["prediction_quality_vs_horizon"]:
        raise RuntimeError("INDEX scientific tables disagree with per-stride results")
    for filename, expected in (("accuracy_vs_communication.json", comparisons),
                               ("prediction_quality_vs_horizon.json", horizons),
                               ("PREPARATION.json", index["preparation"])):
        if filename in original and read(filename) != expected:
            raise RuntimeError(f"duplicate scientific artifact disagrees: {filename}")
    derived = {"SUMMARY.md", "accuracy_vs_communication.csv", "prediction_quality_vs_horizon.csv"}
    unknown = set(original) - used - derived
    if unknown:
        raise RuntimeError(f"unmapped files must be audited before cleanup: {sorted(unknown)}")
    proof["inventory"] = original
    return result, bundle, checkpoints, proof


def validate_compact(root, *, check_checkpoints=True, proof=None, checkpoint_root=None):
    root = Path(root)
    result = json.loads((root / "results.json").read_text())
    if result["schema_version"] != 2 or result["metadata"]["status"] != "complete":
        raise ValueError("not a completed compact artifact")
    if sha256_file(root / "trajectories.npz") != result["metadata"]["trajectories_npz_sha256"]:
        raise RuntimeError("consolidated NPZ integrity mismatch")
    with np.load(root / "trajectories.npz", allow_pickle=False) as bundle:
        for name, manifest in result["metadata"]["trajectories"].items():
            arrays = reconstruct_trajectory(bundle, name)
            if len(arrays["poses"]) != manifest["point_count"] or _array_hash(arrays["poses"]) != manifest["poses_sha256"] or _array_hash(arrays["timestamps_ns"]) != manifest["timestamps_sha256"]:
                raise RuntimeError(f"trajectory array mismatch: {name}")
            for field, digest in manifest.get("admission_arrays_sha256", {}).items():
                value = bundle[name + "__" + field]
                if _array_hash(value) != digest:
                    raise RuntimeError(f"admission array mismatch: {name}/{field}")
                if proof is not None:
                    before = proof["admission"][name][field]
                    if value.dtype != before.dtype or not np.array_equal(value, before):
                        raise RuntimeError(f"admission array changed: {name}/{field}")
            if proof is not None:
                for field, value in arrays.items():
                    before = proof["trajectories"][name][field]
                    if value.dtype != before.dtype or not np.array_equal(value, before):
                        raise RuntimeError(f"trajectory changed: {name}/{field}")
        if proof is not None:
            for stride, old in proof["schedules"].items():
                if reconstruct_schedule(bundle, stride) != old:
                    raise RuntimeError("schedule payload changed")
    if proof is not None:
        if result["full_rgb"] != proof["scientific"]["full_rgb"]:
            raise RuntimeError("Full RGB scientific results changed")
        for stride in result["strides"]:
            if result["strides"][stride] != proof["scientific"][stride]:
                raise RuntimeError(f"scientific results changed: stride {stride}")
    if check_checkpoints:
        import torch
        from .anchor_budget_training import validate_stride_lineage
        for stride, row in result["strides"].items():
            expected = row["provenance"]["checkpoint"]
            path = (Path(checkpoint_root) / Path(expected["relative_path"]).name
                    if checkpoint_root is not None else REPO_ROOT / expected["relative_path"])
            if path.resolve().is_relative_to(root.resolve()) or sha256_file(path) != expected["sha256"]:
                raise RuntimeError("checkpoint location/hash mismatch")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            validate_stride_lineage(checkpoint["training_lineage"], expected["scientific_lineage"], int(stride))
    validate_lightweight_results(root)
    return result


def publish_compact(source, destination, checkpoint_root=CHECKPOINT_ROOT):
    """Aggregate fresh staging only, then replace both canonical directories."""
    from .anchor_budget_figures import render_figures, write_summary
    source, destination, checkpoint_root = Path(source), Path(destination), Path(checkpoint_root)
    if source.resolve() == destination.resolve():
        raise ValueError("fresh staging must be separate from canonical results")
    result, arrays, checkpoints, proof = build_compact(source, checkpoint_root)
    with staged_directory(destination) as staged, staged_directory(checkpoint_root) as staged_checkpoints:
        np.savez_compressed(staged / "trajectories.npz", **arrays)
        result["metadata"]["trajectories_npz_sha256"] = sha256_file(staged / "trajectories.npz")
        atomic_write_json(staged / "results.json", result)
        validate_compact(staged, check_checkpoints=False, proof=proof)
        render_figures(staged)
        write_summary(staged)
        if set(inventory(staged)) != FORMAL_FILES:
            raise RuntimeError("unexpected staging outputs")
        for origin, target, expected in checkpoints:
            staged_path = staged_checkpoints / target.name
            shutil.copy2(origin, staged_path)
            if sha256_file(staged_path) != expected:
                raise RuntimeError("checkpoint staging changed bytes")
        validate_compact(staged, proof=proof, checkpoint_root=staged_checkpoints)
        if {p.name for p in staged_checkpoints.iterdir()} != {p.name for _, p, _ in checkpoints}:
            raise RuntimeError("checkpoint request set mismatch")
        if inventory(source) != proof["inventory"]:
            raise RuntimeError("fresh source changed during aggregation")
        publish_current_canonical(staged, destination, checkpoint_staged=staged_checkpoints,
                                  checkpoint_destination=checkpoint_root)
    return result
