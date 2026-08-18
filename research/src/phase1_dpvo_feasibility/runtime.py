"""Internal runtime for the fixed Experiment 1 protocol.

This module intentionally has no public experiment CLI.  ``run_exp1`` is the
only supported entry point; the small ``--internal-worker`` interface exists
solely to isolate CUDA/DPVO processes from the orchestrator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from evo.core import sync
from evo.core.geometry import GeometryException, umeyama_alignment
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
import evo.main_ape as main_ape


SCHEMA_VERSION = 1
SEQUENCES = ("MH_01_easy", "MH_03_medium", "MH_05_difficult")
UINT64_MASK = (1 << 64) - 1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (value ^ (value >> 31)) & UINT64_MASK


def splitmix64_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64)
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def stable_observation_hash(primary_id: int, update_id: int) -> int:
    return splitmix64((int(primary_id) ^ (int(update_id) * 0xD6E8FEB86659FD93)) & UINT64_MASK)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json_dump(path: str | Path, value: Any) -> None:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(json_ready(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_metadata(root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def resolve_config(*, sequence: str, max_frames: int | None = None) -> dict[str, Any]:
    root = repo_root()
    source = root / "research/configs/phase1_dpvo_feasibility.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if sequence not in SEQUENCES:
        raise ValueError(f"unsupported Experiment 1 sequence: {sequence}")
    config["experiment"]["sequence"] = sequence
    config["experiment"]["max_frames"] = max_frames
    paths = config["paths"]
    paths["euroc_root"] = str((root / paths["euroc_root"]).resolve())
    paths["image_dir"] = str(Path(paths["euroc_root"]) / sequence / "mav0/cam0/data")
    paths["data_csv"] = str(Path(paths["euroc_root"]) / sequence / "mav0/cam0/data.csv")
    paths["groundtruth"] = str((root / paths["groundtruth_pattern"].format(sequence=sequence)).resolve())
    for name in ("calib", "network", "dpvo_config", "output_root"):
        paths[name] = str((root / paths[name]).resolve())
    paths["config_source"] = str(source)
    config["repo_root"] = str(root)
    required = ("image_dir", "data_csv", "groundtruth", "calib", "network", "dpvo_config")
    missing = [name for name in required if not Path(paths[name]).exists()]
    if missing:
        raise FileNotFoundError("missing Experiment 1 inputs: " + ", ".join(missing))
    return config


def load_euroc_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    csv_rows: dict[int, str] = {}
    with Path(config["paths"]["data_csv"]).open(encoding="utf-8") as handle:
        for row in csv.reader(line for line in handle if not line.startswith("#")):
            if row:
                csv_rows[int(row[0])] = row[1]
    image_dir = Path(config["paths"]["image_dir"])
    images = sorted([*image_dir.glob("*.png"), *image_dir.glob("*.jpeg"), *image_dir.glob("*.jpg")])
    experiment = config["experiment"]
    selected = images[int(experiment["skip"])::int(experiment["stride"])]
    if experiment["max_frames"] is not None:
        selected = selected[:int(experiment["max_frames"])]
    records = []
    for index, image in enumerate(selected):
        timestamp = int(image.stem)
        if csv_rows.get(timestamp) != image.name:
            raise ValueError(f"EuRoC filename/data.csv mismatch: {image.name}")
        records.append({"stream_index": index, "euroc_timestamp_ns": timestamp, "image_path": str(image)})
    return records


def trajectory_difference(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape != right.shape:
        raise AssertionError(f"trajectory shape mismatch: {left.shape} vs {right.shape}")
    lhs, rhs = np.asarray(left, dtype=np.float64).copy(), np.asarray(right, dtype=np.float64).copy()
    position = np.linalg.norm(lhs[:, :3] - rhs[:, :3], axis=1)
    ql, qr = lhs[:, 3:7], rhs[:, 3:7]
    ql /= np.linalg.norm(ql, axis=1, keepdims=True); qr /= np.linalg.norm(qr, axis=1, keepdims=True)
    rotation = 2.0 * np.arccos(np.clip(np.abs(np.sum(ql * qr, axis=1)), -1.0, 1.0))
    rotation_matrix, translation, scale = np.eye(3), np.zeros(3), 1.0
    aligned = rhs[:, :3]
    try:
        rotation_matrix, translation, scale = umeyama_alignment(rhs[:, :3].T, lhs[:, :3].T, with_scale=True)
        aligned = (scale * (rotation_matrix @ rhs[:, :3].T) + translation[:, None]).T
    except GeometryException:
        pass
    aligned_error = np.linalg.norm(lhs[:, :3] - aligned, axis=1)
    return {
        "position_rmse": float(np.sqrt(np.mean(position ** 2))),
        "sim3_aligned_position_rmse": float(np.sqrt(np.mean(aligned_error ** 2))),
        "rotation_rmse_rad": float(np.sqrt(np.mean(rotation ** 2))),
    }


def _worker_command(config_path: Path, mode: str, result_json: Path, result_npz: Path, *, trajectory_path: Path | None = None, events_path: Path | None = None, behavior_output: Path | None = None) -> list[str]:
    command = [sys.executable, "-m", "research.src.phase1_dpvo_feasibility.runtime", "--internal-worker", "--config-json", str(config_path), "--mode", mode, "--result-json", str(result_json), "--result-npz", str(result_npz)]
    if trajectory_path: command += ["--trajectory-path", str(trajectory_path)]
    if events_path: command += ["--events-path", str(events_path)]
    if behavior_output: command += ["--behavior-output", str(behavior_output)]
    return command


def _run_worker(root: Path, config: dict[str, Any], mode: str, temporary: Path, *, trajectory_path: Path | None = None, behavior_output: Path | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    config_path, result_json, result_npz = temporary / f"{mode}_config.json", temporary / f"{mode}.json", temporary / f"{mode}.npz"
    atomic_json_dump(config_path, config)
    command = _worker_command(config_path, mode, result_json, result_npz, trajectory_path=trajectory_path, events_path=temporary / f"{mode}.events.jsonl" if mode == "probe" else None, behavior_output=behavior_output)
    environment = os.environ.copy(); environment["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(command, cwd=root, env=environment, check=False)
    result = load_json(result_json) if result_json.exists() else {"status": "error", "error": "worker emitted no result"}
    if completed.returncode or result.get("status") != "ok":
        raise RuntimeError(f"{mode} worker failed: {result}")
    return result, load_npz_arrays(result_npz)


def probe_perturbation(baseline: dict[str, Any], probe: dict[str, Any], baseline_arrays: dict[str, np.ndarray], probe_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Record probe perturbation without turning it into an acceptance envelope."""
    difference = trajectory_difference(baseline_arrays["poses"], probe_arrays["poses"])
    baseline_ate = float(baseline["ate"]["translation_rmse"])
    probe_ate = float(probe["ate"]["translation_rmse"])
    ate_difference = abs(baseline_ate - probe_ate)
    relative_ate_change = ate_difference / baseline_ate if baseline_ate else None
    if relative_ate_change is None:
        label, representativeness = "undefined", "caution"
    elif relative_ate_change < .05:
        label, representativeness = "small", "normal"
    elif relative_ate_change < .10:
        label, representativeness = "moderate", "caution"
    else:
        label, representativeness = "large", "low"
    checks = {
        "internal_timestamps_exact": bool(np.array_equal(baseline_arrays["internal_tstamps"], probe_arrays["internal_tstamps"])),
        "euroc_timestamps_exact": bool(np.array_equal(baseline_arrays["euroc_timestamps_ns"], probe_arrays["euroc_timestamps_ns"])),
        "trajectory_pose_counts_equal": len(baseline_arrays["poses"]) == len(probe_arrays["poses"]),
        "finite_trajectory_diagnostic": bool(np.isfinite([baseline_ate, probe_ate, ate_difference, *difference.values()]).all()),
    }
    return {
        "probe_valid": all(checks.values()),
        "checks": checks,
        "baseline_translation_ate_rmse": baseline_ate,
        "probe_translation_ate_rmse": probe_ate,
        "ate_absolute_difference": ate_difference,
        "relative_ate_change": relative_ate_change,
        "perturbation_label": label,
        "probe_representativeness": representativeness,
        "trajectory_difference": difference,
        "interpretation": "Feasibility-stage engineering diagnostic; not a significance test or numerical acceptance gate.",
    }


def run_sanity_gate(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Run a disposable instrumentation smoke gate, never a parity envelope."""
    sanity = dict(config); sanity["experiment"] = dict(config["experiment"]); sanity["probe"] = dict(config["probe"])
    sanity["experiment"]["max_frames"] = int(config["experiment"]["sanity_frames"])
    sanity["probe"]["sanity_validation"] = True
    with tempfile.TemporaryDirectory(prefix="exp1-sanity-") as name:
        temporary = Path(name)
        baseline, baseline_arrays = _run_worker(root, sanity, "baseline", temporary)
        probe, probe_arrays = _run_worker(root, sanity, "probe", temporary, behavior_output=temporary / "behavior")
        diagnostic = probe_perturbation(baseline, probe, baseline_arrays, probe_arrays)
        checks = {
            **diagnostic["checks"],
            "hooks_restored": bool(probe["probe"]["hooks_restored"]),
            "no_probe_contract_violations": probe["probe"]["violations"] == [],
            "threshold_spool_cleaned": bool(probe["behavior"]["temporary_threshold_spool"]["cleaned"]),
        }
        if not all(checks.values()):
            raise RuntimeError("200-frame instrumentation sanity gate failed")
        return {
            "schema_version": SCHEMA_VERSION,
            "sequence": "MH_01_easy",
            "frames": len(baseline_arrays["poses"]),
            "probe_perturbation": diagnostic,
            "checks": checks,
            "sets_full_sequence_tolerance": False,
            "temporary_artifacts_deleted": True,
        }


def _array_quantiles(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64); finite = finite[np.isfinite(finite)]
    if not len(finite): return {"count": 0, "mean": None, "q20": None, "median": None, "q80": None}
    return {"count": int(len(finite)), "mean": float(finite.mean()), "q20": float(np.quantile(finite, .2)), "median": float(np.quantile(finite, .5)), "q80": float(np.quantile(finite, .8))}


def _artifact_inventory(run_dir: Path) -> list[dict[str, Any]]:
    return [{"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "manifest.json"]


def _reference_provenance(metrics_path: Path | None) -> dict[str, Any] | None:
    if metrics_path is None: return None
    metrics = load_json(metrics_path)
    source = metrics["provenance"]
    return {"reference_run_id": source["run_id"], "metrics_path": f"{source['run_id']}/report/metrics.json", "metrics_sha256": sha256_file(metrics_path), "git_commit": source["git"]["commit"], "config_sha256": source["config_sha256"], "frozen_thresholds": metrics["behavior"]["thresholds"]}


def run_sequence(root: Path, sequence: str, output: Path, sanity_gate: dict[str, Any], reference_metrics: Path | None) -> Path:
    """Run the uninstrumented baseline and behavior probe for one sequence."""
    from .behavior import difficulty_stratification, enrich_sequence_outputs, load_npz, make_sequence_figures, sample_spearman
    config = resolve_config(sequence=sequence)
    config["behavior"]["enabled"] = True
    if reference_metrics:
        reference = _reference_provenance(reference_metrics)
        config["behavior"]["reference_thresholds"] = reference["frozen_thresholds"]
    else:
        reference = None
    records = load_euroc_records(config); config["experiment"]["resolved_frame_count"] = len(records)
    output.mkdir(parents=True, exist_ok=False); report, artifacts = output / "report", output / "artifacts"; report.mkdir(); artifacts.mkdir()
    provenance, config_hash = git_metadata(root), json_sha256(config)
    try:
        with tempfile.TemporaryDirectory(prefix=f"exp1-{sequence}-") as name:
            temporary = Path(name)
            baseline, baseline_arrays = _run_worker(root, config, "baseline", temporary, trajectory_path=report / "trajectory_est.tum")
            probe, probe_arrays = _run_worker(root, config, "probe", temporary, behavior_output=artifacts)
            perturbation = probe_perturbation(baseline, probe, baseline_arrays, probe_arrays)
            print("probe perturbation:\n" + json.dumps(perturbation, indent=2, sort_keys=True), flush=True)
            if not perturbation["probe_valid"]:
                raise RuntimeError(f"{sequence} probe violated a structural trajectory/input contract")
            # Behavior observations and their local trajectory association refer to the same probe run.
            trajectory_analysis = enrich_sequence_outputs(artifacts, probe_arrays["poses"], config["paths"]["groundtruth"], seed=int(config["experiment"]["seed"]))
    except Exception as error:
        atomic_json_dump(report / "summary.json", {"schema_version": SCHEMA_VERSION, "status": "failed", "sequence": sequence, "error": repr(error), "sanity_gate": sanity_gate})
        raise
    frames, patches, observations = (load_npz(artifacts / name) for name in ("frames.npz", "patches.npz", "observations.npz"))
    behavior = probe["behavior"]
    metrics = {"schema_version": SCHEMA_VERSION, "sequence": sequence, "behavior": behavior, **trajectory_analysis, "observation_spearman": sample_spearman(observations), "difficulty_stratification": difficulty_stratification(frames), "association_guardrail": "Descriptive/exploratory association only; no causal or statistical-significance claim.", "provenance": {"run_id": sequence, "git": provenance, "config_sha256": config_hash, "reference": reference}}
    atomic_json_dump(report / "metrics.json", metrics)
    figure_paths = make_sequence_figures(output, sequence)
    total = int(frames["online_observation_count"].sum()); accepted = patches["accepted"]
    checks = {"frame_count_expected": len(frames["stream_index"]) == len(records), "probe_structural_contract": perturbation["probe_valid"], "hooks_restored": bool(probe["probe"]["hooks_restored"]), "no_probe_contract_violations": probe["probe"]["violations"] == [], "threshold_spool_cleaned": bool(behavior["temporary_threshold_spool"]["cleaned"]), "sample_bounded": len(observations["score"]) <= int(config["behavior"]["observation_bottom_k"]), "all_observation_scalars_finite": all(np.isfinite(value).all() for value in observations.values() if value.dtype.kind == "f"), "trajectory_est_is_baseline": (report / "trajectory_est.tum").exists(), "five_core_figures": len(figure_paths) == 5, "dpvo_worktree_unchanged": not subprocess.check_output(["git", "status", "--short", "--", "dpvo"], cwd=root, text=True).splitlines()}
    summary = {"schema_version": SCHEMA_VERSION, "status": "passed" if all(checks.values()) else "failed", "sequence": sequence, "frames_processed": len(records), "trajectory": {"baseline_translation_ate_rmse": baseline["ate"]["translation_rmse"], "probe_translation_ate_rmse": probe["ate"]["translation_rmse"], **trajectory_analysis["trajectory"]}, "headline_performance": {"source": "uninstrumented baseline", "fps": baseline["runtime"]["fps_excluding_decode_h2d_and_terminate"], "peak_allocated_bytes": baseline["runtime"]["peak_allocated_bytes"]}, "probe_perturbation": perturbation, "correlation": {"peak_l0": behavior["moments_exact"]["corr_peak_l0"], "margin_l0": behavior["moments_exact"]["corr_margin_l0"], "entropy_l0": behavior["moments_exact"]["corr_entropy_l0"]}, "weight": behavior["moments_exact"]["weight_mean"], "low_confidence_ratio": int(frames["low_confidence_count"].sum()) / total if total else None, "large_delta_ratio": int(frames["large_delta_count"].sum()) / total if total else None, "patch_residency": _array_quantiles(patches["patch_residency_lifetime"][accepted]), "factor_observation_lifetime": behavior["factor_observation_lifetime"]["lifetime_summary"], "thresholds": behavior["thresholds"], "threshold_source": behavior["threshold_source"], "acceptance_checks": checks, "provenance": {"git": provenance, "checkpoint_sha256": sha256_file(config["paths"]["network"]), "config_sha256": config_hash, "resolved_config": config, "reference": reference}}
    atomic_json_dump(report / "summary.json", summary)
    (report / "REPORT.md").write_text(sequence_report(summary, metrics), encoding="utf-8")
    manifest = {"schema_version": SCHEMA_VERSION, "sequence": sequence, "git_commit": provenance["commit"], "config_sha256": config_hash, "checkpoint_sha256": sha256_file(config["paths"]["network"]), "input_provenance": {"data_csv": config["paths"]["data_csv"], "groundtruth": config["paths"]["groundtruth"]}, "sampling": {key: behavior[key] for key in ("observation_population_count", "observation_sample_count", "observation_bottom_k", "observation_sample_sha256")}, "frozen_thresholds": behavior["thresholds"], "reference": reference, "sanity_gate": sanity_gate, "probe_perturbation": perturbation, "trajectory_est_source": "uninstrumented baseline", "artifacts_compressed": True, "artifact_inventory": _artifact_inventory(output), "no_raw_directory": True, "automatic_zip_created": False}
    atomic_json_dump(report / "manifest.json", manifest)
    if not all(checks.values()): raise RuntimeError(f"{sequence} acceptance checks failed")
    return output


def sequence_report(summary: dict[str, Any], metrics: dict[str, Any]) -> str:
    return f"""# Experiment 1 — {summary['sequence']}

This standalone sequence report is a descriptive DPVO internal-behavior record. Headline ATE, FPS, peak VRAM and `trajectory_est.tum` come exclusively from the uninstrumented baseline; the probe supplies internal-state measurements and a non-blocking perturbation diagnostic.

## Protocol

The fixed protocol uses stride 2, seed 1234, `config/default.yaml` and original DPVO with loop closure disabled. Correlation follows the verified `[E,7,7,3,3,2]` layout with fixed T=1.0. Patch residency and factor-observation lifetimes are distinct graph-policy-dependent quantities.

## Headline

- Processed frames: {summary['frames_processed']}
- Baseline global Sim(3) ATE RMSE: {summary['trajectory']['baseline_translation_ate_rmse']:.6g} m
- Baseline FPS / peak VRAM: {summary['headline_performance']['fps']:.6g} / {summary['headline_performance']['peak_allocated_bytes']} bytes
- Probe perturbation: {summary['probe_perturbation']['perturbation_label']} ({summary['probe_perturbation']['relative_ate_change']!s} relative ATE change); representativeness: {summary['probe_perturbation']['probe_representativeness']}

`moderate` perturbation requires caution. `large` perturbation retains the probe artifact but gives it low representativeness; it must not be used as strong mechanism-level evidence. This diagnostic is not a numerical acceptance gate.

Difficulty proxies and internal-state/trajectory associations are descriptive only. Apparent reprojection motion is model-derived; blur, texture and GT rotation are proxies rather than failure labels. No causal conclusion or Experiment 2 decision is made by this single-sequence report.
"""


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _load_frame(record: dict[str, Any], calibration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(record["image_path"])
    if image is None: raise FileNotFoundError(record["image_path"])
    fx, fy, cx, cy = calibration[:4]
    if len(calibration) > 4:
        matrix = np.eye(3); matrix[0, 0] = fx; matrix[1, 1] = fy; matrix[0, 2] = cx; matrix[1, 2] = cy
        image = cv2.undistort(image, matrix, calibration[4:])
    height, width, _ = image.shape
    return image[:height-height % 16, :width-width % 16], np.asarray([fx, fy, cx, cy])


def _trajectory(poses: np.ndarray, timestamps: np.ndarray) -> PoseTrajectory3D:
    return PoseTrajectory3D(positions_xyz=poses[:, :3], orientations_quat_wxyz=poses[:, [6, 3, 4, 5]], timestamps=timestamps.astype(np.float64))


def _ate(trajectory: PoseTrajectory3D, groundtruth: str) -> dict[str, Any]:
    reference = file_interface.read_tum_trajectory_file(groundtruth); reference, estimate = sync.associate_trajectories(reference, trajectory)
    result = main_ape.ape(reference, estimate, est_name="trajectory", pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
    return {"translation_rmse": float(result.stats["rmse"]), "associated_pose_count": int(estimate.num_poses), "alignment": "Sim(3)"}


def _configure_dpvo(config: dict[str, Any]) -> Any:
    from dpvo.config import cfg as base_cfg
    cfg = base_cfg.clone(); cfg.merge_from_file(config["paths"]["dpvo_config"]); cfg.LOOP_CLOSURE = False; cfg.CLASSIC_LOOP_CLOSURE = False
    return cfg


@torch.no_grad()
def worker(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from dpvo.dpvo import DPVO
    from .behavior import compute_image_difficulty
    from .probe import DPVOTruthProbe
    seed_everything(int(config["experiment"]["seed"])); records = load_euroc_records(config); calibration = np.loadtxt(config["paths"]["calib"], delimiter=" ")
    first, _ = _load_frame(records[0], calibration); slam = DPVO(_configure_dpvo(config), config["paths"]["network"], ht=first.shape[0], wd=first.shape[1], viz=False)
    is_probe = args.mode == "probe"; probe = DPVOTruthProbe(slam, config, args.events_path) if is_probe else None
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); events: list[tuple[Any, Any]] = []; cpu_seconds = 0.; poses = internal = None
    try:
        for record in records:
            image_np, intrinsic_np = _load_frame(record, calibration)
            blur, texture = compute_image_difficulty(image_np) if is_probe else (None, None)
            image, intrinsics = torch.from_numpy(image_np).permute(2, 0, 1).cuda(), torch.from_numpy(intrinsic_np).cuda()
            if probe: probe.before_frame(stream_index=int(record["stream_index"]), euroc_timestamp_ns=int(record["euroc_timestamp_ns"]), image_height=int(image.shape[1]), image_width=int(image.shape[2]), blur_laplacian=blur, texture_gradient=texture)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); start.record(); now = time.perf_counter(); slam(int(record["stream_index"]), image, intrinsics); cpu_seconds += time.perf_counter()-now; end.record(); events.append((start,end))
            if probe: probe.after_frame()
        if probe: probe.before_terminate()
        terminate_start, terminate_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); terminate_start.record(); poses, internal = slam.terminate(); terminate_end.record()
        if probe: probe.after_terminate()
        torch.cuda.synchronize()
    finally:
        if probe: probe.close();
    timestamps = np.asarray([item["euroc_timestamp_ns"] for item in records], dtype=np.uint64)
    if poses is None or len(poses) != len(timestamps) or not np.array_equal(np.asarray(internal), np.arange(len(records), dtype=np.float64)): raise AssertionError("DPVO trajectory timestamp contract failed")
    trajectory = _trajectory(poses, timestamps)
    if args.trajectory_path: Path(args.trajectory_path).parent.mkdir(parents=True, exist_ok=True); file_interface.write_tum_trajectory_file(args.trajectory_path, trajectory)
    behavior = probe.write_behavior_outputs(args.behavior_output, reference_thresholds=config.get("behavior", {}).get("reference_thresholds")) if probe else None
    if behavior: behavior["temporary_threshold_spool"]["cleaned"] = bool(probe.behavior.spool_cleaned)
    frame_ms = sum(start.elapsed_time(end) for start, end in events); properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    np.savez_compressed(args.result_npz, poses=np.asarray(poses, dtype=np.float64), internal_tstamps=np.asarray(internal, dtype=np.float64), euroc_timestamps_ns=timestamps)
    return {"status": "ok", "mode": args.mode, "frame_count": len(records), "trajectory_pose_count": len(poses), "ate": _ate(trajectory, config["paths"]["groundtruth"]), "runtime": {"fps_excluding_decode_h2d_and_terminate": len(records)/(frame_ms/1000.), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "cpu_enqueue_seconds": cpu_seconds, "headline_performance_eligible": not is_probe}, "probe": probe.summary() if probe else None, "behavior": behavior, "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": properties.name}}


def internal_worker_main(args: argparse.Namespace) -> int:
    try:
        result = worker(load_json(args.config_json), args); atomic_json_dump(args.result_json, result); return 0
    except Exception as error:
        atomic_json_dump(args.result_json, {"status": "error", "mode": args.mode, "error": repr(error), "traceback": traceback.format_exc()}); return 1


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False); parser.add_argument("--internal-worker", action="store_true"); parser.add_argument("--config-json"); parser.add_argument("--mode", choices=("baseline", "probe")); parser.add_argument("--result-json"); parser.add_argument("--result-npz"); parser.add_argument("--trajectory-path"); parser.add_argument("--events-path"); parser.add_argument("--behavior-output")
    args = parser.parse_args()
    if not args.internal_worker: raise SystemExit("runtime is internal; use python -m research.src.phase1_dpvo_feasibility.run_exp1")
    return internal_worker_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
