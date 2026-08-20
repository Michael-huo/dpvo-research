"""Single public entry point for the fresh final Experiment 3 reproduction."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from evo.tools import file_interface

from . import SCHEMA_VERSION
from .analyze import SPARSE_TRAJECTORY_NOTE, ate, common_accepted, make_trajectory_figure, recovery_metrics, trajectory


FINAL_SEQUENCES = ("MH_01_easy", "MH_03_medium", "MH_05_difficult")
FINAL_METHODS = ("full_rgb", "sparse_rgb", "oracle_fmap")
FINAL_INTERVAL = 8
FINAL_DECISION = "PARTIAL / CONDITIONAL UPPER-BOUND EVIDENCE"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_run_id(sequence: str, method: str) -> str:
    return f"{sequence}__{method}" + ("__k8" if method != "full_rgb" else "")


def final_run_matrix() -> list[dict[str, Any]]:
    rows = [{"run_id": final_run_id(sequence, method), "sequence": sequence, "method": method, "anchor_interval": None if method == "full_rgb" else FINAL_INTERVAL} for sequence in FINAL_SEQUENCES for method in FINAL_METHODS]
    if len(rows) != 9 or len({row["run_id"] for row in rows}) != 9:
        raise AssertionError("final matrix must contain exactly three sequences times three methods")
    return rows


def build_anchor_schedule(total_candidates: int, bootstrap_end: int, anchor_interval: int = FINAL_INTERVAL) -> tuple[int, ...]:
    if total_candidates <= 0 or not 0 <= bootstrap_end < total_candidates or anchor_interval <= 0:
        raise ValueError("invalid external anchor schedule inputs")
    return tuple(range(bootstrap_end + 1)) + tuple(range(bootstrap_end + 1, total_candidates, anchor_interval))


def resolve_config(root: Path) -> dict[str, Any]:
    source = root / "research/configs/phase1_exp3.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    if tuple(experiment["sequences"]) != FINAL_SEQUENCES:
        raise ValueError("final Experiment 3 sequence protocol changed")
    if int(experiment["post_bootstrap_anchor_interval"]) != FINAL_INTERVAL or float(experiment["nominal_rgb_upload_ratio"]) != .125:
        raise ValueError("final Experiment 3 K=8 protocol changed")
    config["repo_root"] = str(root)
    config["paths"] = {name: str((root / path).resolve()) for name, path in config["paths"].items()}
    config["paths"]["config_source"] = str(source)
    return config


def sequence_config(base: dict[str, Any], sequence: str) -> dict[str, Any]:
    config = copy.deepcopy(base)
    root = Path(config["repo_root"])
    image_dir = Path(config["paths"]["euroc_root"]) / sequence / "mav0/cam0/data"
    images = sorted(image_dir.glob("*.png"))[int(config["experiment"]["skip"])::int(config["experiment"]["stride"])]
    if not images:
        raise RuntimeError(f"no images for {sequence}")
    config["experiment"].update({"sequence": sequence, "processed_frames": len(images), "post_bootstrap_anchor_interval": FINAL_INTERVAL})
    config["paths"].update({"image_dir": str(image_dir), "data_csv": str(image_dir.parent / "data.csv"), "groundtruth": str(root / "datasets/euroc_groundtruth" / f"{sequence}.txt")})
    required = ("image_dir", "data_csv", "groundtruth", "calibration", "checkpoint", "dpvo_config")
    missing = [name for name in required if not Path(config["paths"][name]).is_file() and name not in {"image_dir"}]
    if not image_dir.is_dir() or missing:
        raise FileNotFoundError(f"missing final Experiment 3 assets for {sequence}: image_dir={image_dir}, files={missing}")
    return config


def _repository_state(root: Path) -> dict[str, Any]:
    status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).strip()
    dpvo_diff = subprocess.check_output(["git", "diff", "--", "dpvo"], cwd=root, text=True)
    return {"git_status": status, "dpvo_diff_empty": not dpvo_diff.strip()}


def _preflight(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not _repository_state(root)["dpvo_diff_empty"]:
        raise RuntimeError("git diff -- dpvo must be empty")
    import torch
    import cuda_ba  # noqa: F401
    import dpvo.altcorr  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; final Experiment 3 execution stopped")
    for key in ("checkpoint", "dpvo_config", "calibration"):
        if not Path(config["paths"][key]).is_file():
            raise FileNotFoundError(config["paths"][key])
    return {"python": sys.executable, "cuda_device": torch.cuda.get_device_name(0), "checkpoint_sha256": _sha256(config["paths"]["checkpoint"]), "dpvo_config_sha256": _sha256(config["paths"]["dpvo_config"])}


def _run_worker(root: Path, config_path: Path, mode: str, temporary: Path, schedule_path: Path | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result_json, result_npz = temporary / f"{mode}.json", temporary / f"{mode}.npz"
    command = [sys.executable, "-m", "research.src.phase1_dpvo_feasibility.exp3.runtime", "--internal-worker", "--config-json", str(config_path), "--mode", mode, "--result-json", str(result_json), "--result-npz", str(result_npz)]
    if schedule_path is not None:
        command.extend(["--schedule-json", str(schedule_path)])
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
    if not result_json.is_file():
        raise RuntimeError(f"worker exited without diagnostics ({mode}): {completed.stderr[-2000:]}")
    result = json.loads(result_json.read_text(encoding="utf-8"))
    if completed.returncode or result.get("status") != "ok":
        raise RuntimeError(f"worker failed ({mode}): {result.get('error', completed.stderr[-2000:])}")
    arrays = dict(np.load(result_npz))
    return result, arrays


def _write_tum(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_interface.write_tum_trajectory_file(path, trajectory(arrays["poses"], arrays["timestamps_ns"]))


def oracle_integrity(sparse: dict[str, Any], oracle: dict[str, Any]) -> dict[str, bool]:
    diagnostics = oracle["diagnostics"]
    latent = int(diagnostics["latent_frame_count"])
    return {
        "tracking_complete": bool(sparse["tracking_completed"] and oracle["tracking_completed"]),
        "finite_trajectories": bool(sparse["finite_trajectory"] and oracle["finite_trajectory"]),
        "valid_timestamps": bool(sparse["valid_timestamps"] and oracle["valid_timestamps"]),
        "shared_schedule_exact": sparse["scheduled_anchor_candidate_indices"] == oracle["scheduled_anchor_candidate_indices"],
        "bootstrap_end_exact": sparse["bootstrap_end_candidate_index"] == oracle["bootstrap_end_candidate_index"],
        "shared_no_culling_policy": sparse["culling_policy"] == oracle["culling_policy"] == "no_keyframe_cull_with_upstream_factor_retirement",
        "latent_state_coverage": latent > 0 and diagnostics["latent_frames_with_state_node"] == latent,
        "latent_factor_coverage": latent > 0 and diagnostics["latent_frames_receiving_factors"] == latent,
        "correlation_coverage": diagnostics["correlation"]["latent_frames"] == latent,
        "update_coverage": diagnostics["update"]["latent_frames"] == latent,
        "ba_coverage": diagnostics["ba"]["latent_frames"] == latent and diagnostics["ba"]["successful_calls_with_latent"] > 0,
        "zero_latent_source_factors": diagnostics["latent_source_factor_count"] == 0,
        "zero_placeholder_violations": all(value == 0 for value in diagnostics["placeholder_reference_violations"].values()),
    }


def _report(metrics: dict[str, Any]) -> str:
    lines = ["# Phase 1 / Experiment 3 — Final Representative Reproduction", "", "## Decision", "", f"**{FINAL_DECISION}**", "", "K=8 is a severe-sparsity representative setting, not a tuned optimum.", "", SPARSE_TRAJECTORY_NOTE, "", "## Results", "", "| Sequence | Actual RGB upload | Full dense ATE | Full anchor ATE | Sparse anchor ATE | Oracle anchor ATE | Sparse degradation | Oracle recovery | Relative improvement | Gap recovery |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for sequence in FINAL_SEQUENCES:
        row, evaluation = metrics["sequences"][sequence], metrics["sequences"][sequence]["evaluation"]
        gap = "N/A" if evaluation["gap_recovery"] is None else f"{evaluation['gap_recovery']:.6g}"
        lines.append(f"| {sequence} | {row['actual_rgb_upload_ratio']:.3%} | {evaluation['full_dense_ate']['translation_rmse']:.6g} | {evaluation['full_anchor_ate']['translation_rmse']:.6g} | {evaluation['sparse_anchor_ate']['translation_rmse']:.6g} | {evaluation['oracle_anchor_ate']['translation_rmse']:.6g} | {evaluation['sparse_degradation_abs']:.6g} | {evaluation['oracle_recovery_abs']:.6g} | {evaluation['oracle_relative_improvement']:.3%} | {gap} |")
    lines += ["", "## Interpretation boundary", "", "Full RGB uses upstream normal culling. Sparse RGB and Sparse + Oracle FMap share the research-only no-culling policy and source-factor retirement; their comparison is the causal comparison. An Oracle ATE below Full RGB or gap recovery above one is not evidence that a latent observation is stronger than RGB because graph policies differ.", "", "Earlier smoke, formal, and auxiliary work remains research history in `RESEARCH.md`; it is neither loaded nor reused by this reproduction.", "", "No feature cache, raw tensor dump, archive, extra ratio, fallback, JEPA run, topology ablation, or Experiment 4 implementation was produced.", ""]
    return "\n".join(lines)


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "status": metrics["status"], "decision": FINAL_DECISION, "protocol": metrics["protocol"], "run_matrix": metrics["run_matrix"], "completed_method_ids": metrics["completed_method_ids"], "sequences": {sequence: {key: row[key] for key in ("candidate_count", "actual_rgb_upload_ratio", "anchor_contract", "evaluation", "oracle_integrity")} for sequence, row in metrics["sequences"].items()}, "sparse_trajectory_semantics": SPARSE_TRAJECTORY_NOTE, "interpretation_boundary": "Sparse RGB versus Sparse + Oracle FMap is policy-matched; Full uses upstream culling."}


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in sorted(root.rglob("*")) if path.is_file() and path.name != "manifest.json"]


def _publish(staging: Path, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    staging.replace(output)


def main() -> int:
    root, base = repo_root(), resolve_config(repo_root())
    output = Path(base["paths"]["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".exp3-staging-", dir=output.parent))
    metrics: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "status": "running", "decision": FINAL_DECISION, "protocol": base["experiment"], "run_matrix": final_run_matrix(), "completed_method_ids": [], "sequences": {}}
    try:
        environment = _preflight(root, sequence_config(base, FINAL_SEQUENCES[0]))
        with tempfile.TemporaryDirectory(prefix="phase1-exp3-final-") as temporary_name:
            temporary = Path(temporary_name)
            for sequence in FINAL_SEQUENCES:
                config = sequence_config(base, sequence)
                sequence_dir = temporary / sequence
                sequence_dir.mkdir()
                config_path = sequence_dir / "config.json"
                _atomic_json(config_path, config)
                full, full_arrays = _run_worker(root, config_path, "full_rgb", sequence_dir)
                metrics["completed_method_ids"].append(final_run_id(sequence, "full_rgb"))
                _atomic_json(staging / "artifacts/metrics.json", metrics)
                sparse, sparse_arrays = _run_worker(root, config_path, "sparse_rgb", sequence_dir)
                metrics["completed_method_ids"].append(final_run_id(sequence, "sparse_rgb"))
                bootstrap_end = int(sparse["bootstrap_end_candidate_index"])
                expected = build_anchor_schedule(int(config["experiment"]["processed_frames"]), bootstrap_end)
                if tuple(sparse["scheduled_anchor_candidate_indices"]) != expected:
                    raise AssertionError(f"Sparse schedule violated K=8 external schedule for {sequence}")
                schedule = {"source": "shared_external_candidate_ordinal", "immutable": True, "candidate_indices": list(expected), "timestamps_ns": sparse["scheduled_anchor_timestamps_ns"], "bootstrap_end_candidate_index": bootstrap_end, "anchor_interval": FINAL_INTERVAL}
                schedule_path = sequence_dir / "shared_schedule.json"
                _atomic_json(schedule_path, schedule)
                oracle, oracle_arrays = _run_worker(root, config_path, "oracle_fmap", sequence_dir, schedule_path)
                metrics["completed_method_ids"].append(final_run_id(sequence, "oracle_fmap"))
                if oracle["bootstrap_end_candidate_index"] != bootstrap_end:
                    raise AssertionError(f"Oracle bootstrap mismatch for {sequence}")
                anchors = common_accepted(sparse, oracle)
                common = anchors["common_accepted_anchor_timestamps_ns"]
                if not common:
                    raise AssertionError(f"no common accepted anchors for {sequence}")
                evaluation = {"full_dense_ate": ate(full_arrays, full_arrays["timestamps_ns"], config["paths"]["groundtruth"]), "full_anchor_ate": ate(full_arrays, common, config["paths"]["groundtruth"]), "sparse_anchor_ate": ate(sparse_arrays, common, config["paths"]["groundtruth"]), "oracle_anchor_ate": ate(oracle_arrays, common, config["paths"]["groundtruth"]), "oracle_dense_ate": ate(oracle_arrays, oracle_arrays["timestamps_ns"], config["paths"]["groundtruth"])}
                evaluation.update(recovery_metrics(evaluation["full_anchor_ate"]["translation_rmse"], evaluation["sparse_anchor_ate"]["translation_rmse"], evaluation["oracle_anchor_ate"]["translation_rmse"]))
                integrity = oracle_integrity(sparse, oracle)
                if not all(integrity.values()):
                    raise RuntimeError(f"Oracle structural checks failed for {sequence}: {integrity}")
                row = {"candidate_count": int(config["experiment"]["processed_frames"]), "anchor_interval": FINAL_INTERVAL, "nominal_rgb_upload_ratio": .125, "actual_rgb_upload_ratio": sparse["actual_rgb_upload_ratio"], "shared_external_schedule": schedule, "anchor_contract": anchors, "methods": {"full_rgb": full, "sparse_rgb": sparse, "oracle_fmap": oracle}, "evaluation": evaluation, "oracle_integrity": integrity}
                metrics["sequences"][sequence] = row
                trajectory_dir = staging / "artifacts/trajectories" / sequence
                _write_tum(trajectory_dir / "full_rgb.tum", full_arrays)
                _write_tum(trajectory_dir / "sparse_rgb_k8.tum", sparse_arrays)
                _write_tum(trajectory_dir / "oracle_fmap_k8.tum", oracle_arrays)
                make_trajectory_figure(staging / "report/figures" / f"trajectory_{sequence}.png", {"full_rgb": full_arrays, "sparse_rgb": sparse_arrays, "oracle_fmap": oracle_arrays}, common, config["paths"]["groundtruth"], sequence=sequence, actual_upload_ratio=sparse["actual_rgb_upload_ratio"], full_anchor_ate=evaluation["full_anchor_ate"]["translation_rmse"], sparse_anchor_ate=evaluation["sparse_anchor_ate"]["translation_rmse"], oracle_anchor_ate=evaluation["oracle_anchor_ate"]["translation_rmse"], gap_recovery=evaluation["gap_recovery"])
                _atomic_json(staging / "artifacts/metrics.json", metrics)
        if len(metrics["completed_method_ids"]) != 9:
            raise AssertionError("final representative reproduction did not execute nine methods")
        repository = _repository_state(root)
        if not repository["dpvo_diff_empty"]:
            raise RuntimeError("git diff -- dpvo changed during final reproduction")
        metrics.update({"status": "complete", "environment": environment, "repository": repository})
        _atomic_json(staging / "artifacts/metrics.json", metrics)
        _atomic_json(staging / "report/summary.json", _summary(metrics))
        (staging / "report/REPORT.md").write_text(_report(metrics), encoding="utf-8")
        manifest = {"schema_version": SCHEMA_VERSION, "status": "complete", "decision": FINAL_DECISION, "run_matrix": final_run_matrix(), "fresh_execution": True, "historical_artifacts_loaded": False, "checkpoint_sha256": environment["checkpoint_sha256"], "dpvo_config_sha256": environment["dpvo_config_sha256"], "exp3_config_sha256": _sha256(base["paths"]["config_source"]), "shared_schedule_contract": "Sparse bootstrap determines one immutable K=8 schedule, then Oracle replays that RGB bootstrap and receives the same schedule.", "no_feature_cache": True, "no_raw_tensor_dump": True, "no_archive": True, "repository": repository, "artifact_inventory": _inventory(staging)}
        _atomic_json(staging / "artifacts/manifest.json", manifest)
        _publish(staging, output)
        print(f"Experiment 3 final representative reproduction complete: {output}")
        return 0
    except BaseException as error:
        metrics.update({"status": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        _atomic_json(staging / "artifacts/metrics.json", metrics)
        _atomic_json(staging / "report/summary.json", {"schema_version": SCHEMA_VERSION, "status": "failed", "decision": None, "completed_method_ids": metrics["completed_method_ids"], "error": repr(error)})
        (staging / "report/REPORT.md").write_text(f"# Experiment 3 final reproduction failed\n\nCompleted methods: {len(metrics['completed_method_ids'])}/9.\n\n`{error!r}`\n", encoding="utf-8")
        _atomic_json(staging / "artifacts/manifest.json", {"schema_version": SCHEMA_VERSION, "status": "failed", "completed_method_ids": metrics["completed_method_ids"], "error": repr(error), "artifact_inventory": _inventory(staging)})
        _publish(staging, output)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
