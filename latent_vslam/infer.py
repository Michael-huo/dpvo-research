"""Formal four-mode inference and stride sweeps."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np

from latent_vslam.stride_sweep import (DEFAULT_CONFIG, DEFAULT_STRIDES, TRAINING_ROOT,
                            stride_config, load_protocol,
                            population_summary, prepare_protocol)
from latent_vslam.inference_results import comparison_row, publish_index, save_trajectory
from latent_vslam.stride_training import load_stride_predictor
from latent_vslam.artifact_runtime import staged_directory, publish_current_canonical
from latent_vslam.execution_runtime import (FormalExecution, cpu_numa_layout, fixed_cpu_profile,
                                initialize_formal_main_process, execution_provenance,
                                release_cuda_training_state, require_lifecycle_cleanup)
from latent_vslam.parallel_runtime import run_sequential_trajectory_jobs
from latent_vslam.protocol import (atomic_write_json, load_sequence_records, repo_path,
                       sha256_file)
from latent_vslam.dataset_paths import groundtruth_path
from latent_vslam.inference_runtime import load_config as load_inference_config, run as run_four_modes
from prediction.predictor_checkpoint import load_canonical_predictor
from latent_vslam.inference_runtime import _load_bridge
from prediction.jepa_runtime import sequence_geometry
from latent_vslam.inference_runtime import _repository_provenance


MODES = {
    "full_rgb": "full_rgb_reference",
    "sparse_rgb": "sparse_rgb_reference",
    "oracle_jepa": "oracle_jepa_hidden_reference",
    "ours": "predicted_jepa_hidden",
}


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}; train or restore it before infer")
    return path


def preflight_inference(config):
    """Reject missing model files before initializing any formal GPU runtime."""
    require_file(repo_path(config["paths"]["dpvo_checkpoint"]), "DPVO checkpoint")
    require_file(repo_path(config["paths"]["bridge"]), "Bridge checkpoint")
    require_file(Path(config["jepa"]["checkpoint"]), "V-JEPA checkpoint")
    require_file(repo_path(config["paths"]["predictor"]), "Predictor checkpoint")
    load_canonical_predictor(config)


def preflight_stride_sweep(protocol, canonical, config_path, requested_strides=None):
    require_file(repo_path(canonical["paths"]["dpvo_checkpoint"]), "DPVO checkpoint")
    bridge = require_file(repo_path(canonical["paths"]["bridge"]), "Bridge checkpoint")
    require_file(Path(canonical["jepa"]["checkpoint"]), "V-JEPA checkpoint")
    manifest_path = require_file(TRAINING_ROOT / "manifest.json", "stride training manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    strides = tuple(protocol["anchor_strides"] if requested_strides is None else requested_strides)
    if (manifest.get("schema_version") != 1
            or manifest.get("kind") != "stride_predictor_training"
            or manifest.get("sequence") != protocol["sequence"]
            or manifest.get("anchor_strides") != list(strides)
            or manifest.get("config_sha256") != sha256_file(config_path)
            or manifest.get("bridge_sha256") != sha256_file(bridge)):
        raise RuntimeError("stride training manifest is incompatible with infer config/Bridge")
    for stride in strides:
        manifest_row = manifest.get("strides", {}).get(str(stride))
        if not isinstance(manifest_row, dict):
            raise RuntimeError(f"stride {stride} training metadata is missing")
        source = TRAINING_ROOT / f"stride_{stride}"
        checkpoint = require_file(source / "predictor.pt", f"stride {stride} Predictor checkpoint")
        training = require_file(source / "training.json", f"stride {stride} training record")
        horizon = require_file(source / "horizon_queries.json", f"stride {stride} horizon records")
        if sha256_file(checkpoint) != manifest_row.get("checkpoint_sha256"):
            raise RuntimeError(f"stride {stride} Predictor checkpoint changed")
        if (sha256_file(training) != manifest_row.get("training_sha256")
                or sha256_file(horizon) != manifest_row.get("horizon_sha256")):
            raise RuntimeError(f"stride {stride} training artifacts changed")
        record = json.loads(training.read_text(encoding="utf-8"))
        if (record["bridge"]["file_sha256"] != manifest["bridge_sha256"]
                or record["lineage"]["training_lineage_sha256"] != manifest_row.get("training_lineage_sha256")
                or record["stores_closed_before_deployment"] is not True):
            raise RuntimeError(f"stride {stride} training record is incompatible")
        load_stride_predictor(
            checkpoint, stride_config(canonical, protocol, stride),
            record["lineage"], stride,
        )
    return manifest


def _run_job(task, temporary, runtime, profile):
    # A blocking call with exactly one job; never launch two SLAM processes.
    rows, execution = run_sequential_trajectory_jobs(
        [task], temporary, cpu_profile=profile["components"]["stage_c"],
        hardware=runtime["hardware"])
    if len(rows) != 1 or execution["maximum_concurrent_dpvo_instances"] != 1:
        raise RuntimeError("independent sequential DPVO execution required")
    return rows[0]


def predicted_task(common, budget, trained, config, profile):
    path = trained["checkpoint_path"]
    lineage = trained["record"]["lineage"]
    checkpoint = load_stride_predictor(path, config, lineage, budget["anchor_stride"])
    if trained["record"]["stores_closed_before_deployment"] is not True:
        raise RuntimeError("offline feature stores must close before Ours")
    return {"kind": "infer_ours", **common,
            "intervals": budget["intervals"], "transform": trained["transform"],
            "thresholds": checkpoint["train_only_calibration"],
            "bridge_state": trained["bridge_state"],
            "predictor_state": checkpoint["state_dict"],
            "predictor_checkpoint": str(path),
            "predictor_state_hash": checkpoint["state_dict_sha256"],
            "pipeline_cpu_profile": profile, "execution": dataclasses.asdict(FormalExecution.from_pool())}


def execute_sequence(records, budgets, protocol, canonical, root, temporary, index,
                     training_manifest):
    runtime = initialize_formal_main_process(require_online=True)
    profile = {"schema": "inference_fixed_cpu_profile_v1", "selection": "fixed_cpu_assignment",
               "dynamic_calibration": False,
               "components": fixed_cpu_profile(cpu_numa_layout(runtime["hardware"]))}
    index["execution"] = {"runtime": runtime, "profile": profile,
                          "provenance": execution_provenance(),
                          "maximum_concurrent_dpvo_instances": 1,
                          "measurement_order": ["full_rgb", *[
                              f"stride_{s}/{c}" for s in index["anchor_strides"]
                              for c in ("sparse_rgb", "predicted_jepa")]]}
    sequence_root = root / "sequences" / protocol["sequence"]
    sequence_root.mkdir(parents=True, exist_ok=True)
    calibration = np.loadtxt(repo_path(canonical["paths"]["calibration"]))
    gt = groundtruth_path(protocol["sequence"])
    atomic_write_json(sequence_root / "groundtruth.json", {
        "role": "reference_not_slam_run", "path": str(gt), "sha256": sha256_file(gt)})
    index["groundtruth_ref"] = str((sequence_root / "groundtruth.json").relative_to(root))
    canonical_budget = budgets[5]
    full_records = canonical_budget["records"]
    full = _run_job({"kind": "infer_full_rgb", "records": full_records,
                     "roles": {r.identity.key: "anchor" for r in full_records},
                     "calibration": calibration, "config": canonical},
                    temporary / "full_rgb", runtime, profile)
    full_result = save_trajectory(sequence_root / "full_rgb", full, full_records,
                                  canonical_budget["roles"], canonical, dense=True)
    index["full_rgb"] = {"results": str((sequence_root / "full_rgb/results.json").relative_to(root)),
                         "ate_rmse_m": (full_result["canonical_evaluation"] or {}).get("ate_rmse_m"),
                         "matched_trajectory_wall_seconds": full_result["matched_trajectory_wall_seconds"],
                         "graph_workload": full_result["graph_workload"]}
    del full
    publish_index(root, index)
    (root / "models").mkdir()
    for stride in index["anchor_strides"]:
        budget = budgets[stride]
        config = stride_config(canonical, protocol, stride)
        model_root = root / "models" / f"stride_{stride}"
        shutil.copytree(TRAINING_ROOT / f"stride_{stride}", model_root,
                        ignore=shutil.ignore_patterns("predictor.pt"))
        training_record = json.loads((model_root / "training.json").read_text(encoding="utf-8"))
        transform, _ = sequence_geometry(records[0], calibration, config)
        bridge, _ = _load_bridge(config, transform)
        bridge_state = {name: value.detach().cpu().clone()
                        for name, value in bridge.state_dict().items()}
        bridge.cpu()
        del bridge
        require_lifecycle_cleanup(release_cuda_training_state())
        trained = {"checkpoint_path": TRAINING_ROOT / f"stride_{stride}" / "predictor.pt",
                   "record": training_record, "transform": transform,
                   "bridge_state": bridge_state}
        cleanup = training_manifest["strides"][str(stride)]["cleanup"]
        output = sequence_root / f"stride_{stride}"
        output.mkdir(parents=True, exist_ok=False)
        population = population_summary(budget)
        atomic_write_json(output / "schedule.json", population | {
            "roles": budget["roles"], "intervals": [i.payload() for i in budget["intervals"]]})
        common = {"records": budget["records"], "roles": budget["roles"],
                  "calibration": calibration, "config": config}
        sparse_row = _run_job({"kind": "infer_sparse_rgb", **common},
                              temporary / f"sparse_{stride}", runtime, profile)
        sparse = save_trajectory(output / "sparse_rgb", sparse_row, budget["records"],
                                  budget["roles"], config, dense=False)
        del sparse_row
        task = predicted_task(common, budget, trained, config, profile)
        ours_row = _run_job(task, temporary / f"predicted_{stride}", runtime, profile)
        ours = save_trajectory(output / "predicted_jepa", ours_row, budget["records"],
                                budget["roles"], config, dense=True)
        from latent_vslam.scientific_lineage import deployment_lineage
        ours["deployment_lineage"] = deployment_lineage(
            config, predictor_sha256=task["predictor_state_hash"],
            training_lineage_sha256=trained["record"]["lineage"]["training_lineage_sha256"],
            bridge_sha256=trained["record"]["bridge"]["file_sha256"],
            dpvo_sha256=sha256_file(repo_path(config["paths"]["dpvo_checkpoint"])),
            dpvo_config_sha256=sha256_file(repo_path(config["paths"]["dpvo_config"])),
            schedule_sha256=budget["source_schedule"]["schedule_sha256"],
        )
        atomic_write_json(output / "predicted_jepa/results.json", ours)
        del ours_row, task
        comparison = comparison_row(population, sparse, ours, trained["record"])
        atomic_write_json(output / "results.json", {
            "anchor_stride": stride, "population": population, "comparison": comparison,
            "groundtruth_ref": "../groundtruth.json", "full_rgb_ref": "../full_rgb/results.json",
            "predictor_ref": str(trained["checkpoint_path"].relative_to(repo_path("."))),
            "predictor_path_base": "repository_root",
            "predictor_sha256": sha256_file(trained["checkpoint_path"]),
            "training_ref": str((model_root / "training.json").relative_to(root)),
            "sparse_rgb_ref": "sparse_rgb/results.json", "ours_ref": "predicted_jepa/results.json",
            "training_cleanup": cleanup})
        index["strides"][str(stride)] = str((output / "results.json").relative_to(root))
        index["model_manifest"]["predictors"][str(stride)] = {
            "checkpoint": str(trained["checkpoint_path"].relative_to(repo_path("."))),
            "checkpoint_sha256": sha256_file(trained["checkpoint_path"]),
            "training_lineage_sha256": trained["record"]["lineage"]["training_lineage_sha256"],
        }
        index["deployment_manifest"][str(stride)] = ours["deployment_lineage"]
        index["accuracy_vs_communication"].append(comparison)
        index["prediction_quality_vs_horizon"].extend(trained["record"]["horizon_resolved_quality"])
        del trained
        publish_index(root, index)


def run_stride_sweep(sequence="MH_01_easy", strides=DEFAULT_STRIDES, *, config_path=DEFAULT_CONFIG):
    protocol, canonical, config_path = load_protocol(config_path)
    strides = tuple(strides)
    if sequence != protocol["sequence"]:
        raise ValueError("configured sweep sequence is MH_01_easy")
    if not strides or len(strides) != len(set(strides)) or any(
            isinstance(s, bool) or not isinstance(s, int) or s < 2 for s in strides):
        raise ValueError("choose distinct integer strides >= 2")
    root = repo_path(protocol["output_root"])
    training_manifest = preflight_stride_sweep(protocol, canonical, config_path, strides)
    from latent_vslam.cuda_devices import CudaDevicePool
    CudaDevicePool.discover().online_mapping()
    records = load_sequence_records(canonical, sequence)
    budgets, preparation = prepare_protocol(records, protocol, strides)
    index = {"schema_version": 1, "run_policy": "fresh_current_canonical_replace", "protocol": "stride_sweep_fresh_predictor_v1",
             "sequence": sequence, "anchor_strides": list(strides),
             "trajectory_comparison": ["GT", "Full RGB", "Sparse RGB", "Ours / Predicted JEPA"],
             "status": "running",
             "config": str(config_path), "config_sha256": sha256_file(config_path),
             "preparation": preparation, "strides": {},
             "model_manifest": {"bridge_sha256": training_manifest["bridge_sha256"],
                                "predictors": {}},
             "deployment_manifest": {}, "accuracy_vs_communication": [],
             "prediction_quality_vs_horizon": []}
    # All intermediate models/worker/diagnostic files disappear on either outcome.
    with staged_directory(root) as temporary:
        staged = temporary / "artifacts"
        staged.mkdir()
        index["run_manifest"] = {"repository": _repository_provenance()}
        publish_index(staged, index)
        bridge = repo_path(canonical["paths"]["bridge"])
        bridge_hash = sha256_file(bridge)
        execute_sequence(records, budgets, protocol, canonical, staged,
                         temporary / "workers", index, training_manifest)
        index["status"] = "complete"
        if sha256_file(bridge) != bridge_hash:
            raise RuntimeError("Bridge checkpoint changed during stride sweep")
        publish_index(staged, index)
        publish_current_canonical(staged, root)
    return index


def run(config_path: str | Path) -> dict:
    from latent_vslam.canonical import load_yaml
    config, resolved = load_yaml(config_path)
    if config.get("experiment", {}).get("name") == "four_mode_inference":
        validated, _ = load_inference_config(resolved)
        preflight_inference(validated)
        result = run_four_modes(validated["experiment"]["default_sequences"], config_path=resolved)
        return result | {"infer_modes": list(MODES)}
    if "anchor_strides" in config:
        return run_stride_sweep(config["sequence"], tuple(config["anchor_strides"]),
                                config_path=resolved)
    raise ValueError("infer config must define four-mode inference or a stride sweep")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True, type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
