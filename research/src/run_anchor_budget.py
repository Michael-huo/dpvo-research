"""Anchor budget sensitivity pilot; preparation is CPU-only unless --execute is set."""

from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
from pathlib import Path

import numpy as np

from .anchor_budget import (DEFAULT_CONFIG, PILOT_STRIDES, budget_config, load_protocol,
                            population_summary, prepare_protocol)
from .anchor_budget_results import comparison_row, publish_index, save_trajectory
from .anchor_budget_artifacts import (CHECKPOINT_ROOT, publish_compact,
                                      repository_provenance)
from .anchor_budget_training import (fresh_train_budget, load_budget_predictor,
                                     release_training_before_trajectory)
from .artifact_runtime import staged_directory
from .execution_runtime import (FormalExecution, cpu_numa_layout, fixed_cpu_profile,
                                initialize_formal_main_process, execution_provenance)
from .parallel_runtime import run_sequential_trajectory_jobs
from .protocol import (atomic_write_json, load_sequence_records, repo_path,
                       sha256_file)


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
    checkpoint = load_budget_predictor(path, config, lineage, budget["anchor_stride"])
    if trained["record"]["stores_closed_before_deployment"] is not True:
        raise RuntimeError("offline feature stores must close before Ours")
    return {"kind": "formal_h2_predicted", **common,
            "intervals": budget["intervals"], "transform": trained["transform"],
            "thresholds": checkpoint["train_only_calibration"],
            "bridge_state": trained["bridge_state"],
            "predictor_state": checkpoint["state_dict"],
            "predictor_checkpoint": str(path),
            "predictor_state_hash": checkpoint["state_dict_sha256"],
            "pipeline_cpu_profile": profile, "execution": dataclasses.asdict(FormalExecution.from_pool())}


def execute_sequence(records, budgets, protocol, canonical, root, temporary, index):
    runtime = initialize_formal_main_process(require_online=True)
    profile = {"schema": "research_h2_fixed_cpu_profile_v1", "selection": "fixed_same_as_h0_h1",
               "dynamic_calibration": False,
               "components": fixed_cpu_profile(cpu_numa_layout(runtime["hardware"]))}
    index["execution"] = {"runtime": runtime, "profile": profile,
                          "provenance": execution_provenance(),
                          "maximum_concurrent_dpvo_instances": 1,
                          "measurement_order": ["full_rgb", *[
                              f"stride_{s}/{c}" for s in index["anchor_strides"]
                              for c in ("fresh_train", "sparse_rgb", "predicted_jepa")]]}
    sequence_root = root / "sequences" / protocol["sequence"]
    sequence_root.mkdir(parents=True, exist_ok=True)
    calibration = np.loadtxt(repo_path(canonical["paths"]["calibration"]))
    gt = repo_path(canonical["dataset"]["groundtruth_pattern"].format(sequence=protocol["sequence"]))
    atomic_write_json(sequence_root / "groundtruth.json", {
        "role": "reference_not_slam_run", "path": str(gt), "sha256": sha256_file(gt)})
    index["groundtruth_ref"] = str((sequence_root / "groundtruth.json").relative_to(root))
    canonical_budget = budgets[5]
    full_records = canonical_budget["records"]
    full = _run_job({"kind": "formal_h2_full", "records": full_records,
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
    for stride in index["anchor_strides"]:
        budget = budgets[stride]
        config = budget_config(canonical, protocol, stride)
        model_root = root / "models" / f"stride_{stride}"
        trained = fresh_train_budget(records, budget, protocol, config, model_root,
                                     temporary / f"training_{stride}")
        cleanup = release_training_before_trajectory()
        output = sequence_root / f"stride_{stride}"
        output.mkdir(parents=True, exist_ok=False)
        population = population_summary(budget)
        atomic_write_json(output / "schedule.json", population | {
            "roles": budget["roles"], "intervals": [i.payload() for i in budget["intervals"]]})
        common = {"records": budget["records"], "roles": budget["roles"],
                  "calibration": calibration, "config": config}
        sparse_row = _run_job({"kind": "formal_h2_sparse", **common},
                              temporary / f"sparse_{stride}", runtime, profile)
        sparse = save_trajectory(output / "sparse_rgb", sparse_row, budget["records"],
                                  budget["roles"], config, dense=False)
        del sparse_row
        task = predicted_task(common, budget, trained, config, profile)
        ours_row = _run_job(task, temporary / f"predicted_{stride}", runtime, profile)
        ours = save_trajectory(output / "predicted_jepa", ours_row, budget["records"],
                                budget["roles"], config, dense=True)
        del ours_row, task
        comparison = comparison_row(population, sparse, ours, trained["record"])
        atomic_write_json(output / "results.json", {
            "anchor_stride": stride, "population": population, "comparison": comparison,
            "groundtruth_ref": "../groundtruth.json", "full_rgb_ref": "../full_rgb/results.json",
            "predictor_ref": str(trained["checkpoint_path"].relative_to(root)),
            "predictor_sha256": sha256_file(trained["checkpoint_path"]),
            "training_ref": str((model_root / "training.json").relative_to(root)),
            "sparse_rgb_ref": "sparse_rgb/results.json", "ours_ref": "predicted_jepa/results.json",
            "training_cleanup": cleanup})
        index["strides"][str(stride)] = str((output / "results.json").relative_to(root))
        index["accuracy_vs_communication"].append(comparison)
        index["prediction_quality_vs_horizon"].extend(trained["record"]["horizon_resolved_quality"])
        del trained
        publish_index(root, index)


def run(sequence="MH_01_easy", strides=PILOT_STRIDES, *, config_path=DEFAULT_CONFIG, execute=False):
    protocol, canonical, config_path = load_protocol(config_path)
    strides = tuple(strides)
    if sequence != protocol["sequence"]:
        raise ValueError("only MH_01_easy is allowed in the first pilot")
    if not strides or len(strides) != len(set(strides)) or any(
            isinstance(s, bool) or not isinstance(s, int) or s < 2 for s in strides):
        raise ValueError("choose distinct integer strides >= 2")
    root = repo_path(protocol["output_root"])
    if execute:
        from .cuda_devices import CudaDevicePool
        CudaDevicePool.discover().online_mapping()
    records = load_sequence_records(canonical, sequence)
    budgets, preparation = prepare_protocol(records, protocol, strides)
    index = {"schema_version": 1, "run_policy": "fresh_current_canonical_replace", "protocol": "anchor_budget_fresh_predictor_v1",
             "sequence": sequence, "anchor_strides": list(strides),
             "trajectory_comparison": ["GT", "Full RGB", "Sparse RGB", "Ours / Predicted JEPA"],
             "status": "running" if execute else "prepared_no_experiment_run",
             "config": str(config_path), "config_sha256": sha256_file(config_path),
             "preparation": preparation, "strides": {}, "accuracy_vs_communication": [],
             "prediction_quality_vs_horizon": []}
    # Safe default: no CUDA initialization, extraction, model load, training, SLAM.
    if not execute:
        temporary = Path(tempfile.mkdtemp(prefix="anchor_budget_prepare_", dir="/tmp"))
        atomic_write_json(temporary / "PREPARATION.json", preparation)
        index["preparation_path"] = str(temporary / "PREPARATION.json")
        return index
    # All intermediate models/worker/diagnostic files disappear on either outcome.
    with staged_directory(root) as temporary:
        staged = temporary / "artifacts"
        staged.mkdir()
        index["repository"] = repository_provenance()
        publish_index(staged, index)
        bridge = repo_path(canonical["paths"]["h1_bridge"])
        bridge_hash = sha256_file(bridge)
        execute_sequence(records, budgets, protocol, canonical, staged, temporary / "workers", index)
        index["status"] = "complete"
        if sha256_file(bridge) != bridge_hash:
            raise RuntimeError("canonical H1 bridge changed during Anchor Budget execution")
        publish_index(staged, index)
        publish_compact(staged, root, CHECKPOINT_ROOT)
    return index


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--sequence", choices=["MH_01_easy"], default="MH_01_easy")
    result.add_argument("--strides", nargs="+", type=int, default=list(PILOT_STRIDES))
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--execute", action="store_true",
                        help="fresh Full RGB and per-stride training + Sparse/Ours; replaces the canonical request set")
    return result


def main():
    args = parser().parse_args()
    print("Anchor Budget — Communication-Budget Sensitivity")
    result = run(args.sequence, args.strides, config_path=args.config, execute=args.execute)
    print(json.dumps({"status": result["status"], "sequence": result["sequence"],
                      "strides": result["anchor_strides"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
