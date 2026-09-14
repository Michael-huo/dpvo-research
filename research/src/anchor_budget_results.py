"""Plot-ready anchor-budget tables and shared trajectory references."""

from __future__ import annotations

import csv
import io

import numpy as np

from .evaluation import (dense_hidden_ate, evaluate_paired_trajectory,
                         filter_groundtruth_associable_timestamps,
                         freeze_evaluation_population, population_coverage)
from .profiling import graph_workload_payload
from .protocol import atomic_write_bytes, atomic_write_json, repo_path, sha256_file
from .run_h2 import _compact_runtime


def evaluation_population(records, roles, config):
    sequence = records[0].identity.sequence
    gt = repo_path(config["dataset"]["groundtruth_pattern"].format(sequence=sequence))
    anchors, _ = filter_groundtruth_associable_timestamps(
        gt, [r.identity.timestamp_ns for r in records if roles[r.identity.key] == "anchor"],
        max_difference_ns=int(config["evaluation"]["groundtruth_max_association_ns"]))
    dense, _ = filter_groundtruth_associable_timestamps(
        gt, [r.identity.timestamp_ns for r in records],
        max_difference_ns=int(config["evaluation"]["groundtruth_max_association_ns"]))
    hidden_values = {r.identity.timestamp_ns for r in records if roles[r.identity.key] == "hidden"}
    stride = config["experiment"].get("anchor_stride", 5)
    population = freeze_evaluation_population(
        anchors, horizon_seconds=float(config["evaluation"]["rpe_horizon_seconds"]),
        tolerance_ns=int(config["evaluation"]["rpe_pair_tolerance_ns"]), allow_empty_rpe=True,
        source=f"canonical_sparse_rgb_reference_k{stride}_groundtruth_associable")
    return population, dense, tuple(t for t in dense if t in hidden_values), gt


def save_trajectory(output, row, records, roles, config, *, dense):
    output.mkdir(parents=True, exist_ok=False)
    arrays = row["arrays"]
    population, dense_timestamps, hidden_timestamps, gt = evaluation_population(records, roles, config)
    coverage = population_coverage(arrays, population.common_anchor_timestamps_ns)
    complete = coverage["canonical_pose_coverage"] == 1.0
    # Do not filter an unsuccessful trajectory out of the study, or refit on a
    # convenient subset. Leave paired metrics null and persist the coverage.
    metrics = evaluate_paired_trajectory(arrays, population, gt) if complete else None
    dense_coverage = population_coverage(arrays, dense_timestamps) if dense else None
    dense_metrics = (dense_hidden_ate(arrays, dense_timestamps, hidden_timestamps, gt)
                     if dense and dense_coverage["canonical_pose_coverage"] == 1.0 else None)
    np.savez_compressed(output / "trajectory.npz", **arrays)
    result = {
        "condition": row["condition"], "status": "complete" if complete else "incomplete_coverage",
        "canonical_evaluation": metrics, "canonical_coverage": coverage,
        "canonical_population": population.to_dict(),
        "dense_hidden_evaluation": dense_metrics,
        "dense_coverage": dense_coverage,
        "rpe_status": "available" if population.rpe_pairs_ns else "unavailable_no_pairs_under_frozen_1s_rule",
        "runtime": _compact_runtime(row["runtime"]),
        "graph_workload": graph_workload_payload(row["runtime"]),
        "matched_trajectory_wall_seconds": float(row["runtime"]["elapsed_seconds"]),
        "sequential_execution": row["sequential_execution"],
        "trajectory_sha256": sha256_file(output / "trajectory.npz"),
        "evaluation_role": "in_sequence_trajectory_feasibility_not_held_out_generalization",
        "timing_protocol": "research_matched_online_wall_including_terminate_and_worker_flush",
        "warmup": row.get("warmup", row.get("online_profile", {}).get("warmup")),
    }
    if "online_profile" in row:
        result["h2_stage_profile"] = row["online_profile"]
        result["provider_usage"] = row["runtime"]["provider_usage"]
    atomic_write_json(output / "results.json", result)
    return result


def comparison_row(population, sparse, ours, training):
    if sparse["canonical_population"] != ours["canonical_population"]:
        raise RuntimeError("Sparse/Ours evaluation populations must match exactly")
    communication = population["communication"]
    sparse_ate = (sparse["canonical_evaluation"] or {}).get("ate_rmse_m")
    ours_ate = (ours["canonical_evaluation"] or {}).get("ate_rmse_m")
    profile = ours["h2_stage_profile"]
    return {
        "anchor_stride": population["anchor_stride"],
        "actual_anchor_ratio": communication["actual_anchor_ratio"],
        "encoded_byte_reduction": communication["encoded_byte_reduction"],
        "anchor_count": communication["anchor_count"],
        "hidden_count": communication["hidden_count"],
        "num_effective_observations": communication["num_effective_observations"],
        "encoded_anchor_bytes": communication["encoded_anchor_bytes"],
        "encoded_full_bytes": communication["encoded_full_bytes"],
        "sparse_ate_rmse_m": sparse_ate, "ours_ate_rmse_m": ours_ate,
        "ours_minus_sparse_ate_m": None if sparse_ate is None or ours_ate is None else ours_ate - sparse_ate,
        "sparse_wall_seconds": sparse["matched_trajectory_wall_seconds"],
        "ours_wall_seconds": ours["matched_trajectory_wall_seconds"],
        "predictor_training_wall_seconds": training["summary"]["elapsed_seconds"],
        "context_wait_mean_ms": profile["context_wait_ms"]["mean_ms"],
        "context_wait_p95_ms": profile["context_wait_ms"]["p95_ms"],
        "effective_hidden_delay_mean_ms": profile["effective_hidden_delay_ms"]["mean_ms"],
        "sparse_coverage": sparse["canonical_coverage"]["canonical_pose_coverage"],
        "ours_coverage": ours["canonical_coverage"]["canonical_pose_coverage"],
        "evaluation_population_sha256": sparse["canonical_population"]["population_sha256"],
        "rpe_pair_count": len(sparse["canonical_population"]["rpe_pairs_ns"]),
    }


def write_csv(path, rows):
    if not rows:
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, buffer.getvalue().encode())


def write_tables(root, index):
    accuracy = index.get("accuracy_vs_communication", [])
    horizon = index.get("prediction_quality_vs_horizon", [])
    atomic_write_json(root / "accuracy_vs_communication.json", accuracy)
    atomic_write_json(root / "prediction_quality_vs_horizon.json", horizon)
    write_csv(root / "accuracy_vs_communication.csv", accuracy)
    write_csv(root / "prediction_quality_vs_horizon.csv", horizon)
    lines = ["# Anchor Budget — Communication-Budget Sensitivity", "",
             f"Status: {index['status']}. Sequence: {index['sequence']}.", "",
             "GT is a shared reference. Full RGB runs once; each stride uses a fresh predictor, "
             "then separate sequential Sparse RGB and Ours DPVO processes.", "",
             "| Stride | Anchor % | Byte reduction | Sparse ATE | Ours ATE | Ours-Sparse Δ | Ours wall | Context wait |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def fmt(value, scale=1, suffix=""):
        return "—" if value is None else f"{value * scale:.6f}{suffix}"
    for row in accuracy:
        lines.append("| " + " | ".join([
            str(row["anchor_stride"]), fmt(row["actual_anchor_ratio"], 100, "%"),
            fmt(row["encoded_byte_reduction"], 100, "%"), fmt(row["sparse_ate_rmse_m"]),
            fmt(row["ours_ate_rmse_m"]), fmt(row["ours_minus_sparse_ate_m"]),
            fmt(row["ours_wall_seconds"], suffix=" s"), fmt(row["context_wait_mean_ms"], suffix=" ms"),
        ]) + " |")
    if not accuracy:
        for population in index["preparation"]["populations"]:
            c = population["communication"]
            lines.append(f"| {population['anchor_stride']} | {fmt(c['actual_anchor_ratio'],100,'%')} | "
                         f"{fmt(c['encoded_byte_reduction'],100,'%')} | — | — | — | — | — |")
    if "full_rgb" in index:
        full = index["full_rgb"]
        lines.extend(["", f"Full RGB (shared): ATE {fmt(full['ate_rmse_m'], suffix=' m')}; "
                      f"matched wall {fmt(full['matched_trajectory_wall_seconds'], suffix=' s')}. "
                      f"Metrics and graph workload: `{full['results']}`."])
    lines.extend(["", "ATE uses the unchanged feasibility Sim(3) protocol on each stride's common "
                  "GT-associable anchor timestamps. Sparse and Ours share that population; "
                  "population hashes and RPE pair counts accompany every comparison. "
                  "The shared Full RGB ATE uses the frozen canonical stride-5 evaluation population.", "",
                  "The 1 s RPE rule is unchanged. Stride 3 has zero matching pairs after GT association; "
                  "its RPE is null (unavailable), with pair count 0. Trajectory scoring covers the frozen "
                  "sequence span; representation diagnostics alone use the held-out test region.", "",
                  "Accuracy vs Communication: `accuracy_vs_communication.json` / `.csv`. "
                  "Prediction Quality vs Horizon: `prediction_quality_vs_horizon.json` / `.csv`. "
                  "Per-query timestamps/distances are in `models/stride_K/horizon_queries.json`.", "",
                  "No degradation trend or communication-budget advantage is presumed.", ""])
    atomic_write_bytes(root / "SUMMARY.md", "\n".join(lines).encode())


def publish_index(root, index):
    atomic_write_json(root / "INDEX.json", index)
    write_tables(root, index)
