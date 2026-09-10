"""H0 State: latent-state feasibility for the frozen DPVO state contract."""

from __future__ import annotations

import os

# The public H0 module is a CPU-only coordinator. This must happen before torch,
# DPVO, or profiling modules are imported: several CUDA extension/runtime probes
# can otherwise retain a primary context even though H0 trajectories themselves
# run in fresh children. The worker imports this module by its package name (not
# as __main__) after the launcher has explicitly selected physical GPU0.
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .dpvo_backend import _load_frame, _make_slam, _runtime_classes, _seed_everything
from .canonical import load_yaml, resolve_sequences, trajectory_payload, validate_trajectory_payload
from .evaluation import (dense_hidden_ate, evaluate_paired_trajectory,
                         filter_groundtruth_associable_timestamps,
                         freeze_evaluation_population, plot_canonical_trajectories,
                         population_coverage)
from .efficiency_profiling import (
    PersistentPerformanceAudit, condition_runtime_diagnostics,
    performance_diagnosis,
)
from .oracle_packet import (FMapZeroContextPacket, FrontendPacketWriter,
                            ZERO_PACKET_SCHEMA, _derive_frontend_state,
                            compare_arrays, extract_frontend_packet)
from .protocol import (REPO_ROOT, SUPPORTED_SEQUENCES, load_sequence_records,
                       post_bootstrap_ratio_roles, ratio_schedule_payload,
                       repo_path, sha256_file)
from .runtime import (OnlineFrame, materialize_schedule, run_formal_mode,
                      sanitize_full_oracle_frames)
from .schema import (VISUAL_STATE_CONTRACT, VISUAL_STATE_CONTRACT_SHA256,
                     condition_metadata)
from .registry import (
    base_lineage, complete_lineage, empty_index, publish_current_canonical,
    sequence_entry, validate_module_manifest, write_registry_and_summary,
    write_sequence_metadata,
)
from .execution_runtime import (
    cpu_numa_layout, execution_provenance, initialize_formal_main_process,
    release_cuda_training_state, require_lifecycle_cleanup,
    fixed_cpu_profile,
)
from .parallel_runtime import run_sequential_trajectory_jobs

DEFAULT_CONFIG = REPO_ROOT / "research/configs/phase1_feasibility_h0.yaml"
CONDITIONS = ("full_rgb", "sparse_rgb", "true_fmap")


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    config, resolved = load_yaml(path)
    ratio = float(config["experiment"]["anchor_ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError("anchor_ratio must be in (0, 1]")
    return config, resolved


def _extractor(config: Mapping[str, Any], first_image: torch.Tensor) -> Any:
    runtime_config = {
        "experiment": {"seed": int(config["experiment"]["seed"])},
        "paths": {
            "checkpoint": str(repo_path(config["paths"]["dpvo_checkpoint"])),
            "dpvo_config": str(repo_path(config["paths"]["dpvo_config"])),
            "calibration": str(repo_path(config["dataset"]["calibration"])),
        },
    }
    _seed_everything(int(config["experiment"]["seed"]))
    return _make_slam(_runtime_classes()["NativePacketDPVO"], runtime_config, first_image)


def _compact_runtime(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "condition_name", "input_candidate_count", "processed_observation_count",
        "final_node_count_before_terminate", "final_patch_count_before_terminate",
        "rgb_uploaded_frame_count", "hidden_oracle_packet_count",
        "hidden_online_rgb_violation_count", "factor_count_allocated",
        "hidden_source_factor_count", "hidden_target_factor_count",
        "finite_trajectory", "tracking_success", "trajectory_pose_count",
        "timestamp_contract_exact", "model_training", "gradient_enabled",
        "elapsed_seconds", "peak_gpu_vram_bytes",
    )
    return {key: row.get(key) for key in keys}


def _condition_metadata() -> dict[str, dict[str, Any]]:
    return {
        "full_rgb": condition_metadata(
            experiment_mode="decomposition", input_source="all_rgb_native_dpvo",
            online_allowed_fields=["rgb"], offline_reference_only=False,
            strict_deployment=False, timestamp_causal=True,
            closing_anchor_online_available=True,
        ),
        "sparse_rgb": condition_metadata(
            experiment_mode="decomposition", input_source="anchor_rgb_native_dpvo",
            online_allowed_fields=["anchor_rgb"], offline_reference_only=False,
            strict_deployment=False, timestamp_causal=True,
            closing_anchor_online_available=True,
        ),
        "true_fmap": condition_metadata(
            experiment_mode="decomposition", input_source="anchor_rgb_plus_offline_true_hidden_fmap",
            online_allowed_fields=["anchor_rgb", "hidden_fmap"],
            offline_reference_only=True, strict_deployment=False,
            timestamp_causal=True, closing_anchor_online_available=True,
        ),
    }


def _run_condition(
    condition: str, records: Sequence[Any], calibration: np.ndarray,
    config: Mapping[str, Any], roles: Mapping[str, str], temporary: Path,
) -> dict[str, Any]:
    """Execute one isolated H0 condition inside its assigned GPU worker."""
    if condition == "full_rgb":
        runtime, arrays = run_formal_mode(
            "matched_full_rgb",
            [OnlineFrame(row.identity, row.rgb_path) for row in records],
            calibration, config,
            roles={row.identity.key: "anchor" for row in records},
            condition_name=condition, collect_graph_trace=False,
        )
        return {"condition": condition, "runtime": runtime, "arrays": arrays}
    packet_online = sanitize_full_oracle_frames(records, roles)
    if condition == "sparse_rgb":
        runtime, arrays = run_formal_mode(
            "sparse_rgb", packet_online, calibration, config,
            roles=roles, condition_name=condition, collect_graph_trace=False,
        )
        return {"condition": condition, "runtime": runtime, "arrays": arrays}
    if condition != "true_fmap":
        raise ValueError(f"unsupported H0 condition: {condition}")
    hidden = [row for row in records if roles[row.identity.key] == "hidden"]
    if not hidden:
        raise RuntimeError("decomposition schedule contains no hidden frames")
    first_image, _ = _load_frame({"image_path": records[0].rgb_path}, calibration)
    extractor = _extractor(config, first_image)
    writer = FrontendPacketWriter(
        temporary / "fmap", [row.identity for row in hidden],
        provenance={
            "sequence": records[0].identity.sequence,
            "scope": "offline_oracle_extractor_only",
            "visual_state_contract_sha256": VISUAL_STATE_CONTRACT_SHA256,
        }, schema=ZERO_PACKET_SCHEMA,
    )
    microcheck = None
    for record in hidden:
        image, _ = _load_frame({"image_path": record.rgb_path}, calibration)
        full = extract_frontend_packet(
            extractor, image, record.identity, int(config["experiment"]["seed"]),
        )
        reduced = FMapZeroContextPacket(full.fmap.detach())
        writer.append(record.identity, reduced)
        if microcheck is None:
            left, _ = _derive_frontend_state(
                reduced, record.identity, int(config["experiment"]["seed"]),
                patches_per_image=int(extractor.M), patch_size=int(extractor.P),
                context_dim=int(extractor.DIM),
            )
            right, _ = _derive_frontend_state(
                reduced, record.identity, int(config["experiment"]["seed"]),
                patches_per_image=int(extractor.M), patch_size=int(extractor.P),
                context_dim=int(extractor.DIM),
            )
            microcheck = {
                "fmap_identity": compare_arrays(
                    full.fmap.detach().cpu().numpy(), reduced.fmap.detach().cpu().numpy(),
                ),
                "deterministic_patch_xy": compare_arrays(
                    left.patch_xy.detach().cpu().numpy(), right.patch_xy.detach().cpu().numpy(),
                ),
                "deterministic_gmap": compare_arrays(
                    left.gmap.detach().cpu().numpy(), right.gmap.detach().cpu().numpy(),
                ),
            }
        del image, full, reduced
    del extractor
    preparation_cleanup = release_cuda_training_state()
    require_lifecycle_cleanup(preparation_cleanup)
    store = writer.finalize()
    runtime, arrays = run_formal_mode(
        "fmap_zero_context", packet_online, calibration, config, roles=roles,
        store=store, condition_name=condition,
        collect_graph_trace=False,
    )
    descriptor = store.sanitized_descriptor()
    store.close()
    return {
        "condition": condition, "runtime": runtime, "arrays": arrays,
        "contract_microcheck": microcheck, "oracle_store": descriptor,
        "preparation_cleanup": preparation_cleanup,
    }


def _assemble_sequence(
    config: Mapping[str, Any], sequence: str, records: Sequence[Any],
    roles: Mapping[str, str], schedule: Mapping[str, Any],
    condition_rows: Sequence[Mapping[str, Any]], output: Path,
) -> dict[str, Any]:
    by_condition = {row["condition"]: row for row in condition_rows}
    if set(by_condition) != set(CONDITIONS):
        raise RuntimeError(f"incomplete H0 condition population for {sequence}")
    runtimes = {name: by_condition[name]["runtime"] for name in CONDITIONS}
    arrays = {name: by_condition[name]["arrays"] for name in CONDITIONS}
    true_row = by_condition["true_fmap"]
    if int(runtimes["full_rgb"]["bootstrap_end_candidate_index"]) != int(
        schedule["bootstrap_end_candidate_index"]
    ):
        raise RuntimeError("H0 frozen bootstrap boundary changed in condition worker")
    if "bootstrap_decisions" in schedule:
        frozen_decisions = [
            {
                "candidate_index": int(row["candidate_index"]),
                "motion_accepted": bool(row["motion_accepted"]),
            }
            for row in schedule["bootstrap_decisions"]
        ]
        actual_decisions = [
            {
                "candidate_index": int(row["candidate_index"]),
                "motion_accepted": bool(row["motion_accepted"]),
            }
            for row in runtimes["full_rgb"]["bootstrap_decisions"]
        ]
        if actual_decisions != frozen_decisions:
            raise RuntimeError("H0 frozen bootstrap decisions changed in condition worker")
    hidden = [row for row in records if roles[row.identity.key] == "hidden"]
    anchor = [row for row in records if roles[row.identity.key] == "anchor"]
    groundtruth = repo_path(config["dataset"]["groundtruth_pattern"].format(sequence=sequence))
    anchor_timestamps, anchor_excluded = filter_groundtruth_associable_timestamps(
        groundtruth, [row.identity.timestamp_ns for row in anchor],
        max_difference_ns=int(config["evaluation"]["groundtruth_max_association_ns"]),
    )
    dense_timestamps, _ = filter_groundtruth_associable_timestamps(
        groundtruth, [row.identity.timestamp_ns for row in records],
        max_difference_ns=int(config["evaluation"]["groundtruth_max_association_ns"]),
    )
    hidden_set = {row.identity.timestamp_ns for row in hidden}
    hidden_timestamps = [value for value in dense_timestamps if value in hidden_set]
    population = freeze_evaluation_population(
        anchor_timestamps, horizon_seconds=float(config["evaluation"]["rpe_horizon_seconds"]),
        tolerance_ns=int(config["evaluation"]["rpe_pair_tolerance_ns"]),
    )
    metadata = _condition_metadata()
    conditions = {}
    for name in CONDITIONS:
        coverage = population_coverage(arrays[name], population.common_anchor_timestamps_ns)
        if coverage["canonical_pose_coverage"] != 1.0:
            raise RuntimeError(f"incomplete canonical coverage: {sequence}/{name}")
        dense = None
        if name != "sparse_rgb":
            dense = dense_hidden_ate(arrays[name], dense_timestamps, hidden_timestamps, groundtruth)
        conditions[name] = {
            **metadata[name], "runtime": _compact_runtime(runtimes[name]),
            "canonical_coverage": coverage,
            "canonical_evaluation": evaluate_paired_trajectory(arrays[name], population, groundtruth),
            "dense_hidden_evaluation": dense,
        }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **trajectory_payload(
        records, roles, arrays, population, groundtruth,
    ))
    validate_trajectory_payload(output / "trajectories.npz", CONDITIONS)
    plot_canonical_trajectories(
        output / "trajectory.png", arrays, population, groundtruth, sequence=sequence,
        labels={"full_rgb": "Full RGB", "sparse_rgb": "Sparse RGB",
                "true_fmap": "True FMap"},
    )
    return {
        "sequence": sequence, "schedule": schedule,
        "candidate_count": len(records), "anchor_count": len(anchor), "hidden_count": len(hidden),
        "canonical_population": population.to_dict(), "gt_excluded_anchor_count": len(anchor_excluded),
        "canonical_population_sha256": population.population_sha256,
        "conditions": conditions,
        "visual_state_contract": {**VISUAL_STATE_CONTRACT,
                                  "contract_sha256": VISUAL_STATE_CONTRACT_SHA256},
        "contract_microcheck": true_row["contract_microcheck"],
        "oracle_store": {
            "schema": true_row["oracle_store"]["schema"],
            "contains_hidden_rgb_or_path": true_row["oracle_store"]["contains_hidden_rgb_or_path"],
            "identity_list_sha256": true_row["oracle_store"]["identity_list_sha256"],
        },
    }


def run(sequences: Sequence[str]) -> dict[str, Any]:
    command_started = time.perf_counter()
    config, config_path = load_config()
    requested = resolve_sequences(sequences, config["experiment"]["default_sequences"])
    formal_runtime = initialize_formal_main_process(training_device=None)
    root = repo_path(config["paths"]["output_root"])
    root.parent.mkdir(parents=True, exist_ok=True)
    source_files = (
        __file__, Path(__file__).with_name("runtime.py"),
        Path(__file__).with_name("dpvo_backend.py"),
        Path(__file__).with_name("oracle_packet.py"),
        Path(__file__).with_name("protocol.py"),
        Path(__file__).with_name("evaluation.py"),
        Path(__file__).with_name("registry.py"),
        Path(__file__).with_name("schema.py"),
        Path(__file__).with_name("efficiency_profiling.py"),
    )
    with tempfile.TemporaryDirectory(prefix=".phase1_h0_", dir=root.parent) as name:
        staged_root = Path(name) / "h0_state"
        (staged_root / "sequences").mkdir(parents=True)
        index = empty_index("h0_state", requested)
        calibration = np.loadtxt(
            repo_path(config["dataset"]["calibration"]), delimiter=" ",
        )
        layout = cpu_numa_layout(formal_runtime["hardware"])
        stage_c = fixed_cpu_profile(layout)["stage_c"]
        records_by_sequence = {
            sequence: load_sequence_records(config, sequence) for sequence in requested
        }
        schedule_rows, schedule_execution = run_sequential_trajectory_jobs(
            [
                {"kind": "materialize_schedule", "records": records_by_sequence[sequence],
                 "calibration": calibration, "config": config, "sequence": sequence}
                for sequence in requested
            ],
            Path(name) / "h0_schedule_jobs", cpu_profile=stage_c,
            hardware=formal_runtime["hardware"],
        )
        schedules = {row["sequence"]: row["schedule"] for row in schedule_rows}
        frozen = {}
        for sequence in requested:
            records = records_by_sequence[sequence]
            bootstrap = schedules[sequence]
            bootstrap_end = int(bootstrap["bootstrap_end_candidate_index"])
            identities = [row.identity for row in records]
            roles = post_bootstrap_ratio_roles(
                identities, bootstrap_end_candidate_index=bootstrap_end,
                anchor_ratio=float(config["experiment"]["anchor_ratio"]),
            )
            schedule = ratio_schedule_payload(
                identities, bootstrap_end_candidate_index=bootstrap_end,
                anchor_ratio=float(config["experiment"]["anchor_ratio"]),
            )
            schedule["bootstrap_decisions"] = bootstrap["bootstrap_decisions"]
            schedule["bootstrap_decisions_sha256"] = bootstrap[
                "bootstrap_decisions_sha256"
            ]
            frozen[sequence] = (records, roles, schedule)
        jobs = []
        for sequence in requested:
            records, roles, _ = frozen[sequence]
            for condition in CONDITIONS:
                jobs.append({
                    "kind": "formal_h0_condition", "config": config,
                    "sequence": sequence, "condition": condition,
                    "records": records, "roles": roles,
                    "calibration": calibration,
                })
        with PersistentPerformanceAudit(
            "h0_state", "formal_sequential_evaluation",
            components=("formal_coordinator",),
        ) as command_performance:
            with command_performance.phase("sequential_gpu0_condition_evaluation"):
                cleanup_before_trajectories = release_cuda_training_state()
                require_lifecycle_cleanup(cleanup_before_trajectories)
                completed, scheduler = run_sequential_trajectory_jobs(
                    jobs, Path(name) / "h0_gpu_jobs", cpu_profile=stage_c,
                    hardware=formal_runtime["hardware"],
                )
        command_performance_payload = command_performance.payload()
        completed_by_sequence = {sequence: [] for sequence in requested}
        for job in completed:
            completed_by_sequence[job["sequence"]].append(job)
        for sequence in requested:
            base, provenance = base_lineage(
                config, sequence, source_files,
                h0_contract_sha256=VISUAL_STATE_CONTRACT_SHA256,
            )
            provenance = provenance | {
                "config_file": str(config_path.relative_to(REPO_ROOT)),
                "config_file_sha256": sha256_file(config_path),
                "execution_backend_provenance": execution_provenance(),
                "formal_hardware": formal_runtime,
            }
            output = staged_root / "sequences" / sequence
            records, roles, schedule = frozen[sequence]
            sequence_jobs = completed_by_sequence[sequence]
            result = _assemble_sequence(
                config, sequence, records, roles, schedule, sequence_jobs, output,
            )
            performance_payload = {
                "schema": "phase1_condition_job_performance_v1",
                "conditions": {
                    row["condition"]: row["performance"] for row in sequence_jobs
                },
                "condition_runtime": condition_runtime_diagnostics(result),
            }
            performance_payload["diagnosis"] = performance_diagnosis(
                performance_payload
            )
            performance_payload["sequential_evaluation"] = scheduler
            performance_payload["formal_command"] = command_performance_payload
            performance_payload["execution_backend_provenance"] = execution_provenance()
            result["performance_diagnostics"] = performance_payload
            lineage = complete_lineage(base, result["schedule"]["schedule_sha256"])
            write_sequence_metadata(
                output, "h0_state", sequence, result, lineage, provenance,
            )
            index["sequences"][sequence] = sequence_entry(
                output, "h0_state", lineage,
                bootstrap_end_candidate_index=result["schedule"]["bootstrap_end_candidate_index"],
            )
        total_makespan = time.perf_counter() - command_started
        index["execution"] = {
            "schema": "phase1_formal_execution_summary_v1",
            "hardware": formal_runtime,
            "provenance": execution_provenance(),
            "sequential_evaluation": scheduler,
            "sequential_schedule_materialization": schedule_execution,
            "cleanup_before_trajectories": cleanup_before_trajectories,
            "total_makespan_seconds": total_makespan,
            "domain": "research_throughput",
        }
        write_registry_and_summary(staged_root, "h0_state", index)
        validate_module_manifest(staged_root, "h0_state", index)
        publish_current_canonical(staged_root, root)
    return {
        "status": "complete", "module": "h0_state",
        "output": str(root.relative_to(REPO_ROOT)),
        "requested_sequences": list(requested),
        "fresh_sequences": list(requested),
        "run_policy": "fresh_current_canonical_replace",
        "performance_diagnostics": {
            "persistent": True,
            "sequence_result_field": "result.performance_diagnostics",
            "aggregate_summary": "SUMMARY_H0.md",
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--sequences", nargs="+", choices=SUPPORTED_SEQUENCES,
        help="fresh sequence evaluations; this request replaces the current canonical set",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    config, _ = load_config()
    requested = resolve_sequences(args.sequences, config["experiment"]["default_sequences"])
    print("Phase 1: H0 State — Latent-State Feasibility")
    print("Input modes: full_rgb, sparse_rgb, true_fmap")
    print(json.dumps(run(requested), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
