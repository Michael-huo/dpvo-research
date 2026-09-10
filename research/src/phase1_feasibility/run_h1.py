"""H1 Interface: Oracle-JEPA representation-interface feasibility."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .canonical import resolve_sequences, trajectory_payload, validate_trajectory_payload
from .bridge_checkpoint import (
    H1_CONFIG_PATH, H1_TRAINING_SEQUENCE, h1_training_context, h1_training_input,
    h1_training_sources, load_compatible_bridge, load_h1_config,
)
from .evaluation import (dense_hidden_ate, evaluate_paired_trajectory,
                         filter_groundtruth_associable_timestamps,
                         freeze_evaluation_population, plot_canonical_trajectories,
                         population_coverage)
from .efficiency_profiling import (
    PerformanceRecorder, PersistentPerformanceAudit, add_cuda_worker_mapping,
    condition_runtime_diagnostics, performance_diagnosis,
)
from .jepa_runtime import sequence_geometry, state_dict_sha256
from .jepa_fmap import (
    contiguous_split, coordinate_protocol_metadata, hidden_split_keys,
)
from .protocol import (REPO_ROOT, SUPPORTED_SEQUENCES, canonical_sha256,
                       load_sequence_records, post_bootstrap_ratio_roles,
                       ratio_schedule_payload, repo_path, sha256_file)
from .h1_training import (FeatureStore, HiddenFMapProvider, _previous_anchor_mapping,
                          evaluate_representation_control, train_bridge)
from .runtime import (OnlineFrame, materialize_schedule, run_formal_mode,
                      sanitize_full_oracle_frames)
from .schema import condition_metadata
from .schema import VISUAL_STATE_CONTRACT_SHA256
from .registry import (
    base_lineage, complete_lineage, empty_index, publish_current_canonical,
    sequence_entry, validate_module_manifest, write_registry_and_summary,
    write_sequence_metadata,
)
from .execution_runtime import (
    cpu_numa_layout, execution_provenance, initialize_formal_main_process,
    release_cuda_training_state, require_lifecycle_cleanup, runtime_provenance,
    fixed_cpu_profile,
)
from .parallel_runtime import extract_parallel, run_sequential_trajectory_jobs
from .training_runtime import ResidentH1View

DEFAULT_CONFIG = H1_CONFIG_PATH
TRAINING_SEQUENCE = H1_TRAINING_SEQUENCE
CONDITIONS = ("full_rgb", "sparse_rgb", "true_fmap", "oracle_jepa_bridge")


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    return load_h1_config(path)


def _compact_runtime(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "condition_name", "processed_observation_count",
        "final_node_count_before_terminate", "final_patch_count_before_terminate",
        "rgb_uploaded_frame_count", "hidden_oracle_packet_count",
        "hidden_online_rgb_violation_count", "factor_count_allocated",
        "hidden_source_factor_count", "hidden_target_factor_count",
        "finite_trajectory", "tracking_success", "trajectory_pose_count",
        "timestamp_contract_exact", "model_training", "gradient_enabled",
        "elapsed_seconds", "peak_gpu_vram_bytes",
        "representation", "hidden_provider_usage",
    )
    return {key: row.get(key) for key in keys}


def _metadata() -> dict[str, dict[str, Any]]:
    return {
        "full_rgb": condition_metadata(
            experiment_mode="h1_interface", input_source="all_rgb_native_dpvo",
            online_allowed_fields=["rgb"], offline_reference_only=False,
            strict_deployment=False, timestamp_causal=True,
            closing_anchor_online_available=True,
        ),
        "sparse_rgb": condition_metadata(
            experiment_mode="h1_interface", input_source="anchor_rgb_native_dpvo",
            online_allowed_fields=["anchor_rgb"], offline_reference_only=False,
            strict_deployment=False, timestamp_causal=True,
            closing_anchor_online_available=True,
        ),
        "true_fmap": condition_metadata(
            experiment_mode="h1_interface", input_source="offline_true_hidden_fmap",
            online_allowed_fields=["anchor_rgb", "hidden_fmap"],
            offline_reference_only=True, strict_deployment=False,
            timestamp_causal=True, closing_anchor_online_available=True,
        ),
        "oracle_jepa_bridge": condition_metadata(
            experiment_mode="h1_interface", input_source="offline_oracle_hidden_jepa_to_bridge",
            online_allowed_fields=["anchor_rgb", "hidden_oracle_jepa"],
            offline_reference_only=True, strict_deployment=False,
            timestamp_causal=True, closing_anchor_online_available=True,
        ),
    }


@torch.no_grad()
def _feature_diagnostics(path: Path, store: FeatureStore, model: torch.nn.Module,
                         hidden_keys: Sequence[str], seed: int) -> dict[str, Any]:
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phase1-exp6")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    count = min(6, len(hidden_keys))
    order = np.random.default_rng(seed).choice(len(hidden_keys), count, replace=False)
    selected = [hidden_keys[int(index)] for index in sorted(order.tolist())]
    figure, axes = plt.subplots(count, 3, figsize=(10.2, 2.7 * count), squeeze=False)
    rows = []
    for row, key in enumerate(selected):
        index = store.index[key]
        tokens = torch.from_numpy(np.asarray(store.block5[index:index + 1], np.float32)).cuda()
        true = torch.from_numpy(np.asarray(store.teacher[index], np.float32)).cuda()
        predicted = model(tokens)[0]
        error = 1.0 - F.cosine_similarity(predicted.float(), true.float(), dim=0, eps=1e-8)
        true_norm = true.float().norm(dim=0).cpu().numpy()
        predicted_norm = predicted.float().norm(dim=0).cpu().numpy()
        error_np = error.cpu().numpy()
        for axis, image, title in zip(
            axes[row], (true_norm, predicted_norm, error_np),
            ("True FMap norm", "Oracle-JEPA bridge norm", "Cosine error"),
        ):
            axis.imshow(image, cmap="viridis")
            axis.set_title(title); axis.axis("off")
        rows.append({"identity_key": key, "mean_cosine_error": float(error.mean())})
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)
    return {"sample_count": count, "seed": seed, "rows": rows,
            "online_model_or_dpvo_input": False}


def _feature_store_descriptor(store: FeatureStore) -> dict[str, Any]:
    return {
        "block5_path": str(Path(store.block5.filename).resolve()),
        "block5_shape": list(store.block5.shape),
        "teacher_path": str(Path(store.teacher.filename).resolve()),
        "teacher_shape": list(store.teacher.shape),
        "index": dict(store.index),
        "identity_keys": list(store.identity_keys),
    }


def _run_condition(
    condition: str, records: Sequence[Any], calibration: np.ndarray,
    roles: Mapping[str, str], store: FeatureStore | None,
    model: torch.nn.Module | None, config: Mapping[str, Any], temporary: Path,
) -> dict[str, Any]:
    sequence = records[0].identity.sequence
    all_online = [OnlineFrame(row.identity, row.rgb_path) for row in records]
    packet_online = sanitize_full_oracle_frames(records, roles)
    if condition == "full_rgb":
        runtime, arrays = run_formal_mode(
            "matched_full_rgb", all_online, calibration, config,
            roles={row.identity.key: "anchor" for row in records},
            condition_name=condition, collect_graph_trace=False,
        )
        return {"condition": condition, "runtime": runtime, "arrays": arrays}
    if condition == "sparse_rgb":
        runtime, arrays = run_formal_mode(
            "sparse_rgb", packet_online, calibration, config,
            roles=roles, condition_name=condition, collect_graph_trace=False,
        )
        return {"condition": condition, "runtime": runtime, "arrays": arrays}
    if condition not in {"true_fmap", "oracle_jepa_bridge"} or store is None:
        raise ValueError(f"unsupported or unprepared H1 condition: {condition}")
    provider = HiddenFMapProvider(
        condition, store, _previous_anchor_mapping(records, roles),
        model if condition == "oracle_jepa_bridge" else None,
    )
    runtime, arrays = run_formal_mode(
        "fmap_zero_context", packet_online, calibration, config, roles=roles,
        hidden_provider=provider, condition_name=condition,
        collect_graph_trace=False,
    )
    result = {"condition": condition, "runtime": runtime, "arrays": arrays}
    if condition == "oracle_jepa_bridge":
        if model is None:
            raise RuntimeError("frozen H1 bridge is missing")
        path = temporary / "feature_diagnostics.png"
        result["feature_diagnostics"] = _feature_diagnostics(
            path, store, model,
            [row.identity.key for row in records if roles[row.identity.key] == "hidden"],
            int(config["experiment"]["seed"]),
        )
        result["feature_diagnostics_path"] = str(path)
    return result


def _assemble_sequence(
    records: Sequence[Any], schedule: Mapping[str, Any], roles: Mapping[str, str],
    condition_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    sequence = records[0].identity.sequence
    by_condition = {row["condition"]: row for row in condition_rows}
    if set(by_condition) != set(CONDITIONS):
        raise RuntimeError(f"incomplete H1 condition population for {sequence}")
    runtimes = {name: by_condition[name]["runtime"] for name in CONDITIONS}
    arrays = {name: by_condition[name]["arrays"] for name in CONDITIONS}
    expected_bootstrap = int(schedule["bootstrap_end_candidate_index"])
    changed = {
        name: row.get("bootstrap_end_candidate_index")
        for name, row in runtimes.items()
        if int(row.get("bootstrap_end_candidate_index", -1)) != expected_bootstrap
    }
    if changed:
        raise RuntimeError(f"H1 frozen bootstrap boundary changed: {changed}")
    groundtruth = repo_path(config["dataset"]["groundtruth_pattern"].format(sequence=sequence))
    anchors = [row for row in records if roles[row.identity.key] == "anchor"]
    hidden = [row for row in records if roles[row.identity.key] == "hidden"]
    anchor_timestamps, excluded = filter_groundtruth_associable_timestamps(
        groundtruth, [row.identity.timestamp_ns for row in anchors],
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
    metadata = _metadata()
    conditions = {}
    for name in CONDITIONS:
        coverage = population_coverage(arrays[name], population.common_anchor_timestamps_ns)
        if coverage["canonical_pose_coverage"] != 1.0:
            raise RuntimeError(f"incomplete H1 coverage: {sequence}/{name}")
        dense = None if name == "sparse_rgb" else dense_hidden_ate(
            arrays[name], dense_timestamps, hidden_timestamps, groundtruth,
        )
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
                "true_fmap": "True FMap", "oracle_jepa_bridge": "Oracle JEPA→Bridge"},
    )
    oracle = by_condition["oracle_jepa_bridge"]
    shutil.copy2(oracle["feature_diagnostics_path"], output / "feature_diagnostics.png")
    diagnostics = oracle["feature_diagnostics"]
    return {
        "evaluation_role": ("bridge_development_in_sequence_feasibility"
                            if sequence == TRAINING_SEQUENCE else "frozen_bridge_zero_shot"),
        "per_sequence_training": False, "schedule": dict(schedule),
        "candidate_count": len(records), "anchor_count": len(anchors), "hidden_count": len(hidden),
        "canonical_population": population.to_dict(), "gt_excluded_anchor_count": len(excluded),
        "canonical_population_sha256": population.population_sha256,
        "conditions": conditions, "feature_diagnostics": diagnostics,
    }


def _training_details(
    records: Sequence[Any], bootstrap_end: int, calibration: np.ndarray,
    config: Mapping[str, Any], base: Mapping[str, Any],
) -> dict[str, Any]:
    roles = post_bootstrap_ratio_roles(
        [row.identity for row in records],
        bootstrap_end_candidate_index=int(bootstrap_end),
        anchor_ratio=float(config["experiment"]["anchor_ratio"]),
    )
    schedule = ratio_schedule_payload(
        [row.identity for row in records],
        bootstrap_end_candidate_index=int(bootstrap_end),
        anchor_ratio=float(config["experiment"]["anchor_ratio"]),
    )
    split = contiguous_split([row.identity for row in records])
    split_keys = hidden_split_keys([row.identity for row in records], roles, split)
    counts = {key: len(value) for key, value in split_keys.items()}
    if counts != dict(config["split"]["expected_hidden_counts"]):
        raise RuntimeError(f"H1 split population changed: {counts}")
    transform, geometry = sequence_geometry(records[0], calibration, config)
    lineage = {
        "training_input": {
            key: value for key, value in base.items()
            if key not in {"schedule_sha256", "h1_bridge_sha256", "h2_predictor_sha256"}
        },
        "bootstrap_end_candidate_index": int(bootstrap_end),
        "schedule_sha256": schedule["schedule_sha256"],
        "split_sha256": split["split_sha256"],
        "hidden_split_counts": counts,
        "coordinate_transform_sha256": geometry["transform_sha256"],
        "coordinate_protocol": coordinate_protocol_metadata(
            target_height=int(config["jepa"]["target_height"]),
            patch_size=int(config["jepa"]["patch_size"]),
            fmap_scale=int(config["teacher"]["fmap_scale"]),
        ),
    }
    lineage["training_lineage_sha256"] = canonical_sha256(lineage)
    return {
        "roles": roles, "schedule": schedule, "split": split,
        "split_keys": split_keys, "counts": counts, "transform": transform,
        "geometry": geometry, "lineage": lineage,
    }


def run(sequences: Sequence[str]) -> dict[str, Any]:
    command_started = time.perf_counter()
    config, config_path = load_config()
    requested = resolve_sequences(sequences, config["experiment"]["default_sequences"])
    formal_runtime = initialize_formal_main_process()
    root = repo_path(config["paths"]["output_root"])
    root.parent.mkdir(parents=True, exist_ok=True)
    calibration = np.loadtxt(repo_path(config["paths"]["calibration"]), delimiter=" ")
    evaluation_source_files = h1_training_sources() + tuple(
        Path(__file__).with_name(name) for name in (
            "runtime.py", "dpvo_backend.py", "evaluation.py", "registry.py", "canonical.py",
            "efficiency_profiling.py",
        )
    )
    training_records, training_base, training_provenance = h1_training_context(config)

    with tempfile.TemporaryDirectory(prefix=".phase1_h1_", dir=root.parent) as name:
        temporary = Path(name)
        staged_root = temporary / "h1_interface"
        (staged_root / "sequences").mkdir(parents=True)
        checkpoint_path = staged_root / "bridge.pt"
        index = empty_index("h1_interface", requested)
        layout = cpu_numa_layout(formal_runtime["hardware"])
        stage_c = fixed_cpu_profile(layout)["stage_c"]
        schedule_rows, training_schedule_execution = run_sequential_trajectory_jobs(
            [{"kind": "materialize_schedule", "records": training_records,
              "calibration": calibration, "config": config,
              "sequence": TRAINING_SEQUENCE}],
            temporary / "h1_training_schedule", cpu_profile=stage_c,
            hardware=formal_runtime["hardware"],
        )
        training_bootstrap = schedule_rows[0]["schedule"]

        with PersistentPerformanceAudit(
            "h1_interface", "canonical_bridge_training",
            components=("bridge_training_main_process",),
        ) as training_performance:
            with training_performance.phase("schedule_and_split"):
                training = _training_details(
                    training_records,
                    int(training_bootstrap["bootstrap_end_candidate_index"]),
                    calibration, config, training_base,
                )
                training_temp = temporary / "training"
                training_temp.mkdir()
            with training_performance.phase("offline_feature_extraction"):
                training_store, training_extraction = extract_parallel(
                    training_records,
                    set().union(*map(set, training["split_keys"].values())),
                    calibration, config, training_temp, training["transform"],
                    devices=(0, 1, 2),
                    h1_hidden_keys=set().union(*map(set, training["split_keys"].values())),
                )
            with training_performance.phase("resident_initialization"):
                resident_training = ResidentH1View(
                    training_store, training["split_keys"], device=torch.device("cuda:1"),
                )
            with training_performance.phase("bridge_training"):
                training_batch_performance = PerformanceRecorder(enable_cuda=True)
                validation_batch_performance = PerformanceRecorder(enable_cuda=True)
                model, training_summary = train_bridge(
                    resident_training, training["split_keys"], training["transform"],
                    config, checkpoint_path, training["lineage"],
                    profiler=training_batch_performance,
                    validation_profiler=validation_batch_performance,
                )
            residency = dict(resident_training.rows.diagnostics)
            residency["full_train_validation_residency"] = (
                residency["resident_rows"] == residency["allowed_rows"]
            )
            residency["eliminated_memmap_h2d_bytes_per_epoch"] = (
                residency["native_bytes"]
            )
            resident_training.close()
            with training_performance.phase("held_out_representation"):
                representation = evaluate_representation_control(
                    model, training_store, training["split_keys"]["test"],
                    training["transform"], config,
                )
            with training_performance.phase("checkpoint_validation"):
                checked_model, _, _ = load_compatible_bridge(
                    checkpoint_path, training["transform"],
                    hidden_channels=int(config["bridge"]["hidden_channels"]),
                    expected_training_input=h1_training_input(training_base),
                    expected_training_lineage=training["lineage"],
                )
                del checked_model
        training_performance_payload = training_performance.payload()
        add_cuda_worker_mapping(
            training_performance_payload, training_extraction["workers"][0],
        )
        training_performance_payload["execution_backend_provenance"] = (
            execution_provenance()
        )
        training_performance_payload["formal_hardware"] = formal_runtime
        training_performance_payload["training_runtime"] = runtime_provenance(
            formal_runtime["runtime_settings"], component="h1_bridge_post_training",
            model=model, amp=True,
        )
        training_performance_payload["residency"] = residency
        training_performance_payload["efficiency"] = {
            "domain": "research_throughput",
            "parallel_preparation_seconds": training_extraction["elapsed_seconds"],
            "resident_training_wall_seconds": training_summary["elapsed_seconds"],
            "resident_epoch_wall_seconds": [
                row["epoch_wall_seconds"] for row in training_summary["history"]
            ],
            "formal_measurement": True,
            "ddp": "not_implemented; resident path preserves canonical batch and optimizer semantics",
        }
        training_performance_payload["training_throughput"] = {
            "training_batches": training_batch_performance.payload(),
            "validation_batches": validation_batch_performance.payload(),
        }
        training_performance_payload["diagnosis"] = performance_diagnosis(
            training_performance_payload
        )
        training_record = {
            "sequence": TRAINING_SEQUENCE, "lineage": training["lineage"],
            "schedule": training["schedule"], "split": training["split"],
            "hidden_split_counts": training["counts"],
            "coordinate_transform": training["geometry"],
            "summary": training_summary,
            "held_out_representation": representation,
            "performance_diagnostics": training_performance_payload,
            "execution_backend_provenance": execution_provenance(),
            "provenance": training_provenance | {
                "config_file": str(config_path.relative_to(REPO_ROOT)),
                "config_file_sha256": sha256_file(config_path),
                "execution_backend_provenance": execution_provenance(),
                "formal_hardware": formal_runtime,
            },
        }
        bridge_sha256 = sha256_file(checkpoint_path)
        bridge_state_sha256 = state_dict_sha256(model.state_dict())
        model.cpu()
        schedule_cleanup = release_cuda_training_state(resident_training)
        require_lifecycle_cleanup(schedule_cleanup)
        del resident_training, model
        evaluation_inputs = {}
        evaluation_stores = {}
        evaluation_preparation = {}
        nontraining_records = {
            sequence: load_sequence_records(config, sequence)
            for sequence in requested if sequence != TRAINING_SEQUENCE
        }
        if nontraining_records:
            schedule_rows, evaluation_schedule_execution = run_sequential_trajectory_jobs(
                [
                    {"kind": "materialize_schedule", "records": records,
                     "calibration": calibration, "config": config,
                     "sequence": sequence}
                    for sequence, records in nontraining_records.items()
                ],
                temporary / "h1_evaluation_schedules", cpu_profile=stage_c,
                hardware=formal_runtime["hardware"],
            )
            evaluation_schedules = {
                row["sequence"]: row["schedule"] for row in schedule_rows
            }
        else:
            evaluation_schedule_execution = None
            evaluation_schedules = {}
        for sequence in requested:
            if sequence == TRAINING_SEQUENCE:
                records = training_records
                roles = training["roles"]
                schedule = training["schedule"]
                transform = training["transform"]
                geometry = training["geometry"]
                store = training_store
                extraction = training_extraction
            else:
                records = nontraining_records[sequence]
                bootstrap = evaluation_schedules[sequence]
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
                transform, geometry = sequence_geometry(
                    records[0], calibration, config,
                )
                hidden_keys = {
                    row.identity.key for row in records
                    if roles[row.identity.key] == "hidden"
                }
                sequence_prepare = temporary / f"h1_prepare_{sequence}"
                store, extraction = extract_parallel(
                    records, hidden_keys, calibration, config, sequence_prepare,
                    transform, devices=(0, 1, 2), h1_hidden_keys=hidden_keys,
                )
            evaluation_inputs[sequence] = {
                "records": records, "roles": roles, "schedule": schedule,
                "transform": transform, "geometry": geometry,
            }
            evaluation_stores[sequence] = store
            evaluation_preparation[sequence] = extraction
        evaluation_jobs = []
        for sequence in requested:
            values = evaluation_inputs[sequence]
            descriptor = _feature_store_descriptor(evaluation_stores[sequence])
            for condition in CONDITIONS:
                evaluation_jobs.append({
                    "kind": "formal_h1_condition", "config": config,
                    "sequence": sequence, "condition": condition,
                    "calibration": calibration, "checkpoint": str(checkpoint_path),
                    "records": values["records"], "roles": values["roles"],
                    "transform": values["transform"], "store": descriptor,
                })
        cleanup_before_trajectories = release_cuda_training_state()
        require_lifecycle_cleanup(cleanup_before_trajectories)
        completed, evaluation_scheduler = run_sequential_trajectory_jobs(
            evaluation_jobs, temporary / "h1_evaluation_jobs", cpu_profile=stage_c,
            hardware=formal_runtime["hardware"],
        )
        completed_by_sequence = {sequence: [] for sequence in requested}
        for job in completed:
            completed_by_sequence[job["sequence"]].append(job)
        for sequence in requested:
            base, provenance = base_lineage(
                config, sequence, evaluation_source_files,
                h0_contract_sha256=VISUAL_STATE_CONTRACT_SHA256,
                h1_bridge_sha256=bridge_sha256,
            )
            provenance = provenance | {
                "config_file": str(config_path.relative_to(REPO_ROOT)),
                "config_file_sha256": sha256_file(config_path),
                "execution_backend_provenance": execution_provenance(),
                "formal_hardware": formal_runtime,
            }
            output = staged_root / "sequences" / sequence
            values = evaluation_inputs[sequence]
            sequence_jobs = completed_by_sequence[sequence]
            result = _assemble_sequence(
                values["records"], values["schedule"], values["roles"],
                sequence_jobs, config, output,
            )
            extraction = evaluation_preparation[sequence]
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
            performance_payload["sequential_evaluation"] = evaluation_scheduler
            performance_payload["execution_backend_provenance"] = execution_provenance()
            result["performance_diagnostics"] = performance_payload
            result["oracle_extraction"] = extraction
            schedule = result["schedule"]
            lineage = complete_lineage(base, schedule["schedule_sha256"])
            write_sequence_metadata(
                output, "h1_interface", sequence, result, lineage, provenance,
            )
            index["sequences"][sequence] = sequence_entry(
                output, "h1_interface", lineage,
                bootstrap_end_candidate_index=result["schedule"]["bootstrap_end_candidate_index"],
            )

        for sequence, store in evaluation_stores.items():
            if sequence != TRAINING_SEQUENCE:
                store.close()

        training_store.close()
        index["canonical_checkpoint"] = {
            "file": "bridge.pt", "file_sha256": sha256_file(checkpoint_path),
            "state_dict_sha256": bridge_state_sha256,
            "training_lineage_sha256": training["lineage"]["training_lineage_sha256"],
            "training": training_record,
        }
        total_makespan = time.perf_counter() - command_started
        index["execution"] = {
            "schema": "phase1_formal_execution_summary_v1",
            "hardware": formal_runtime,
            "provenance": execution_provenance(),
            "parallel_preparation": training_extraction,
            "evaluation_preparation": evaluation_preparation,
            "sequential_evaluation": evaluation_scheduler,
            "training_schedule_materialization": training_schedule_execution,
            "evaluation_schedule_materialization": evaluation_schedule_execution,
            "cleanup_before_evaluation_schedules": schedule_cleanup,
            "cleanup_before_trajectories": cleanup_before_trajectories,
            "residency": residency,
            "total_makespan_seconds": total_makespan,
            "domain": "research_throughput",
        }
        write_registry_and_summary(staged_root, "h1_interface", index)
        validate_module_manifest(staged_root, "h1_interface", index)
        publish_current_canonical(staged_root, root)
    return {
        "status": "complete", "module": "h1_interface",
        "output": str(root.relative_to(REPO_ROOT)),
        "requested_sequences": list(requested),
        "fresh_sequences": list(requested),
        "checkpoint_training": "fresh",
        "run_policy": "fresh_current_canonical_replace",
        "performance_diagnostics": {
            "persistent": True,
            "sequence_result_field": "result.performance_diagnostics",
            "training_field": "canonical_checkpoint.training.performance_diagnostics",
            "aggregate_summary": "SUMMARY_H1.md",
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--sequences", nargs="+", choices=SUPPORTED_SEQUENCES,
        help="fresh bridge training and evaluation; this request replaces the canonical set",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    config, _ = load_config()
    requested = resolve_sequences(args.sequences, config["experiment"]["default_sequences"])
    print("Phase 1: H1 Interface — Representation-Interface Feasibility")
    print("Input modes: full_rgb, sparse_rgb, true_fmap, oracle_jepa_bridge")
    print(json.dumps(run(requested), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
