"""Four-mode delayed sparse-anchor inference runtime."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from latent_vslam.canonical import load_yaml, resolve_sequences, trajectory_payload, validate_trajectory_payload
from latent_vslam.bridge_checkpoint import (
    bridge_training_context, bridge_training_input, load_compatible_bridge, load_bridge_config,
)
from latent_vslam.evaluation import (admitted_native_evaluation, dense_hidden_ate, evaluate_paired_trajectory,
                         filter_groundtruth_associable_timestamps,
                         freeze_evaluation_population, plot_canonical_trajectories,
                         population_coverage)
from latent_vslam.efficiency_profiling import (
    PerformanceRecorder, PersistentPerformanceAudit, TransferLedger,
    add_cuda_worker_mapping, condition_runtime_diagnostics, performance_diagnosis,
)
from latent_vslam.prediction_pipeline import PredictionPipeline
from prediction.jepa_runtime import CompactFeatureStore, load_dpvo_domain, sequence_geometry
from latent_vslam.oracle_packet import FMapZeroContextPacket
from prediction.predictor import AnchorInterval, build_anchor_intervals, effective_records, predictor_state_sha256
from latent_vslam.profiling import (OnlineProfiler, break_even_payload,
                        graph_workload_payload, matched_wall_clock_payload,
                        transmission_payload)
from latent_vslam.protocol import (REPO_ROOT, SUPPORTED_SEQUENCES,
                       canonical_sha256, load_sequence_records,
                       post_bootstrap_ratio_roles, ratio_schedule_payload,
                       repo_path, sha256_file)
from latent_vslam.predictor_training import _field
from prediction.predictor_checkpoint import load_canonical_predictor
from latent_vslam.uniform_admission import ADMISSION_CONTRACT, validate_admission_trajectory
from latent_vslam.scientific_lineage import deployment_lineage
from latent_vslam.runtime import (PacketObservation,
                      run_deployment_observations, warmup_dpvo_frontend)
from latent_vslam.schema import VISUAL_STATE_CONTRACT_SHA256, condition_metadata
from latent_vslam.manifests import (
    base_lineage, complete_lineage, empty_index, publish_current_canonical,
    sequence_entry, validate_module_manifest, write_registry_and_summary,
    write_sequence_metadata,
)
from latent_vslam.artifact_runtime import staged_directory
from latent_vslam.execution_runtime import (
    FormalExecution, cpu_numa_layout, capture_runtime,
    execution_provenance, initialize_formal_main_process, runtime_provenance,
    release_cuda_training_state,
    require_lifecycle_cleanup, fixed_cpu_profile,
)
from latent_vslam.parallel_runtime import run_sequential_trajectory_jobs
from latent_vslam.dataset_paths import groundtruth_path

DEFAULT_CONFIG = REPO_ROOT / "configs/infer/mh01_four_modes.yaml"
TRAINING_SEQUENCE = "MH_01_easy"
TRAINING_ANCHOR_RATIO = 0.2
CONDITIONS = (
    "full_rgb_reference", "sparse_rgb_reference",
    "oracle_jepa_hidden_reference", "predicted_jepa_hidden",
)


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    config, resolved = load_yaml(path)
    fixed = {
        "seed": 1234, "training_sequence": TRAINING_SEQUENCE,
        "bootstrap_accepted_nodes": 8, "post_bootstrap_anchor_interval": 5,
        "deployment_mode": "delayed_bracketed",
        "strict_deployment": True, "timestamp_causal": False,
    }
    for key, value in fixed.items():
        if config["experiment"].get(key) != value:
            raise ValueError(f"frozen inference field changed: {key}")
    ratio = float(config["experiment"]["anchor_ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError("anchor_ratio must be in (0, 1]")
    expected = {
        "train": {"intervals": 219, "queries": 876},
        "validation": {"intervals": 72, "queries": 288},
        "test": {"intervals": 73, "queries": 292},
    }
    if config["split"].get("expected_counts") != expected:
        raise ValueError("frozen 219/72/73 training split changed")
    if config.get("admission") != ADMISSION_CONTRACT:
        raise ValueError("inference requires the Uniform Budgeted Admission contract")
    return config, resolved


def _repository_provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    return {
        "head_commit": git("rev-parse", "HEAD"),
        "worktree_dirty": bool(git("status", "--short")),
        "source_hash_is_authoritative_when_dirty": True,
    }


def _unique_identities(intervals: Sequence[AnchorInterval]) -> tuple[Any, ...]:
    rows = {identity.key: identity for interval in intervals
            for identity in (interval.anchor0, *[query.identity for query in interval.hidden],
                             interval.anchor1)}
    return tuple(sorted(rows.values(), key=lambda item: item.candidate_index))


def _load_bridge(config: Mapping[str, Any], transform: Any) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = repo_path(config["paths"]["bridge"])
    bridge_config, _ = load_bridge_config()
    _, bridge_base, _ = bridge_training_context(bridge_config)
    model, metadata, _ = load_compatible_bridge(
        path, transform, hidden_channels=int(bridge_config["bridge"]["hidden_channels"]),
        expected_training_input=bridge_training_input(bridge_base),
    )
    return model, metadata


def _evaluation_protocol(
    records: Sequence[Any], bootstrap_end: int, config: Mapping[str, Any],
) -> dict[str, Any]:
    roles = post_bootstrap_ratio_roles(
        [row.identity for row in records],
        bootstrap_end_candidate_index=int(bootstrap_end),
        anchor_ratio=float(config["experiment"]["anchor_ratio"]),
    )
    intervals = build_anchor_intervals(records, roles)
    effective, effective_payload = effective_records(records, intervals)
    keys = {row.identity.key for row in effective}
    effective_intervals = tuple(row for row in intervals if row.anchor1.key in keys)
    effective_roles = {row.identity.key: roles[row.identity.key] for row in effective}
    schedule = ratio_schedule_payload(
        [row.identity for row in effective],
        bootstrap_end_candidate_index=int(bootstrap_end),
        anchor_ratio=float(config["experiment"]["anchor_ratio"]),
    )
    return {
        "records": effective, "roles": effective_roles,
        "intervals": effective_intervals, "schedule": schedule,
        "effective_population": effective_payload,
    }


def _metadata() -> dict[str, dict[str, Any]]:
    return {
        "full_rgb_reference": condition_metadata(
            experiment_mode="inference", input_source="all_rgb_native_dpvo",
            online_allowed_fields=["rgb"], offline_reference_only=True,
            strict_deployment=False, timestamp_causal=True,
            closing_anchor_online_available=True,
        ),
        "sparse_rgb_reference": condition_metadata(
            experiment_mode="inference", input_source="anchor_rgb_native_dpvo",
            online_allowed_fields=["anchor_rgb"], offline_reference_only=True,
            strict_deployment=False, timestamp_causal=True,
            closing_anchor_online_available=True,
        ),
        "oracle_jepa_hidden_reference": condition_metadata(
            experiment_mode="inference", input_source="all_frame_oracle_jepa_to_bridge",
            online_allowed_fields=["oracle_jepa"], offline_reference_only=True,
            strict_deployment=False, timestamp_causal=True,
            closing_anchor_online_available=True,
        ),
        "predicted_jepa_hidden": condition_metadata(
            experiment_mode="inference",
            input_source="native_anchor_rgb_plus_uniform_budgeted_predicted_hidden_jepa_bridge",
            online_allowed_fields=["anchor_rgb", "anchor_identity", "hidden_identity_timestamp"],
            offline_reference_only=False, strict_deployment=True,
            timestamp_causal=False, closing_anchor_online_available=True,
        ),
    }


def _compact_runtime(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "condition_name", "processed_observation_count",
        "final_node_count_before_terminate", "final_patch_count_before_terminate",
        "rgb_uploaded_frame_count", "hidden_packet_count", "hidden_oracle_packet_count",
        "hidden_online_rgb_violation_count", "factor_count_allocated",
        "hidden_source_factor_count", "hidden_target_factor_count",
        "native_anchor_frontend_count", "dpvo_insertion_count",
        "candidate_consumed_exactly_once", "candidate_identity_order_sha256",
        "finite_trajectory", "tracking_success", "trajectory_pose_count",
        "model_training", "gradient_enabled",
        "timestamp_contract_exact", "elapsed_seconds", "peak_gpu_vram_bytes",
        "final_active_factor_count", "dpvo_graph_runtime_total_ms",
        "dpvo_graph_runtime_mean_ms_per_processed_observation",
        "dpvo_graph_runtime_definition", "dpvo_graph_runtime_legacy_definition",
        "stage_c_timing",
        "cross_process_timing_barrier",
        "representation", "admission", "observation_sampling_provenance",
        "candidate_received_exactly_once",
    )
    return {key: row.get(key) for key in keys}


@torch.no_grad()
def _store_observations(
    identities: Sequence[Any], roles: Mapping[str, str], store: CompactFeatureStore,
    transform: Any, bridge: torch.nn.Module, *, include_hidden: bool,
) -> Iterator[PacketObservation]:
    for identity in identities:
        role = roles[identity.key]
        if role == "hidden" and not include_hidden:
            continue
        field = _field(store, identity, transform, torch.device("cuda"))[None]
        fmap = bridge(field.flatten(2).transpose(1, 2))
        yield PacketObservation(
            identity, FMapZeroContextPacket(fmap[:, None]), role,
            identity.timestamp_ns,
        )


def _evaluation_population(records: Sequence[Any], roles: Mapping[str, str],
                           config: Mapping[str, Any]) -> tuple[Any, tuple[int, ...], tuple[int, ...], Path]:
    sequence = records[0].identity.sequence
    groundtruth = groundtruth_path(sequence)
    anchor_timestamps, _ = filter_groundtruth_associable_timestamps(
        groundtruth, [row.identity.timestamp_ns for row in records if roles[row.identity.key] == "anchor"],
        max_difference_ns=int(config["evaluation"]["groundtruth_max_association_ns"]),
    )
    dense_timestamps, _ = filter_groundtruth_associable_timestamps(
        groundtruth, [row.identity.timestamp_ns for row in records],
        max_difference_ns=int(config["evaluation"]["groundtruth_max_association_ns"]),
    )
    hidden_values = {row.identity.timestamp_ns for row in records if roles[row.identity.key] == "hidden"}
    hidden_timestamps = tuple(value for value in dense_timestamps if value in hidden_values)
    population = freeze_evaluation_population(
        anchor_timestamps, horizon_seconds=float(config["evaluation"]["rpe_horizon_seconds"]),
        tolerance_ns=int(config["evaluation"]["rpe_pair_tolerance_ns"]),
    )
    return population, dense_timestamps, hidden_timestamps, groundtruth


def _run_strict_replay(
    records: Sequence[Any], roles: Mapping[str, str], intervals: Sequence[AnchorInterval],
    calibration: np.ndarray, transform: Any, bridge: torch.nn.Module,
    predictor: torch.nn.Module, thresholds: Mapping[str, Any],
    config: Mapping[str, Any], temporary: Path, *,
    performance: PerformanceRecorder | None = None,
    transfer_ledger: TransferLedger | None = None,
    predictor_checkpoint: Path,
    predictor_state_hash: str,
    execution: FormalExecution | None = None,
    cpu_profile: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    height, width = load_dpvo_domain(records[0].rgb_path, calibration)[0].shape[:2]
    anchor_paths = {
        row.identity.key: row.rgb_path for row in records
        if roles[row.identity.key] == "anchor"
    }
    profiler = OnlineProfiler()
    provider = PredictionPipeline(
        anchor_paths=anchor_paths, identities=[row.identity for row in records],
        intervals=intervals, transform=transform, calibration=calibration,
        config=config, temporary=temporary, bridge=bridge, predictor=predictor,
        transport_calibration=thresholds, profiler=profiler, predict_hidden=True,
        performance=performance, transfer_ledger=transfer_ledger,
        predictor_checkpoint=predictor_checkpoint,
        predictor_state_sha256=predictor_state_hash,
        execution=execution or FormalExecution(), cpu_profile=cpu_profile,
        worker_settings=capture_runtime(int(config["experiment"]["seed"])),
    )
    temporary.mkdir(parents=True, exist_ok=False)
    warmup = warmup_dpvo_frontend(
        records[0], calibration, config, packet_runtime=True,
    )
    with provider.online_session():
        runtime, arrays = run_deployment_observations(
            provider.observations(), calibration, config,
            image_height=height, image_width=width,
            condition_name="predicted_jepa_hidden", expected_roles=roles,
            profiler=profiler, worker_barrier=provider,
            on_tracked=provider.on_tracked, intervals=intervals,
        )
    usage = provider.usage_payload()
    runtime["provider_usage"] = usage
    if len(provider.consumed_hidden_keys) != runtime["admission"]["generated_hidden_count"]:
        raise RuntimeError("hidden generation and receipt counts disagree")
    # GPUWorker retains the ready response envelope; runtime settings live
    # inside its provenance payload, not beside the response status.
    worker_seeds = {name: row["provenance"]["settings"]["seed"]
                    for name, row in usage["worker_provenance"].items()}
    if set(worker_seeds) != {"encoder", "predictor"} or set(worker_seeds.values()) != {config["experiment"]["seed"]}:
        raise RuntimeError("pipeline worker scientific seed mismatch")
    runtime["observation_sampling_provenance"]["worker_seeds"] = worker_seeds
    transfer_boundaries = {
        "stage_c": usage.get("stage_c_transfer"),
        "encoder": usage.get("workers", {}).get("encoder", {}).get("transfer"),
        "predictor": usage.get("workers", {}).get("predictor", {}).get("transfer"),
    }
    if (execution or FormalExecution()).verify_transfers:
        for boundary, payload in transfer_boundaries.items():
            verification = (payload or {}).get("payload_verification", {})
            if not verification.get("all_exact"):
                raise RuntimeError(f"{boundary} D2H/H2D payload verification is not exact")
    if not usage["anchor_encoded_exactly_once"]:
        raise RuntimeError("strict replay did not encode every anchor exactly once")
    if not usage["candidate_yielded_exactly_once"]:
        raise RuntimeError("strict replay did not yield every candidate exactly once")
    if not usage["hidden_consumed_exactly_once"]:
        raise RuntimeError("strict replay did not consume every hidden observation exactly once")
    if (usage["contains_hidden_rgb_or_path_capability"]
            or usage["contains_hidden_reference_or_groundtruth_capability"]):
        raise PermissionError("strict replay acquired a forbidden hidden/reference capability")
    online_profile = profiler.payload(
        peak_online_vram_bytes=int(runtime["peak_gpu_vram_bytes"]),
    )
    online_profile["pipeline_transfers"] = {
        "stage_c": usage.get("stage_c_transfer"),
        "host_shared_memory": usage.get("host_shared_memory_transfer"),
        "encoder": usage.get("workers", {}).get("encoder", {}).get("transfer"),
        "predictor": usage.get("workers", {}).get("predictor", {}).get("transfer"),
    }
    if (execution or FormalExecution()).decision_trace:
        online_profile["correspondence_decision_traces"] = usage.get(
            "correspondence_decision_traces", []
        )
        online_profile["numerical_acceptance"] = usage.get("numerical_acceptance")
    online_profile["pipeline_wall_segments"] = usage.get("pipeline_wall_segments")
    online_profile["anchor_completion_latency"] = usage.get("anchor_completion_latency")
    online_profile["producer_queue_backpressure"] = usage.get(
        "producer_queue_backpressure"
    )
    online_profile["peak_online_vram_main_process_bytes"] = int(
        runtime["peak_gpu_vram_bytes"]
    )
    online_profile["peak_online_vram_jepa_worker_bytes"] = int(
        provider.jepa_peak_online_vram_bytes
    )
    online_profile["peak_online_vram_predictor_worker_bytes"] = int(
        provider.worker_diagnostics["predictor"]["peak_vram_bytes"]
    )
    online_profile["peak_online_vram_combined_process_sum_bytes"] = (
        online_profile["peak_online_vram_main_process_bytes"]
        + online_profile["peak_online_vram_jepa_worker_bytes"]
        + online_profile["peak_online_vram_predictor_worker_bytes"]
    )
    online_profile["stage_c_runtime"] = runtime_provenance(
        provider.settings, component="stage_c_native_frontend_bridge_dpvo",
        model=bridge, amp=False,
    )
    online_profile["warmup"] = warmup | {
        "additional_prediction_components": ["v_jepa_encoder", "predictor", "frozen_bridge"],
        "worker_online_stats_reset_after_warmup": True,
    }
    if performance is not None:
        online_profile["predictor_fine_profile"] = performance.payload()
    if transfer_ledger is not None:
        online_profile["transfer_ipc"] = transfer_ledger.payload()
    return runtime, arrays, online_profile


def _run_sequence(
    records: Sequence[Any], roles: Mapping[str, str], intervals: Sequence[AnchorInterval],
    schedule: Mapping[str, Any], calibration: np.ndarray, transform: Any,
    thresholds: Mapping[str, Any], config: Mapping[str, Any], temporary: Path,
    output: Path, *, predictor_checkpoint: Path,
    condition_rows: Sequence[Mapping[str, Any]],
    trajectory_execution: Mapping[str, Any],
) -> dict[str, Any]:
    sequence = records[0].identity.sequence
    height, width = load_dpvo_domain(records[0].rgb_path, calibration)[0].shape[:2]
    anchor_paths = {row.identity.key: row.rgb_path for row in records
                    if roles[row.identity.key] == "anchor"}
    identities = [row.identity for row in records]
    if not condition_rows:
        raise RuntimeError("inference trajectory results are missing")
    by_condition = {row["condition"]: row for row in condition_rows}
    full_row = by_condition["full_rgb_reference"]
    sparse_row = by_condition["sparse_rgb_reference"]
    oracle_row = by_condition["oracle_jepa_hidden_reference"]
    predicted_row = by_condition["predicted_jepa_hidden"]
    full_runtime, full_arrays = full_row["runtime"], full_row["arrays"]
    sparse_runtime, sparse_arrays = sparse_row["runtime"], sparse_row["arrays"]
    full_warmup, sparse_warmup = full_row["warmup"], sparse_row["warmup"]
    oracle_runtime, oracle_arrays = oracle_row["runtime"], oracle_row["arrays"]
    predicted_runtime = predicted_row["runtime"]
    predicted_arrays = predicted_row["arrays"]
    online_profile = predicted_row["online_profile"]
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(oracle_row["diagnostic_path"], output / "feature_diagnostics.png")
    diagnostics = oracle_row["diagnostics"]
    runtimes = {
        "full_rgb_reference": full_runtime, "sparse_rgb_reference": sparse_runtime,
        "oracle_jepa_hidden_reference": oracle_runtime,
        "predicted_jepa_hidden": predicted_runtime,
    }
    arrays = {
        "full_rgb_reference": full_arrays, "sparse_rgb_reference": sparse_arrays,
        "oracle_jepa_hidden_reference": oracle_arrays,
        "predicted_jepa_hidden": predicted_arrays,
    }
    population, dense_timestamps, hidden_timestamps, groundtruth = _evaluation_population(
        records, roles, config,
    )
    metadata = _metadata()
    conditions = {}
    for name in CONDITIONS:
        coverage = population_coverage(arrays[name], population.common_anchor_timestamps_ns)
        if coverage["canonical_pose_coverage"] != 1.0:
            raise RuntimeError(f"incomplete inference coverage: {sequence}/{name}")
        dense = None
        if name in {"full_rgb_reference", "oracle_jepa_hidden_reference"}:
            dense = dense_hidden_ate(arrays[name], dense_timestamps, hidden_timestamps, groundtruth)
        conditions[name] = {
            **metadata[name], "runtime": _compact_runtime(runtimes[name]),
            "canonical_coverage": coverage,
            "canonical_evaluation": evaluate_paired_trajectory(arrays[name], population, groundtruth),
            "dense_hidden_evaluation": dense,
        }
        if name == "predicted_jepa_hidden":
            if not predicted_runtime["tracking_success"]:
                raise RuntimeError("canonical Ours tracking failed")
            conditions[name]["secondary_native_evaluation"] = admitted_native_evaluation(
                arrays[name], records, roles, groundtruth, config)
    conditions["predicted_jepa_hidden"]["runtime"]["provider_usage"] = (
        predicted_runtime["provider_usage"]
    )
    for name, row in (("oracle_jepa_hidden_reference", oracle_row),):
        if row.get("condition_timing_scopes") is not None:
            conditions[name]["execution_timing_scopes"] = row[
                "condition_timing_scopes"
            ]
    bundle = trajectory_payload(records, roles, arrays, population, groundtruth)
    for key in ("admission_identities", "admission_insert"):
        bundle["predicted_jepa_hidden__" + key] = predicted_arrays[key]
    np.savez_compressed(output / "trajectories.npz", **bundle)
    validate_trajectory_payload(output / "trajectories.npz", CONDITIONS)
    with np.load(output / "trajectories.npz", allow_pickle=False) as stored:
        prefix = "predicted_jepa_hidden__"
        restored = {key: stored[prefix + key] for key in (
            "admission_identities", "admission_insert", "timestamps_ns")}
        restored["poses"] = stored[prefix + "poses_raw"]
        validate_admission_trajectory(
            restored, records, roles, int(config["experiment"]["post_bootstrap_anchor_interval"]))
    plot_canonical_trajectories(
        output / "trajectory.png", arrays, population, groundtruth, sequence=sequence,
        labels={
            "full_rgb_reference": "Full RGB", "sparse_rgb_reference": "Sparse RGB",
            "oracle_jepa_hidden_reference": "Oracle JEPA hidden",
            "predicted_jepa_hidden": "Predicted JEPA hidden",
        },
    )
    transmission = transmission_payload(records, dict(roles))
    matched_wall = matched_wall_clock_payload(
        full_rgb_seconds=float(full_runtime["elapsed_seconds"]),
        ours_seconds=float(predicted_runtime["elapsed_seconds"]),
        sparse_rgb_seconds=float(sparse_runtime["elapsed_seconds"]),
    )
    graph_workload = {
        "full_rgb": graph_workload_payload(full_runtime),
        "h2": graph_workload_payload(predicted_runtime),
        "sparse_rgb": graph_workload_payload(sparse_runtime),
    }
    efficiency = {
        "transmission": transmission,
        "timing_protocol": {
            "schema": "matched_online_wall_clock_cross_process_v1",
            "sequence": sequence,
            "effective_candidate_count": len(records),
            "effective_candidate_identity_sha256": canonical_sha256(
                [identity.key for identity in identities]
            ),
            "measurement_order": list(CONDITIONS),
            "control_condition_timing_concurrent": False,
            "maximum_concurrent_dpvo_instances": 1,
            "timer": "time_perf_counter_with_cuda_synchronize_boundaries",
            "start": "after_model_load_component_warmup_worker_online_ready_and_main_cuda_sync",
            "stop": "after_dpvo_terminate_worker_flush_ack_and_main_cuda_sync",
            "included": ["online_decode_preprocess", "online_ipc", "online_inference",
                         "dpvo_frontend", "dpvo_graph_runtime", "worker_flush_ack"],
            "excluded": ["training", "model_load", "checkpoint_load", "warmup",
                         "offline_oracle_extraction", "offline_teacher_extraction",
                         "diagnostics", "artifact_serialization"],
            "warmup": {"h2": online_profile["warmup"],
                       "full_rgb": full_warmup, "sparse_rgb": sparse_warmup},
            "cross_process_barrier": predicted_runtime["cross_process_timing_barrier"],
            "stateful_dpvo_graph_warmup": False,
            "timed_dpvo_instances_start_fresh": True,
        },
        "matched_online_wall_clock": matched_wall,
        "prediction_stage_profile": online_profile,
        "graph_workload": graph_workload,
        "break_even_uplink_bandwidth": break_even_payload(
            encoded_full_bytes=transmission["encoded_full_bytes"],
            encoded_anchor_bytes=transmission["encoded_anchor_bytes"],
            extra_cloud_compute_s=matched_wall["extra_cloud_compute_s"],
        ),
        "interpretation_guard": (
            "wall_time_must_be_interpreted_with_trajectory_quality_and_graph_workload;"
            "a_shorter_prediction_runtime_is_not_automatically_a_compute_efficiency_improvement"
        ),
        "trajectory_execution": trajectory_execution,
    }
    return {
        "evaluation_role": ("predictor_development_in_sequence_feasibility"
                            if sequence == TRAINING_SEQUENCE else "frozen_predictor_zero_shot"),
        "schedule": dict(schedule), "candidate_count": len(records),
        "anchor_count": len(anchor_paths), "hidden_count": len(records) - len(anchor_paths),
        "canonical_population": population.to_dict(),
        "canonical_population_sha256": population.population_sha256,
        "conditions": conditions, "efficiency": efficiency,
        "feature_diagnostics": diagnostics,
        "offline_reference": {
            "closed_before_strict_deployment": True,
            "control_workers_released_before_strict_deployment": True,
            "oracle_condition_timing_scopes": oracle_row.get(
                "condition_timing_scopes"
            ),
            **oracle_row["offline_reference"],
        },
    }


def run(sequences: Sequence[str], *, config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Evaluate the verified canonical predictor; never train or recalibrate it."""
    command_started = time.perf_counter()
    config, config_path = load_config(config_path)
    requested = resolve_sequences(sequences, config["experiment"]["default_sequences"])
    checkpoint, checkpoint_path, predictor_sha256 = load_canonical_predictor(config)
    formal_runtime = initialize_formal_main_process(require_online=True)
    root = repo_path(config["paths"]["output_root"])
    calibration = np.loadtxt(repo_path(config["paths"]["calibration"]))
    records_by_sequence = {sequence: load_sequence_records(config, sequence) for sequence in requested}
    first_records = records_by_sequence[requested[0]]
    transform, _ = sequence_geometry(first_records[0], calibration, config)
    bridge, bridge_meta = _load_bridge(config, transform)
    thresholds = checkpoint["train_only_calibration"]
    bridge_state = {name: value.detach().cpu().clone() for name, value in bridge.state_dict().items()}
    predictor_state = checkpoint["state_dict"]
    bridge.cpu()
    del bridge
    require_lifecycle_cleanup(release_cuda_training_state())
    prediction_sources = {"predictor.py", "predictor_checkpoint.py", "transport.py", "jepa_runtime.py"}
    evaluation_source_files = tuple(
        REPO_ROOT / "prediction" / name if name in prediction_sources else Path(__file__).with_name(name)
        for name in (
        "inference_runtime.py", "prediction_pipeline.py", "prediction_deployment.py", "pipeline_worker.py",
        "predictor.py", "predictor_checkpoint.py", "transport.py", "jepa_fmap.py",
        "jepa_runtime.py", "oracle_packet.py", "protocol.py", "schema.py",
        "runtime.py", "dpvo_backend.py", "evaluation.py", "manifests.py", "canonical.py",
        "uniform_admission.py", "observation_sampling.py", "scientific_lineage.py",
    ))
    with staged_directory(root) as temporary:
        staged_root = temporary / "results"
        (staged_root / "sequences").mkdir(parents=True)
        index = empty_index("inference", requested)
        layout = cpu_numa_layout(formal_runtime["hardware"])
        selected_profile = {
            "schema": "inference_fixed_cpu_profile_v1",
            "selection": "fixed_cpu_assignment", "dynamic_calibration": False,
            "components": fixed_cpu_profile(layout),
        }
        schedule_rows, evaluation_schedule_execution = run_sequential_trajectory_jobs(
            [{"kind": "materialize_schedule", "records": records,
              "calibration": calibration, "config": config, "sequence": sequence}
             for sequence, records in records_by_sequence.items()],
            temporary / "evaluation_schedules", cpu_profile=selected_profile["components"]["stage_c"],
            hardware=formal_runtime["hardware"],
        )
        evaluation_by_sequence = {}
        for row in schedule_rows:
            sequence = row["sequence"]
            evaluation = _evaluation_protocol(records_by_sequence[sequence],
                int(row["schedule"]["bootstrap_end_candidate_index"]), config)
            transform_eval, geometry_eval = sequence_geometry(evaluation["records"][0], calibration, config)
            evaluation_by_sequence[sequence] = evaluation, transform_eval, geometry_eval
        cleanup_before_trajectories = release_cuda_training_state()
        require_lifecycle_cleanup(cleanup_before_trajectories)
        trajectory_tasks = []
        for sequence in requested:
            evaluation, transform_eval, _ = evaluation_by_sequence[sequence]
            common = {
                "records": evaluation["records"], "roles": evaluation["roles"],
                "calibration": calibration, "config": config,
            }
            trajectory_tasks.extend((
                {"kind": "infer_full_rgb", **common},
                {"kind": "infer_sparse_rgb", **common},
                {"kind": "infer_oracle_jepa", **common,
                 "condition": "oracle_jepa_hidden_reference",
                 "intervals": evaluation["intervals"], "transform": transform_eval,
                 "thresholds": thresholds, "bridge_state": bridge_state,
                 "predictor_state": predictor_state},
                {"kind": "infer_ours", **common,
                 "intervals": evaluation["intervals"], "transform": transform_eval,
                 "thresholds": thresholds, "bridge_state": bridge_state,
                 "predictor_state": predictor_state,
                 "predictor_checkpoint": str(checkpoint_path),
                 "predictor_state_hash": predictor_state_sha256(predictor_state),
                 "pipeline_cpu_profile": selected_profile,
                 "execution": dataclasses.asdict(FormalExecution.from_pool())},
            ))
        with PersistentPerformanceAudit(
            "inference", "sequential_trajectory_evaluation",
            components=("formal_coordinator",),
        ) as trajectory_performance:
            with trajectory_performance.phase("sequential_gpu0_trajectories"):
                trajectory_rows, trajectory_execution = run_sequential_trajectory_jobs(
                    trajectory_tasks, temporary / "inference_trajectory_jobs",
                    cpu_profile=selected_profile["components"]["stage_c"],
                    hardware=formal_runtime["hardware"],
                )
        trajectory_performance_payload = trajectory_performance.payload()
        trajectory_performance_payload["sequential_evaluation"] = trajectory_execution
        trajectory_performance_payload["diagnosis"] = performance_diagnosis(
            trajectory_performance_payload
        )
        trajectories_by_sequence = {sequence: [] for sequence in requested}
        for row in trajectory_rows:
            trajectories_by_sequence[row["sequence"]].append(row)
        predicted_sequence_components = {}
        for sequence in requested:
            records = records_by_sequence[sequence]
            base, provenance = base_lineage(
                config, sequence, evaluation_source_files,
                h0_contract_sha256=VISUAL_STATE_CONTRACT_SHA256,
                h1_bridge_sha256=bridge_meta["file_sha256"],
                h2_predictor_sha256=predictor_sha256,
            )
            deployment = deployment_lineage(
                config, predictor_sha256=checkpoint["state_dict_sha256"],
                training_lineage_sha256=checkpoint["training_lineage"]["training_lineage_sha256"],
                bridge_sha256=bridge_meta["file_sha256"],
                dpvo_sha256=provenance["config_protocol"]["dpvo_checkpoint_sha256"],
                dpvo_config_sha256=provenance["config_protocol"]["dpvo_config_sha256"],
                schedule_sha256=evaluation_by_sequence[sequence][0]["schedule"]["schedule_sha256"],
            )
            provenance = provenance | {
                "predictor_training_lineage_sha256": checkpoint["training_lineage"]["training_lineage_sha256"],
                "deployment_lineage": deployment,
                "config_file": str(config_path.relative_to(REPO_ROOT)),
                "config_file_sha256": sha256_file(config_path),
                "prior_results_json_reads": 0,
                "repository": _repository_provenance(),
                "checkpoint_lineage": {
                    "bridge_sha256": bridge_meta["file_sha256"],
                    "predictor_sha256": predictor_sha256,
                    "dpvo_checkpoint_sha256": provenance["config_protocol"]["dpvo_checkpoint_sha256"],
                    "vjepa_checkpoint_sha256": config["jepa"]["checkpoint_sha256"],
                },
                "execution_backend_provenance": execution_provenance(),
                "formal_hardware": formal_runtime,
            }
            evaluation, transform_eval, geometry_eval = evaluation_by_sequence[sequence]
            sequence_temp = temporary / f"sequence_work_{sequence}"
            sequence_temp.mkdir()
            output = staged_root / "sequences" / sequence
            with PersistentPerformanceAudit(
                "inference", f"sequence:{sequence}",
                components=("artifact_assembly_main_process",),
            ) as performance:
                with performance.phase("sequence_evaluation_and_artifact_generation"):
                    sequence_started = time.perf_counter()
                    result = _run_sequence(
                        evaluation["records"], evaluation["roles"],
                        evaluation["intervals"], evaluation["schedule"], calibration,
                        transform_eval, thresholds, config,
                        sequence_temp, output,
                        predictor_checkpoint=checkpoint_path,
                        condition_rows=trajectories_by_sequence[sequence],
                        trajectory_execution=trajectory_execution,
                    )
                    sequence_total_seconds = time.perf_counter() - sequence_started
            online_seconds = float(
                result["conditions"]["predicted_jepa_hidden"]["runtime"][
                    "elapsed_seconds"
                ]
            )
            predicted_sequence_components[sequence] = {
                "online_pipeline_seconds": online_seconds,
                "artifact_and_evaluation_seconds": sequence_total_seconds,
                "total_seconds": sequence_total_seconds + online_seconds,
            }
            performance_payload = performance.payload()
            performance_payload["condition_runtime"] = condition_runtime_diagnostics(result)
            worker_usage = result["conditions"]["predicted_jepa_hidden"]["runtime"][
                "provider_usage"
            ]
            add_cuda_worker_mapping(performance_payload, {
                "worker_pid": worker_usage.get("jepa_worker_pid"),
                "logical_cuda_ordinal": worker_usage.get(
                    "jepa_worker_logical_cuda_ordinal", int(FormalExecution.from_pool().encoder_device),
                ),
            })
            performance_payload["strict_predictor"] = result["efficiency"][
                "prediction_stage_profile"
            ].get("predictor_fine_profile")
            performance_payload["strict_transfer_ipc"] = result["efficiency"][
                "prediction_stage_profile"
            ].get("transfer_ipc")
            performance_payload["diagnosis"] = performance_diagnosis(performance_payload)
            performance_payload["execution_backend_provenance"] = execution_provenance()
            performance_payload["formal_hardware"] = formal_runtime
            result["deployment_lineage"] = deployment
            result["performance_diagnostics"] = performance_payload
            result["coordinate_transform"] = geometry_eval
            result["effective_population"] = evaluation["effective_population"]
            lineage = complete_lineage(base, evaluation["schedule"]["schedule_sha256"])
            write_sequence_metadata(
                output, "inference", sequence, result, lineage, provenance,
            )
            index["sequences"][sequence] = sequence_entry(
                output, "inference", lineage,
                bootstrap_end_candidate_index=result["schedule"]["bootstrap_end_candidate_index"],
            )

        index["model_manifest"] = {"predictor": {
            "file": str(checkpoint_path.relative_to(REPO_ROOT)), "path_base": "repository_root",
            "seed": checkpoint["training_recipe"]["seed"], "best_epoch": checkpoint["best_epoch"],
            "file_sha256": predictor_sha256, "state_dict_sha256": checkpoint["state_dict_sha256"],
            "training_lineage_sha256": checkpoint["training_lineage"]["training_lineage_sha256"],
            "scientific_lineage": checkpoint["training_lineage"],
            "training_performed": False,
            "promotion_provenance": {k: v for k, v in checkpoint.get("promotion_provenance", {}).items()
                                     if k != "source_checkpoint_metadata"},
        }}
        index["model_manifest"]["bridge"] = bridge_meta
        index["deployment_manifest"] = {
            sequence: json.loads((staged_root / "sequences" / sequence / "results.json").read_text())[
                "result"]["deployment_lineage"] for sequence in requested
        }
        index["run_manifest"] = {
            "hardware": formal_runtime, "provenance": execution_provenance(),
            "evaluation_schedule_materialization": evaluation_schedule_execution,
            "sequential_trajectory_execution": trajectory_execution,
            "trajectory_performance": trajectory_performance_payload,
            "cpu_numa_profile": selected_profile, "cleanup_before_trajectories": cleanup_before_trajectories,
            "predicted_jepa_sequence_policy": "sequential_exclusive_three_gpu_pipeline",
            "predicted_jepa_total_seconds": sum(
                row["online_pipeline_seconds"] for row in predicted_sequence_components.values()),
            "predicted_jepa_sequence_components": predicted_sequence_components,
            "total_makespan_seconds": time.perf_counter() - command_started,
            "training_performed": False,
        }
        write_registry_and_summary(staged_root, "inference", index)
        validate_module_manifest(staged_root, "inference", index)
        if sha256_file(checkpoint_path) != predictor_sha256:
            raise RuntimeError("predictor changed during inference")
        if sha256_file(repo_path(config["paths"]["bridge"])) != bridge_meta["file_sha256"]:
            raise RuntimeError("Bridge changed during inference")
        publish_current_canonical(staged_root, root)
    return {
        "status": "complete", "module": "inference", "output": str(root.relative_to(REPO_ROOT)),
        "requested_sequences": list(requested), "fresh_sequences": list(requested),
        "checkpoint_training": "not_performed", "run_policy": "fresh_current_canonical_replace",
        "predictor_state_dict_sha256": checkpoint["state_dict_sha256"],
    }
