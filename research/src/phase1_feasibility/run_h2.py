"""H2 Prediction: strict delayed sparse-anchor prediction feasibility."""

from __future__ import annotations

import argparse
import copy
import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from .canonical import load_yaml, resolve_sequences, trajectory_payload, validate_trajectory_payload
from .bridge_checkpoint import (
    h1_training_context, h1_training_input, load_compatible_bridge, load_h1_config,
)
from .evaluation import (dense_hidden_ate, evaluate_paired_trajectory,
                         filter_groundtruth_associable_timestamps,
                         freeze_evaluation_population, plot_canonical_trajectories,
                         population_coverage)
from .efficiency_profiling import (
    PerformanceRecorder, PersistentPerformanceAudit, TransferLedger,
    add_cuda_worker_mapping, condition_runtime_diagnostics, performance_diagnosis,
)
from .h2_deployment import DelayedDeploymentProvider
from .jepa_fmap import build_bridge, coordinate_masks, tokens_to_field
from .jepa_runtime import (CompactFeatureStore, RestrictedFeatureView, extract_block5_store,
                           extract_true_fmap_store, load_dpvo_domain,
                           sequence_geometry)
from .oracle_packet import FMapZeroContextPacket
from .predictor import (AnchorInterval, RobustTransportBlock5Predictor,
                        build_anchor_intervals, effective_records,
                        predictor_metadata, predictor_state_sha256,
                        split_anchor_intervals)
from .profiling import (OnlineProfiler, break_even_payload,
                        graph_workload_payload, matched_wall_clock_payload,
                        transmission_payload)
from .protocol import (REPO_ROOT, SUPPORTED_SEQUENCES, atomic_write_bytes,
                       canonical_sha256, load_sequence_records,
                       post_bootstrap_ratio_roles, ratio_schedule_payload,
                       repo_path, sha256_file)
from .h2_training import (_field, _hidden_identities, _plot_feature_diagnostics,
                          build_robust_correspondence_store,
                          calibrate_train_only_thresholds, held_out_representation,
                          tiny_overfit, train_predictor)
from .transport import robust_protocol_metadata
from .runtime import (OnlineFrame, PacketObservation, materialize_schedule,
                      run_deployment_observations, run_formal_mode,
                      run_packet_observations,
                      sanitize_full_oracle_frames, warmup_dpvo_frontend)
from .schema import VISUAL_STATE_CONTRACT_SHA256, condition_metadata
from .registry import (
    base_lineage, complete_lineage, empty_index, publish_current_canonical,
    sequence_entry, validate_module_manifest, write_registry_and_summary,
    write_sequence_metadata,
)

DEFAULT_CONFIG = REPO_ROOT / "research/configs/phase1_feasibility_h2.yaml"
TRAINING_SEQUENCE = "MH_01_easy"
TRAINING_ANCHOR_RATIO = 0.2
CONDITIONS = (
    "full_rgb_reference", "sparse_rgb_reference", "anchor_jepa_only",
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
            raise ValueError(f"frozen H2 field changed: {key}")
    ratio = float(config["experiment"]["anchor_ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError("anchor_ratio must be in (0, 1]")
    expected = {
        "train": {"intervals": 219, "queries": 876},
        "validation": {"intervals": 72, "queries": 288},
        "test": {"intervals": 73, "queries": 292},
    }
    if config["split"].get("expected_counts") != expected:
        raise ValueError("frozen H2 219/72/73 split changed")
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
    path = repo_path(config["paths"]["h1_bridge"])
    h1_config, _ = load_h1_config()
    _, h1_base, _ = h1_training_context(h1_config)
    model, metadata, _ = load_compatible_bridge(
        path, transform, hidden_channels=int(h1_config["bridge"]["hidden_channels"]),
        expected_training_input=h1_training_input(h1_base),
    )
    return model, metadata


def _save_predictor(path: Path, model: torch.nn.Module, config: Mapping[str, Any],
                    calibration: Mapping[str, Any], lineage: Mapping[str, Any]) -> dict[str, Any]:
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    payload = {
        "schema_version": 1, "state_dict": state,
        "state_dict_sha256": predictor_state_sha256(state),
        "architecture": predictor_metadata(model),
        "deployment_protocol": {
            "mode": "delayed_bracketed", "strict_deployment": True,
            "timestamp_causal": False, "closing_anchor_online_availability_required": True,
            "online_input": "uploaded_anchor_rgb_only",
        },
        "training_recipe": dict(config["training"]) | {
            "seed": 1234, "training_sequence": TRAINING_SEQUENCE,
            "target": "offline_oracle_hidden_jepa_block5",
            "checkpoint_contains_optimizer_or_scaler": False,
        },
        "train_only_calibration": dict(calibration),
        "training_lineage": dict(lineage),
    }
    buffer = io.BytesIO(); torch.save(payload, buffer)
    atomic_write_bytes(path, buffer.getvalue())
    return payload


def _new_predictor(config: Mapping[str, Any]) -> RobustTransportBlock5Predictor:
    value = config["predictor"]
    return RobustTransportBlock5Predictor(
        int(value["feature_dim"]), int(value["hidden_dim"]),
        int(value["difference_dim"]), int(value["reliability_dim"]),
        int(value["residual_blocks"]), int(value["group_norm_groups"]),
        int(value["time_hidden_dim"]),
    )


def _predictor_training_details(
    records: Sequence[Any], bootstrap_end: int, calibration: np.ndarray,
    config: Mapping[str, Any], base: Mapping[str, Any],
) -> dict[str, Any]:
    roles = post_bootstrap_ratio_roles(
        [row.identity for row in records],
        bootstrap_end_candidate_index=int(bootstrap_end),
        anchor_ratio=TRAINING_ANCHOR_RATIO,
    )
    intervals = build_anchor_intervals(records, roles)
    split, split_payload = split_anchor_intervals(
        intervals, float(config["split"]["train_fraction"]),
        float(config["split"]["validation_fraction"]),
    )
    counts = {
        key: {"intervals": len(values), "queries": sum(len(row.hidden) for row in values)}
        for key, values in split.items()
    }
    if counts != config["split"]["expected_counts"]:
        raise RuntimeError(f"H2 split changed: {counts}")
    schedule = ratio_schedule_payload(
        [row.identity for row in records],
        bootstrap_end_candidate_index=int(bootstrap_end),
        anchor_ratio=TRAINING_ANCHOR_RATIO,
    )
    transform, geometry = sequence_geometry(records[0], calibration, config)
    lineage = {
        "training_input": {
            key: value for key, value in base.items()
            if key not in {"schedule_sha256", "h2_predictor_sha256"}
        },
        "bootstrap_end_candidate_index": int(bootstrap_end),
        "training_schedule_sha256": schedule["schedule_sha256"],
        "split_sha256": split_payload["split_sha256"],
        "split_counts": counts,
        "coordinate_transform_sha256": geometry["transform_sha256"],
        "transport_calibration_protocol_sha256": canonical_sha256(
            robust_protocol_metadata()
        ),
    }
    return {
        "roles": roles, "intervals": intervals, "split": split,
        "split_payload": split_payload, "counts": counts, "schedule": schedule,
        "transform": transform, "geometry": geometry, "lineage": lineage,
    }


def _validate_fresh_predictor(
    path: Path, config: Mapping[str, Any], expected_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError("fresh H2 predictor checkpoint is unreadable") from error
    reasons: list[str] = []
    forbidden = {"optimizer", "optimizer_state_dict", "grad_scaler", "best_state"}
    if forbidden & set(checkpoint):
        reasons.append("checkpoint_contains_training_state")
    if checkpoint.get("schema_version") != 1 or not isinstance(checkpoint.get("training_recipe"), Mapping):
        reasons.append("checkpoint_schema_mismatch")
    if "state_dict" not in checkpoint:
        reasons.append("checkpoint_schema_mismatch")
    elif predictor_state_sha256(checkpoint["state_dict"]) != checkpoint.get("state_dict_sha256"):
        reasons.append("state_dict_integrity_mismatch")
    stored = checkpoint.get("training_lineage")
    if not isinstance(stored, Mapping):
        reasons.append("training_lineage_missing")
    else:
        if dict(stored) != dict(expected_lineage):
            reasons.append("training_lineage_mismatch")
        lineage_hash = stored.get("training_lineage_sha256")
        lineage_body = {key: value for key, value in stored.items()
                        if key != "training_lineage_sha256"}
        if lineage_hash != canonical_sha256(lineage_body):
            reasons.append("training_lineage_integrity_mismatch")
        threshold = checkpoint.get("train_only_calibration")
        if not isinstance(threshold, Mapping):
            reasons.append("train_calibration_missing")
        else:
            body = dict(threshold)
            stored_calibration_hash = body.pop("calibration_sha256", None)
            if stored_calibration_hash != canonical_sha256(body):
                reasons.append("train_calibration_integrity_mismatch")
            if stored.get("train_only_calibration_sha256") != stored_calibration_hash:
                reasons.append("training_calibration_lineage_mismatch")
    model = _new_predictor(config)
    if checkpoint.get("architecture") != predictor_metadata(model):
        reasons.append("architecture_mismatch")
    deployment = checkpoint.get("deployment_protocol", {})
    if deployment.get("mode") != "delayed_bracketed" or deployment.get("timestamp_causal") is not False:
        reasons.append("deployment_protocol_mismatch")
    if reasons:
        raise RuntimeError(f"fresh H2 predictor checkpoint validation failed: {sorted(set(reasons))}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return checkpoint


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
            experiment_mode="h2_prediction", input_source="all_rgb_native_dpvo",
            online_allowed_fields=["rgb"], offline_reference_only=True,
            strict_deployment=False, timestamp_causal=True,
            closing_anchor_online_available=True,
        ),
        "sparse_rgb_reference": condition_metadata(
            experiment_mode="h2_prediction", input_source="anchor_rgb_native_dpvo",
            online_allowed_fields=["anchor_rgb"], offline_reference_only=True,
            strict_deployment=False, timestamp_causal=True,
            closing_anchor_online_available=True,
        ),
        "anchor_jepa_only": condition_metadata(
            experiment_mode="h2_prediction", input_source="anchor_rgb_to_jepa_to_bridge",
            online_allowed_fields=["anchor_rgb"], offline_reference_only=True,
            strict_deployment=False, timestamp_causal=True,
            closing_anchor_online_available=True,
        ),
        "oracle_jepa_hidden_reference": condition_metadata(
            experiment_mode="h2_prediction", input_source="all_frame_oracle_jepa_to_bridge",
            online_allowed_fields=["oracle_jepa"], offline_reference_only=True,
            strict_deployment=False, timestamp_causal=True,
            closing_anchor_online_available=True,
        ),
        "predicted_jepa_hidden": condition_metadata(
            experiment_mode="h2_prediction",
            input_source="native_anchor_rgb_plus_predicted_hidden_jepa_bridge",
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
        "timestamp_contract_exact", "elapsed_seconds", "peak_gpu_vram_bytes",
        "final_active_factor_count", "dpvo_graph_runtime_total_ms",
        "dpvo_graph_runtime_mean_ms_per_processed_observation",
        "cross_process_timing_barrier",
        "representation",
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
    groundtruth = repo_path(config["dataset"]["groundtruth_pattern"].format(sequence=sequence))
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
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    height, width = load_dpvo_domain(records[0].rgb_path, calibration)[0].shape[:2]
    anchor_paths = {
        row.identity.key: row.rgb_path for row in records
        if roles[row.identity.key] == "anchor"
    }
    profiler = OnlineProfiler()
    provider = DelayedDeploymentProvider(
        anchor_paths=anchor_paths, identities=[row.identity for row in records],
        intervals=intervals, transform=transform, calibration=calibration,
        config=config, temporary=temporary, bridge=bridge, predictor=predictor,
        transport_calibration=thresholds, profiler=profiler, predict_hidden=True,
        performance=performance, transfer_ledger=transfer_ledger,
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
            on_tracked=provider.on_tracked,
        )
    usage = provider.usage_payload()
    runtime["provider_usage"] = usage
    if not usage["anchor_encoded_exactly_once"]:
        raise RuntimeError("strict replay did not encode every anchor exactly once")
    if not usage["candidate_yielded_exactly_once"]:
        raise RuntimeError("strict replay did not yield every candidate exactly once")
    online_profile = profiler.payload(
        peak_online_vram_bytes=int(runtime["peak_gpu_vram_bytes"]),
    )
    online_profile["peak_online_vram_main_process_bytes"] = int(
        runtime["peak_gpu_vram_bytes"]
    )
    online_profile["peak_online_vram_jepa_worker_bytes"] = int(
        provider.jepa_peak_online_vram_bytes
    )
    online_profile["peak_online_vram_combined_process_sum_bytes"] = (
        online_profile["peak_online_vram_main_process_bytes"]
        + online_profile["peak_online_vram_jepa_worker_bytes"]
    )
    online_profile["warmup"] = warmup | {
        "additional_h2_components": ["v_jepa_encoder", "predictor", "frozen_h1_bridge"],
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
    bridge: torch.nn.Module, predictor: torch.nn.Module,
    thresholds: Mapping[str, Any], config: Mapping[str, Any], temporary: Path,
    output: Path,
) -> dict[str, Any]:
    sequence = records[0].identity.sequence
    height, width = load_dpvo_domain(records[0].rgb_path, calibration)[0].shape[:2]
    anchor_paths = {row.identity.key: row.rgb_path for row in records
                    if roles[row.identity.key] == "anchor"}
    identities = [row.identity for row in records]
    predictor_performance = PerformanceRecorder(enable_cuda=True)
    transfer_ledger = TransferLedger()
    predicted_runtime, predicted_arrays, online_profile = _run_strict_replay(
        records, roles, intervals, calibration, transform, bridge, predictor,
        thresholds, config, temporary / "strict_online",
        performance=predictor_performance, transfer_ledger=transfer_ledger,
    )
    # Only after strict deployment is complete may raw/oracle reference stores exist.
    all_online = [OnlineFrame(row.identity, row.rgb_path) for row in records]
    packet_online = sanitize_full_oracle_frames(records, roles)
    all_anchor = {row.identity.key: "anchor" for row in records}
    full_warmup = warmup_dpvo_frontend(
        records[0], calibration, config, packet_runtime=False,
    )
    full_runtime, full_arrays = run_formal_mode(
        "matched_full_rgb", all_online, calibration, config, roles=all_anchor,
        condition_name="full_rgb_reference",
        matched_timing=True, profile_graph_runtime=True, collect_graph_trace=False,
    )
    sparse_warmup = warmup_dpvo_frontend(
        records[0], calibration, config, packet_runtime=False,
    )
    sparse_runtime, sparse_arrays = run_formal_mode(
        "sparse_rgb", packet_online, calibration, config, roles=roles,
        condition_name="sparse_rgb_reference",
        matched_timing=True, profile_graph_runtime=True, collect_graph_trace=False,
    )
    reference_temp = temporary / "offline_reference"
    reference_temp.mkdir()
    all_store, all_extraction = extract_block5_store(
        records, identities, calibration, config, reference_temp, transform,
    )
    anchor_runtime, anchor_arrays = run_packet_observations(
        _store_observations(identities, roles, all_store, transform, bridge, include_hidden=False),
        calibration, config, image_height=height, image_width=width,
        condition_name="anchor_jepa_only",
    )
    oracle_runtime, oracle_arrays = run_packet_observations(
        _store_observations(identities, roles, all_store, transform, bridge, include_hidden=True),
        calibration, config, image_height=height, image_width=width,
        condition_name="oracle_jepa_hidden_reference",
    )
    hidden_identities = tuple(row.identity for row in records if roles[row.identity.key] == "hidden")
    true_store, true_extraction = extract_true_fmap_store(
        records, hidden_identities, calibration, config, reference_temp, transform,
    )
    mask = torch.from_numpy(coordinate_masks(transform)["valid_token_mask"]).cuda()
    robust, robust_meta = build_robust_correspondence_store(
        intervals, all_store, transform, mask, thresholds,
    )
    output.mkdir(parents=True, exist_ok=False)
    diagnostics = _plot_feature_diagnostics(
        output / "feature_diagnostics.png", sequence, intervals, all_store, true_store,
        transform, bridge, robust, predictor, config,
    )
    runtimes = {
        "full_rgb_reference": full_runtime, "sparse_rgb_reference": sparse_runtime,
        "anchor_jepa_only": anchor_runtime,
        "oracle_jepa_hidden_reference": oracle_runtime,
        "predicted_jepa_hidden": predicted_runtime,
    }
    arrays = {
        "full_rgb_reference": full_arrays, "sparse_rgb_reference": sparse_arrays,
        "anchor_jepa_only": anchor_arrays,
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
            raise RuntimeError(f"incomplete H2 coverage: {sequence}/{name}")
        dense = None
        if name in {"full_rgb_reference", "oracle_jepa_hidden_reference", "predicted_jepa_hidden"}:
            dense = dense_hidden_ate(arrays[name], dense_timestamps, hidden_timestamps, groundtruth)
        conditions[name] = {
            **metadata[name], "runtime": _compact_runtime(runtimes[name]),
            "canonical_coverage": coverage,
            "canonical_evaluation": evaluate_paired_trajectory(arrays[name], population, groundtruth),
            "dense_hidden_evaluation": dense,
        }
    conditions["predicted_jepa_hidden"]["runtime"]["provider_usage"] = (
        predicted_runtime["provider_usage"]
    )
    np.savez_compressed(output / "trajectories.npz", **trajectory_payload(
        records, roles, arrays, population, groundtruth,
    ))
    validate_trajectory_payload(output / "trajectories.npz", CONDITIONS)
    plot_canonical_trajectories(
        output / "trajectory.png", arrays, population, groundtruth, sequence=sequence,
        labels={
            "full_rgb_reference": "Full RGB", "sparse_rgb_reference": "Sparse RGB",
            "anchor_jepa_only": "Anchor JEPA only",
            "oracle_jepa_hidden_reference": "Oracle JEPA hidden",
            "predicted_jepa_hidden": "Predicted JEPA hidden",
        },
    )
    transmission = transmission_payload(records, dict(roles))
    matched_wall = matched_wall_clock_payload(
        full_rgb_seconds=float(full_runtime["elapsed_seconds"]),
        h2_seconds=float(predicted_runtime["elapsed_seconds"]),
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
            "measurement_order": ["h2", "full_rgb", "sparse_rgb"],
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
        "h2_stage_profile": online_profile,
        "graph_workload": graph_workload,
        "break_even_uplink_bandwidth": break_even_payload(
            encoded_full_bytes=transmission["encoded_full_bytes"],
            encoded_anchor_bytes=transmission["encoded_anchor_bytes"],
            extra_cloud_compute_s=matched_wall["extra_cloud_compute_s"],
        ),
        "interpretation_guard": (
            "wall_time_must_be_interpreted_with_trajectory_quality_and_graph_workload;"
            "a_shorter_h2_runtime_is_not_automatically_a_compute_efficiency_improvement"
        ),
    }
    all_store.close(); true_store.close(); shutil.rmtree(reference_temp)
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
            "created_after_strict_deployment": True,
            "block5_extraction": all_extraction,
            "true_fmap_extraction": true_extraction,
            "robust_correspondence": robust_meta,
        },
    }


def run(sequences: Sequence[str]) -> dict[str, Any]:
    config, config_path = load_config()
    requested = resolve_sequences(sequences, config["experiment"]["default_sequences"])
    root = repo_path(config["paths"]["output_root"])
    root.parent.mkdir(parents=True, exist_ok=True)
    calibration = np.loadtxt(repo_path(config["paths"]["calibration"]), delimiter=" ")
    training_source_files = tuple(Path(__file__).with_name(name) for name in (
        "run_h2.py", "h2_training.py", "predictor.py",
        "transport.py", "jepa_fmap.py", "jepa_runtime.py", "jepa_worker.py",
        "oracle_packet.py", "protocol.py", "schema.py", "efficiency_profiling.py",
    ))
    evaluation_source_files = training_source_files + tuple(
        Path(__file__).with_name(name) for name in (
            "h2_deployment.py", "profiling.py", "runtime.py", "dpvo_backend.py", "evaluation.py",
            "registry.py", "canonical.py",
        )
    )
    training_records = load_sequence_records(config, TRAINING_SEQUENCE)
    training_transform, _ = sequence_geometry(training_records[0], calibration, config)
    bridge, bridge_meta = _load_bridge(config, training_transform)
    training_config = copy.deepcopy(config)
    training_config.pop("evaluation", None)
    training_config.pop("diagnostics", None)
    training_base, _ = base_lineage(
        training_config, TRAINING_SEQUENCE, training_source_files,
        h0_contract_sha256=VISUAL_STATE_CONTRACT_SHA256,
        h1_bridge_sha256=bridge_meta["file_sha256"],
    )

    with tempfile.TemporaryDirectory(prefix=".phase1_h2_", dir=root.parent) as name:
        temporary = Path(name)
        staged_root = temporary / "h2_prediction"
        (staged_root / "sequences").mkdir(parents=True)
        checkpoint_path = staged_root / "predictor.pt"
        index = empty_index("h2_prediction", requested)

        with PersistentPerformanceAudit(
            "h2_prediction", "canonical_predictor_training",
            components=("predictor_bridge_training_main_process",),
        ) as training_performance:
            with training_performance.phase("schedule_and_split"):
                bootstrap = materialize_schedule(training_records, calibration, config)
                training = _predictor_training_details(
                    training_records, int(bootstrap["bootstrap_end_candidate_index"]),
                    calibration, config, training_base,
                )
                development = (
                    *training["split"]["train"], *training["split"]["validation"],
                )
                training_temp = temporary / "predictor_training"
            with training_performance.phase("development_jepa_extraction"):
                dev_store, dev_extraction = extract_block5_store(
                    training_records, _unique_identities(development), calibration,
                    config, training_temp, training["transform"],
                )
            with training_performance.phase("train_only_threshold_calibration"):
                mask = torch.from_numpy(
                    coordinate_masks(training["transform"])["valid_token_mask"]
                ).cuda()
                thresholds = calibrate_train_only_thresholds(
                    training["split"]["train"], dev_store,
                    training["transform"], mask,
                )
            with training_performance.phase("development_correspondence_precompute"):
                robust_dev, robust_dev_meta = build_robust_correspondence_store(
                    development, dev_store, training["transform"], mask, thresholds,
                )
            with training_performance.phase("tiny_overfit"):
                tiny = tiny_overfit(
                    dev_store, training["split"]["train"], training["transform"],
                    mask, config, robust_dev,
                )
            with training_performance.phase("predictor_training"):
                training_batch_performance = PerformanceRecorder(enable_cuda=True)
                validation_batch_performance = PerformanceRecorder(enable_cuda=True)
                predictor, training_summary = train_predictor(
                    dev_store, training["split"], training["transform"], mask,
                    config, robust_dev, profiler=training_batch_performance,
                    validation_profiler=validation_batch_performance,
                )
            with training_performance.phase("checkpoint_save_and_validation"):
                lineage = dict(training["lineage"])
                lineage["train_only_calibration_sha256"] = thresholds["calibration_sha256"]
                lineage["training_lineage_sha256"] = canonical_sha256(lineage)
                checkpoint = _save_predictor(
                    checkpoint_path, predictor, config, thresholds, lineage,
                )
                _validate_fresh_predictor(checkpoint_path, config, lineage)
                dev_store.close()
                test_temp = temporary / "predictor_test"
            with training_performance.phase("test_jepa_extraction"):
                test_store, test_extraction = extract_block5_store(
                    training_records, _unique_identities(training["split"]["test"]),
                    calibration, config, test_temp, training["transform"],
                )
            with training_performance.phase("test_true_fmap_extraction"):
                test_teacher_store, test_teacher_extraction = extract_true_fmap_store(
                    training_records, _hidden_identities(training["split"]["test"]),
                    calibration, config, test_temp / "test_teacher",
                    training["transform"],
                )
                test_teacher = RestrictedFeatureView(
                    test_teacher_store,
                    {item.key for item in _hidden_identities(training["split"]["test"])},
                    "h2_test_true_fmap_teacher",
                )
            with training_performance.phase("test_correspondence_precompute"):
                robust_test, robust_test_meta = build_robust_correspondence_store(
                    training["split"]["test"], test_store, training["transform"],
                    mask, thresholds,
                )
            with training_performance.phase("held_out_representation"):
                held_out = held_out_representation(
                    training["split"]["test"], test_store, training["transform"],
                    mask, robust_test, predictor, bridge, test_teacher,
                )
                test_teacher_usage = test_teacher.usage_payload()
                test_teacher_store.close(); test_store.close()
                stores_closed = all(
                    store.closed for store in (dev_store, test_store, test_teacher_store)
                )
                if not stores_closed:
                    raise RuntimeError(
                        "all H2 training/reference stores must close before deployment"
                    )
        training_performance_payload = training_performance.payload()
        add_cuda_worker_mapping(
            training_performance_payload, dev_extraction["jepa"],
        )
        training_performance_payload["training_throughput"] = {
            "training_batches": training_batch_performance.payload(),
            "validation_batches": validation_batch_performance.payload(),
        }
        training_performance_payload["diagnosis"] = performance_diagnosis(
            training_performance_payload
        )
        training_record = {
            "sequence": TRAINING_SEQUENCE, "lineage": lineage,
            "schedule": training["schedule"], "split": training["split_payload"],
            "split_counts": training["counts"],
            "coordinate_transform": training["geometry"],
            "training_anchor_ratio": TRAINING_ANCHOR_RATIO,
            "tiny_overfit": tiny, "summary": training_summary,
            "held_out_representation": held_out,
            "development_extraction": dev_extraction,
            "test_extraction": test_extraction,
            "offline_true_fmap_diagnostics": {
                "held_out_test": {
                    "extraction": test_teacher_extraction,
                    "usage": test_teacher_usage | {"store_closed": True},
                    "created_after_checkpoint_selection": True,
                    "created_after_checkpoint_freeze_save_and_validation": True,
                },
                "test_was_read_during_training_or_selection": False,
            },
            "development_correspondence": robust_dev_meta,
            "test_correspondence": robust_test_meta,
            "stores_closed_before_deployment": stores_closed,
            "performance_diagnostics": training_performance_payload,
        }
        del test_teacher
        del test_teacher_store
        del dev_store, test_store
        predictor_sha256 = sha256_file(checkpoint_path)
        records_by_sequence: dict[str, Sequence[Any]] = {TRAINING_SEQUENCE: training_records}
        for sequence in requested:
            if sequence not in records_by_sequence:
                records_by_sequence[sequence] = load_sequence_records(config, sequence)
            records = records_by_sequence[sequence]
            base, provenance = base_lineage(
                config, sequence, evaluation_source_files,
                h0_contract_sha256=VISUAL_STATE_CONTRACT_SHA256,
                h1_bridge_sha256=bridge_meta["file_sha256"],
                h2_predictor_sha256=predictor_sha256,
            )
            provenance = provenance | {
                "config_file": str(config_path.relative_to(REPO_ROOT)),
                "config_file_sha256": sha256_file(config_path),
                "h1_results_json_reads": 0,
                "repository": _repository_provenance(),
                "checkpoint_lineage": {
                    "h1_bridge_sha256": bridge_meta["file_sha256"],
                    "h2_predictor_sha256": predictor_sha256,
                    "dpvo_checkpoint_sha256": provenance["config_protocol"]["dpvo_checkpoint_sha256"],
                    "vjepa_checkpoint_sha256": config["jepa"]["checkpoint_sha256"],
                },
            }
            if sequence == TRAINING_SEQUENCE:
                bootstrap_end = training["lineage"]["bootstrap_end_candidate_index"]
            else:
                bootstrap = materialize_schedule(records, calibration, config)
                bootstrap_end = int(bootstrap["bootstrap_end_candidate_index"])
            evaluation = _evaluation_protocol(records, bootstrap_end, config)
            transform_eval, geometry_eval = sequence_geometry(
                evaluation["records"][0], calibration, config,
            )
            bridge_eval = build_bridge(
                transform_eval, channels=int(config["bridge"]["hidden_channels"]),
            ).cuda().eval()
            bridge_eval.load_state_dict(bridge.state_dict(), strict=True)
            bridge_eval.requires_grad_(False)
            sequence_temp = temporary / f"sequence_work_{sequence}"
            sequence_temp.mkdir()
            output = staged_root / "sequences" / sequence
            with PersistentPerformanceAudit(
                "h2_prediction", f"sequence:{sequence}",
                components=("dpvo_predictor_bridge_main_process",),
            ) as performance:
                with performance.phase("sequence_evaluation_and_artifact_generation"):
                    result = _run_sequence(
                        evaluation["records"], evaluation["roles"],
                        evaluation["intervals"], evaluation["schedule"], calibration,
                        transform_eval, bridge_eval, predictor, thresholds, config,
                        sequence_temp, output,
                    )
            performance_payload = performance.payload()
            performance_payload["condition_runtime"] = condition_runtime_diagnostics(result)
            worker_usage = result["conditions"]["predicted_jepa_hidden"]["runtime"][
                "provider_usage"
            ]
            add_cuda_worker_mapping(performance_payload, {
                "worker_pid": worker_usage.get("jepa_worker_pid"),
                "logical_cuda_ordinal": worker_usage.get(
                    "jepa_worker_logical_cuda_ordinal", 0,
                ),
            })
            performance_payload["strict_h2_predictor"] = result["efficiency"][
                "h2_stage_profile"
            ].get("predictor_fine_profile")
            performance_payload["strict_h2_transfer_ipc"] = result["efficiency"][
                "h2_stage_profile"
            ].get("transfer_ipc")
            performance_payload["diagnosis"] = performance_diagnosis(performance_payload)
            result["performance_diagnostics"] = performance_payload
            result["coordinate_transform"] = geometry_eval
            result["effective_population"] = evaluation["effective_population"]
            lineage = complete_lineage(base, evaluation["schedule"]["schedule_sha256"])
            write_sequence_metadata(
                output, "h2_prediction", sequence, result, lineage, provenance,
            )
            index["sequences"][sequence] = sequence_entry(
                output, "h2_prediction", lineage,
                bootstrap_end_candidate_index=result["schedule"]["bootstrap_end_candidate_index"],
            )
            del bridge_eval
            torch.cuda.empty_cache()

        index["canonical_checkpoint"] = {
            "file": "predictor.pt", "file_sha256": sha256_file(checkpoint_path),
            "state_dict_sha256": predictor_state_sha256(predictor.state_dict()),
            "training_lineage_sha256": checkpoint["training_lineage"]["training_lineage_sha256"],
            "h1_bridge_sha256": bridge_meta["file_sha256"],
            "training": training_record,
        }
        write_registry_and_summary(staged_root, "h2_prediction", index)
        validate_module_manifest(staged_root, "h2_prediction", index)
        if sha256_file(repo_path(config["paths"]["h1_bridge"])) != bridge_meta["file_sha256"]:
            raise RuntimeError("canonical H1 bridge changed during H2 run; run run_h2 again")
        publish_current_canonical(staged_root, root)
        del predictor, bridge
        torch.cuda.empty_cache()
    return {
        "status": "complete",
        "module": "h2_prediction",
        "output": str(root.relative_to(REPO_ROOT)),
        "requested_sequences": list(requested),
        "fresh_sequences": list(requested),
        "checkpoint_training": "fresh",
        "run_policy": "fresh_current_canonical_replace",
        "performance_diagnostics": {
            "persistent": True,
            "sequence_result_field": "result.performance_diagnostics",
            "training_field": "canonical_checkpoint.training.performance_diagnostics",
            "aggregate_summary": "SUMMARY_H2.md",
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--sequences", nargs="+", choices=SUPPORTED_SEQUENCES,
        help="fresh predictor training and evaluation; this request replaces the canonical set",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    config, _ = load_config()
    requested = resolve_sequences(args.sequences, config["experiment"]["default_sequences"])
    print("Phase 1: H2 Prediction — Sparse-Anchor Prediction Feasibility")
    print("Input mode: native_anchor_rgb_plus_predicted_hidden_jepa_bridge")
    print("Deployment: delayed/bracketed; timestamp_causal=false; closing A5 must be online")
    print(json.dumps(run(requested), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
