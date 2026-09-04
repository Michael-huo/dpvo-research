"""Final Exp6 H2: strict delayed sparse-anchor prediction and efficiency."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from .canonical import (input_provenance, load_yaml, publish_tree, resolve_sequences, source_paths,
                        trajectory_payload, validate_trajectory_payload, write_report)
from .evaluation import (dense_hidden_ate, evaluate_paired_trajectory,
                         filter_groundtruth_associable_timestamps,
                         freeze_evaluation_population, plot_canonical_trajectories,
                         population_coverage)
from .h2_deployment import DelayedDeploymentProvider
from .jepa_fmap import BRIDGE_ARCHITECTURE, build_bridge, coordinate_masks, tokens_to_field
from .jepa_runtime import (CompactFeatureStore, extract_block5_store,
                           extract_true_fmap_store, load_dpvo_domain,
                           sequence_geometry)
from .oracle_packet import FMapZeroContextPacket
from .predictor import (AnchorInterval, RobustTransportBlock5Predictor,
                        build_anchor_intervals, effective_records,
                        predictor_metadata, predictor_state_sha256,
                        split_anchor_intervals)
from .profiling import (OnlineProfiler, break_even_payload,
                        transmission_payload)
from .protocol import (REPO_ROOT, SUPPORTED_SEQUENCES, atomic_write_bytes,
                       atomic_write_json, canonical_sha256, load_sequence_records,
                       post_bootstrap_ratio_roles, ratio_schedule_payload,
                       repo_path, sha256_file)
from .h2_training import (_field, _plot_feature_diagnostics,
                          build_robust_correspondence_store,
                          calibrate_train_only_thresholds, held_out_representation,
                          tiny_overfit, train_predictor)
from .runtime import (OnlineFrame, PacketObservation, materialize_schedule,
                      run_deployment_observations, run_formal_mode,
                      run_packet_observations,
                      sanitize_full_oracle_frames)
from .schema import condition_metadata

DEFAULT_CONFIG = REPO_ROOT / "research/configs/exp6_h2.yaml"
TRAINING_SEQUENCE = "MH_01_easy"
TRAINING_ANCHOR_RATIO = 0.2
CONDITIONS = (
    "full_rgb_reference", "sparse_rgb_reference", "anchor_jepa_only",
    "oracle_jepa_hidden_reference", "predicted_jepa_hidden",
)
_SUMMARY_CONDITIONS = (
    ("full_rgb_reference", "Full RGB"),
    ("sparse_rgb_reference", "Sparse RGB"),
    ("oracle_jepa_hidden_reference", "Oracle JEPA"),
    ("predicted_jepa_hidden", "Predicted JEPA"),
    ("anchor_jepa_only", "Anchor JEPA only"),
)


def _summary_scalar(value: Any) -> str:
    """Serialize a result scalar without introducing display-only rounding."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"SUMMARY value is not numeric: {value!r}")
    return json.dumps(value, allow_nan=False)


def _render_sequence_summary(results: Mapping[str, Any], sequence: str) -> str:
    """Render one H2 summary solely from the current run's canonical results."""
    if "delayed_bracketed" not in str(results["method_role"]):
        raise ValueError("SUMMARY requires delayed/bracketed H2 results")
    if results["protocol"]["timestamp_causal"] is not False:
        raise ValueError("SUMMARY requires non-causal H2 results")

    sequence_result = results["sequences"][sequence]
    conditions = sequence_result["conditions"]
    lines = [
        f"# Exp6 H2 — {sequence}",
        "",
        "| Condition | ATE RMSE (m) | translation RPE@1s (m) | rotation RPE@1s (deg) | coverage | final node count |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition_key, label in _SUMMARY_CONDITIONS:
        condition = conditions[condition_key]
        evaluation = condition["canonical_evaluation"]
        lines.append(
            f"| {label} | {_summary_scalar(evaluation['ate_rmse_m'])} | "
            f"{_summary_scalar(evaluation['translation_rpe_rmse_m'])} | "
            f"{_summary_scalar(evaluation['rotation_rpe_rmse_deg'])} | "
            f"{_summary_scalar(condition['canonical_coverage']['canonical_pose_coverage'])} | "
            f"{_summary_scalar(condition['runtime']['final_node_count_before_terminate'])} |"
        )

    predicted_runtime = conditions["predicted_jepa_hidden"]["runtime"]
    lines.extend((
        "",
        f"- Anchor ratio: {_summary_scalar(sequence_result['schedule']['actual_full_sequence_anchor_ratio'])}",
        "- Uploaded anchor count / total candidates: "
        f"{_summary_scalar(predicted_runtime['rgb_uploaded_frame_count'])} / "
        f"{_summary_scalar(sequence_result['candidate_count'])}",
        f"- Hidden count: {_summary_scalar(sequence_result['hidden_count'])}",
        "- Hidden RGB violation count: "
        f"{_summary_scalar(predicted_runtime['hidden_online_rgb_violation_count'])}",
        "- Deployment mode: delayed/bracketed, non-causal",
        "",
        "This is a human-readable summary. The root `results.json` remains the complete, "
        "authoritative data source.",
        "",
    ))
    return "\n".join(lines)


def _write_sequence_summary(path: Path, results: Mapping[str, Any], sequence: str) -> None:
    atomic_write_bytes(path, _render_sequence_summary(results, sequence).encode("utf-8"))


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


def _unique_identities(intervals: Sequence[AnchorInterval]) -> tuple[Any, ...]:
    rows = {identity.key: identity for interval in intervals
            for identity in (interval.anchor0, *[query.identity for query in interval.hidden],
                             interval.anchor1)}
    return tuple(sorted(rows.values(), key=lambda item: item.candidate_index))


def _load_bridge(config: Mapping[str, Any], transform: Any) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = repo_path(config["paths"]["h1_bridge"])
    if not path.is_file():
        raise FileNotFoundError(
            f"canonical H1 bridge is required; run run_h1 first: {path}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    forbidden = {"optimizer", "optimizer_state_dict", "grad_scaler", "best_state"}
    if forbidden & set(checkpoint) or checkpoint.get("architecture") != BRIDGE_ARCHITECTURE:
        raise RuntimeError("H1 bridge checkpoint violates the canonical contract")
    if checkpoint.get("layer_zero_based") != 5 or "state_dict" not in checkpoint:
        raise RuntimeError("H1 bridge checkpoint metadata is incomplete")
    model = build_bridge(transform, channels=int(config["bridge"]["hidden_channels"])).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.requires_grad_(False)
    return model, {
        "file": str(path.relative_to(REPO_ROOT)), "file_sha256": sha256_file(path),
        "architecture": checkpoint["architecture"], "layer_zero_based": 5,
        "coordinate_protocol": checkpoint.get("coordinate_protocol"),
        "loaded_h1_results_json": False, "bridge_retrained_by_h2": False,
    }


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
        "train_only_calibration": dict(calibration), "lineage": dict(lineage),
    }
    buffer = io.BytesIO(); torch.save(payload, buffer)
    atomic_write_bytes(path, buffer.getvalue())
    return payload


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
    config: Mapping[str, Any], temporary: Path,
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
    )
    temporary.mkdir(parents=True, exist_ok=False)
    runtime, arrays = run_deployment_observations(
        provider.observations(), calibration, config,
        image_height=height, image_width=width,
        condition_name="predicted_jepa_hidden", expected_roles=roles,
        profiler=profiler, on_tracked=provider.on_tracked,
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
    predicted_runtime, predicted_arrays, online_profile = _run_strict_replay(
        records, roles, intervals, calibration, transform, bridge, predictor,
        thresholds, config, temporary / "strict_online",
    )
    # Only after strict deployment is complete may raw/oracle reference stores exist.
    all_online = [OnlineFrame(row.identity, row.rgb_path) for row in records]
    packet_online = sanitize_full_oracle_frames(records, roles)
    all_anchor = {row.identity.key: "anchor" for row in records}
    full_runtime, full_arrays = run_formal_mode(
        "matched_full_rgb", all_online, calibration, config, roles=all_anchor,
        condition_name="full_rgb_reference",
    )
    sparse_runtime, sparse_arrays = run_formal_mode(
        "sparse_rgb", packet_online, calibration, config, roles=roles,
        condition_name="sparse_rgb_reference",
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
    full_compute_ms = float(full_runtime["elapsed_seconds"]) * 1000.0
    efficiency = {
        "transmission": transmission,
        "online_cloud": online_profile,
        "full_rgb_reference_cloud_compute_ms": full_compute_ms,
        "break_even_uplink_bandwidth": break_even_payload(
            encoded_full_bytes=transmission["encoded_full_bytes"],
            encoded_anchor_bytes=transmission["encoded_anchor_bytes"],
            h2_cloud_compute_ms=online_profile["cloud_compute_ms"],
            full_rgb_cloud_compute_ms=full_compute_ms,
        ),
    }
    all_store.close(); true_store.close(); shutil.rmtree(reference_temp)
    return {
        "evaluation_role": ("predictor_development_in_sequence_feasibility"
                            if sequence == TRAINING_SEQUENCE else "frozen_predictor_zero_shot"),
        "schedule": dict(schedule), "candidate_count": len(records),
        "anchor_count": len(anchor_paths), "hidden_count": len(records) - len(anchor_paths),
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
    destination = repo_path(config["paths"]["output_root"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".exp6_h2_", dir=destination.parent) as name:
        temporary = Path(name)
        staging = temporary / "h2_prediction"
        staging.mkdir()
        calibration = np.loadtxt(repo_path(config["paths"]["calibration"]), delimiter=" ")
        training_records = load_sequence_records(config, TRAINING_SEQUENCE)
        bootstrap = materialize_schedule(training_records, calibration, config)
        training_roles = post_bootstrap_ratio_roles(
            [row.identity for row in training_records],
            bootstrap_end_candidate_index=int(bootstrap["bootstrap_end_candidate_index"]),
            anchor_ratio=TRAINING_ANCHOR_RATIO,
        )
        training_intervals = build_anchor_intervals(training_records, training_roles)
        split, split_payload = split_anchor_intervals(
            training_intervals, float(config["split"]["train_fraction"]),
            float(config["split"]["validation_fraction"]),
        )
        counts = {
            key: {
                "intervals": len(values),
                "queries": sum(len(row.hidden) for row in values),
            }
            for key, values in split.items()
        }
        if counts != config["split"]["expected_counts"]:
            raise RuntimeError(f"H2 split changed: {counts}")

        transform, geometry = sequence_geometry(training_records[0], calibration, config)
        bridge, bridge_meta = _load_bridge(config, transform)
        development = (*split["train"], *split["validation"])
        training_temp = temporary / "predictor_training"
        dev_store, dev_extraction = extract_block5_store(
            training_records, _unique_identities(development), calibration,
            config, training_temp, transform,
        )
        mask = torch.from_numpy(coordinate_masks(transform)["valid_token_mask"]).cuda()
        thresholds = calibrate_train_only_thresholds(
            split["train"], dev_store, transform, mask,
        )
        robust_dev, robust_dev_meta = build_robust_correspondence_store(
            development, dev_store, transform, mask, thresholds,
        )
        tiny = tiny_overfit(
            dev_store, split["train"], transform, mask, config, robust_dev,
        )
        predictor, training_summary = train_predictor(
            dev_store, split, transform, mask, config, robust_dev,
        )

        test_temp = temporary / "predictor_test"
        test_store, test_extraction = extract_block5_store(
            training_records, _unique_identities(split["test"]), calibration,
            config, test_temp, transform,
        )
        robust_test, robust_test_meta = build_robust_correspondence_store(
            split["test"], test_store, transform, mask, thresholds,
        )
        held_out = held_out_representation(
            split["test"], test_store, transform, mask, robust_test, predictor, bridge,
        )
        training_schedule = ratio_schedule_payload(
            [row.identity for row in training_records],
            bootstrap_end_candidate_index=int(bootstrap["bootstrap_end_candidate_index"]),
            anchor_ratio=TRAINING_ANCHOR_RATIO,
        )
        lineage = {
            "config_sha256": sha256_file(config_path),
            "h1_bridge_file_sha256": bridge_meta["file_sha256"],
            "training_schedule_sha256": training_schedule["schedule_sha256"],
            "split_sha256": split_payload["split_sha256"],
            "train_only_calibration_sha256": thresholds["calibration_sha256"],
        }
        checkpoint = _save_predictor(
            staging / "predictor.pt", predictor, config, thresholds, lineage,
        )

        # No training or Oracle reference store survives into deployment.
        dev_store.close()
        test_store.close()
        shutil.rmtree(training_temp)
        shutil.rmtree(test_temp)

        sequence_results = {}
        for sequence in requested:
            records = load_sequence_records(config, sequence)
            sequence_bootstrap = materialize_schedule(records, calibration, config)
            roles = post_bootstrap_ratio_roles(
                [row.identity for row in records],
                bootstrap_end_candidate_index=int(
                    sequence_bootstrap["bootstrap_end_candidate_index"]
                ),
                anchor_ratio=float(config["experiment"]["anchor_ratio"]),
            )
            intervals = build_anchor_intervals(records, roles)
            effective, _ = effective_records(records, intervals)
            keys = {row.identity.key for row in effective}
            effective_intervals = tuple(
                row for row in intervals if row.anchor1.key in keys
            )
            effective_roles = {
                row.identity.key: roles[row.identity.key] for row in effective
            }
            schedule = ratio_schedule_payload(
                [row.identity for row in effective],
                bootstrap_end_candidate_index=int(
                    sequence_bootstrap["bootstrap_end_candidate_index"]
                ),
                anchor_ratio=float(config["experiment"]["anchor_ratio"]),
            )
            transform_eval, geometry_eval = sequence_geometry(
                effective[0], calibration, config,
            )
            bridge_eval, bridge_eval_meta = _load_bridge(config, transform_eval)
            if bridge_eval_meta["file_sha256"] != bridge_meta["file_sha256"]:
                raise RuntimeError("H1 bridge changed during H2 run")
            sequence_temp = temporary / f"sequence_{sequence}"
            sequence_temp.mkdir()
            result = _run_sequence(
                effective, effective_roles, effective_intervals, schedule, calibration,
                transform_eval, bridge_eval, predictor, thresholds, config,
                sequence_temp, staging / sequence,
            )
            result["coordinate_transform"] = geometry_eval
            sequence_results[sequence] = result
            del bridge_eval
            torch.cuda.empty_cache()
            shutil.rmtree(sequence_temp)

        results = {
            "schema_version": 1,
            "status": "complete",
            "experiment": "Exp6 H2 Prediction",
            "method_role": "strict_capability_isolated_delayed_bracketed_deployment",
            "requested_sequences": list(requested),
            "conditions": list(CONDITIONS),
            "protocol": {
                "online_input": "uploaded_anchor_rgb_only",
                "anchor_observation": "native_dpvo_fnet_patchifier",
                "hidden_observation": "predicted_jepa_to_frozen_bridge_fmap_only",
                "strict_deployment": True,
                "timestamp_causal": False,
                "closing_anchor_online_availability_required": True,
                "causal_or_real_time_claim": False,
            },
            "bridge": bridge_meta,
            "predictor": {
                "file": "predictor.pt",
                "file_sha256": sha256_file(staging / "predictor.pt"),
                "state_dict_sha256": checkpoint["state_dict_sha256"],
            },
            "training": {
                "sequence": TRAINING_SEQUENCE,
                "schedule": training_schedule,
                "split": split_payload,
                "split_counts": counts,
                "coordinate_transform": geometry,
                "training_anchor_ratio": TRAINING_ANCHOR_RATIO,
                "tiny_overfit": tiny,
                "summary": training_summary,
                "held_out_representation": held_out,
                "development_extraction": dev_extraction,
                "test_extraction": test_extraction,
                "development_correspondence": robust_dev_meta,
                "test_correspondence": robust_test_meta,
                "stores_closed_before_deployment": True,
            },
            "sequences": sequence_results,
            "provenance": {
                "lineage": lineage,
                "config_sha256": sha256_file(config_path),
                "inputs": input_provenance(
                    config, tuple(dict.fromkeys((TRAINING_SEQUENCE, *requested))),
                ),
                "sources": source_paths(
                    __file__,
                    Path(__file__).with_name("h2_training.py"),
                    Path(__file__).with_name("h2_deployment.py"),
                    Path(__file__).with_name("profiling.py"),
                    config_path,
                ),
                "h1_results_json_reads": 0,
            },
        }
        atomic_write_json(staging / "results.json", results)
        for sequence in requested:
            _write_sequence_summary(staging / sequence / "SUMMARY.md", results, sequence)
        write_report(staging / "REPORT.md", "Exp6 H2 — Prediction & Efficiency", [
            "Strict online capability: uploaded anchor RGB only; anchors use native DPVO frontend.",
            "JEPA anchor representations are predictor context only and never DPVO anchor observations.",
            "Closing anchor A5 must arrive and encode before buffered hidden reconstruction.",
            "This is delayed/bracketed and timestamp-non-causal; no causal or real-time claim is made.",
            f"Sequences: {', '.join(requested)}",
        ])
        expected = {"REPORT.md", "results.json", "predictor.pt", *requested}
        if {path.name for path in staging.iterdir()} != expected:
            raise RuntimeError("invalid H2 artifact tree")
        if any(not (staging / sequence / "SUMMARY.md").is_file()
               for sequence in requested):
            raise RuntimeError("missing H2 sequence SUMMARY.md")
        publish_tree(staging, destination)
        del predictor, bridge
        torch.cuda.empty_cache()
    return {
        "status": "complete",
        "output": str(destination.relative_to(REPO_ROOT)),
        "requested_sequences": list(requested),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--sequences", nargs="+", choices=SUPPORTED_SEQUENCES,
        help=("complete canonical output set; predictor is freshly trained on MH01 once, "
              "and other requested sequences are frozen zero-shot evaluations"),
    )
    return result


def main() -> int:
    args = parser().parse_args()
    config, _ = load_config()
    requested = resolve_sequences(args.sequences, config["experiment"]["default_sequences"])
    print("Experiment: h2_prediction")
    print("Input mode: native_anchor_rgb_plus_predicted_hidden_jepa_bridge")
    print("Deployment: delayed/bracketed; timestamp_causal=false; closing A5 must be online")
    print(json.dumps(run(requested), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
