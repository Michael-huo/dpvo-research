"""H1 Interface: Oracle-JEPA representation-interface feasibility."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
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
    build_bridge, contiguous_split,
    coordinate_protocol_metadata, hidden_split_keys,
)
from .protocol import (REPO_ROOT, SUPPORTED_SEQUENCES, canonical_sha256,
                       load_sequence_records, post_bootstrap_ratio_roles,
                       ratio_schedule_payload, repo_path, sha256_file)
from .h1_training import (FeatureStore, HiddenFMapProvider, _previous_anchor_mapping,
                          evaluate_representation_control, extract_feature_store,
                          train_bridge)
from .runtime import (OnlineFrame, materialize_schedule, run_formal_mode,
                      sanitize_full_oracle_frames)
from .schema import condition_metadata
from .schema import VISUAL_STATE_CONTRACT_SHA256
from .registry import (
    base_lineage, complete_lineage, empty_index, publish_current_canonical,
    sequence_entry, validate_module_manifest, write_registry_and_summary,
    write_sequence_metadata,
)

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
        "timestamp_contract_exact", "elapsed_seconds", "peak_gpu_vram_bytes",
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


def _run_sequence(
    records: Sequence[Any], calibration: np.ndarray, schedule: Mapping[str, Any],
    roles: Mapping[str, str], store: FeatureStore, model: torch.nn.Module,
    config: Mapping[str, Any], output: Path,
) -> dict[str, Any]:
    sequence = records[0].identity.sequence
    previous_anchor = _previous_anchor_mapping(records, roles)
    all_online = [OnlineFrame(row.identity, row.rgb_path) for row in records]
    packet_online = sanitize_full_oracle_frames(records, roles)
    all_anchor_roles = {row.identity.key: "anchor" for row in records}
    runtimes: dict[str, Any] = {}
    arrays: dict[str, Any] = {}
    runtimes["full_rgb"], arrays["full_rgb"] = run_formal_mode(
        "matched_full_rgb", all_online, calibration, config,
        roles=all_anchor_roles, condition_name="full_rgb",
    )
    runtimes["sparse_rgb"], arrays["sparse_rgb"] = run_formal_mode(
        "sparse_rgb", packet_online, calibration, config,
        roles=roles, condition_name="sparse_rgb",
    )
    for name in ("true_fmap", "oracle_jepa_bridge"):
        provider = HiddenFMapProvider(
            name, store, previous_anchor, model if name == "oracle_jepa_bridge" else None,
        )
        runtimes[name], arrays[name] = run_formal_mode(
            "fmap_zero_context", packet_online, calibration, config, roles=roles,
            hidden_provider=provider, condition_name=name,
        )
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
    diagnostics = _feature_diagnostics(
        output / "feature_diagnostics.png", store, model,
        [row.identity.key for row in hidden], int(config["experiment"]["seed"]),
    )
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
    config, config_path = load_config()
    requested = resolve_sequences(sequences, config["experiment"]["default_sequences"])
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

        with PersistentPerformanceAudit(
            "h1_interface", "canonical_bridge_training",
            components=("bridge_training_main_process",),
        ) as training_performance:
            with training_performance.phase("schedule_and_split"):
                bootstrap = materialize_schedule(training_records, calibration, config)
                training = _training_details(
                    training_records, int(bootstrap["bootstrap_end_candidate_index"]),
                    calibration, config, training_base,
                )
                training_temp = temporary / "training"
                training_temp.mkdir()
            with training_performance.phase("offline_feature_extraction"):
                training_store, training_extraction = extract_feature_store(
                    training_records,
                    set().union(*map(set, training["split_keys"].values())),
                    calibration, config, training_temp, training["transform"],
                )
            with training_performance.phase("bridge_training"):
                training_batch_performance = PerformanceRecorder(enable_cuda=True)
                validation_batch_performance = PerformanceRecorder(enable_cuda=True)
                model, training_summary = train_bridge(
                    training_store, training["split_keys"], training["transform"],
                    config, checkpoint_path, training["lineage"],
                    profiler=training_batch_performance,
                    validation_profiler=validation_batch_performance,
                )
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
            training_performance_payload, training_extraction["jepa"],
        )
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
            "provenance": training_provenance | {
                "config_file": str(config_path.relative_to(REPO_ROOT)),
                "config_file_sha256": sha256_file(config_path),
            },
        }
        bridge_sha256 = sha256_file(checkpoint_path)
        records_by_sequence: dict[str, Sequence[Any]] = {TRAINING_SEQUENCE: training_records}
        for sequence in requested:
            if sequence not in records_by_sequence:
                records_by_sequence[sequence] = load_sequence_records(config, sequence)
            records = records_by_sequence[sequence]
            base, provenance = base_lineage(
                config, sequence, evaluation_source_files,
                h0_contract_sha256=VISUAL_STATE_CONTRACT_SHA256,
                h1_bridge_sha256=bridge_sha256,
            )
            provenance = provenance | {
                "config_file": str(config_path.relative_to(REPO_ROOT)),
                "config_file_sha256": sha256_file(config_path),
            }
            with PersistentPerformanceAudit(
                "h1_interface", f"sequence:{sequence}",
                components=("dpvo_bridge_main_process",),
            ) as performance:
                with performance.phase("schedule_and_feature_preparation"):
                    if sequence == TRAINING_SEQUENCE:
                        roles, schedule, transform = (
                            training["roles"], training["schedule"],
                            training["transform"],
                        )
                        current_store, extraction = training_store, training_extraction
                    else:
                        bootstrap = materialize_schedule(records, calibration, config)
                        roles = post_bootstrap_ratio_roles(
                            [row.identity for row in records],
                            bootstrap_end_candidate_index=int(
                                bootstrap["bootstrap_end_candidate_index"]
                            ),
                            anchor_ratio=float(config["experiment"]["anchor_ratio"]),
                        )
                        schedule = ratio_schedule_payload(
                            [row.identity for row in records],
                            bootstrap_end_candidate_index=int(
                                bootstrap["bootstrap_end_candidate_index"]
                            ),
                            anchor_ratio=float(config["experiment"]["anchor_ratio"]),
                        )
                        transform, _ = sequence_geometry(
                            records[0], calibration, config,
                        )
                        evaluation_temp = temporary / f"evaluation_{sequence}"
                        evaluation_temp.mkdir()
                        hidden = {
                            row.identity.key for row in records
                            if roles[row.identity.key] == "hidden"
                        }
                        current_store, extraction = extract_feature_store(
                            records, hidden, calibration, config, evaluation_temp,
                            transform,
                        )
                with performance.phase("bridge_model_setup"):
                    evaluation_model = build_bridge(
                        transform, channels=160,
                    ).cuda().eval()
                    evaluation_model.load_state_dict(model.state_dict(), strict=True)
                    output = staged_root / "sequences" / sequence
                with performance.phase("sequence_evaluation_and_artifact_generation"):
                    result = _run_sequence(
                        records, calibration, schedule, roles, current_store,
                        evaluation_model, config, output,
                    )
            performance_payload = performance.payload()
            if sequence != TRAINING_SEQUENCE:
                add_cuda_worker_mapping(performance_payload, extraction["jepa"])
            performance_payload["condition_runtime"] = condition_runtime_diagnostics(result)
            performance_payload["diagnosis"] = performance_diagnosis(performance_payload)
            result["performance_diagnostics"] = performance_payload
            result["oracle_extraction"] = extraction
            lineage = complete_lineage(base, schedule["schedule_sha256"])
            write_sequence_metadata(
                output, "h1_interface", sequence, result, lineage, provenance,
            )
            index["sequences"][sequence] = sequence_entry(
                output, "h1_interface", lineage,
                bootstrap_end_candidate_index=result["schedule"]["bootstrap_end_candidate_index"],
            )
            del evaluation_model
            if current_store is not training_store:
                current_store.close()

        training_store.close()
        index["canonical_checkpoint"] = {
            "file": "bridge.pt", "file_sha256": sha256_file(checkpoint_path),
            "state_dict_sha256": state_dict_sha256(model.state_dict()),
            "training_lineage_sha256": training["lineage"]["training_lineage_sha256"],
            "training": training_record,
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
