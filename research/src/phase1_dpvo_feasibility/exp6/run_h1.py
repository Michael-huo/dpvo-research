"""Final Exp6 H1: fresh Oracle-JEPA to DPVO interface validation."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .canonical import (input_provenance, load_yaml, publish_tree, resolve_sequences, source_paths,
                        trajectory_payload, validate_trajectory_payload, write_report)
from .evaluation import (dense_hidden_ate, evaluate_paired_trajectory,
                         filter_groundtruth_associable_timestamps,
                         freeze_evaluation_population, plot_canonical_trajectories,
                         population_coverage)
from .jepa_runtime import sequence_geometry
from .jepa_fmap import build_bridge, contiguous_split, hidden_split_keys
from .protocol import (REPO_ROOT, SUPPORTED_SEQUENCES, atomic_write_json,
                       load_sequence_records, post_bootstrap_ratio_roles,
                       ratio_schedule_payload, repo_path, sha256_file)
from .h1_training import (FeatureStore, HiddenFMapProvider, _previous_anchor_mapping,
                          evaluate_representation_control, extract_feature_store,
                          train_bridge)
from .runtime import (OnlineFrame, materialize_schedule, run_formal_mode,
                      sanitize_full_oracle_frames)
from .schema import condition_metadata

DEFAULT_CONFIG = REPO_ROOT / "research/configs/exp6_h1.yaml"
TRAINING_SEQUENCE = "MH_01_easy"
CONDITIONS = ("full_rgb", "sparse_rgb", "true_fmap", "oracle_jepa_bridge")


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    config, resolved = load_yaml(path)
    fixed = {
        "seed": 1234, "bridge_initialization_seed": 1236,
        "training_sequence": TRAINING_SEQUENCE, "bootstrap_accepted_nodes": 8,
        "post_bootstrap_anchor_interval": 5,
    }
    for key, value in fixed.items():
        if config["experiment"].get(key) != value:
            raise ValueError(f"frozen H1 field changed: {key}")
    ratio = float(config["experiment"]["anchor_ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError("anchor_ratio must be in (0, 1]")
    bridge = config["bridge"]
    required = {
        "hidden_channels": 160, "passes": 2, "epochs_per_pass": 30,
        "batch_size": 4, "learning_rate": 1e-4, "weight_decay": 1e-4,
        "cosine_weight": 1.0, "smooth_l1_weight": 0.1,
        "reset_optimizer_and_grad_scaler_between_passes": True,
        "repeat_epoch_local_batch_order_each_pass": True,
    }
    for key, value in required.items():
        if bridge.get(key) != value:
            raise ValueError(f"frozen H1 bridge recipe changed: {key}")
    return config, resolved


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
        "conditions": conditions, "feature_diagnostics": diagnostics,
    }


def run(sequences: Sequence[str]) -> dict[str, Any]:
    config, config_path = load_config()
    requested = resolve_sequences(sequences, config["experiment"]["default_sequences"])
    destination = repo_path(config["paths"]["output_root"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".exp6_h1_", dir=destination.parent) as name:
        temporary = Path(name)
        staging = temporary / "h1_interface"
        staging.mkdir()
        calibration = np.loadtxt(repo_path(config["paths"]["calibration"]), delimiter=" ")
        training_records = load_sequence_records(config, TRAINING_SEQUENCE)
        bootstrap = materialize_schedule(training_records, calibration, config)
        training_roles = post_bootstrap_ratio_roles(
            [row.identity for row in training_records],
            bootstrap_end_candidate_index=int(bootstrap["bootstrap_end_candidate_index"]),
            anchor_ratio=float(config["experiment"]["anchor_ratio"]),
        )
        training_schedule = ratio_schedule_payload(
            [row.identity for row in training_records],
            bootstrap_end_candidate_index=int(bootstrap["bootstrap_end_candidate_index"]),
            anchor_ratio=float(config["experiment"]["anchor_ratio"]),
        )
        split = contiguous_split([row.identity for row in training_records])
        split_keys = hidden_split_keys(
            [row.identity for row in training_records], training_roles, split,
        )
        counts = {key: len(value) for key, value in split_keys.items()}
        if counts != dict(config["split"]["expected_hidden_counts"]):
            raise RuntimeError(f"H1 split population changed: {counts}")
        transform, geometry = sequence_geometry(training_records[0], calibration, config)
        training_temp = temporary / "mh01_training"
        training_temp.mkdir()
        store, extraction = extract_feature_store(
            training_records, set().union(*map(set, split_keys.values())), calibration,
            config, training_temp, transform,
        )
        model, training_summary = train_bridge(
            store, split_keys, transform, config, staging / "bridge.pt",
        )
        representation = evaluate_representation_control(
            model, store, split_keys["test"], transform, config,
        )
        sequence_results = {}
        for sequence in requested:
            if sequence == TRAINING_SEQUENCE:
                records, roles, schedule, current_store, current_transform = (
                    training_records, training_roles, training_schedule, store, transform,
                )
                current_extraction = extraction
            else:
                records = load_sequence_records(config, sequence)
                sequence_bootstrap = materialize_schedule(records, calibration, config)
                roles = post_bootstrap_ratio_roles(
                    [row.identity for row in records],
                    bootstrap_end_candidate_index=int(sequence_bootstrap["bootstrap_end_candidate_index"]),
                    anchor_ratio=float(config["experiment"]["anchor_ratio"]),
                )
                schedule = ratio_schedule_payload(
                    [row.identity for row in records],
                    bootstrap_end_candidate_index=int(sequence_bootstrap["bootstrap_end_candidate_index"]),
                    anchor_ratio=float(config["experiment"]["anchor_ratio"]),
                )
                current_transform, _ = sequence_geometry(records[0], calibration, config)
                evaluation_temp = temporary / f"evaluation_{sequence}"
                evaluation_temp.mkdir()
                hidden = {row.identity.key for row in records if roles[row.identity.key] == "hidden"}
                current_store, current_extraction = extract_feature_store(
                    records, hidden, calibration, config, evaluation_temp, current_transform,
                )
            evaluation_model = build_bridge(current_transform, channels=160).cuda().eval()
            evaluation_model.load_state_dict(model.state_dict(), strict=True)
            sequence_results[sequence] = _run_sequence(
                records, calibration, schedule, roles, current_store, evaluation_model,
                config, staging / sequence,
            )
            sequence_results[sequence]["oracle_extraction"] = current_extraction
            del evaluation_model
            if sequence != TRAINING_SEQUENCE:
                current_store.close(); shutil.rmtree(temporary / f"evaluation_{sequence}")
        store.close(); shutil.rmtree(training_temp)
        checkpoint = torch.load(staging / "bridge.pt", map_location="cpu", weights_only=False)
        if "state_dict" not in checkpoint or {"optimizer", "grad_scaler", "best_state"} & set(checkpoint):
            raise RuntimeError("H1 bridge checkpoint is not canonical")
        results = {
            "schema_version": 1, "status": "complete", "experiment": "Exp6 H1 Interface",
            "requested_sequences": list(requested), "conditions": list(CONDITIONS),
            "training": {
                "sequence": TRAINING_SEQUENCE, "schedule": training_schedule,
                "split": split, "hidden_split_counts": counts,
                "summary": training_summary, "held_out_representation": representation,
                "coordinate_transform": geometry, "trajectory_generalization_claim": False,
            },
            "bridge": {"file": "bridge.pt", "sha256": sha256_file(staging / "bridge.pt")},
            "sequences": sequence_results,
            "provenance": {
                "inputs": input_provenance(
                    config, tuple(dict.fromkeys((TRAINING_SEQUENCE, *requested))),
                ),
                "config_sha256": sha256_file(config_path),
                "sources": source_paths(__file__, Path(__file__).with_name("h1_training.py"),
                                        Path(__file__).with_name("jepa_fmap.py"),
                                        Path(__file__).with_name("jepa_runtime.py"),
                                        Path(__file__).with_name("runtime.py"), config_path),
            },
        }
        atomic_write_json(staging / "results.json", results)
        write_report(staging / "REPORT.md", "Exp6 H1 — JEPA → DPVO Interface", [
            "This is an Oracle-JEPA interface upper bound, not a deployment experiment.",
            "MH01 trains the bridge; MH03/MH05 are frozen zero-shot evaluations.",
            f"Sequences: {', '.join(requested)}",
        ])
        expected = {"REPORT.md", "results.json", "bridge.pt", *requested}
        if {path.name for path in staging.iterdir()} != expected:
            raise RuntimeError("invalid H1 artifact tree")
        publish_tree(staging, destination)
    return {"status": "complete", "output": str(destination.relative_to(REPO_ROOT)),
            "requested_sequences": list(requested)}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--sequences", nargs="+", choices=SUPPORTED_SEQUENCES,
        help=("complete canonical output set; bridge is freshly trained on MH01 once, "
              "and other requested sequences are frozen zero-shot evaluations"),
    )
    return result


def main() -> int:
    args = parser().parse_args()
    config, _ = load_config()
    requested = resolve_sequences(args.sequences, config["experiment"]["default_sequences"])
    print("Experiment: h1_interface")
    print("Input modes: full_rgb, sparse_rgb, true_fmap, oracle_jepa_bridge")
    print(json.dumps(run(requested), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
