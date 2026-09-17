"""Train the Bridge with the fixed MH_01_easy recipe."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from latent_vslam.canonical import resolve_sequences
from latent_vslam.bridge_checkpoint import (
    BRIDGE_CONFIG_PATH, BRIDGE_TRAINING_SEQUENCE, bridge_training_context, bridge_training_input,
    bridge_training_sources, load_compatible_bridge, load_bridge_config,
)
from latent_vslam.efficiency_profiling import (
    PerformanceRecorder, PersistentPerformanceAudit, add_cuda_worker_mapping,
    performance_diagnosis,
)
from prediction.jepa_runtime import sequence_geometry, state_dict_sha256
from latent_vslam.jepa_fmap import (
    contiguous_split, coordinate_protocol_metadata, hidden_split_keys,
)
from latent_vslam.protocol import (REPO_ROOT, atomic_write_json, canonical_sha256,
                       post_bootstrap_ratio_roles,
                       ratio_schedule_payload, repo_path, sha256_file)
from latent_vslam.bridge_training import evaluate_representation_control, train_bridge
from latent_vslam.artifact_runtime import publish_checkpoint_tree, staged_directory
from latent_vslam.manifests import CHECKPOINT_ROOTS
from latent_vslam.cuda_devices import CudaDevicePool
from latent_vslam.execution_runtime import (
    cpu_numa_layout, execution_provenance, initialize_formal_main_process,
    release_cuda_training_state, require_lifecycle_cleanup, runtime_provenance,
    fixed_cpu_profile,
)
from latent_vslam.parallel_runtime import extract_parallel, run_sequential_trajectory_jobs
from latent_vslam.training_runtime import ResidentBridgeView

DEFAULT_CONFIG = BRIDGE_CONFIG_PATH
TRAINING_SEQUENCE = BRIDGE_TRAINING_SEQUENCE


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    return load_bridge_config(path)


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
        raise RuntimeError(f"Bridge training split population changed: {counts}")
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


def run(sequences: Sequence[str], *, config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, config_path = load_config(config_path)
    requested = resolve_sequences(sequences, config["experiment"]["default_sequences"])
    formal_runtime = initialize_formal_main_process()
    root = repo_path(config["paths"]["output_root"])
    root.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root = CHECKPOINT_ROOTS["bridge"]
    calibration = np.loadtxt(repo_path(config["paths"]["calibration"]), delimiter=" ")
    training_records, training_base, training_provenance = bridge_training_context(config)

    with tempfile.TemporaryDirectory(prefix=".bridge_training_", dir=root.parent) as name, staged_directory(checkpoint_root) as staged_checkpoints:
        temporary = Path(name)
        checkpoint_path = staged_checkpoints / "bridge.pt"
        layout = cpu_numa_layout(formal_runtime["hardware"])
        stage_c = fixed_cpu_profile(layout)["stage_c"]
        schedule_rows, training_schedule_execution = run_sequential_trajectory_jobs(
            [{"kind": "materialize_schedule", "records": training_records,
              "calibration": calibration, "config": config,
              "sequence": TRAINING_SEQUENCE}],
            temporary / "bridge_training_schedule", cpu_profile=stage_c,
            hardware=formal_runtime["hardware"],
        )
        training_bootstrap = schedule_rows[0]["schedule"]

        with PersistentPerformanceAudit(
            "bridge_training", "canonical_bridge_training",
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

                    bridge_hidden_keys=set().union(*map(set, training["split_keys"].values())),
                )
            with training_performance.phase("resident_initialization"):
                resident_training = ResidentBridgeView(
                    training_store, training["split_keys"], device=torch.device(CudaDevicePool.discover().primary_device),
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
                    expected_training_input=bridge_training_input(training_base),
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
            formal_runtime["runtime_settings"], component="bridge_post_training",
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
        training_store.close()
        atomic_write_json(staged_checkpoints / "training.json", training_record)
        publish_checkpoint_tree(staged_checkpoints, checkpoint_root)
        return {
            "status": "complete", "module": "bridge", "training": "bridge",
            "checkpoint": str((checkpoint_root / "bridge.pt").relative_to(REPO_ROOT)),
            "checkpoint_sha256": bridge_sha256,
            "training_lineage_sha256": training["lineage"]["training_lineage_sha256"],
        }
