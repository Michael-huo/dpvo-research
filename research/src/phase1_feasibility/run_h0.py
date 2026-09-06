"""H0 State: latent-state feasibility for the frozen DPVO state contract."""

from __future__ import annotations

import argparse
import json
import tempfile
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
from .oracle_packet import (FMapZeroContextPacket, FrontendPacketWriter,
                            ZERO_PACKET_SCHEMA, _derive_frontend_state,
                            compare_arrays, extract_frontend_packet)
from .protocol import (REPO_ROOT, SUPPORTED_SEQUENCES, load_sequence_records,
                       post_bootstrap_ratio_roles, ratio_schedule_payload,
                       repo_path, sha256_file)
from .runtime import OnlineFrame, run_formal_mode, sanitize_full_oracle_frames
from .schema import (VISUAL_STATE_CONTRACT, VISUAL_STATE_CONTRACT_SHA256,
                     condition_metadata)
from .registry import (
    base_lineage, complete_lineage, empty_index, publish_current_canonical,
    sequence_entry, validate_module_manifest, write_registry_and_summary,
    write_sequence_metadata,
)

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
        "timestamp_contract_exact", "elapsed_seconds", "peak_gpu_vram_bytes",
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


def _run_sequence(config: Mapping[str, Any], sequence: str, output: Path) -> dict[str, Any]:
    records = load_sequence_records(config, sequence)
    calibration = np.loadtxt(repo_path(config["dataset"]["calibration"]), delimiter=" ")
    all_online = [OnlineFrame(row.identity, row.rgb_path) for row in records]
    all_anchor_roles = {row.identity.key: "anchor" for row in records}
    full_runtime, full_arrays = run_formal_mode(
        "matched_full_rgb", all_online, calibration, config,
        roles=all_anchor_roles, condition_name="full_rgb",
    )
    bootstrap_end = int(full_runtime["bootstrap_end_candidate_index"])
    identities = [row.identity for row in records]
    roles = post_bootstrap_ratio_roles(
        identities, bootstrap_end_candidate_index=bootstrap_end,
        anchor_ratio=float(config["experiment"]["anchor_ratio"]),
    )
    schedule = ratio_schedule_payload(
        identities, bootstrap_end_candidate_index=bootstrap_end,
        anchor_ratio=float(config["experiment"]["anchor_ratio"]),
    )
    packet_online = sanitize_full_oracle_frames(records, roles)
    sparse_runtime, sparse_arrays = run_formal_mode(
        "sparse_rgb", packet_online, calibration, config, roles=roles,
        condition_name="sparse_rgb",
    )
    hidden = [row for row in records if roles[row.identity.key] == "hidden"]
    first_image, _ = _load_frame({"image_path": records[0].rgb_path}, calibration)
    microcheck: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="exp6_decomposition_fmap_", dir=output.parent) as name:
        extractor = _extractor(config, first_image)
        writer = FrontendPacketWriter(
            Path(name) / "fmap", [row.identity for row in hidden],
            provenance={
                "sequence": sequence, "scope": "offline_oracle_extractor_only",
                "visual_state_contract_sha256": VISUAL_STATE_CONTRACT_SHA256,
            }, schema=ZERO_PACKET_SCHEMA,
        )
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
        torch.cuda.empty_cache()
        store = writer.finalize()
        true_runtime, true_arrays = run_formal_mode(
            "fmap_zero_context", packet_online, calibration, config, roles=roles,
            store=store, condition_name="true_fmap",
        )
        store_descriptor = store.sanitized_descriptor()
        store.close()
    if microcheck is None:
        raise RuntimeError("decomposition schedule contains no hidden frames")
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
    runtimes = {"full_rgb": full_runtime, "sparse_rgb": sparse_runtime, "true_fmap": true_runtime}
    arrays = {"full_rgb": full_arrays, "sparse_rgb": sparse_arrays, "true_fmap": true_arrays}
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
        "contract_microcheck": microcheck,
        "oracle_store": {
            "schema": store_descriptor["schema"],
            "contains_hidden_rgb_or_path": store_descriptor["contains_hidden_rgb_or_path"],
            "identity_list_sha256": store_descriptor["identity_list_sha256"],
        },
    }


def run(sequences: Sequence[str]) -> dict[str, Any]:
    config, config_path = load_config()
    requested = resolve_sequences(sequences, config["experiment"]["default_sequences"])
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
    )
    with tempfile.TemporaryDirectory(prefix=".phase1_h0_", dir=root.parent) as name:
        staged_root = Path(name) / "h0_state"
        (staged_root / "sequences").mkdir(parents=True)
        index = empty_index("h0_state", requested)
        for sequence in requested:
            base, provenance = base_lineage(
                config, sequence, source_files,
                h0_contract_sha256=VISUAL_STATE_CONTRACT_SHA256,
            )
            provenance = provenance | {
                "config_file": str(config_path.relative_to(REPO_ROOT)),
                "config_file_sha256": sha256_file(config_path),
            }
            output = staged_root / "sequences" / sequence
            result = _run_sequence(config, sequence, output)
            lineage = complete_lineage(base, result["schedule"]["schedule_sha256"])
            write_sequence_metadata(
                output, "h0_state", sequence, result, lineage, provenance,
            )
            index["sequences"][sequence] = sequence_entry(
                output, "h0_state", lineage,
                bootstrap_end_candidate_index=result["schedule"]["bootstrap_end_candidate_index"],
            )
        write_registry_and_summary(staged_root, "h0_state", index)
        validate_module_manifest(staged_root, "h0_state", index)
        publish_current_canonical(staged_root, root)
    return {
        "status": "complete", "module": "h0_state",
        "output": str(root.relative_to(REPO_ROOT)),
        "requested_sequences": list(requested),
        "fresh_sequences": list(requested),
        "run_policy": "fresh_current_canonical_replace",
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
