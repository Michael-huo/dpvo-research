"""B1 sequence-loop training using the existing Bridge and Predictor recipes."""

from __future__ import annotations

import copy
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from latent_vslam.artifact_runtime import publish_checkpoint_tree, staged_directory
from latent_vslam.canonical import resolve_sequences
from latent_vslam.execution_runtime import (
    cpu_numa_layout, fixed_cpu_profile, initialize_formal_main_process,
)
from latent_vslam.inference_runtime import load_config as load_inference_config
from latent_vslam.parallel_runtime import run_sequential_trajectory_jobs
from latent_vslam.protocol import (
    REPO_ROOT, atomic_write_json, canonical_sha256, load_sequence_records,
    post_bootstrap_ratio_roles, ratio_schedule_payload, repo_path, sha256_file,
)
from latent_vslam.stride_training import release_training_before_trajectory, train_stride_predictor
from prediction.jepa_runtime import sequence_geometry
from prediction.predictor import build_anchor_intervals, effective_records, split_anchor_intervals


def checkpoint_root(kind: str, sequences: Sequence[str]) -> Path:
    if kind not in {"bridge", "predictor"}:
        raise ValueError(f"unknown B1 training kind: {kind}")
    return REPO_ROOT / "checkpoints" / "b1" / kind / "__".join(sequences)


def prepare_predictor_samples(
    records_by_sequence: Mapping[str, Sequence[Any]],
    bootstrap_by_sequence: Mapping[str, int],
    sequences: Sequence[str],
    *, anchor_stride: int = 5,
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    """Build each sequence in isolation, then concatenate corresponding splits."""
    if anchor_stride != 5:
        raise ValueError("B1 uses the existing stride-5 Predictor recipe")
    records: list[Any] = []
    split: dict[str, list[Any]] = {name: [] for name in ("train", "validation", "test")}
    per_sequence: dict[str, Any] = {}
    next_index = 0
    for sequence in sequences:
        source = tuple(records_by_sequence[sequence])
        if not source or any(row.identity.sequence != sequence for row in source):
            raise ValueError(f"invalid B1 records for {sequence}")
        bootstrap = int(bootstrap_by_sequence[sequence])
        identities = [row.identity for row in source]
        roles = post_bootstrap_ratio_roles(
            identities, bootstrap_end_candidate_index=bootstrap,
            anchor_ratio=1.0 / anchor_stride,
        )
        local_intervals = build_anchor_intervals(source, roles)
        effective, tail = effective_records(source, local_intervals)
        local_split, split_payload = split_anchor_intervals(local_intervals)
        schedule = ratio_schedule_payload(
            identities, bootstrap_end_candidate_index=bootstrap,
            anchor_ratio=1.0 / anchor_stride,
        )
        translated = {}
        for interval in local_intervals:
            translated[interval.interval_index] = replace(
                interval, interval_index=next_index,
                local_interval_index=interval.interval_index,
                global_interval_index=next_index,
            )
            next_index += 1
        for name in split:
            split[name].extend(translated[row.interval_index] for row in local_split[name])
        counts = {name: {
            "intervals": len(local_split[name]),
            "queries": sum(len(row.hidden) for row in local_split[name]),
        } for name in split}
        counts["all"] = {key: sum(counts[name][key] for name in split)
                         for key in ("intervals", "queries")}
        per_sequence[sequence] = {
            "candidate_count": len(source), "effective_candidate_count": len(effective),
            "complete_interval_count": len(local_intervals),
            "sample_counts": counts, "schedule": schedule,
            "split": split_payload, "effective_population": tail,
        }
        records.extend(effective)
    merged = {name: tuple(rows) for name, rows in split.items()}
    indices = [row.interval_index for rows in merged.values() for row in rows]
    if len(indices) != len(set(indices)):
        raise RuntimeError("B1 global interval indices are not unique")
    for rows in merged.values():
        for row in rows:
            if {row.anchor0.sequence, row.anchor1.sequence, *
                (query.identity.sequence for query in row.hidden)} != {row.anchor0.sequence}:
                raise RuntimeError("B1 interval crosses sequences")
    totals = {name: {
        "intervals": len(merged[name]),
        "queries": sum(len(row.hidden) for row in merged[name]),
    } for name in merged}
    totals["all"] = {key: sum(totals[name][key] for name in merged)
                     for key in ("intervals", "queries")}
    sample_counts = {
        "by_sequence": {sequence: per_sequence[sequence]["sample_counts"]
                        for sequence in sequences},
        "total": totals,
    }
    split_definition = {sequence: per_sequence[sequence]["split"]
                        for sequence in sequences}
    budget = {
        "anchor_stride": anchor_stride, "sequences": tuple(sequences),
        "split": merged, "sample_counts": sample_counts,
        "split_payload": {"by_sequence": split_definition,
                          "split_sha256": canonical_sha256(split_definition)},
        "source_schedule": {"by_sequence": {
            sequence: per_sequence[sequence]["schedule"] for sequence in sequences},
            "schedule_sha256": canonical_sha256({
                sequence: per_sequence[sequence]["schedule"]["schedule_sha256"]
                for sequence in sequences})},
    }
    protocol = {"fixed_split": {
        "method": "per_sequence_existing_interval_split",
        "by_sequence": split_definition,
    }}
    return tuple(records), budget, {"protocol": protocol, "per_sequence": per_sequence}


def run_predictor(sequences: Sequence[str], *, base_config_path: str | Path,
                  wrapper_path: str | Path) -> dict[str, Any]:
    requested = resolve_sequences(sequences, ())
    config, _ = load_inference_config(base_config_path)
    config = copy.deepcopy(config)
    config.pop("canonical_predictor", None)
    config["experiment"]["training_sequences"] = list(requested)
    config["paths"]["bridge"] = str(checkpoint_root("bridge", requested) / "bridge.pt")
    root = checkpoint_root("predictor", requested)
    calibration = np.loadtxt(repo_path(config["paths"]["calibration"]))
    records_by_sequence = {sequence: load_sequence_records(config, sequence)
                           for sequence in requested}
    geometries = {sequence: sequence_geometry(records_by_sequence[sequence][0], calibration, config)[1]
                  for sequence in requested}
    if len({row["transform_sha256"] for row in geometries.values()}) != 1:
        raise RuntimeError("B1 sequences have different coordinate geometry")
    if not repo_path(config["paths"]["bridge"]).is_file():
        raise FileNotFoundError(f"B1 Bridge is missing: {config['paths']['bridge']}")
    formal_runtime = initialize_formal_main_process(require_online=True)
    layout = cpu_numa_layout(formal_runtime["hardware"])
    stage_c = fixed_cpu_profile(layout)["stage_c"]
    with tempfile.TemporaryDirectory(prefix="b1_predictor_", dir="/tmp") as name:
        temporary = Path(name)
        schedule_rows, _ = run_sequential_trajectory_jobs(
            [{"kind": "materialize_schedule", "records": records_by_sequence[sequence],
              "calibration": calibration, "config": config, "sequence": sequence}
             for sequence in requested],
            temporary / "schedules", cpu_profile=stage_c,
            hardware=formal_runtime["hardware"],
        )
        bootstrap = {sequence: int(schedule_rows[index]["schedule"][
            "bootstrap_end_candidate_index"]) for index, sequence in enumerate(requested)}
        records, budget, preparation = prepare_predictor_samples(
            records_by_sequence, bootstrap, requested,
        )
        with staged_directory(root) as staged:
            trained = train_stride_predictor(
                records, budget, preparation["protocol"], config,
                staged / "model", temporary / "model",
            )
            cleanup = release_training_before_trajectory()
            atomic_write_json(staged / "training.json", {
                "sequences": list(requested), "sample_counts": budget["sample_counts"],
                "per_sequence": preparation["per_sequence"],
                "predictor_training": trained["record"],
                "checkpoint_sha256": sha256_file(trained["checkpoint_path"]),
                "config_file_sha256": sha256_file(wrapper_path),
                "cleanup": cleanup,
                "held_out_scope": "training_sequences_internal_only",
            })
            publish_checkpoint_tree(staged, root)
    return {"status": "complete", "training": "b1_predictor",
            "sequences": list(requested), "sample_counts": budget["sample_counts"],
            "checkpoint": str((root / "model" / "predictor.pt").relative_to(REPO_ROOT)),
            "training_record": str((root / "training.json").relative_to(REPO_ROOT))}
