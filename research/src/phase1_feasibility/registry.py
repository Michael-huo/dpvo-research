"""Sequence-local artifact registry and inexpensive lineage fingerprints."""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import (
    REPO_ROOT, atomic_write_bytes, atomic_write_json, canonical_sha256,
    load_sequence_records, repo_path, sha256_file,
)

INDEX_SCHEMA_VERSION = 2
MODULES = ("h0_state", "h1_interface", "h2_prediction")
SUMMARY_NAMES = {
    "h0_state": "SUMMARY_H0.md",
    "h1_interface": "SUMMARY_H1.md",
    "h2_prediction": "SUMMARY_H2.md",
}
SEQUENCE_FILES = {
    "h0_state": ("results.json", "SUMMARY_H0.md", "trajectory.png", "trajectories.npz"),
    "h1_interface": (
        "results.json", "SUMMARY_H1.md", "trajectory.png", "trajectories.npz",
        "feature_diagnostics.png",
    ),
    "h2_prediction": (
        "results.json", "SUMMARY_H2.md", "trajectory.png", "trajectories.npz",
        "feature_diagnostics.png",
    ),
}
LINEAGE_FIELDS = (
    "dataset_sha256", "config_protocol_sha256", "source_sha256",
    "schedule_sha256", "h0_state_contract_sha256", "h1_bridge_sha256",
    "h2_predictor_sha256",
)


def _calibration_path(config: Mapping[str, Any]) -> Path:
    value = config.get("paths", {}).get("calibration")
    if value is None:
        value = config.get("dataset", {}).get("calibration")
    if value is None:
        raise KeyError("Phase 1 config has no calibration path")
    return repo_path(value)


def dataset_fingerprint(config: Mapping[str, Any], sequence: str) -> dict[str, Any]:
    """Hash manifests and evaluation inputs without reading every RGB payload."""
    records = load_sequence_records(config, sequence)
    first_rgb = Path(records[0].rgb_path)
    camera = first_rgb.parents[1]
    groundtruth = repo_path(config["dataset"]["groundtruth_pattern"].format(sequence=sequence))
    manifest = [
        {
            "identity_key": row.identity.key,
            "candidate_index": row.identity.candidate_index,
            "filename": Path(row.rgb_path).name,
        }
        for row in records
    ]
    payload = {
        "method": "normalized_manifest_no_rgb_payload_v1",
        "sequence": sequence,
        "selected_frames": manifest,
        "selected_frame_manifest_sha256": canonical_sha256(manifest),
        "camera_data_csv_sha256": sha256_file(camera / "data.csv"),
        "calibration_sha256": sha256_file(_calibration_path(config)),
        "groundtruth_sha256": sha256_file(groundtruth),
        "rgb_payload_hashed": False,
    }
    payload["dataset_sha256"] = canonical_sha256(payload)
    return payload


def _scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(config))
    experiment = payload.get("experiment", {})
    for key in ("name", "default_sequences", "supported_sequences"):
        experiment.pop(key, None)
    paths = payload.get("paths", {})
    paths.pop("output_root", None)
    paths.pop("h1_bridge", None)
    runtime = payload.get("runtime", {})
    runtime.pop("jepa_python", None)
    return payload


def config_protocol_fingerprint(config: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "scientific_config": _scientific_config(config),
        "calibration_sha256": sha256_file(_calibration_path(config)),
        "dpvo_checkpoint_sha256": sha256_file(repo_path(config["paths"]["dpvo_checkpoint"])),
        "dpvo_config_sha256": sha256_file(repo_path(config["paths"]["dpvo_config"])),
    }
    if "jepa" in config:
        jepa = config["jepa"]
        payload["vjepa"] = {
            "expected_git_commit": jepa["expected_git_commit"],
            "checkpoint_sha256": jepa["checkpoint_sha256"],
        }
    payload["config_protocol_sha256"] = canonical_sha256(payload)
    return payload


def source_fingerprint(paths: Sequence[str | Path]) -> dict[str, Any]:
    rows: dict[str, str] = {}
    for value in paths:
        path = Path(value).resolve()
        rows[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
    return {"files": rows, "source_sha256": canonical_sha256(rows)}


def base_lineage(
    config: Mapping[str, Any], sequence: str, source_paths: Sequence[str | Path], *,
    h0_contract_sha256: str, h1_bridge_sha256: str | None = None,
    h2_predictor_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = dataset_fingerprint(config, sequence)
    protocol = config_protocol_fingerprint(config)
    sources = source_fingerprint(source_paths)
    lineage = {
        "dataset_sha256": dataset["dataset_sha256"],
        "config_protocol_sha256": protocol["config_protocol_sha256"],
        "source_sha256": sources["source_sha256"],
        "schedule_sha256": None,
        "h0_state_contract_sha256": h0_contract_sha256,
        "h1_bridge_sha256": h1_bridge_sha256,
        "h2_predictor_sha256": h2_predictor_sha256,
    }
    return lineage, {"dataset": dataset, "config_protocol": protocol, "sources": sources}


def complete_lineage(base: Mapping[str, Any], schedule_sha256: str) -> dict[str, Any]:
    lineage = dict(base)
    lineage["schedule_sha256"] = str(schedule_sha256)
    if tuple(lineage) != LINEAGE_FIELDS:
        raise ValueError("sequence lineage schema changed")
    return lineage


def empty_index(module: str, requested_sequences: Sequence[str]) -> dict[str, Any]:
    if module not in MODULES:
        raise ValueError(module)
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "module": module,
        "run_policy": "fresh_current_canonical_replace",
        "requested_sequences": list(requested_sequences),
        "canonical_checkpoint": None,
        "sequences": {},
    }


def artifact_hashes(directory: Path, module: str) -> dict[str, str]:
    required = SEQUENCE_FILES[module]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"missing sequence artifacts in {directory}: {missing}")
    present = {path.name for path in directory.iterdir()}
    unexpected = sorted(present - set(required))
    if unexpected:
        raise RuntimeError(f"unexpected sequence artifacts in {directory}: {unexpected}")
    return {name: sha256_file(directory / name) for name in required}


def sequence_entry(
    directory: Path, module: str, lineage: Mapping[str, Any], *,
    bootstrap_end_candidate_index: int,
) -> dict[str, Any]:
    return {
        "status": "valid",
        "bootstrap_end_candidate_index": int(bootstrap_end_candidate_index),
        "lineage": dict(lineage),
        "artifacts": artifact_hashes(directory, module),
    }


def publish_current_canonical(staged: Path, destination: Path) -> None:
    """Replace one complete module only after its fresh run validates."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        raise RuntimeError(f"stale module publish backup exists: {backup}")
    if destination.exists():
        destination.rename(backup)
    try:
        staged.rename(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


CONDITION_LABELS = {
    "full_rgb": "Full RGB", "sparse_rgb": "Sparse RGB", "true_fmap": "True FMap",
    "oracle_jepa_bridge": "Oracle JEPA→Bridge",
    "full_rgb_reference": "Full RGB", "sparse_rgb_reference": "Sparse RGB",
    "anchor_jepa_only": "Anchor JEPA only",
    "oracle_jepa_hidden_reference": "Oracle JEPA",
    "predicted_jepa_hidden": "Predicted JEPA",
}
CONDITION_ORDER = {
    "h0_state": ("full_rgb", "sparse_rgb", "true_fmap"),
    "h1_interface": ("full_rgb", "sparse_rgb", "true_fmap", "oracle_jepa_bridge"),
    "h2_prediction": (
        "full_rgb_reference", "sparse_rgb_reference", "anchor_jepa_only",
        "oracle_jepa_hidden_reference", "predicted_jepa_hidden",
    ),
}
SUMMARY_TITLES = {
    "h0_state": "H0 State — Latent-State Feasibility",
    "h1_interface": "H1 Interface — Representation-Interface Feasibility",
    "h2_prediction": "H2 Prediction — Sparse-Anchor Prediction Feasibility",
}


def _fixed(value: Any, digits: int) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _percentage(value: Any, digits: int) -> str:
    return "—" if value is None else f"{100.0 * float(value):.{digits}f}%"


def _count(value: Any) -> str:
    return "—" if value is None else str(int(value))


def _result_summary_lines(module: str, result: Mapping[str, Any]) -> list[str]:
    """Render display-only values without mutating the authoritative result."""
    lines = [
        "| Condition | ATE RMSE (m) | translation RPE@1s (m) | rotation RPE@1s (deg) | coverage | nodes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    conditions = result["conditions"]
    for key in CONDITION_ORDER[module]:
        condition = conditions[key]
        evaluation = condition["canonical_evaluation"]
        runtime = condition["runtime"]
        lines.append(
            f"| {CONDITION_LABELS[key]} | {_fixed(evaluation['ate_rmse_m'], 4)} | "
            f"{_fixed(evaluation['translation_rpe_rmse_m'], 4)} | "
            f"{_fixed(evaluation['rotation_rpe_rmse_deg'], 2)} | "
            f"{_percentage(condition['canonical_coverage']['canonical_pose_coverage'], 1)} | "
            f"{_count(runtime['final_node_count_before_terminate'])} |"
        )
    if module == "h2_prediction":
        predicted = conditions["predicted_jepa_hidden"]
        if not predicted["strict_deployment"] or predicted["timestamp_causal"] is not False:
            raise ValueError("H2 summary requires strict non-causal deployment results")
        if predicted["closing_anchor_online_available"] is not True:
            raise ValueError("H2 summary requires bracketed closing-anchor availability")
        efficiency = result["efficiency"]
        transmission = efficiency["transmission"]
        wall = efficiency["matched_online_wall_clock"]
        stages = efficiency["h2_stage_profile"]["stages"]
        context = efficiency["h2_stage_profile"]["context_wait_ms"]
        break_even = efficiency["break_even_uplink_bandwidth"]
        bandwidth = break_even["break_even_uplink_bandwidth_mbps"]
        bandwidth_text = (f"{float(bandwidth):.3f} Mbps" if bandwidth is not None
                          else break_even["status"])
        lines.extend((
            "",
            "## Efficiency",
            "",
            "- Communication: anchor ratio "
            f"{_percentage(transmission['anchor_ratio'], 2)}; encoded byte reduction "
            f"{_percentage(transmission['encoded_byte_reduction'], 2)}",
            "- Cloud compute: Full RGB "
            f"{_fixed(wall['full_rgb_total_s'], 3)} s; H2 {_fixed(wall['h2_total_s'], 3)} s; "
            f"H2 / Full RGB {_fixed(wall['h2_over_full_rgb_ratio'], 3)}x; "
            f"extra {_fixed(wall['extra_cloud_compute_s'], 3)} s",
            "- H2 stage totals: JEPA encoder "
            f"{_fixed(stages['jepa_encoder']['total_ms'] / 1000.0, 3)} s; predictor "
            f"{_fixed(stages['jepa_predictor']['total_ms'] / 1000.0, 3)} s; bridge "
            f"{_fixed(stages['bridge']['total_ms'] / 1000.0, 3)} s; DPVO graph/runtime "
            f"{_fixed(stages['dpvo_graph_runtime']['total_ms'] / 1000.0, 3)} s",
            "- Latency: context wait mean "
            f"{_fixed(context['mean_ms'], 2)} ms; P95 {_fixed(context['p95_ms'], 2)} ms",
            f"- System: break-even uplink bandwidth {bandwidth_text}",
        ))
    return lines


def render_sequence_summary(module: str, sequence: str, result: Mapping[str, Any]) -> str:
    lines = [f"# {SUMMARY_TITLES[module]} — {sequence}", ""]
    lines.extend(_result_summary_lines(module, result))
    lines.extend(("", "`results.json` is the authoritative sequence result.", ""))
    return "\n".join(lines)


def write_sequence_metadata(
    directory: Path, module: str, sequence: str, result: Mapping[str, Any],
    lineage: Mapping[str, Any], provenance: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": 1, "module": module, "sequence": sequence,
        "lineage": dict(lineage), "provenance": dict(provenance), "result": dict(result),
    }
    atomic_write_json(directory / "results.json", payload)
    atomic_write_bytes(
        directory / SUMMARY_NAMES[module],
        render_sequence_summary(module, sequence, result).encode("utf-8"),
    )


def render_aggregate_summary(root: Path, module: str, index: Mapping[str, Any]) -> str:
    requested = list(index["requested_sequences"])
    lines = [f"# {SUMMARY_TITLES[module]}", "", "## Valid sequences", ""]
    for sequence in requested:
        payload = json.loads(
            (root / "sequences" / sequence / "results.json").read_text(encoding="utf-8")
        )
        lines.extend((f"### {sequence}", ""))
        lines.extend(_result_summary_lines(module, payload["result"]))
        lines.append("")
    return "\n".join(lines)


def write_registry_and_summary(root: Path, module: str, index: Mapping[str, Any]) -> None:
    summary_path = root / SUMMARY_NAMES[module]
    summary_temp = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    summary_temp.write_text(render_aggregate_summary(root, module, index), encoding="utf-8")
    os.replace(summary_temp, summary_path)
    atomic_write_json(root / "INDEX.json", index)


def validate_module_manifest(root: Path, module: str, index: Mapping[str, Any]) -> None:
    """Fail closed before publishing a freshly constructed canonical module."""
    if index.get("schema_version") != INDEX_SCHEMA_VERSION or index.get("module") != module:
        raise RuntimeError("canonical manifest schema/module mismatch")
    if index.get("run_policy") != "fresh_current_canonical_replace":
        raise RuntimeError("canonical manifest run policy mismatch")
    requested = list(index.get("requested_sequences", ()))
    if not requested or len(requested) != len(set(requested)):
        raise RuntimeError("canonical manifest has invalid requested sequences")
    entries = index.get("sequences")
    if not isinstance(entries, Mapping) or set(entries) != set(requested):
        raise RuntimeError("canonical manifest sequence set mismatch")

    checkpoint_name = {
        "h0_state": None, "h1_interface": "bridge.pt", "h2_prediction": "predictor.pt",
    }[module]
    expected_root = {"INDEX.json", SUMMARY_NAMES[module], "sequences"}
    checkpoint = index.get("canonical_checkpoint")
    if checkpoint_name is None:
        if checkpoint is not None:
            raise RuntimeError("H0 canonical manifest must not contain a checkpoint")
    else:
        expected_root.add(checkpoint_name)
        if not isinstance(checkpoint, Mapping) or checkpoint.get("file") != checkpoint_name:
            raise RuntimeError("canonical checkpoint manifest is missing")
        if sha256_file(root / checkpoint_name) != checkpoint.get("file_sha256"):
            raise RuntimeError("canonical checkpoint artifact hash mismatch")
    if {path.name for path in root.iterdir()} != expected_root:
        raise RuntimeError("canonical module contains unexpected root artifacts")

    sequence_root = root / "sequences"
    if {path.name for path in sequence_root.iterdir()} != set(requested):
        raise RuntimeError("canonical sequence directory set mismatch")
    for sequence in requested:
        entry = entries[sequence]
        if entry.get("status") != "valid":
            raise RuntimeError(f"canonical sequence is not valid: {sequence}")
        directory = sequence_root / sequence
        if artifact_hashes(directory, module) != entry.get("artifacts"):
            raise RuntimeError(f"canonical sequence artifact hash mismatch: {sequence}")
        try:
            payload = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"canonical sequence results are invalid: {sequence}") from error
        if payload.get("module") != module or payload.get("sequence") != sequence:
            raise RuntimeError(f"canonical sequence identity mismatch: {sequence}")
        if payload.get("lineage") != entry.get("lineage"):
            raise RuntimeError(f"canonical sequence lineage mismatch: {sequence}")
