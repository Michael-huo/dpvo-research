"""Model, deployment, and run manifests with content-based scientific lineage."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from latent_vslam.protocol import (
    REPO_ROOT, atomic_write_bytes, atomic_write_json, canonical_sha256,
    load_sequence_records, repo_path, sha256_file,
)

from latent_vslam.artifact_runtime import (
    publish_current_canonical, validate_lightweight_results, without_worker_log_paths,
)
from latent_vslam.dataset_paths import groundtruth_path

CHECKPOINT_ROOTS = {"bridge": REPO_ROOT / "checkpoints/bridge",
                    "inference": REPO_ROOT / "checkpoints/predictor"}

INDEX_SCHEMA_VERSION = 2
MODULES = ("inference",)
SUMMARY_NAMES = {"inference": "SUMMARY.md"}
SEQUENCE_FILES = {"inference": (
    "results.json", "SUMMARY.md", "trajectory.png", "trajectories.npz",
    "feature_diagnostics.png",
)}
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
        raise KeyError("config has no calibration path")
    return repo_path(value)


def dataset_fingerprint(config: Mapping[str, Any], sequence: str) -> dict[str, Any]:
    """Hash manifests and evaluation inputs without reading every RGB payload."""
    records = load_sequence_records(config, sequence)
    first_rgb = Path(records[0].rgb_path)
    camera = first_rgb.parents[1]
    groundtruth = groundtruth_path(sequence)
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
    dataset = payload.get("dataset", {})
    dataset.pop("root", None)
    dataset.pop("groundtruth_pattern", None)
    jepa = payload.get("jepa", {})
    jepa.pop("checkpoint", None)
    # All paths in this block are represented by content hashes below or are
    # execution locations. No filesystem location is a scientific parameter.
    payload.pop("paths", None)
    payload.pop("runtime", None)
    def reject_absolute(value: Any) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                reject_absolute(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                reject_absolute(child)
        elif isinstance(value, str) and Path(value).is_absolute():
            raise ValueError(f"absolute filesystem path in scientific config: {value}")
    reject_absolute(payload)
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
    source_names = {Path(p).name for p in source_paths}
    scientific = None
    if "bridge_training_runtime.py" in source_names or "inference_runtime.py" in source_names:
        from latent_vslam.scientific_lineage import scientific_fingerprint
        scientific = scientific_fingerprint("h2" if "inference_runtime.py" in source_names else "h1")
        sources["scientific_dependencies"] = scientific
    lineage = {
        "dataset_sha256": dataset["dataset_sha256"],
        "config_protocol_sha256": protocol["config_protocol_sha256"],
        "source_sha256": (scientific or sources)["source_sha256"],
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
        "model_manifest": None,
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
        "model_lineage": dict(lineage),
        "artifacts": artifact_hashes(directory, module),
    }


CONDITION_LABELS = {
    "full_rgb_reference": "Full RGB", "sparse_rgb_reference": "Sparse RGB",
    "oracle_jepa_hidden_reference": "Oracle JEPA",
    "predicted_jepa_hidden": "Predicted JEPA",
}
CONDITION_ORDER = {"inference": (
    "full_rgb_reference", "sparse_rgb_reference",
    "oracle_jepa_hidden_reference", "predicted_jepa_hidden",
)}
SUMMARY_TITLES = {"inference": "Inference"}


def _fixed(value: Any, digits: int) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _percentage(value: Any, digits: int) -> str:
    return "—" if value is None else f"{100.0 * float(value):.{digits}f}%"


def _count(value: Any) -> str:
    return "—" if value is None else str(int(value))


def _performance_summary_lines(
    performance: Mapping[str, Any] | None, *, title: str = "Performance diagnostics",
) -> list[str]:
    if not performance:
        return []
    cpu = performance.get("cpu_wall", {})
    outer = cpu.get("outer", {})
    utilization = performance.get("gpu_utilization", {}).get("devices", {})
    gpu_rows = []
    for row in sorted(utilization.values(), key=lambda value: value.get("physical_index", 999)):
        stats = row.get("gpu_utilization_percent", {})
        gpu_rows.append(
            f"GPU{row.get('physical_index', '?')} mean {_fixed(stats.get('mean'), 1)}%, "
            f"P95 {_fixed(stats.get('p95'), 1)}%"
        )
    lines = ["", f"## {title}", "",
             "- Diagnostic-only; excluded from scientific metrics and decisions.",
             f"- Scope `{performance.get('scope', 'unknown')}` CPU wall: "
             f"{_fixed(float(outer.get('total_ms', 0.0)) / 1000.0, 3)} s."]
    if gpu_rows:
        lines.append("- GPU utilization: " + "; ".join(gpu_rows) + ".")
    residency = performance.get("residency")
    if isinstance(residency, Mapping):
        lines.append(
            "- Training residency: "
            f"{_count(residency.get('resident_rows'))}/"
            f"{_count(residency.get('allowed_rows'))} rows, "
            f"{_fixed(float(residency.get('resident_bytes', 0)) / 2**30, 3)} GiB, "
            f"initialization {_fixed(residency.get('initialization_seconds'), 3)} s; "
            f"remaining batch H2D {_fixed(residency.get('batch_h2d_seconds'), 6)} s."
        )
        lines.append(
            "- Resident VRAM delta: allocated "
            f"{_fixed(float(residency.get('allocated_delta_bytes', 0)) / 2**30, 3)} GiB; "
            "reserved "
            f"{_fixed(float(residency.get('reserved_delta_bytes', 0)) / 2**30, 3)} GiB."
        )
        if residency.get("correspondence_resident"):
            lines.append(
                "- Predictor correspondence residency: enabled, "
                f"{_fixed(float(residency.get('correspondence_resident_bytes', 0)) / 2**30, 3)} GiB."
            )
    efficiency = performance.get("efficiency")
    if isinstance(efficiency, Mapping):
        lines.append(
            "- Training pipeline: parallel preparation "
            f"{_fixed(efficiency.get('parallel_preparation_seconds'), 3)} s; "
            f"resident training {_fixed(efficiency.get('resident_training_wall_seconds'), 3)} s."
        )
        if efficiency.get("parallel_correspondence_seconds") is not None:
            lines.append(
                "- Multi-GPU correspondence precompute: "
                f"{_fixed(efficiency.get('parallel_correspondence_seconds'), 3)} s."
            )
    scheduler = performance.get("sequential_evaluation")
    if isinstance(scheduler, Mapping):
        assignments = ", ".join(
            f"job{index}({row.get('sequence', '?')}/"
            f"{row.get('condition') or row.get('kind', '?')})"
            f"→GPU{row.get('logical_device')}"
            for index, row in enumerate(scheduler.get("jobs", ()))
        )
        lines.append(
            f"- Canonical sequential evaluation: {scheduler.get('job_count', 0)} logical cuda:0 jobs; "
            f"makespan {_fixed(scheduler.get('makespan_seconds'), 3)} s"
            + (f"; {assignments}." if assignments else ".")
        )
    strict = performance.get("strict_predictor")
    if strict:
        cpu_outer = strict.get("cpu", {}).get("outer", {}).get("total_ms")
        cuda_outer = strict.get("cuda", {}).get("outer", {}).get("total_ms")
        lines.append(
            "- Predictor timing domains: CPU outer "
            f"{_fixed(cpu_outer / 1000.0 if cpu_outer is not None else None, 3)} s; "
            "CUDA-event outer "
            f"{_fixed(cuda_outer / 1000.0 if cuda_outer is not None else None, 3)} s; "
            "the two domains are not combined."
        )
    return lines


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
    if module == "inference":
        predicted = conditions["predicted_jepa_hidden"]
        if not predicted["strict_deployment"] or predicted["timestamp_causal"] is not False:
            raise ValueError("Inference summary requires strict non-causal deployment results")
        if predicted["closing_anchor_online_available"] is not True:
            raise ValueError("Inference summary requires bracketed closing-anchor availability")
        efficiency = result["efficiency"]
        transmission = efficiency["transmission"]
        wall = efficiency["matched_online_wall_clock"]
        stages = efficiency["prediction_stage_profile"]["stages"]
        stage_c_timing = efficiency["prediction_stage_profile"].get("stage_c_timing", {})
        stage_c_cpu = stage_c_timing.get("cpu_wall_exclusive", {})
        stage_c_cuda = stage_c_timing.get("cuda_event", {})
        queue_breakdown = stage_c_timing.get("queue_wait_breakdown", {})
        context = efficiency["prediction_stage_profile"]["context_wait_ms"]
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
            f"{_fixed(wall['full_rgb_total_s'], 3)} s; Ours {_fixed(wall['ours_total_s'], 3)} s; "
            f"Ours / Full RGB {_fixed(wall['ours_over_full_rgb_ratio'], 3)}x; "
            f"extra {_fixed(wall['extra_cloud_compute_s'], 3)} s",
            "- Prediction stage totals: JEPA encoder "
            f"{_fixed(stages['jepa_encoder']['total_ms'] / 1000.0, 3)} s; predictor "
            f"{_fixed(stages['jepa_predictor']['total_ms'] / 1000.0, 3)} s; bridge "
            f"{_fixed(stages['bridge']['total_ms'] / 1000.0, 3)} s; DPVO graph CUDA-event span "
            f"{_fixed(stages['dpvo_graph_runtime']['total_ms'] / 1000.0, 3)} s",
            "- Latency: context wait mean "
            f"{_fixed(context['mean_ms'], 2)} ms; P95 {_fixed(context['p95_ms'], 2)} ms",
            f"- System: break-even uplink bandwidth {bandwidth_text}",
        ))
        if stage_c_cpu:
            lines.append(
                "- Stage C exclusive CPU wall: queue "
                f"{_fixed(stage_c_cpu['stage_c_queue_wait']['total_ms'] / 1000.0, 3)} s; "
                "transfer "
                f"{_fixed(stage_c_cpu['stage_c_transfer_wait']['total_ms'] / 1000.0, 3)} s; "
                "bridge "
                f"{_fixed(stage_c_cpu['bridge_compute']['total_ms'] / 1000.0, 3)} s; "
                "native frontend "
                f"{_fixed(stage_c_cpu['native_frontend_compute']['total_ms'] / 1000.0, 3)} s; "
                "DPVO call/launch "
                f"{_fixed(stage_c_cpu['dpvo_graph_compute']['total_ms'] / 1000.0, 3)} s; "
                "DPVO sync "
                f"{_fixed(stage_c_cpu['dpvo_sync_wait']['total_ms'] / 1000.0, 3)} s; "
                "Python/control "
                f"{_fixed(stage_c_cpu['stage_c_python_other']['total_ms'] / 1000.0, 3)} s."
            )
            lines.append(
                "- Stage C CUDA events (separate time domain): bridge "
                f"{_fixed(stage_c_cuda['bridge_compute']['total_ms'] / 1000.0, 3)} s; "
                "native frontend "
                f"{_fixed(stage_c_cuda['native_frontend_compute']['total_ms'] / 1000.0, 3)} s; "
                "DPVO graph CUDA-event span "
                f"{_fixed(stage_c_cuda['dpvo_graph_compute']['total_ms'] / 1000.0, 3)} s."
            )
            lines.append(
                "- Stage C queue detail: prediction-ready "
                f"{_fixed(queue_breakdown.get('prediction_ready_wait', {}).get('total_ms'), 2)} ms; "
                "consumer dequeue "
                f"{_fixed(queue_breakdown.get('consume_queue_wait', {}).get('total_ms'), 2)} ms; "
                "worker flush "
                f"{_fixed(queue_breakdown.get('worker_flush_wait', {}).get('total_ms'), 2)} ms."
            )
            backpressure = efficiency["prediction_stage_profile"].get(
                "producer_queue_backpressure", {}
            )
            lines.append(
                "- Producer queue backpressure: "
                f"{_fixed(backpressure.get('total_ms'), 2)} ms total; P95 "
                f"{_fixed(backpressure.get('p95_ms'), 2)} ms."
            )
        acceptance = efficiency["prediction_stage_profile"].get("numerical_acceptance")
        if isinstance(acceptance, Mapping):
            lines.append(
                "- Three-GPU numerical acceptance: "
                f"{'passed' if acceptance.get('accepted') else 'failed'}; "
                "identity exact "
                f"{acceptance.get('identity_exact')}; "
                "correspondence decisions exact "
                f"{acceptance.get('correspondence_decisions_exact')}."
            )
            for boundary in (
                "anchor_jepa", "predictor_jepa", "bridge_fmap", "packet_payload",
            ):
                row = acceptance.get("continuous", {}).get(boundary, {})
                lines.append(
                    f"  - `{boundary}`: max_abs {_fixed(row.get('max_abs'), 9)}, "
                    f"MSE {_fixed(row.get('MSE'), 12)}, normalized MSE "
                    f"{_fixed(row.get('normalized_MSE'), 12)}, cosine "
                    f"{_fixed(row.get('cosine'), 12)}."
                )
        stage_profile = efficiency["prediction_stage_profile"]
        segments = stage_profile.get("pipeline_wall_segments", {})
        if segments:
            lines.append(
                "- Pipeline wall: fill "
                f"{_fixed(float(segments.get('fill_ms', 0.0)) / 1000.0, 3)} s; "
                "steady "
                f"{_fixed(float(segments.get('steady_ms', 0.0)) / 1000.0, 3)} s; "
                "drain "
                f"{_fixed(float(segments.get('drain_ms', 0.0)) / 1000.0, 3)} s."
            )
        lines.append(
            "- Peak VRAM by process: logical cuda:2 V-JEPA "
            f"{_fixed(float(stage_profile.get('peak_online_vram_jepa_worker_bytes', 0)) / 2**30, 3)} GiB; "
            "logical cuda:1 predictor "
            f"{_fixed(float(stage_profile.get('peak_online_vram_predictor_worker_bytes', 0)) / 2**30, 3)} GiB; "
            "logical cuda:0 frontend/bridge/DPVO "
            f"{_fixed(float(stage_profile.get('peak_online_vram_main_process_bytes', 0)) / 2**30, 3)} GiB."
        )
    lines.extend(_performance_summary_lines(result.get("performance_diagnostics")))
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
        "model_manifest": {"lineage": dict(lineage)},
        "deployment_manifest": dict(result.get("deployment_lineage", {})),
        "run_manifest": dict(provenance), "result": dict(result),
    }
    atomic_write_json(directory / "results.json", without_worker_log_paths(payload))
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
    checkpoint = index.get("model_manifest", {}).get("predictor")
    if isinstance(checkpoint, Mapping):
        training = checkpoint.get("training", {})
        lines.extend(_performance_summary_lines(
            training.get("performance_diagnostics"),
            title="Canonical training performance diagnostics",
        ))
        if lines and lines[-1] != "":
            lines.append("")
    execution = index.get("run_manifest")
    if isinstance(execution, Mapping):
        lines.extend(("## Formal execution", ""))
        lines.append(
            "- Execution implementation is provenance-only and excluded from scientific "
            "checkpoint compatibility."
        )
        if execution.get("total_makespan_seconds") is not None:
            lines.append(
                "- Formal command makespan: "
                f"{_fixed(execution.get('total_makespan_seconds'), 3)} s."
            )
        if execution.get("estimated") is True:
            lines.append("- Full three-sequence wall time remains an estimate until this command completes.")
        mapping = execution.get("hardware", {}).get("hardware", {}).get("mapping")
        if mapping:
            lines.append(
                f"- GPU mapping: preparation {mapping['preparation']}; standalone trajectories {mapping['standalone_trajectory']}; "
                "Predicted JEPA "
                "logical cuda:2 V-JEPA, logical cuda:1 predictor, logical cuda:0 native frontend/bridge/DPVO."
            )
        scheduler = execution.get("sequential_evaluation") or execution.get(
            "sequential_trajectory_execution"
        )
        if isinstance(scheduler, Mapping):
            lines.append(
                f"- Sequential logical cuda:0 trajectory makespan: "
                f"{_fixed(scheduler.get('makespan_seconds'), 3)} s for "
                f"{scheduler.get('job_count', 0)} jobs; maximum concurrent DPVO instances 1."
            )
        preparation = execution.get("parallel_preparation")
        if isinstance(preparation, Mapping):
            lines.append(
                "- Multi-GPU preparation: "
                f"{_fixed(preparation.get('elapsed_seconds'), 3)} s; "
                f"ordered payload hash `{preparation.get('ordered_content_sha256', 'n/a')}`."
            )
        evaluation_preparation = execution.get("evaluation_preparation")
        if isinstance(evaluation_preparation, Mapping):
            lines.append(
                "- Bridge evaluation preparation: "
                + "; ".join(
                    f"{sequence} {_fixed(row.get('elapsed_seconds'), 3)} s"
                    for sequence, row in evaluation_preparation.items()
                ) + "."
            )
        correspondence = execution.get("parallel_correspondence")
        if isinstance(correspondence, Mapping):
            lines.append(
                "- Multi-GPU correspondence precompute: "
                f"{_fixed(correspondence.get('elapsed_seconds'), 3)} s; "
                f"ordered payload hash `{correspondence.get('ordered_content_sha256', 'n/a')}`."
            )
        cpu_profile = execution.get("cpu_numa_profile")
        if isinstance(cpu_profile, Mapping):
            components = cpu_profile.get("components", {})
            stage_c = components.get("stage_c", {})
            predictor = components.get("predictor", {})
            encoder = components.get("encoder", {})
            lines.append(
                "- Inference CPU/NUMA profile: "
                f"`{cpu_profile.get('selection')}`; dynamic calibration "
                f"{cpu_profile.get('dynamic_calibration')}; "
                "logical cuda:0 Stage C OMP/MKL/intra/inter "
                f"{stage_c.get('omp_num_threads')}/{stage_c.get('mkl_num_threads')}/"
                f"{stage_c.get('intraop_threads')}/{stage_c.get('interop_threads')}; "
                "logical cuda:1 predictor "
                f"{predictor.get('omp_num_threads')}/{predictor.get('mkl_num_threads')}/"
                f"{predictor.get('intraop_threads')}/{predictor.get('interop_threads')}; "
                "logical cuda:2 V-JEPA "
                f"{encoder.get('omp_num_threads')}/{encoder.get('mkl_num_threads')}/"
                f"{encoder.get('intraop_threads')}/{encoder.get('interop_threads')}."
            )
        if execution.get("predicted_jepa_sequence_policy"):
            lines.append(
                "- Predicted JEPA sequences use the same exclusive three-GPU pipeline "
                f"sequentially; online wall total {_fixed(execution.get('predicted_jepa_total_seconds'), 3)} s."
            )
        lines.append("")
        control_performance = execution.get("trajectory_performance")
        if isinstance(control_performance, Mapping):
            lines.extend(_performance_summary_lines(
                control_performance, title="Sequential trajectory performance",
            ))
            if lines and lines[-1] != "":
                lines.append("")
    return "\n".join(lines)


def write_registry_and_summary(root: Path, module: str, index: Mapping[str, Any]) -> None:
    summary_path = root / SUMMARY_NAMES[module]
    summary_temp = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    summary_temp.write_text(render_aggregate_summary(root, module, index), encoding="utf-8")
    os.replace(summary_temp, summary_path)
    atomic_write_json(root / "INDEX.json", without_worker_log_paths(index))


def validate_module_manifest(root: Path, module: str, index: Mapping[str, Any], *,
                             checkpoint_root: Path | None = None) -> None:
    """Fail closed before publishing a freshly constructed canonical module."""
    validate_lightweight_results(root)
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

    checkpoint_name = "predictor.pt"
    expected_root = {"INDEX.json", SUMMARY_NAMES[module], "sequences"}
    model_manifest = index.get("model_manifest")
    checkpoint = model_manifest.get("predictor") if isinstance(model_manifest, Mapping) else None
    canonical_path = CHECKPOINT_ROOTS[module] / checkpoint_name
    if not isinstance(checkpoint, Mapping) or checkpoint.get("file") != str(canonical_path.relative_to(REPO_ROOT)):
        raise RuntimeError("predictor model manifest is missing")
    checkpoint_root = checkpoint_root or CHECKPOINT_ROOTS[module]
    if not (checkpoint_root / checkpoint_name).is_file():
        raise RuntimeError("predictor checkpoint is missing")
    if sha256_file(checkpoint_root / checkpoint_name) != checkpoint.get("file_sha256"):
        raise RuntimeError("predictor checkpoint hash mismatch")
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
        if payload.get("model_manifest", {}).get("lineage") != entry.get("model_lineage"):
            raise RuntimeError(f"canonical sequence lineage mismatch: {sequence}")
