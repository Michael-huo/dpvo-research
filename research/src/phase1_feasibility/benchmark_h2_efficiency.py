"""Read-only H2 hardware-utilization and bottleneck benchmark.

This command never calls the formal H2 runner, evaluates a trajectory, saves a
checkpoint, or publishes canonical results.  GPU-bearing modes are intended to
be launched manually on the three-RTX4090 server.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .efficiency_profiling import (
    PerformanceRecorder, TransferLedger, NvidiaSmiSampler, atomic_write_json,
    bootstrap_pipeline, canonical_artifact_snapshot, current_cuda_device, gpu_process_snapshot,
    gpu_topology_audit, pairwise_copy_benchmark, simulate_pipeline,
    validate_output_path,
)
from .h2_deployment import DelayedDeploymentProvider
from .h2_training import (
    _batches, _predict, _transport_batch, build_robust_correspondence_store,
)
from .jepa_fmap import coordinate_masks
from .jepa_runtime import extract_block5_store, load_dpvo_domain, sequence_geometry
from .predictor import predictor_metadata, predictor_state_sha256
from .profiling import OnlineProfiler
from .protocol import REPO_ROOT, load_sequence_records, repo_path, sha256_file
from .run_h2 import (
    TRAINING_SEQUENCE, _evaluation_protocol, _load_bridge, _new_predictor,
    _predictor_training_details, load_config,
)
from .runtime import materialize_schedule, run_deployment_observations, warmup_dpvo_frontend
from .dpvo_backend import _seed_everything
from .predictor import prediction_loss


SCHEMA = "h2_efficiency_audit_v1"


def _load_canonical_predictor(config: Mapping[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = repo_path(config["paths"]["output_root"]) / "predictor.pt"
    if not path.is_file():
        raise RuntimeError(f"canonical H2 predictor is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    forbidden = {"optimizer", "optimizer_state_dict", "grad_scaler", "best_state"}
    if forbidden & set(checkpoint) or "state_dict" not in checkpoint:
        raise RuntimeError("canonical predictor checkpoint schema is unsafe")
    if predictor_state_sha256(checkpoint["state_dict"]) != checkpoint.get("state_dict_sha256"):
        raise RuntimeError("canonical predictor state hash mismatch")
    model = _new_predictor(config)
    if predictor_metadata(model) != checkpoint.get("architecture"):
        raise RuntimeError("canonical predictor architecture mismatch")
    deployment = checkpoint.get("deployment_protocol", {})
    if (deployment.get("mode") != "delayed_bracketed"
            or deployment.get("timestamp_causal") is not False):
        raise RuntimeError("canonical predictor deployment protocol mismatch")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.cuda().requires_grad_(False).eval()
    return model, {
        "path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path),
        "state_dict_sha256": checkpoint["state_dict_sha256"],
        "training_lineage_sha256": checkpoint.get("training_lineage", {}).get(
            "training_lineage_sha256"
        ),
        "train_only_calibration": checkpoint["train_only_calibration"],
    }


def _selected_online(config: Mapping[str, Any], interval_count: int | None) -> dict[str, Any]:
    calibration = np.loadtxt(repo_path(config["paths"]["calibration"]), delimiter=" ")
    records = load_sequence_records(config, TRAINING_SEQUENCE)
    bootstrap = materialize_schedule(records, calibration, config)
    evaluation = _evaluation_protocol(
        records, int(bootstrap["bootstrap_end_candidate_index"]), config,
    )
    intervals = tuple(evaluation["intervals"])
    if interval_count is not None:
        if not 0 < interval_count <= len(intervals):
            raise ValueError(f"--intervals must be in [1,{len(intervals)}]")
        intervals = intervals[:interval_count]
        closing = intervals[-1].anchor1.key
        end = next(index for index, row in enumerate(evaluation["records"])
                   if row.identity.key == closing)
        records = tuple(evaluation["records"][:end + 1])
    else:
        records = tuple(evaluation["records"])
    keys = {row.identity.key for row in records}
    roles = {key: value for key, value in evaluation["roles"].items() if key in keys}
    return {"calibration": calibration, "records": records, "roles": roles,
            "intervals": intervals, "bootstrap": bootstrap}


def _copy_specs(transform: Any, records: Sequence[Any], intervals: Sequence[Any],
                calibration: np.ndarray,
                ) -> dict[str, tuple[Sequence[int], torch.dtype]]:
    height, width = load_dpvo_domain(records[0].rgb_path, calibration)[0].shape[:2]
    hidden = max(len(row.hidden) for row in intervals)
    return {
        "endpoint_block5": ((1, 768, transform.token_grid_height,
                              transform.token_grid_width), torch.float32),
        "predicted_block5": ((hidden, 768, transform.token_grid_height,
                               transform.token_grid_width), torch.float32),
        "bridge_fmap": ((hidden, 128, transform.fmap_height,
                          transform.fmap_width), torch.float32),
        "anchor_image": ((3, height, width), torch.uint8),
    }


def _copy_mean(payload: Mapping[str, Any], source: int, destination: int,
               tensor_name: str) -> float:
    for row in payload.get("pairs", []):
        if row["source"] == source and row["destination"] == destination:
            value = row["measurements"][tensor_name]
            if value["path"] == "direct_device_copy":
                return float(value["cuda_copy_ms"]["mean_ms"])
            return float(value["cpu_wall_ms"]["mean_ms"])
    raise RuntimeError(
        f"no measured transfer model for cuda:{source}->cuda:{destination}/{tensor_name}"
    )


def _pipeline_rows(provider: DelayedDeploymentProvider, profiler: OnlineProfiler,
                   performance: PerformanceRecorder, copy_audit: Mapping[str, Any]
                   ) -> tuple[list[dict[str, float]], dict[str, float], float]:
    intervals = provider.intervals
    first_left = intervals[0].anchor0.key
    bootstrap_count = next(index for index, identity in enumerate(provider.identities)
                           if identity.key == first_left) + 1
    encode_operation_names = (
        "temporary_npy_main_to_worker_write", "worker_main_request_response_wall",
        "temporary_npy_worker_to_main_read", "main_cpu_to_gpu_block5",
    )
    encoder_all = [sum(
        float(provider.transfer_ledger.rows[name][index]["cpu_ms"])
        for name in encode_operation_names
    ) for index in range(len(provider.transfer_ledger.rows[encode_operation_names[0]]))]
    encoder = encoder_all[-len(intervals):]
    decode = profiler.samples["anchor_decode_preprocess"][-len(intervals):]
    predictor = performance.outer_cpu_ms
    bridge = [float(row["cpu_ms"]) for row in provider.transfer_ledger.rows["bridge_forward_wall"]]
    graph = profiler.samples["dpvo_graph_runtime"][:-1]  # terminate is pipeline drain
    frontend = profiler.samples["native_dpvo_frontend"]
    if not all(len(values) == len(intervals) for values in (encoder, decode, predictor, bridge)):
        raise RuntimeError("per-interval stage samples do not align")
    transfer01 = _copy_mean(copy_audit, 0, 1, "endpoint_block5")
    transfer12 = _copy_mean(copy_audit, 1, 2, "bridge_fmap")
    rows = []
    cursor = bootstrap_count
    for index, interval in enumerate(intervals):
        observation_count = len(interval.hidden) + 1
        dpvo_ms = float(sum(graph[cursor:cursor + observation_count])
                        + frontend[-len(intervals):][index])
        cursor += observation_count
        rows.append({
            "anchor_available_ms": (
                interval.anchor1.timestamp_ns - provider.identities[0].timestamp_ns
            ) / 1e6,
            "encode_ms": float(decode[index] + encoder[index] + transfer01),
            "predict_ms": float(predictor[index]),
            "bridge_ms": float(bridge[index]),
            "transfer_ms": float(transfer12),
            "dpvo_ms": dpvo_ms,
        })
    startup = {
        "gpu0_ms": float(sum(
            profiler.samples["anchor_decode_preprocess"][:bootstrap_count]
        ) + sum(encoder_all[:bootstrap_count])),
        "gpu1_ms": 0.0,
        "gpu2_ms": float(sum(graph[:bootstrap_count]) + sum(frontend[:bootstrap_count])),
    }
    drain_ms = float(profiler.samples["dpvo_graph_runtime"][-1])
    return rows, startup, drain_ms


def _common_metadata(command: str, before: Mapping[str, str],
                     after: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "command": command, "pid": os.getpid(),
        "scientific_outputs_generated": False,
        "formal_h2_runner_called": False,
        "canonical_artifacts_before": dict(before),
        "canonical_artifacts_after": dict(after),
        "canonical_artifacts_unchanged": dict(before) == dict(after),
    }


def _online_diagnosis(
    performance: PerformanceRecorder, profiler: OnlineProfiler,
    ledger: TransferLedger, pipeline_rows: Sequence[Mapping[str, float]],
    compute_pipeline: Mapping[str, Any], sampler_payload: Mapping[str, Any],
    peak_vram_bytes: int,
) -> dict[str, Any]:
    cpu_fine = {
        name: float(sum(values)) for name, values in performance.cpu_samples.items()
        if name != "synchronization_wait"
    }
    cuda_fine = {name: float(sum(values))
                 for name, values in performance.cuda_samples.items()}
    cpu_system = {
        "predictor_outer_cpu_wall": float(sum(performance.outer_cpu_ms)),
        "anchor_decode_preprocess_cpu_wall": float(sum(
            profiler.samples["anchor_decode_preprocess"]
        )),
        "dpvo_graph_runtime_cpu_wall": float(sum(profiler.samples["dpvo_graph_runtime"])),
        "worker_main_request_response_cpu_wall": float(sum(
            row["cpu_ms"] for row in ledger.rows["worker_main_request_response_wall"]
        )),
    }
    cuda_system = {
        "predictor_outer_cuda_event": float(sum(performance.outer_cuda_ms)),
        "vjepa_encoder_cuda_event": float(sum(profiler.samples["jepa_encoder"])),
        "bridge_cuda_event": float(sum(profiler.samples["bridge"])),
        "native_dpvo_frontend_cuda_event": float(sum(
            profiler.samples["native_dpvo_frontend"]
        )),
    }
    devices = sampler_payload.get("devices", {})
    gpu_means = [row["gpu_utilization_percent"]["mean"] for row in devices.values()
                 if row["gpu_utilization_percent"]["mean"] is not None]
    total_memory = torch.cuda.get_device_properties(0).total_memory
    host_roundtrip_ms = float(sum(
        row["cpu_ms"] for name in (
            "temporary_npy_main_to_worker_write", "temporary_npy_worker_to_main_read",
            "main_cpu_to_gpu_block5",
            "previous_anchor_gpu_to_cpu", "previous_anchor_cpu_to_gpu",
        ) for row in ledger.rows[name]
    ))
    serial_interval_work = float(sum(
        row["encode_ms"] + row["predict_ms"] + row["bridge_ms"]
        + row["transfer_ms"] + row["dpvo_ms"] for row in pipeline_rows
    ))
    overlap_ms = max(0.0, serial_interval_work - float(compute_pipeline["makespan_ms"]))
    dominant_name, dominant_ms = max(cpu_fine.items(), key=lambda item: item[1])
    candidates = sorted((
        {"candidate": f"optimize_{dominant_name}", "evidence_ms": dominant_ms},
        {"candidate": "remove_gpu_cpu_npy_cpu_gpu_round_trips", "evidence_ms": host_roundtrip_ms},
        {"candidate": "implement_strict_three_stage_gpu_pipeline", "evidence_ms": overlap_ms},
    ), key=lambda row: row["evidence_ms"], reverse=True)
    return {
        "bottleneck_ranking_by_timing_domain": {
            "cpu_wall": sorted(cpu_system.items(), key=lambda item: item[1], reverse=True),
            "predictor_cpu_fine": sorted(cpu_fine.items(), key=lambda item: item[1], reverse=True),
            "cuda_event": sorted(cuda_system.items(), key=lambda item: item[1], reverse=True),
            "predictor_cuda_fine": sorted(cuda_fine.items(), key=lambda item: item[1], reverse=True),
            "cross_domain_values_are_not_combined": True,
        },
        "single_gpu_headroom": {
            "peak_allocated_fraction_of_gpu0_memory": peak_vram_bytes / total_memory,
            "maximum_sampled_gpu_utilization_mean_percent": max(gpu_means) if gpu_means else None,
            "interpretation": "use with power, memory utilization, synchronization and Nsight data",
        },
        "current_gpu_cpu_gpu_round_trip_present": True,
        "next_optimization_candidates_ranked_by_measured_opportunity": candidates,
    }


def _training_diagnosis(recorder: PerformanceRecorder, sampler: Mapping[str, Any],
                        peak_bytes: int) -> dict[str, Any]:
    total = float(sum(recorder.outer_cpu_ms))
    forward_backward = float(sum(recorder.cpu_samples["forward_loss"])
                             + sum(recorder.cpu_samples["backward"]))
    total_memory = torch.cuda.get_device_properties(0).total_memory
    devices = sampler.get("devices", {})
    utilization = [row["gpu_utilization_percent"]["mean"] for row in devices.values()
                   if row["gpu_utilization_percent"]["mean"] is not None]
    neural_fraction = forward_backward / total if total else 0.0
    memory_fraction = peak_bytes / total_memory
    mean_utilization = max(utilization) if utilization else None
    return {
        "larger_batch": {
            "candidate": memory_fraction < .60,
            "evidence": {"peak_memory_fraction": memory_fraction,
                         "sampled_gpu_utilization_mean_percent": mean_utilization},
        },
        "amp": {"currently_enabled": True, "change_made": False},
        "torch_compile": {
            "candidate": neural_fraction >= .50,
            "evidence_neural_forward_backward_cpu_wall_fraction": neural_fraction,
        },
        "multi_gpu_training": {
            "candidate": neural_fraction >= .50 and mean_utilization is not None
                         and mean_utilization >= 70.0,
            "reason": "recommended only when neural forward/backward dominates a busy GPU",
        },
    }


def _write_report(output: Path, payload: Mapping[str, Any]) -> None:
    sections = (
        "1. Current GPU device mapping", "2. Per-GPU RTX4090 utilization",
        "3. Predictor 81s decomposition", "4. Training time decomposition",
        "5. CPU/GPU transfer and IPC overhead", "6. Current bottleneck ranking",
        "7. Single-GPU software headroom", "8. Three-GPU parallelizable work",
        "9. Strict delayed/bracketed pipeline", "10. Online wall-clock lower bounds",
        "11. Three highest-value next optimizations",
    )
    online = payload.get("online", {})
    training = payload.get("training", {})
    timing = online.get("predictor_fine_profile", {})
    cpu_stages = timing.get("cpu", {}).get("exclusive_stages", {})
    ranking = payload.get("diagnosis", {}).get(
        "bottleneck_ranking_by_timing_domain",
        sorted(((name, float(value["total_ms"])) for name, value in cpu_stages.items()
                if name != "synchronization_wait"),
               key=lambda item: item[1], reverse=True),
    )
    lines = ["# H2 Efficiency Audit", "",
             "Performance diagnostics only; no scientific metric or canonical artifact was produced.", ""]
    bodies = {
        sections[0]: json.dumps(payload.get("devices", {}), indent=2, sort_keys=True),
        sections[1]: json.dumps(payload.get("gpu_utilization", {}), indent=2, sort_keys=True),
        sections[2]: json.dumps(timing, indent=2, sort_keys=True),
        sections[3]: json.dumps(training or {"status": "not measured by this invocation"}, indent=2),
        sections[4]: json.dumps(payload.get("transfer_ipc", {}), indent=2, sort_keys=True),
        sections[5]: json.dumps(ranking, indent=2),
        sections[6]: json.dumps(payload.get("diagnosis", {}).get(
            "single_gpu_headroom", payload.get("diagnosis", {}).get(
                "training_optimization_suitability", {}
            )), indent=2, sort_keys=True),
        sections[7]: "Legal overlap: GPU0 encode k+1, GPU1 predict+bridge k, GPU2 consume k-1.",
        sections[8]: "Closing anchor must be online before prediction; DPVO remains timestamp-ordered and stateful.",
        sections[9]: json.dumps(payload.get("pipeline", {}), indent=2, sort_keys=True),
        sections[10]: json.dumps(payload.get("diagnosis", {}).get(
            "next_optimization_candidates_ranked_by_measured_opportunity",
            {"status": "requires full online measurement"},
        ), indent=2, sort_keys=True),
    }
    for title in sections:
        lines.extend((f"## {title}", "", "```json" if bodies[title].startswith(("{", "[")) else "",
                      bodies[title], "```" if bodies[title].startswith(("{", "[")) else "", ""))
    (output / "AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def _new_output(path: str | Path) -> Path:
    output = validate_output_path(path)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"benchmark output must be absent or empty: {output}")
    return output


def _finalize(output: Path, payload: dict[str, Any], before: Mapping[str, str]) -> None:
    after = canonical_artifact_snapshot()
    payload.update(_common_metadata(payload["mode"], before, after))
    if before != after:
        raise RuntimeError("canonical artifacts changed during efficiency benchmark")
    output.mkdir(parents=True, exist_ok=True)
    audit = output / "audit.json"
    if audit.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark: {audit}")
    atomic_write_json(audit, payload)
    _write_report(output, payload)


@torch.no_grad()
def run_online(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("online efficiency benchmark requires CUDA")
    output = _new_output(args.output)
    before = canonical_artifact_snapshot()
    config, config_path = load_config()
    interval_count = None if args.intervals == "all" else int(args.intervals)
    selection = _selected_online(config, interval_count)
    records, roles, intervals = selection["records"], selection["roles"], selection["intervals"]
    transform, geometry = sequence_geometry(records[0], selection["calibration"], config)
    bridge, bridge_meta = _load_bridge(config, transform)
    predictor, predictor_meta = _load_canonical_predictor(config)
    performance = PerformanceRecorder(enable_cuda=True)
    ledger = TransferLedger(); online_profiler = OnlineProfiler()
    sampler = NvidiaSmiSampler(args.sample_ms)
    topology = gpu_topology_audit()
    devices = {
        name: current_cuda_device(name) for name in ("dpvo", "predictor", "bridge")
    }
    logical_mapping = {row["logical_index"]: row for row in topology["logical_devices"]}
    for row in devices.values():
        if row.get("logical_index") in logical_mapping:
            row.update({key: value for key, value in
                        logical_mapping[row["logical_index"]].items()
                        if key in {"physical_index", "pci_bus_id", "uuid",
                                   "mapping_basis", "mapping_error"}})
    with tempfile.TemporaryDirectory(prefix="h2_efficiency_online_") as temporary_name:
        temporary = Path(temporary_name)
        anchor_paths = {row.identity.key: row.rgb_path for row in records
                        if roles[row.identity.key] == "anchor"}
        provider = DelayedDeploymentProvider(
            anchor_paths=anchor_paths, identities=[row.identity for row in records],
            intervals=intervals, transform=transform, calibration=selection["calibration"],
            config=config, temporary=temporary / "provider", bridge=bridge,
            predictor=predictor, transport_calibration=predictor_meta["train_only_calibration"],
            profiler=online_profiler, performance=performance, transfer_ledger=ledger,
        )
        provider.temporary.mkdir(parents=True)
        warmup = warmup_dpvo_frontend(records[0], selection["calibration"], config,
                                      packet_runtime=True)
        process_before = gpu_process_snapshot()
        with provider.online_session():
            assert provider._sidecar is not None
            devices["v_jepa_worker"] = current_cuda_device(
                "v_jepa_worker", pid=provider._sidecar.process.pid,
            ) | {"mapping_basis": "worker hardcodes cuda:0 and inherits visibility"}
            if 0 in logical_mapping:
                devices["v_jepa_worker"].update({
                    key: value for key, value in logical_mapping[0].items()
                    if key in {"physical_index", "pci_bus_id", "uuid",
                               "mapping_basis", "mapping_error"}
                })
            process_active = gpu_process_snapshot()
            expected_pids = {os.getpid(), provider._sidecar.process.pid}
            process_active["unrelated_compute_processes"] = [
                row for row in process_active["rows"] if row["pid"] not in expected_pids
            ]
            process_active["sampling_contaminated"] = bool(
                process_active["unrelated_compute_processes"]
            )
            sampler.start()
            try:
                runtime, arrays = run_deployment_observations(
                    provider.observations(), selection["calibration"], config,
                    image_height=geometry["dpvo_domain_height"],
                    image_width=geometry["dpvo_domain_width"],
                    condition_name="h2_efficiency_audit_only", expected_roles=roles,
                    profiler=online_profiler, worker_barrier=provider,
                    on_tracked=provider.on_tracked,
                )
            finally:
                sampler.stop()
        del arrays
        process_after = gpu_process_snapshot()
        # Run topology copies only after utilization sampling, otherwise CUDA
        # allocator caches on GPU1/GPU2 would falsely imply current H2 usage.
        copy_audit = pairwise_copy_benchmark(_copy_specs(
            transform, records, intervals, selection["calibration"],
        ))
        if copy_audit["available"] and torch.cuda.device_count() >= 3:
            pipeline_rows, pipeline_startup, pipeline_drain = _pipeline_rows(
                provider, online_profiler, performance, copy_audit,
            )
            compute_pipeline = simulate_pipeline(
                pipeline_rows, sensor_paced=False, startup=pipeline_startup,
                drain_ms=pipeline_drain,
            )
            first_interval_start_ms = (
                intervals[0].anchor0.timestamp_ns - records[0].identity.timestamp_ns
            ) / 1e6
            sensor_startup = {
                name: max(value, first_interval_start_ms)
                for name, value in pipeline_startup.items()
            }
            sensor_pipeline = simulate_pipeline(
                pipeline_rows, sensor_paced=True, startup=sensor_startup,
                drain_ms=pipeline_drain,
            )
            compute_empirical = bootstrap_pipeline(
                pipeline_rows, sensor_paced=False, startup=pipeline_startup,
                drain_ms=pipeline_drain,
            )
            sensor_empirical = bootstrap_pipeline(
                pipeline_rows, sensor_paced=True, startup=sensor_startup,
                drain_ms=pipeline_drain,
            )
            pipeline_status = "modeled_from_measured_pairwise_transfers"
        else:
            pipeline_rows = []
            pipeline_startup = {}; pipeline_drain = 0.0
            compute_pipeline = sensor_pipeline = None
            compute_empirical = sensor_empirical = None
            pipeline_status = "unavailable_requires_three_visible_cuda_devices"
    sampler_payload = sampler.payload()
    sampler_payload["scope"] = (
        "strict_online_replay_only; excludes post-replay pairwise copy microbenchmark"
    )
    payload = {
        "mode": "online", "config_file": str(config_path.relative_to(REPO_ROOT)),
        "selection": {"sequence": TRAINING_SEQUENCE, "candidate_count": len(records),
                      "interval_count": len(intervals)},
        "artifacts": {"bridge": bridge_meta, "predictor": predictor_meta},
        "devices": {"components": devices, "topology": topology,
                    "processes_before": process_before, "processes_active": process_active,
                    "processes_after": process_after,
                    "current_mapping_conclusion": "all current components use logical cuda:0"},
        "gpu_utilization": sampler_payload,
        "transfer_ipc": {"current_path": ledger.payload(),
                         "temporary_handoff": "filesystem_npy",
                         "shared_memory_handoff_count": 0,
                         "confirmed_round_trips": [
                             "vjepa_gpu_to_cpu_to_npy_to_cpu_to_current_gpu",
                             "previous_anchor_gpu_to_cpu_to_current_gpu",
                         ],
                         "proposed_three_gpu_copy_measurements": copy_audit},
        "online": {
            "wall_clock_seconds": float(runtime["elapsed_seconds"]),
            "processed_observations": int(runtime["processed_observation_count"]),
            "dpvo_graph_runtime_total_ms": float(runtime["dpvo_graph_runtime_total_ms"]),
            "peak_gpu_vram_bytes": int(runtime["peak_gpu_vram_bytes"]),
            "jepa_worker_peak_gpu_vram_bytes": int(provider.jepa_peak_online_vram_bytes),
            "legacy_stage_profile": online_profiler.payload(
                peak_online_vram_bytes=int(runtime["peak_gpu_vram_bytes"]),
            ),
            "predictor_fine_profile": performance.payload(), "warmup": warmup,
            "scientific_trajectory_discarded": True,
        },
        "pipeline": {
            "compute_only": compute_pipeline,
            "sensor_paced": sensor_pipeline,
            "empirical_attainable_compute_only": compute_empirical,
            "empirical_attainable_sensor_paced": sensor_empirical,
            "status": pipeline_status,
            "transfer_model_is_measured": compute_pipeline is not None,
            "old_83_27s_label": (
                "current-kernel + proposed one-GPU-per-stage mapping resource lower bound"
            ),
        },
        "diagnosis": (_online_diagnosis(
            performance, online_profiler, ledger, pipeline_rows, compute_pipeline,
            sampler_payload, int(runtime["peak_gpu_vram_bytes"]),
        ) if compute_pipeline is not None else {
            "status": "online profile complete; three-GPU diagnosis requires three visible devices",
        }),
    }
    _finalize(output, payload, before)
    del predictor, bridge
    torch.cuda.empty_cache()
    return payload


def _training_batch(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: Any,
    intervals: Sequence[Any], store: Any, transform: Any, mask: torch.Tensor,
    robust: Any, recorder: PerformanceRecorder, *, amp: bool,
) -> tuple[float, int]:
    recorder.begin_outer()
    transported, target, alpha, delta = _transport_batch(
        intervals, store, transform, mask, robust, profiler=recorder,
    )
    optimizer.zero_grad(set_to_none=True)
    with recorder.stage("forward_loss"):
        with torch.cuda.amp.autocast(enabled=bool(amp)):
            loss = prediction_loss(_predict(model, transported, alpha, delta), target, mask)["total"]
    with recorder.stage("backward"):
        scaler.scale(loss).backward()
    with recorder.stage("optimizer_scaler"):
        scaler.step(optimizer); scaler.update()
    with recorder.stage("synchronization_wait", cuda=False):
        value = float(loss.detach())
    recorder.finish_outer()
    return value, len(target)


@torch.no_grad()
def _validation_batch(
    model: torch.nn.Module, intervals: Sequence[Any], store: Any, transform: Any,
    mask: torch.Tensor, robust: Any, recorder: PerformanceRecorder,
) -> tuple[int, float]:
    recorder.begin_outer()
    transported, target, alpha, delta = _transport_batch(
        intervals, store, transform, mask, robust, profiler=recorder,
    )
    with recorder.stage("validation_forward_loss"):
        metrics = prediction_loss(_predict(model, transported, alpha, delta), target, mask)
    with recorder.stage("synchronization_wait", cuda=False):
        # Preserve the canonical validation synchronization pattern without
        # retaining scientific metric values in the audit output.
        values = {key: float(metrics[key]) for key in
                  ("total", "cosine", "mse", "smooth_l1", "norm_ratio")}
    recorder.finish_outer()
    return len(target), values["total"]


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("training efficiency benchmark requires CUDA")
    output = _new_output(args.output); before = canonical_artifact_snapshot()
    config, config_path = load_config()
    predictor_checkpoint, predictor_meta = _load_canonical_predictor(config)
    del predictor_checkpoint
    calibration = np.loadtxt(repo_path(config["paths"]["calibration"]), delimiter=" ")
    records = load_sequence_records(config, TRAINING_SEQUENCE)
    bootstrap = materialize_schedule(records, calibration, config)
    transform, _ = sequence_geometry(records[0], calibration, config)
    # Use the same frozen split without invoking formal training lineage/publication.
    details = _predictor_training_details(
        records, int(bootstrap["bootstrap_end_candidate_index"]), calibration,
        config, {"performance_audit_only": True},
    )
    development = (*details["split"]["train"], *details["split"]["validation"])
    sampler = NvidiaSmiSampler(args.sample_ms); topology = gpu_topology_audit()
    recorder = PerformanceRecorder(enable_cuda=True)
    validation_recorder = PerformanceRecorder(enable_cuda=True)
    ledger = TransferLedger()
    with tempfile.TemporaryDirectory(prefix="h2_efficiency_training_") as temporary_name:
        temporary = Path(temporary_name)
        extraction_started = time.perf_counter()
        store, extraction = extract_block5_store(
            records,
            tuple({identity.key: identity for interval in development for identity in
                   (interval.anchor0, *[query.identity for query in interval.hidden],
                    interval.anchor1)}.values()),
            calibration, config, temporary, transform,
        )
        extraction_seconds = time.perf_counter() - extraction_started
        mask = torch.from_numpy(coordinate_masks(transform)["valid_token_mask"]).cuda()
        robust_started = time.perf_counter()
        robust, robust_meta = build_robust_correspondence_store(
            development, store, transform, mask,
            predictor_meta["train_only_calibration"],
        )
        correspondence_seconds = time.perf_counter() - robust_started
        _seed_everything(1234)
        model = _new_predictor(config).cuda().train()
        training_config = config["training"]
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=bool(training_config["amp"]))
        sample, _, sample_alpha, sample_delta = _transport_batch(
            details["split"]["train"][:1], store, transform, mask, robust,
        )
        if not torch.equal(_predict(model, sample, sample_alpha, sample_delta), sample.field):
            raise RuntimeError("canonical zero-initialization preflight changed")
        intervals_per_batch = int(training_config["intervals_per_batch"])
        batches = list(_batches(
            details["split"]["train"], intervals_per_batch, 1234,
        ))
        if not args.one_epoch:
            batches = batches[:int(args.batches)]
        torch.cuda.reset_peak_memory_stats()
        sampler.start(); epoch_started = time.perf_counter()
        total = 0.0; queries = 0; queries_per_batch = []
        validation_queries = 0; validation_seconds = None
        checkpoint_selection_seconds = None
        try:
            for batch in batches:
                before_data_samples = len(recorder.cpu_samples["batch_data_memmap_h2d"])
                value, count = _training_batch(
                    model, optimizer, scaler, batch, store, transform, mask, robust, recorder,
                    amp=bool(training_config["amp"]),
                )
                if len(recorder.cpu_samples["batch_data_memmap_h2d"]) != before_data_samples + 1:
                    raise RuntimeError("training data-transfer timing count changed")
                field_bytes = 768 * transform.token_grid_height * transform.token_grid_width * 4
                correspondence_bytes = sum(
                    getattr(robust.rows[interval.interval_index], name).numel() * 4
                    * len(interval.hidden)
                    for interval in batch for name in robust.rows[interval.interval_index].__dataclass_fields__
                )
                ledger.add(
                    "training_memmap_and_correspondence_cpu_to_gpu",
                    byte_count=(2 * len(batch) + count) * field_bytes + correspondence_bytes,
                    cpu_ms=recorder.cpu_samples["batch_data_memmap_h2d"][-1],
                )
                total += value * count; queries += count
                queries_per_batch.append(count)
            if args.one_epoch:
                model.eval(); validation_started = time.perf_counter()
                validation_total = 0.0
                for batch in _batches(
                    details["split"]["validation"], intervals_per_batch,
                ):
                    count, value = _validation_batch(
                        model, batch, store, transform, mask, robust, validation_recorder,
                    )
                    validation_queries += count
                    validation_total += value * count
                validation_seconds = time.perf_counter() - validation_started
                # The first epoch is necessarily the best-so-far epoch in the
                # canonical loop, so include its state clone in epoch throughput.
                selection_started = time.perf_counter()
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                checkpoint_selection_seconds = time.perf_counter() - selection_started
                ledger.add(
                    "checkpoint_selection_gpu_to_cpu",
                    byte_count=sum(value.numel() * value.element_size()
                                   for value in best_state.values()),
                    cpu_ms=checkpoint_selection_seconds * 1000.0,
                )
                del best_state, validation_total
        finally:
            sampler.stop()
        epoch_seconds = time.perf_counter() - epoch_started
        store.close()
    sampler_payload = sampler.payload()
    sampler_payload["scope"] = (
        "train_epoch_plus_validation_and_state_clone; excludes offline JEPA extraction "
        "and correspondence precompute"
    )
    peak_allocated = int(torch.cuda.max_memory_allocated())
    training_device = current_cuda_device("training")
    logical_mapping = {row["logical_index"]: row for row in topology["logical_devices"]}
    if training_device.get("logical_index") in logical_mapping:
        training_device.update({
            key: value
            for key, value in logical_mapping[training_device["logical_index"]].items()
            if key in {"physical_index", "pci_bus_id", "uuid", "mapping_basis",
                       "mapping_error"}
        })
    payload = {
        "mode": "training", "config_file": str(config_path.relative_to(REPO_ROOT)),
        "artifacts": {"predictor": predictor_meta},
        "devices": {"components": {"training": training_device},
                    "topology": topology},
        "gpu_utilization": sampler_payload, "transfer_ipc": ledger.payload(),
        "training": {
            "one_complete_train_epoch": bool(args.one_epoch),
            "validation_executed": bool(args.one_epoch),
            "validation_hidden_query_count": validation_queries,
            "validation_seconds": validation_seconds,
            "checkpoint_selection_state_clone_seconds": checkpoint_selection_seconds,
            "batch_count": len(batches), "hidden_query_count": queries,
            "hidden_queries_per_batch": {
                "min": min(queries_per_batch), "max": max(queries_per_batch),
                "mean": sum(queries_per_batch) / len(queries_per_batch),
            },
            "epoch_or_partial_epoch_seconds": epoch_seconds,
            "offline_jepa_extraction_seconds": extraction_seconds,
            "correspondence_precompute_seconds": correspondence_seconds,
            "correspondence_metadata": robust_meta,
            "offline_jepa_extraction_metadata": extraction,
            "batch_profile": recorder.payload(),
            "validation_profile": validation_recorder.payload(),
            "current_intervals_per_batch": intervals_per_batch,
            "amp": bool(training_config["amp"]),
            "parameter_dtype": str(next(model.parameters()).dtype),
            "autocast_dtype": str(torch.get_autocast_gpu_dtype()),
            "peak_allocated_vram_bytes": peak_allocated,
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
            "optimizer_or_checkpoint_saved": False,
        },
        "pipeline": {},
        "diagnosis": {"training_optimization_suitability": _training_diagnosis(
            recorder, sampler_payload, peak_allocated,
        )},
    }
    _finalize(output, payload, before)
    del model, optimizer, scaler, robust
    torch.cuda.empty_cache()
    return payload


def run_smoke() -> dict[str, Any]:
    recorder = PerformanceRecorder(enable_cuda=False)
    recorder.begin_outer()
    with recorder.stage("endpoint_descriptor_matching", cuda=False):
        _ = sum(range(100))
    with recorder.stage("synchronization_wait", cuda=False):
        pass
    recorder.finish_outer()
    rows = [
        {"anchor_available_ms": 100.0, "encode_ms": 3.0, "predict_ms": 7.0,
         "bridge_ms": 1.0, "transfer_ms": 0.5, "dpvo_ms": 5.0},
        {"anchor_available_ms": 200.0, "encode_ms": 3.0, "predict_ms": 7.0,
         "bridge_ms": 1.0, "transfer_ms": 0.5, "dpvo_ms": 5.0},
    ]
    return {"schema": SCHEMA, "mode": "smoke", "timing": recorder.payload(),
            "topology": gpu_topology_audit(),
            "pipeline": {"compute_only": simulate_pipeline(rows, sensor_paced=False),
                         "sensor_paced": simulate_pipeline(rows, sensor_paced=True)}}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("smoke", help="CPU/synthetic instrumentation smoke")
    online = subparsers.add_parser("online", help="read-only strict online replay")
    online.add_argument("--intervals", default="4",
                        help="positive complete interval count or 'all'")
    online.add_argument("--sample-ms", type=int, default=200)
    online.add_argument("--output", required=True)
    training = subparsers.add_parser("training", help="discarded-state training throughput audit")
    group = training.add_mutually_exclusive_group(required=True)
    group.add_argument("--batches", type=int)
    group.add_argument("--one-epoch", action="store_true")
    training.add_argument("--sample-ms", type=int, default=200)
    training.add_argument("--output", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.mode == "smoke":
        payload = run_smoke()
    elif args.mode == "online":
        if args.intervals != "all" and int(args.intervals) <= 0:
            raise ValueError("--intervals must be positive or 'all'")
        payload = run_online(args)
    else:
        if args.batches is not None and args.batches <= 0:
            raise ValueError("--batches must be positive")
        payload = run_training(args)
    print(json.dumps({"status": "complete", "mode": args.mode,
                      "schema": payload.get("schema", SCHEMA)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
