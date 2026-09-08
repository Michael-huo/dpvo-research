"""Performance-only instrumentation for the frozen H2 implementation.

This module deliberately owns no scientific configuration, model, checkpoint,
evaluation, or publication logic.  It provides small timing/accounting helpers
used by the standalone H2 efficiency benchmark.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from .profiling import distribution_ms
from .protocol import REPO_ROOT


CANONICAL_RESULTS_ROOT = REPO_ROOT / "research/results/phase1-feasibility"
PREDICTOR_STAGES = (
    "endpoint_descriptor_matching",
    "bidirectional_topk_correspondence",
    "cycle_confidence_filtering",
    "global_affine_estimation",
    "coarse_residual_estimation",
    "correspondence_query_assembly",
    "soft_transport_warp",
    "neural_residual_predictor_forward",
    "predictor_output_assembly",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_artifact_snapshot(root: Path = CANONICAL_RESULTS_ROOT) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def validate_output_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    canonical = CANONICAL_RESULTS_ROOT.resolve()
    if resolved == canonical or canonical in resolved.parents:
        raise ValueError("efficiency benchmark output may not be inside canonical results")
    return resolved


@dataclass
class _PendingCuda:
    stage: str
    start: Any
    end: Any


class PerformanceRecorder:
    """Exclusive CPU/CUDA timing with independent same-domain reconciliation."""

    def __init__(self, *, enable_cuda: bool | None = None,
                 negative_tolerance_ms: float = 0.05) -> None:
        self.enable_cuda = bool(torch.cuda.is_available() if enable_cuda is None else enable_cuda)
        self.negative_tolerance_ms = float(negative_tolerance_ms)
        self.cpu_samples: dict[str, list[float]] = defaultdict(list)
        self.cuda_samples: dict[str, list[float]] = defaultdict(list)
        self.outer_cpu_ms: list[float] = []
        self.outer_cuda_ms: list[float] = []
        self.sync_wait_ms: list[float] = []
        self.nested_wait_diagnostics: dict[str, list[float]] = defaultdict(list)
        self._active_stage: str | None = None
        self._outer: dict[str, Any] | None = None
        self._pending: list[_PendingCuda] = []

    @contextlib.contextmanager
    def stage(self, name: str, *, cuda: bool = True) -> Iterator[None]:
        if self._active_stage is not None:
            raise RuntimeError(
                f"exclusive profiler stage nesting is forbidden: {self._active_stage}/{name}"
            )
        self._active_stage = str(name)
        start_event = end_event = None
        nvtx_active = False
        if cuda and self.enable_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(f"h2_efficiency::{name}")
            nvtx_active = True
            start_event.record()
        started = time.perf_counter()
        try:
            yield
        finally:
            cpu_ms = (time.perf_counter() - started) * 1000.0
            if end_event is not None:
                end_event.record()
                self._pending.append(_PendingCuda(str(name), start_event, end_event))
            if nvtx_active:
                torch.cuda.nvtx.range_pop()
            self.cpu_samples[str(name)].append(float(cpu_ms))
            self._active_stage = None

    def begin_outer(self) -> None:
        if self._outer is not None or self._active_stage is not None:
            raise RuntimeError("outer timing scope is already active")
        event = torch.cuda.Event(enable_timing=True) if self.enable_cuda else None
        if event is not None:
            event.record()
        self._outer = {"cpu_start": time.perf_counter(), "cuda_start": event,
                       "pending_start": len(self._pending)}

    def finish_outer(self) -> dict[str, float | None]:
        if self._outer is None or self._active_stage is not None:
            raise RuntimeError("outer timing scope is not active or a stage is unfinished")
        end_event = torch.cuda.Event(enable_timing=True) if self.enable_cuda else None
        if end_event is not None:
            end_event.record()
            sync_started = time.perf_counter()
            end_event.synchronize()
            sync_ms = (time.perf_counter() - sync_started) * 1000.0
            outer_cuda = float(self._outer["cuda_start"].elapsed_time(end_event))
        else:
            sync_ms = 0.0
            outer_cuda = None
        cpu_outer = (time.perf_counter() - float(self._outer["cpu_start"])) * 1000.0
        self.outer_cpu_ms.append(float(cpu_outer))
        self.sync_wait_ms.append(float(sync_ms))
        self.cpu_samples["synchronization_wait"].append(float(sync_ms))
        if outer_cuda is not None:
            self.outer_cuda_ms.append(outer_cuda)
            pending_start = int(self._outer["pending_start"])
            for item in self._pending[pending_start:]:
                self.cuda_samples[item.stage].append(float(item.start.elapsed_time(item.end)))
            del self._pending[pending_start:]
        self._outer = None
        return {"cpu_outer_ms": float(cpu_outer), "cuda_outer_ms": outer_cuda,
                "synchronization_wait_ms": float(sync_ms)}

    def add_cpu(self, name: str, milliseconds: float) -> None:
        value = float(milliseconds)
        if value < 0 or not np.isfinite(value):
            raise ValueError("CPU latency must be finite and non-negative")
        self.cpu_samples[str(name)].append(value)

    def add_nested_wait_diagnostic(self, name: str, milliseconds: float) -> None:
        """Record explanatory wait samples excluded from exclusive reconciliation."""
        value = float(milliseconds)
        if value < 0 or not np.isfinite(value):
            raise ValueError("wait latency must be finite and non-negative")
        self.nested_wait_diagnostics[str(name)].append(value)

    def payload(self) -> dict[str, Any]:
        if self._outer is not None or self._active_stage is not None:
            raise RuntimeError("cannot aggregate an active timing scope")
        cpu_stages = {name: distribution_ms(values)
                      for name, values in sorted(self.cpu_samples.items())}
        cuda_stages = {name: distribution_ms(values)
                       for name, values in sorted(self.cuda_samples.items())}
        outer_cpu_total = float(sum(self.outer_cpu_ms))
        cpu_exclusive_total = float(sum(
            sum(values) for values in self.cpu_samples.values()
        ))
        cpu_residual = outer_cpu_total - cpu_exclusive_total
        outer_cuda_total = float(sum(self.outer_cuda_ms))
        cuda_exclusive_total = float(sum(
            sum(values) for values in self.cuda_samples.values()
        ))
        cuda_residual = outer_cuda_total - cuda_exclusive_total
        if cpu_residual < -self.negative_tolerance_ms:
            raise RuntimeError(f"CPU timing stages overlap: residual={cpu_residual:.6f}ms")
        if cuda_residual < -self.negative_tolerance_ms:
            raise RuntimeError(f"CUDA timing stages overlap: residual={cuda_residual:.6f}ms")
        return {
            "schema": "h2_efficiency_exclusive_timing_v1",
            "domains_must_not_be_combined": True,
            "cpu": {
                "outer": distribution_ms(self.outer_cpu_ms),
                "exclusive_stages": cpu_stages,
                "outer_end_cuda_event_synchronization_wait": distribution_ms(
                    self.sync_wait_ms
                ),
                "nested_wait_diagnostics_excluded_from_reconciliation": {
                    name: distribution_ms(values)
                    for name, values in sorted(self.nested_wait_diagnostics.items())
                },
                "exclusive_total_ms": cpu_exclusive_total,
                "uninstrumented_ms": cpu_residual,
                "reconciliation": "outer_cpu_wall-sum(exclusive_cpu_stages)",
            },
            "cuda": {
                "available": self.enable_cuda,
                "outer": distribution_ms(self.outer_cuda_ms),
                "exclusive_stages": cuda_stages,
                "exclusive_total_ms": cuda_exclusive_total,
                "uninstrumented_ms": cuda_residual,
                "reconciliation": "outer_cuda_event-sum(exclusive_cuda_stages)",
            },
        }


@dataclass
class TransferLedger:
    rows: dict[str, list[dict[str, float | int]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add(self, name: str, *, byte_count: int = 0, cpu_ms: float = 0.0,
            cuda_ms: float | None = None) -> None:
        row: dict[str, float | int] = {
            "bytes": int(byte_count), "cpu_ms": float(cpu_ms),
        }
        if cuda_ms is not None:
            row["cuda_ms"] = float(cuda_ms)
        if row["bytes"] < 0 or row["cpu_ms"] < 0:
            raise ValueError("transfer accounting values must be non-negative")
        self.rows[str(name)].append(row)

    def payload(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, rows in sorted(self.rows.items()):
            cuda = [float(row["cuda_ms"]) for row in rows if "cuda_ms" in row]
            result[name] = {
                "count": len(rows),
                "total_bytes": int(sum(int(row["bytes"]) for row in rows)),
                "cpu_wall": distribution_ms([float(row["cpu_ms"]) for row in rows]),
                "cuda_copy": distribution_ms(cuda),
            }
        return {"schema": "h2_transfer_ipc_audit_v1", "operations": result}


GPU_QUERY_FIELDS = (
    "timestamp", "index", "uuid", "pci.bus_id", "name", "utilization.gpu",
    "utilization.memory", "memory.used", "memory.total", "power.draw", "power.limit",
)


def parse_nvidia_smi_row(line: str) -> dict[str, Any]:
    values = next(csv.reader([line]))
    if len(values) != len(GPU_QUERY_FIELDS):
        raise ValueError(f"unexpected nvidia-smi row with {len(values)} fields")
    row: dict[str, Any] = {key: value.strip() for key, value in zip(GPU_QUERY_FIELDS, values)}
    row["index"] = int(row["index"])
    for key in ("utilization.gpu", "utilization.memory", "memory.used", "memory.total",
                "power.draw", "power.limit"):
        try:
            row[key] = float(row[key])
        except (TypeError, ValueError):
            row[key] = None
    return row


def scalar_distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "total": 0.0, "mean": None, "p50": None,
                "p95": None, "max": None}
    return {"count": int(len(array)), "total": float(array.sum()),
            "mean": float(array.mean()), "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())}


class NvidiaSmiSampler:
    def __init__(self, sample_ms: int = 200) -> None:
        if sample_ms < 100:
            raise ValueError("nvidia-smi sample period must be at least 100ms")
        self.sample_ms = int(sample_ms)
        self.rows: list[dict[str, Any]] = []
        self.error: str | None = None
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            for line in self.process.stdout:
                if line.strip():
                    self.rows.append(parse_nvidia_smi_row(line))
        except Exception as error:  # sampler failure must remain diagnostic-only
            self.error = f"{type(error).__name__}: {error}"

    def start(self) -> None:
        command = [
            "nvidia-smi", "--query-gpu=" + ",".join(GPU_QUERY_FIELDS),
            "--format=csv,noheader,nounits", f"--loop-ms={self.sample_ms}",
        ]
        try:
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            return
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill(); self.process.wait(timeout=5)
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.process.stderr is not None:
            stderr = self.process.stderr.read().strip()
            if stderr and not self.rows:
                self.error = stderr

    def payload(self) -> dict[str, Any]:
        by_uuid: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            by_uuid[str(row["uuid"])].append(row)
        devices = {}
        for uuid, rows in by_uuid.items():
            first = rows[0]
            devices[uuid] = {
                "physical_index": first["index"], "name": first["name"],
                "pci_bus_id": first["pci.bus_id"], "sample_count": len(rows),
                "gpu_utilization_percent": scalar_distribution([
                    row["utilization.gpu"] for row in rows
                    if row["utilization.gpu"] is not None
                ]),
                "memory_utilization_percent": scalar_distribution([
                    row["utilization.memory"] for row in rows
                    if row["utilization.memory"] is not None
                ]),
                "vram_used_mib": scalar_distribution([
                    row["memory.used"] for row in rows if row["memory.used"] is not None
                ]),
                "power_draw_w": scalar_distribution([
                    row["power.draw"] for row in rows if row["power.draw"] is not None
                ]),
                "power_limit_w": first["power.limit"],
            }
        return {"sample_period_ms": self.sample_ms, "error": self.error,
                "devices": devices}


class PersistentPerformanceAudit:
    """Run-integrated performance provenance for canonical Phase 1 commands.

    This recorder owns diagnostic timing and hardware sampling only.  It does
    not inspect evaluation values or participate in any scientific decision.
    CPU phase scopes are exclusive and non-nested; GPU activity is reported by
    the independent nvidia-smi sampler rather than mixed into CPU wall time.
    """

    def __init__(self, module: str, scope: str, *, sample_ms: int = 200,
                 components: Sequence[str] = ("main_process",)) -> None:
        self.module = str(module)
        self.scope = str(scope)
        self.sample_ms = int(sample_ms)
        self.components = tuple(str(value) for value in components)
        self.sampler = NvidiaSmiSampler(sample_ms)
        self.topology: dict[str, Any] | None = None
        self.devices: dict[str, dict[str, Any]] = {}
        self.processes_before: dict[str, Any] | None = None
        self.processes_after: dict[str, Any] | None = None
        self.cpu_phases: dict[str, list[float]] = defaultdict(list)
        self._started: float | None = None
        self._outer_cpu_ms: float | None = None
        self._active_phase: str | None = None
        self._payload: dict[str, Any] | None = None

    def __enter__(self) -> "PersistentPerformanceAudit":
        if self._started is not None:
            raise RuntimeError("persistent performance audit is already active")
        self.topology = gpu_topology_audit()
        logical = {
            row["logical_index"]: row
            for row in self.topology.get("logical_devices", [])
        }
        for component in self.components:
            device = current_cuda_device(component)
            mapping = logical.get(device.get("logical_index"))
            if mapping is not None:
                device.update({
                    key: value for key, value in mapping.items()
                    if key in {"physical_index", "pci_bus_id", "uuid",
                               "mapping_basis", "mapping_error"}
                })
            self.devices[component] = device
        self.processes_before = gpu_process_snapshot()
        self.sampler.start()
        self._started = time.perf_counter()
        return self

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if self._started is None or self._outer_cpu_ms is not None:
            raise RuntimeError("persistent performance audit is not active")
        if self._active_phase is not None:
            raise RuntimeError(
                f"persistent performance phase nesting is forbidden: "
                f"{self._active_phase}/{name}"
            )
        self._active_phase = str(name)
        started = time.perf_counter()
        try:
            yield
        finally:
            self.cpu_phases[str(name)].append(
                (time.perf_counter() - started) * 1000.0
            )
            self._active_phase = None

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> None:
        if self._active_phase is not None:
            raise RuntimeError("persistent performance phase was left active")
        assert self._started is not None
        self._outer_cpu_ms = (time.perf_counter() - self._started) * 1000.0
        self.sampler.stop()
        self.processes_after = gpu_process_snapshot()
        phase_total = float(sum(sum(values) for values in self.cpu_phases.values()))
        residual = self._outer_cpu_ms - phase_total
        if residual < -0.05:
            raise RuntimeError(
                f"persistent CPU phases overlap: residual={residual:.6f}ms"
            )
        self._payload = {
            "schema": "phase1_persistent_performance_diagnostics_v1",
            "module": self.module,
            "scope": self.scope,
            "diagnostic_only": True,
            "excluded_from_scientific_metrics_and_decisions": True,
            "timing_domains_must_not_be_combined": True,
            "cpu_wall": {
                "outer": distribution_ms([self._outer_cpu_ms]),
                "exclusive_phases": {
                    name: distribution_ms(values)
                    for name, values in sorted(self.cpu_phases.items())
                },
                "exclusive_total_ms": phase_total,
                "uninstrumented_ms": residual,
                "reconciliation": "outer_cpu_wall-sum(exclusive_cpu_phases)",
            },
            "cuda_execution": {
                "source": "component-specific CUDA events where available",
                "not_subtracted_from_cpu_wall": True,
            },
            "gpu_utilization": self.sampler.payload(),
            "devices": {
                "components": self.devices,
                "topology": self.topology,
                "processes_before": self.processes_before,
                "processes_after": self.processes_after,
            },
        }

    def payload(self) -> dict[str, Any]:
        if self._payload is None:
            raise RuntimeError("persistent performance audit has not completed")
        return dict(self._payload)


def condition_runtime_diagnostics(result: Mapping[str, Any]) -> dict[str, Any]:
    """Copy runtime-only condition timings without touching evaluation values."""
    rows: dict[str, Any] = {}
    for name, condition in result.get("conditions", {}).items():
        runtime = condition.get("runtime", {})
        rows[str(name)] = {
            key: runtime.get(key) for key in (
                "elapsed_seconds", "processed_observation_count",
                "dpvo_graph_runtime_total_ms",
                "dpvo_graph_runtime_mean_ms_per_processed_observation",
                "peak_gpu_vram_bytes",
            ) if key in runtime
        }
    return rows


def add_cuda_worker_mapping(
    payload: dict[str, Any], worker: Mapping[str, Any], *,
    component: str = "v_jepa_worker",
) -> None:
    """Attach a sidecar's real PID and inherited logical cuda:0 mapping."""
    logical_index = int(worker.get("logical_cuda_ordinal", 0))
    topology = payload.get("devices", {}).get("topology", {})
    mapping = next((
        row for row in topology.get("logical_devices", [])
        if row.get("logical_index") == logical_index
    ), {})
    payload["devices"]["components"][str(component)] = {
        "component": str(component),
        "pid": worker.get("worker_pid"),
        "logical_index": logical_index,
        "cuda_visible_devices": worker.get("cuda_visible_devices"),
        "name": worker.get("cuda_device_name", mapping.get("name")),
        "mapping_basis": "worker hardcodes cuda:0 and inherits visibility",
        **{key: mapping.get(key) for key in (
            "physical_index", "uuid", "pci_bus_id",
        )},
    }


def performance_diagnosis(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Produce timing-only rankings without coupling them to scientific gates."""
    phases = payload.get("cpu_wall", {}).get("exclusive_phases", {})
    conditions = payload.get("condition_runtime", {})
    result: dict[str, Any] = {
        "cpu_phase_wall_ranking": sorted(
            ((name, row.get("total_ms")) for name, row in phases.items()),
            key=lambda item: float(item[1] or 0.0), reverse=True,
        ),
        "condition_wall_ranking": sorted(
            ((name, row.get("elapsed_seconds")) for name, row in conditions.items()),
            key=lambda item: float(item[1] or 0.0), reverse=True,
        ),
        "cpu_cuda_values_are_not_combined": True,
    }
    strict = payload.get("strict_h2_predictor")
    if isinstance(strict, Mapping):
        for domain in ("cpu", "cuda"):
            stages = strict.get(domain, {}).get("exclusive_stages", {})
            result[f"strict_h2_predictor_{domain}_ranking"] = sorted(
                ((name, row.get("total_ms")) for name, row in stages.items()),
                key=lambda item: float(item[1] or 0.0), reverse=True,
            )
        cpu_ranking = result.get("strict_h2_predictor_cpu_ranking", [])
        result["next_optimization_candidates"] = [
            {
                "candidate": f"optimize_{name}",
                "evidence_cpu_wall_ms": total,
                "implemented_by_this_run": False,
            }
            for name, total in cpu_ranking
            if name != "synchronization_wait"
        ][:3]
    throughput = payload.get("training_throughput")
    if isinstance(throughput, Mapping):
        training_ranking: dict[str, Any] = {}
        for scope, profile in throughput.items():
            if not isinstance(profile, Mapping):
                continue
            for domain in ("cpu", "cuda"):
                stages = profile.get(domain, {}).get("exclusive_stages", {})
                training_ranking[f"{scope}_{domain}"] = sorted(
                    ((name, row.get("total_ms")) for name, row in stages.items()),
                    key=lambda item: float(item[1] or 0.0), reverse=True,
                )
        result["training_stage_ranking_by_timing_domain"] = training_ranking
    return result


def _run_text(command: Sequence[str]) -> tuple[str | None, str | None]:
    try:
        value = subprocess.run(command, check=True, capture_output=True, text=True)
        return value.stdout.strip(), None
    except Exception as error:
        detail = getattr(error, "stderr", None) or str(error)
        return None, str(detail).strip()


def parse_nvidia_topology(value: str | None) -> list[dict[str, Any]]:
    if not value:
        return []
    # ``nvidia-smi`` may underline the header with ANSI escapes when invoked
    # from a terminal.  Selecting the first line that starts with GPU0 is not
    # sufficient because that can be the first data row.  The real header is
    # the line containing the greatest number of GPU column labels.
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    lines = [ansi.sub("", line).rstrip() for line in value.splitlines() if line.strip()]
    header_line = max(
        lines,
        key=lambda line: sum(bool(re.fullmatch(r"GPU\d+", token))
                             for token in line.split()),
        default="",
    )
    header = header_line.split()
    gpu_columns = [item for item in header if re.fullmatch(r"GPU\d+", item)]
    if len(gpu_columns) < 2:
        return []
    pairs = []
    for line in lines:
        fields = line.split()
        if not fields or not fields[0].startswith("GPU") or line == header_line:
            continue
        source = fields[0]
        if len(fields) < len(gpu_columns) + 1:
            continue
        for offset, destination in enumerate(gpu_columns, start=1):
            if source != destination:
                pairs.append({"source": source, "destination": destination,
                              "link": fields[offset]})
    return pairs


def logical_physical_mapping(
    logical_count: int, physical_devices: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve CUDA ordinals to nvidia-smi/NVML identities.

    Older PyTorch device properties do not expose UUID or PCI bus ID.  PyTorch's
    NVML-index resolver still accounts for CUDA_VISIBLE_DEVICES (including UUID
    selectors), so use it first and retain an explicit fallback/basis in the
    audit instead of silently reporting an assumed direct mapping.
    """
    physical_by_index = {
        int(row["physical_index"]): row for row in physical_devices
        if row.get("physical_index") is not None
    }
    result: list[dict[str, Any]] = []
    for logical in range(int(logical_count)):
        physical_index: int | None = None
        mapping_basis: str
        mapping_error: str | None = None
        try:
            physical_index = int(torch.cuda._get_nvml_device_index(logical))
            mapping_basis = "torch.cuda._get_nvml_device_index"
        except Exception as error:
            mapping_error = f"{type(error).__name__}: {error}"
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible is None:
                physical_index = logical
                mapping_basis = "unmasked_cuda_ordinal_fallback"
            else:
                selectors = [item.strip() for item in visible.split(",")]
                selector = selectors[logical] if logical < len(selectors) else ""
                if selector.isdigit():
                    physical_index = int(selector)
                    mapping_basis = "CUDA_VISIBLE_DEVICES_numeric_fallback"
                else:
                    match = next((
                        row for row in physical_devices
                        if str(row.get("uuid", "")).startswith(selector)
                    ), None)
                    physical_index = (
                        int(match["physical_index"]) if match is not None else None
                    )
                    mapping_basis = "CUDA_VISIBLE_DEVICES_uuid_fallback"
        physical = physical_by_index.get(physical_index) if physical_index is not None else None
        result.append({
            "logical_index": logical,
            "physical_index": physical_index,
            "uuid": physical.get("uuid") if physical else None,
            "pci_bus_id": physical.get("pci_bus_id") if physical else None,
            "physical_name": physical.get("name") if physical else None,
            "mapping_basis": mapping_basis,
            "mapping_error": mapping_error,
        })
    return result


def gpu_topology_audit() -> dict[str, Any]:
    topology, topology_error = _run_text(("nvidia-smi", "topo", "-m"))
    inventory, inventory_error = _run_text((
        "nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
        "--format=csv,noheader,nounits",
    ))
    peer: list[dict[str, Any]] = []
    physical_devices: list[dict[str, Any]] = []
    if inventory:
        for line in inventory.splitlines():
            fields = [item.strip() for item in next(csv.reader([line]))]
            if len(fields) == 5:
                try:
                    physical_devices.append({
                        "physical_index": int(fields[0]), "uuid": fields[1],
                        "pci_bus_id": fields[2], "name": fields[3],
                        "memory_total_mib": int(fields[4]),
                    })
                except ValueError:
                    pass
    logical_devices: list[dict[str, Any]] = []
    cuda_error = None
    try:
        count = torch.cuda.device_count()
        resolved = {
            row["logical_index"]: row
            for row in logical_physical_mapping(count, physical_devices)
        }
        for source in range(count):
            properties = torch.cuda.get_device_properties(source)
            mapping = resolved[source]
            logical_devices.append({
                "logical_index": source, "name": properties.name,
                "physical_index": mapping["physical_index"],
                "uuid": mapping["uuid"], "pci_bus_id": mapping["pci_bus_id"],
                "mapping_basis": mapping["mapping_basis"],
                "mapping_error": mapping["mapping_error"],
            })
            for destination in range(count):
                if source == destination:
                    continue
                try:
                    allowed = bool(torch.cuda.can_device_access_peer(source, destination))
                    peer.append({"source": source, "destination": destination,
                                 "can_access_peer": allowed, "error": None})
                except Exception as error:
                    peer.append({"source": source, "destination": destination,
                                 "can_access_peer": False,
                                 "error": f"{type(error).__name__}: {error}"})
    except Exception as error:
        cuda_error = f"{type(error).__name__}: {error}"
    pair_summary = []
    for left, right in ((0, 1), (1, 2), (0, 2)):
        forward = next((row for row in peer
                        if row["source"] == left and row["destination"] == right), None)
        reverse = next((row for row in peer
                        if row["source"] == right and row["destination"] == left), None)
        pair_summary.append({"pair": [left, right], "forward": forward,
                             "reverse": reverse,
                             "bidirectional": bool(
                                 forward and reverse and forward["can_access_peer"]
                                 and reverse["can_access_peer"]
                             )})
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_count": int(torch.cuda.device_count()),
        "nvidia_smi_inventory": inventory,
        "physical_devices": physical_devices,
        "logical_devices": logical_devices,
        "nvidia_smi_inventory_error": inventory_error,
        "nvidia_smi_topology_raw": topology,
        "nvidia_smi_topology_pairs": parse_nvidia_topology(topology),
        "nvidia_smi_topology_error": topology_error,
        "peer_access": peer, "peer_pair_summary": pair_summary,
        "cuda_error": cuda_error,
    }


def current_cuda_device(component: str, *, pid: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "component": str(component), "pid": int(os.getpid() if pid is None else pid),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if not torch.cuda.is_available():
        return result | {"available": False, "error": "torch.cuda.is_available() is false"}
    logical = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(logical)
    return result | {
        "available": True, "logical_index": logical, "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "uuid": str(getattr(properties, "uuid", "unknown")),
    }


def gpu_process_snapshot() -> dict[str, Any]:
    value, error = _run_text((
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ))
    rows = []
    if value:
        for line in value.splitlines():
            fields = [item.strip() for item in next(csv.reader([line]))]
            if len(fields) == 4:
                try:
                    memory: int | None = int(fields[3])
                except ValueError:
                    memory = None
                rows.append({"pid": int(fields[0]), "gpu_uuid": fields[1],
                             "process_name": fields[2], "used_gpu_memory_mib": memory})
    return {"rows": rows, "error": error}


def pairwise_copy_benchmark(
    tensor_specs: Mapping[str, tuple[Sequence[int], torch.dtype]], *, repeats: int = 10,
) -> dict[str, Any]:
    """Measure representative peer or pinned-host-staged transfers only."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return {"available": False, "reason": "at least two visible CUDA devices are required",
                "pairs": []}
    pairs = []
    for source in range(torch.cuda.device_count()):
        for destination in range(torch.cuda.device_count()):
            if source == destination:
                continue
            peer = bool(torch.cuda.can_device_access_peer(source, destination))
            measurements = {}
            for name, (shape, dtype) in tensor_specs.items():
                with torch.cuda.device(source):
                    source_tensor = torch.zeros(tuple(shape), device=f"cuda:{source}", dtype=dtype)
                bytes_count = source_tensor.numel() * source_tensor.element_size()
                if peer:
                    with torch.cuda.device(destination):
                        destination_tensor = torch.empty(
                            tuple(shape), device=f"cuda:{destination}", dtype=dtype,
                        )
                        destination_tensor.copy_(source_tensor)
                        torch.cuda.synchronize(destination)
                        samples = []
                        for _ in range(int(repeats)):
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record(); destination_tensor.copy_(source_tensor); end.record()
                            end.synchronize(); samples.append(float(start.elapsed_time(end)))
                    measurements[name] = {
                        "path": "direct_device_copy", "bytes": int(bytes_count),
                        "cuda_copy_ms": distribution_ms(samples),
                        "bandwidth_gbps_from_mean": (
                            bytes_count / (float(np.mean(samples)) / 1000.0) / 1e9
                            if samples and float(np.mean(samples)) > 0 else None
                        ),
                    }
                    del destination_tensor
                else:
                    host = torch.empty(tuple(shape), dtype=dtype, pin_memory=True)
                    with torch.cuda.device(destination):
                        destination_tensor = torch.empty(
                            tuple(shape), device=f"cuda:{destination}", dtype=dtype,
                        )
                    d2h, h2d, wall = [], [], []
                    for _ in range(int(repeats)):
                        wall_started = time.perf_counter()
                        with torch.cuda.device(source):
                            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                            start.record(); host.copy_(source_tensor, non_blocking=True); end.record()
                            end.synchronize(); d2h.append(float(start.elapsed_time(end)))
                        with torch.cuda.device(destination):
                            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                            start.record(); destination_tensor.copy_(host, non_blocking=True); end.record()
                            end.synchronize(); h2d.append(float(start.elapsed_time(end)))
                        wall.append((time.perf_counter() - wall_started) * 1000.0)
                    measurements[name] = {
                        "path": "pinned_host_staging", "bytes": int(bytes_count),
                        "d2h_cuda_ms": distribution_ms(d2h),
                        "h2d_cuda_ms": distribution_ms(h2d),
                        "cpu_wall_ms": distribution_ms(wall),
                    }
                    del host, destination_tensor
                del source_tensor
            pairs.append({"source": source, "destination": destination,
                          "can_access_peer": peer, "measurements": measurements})
    return {"available": True, "repeats": int(repeats), "pairs": pairs}


def simulate_pipeline(
    intervals: Sequence[Mapping[str, float]], *, sensor_paced: bool,
    startup: Mapping[str, float] | None = None, drain_ms: float = 0.0,
) -> dict[str, Any]:
    """Simulate the frozen dependency DAG on three independent resources."""
    startup = dict(startup or {})
    gpu0 = float(startup.get("gpu0_ms", 0.0))
    gpu1 = float(startup.get("gpu1_ms", 0.0))
    gpu2 = float(startup.get("gpu2_ms", 0.0))
    events = []
    resource = {"gpu0_encode_ms": gpu0, "gpu1_predict_bridge_transfer_ms": gpu1,
                "gpu2_dpvo_ms": gpu2 + float(drain_ms)}
    for index, row in enumerate(intervals):
        arrival = float(row.get("anchor_available_ms", 0.0)) if sensor_paced else 0.0
        encode = float(row["encode_ms"])
        predict = float(row["predict_ms"]) + float(row.get("bridge_ms", 0.0)) \
            + float(row.get("transfer_ms", 0.0))
        dpvo = float(row["dpvo_ms"])
        encode_start = max(arrival, gpu0); encode_ready = encode_start + encode
        predict_start = max(encode_ready, gpu1); predict_ready = predict_start + predict
        dpvo_start = max(predict_ready, gpu2); dpvo_ready = dpvo_start + dpvo
        gpu0, gpu1, gpu2 = encode_ready, predict_ready, dpvo_ready
        resource["gpu0_encode_ms"] += encode
        resource["gpu1_predict_bridge_transfer_ms"] += predict
        resource["gpu2_dpvo_ms"] += dpvo
        events.append({"interval": index, "anchor_available_ms": arrival,
                       "encode_ready_ms": encode_ready, "predict_ready_ms": predict_ready,
                       "dpvo_ready_ms": dpvo_ready})
    makespan = float((gpu2 if intervals else max(gpu0, gpu1, gpu2)) + float(drain_ms))
    resource_bound = float(max(resource.values(), default=0.0))
    return {
        "sensor_paced": bool(sensor_paced), "interval_count": len(intervals),
        "startup_ms": startup, "drain_ms": float(drain_ms),
        "makespan_ms": makespan, "dependency_critical_path_lower_bound_ms": makespan,
        "per_device_resource_totals_ms": resource,
        "per_device_resource_lower_bound_ms": resource_bound,
        "bound_scope": (
            "current measured kernels and proposed one-GPU-per-stage mapping; "
            "not an RTX4090 hardware or H2 algorithmic lower bound"
        ),
        "events": events,
        "excludes": ["rgb_encoding", "network_queue", "uplink_transmission",
                     "packet_loss", "network_jitter"],
    }


def bootstrap_pipeline(
    intervals: Sequence[Mapping[str, float]], *, sensor_paced: bool,
    startup: Mapping[str, float] | None = None, drain_ms: float = 0.0,
    repetitions: int = 1000, seed: int = 1234,
) -> dict[str, Any]:
    if not intervals:
        raise ValueError("pipeline bootstrap requires interval samples")
    rng = np.random.default_rng(int(seed)); values = []
    availability = [float(row.get("anchor_available_ms", 0.0)) for row in intervals]
    for _ in range(int(repetitions)):
        indices = rng.integers(0, len(intervals), size=len(intervals))
        sampled = []
        for position, index in enumerate(indices):
            row = dict(intervals[int(index)])
            if sensor_paced:
                row["anchor_available_ms"] = availability[position]
            sampled.append(row)
        values.append(simulate_pipeline(
            sampled, sensor_paced=sensor_paced, startup=startup, drain_ms=drain_ms,
        )["makespan_ms"])
    result = scalar_distribution(values)
    return {"repetitions": int(repetitions), "seed": int(seed),
            "makespan_ms": result}


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
