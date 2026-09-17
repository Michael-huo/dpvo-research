"""Execution settings for functional experiment runners.

These settings describe how a scientific contract is executed.  They are
recorded in provenance and never participate in checkpoint compatibility.
"""
from __future__ import annotations

import dataclasses
import gc
import hashlib
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from latent_vslam.protocol import canonical_sha256


from latent_vslam.cuda_devices import CudaDevicePool, logical_device_telemetry


def _parse_cpu_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.strip().split(","):
        if not part:
            continue
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            result.extend(range(first, last + 1))
        else:
            result.append(int(part))
    return result


def _physical_cpus_for_node(node: int) -> list[int]:
    cpulist = Path(f"/sys/devices/system/node/node{node}/cpulist")
    if not cpulist.is_file():
        raise RuntimeError(f"CPU list is unavailable for NUMA node {node}")
    allowed = set(os.sched_getaffinity(0))
    selected: dict[tuple[int, int], int] = {}
    for cpu in _parse_cpu_list(cpulist.read_text()):
        if cpu not in allowed:
            continue
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = int((topology / "physical_package_id").read_text())
        core = int((topology / "core_id").read_text())
        selected.setdefault((package, core), cpu)
    if not selected:
        raise RuntimeError(f"no allowed physical CPU cores for NUMA node {node}")
    return sorted(selected.values())


def _allowed_physical_cpus() -> list[int]:
    allowed = sorted(os.sched_getaffinity(0))
    selected: dict[tuple[int, int], int] = {}
    try:
        for cpu in allowed:
            topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            package = int((topology / "physical_package_id").read_text())
            core = int((topology / "core_id").read_text())
            selected.setdefault((package, core), cpu)
    except (OSError, ValueError):
        return allowed
    return sorted(selected.values()) or allowed


def cpu_numa_layout(hardware: dict[str, Any]) -> dict[str, Any]:
    devices = hardware.get("devices", [])
    basis = "gpu_sysfs_numa"
    try:
        by_index = {int(row["logical_index"]): row for row in devices}
        stage_node = int(by_index[0]["numa_node"])
        remote_node = int(by_index[1]["numa_node"])
        if stage_node == remote_node or int(by_index[2]["numa_node"]) != remote_node:
            raise ValueError("preferred GPU NUMA relation is unavailable")
        stage = _physical_cpus_for_node(stage_node)
        remote = _physical_cpus_for_node(remote_node)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        basis = "allowed_cpuset_fallback"
        cpus = _allowed_physical_cpus()
        split = max(1, len(cpus) // 2)
        stage, remote = cpus[:split], cpus[split:]
        if len(remote) < 2:
            stage = remote = cpus
        stage_node = remote_node = None
    # Keep the remote workers disjoint.  V-JEPA receives an odd remainder.
    predictor_count = len(remote) // 2
    predictor = remote[:predictor_count]
    encoder = remote[predictor_count:]
    if not predictor or not encoder:
        predictor = encoder = remote
    payload = {
        "schema": "research_cpu_numa_layout_v1",
        "stage_c": {"device": 0, "numa_node": stage_node, "cpus": stage},
        "predictor": {"device": 1, "numa_node": remote_node, "cpus": predictor},
        "encoder": {"device": 2, "numa_node": remote_node, "cpus": encoder},
        "affinity_basis": basis,
        "smt_policy": "one_logical_cpu_per_physical_core",
        "sets_disjoint": not bool(set(stage) & set(predictor + encoder))
        and not bool(set(predictor) & set(encoder)),
    }
    return payload | {"layout_sha256": canonical_sha256(payload)}


def fixed_cpu_profile(layout: dict[str, Any]) -> dict[str, Any]:
    """Keep standard DPVO CPU resources and isolate the two prediction workers."""
    components = {}
    for name in ("stage_c", "predictor", "encoder"):
        cpus = list(layout[name]["cpus"])
        threads = len(cpus) if name == "stage_c" else 1
        components[name] = {
            **layout[name], "intraop_threads": threads,
            "interop_threads": 1, "omp_num_threads": threads,
            "mkl_num_threads": threads,
        }
    return components


def apply_cpu_profile(component: dict[str, Any]) -> None:
    cpus = {int(value) for value in component["cpus"]}
    os.sched_setaffinity(0, cpus)
    os.environ["OMP_NUM_THREADS"] = str(component["omp_num_threads"])
    os.environ["MKL_NUM_THREADS"] = str(component["mkl_num_threads"])
    torch.set_num_threads(int(component["intraop_threads"]))
    # PyTorch only permits setting this once and before inter-op work begins.
    try:
        torch.set_num_interop_threads(int(component["interop_threads"]))
    except RuntimeError:
        if torch.get_num_interop_threads() != int(component["interop_threads"]):
            raise
    if set(os.sched_getaffinity(0)) != cpus:
        raise RuntimeError("CPU affinity application failed")


def validate_formal_hardware() -> dict[str, Any]:
    """Collect best-effort hardware telemetry without making it an admission gate."""
    inventory_error = None
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,memory.total,name",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        output = ""
        inventory_error = f"{type(error).__name__}: {error}"
    topology = []
    for line in output.splitlines():
        try:
            fields = [value.strip() for value in line.split(",", 4)]
            if len(fields) != 5:
                raise ValueError(f"invalid inventory row: {line!r}")
            topology.append({
                "physical_index": int(fields[0]), "uuid": fields[1],
                "pci_bus_id": fields[2], "memory_total_mib": int(fields[3]),
                "name": fields[4],
            })
        except ValueError as error:
            inventory_error = f"{type(error).__name__}: {error}"
    for row in topology:
        try:
            bus = row["pci_bus_id"].lower()
            domain, suffix = bus.split(":", 1)
            sysfs_bus = f"{int(domain, 16) & 0xffff:04x}:{suffix}"
            row["numa_node"] = int(
                (Path("/sys/bus/pci/devices") / sysfs_bus / "numa_node").read_text()
            )
            row["numa_error"] = None
        except (OSError, ValueError) as error:
            row["numa_node"] = None
            row["numa_error"] = f"{type(error).__name__}: {error}"
    pool = CudaDevicePool.discover()
    devices = logical_device_telemetry(pool.device_count, topology)
    payload = {
        "schema": "research_formal_hardware_v2",
        "devices": devices, "physical_inventory": topology,
        "nvidia_smi_inventory_error": inventory_error, "telemetry_only": True,
        **pool.provenance(),
        "mapping": {"preparation": list(pool.logical_ids),
                    "standalone_trajectory": pool.primary_device if pool.device_count else None,
                    "predicted_jepa": pool.online_mapping() if pool.device_count >= 3 else None},
    }
    return payload | {"hardware_sha256": canonical_sha256(payload)}


def initialize_formal_main_process(*, training_device: int | None = 0,
                                   require_online: bool = False) -> dict[str, Any]:
    pool = CudaDevicePool.discover()
    if require_online:
        pool.online_mapping()
    pool.require(0)
    hardware = validate_formal_hardware()
    settings = capture_runtime()
    if training_device is not None:
        pool.require(training_device)
        if training_device != 0:
            raise ValueError("formal training belongs to the primary logical device")
        torch.cuda.set_device(training_device)
        apply_runtime(settings)
    return {"hardware": hardware, "device_pool": pool.provenance(),
            "main_process_device": training_device,
            "runtime_settings": settings}


@dataclasses.dataclass(frozen=True)
class FormalExecution:
    encoder_device: str = "2"
    predictor_device: str = "1"
    consumer_device: str = "0"
    queue_capacity: int = 2
    timeout_seconds: float = 120.0
    sensor_paced: bool = False
    verify_transfers: bool = False
    decision_trace: bool = False
    # Formal execution keeps pre-online A/B acceptance explicitly disabled.
    preonline_acceptance: bool = False

    @classmethod
    def from_pool(cls, pool=None):
        mapping = (pool or CudaDevicePool.discover()).online_mapping()
        return cls(encoder_device=mapping["encoder"].split(":")[1],
                   predictor_device=mapping["predictor"].split(":")[1],
                   consumer_device=mapping["stage_c"].split(":")[1])

    def __post_init__(self):
        if self.preonline_acceptance:
            raise ValueError("formal execution does not support pre-online A/B acceptance")
        if self.queue_capacity < 1 or self.timeout_seconds <= 0:
            raise ValueError("queue capacity and timeout must be positive")
        if len({self.encoder_device, self.predictor_device, self.consumer_device}) != 3:
            raise ValueError("formal execution needs three distinct logical CUDA devices")
        if (self.encoder_device, self.predictor_device, self.consumer_device) != ("2", "1", "0"):
            raise ValueError("inference logical mapping is encoder=2, predictor=1, Stage C=0")


def capture_runtime(seed: int = 1234) -> dict[str, Any]:
    return {
        "seed": int(seed), "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "autocast_dtype": str(torch.get_autocast_gpu_dtype()).removeprefix("torch."),
        "identity_rng": "frontend_seed(FrameIdentity,experiment_seed)",
    }


def apply_runtime(settings: dict[str, Any]) -> None:
    """Call before model construction; do not reset RNG at task boundaries."""
    workspace = settings["cublas_workspace_config"]
    if workspace is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    seed = int(settings["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision(settings["matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = settings["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = settings["cudnn_allow_tf32"]
    torch.backends.cudnn.benchmark = settings["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = settings["cudnn_deterministic"]
    torch.use_deterministic_algorithms(settings["deterministic_algorithms"],
                                     warn_only=settings["deterministic_warn_only"])
    torch.set_autocast_gpu_dtype(getattr(torch, settings["autocast_dtype"]))


def runtime_provenance(settings: dict[str, Any], *, component: str,
                       model: Any = None, amp: bool = False,
                       autocast_dtype: str | None = None) -> dict[str, Any]:
    actual = capture_runtime(settings["seed"])
    if actual != settings:
        raise RuntimeError(f"worker runtime settings differ: {actual} != {settings}")
    cuda_available = bool(torch.cuda.is_available())
    visible_count = int(torch.cuda.device_count()) if cuda_available else 0
    current_device = int(torch.cuda.current_device()) if visible_count else None
    return {"component": component, "settings": actual, "amp": bool(amp),
            "component_autocast_dtype": autocast_dtype or actual["autocast_dtype"],
            "model_training": None if model is None else model.training,
            "model_requires_grad": None if model is None else any(p.requires_grad for p in model.parameters()),
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "python_version": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_cuda_device_count": visible_count,
            "current_logical_cuda_device": current_device,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads()}


def release_cuda_training_state(*objects: Any) -> dict[str, Any]:
    """Release resources owned by the completed stage; memory is diagnostic only."""
    closed = 0
    lifecycle_errors = []
    for value in objects:
        close = getattr(value, "close", None)
        if callable(close):
            try:
                close()
                closed += 1
            except Exception as error:
                lifecycle_errors.append(
                    f"owner close: {type(error).__name__}: {error}"
                )
    initialized = bool(torch.cuda.is_initialized())
    active = None
    if initialized:
        try:
            active = int(torch.cuda.current_device())
        except Exception as error:
            lifecycle_errors.append(
                f"current CUDA device: {type(error).__name__}: {error}"
            )
    if initialized:
        try:
            if active is None:
                torch.cuda.synchronize()
            else:
                torch.cuda.synchronize(active)
        except Exception as error:
            lifecycle_errors.append(
                f"CUDA synchronize: {type(error).__name__}: {error}"
            )
    try:
        gc.collect()
    except Exception as error:
        lifecycle_errors.append(f"gc: {type(error).__name__}: {error}")
    if initialized:
        try:
            torch.cuda.empty_cache()
        except Exception as error:
            lifecycle_errors.append(
                f"empty_cache: {type(error).__name__}: {error}"
            )
    devices = []
    telemetry_error = None
    if initialized and active is not None:
        try:
            devices.append({
                "logical_device": active,
                "allocated_bytes": int(torch.cuda.memory_allocated(active)),
                "reserved_bytes": int(torch.cuda.memory_reserved(active)),
            })
        except Exception as error:
            telemetry_error = f"{type(error).__name__}: {error}"
    return {
        "cleanup_complete": not lifecycle_errors,
        "lifecycle_errors": lifecycle_errors,
        "closed_owner_count": closed,
        "cuda_synchronized": initialized and not any(
            value.startswith("CUDA synchronize:") for value in lifecycle_errors
        ),
        "gc_collected": not any(
            value.startswith("gc:") for value in lifecycle_errors
        ),
        "empty_cache_called": initialized and not any(
            value.startswith("empty_cache:") for value in lifecycle_errors
        ),
        "devices": devices,
        "memory_telemetry_only": True, "telemetry_error": telemetry_error,
        "pid": os.getpid(),
    }


def require_lifecycle_cleanup(cleanup: dict[str, Any]) -> None:
    if cleanup.get("cleanup_complete") is not True:
        raise RuntimeError(f"stage resource lifecycle cleanup failed: {cleanup}")


def execution_provenance(options: FormalExecution | None = None) -> dict[str, Any]:
    names = (
        "execution_runtime.py", "cuda_devices.py", "artifact_runtime.py", "training_runtime.py", "parallel_runtime.py",
        "prediction_pipeline.py", "pipeline_worker.py", "staged_transfer.py",
        "jepa_runtime.py", "jepa_worker.py", "runtime.py",
        "efficiency_profiling.py", "bridge_training_runtime.py", "inference_runtime.py",
        "train.py", "infer.py", "stride_sweep.py", "stride_training.py",
        "inference_results.py", "scientific_lineage.py", "manifests.py",
        "predictor.py", "transport.py", "bridge_training.py", "predictor_training.py",
        "jepa_fmap.py", "prediction_deployment.py",
        "predictor_checkpoint.py", "uniform_admission.py", "observation_sampling.py",
    )
    source_root = Path(__file__).resolve().parent.parent
    prediction_sources = {
        "jepa_runtime.py", "jepa_worker.py", "predictor.py", "transport.py",
        "predictor_checkpoint.py",
    }
    files = {
        name: hashlib.sha256((source_root / (
            "prediction" if name in prediction_sources else "latent_vslam"
        ) / name).read_bytes()).hexdigest()
        for name in names
    }
    files.update({
        f"vjepa2/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((source_root / "prediction" / "vjepa2").glob("*.py"))
    })
    payload = {"version": "research_formal_execution_v2_single_dpvo_numa", "source_manifest": files,
               "options": dataclasses.asdict(options) if options is not None else None,
               "excluded_from_checkpoint_compatibility": True}
    return payload | {"execution_sha256": canonical_sha256(payload)}
