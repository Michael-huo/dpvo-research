"""Shared runtime for the three canonical Exp6 modules."""
from __future__ import annotations

import contextlib
import gc
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .dpvo_backend import _load_frame, _make_slam, _runtime_classes, _seed_everything
from .oracle_packet import (
    FrontendPacketStore,
    ZERO_PACKET_SCHEMA,
    extract_frontend_packet,
    frontend_rng_scope,
    frontend_seed,
    packet_to_native,
    representation_contract,
)
from .protocol import (
    FrameIdentity,
    canonical_sha256,
    repo_path,
)


FORMAL_MODES = (
    "matched_full_rgb", "sparse_rgb", "fmap_zero_context",
)
PACKET_MODES = frozenset({"fmap_zero_context"})


@dataclass(frozen=True)
class OnlineFrame:
    identity: FrameIdentity
    rgb_path: str | None


@dataclass(frozen=True)
class PacketObservation:
    """A timestamp-ordered FMap-only observation with no RGB capability."""

    identity: FrameIdentity
    packet: Any
    kind: str
    availability_timestamp_ns: int

    def __post_init__(self) -> None:
        if self.kind not in {"anchor", "hidden"}:
            raise ValueError(self.kind)
        if self.availability_timestamp_ns < self.identity.timestamp_ns:
            raise ValueError("observation cannot be available before its sensor timestamp")


@dataclass(frozen=True)
class AnchorRGBObservation:
    """An uploaded anchor decoded for the native DPVO frontend only."""

    identity: FrameIdentity
    image: torch.Tensor
    intrinsics: torch.Tensor
    availability_timestamp_ns: int

    def __post_init__(self) -> None:
        if self.availability_timestamp_ns < self.identity.timestamp_ns:
            raise ValueError("anchor cannot be available before its sensor timestamp")
        if self.image.ndim != 3 or self.image.shape[0] != 3:
            raise ValueError("anchor image must have CHW RGB/BGR tensor layout")
        if self.intrinsics.numel() != 4:
            raise ValueError("anchor intrinsics must contain fx/fy/cx/cy")


def sanitize_full_oracle_frames(
    records: Sequence[Any], roles: Mapping[str, str],
) -> list[OnlineFrame]:
    result = [
        OnlineFrame(
            record.identity,
            record.rgb_path if roles[record.identity.key] == "anchor" else None,
        )
        for record in records
    ]
    if any(frame.rgb_path is not None for frame in result if roles[frame.identity.key] == "hidden"):
        raise AssertionError("hidden RGB path survived online boundary sanitization")
    return result




@contextlib.contextmanager
def _deterministic_patchifier(slam: Any, identity_box: dict[str, FrameIdentity], seed: int,
                              frontend_samples_ms: list[float] | None = None):
    original = slam.network.patchify.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        identity = identity_box.get("identity")
        if identity is None:
            raise AssertionError("Patchifier called without FrameIdentity")
        with frontend_rng_scope(frontend_seed(identity, seed)):
            if frontend_samples_ms is None:
                return original(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(); result = original(*args, **kwargs); end.record(); end.synchronize()
            frontend_samples_ms.append(float(start.elapsed_time(end)))
            return result

    slam.network.patchify.forward = forward
    try:
        yield
    finally:
        slam.network.patchify.forward = original


def _runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    calibration = config["dataset"].get("calibration", config["paths"].get("calibration"))
    if calibration is None:
        raise KeyError("Exp6 config has no calibration path")
    return {
        "experiment": {"seed": int(config["experiment"]["seed"])},
        "paths": {
            "checkpoint": str(repo_path(config["paths"]["dpvo_checkpoint"])),
            "dpvo_config": str(repo_path(config["paths"]["dpvo_config"])),
            "calibration": str(repo_path(calibration)),
        },
    }


def _formal_classes() -> tuple[type[Any], type[Any]]:
    classes = _runtime_classes()

    class FactorAccountingMixin:
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.exp6_hidden_timestamps: set[int] = set()
            self.exp6_factor_count = 0
            self.exp6_hidden_source_factor_count = 0
            self.exp6_hidden_target_factor_count = 0

        def append_factors(self, kk: torch.Tensor, jj: torch.Tensor) -> None:
            if kk.numel():
                source_nodes = self.ix[kk.long()].detach().cpu().numpy()
                target_nodes = jj.long().detach().cpu().numpy()
                source_counters = self.pg.tstamps_[source_nodes]
                target_counters = self.pg.tstamps_[target_nodes]
                self.exp6_factor_count += int(kk.numel())
                self.exp6_hidden_source_factor_count += sum(
                    int(self.tlist[int(value)]) in self.exp6_hidden_timestamps
                    for value in source_counters.tolist()
                )
                self.exp6_hidden_target_factor_count += sum(
                    int(self.tlist[int(value)]) in self.exp6_hidden_timestamps
                    for value in target_counters.tolist()
                )
            super().append_factors(kk, jj)

    class FormalRGBDPVO(FactorAccountingMixin, classes["DPVO"]):
        pass

    class FormalPacketDPVO(FactorAccountingMixin, classes["NativePacketDPVO"]):
        pass

    return FormalRGBDPVO, FormalPacketDPVO


@torch.no_grad()
def warmup_dpvo_frontend(record: Any, calibration: np.ndarray,
                         config: Mapping[str, Any], *, packet_runtime: bool) -> dict[str, Any]:
    """Warm only the independent frontend on a disposable DPVO instance."""
    image, _ = _load_frame({"image_path": record.rgb_path}, calibration)
    rgb_class, packet_class = _formal_classes()
    cls = packet_class if packet_runtime else rgb_class
    seed = int(config["experiment"]["seed"])
    _seed_everything(seed)
    slam = _make_slam(cls, _runtime_config(config), image)
    extract_frontend_packet(slam, image, record.identity, seed)
    torch.cuda.synchronize()
    del slam
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "rule": "one_representative_forward_per_independent_online_component",
        "instance": "throwaway_dpvo",
        "warmed_components": ["native_dpvo_frontend"],
        "stateful_dpvo_graph_warmed": False,
        "timed_dpvo_starts_from_fresh_state": True,
    }


def _state_is_finite(slam: Any) -> bool:
    return bool(
        torch.isfinite(slam.pg.poses_[: int(slam.n)]).all().item()
        and torch.isfinite(slam.pg.patches_[: int(slam.n), :, 2]).all().item()
    )


@torch.no_grad()
def materialize_schedule(
    records: Sequence[Any], calibration: np.ndarray, config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run real DPVO bootstrap and return the frozen bootstrap boundary."""
    if not records:
        raise ValueError("schedule materialization requires EuRoC records")
    first_image, _ = _load_frame({"image_path": records[0].rgb_path}, calibration)
    rgb_class, _ = _formal_classes()
    seed = int(config["experiment"]["seed"])
    _seed_everything(seed)
    slam = _make_slam(rgb_class, _runtime_config(config), first_image)
    identity_box: dict[str, FrameIdentity] = {}
    decisions: list[dict[str, Any]] = []
    with _deterministic_patchifier(slam, identity_box, seed):
        for record in records:
            identity = record.identity
            identity_box["identity"] = identity
            image, intrinsics = _load_frame({"image_path": record.rgb_path}, calibration)
            before = int(slam.n)
            slam(int(identity.timestamp_ns), image, intrinsics)
            decisions.append({
                "candidate_index": identity.candidate_index,
                "identity_key": identity.key,
                "motion_accepted": int(slam.n) > before,
            })
            if slam.is_initialized:
                break
    if not slam.is_initialized:
        raise RuntimeError("DPVO bootstrap did not reach eight accepted nodes")
    result = {
        "sequence": records[0].identity.sequence,
        "bootstrap_end_candidate_index": int(decisions[-1]["candidate_index"]),
        "bootstrap_decisions": decisions,
        "bootstrap_decisions_sha256": canonical_sha256(decisions),
        "upload_schedule_derived_after_bootstrap": True,
    }
    del slam
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def run_formal_mode(
    mode: str, frames: Sequence[OnlineFrame], calibration: np.ndarray,
    config: Mapping[str, Any], *, roles: Mapping[str, str],
    store: FrontendPacketStore | None = None,
    hidden_provider: Any | None = None,
    condition_name: str | None = None,
    matched_timing: bool = False,
    profile_graph_runtime: bool = False,
    collect_graph_trace: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if mode not in FORMAL_MODES:
        raise ValueError(mode)
    if not frames or frames[0].rgb_path is None:
        raise ValueError("formal runtime requires an RGB bootstrap frame")
    if mode in PACKET_MODES:
        leaked = [frame.identity.key for frame in frames if roles[frame.identity.key] == "hidden" and frame.rgb_path is not None]
        if leaked:
            raise AssertionError("FullOracle online payload contains hidden RGB path")
        if store is None and hidden_provider is None:
            raise ValueError("FullOracle packet store or hidden provider is required")
        if store is not None and hidden_provider is not None:
            raise ValueError("packet store and hidden provider are mutually exclusive")
    first_image, _ = _load_frame({"image_path": frames[0].rgb_path}, calibration)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rgb_class, packet_class = _formal_classes()
    cls = packet_class if mode in PACKET_MODES else rgb_class
    seed = int(config["experiment"]["seed"])
    _seed_everything(seed)
    slam = _make_slam(cls, _runtime_config(config), first_image)
    identity_box: dict[str, FrameIdentity] = {}
    accepted_count = 0
    rgb_count = 0
    hidden_packet_count = 0
    bootstrap_decisions: list[dict[str, Any]] = []
    bootstrap_end: int | None = None
    first_nonfinite: dict[str, Any] | None = None
    graph_trace: list[dict[str, Any]] = []
    frontend_samples_ms: list[float] = []
    graph_samples_ms: list[float] = []
    if matched_timing:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    scope = (_deterministic_patchifier(
        slam, identity_box, seed,
        frontend_samples_ms if profile_graph_runtime else None,
    ) if mode not in PACKET_MODES else contextlib.nullcontext())
    with scope:
        for frame in frames:
            identity = frame.identity
            role = roles[identity.key]
            if mode == "sparse_rgb" and role == "hidden":
                continue
            before_n = int(slam.n)
            initialized_before = bool(slam.is_initialized)
            if mode in {"matched_full_rgb", "sparse_rgb"}:
                if frame.rgb_path is None:
                    raise AssertionError("RGB runtime frame lacks RGB capability")
                identity_box["identity"] = identity
                image, intrinsics = _load_frame({"image_path": frame.rgb_path}, calibration)
                if profile_graph_runtime:
                    torch.cuda.synchronize(); call_started = time.perf_counter()
                    frontend_before = len(frontend_samples_ms)
                slam(int(identity.timestamp_ns), image, intrinsics)
                if profile_graph_runtime:
                    torch.cuda.synchronize()
                    call_ms = (time.perf_counter() - call_started) * 1000.0
                    if len(frontend_samples_ms) != frontend_before + 1:
                        raise RuntimeError("matched RGB frontend profiling count changed")
                    graph_samples_ms.append(max(0.0, call_ms - frontend_samples_ms[-1]))
                rgb_count += 1
            else:
                if role == "hidden":
                    packet = (hidden_provider.get(identity) if hidden_provider is not None
                              else store.get(identity))
                    intrinsics = torch.as_tensor(calibration[:4], dtype=torch.float32, device="cuda")
                    slam.exp6_hidden_timestamps.add(int(identity.timestamp_ns))
                    hidden_packet_count += 1
                else:
                    if frame.rgb_path is None:
                        raise AssertionError("FullOracle anchor lacks online RGB")
                    image, intrinsics = _load_frame({"image_path": frame.rgb_path}, calibration)
                    packet = extract_frontend_packet(slam, image, identity, seed)
                    if hidden_provider is not None:
                        hidden_provider.observe_anchor(identity, packet)
                    rgb_count += 1
                slam.track_packet(
                    int(identity.timestamp_ns), intrinsics,
                    packet=packet_to_native(
                        packet, identity=identity, experiment_seed=seed,
                        patches_per_image=int(slam.M), patch_size=int(slam.P),
                        context_dim=int(slam.DIM),
                    ), kind=role,
                )
                del packet
            accepted = int(slam.n) > before_n
            accepted_count += int(accepted)
            if not initialized_before:
                bootstrap_decisions.append({
                    "candidate_index": int(identity.candidate_index),
                    "motion_accepted": bool(accepted),
                })
            if bootstrap_end is None and slam.is_initialized:
                bootstrap_end = int(identity.candidate_index)
            if first_nonfinite is None and not _state_is_finite(slam):
                first_nonfinite = {
                    "candidate_index": int(identity.candidate_index),
                    "stage": "post_frame_pose_or_depth",
                }
            if collect_graph_trace:
                graph_trace.append({
                "candidate_index": int(identity.candidate_index),
                "node_count": int(slam.n), "patch_count": int(slam.m),
                "active_timestamps_sha256": canonical_sha256([
                    int(value) for value in slam.tlist[: int(slam.n)]
                ]),
                "factor_topology_sha256": canonical_sha256({
                    name: tensor.detach().cpu().tolist()
                    for name, tensor in (
                        ("ii", slam.pg.ii), ("jj", slam.pg.jj), ("kk", slam.pg.kk),
                    )
                }),
                "pose_state_sha256": (
                    canonical_sha256(slam.pg.poses_[: int(slam.n)].detach().cpu().tolist())
                    if bool(torch.isfinite(slam.pg.poses_[: int(slam.n)]).all().item()) else None
                ),
                "depth_state_sha256": (
                    canonical_sha256(slam.pg.patches_[: int(slam.n), :, 2].detach().cpu().tolist())
                    if bool(torch.isfinite(slam.pg.patches_[: int(slam.n), :, 2]).all().item()) else None
                ),
                "pose_finite": bool(torch.isfinite(slam.pg.poses_[: int(slam.n)]).all().item()),
                "depth_finite": bool(torch.isfinite(slam.pg.patches_[: int(slam.n), :, 2]).all().item()),
                })
    final_node_count_before_terminate = int(slam.n)
    final_patch_count_before_terminate = int(slam.m)
    final_active_factor_count_before_terminate = int(slam.pg.ii.numel())
    if profile_graph_runtime:
        torch.cuda.synchronize(); terminate_started = time.perf_counter()
    poses, timestamps = slam.terminate()
    torch.cuda.synchronize()
    if profile_graph_runtime:
        graph_samples_ms.append((time.perf_counter() - terminate_started) * 1000.0)
    matched_elapsed = float(time.perf_counter() - started) if matched_timing else None
    pose_array = np.asarray(poses, dtype=np.float64)
    timestamp_array = np.asarray(timestamps, dtype=np.uint64)
    if first_nonfinite is None and not np.isfinite(pose_array).all():
        first_nonfinite = {"candidate_index": None, "stage": "terminate_trajectory"}
    metrics = {
        "mode": mode,
        "condition_name": condition_name or mode,
        "model_training": bool(slam.network.training),
        "gradient_enabled": bool(torch.is_grad_enabled()),
        "input_candidate_count": len(frames),
        "processed_observation_count": int(slam.counter),
        "accepted_node_count_before_culling_sum": accepted_count,
        "final_node_count_before_terminate": final_node_count_before_terminate,
        "final_patch_count_before_terminate": final_patch_count_before_terminate,
        "bootstrap_end_candidate_index": bootstrap_end,
        "bootstrap_decisions": bootstrap_decisions,
        "rgb_uploaded_frame_count": rgb_count,
        "hidden_oracle_packet_count": hidden_packet_count,
        "hidden_online_rgb_violation_count": 0,
        "factor_count_allocated": int(slam.exp6_factor_count),
        "final_active_factor_count": final_active_factor_count_before_terminate,
        "hidden_source_factor_count": int(slam.exp6_hidden_source_factor_count),
        "hidden_target_factor_count": int(slam.exp6_hidden_target_factor_count),
        "packet_integrity_failure_count": 0,
        "representation": (
            hidden_provider.sanitized_descriptor()
            if hidden_provider is not None
            else (representation_contract(store.store.descriptor["schema"])
                  if mode in PACKET_MODES and store is not None and hasattr(store, "store")
                  else None)
        ),
        "hidden_provider_usage": (
            hidden_provider.usage_payload() if hidden_provider is not None else None
        ),
        "graph_trace": graph_trace,
        "first_nonfinite_event": first_nonfinite,
        "finite_trajectory": bool(np.isfinite(pose_array).all()),
        "tracking_success": first_nonfinite is None and len(pose_array) == int(slam.counter),
        "trajectory_pose_count": len(pose_array),
        "timestamp_contract_exact": bool(
            len(timestamp_array) == int(slam.counter)
            and np.array_equal(timestamp_array, np.asarray(slam.tlist, dtype=np.uint64))
        ),
        "elapsed_seconds": (matched_elapsed if matched_elapsed is not None
                            else float(time.perf_counter() - started)),
        "dpvo_graph_runtime_total_ms": float(sum(graph_samples_ms)),
        "dpvo_graph_runtime_mean_ms_per_processed_observation": (
            float(sum(graph_samples_ms)) / int(slam.counter) if graph_samples_ms else None
        ),
        "peak_gpu_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    arrays = {"poses": pose_array, "timestamps_ns": timestamp_array}
    del slam
    torch.cuda.empty_cache()
    return metrics, arrays


@torch.no_grad()
def run_packet_observations(
    observations: Sequence[PacketObservation] | Any,
    calibration: np.ndarray, config: Mapping[str, Any], *,
    image_height: int, image_width: int, condition_name: str,
    profiler: Any | None = None, on_tracked: Any | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run DPVO using only pre-authorized FMap packets.

    The iterable may be lazy: H2 uses this to make the closing anchor arrive and
    finish encoding before buffered hidden observations are materialized.
    """
    if image_height <= 0 or image_width <= 0:
        raise ValueError("positive DPVO image geometry is required")
    dummy = torch.empty((3, int(image_height), int(image_width)), dtype=torch.uint8)
    _, packet_class = _formal_classes()
    seed = int(config["experiment"]["seed"])
    _seed_everything(seed)
    slam = _make_slam(packet_class, _runtime_config(config), dummy)
    intrinsics = torch.as_tensor(calibration[:4], dtype=torch.float32, device="cuda")
    hidden_timestamps: set[int] = set()
    processed = 0
    previous_timestamp = -1
    first_nonfinite: dict[str, Any] | None = None
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for observation in observations:
        if not isinstance(observation, PacketObservation):
            raise TypeError("packet iterable yielded a non-PacketObservation")
        identity = observation.identity
        if identity.timestamp_ns <= previous_timestamp:
            raise RuntimeError("packet observations must be strictly timestamp ordered")
        previous_timestamp = identity.timestamp_ns
        if observation.kind == "hidden":
            slam.exp6_hidden_timestamps.add(int(identity.timestamp_ns))
            hidden_timestamps.add(int(identity.timestamp_ns))
        torch.cuda.synchronize()
        stage_started = time.perf_counter()
        slam.track_packet(
            int(identity.timestamp_ns), intrinsics,
            packet=packet_to_native(
                observation.packet, identity=identity, experiment_seed=seed,
                patches_per_image=int(slam.M), patch_size=int(slam.P),
                context_dim=int(slam.DIM),
            ),
            kind=observation.kind,
        )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - stage_started) * 1000.0
        if profiler is not None:
            profiler.add("dpvo", elapsed_ms)
        processed += 1
        if first_nonfinite is None and not _state_is_finite(slam):
            first_nonfinite = {"candidate_index": identity.candidate_index,
                               "stage": "post_packet_pose_or_depth"}
        if on_tracked is not None:
            on_tracked(observation, elapsed_ms)
    final_nodes = int(slam.n)
    final_patches = int(slam.m)
    torch.cuda.synchronize()
    terminate_started = time.perf_counter()
    poses, timestamps = slam.terminate()
    torch.cuda.synchronize()
    terminate_ms = (time.perf_counter() - terminate_started) * 1000.0
    if profiler is not None:
        profiler.add("dpvo", terminate_ms)
    pose_array = np.asarray(poses, dtype=np.float64)
    timestamp_array = np.asarray(timestamps, dtype=np.uint64)
    if first_nonfinite is None and not np.isfinite(pose_array).all():
        first_nonfinite = {"candidate_index": None, "stage": "terminate_trajectory"}
    metrics = {
        "condition_name": condition_name,
        "model_training": bool(slam.network.training),
        "gradient_enabled": bool(torch.is_grad_enabled()),
        "processed_observation_count": processed,
        "final_node_count_before_terminate": final_nodes,
        "final_patch_count_before_terminate": final_patches,
        "rgb_uploaded_frame_count": 0,
        "hidden_packet_count": len(hidden_timestamps),
        "hidden_online_rgb_violation_count": 0,
        "factor_count_allocated": int(slam.exp6_factor_count),
        "hidden_source_factor_count": int(slam.exp6_hidden_source_factor_count),
        "hidden_target_factor_count": int(slam.exp6_hidden_target_factor_count),
        "finite_trajectory": bool(np.isfinite(pose_array).all()),
        "tracking_success": first_nonfinite is None and len(pose_array) == int(slam.counter),
        "first_nonfinite_event": first_nonfinite,
        "trajectory_pose_count": len(pose_array),
        "timestamp_contract_exact": bool(
            len(timestamp_array) == int(slam.counter)
            and np.array_equal(timestamp_array, np.asarray(slam.tlist, dtype=np.uint64))
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_gpu_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "representation": representation_contract(ZERO_PACKET_SCHEMA),
    }
    arrays = {"poses": pose_array, "timestamps_ns": timestamp_array}
    del slam
    torch.cuda.empty_cache()
    return metrics, arrays


@torch.no_grad()
def run_deployment_observations(
    observations: Any, calibration: np.ndarray, config: Mapping[str, Any], *,
    image_height: int, image_width: int, condition_name: str,
    expected_roles: Mapping[str, str], profiler: Any,
    worker_barrier: Any | None = None,
    on_tracked: Any | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run strict H2 with native RGB anchors and FMap-only hidden packets."""
    if image_height <= 0 or image_width <= 0:
        raise ValueError("positive DPVO image geometry is required")
    expected_keys = list(expected_roles)
    if not expected_keys or set(expected_roles.values()) - {"anchor", "hidden"}:
        raise ValueError("expected_roles must define a non-empty anchor/hidden population")
    dummy = torch.empty((3, int(image_height), int(image_width)), dtype=torch.uint8)
    _, packet_class = _formal_classes()
    seed = int(config["experiment"]["seed"])
    _seed_everything(seed)
    # Model/checkpoint load is completed before online profiling begins.
    slam = _make_slam(packet_class, _runtime_config(config), dummy)
    hidden_intrinsics = torch.as_tensor(
        calibration[:4], dtype=torch.float32, device="cuda",
    )
    torch.cuda.empty_cache()
    processed_keys: list[str] = []
    processed_set: set[str] = set()
    native_frontend_keys: set[str] = set()
    hidden_timestamps: set[int] = set()
    previous_timestamp = -1
    first_nonfinite: dict[str, Any] | None = None
    # Execution-only early frontend cache: exposes no graph state to provider.
    early_frontend = {}
    if worker_barrier is not None and hasattr(worker_barrier, "attach_anchor_frontend"):
        from types import SimpleNamespace
        frontend = SimpleNamespace(network=slam.network, cfg=slam.cfg, M=slam.M)
        def prepare_anchor(observation):
            if (not isinstance(observation, AnchorRGBObservation)
                    or expected_roles.get(observation.identity.key) != "anchor"):
                raise PermissionError("early frontend accepts uploaded anchors only")
            key = observation.identity.key
            if key in early_frontend or key in native_frontend_keys:
                raise RuntimeError("native anchor frontend computed twice")
            cpu_started = time.perf_counter()
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record()
            early_frontend[key] = extract_frontend_packet(frontend, observation.image, observation.identity, seed)
            end.record(); end.synchronize()
            cuda_ms = float(start.elapsed_time(end))
            profiler.add("native_dpvo_frontend", cuda_ms)
            profiler.add_stage_c_cpu(
                "native_frontend_compute",
                (time.perf_counter() - cpu_started) * 1000.0,
            )
            profiler.add_stage_c_cuda("native_frontend_compute", cuda_ms)
            native_frontend_keys.add(key)
        worker_barrier.attach_anchor_frontend(prepare_anchor)
    ready_ack = worker_barrier.prepare_online() if worker_barrier is not None else None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    matched_started = time.perf_counter()
    for observation in observations:
        if not isinstance(observation, (AnchorRGBObservation, PacketObservation)):
            raise TypeError("deployment iterable yielded an unauthorized observation type")
        identity = observation.identity
        role = expected_roles.get(identity.key)
        if role is None:
            raise RuntimeError(f"observation identity is outside deployment schedule: {identity.key}")
        if identity.key in processed_set:
            raise RuntimeError(f"deployment candidate was inserted twice: {identity.key}")
        if identity.timestamp_ns <= previous_timestamp:
            raise RuntimeError("deployment observations must be strictly timestamp ordered")
        previous_timestamp = identity.timestamp_ns
        if role == "anchor":
            if not isinstance(observation, AnchorRGBObservation):
                raise RuntimeError("scheduled anchor lacks native RGB observation")
            if identity.key in early_frontend:
                packet = early_frontend.pop(identity.key)
            else:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                frontend_cpu_started = time.perf_counter()
                start_event.record()
                packet = extract_frontend_packet(slam, observation.image, identity, seed)
                end_event.record(); end_event.synchronize()
                frontend_cuda_ms = float(start_event.elapsed_time(end_event))
                profiler.add("native_dpvo_frontend", frontend_cuda_ms)
                profiler.add_stage_c_cpu(
                    "native_frontend_compute",
                    (time.perf_counter() - frontend_cpu_started) * 1000.0,
                )
                profiler.add_stage_c_cuda(
                    "native_frontend_compute", frontend_cuda_ms,
                )
            intrinsics = observation.intrinsics
            native_frontend_keys.add(identity.key)
            kind = "anchor"
        else:
            if not isinstance(observation, PacketObservation) or observation.kind != "hidden":
                raise PermissionError("hidden RGB capability entered deployment runtime")
            packet = observation.packet
            intrinsics = hidden_intrinsics
            slam.exp6_hidden_timestamps.add(int(identity.timestamp_ns))
            hidden_timestamps.add(int(identity.timestamp_ns))
            kind = "hidden"
        sync_started = time.perf_counter()
        torch.cuda.synchronize()
        profiler.add_stage_c_cpu(
            "dpvo_sync_wait", (time.perf_counter() - sync_started) * 1000.0,
        )
        conversion_start = torch.cuda.Event(enable_timing=True)
        conversion_end = torch.cuda.Event(enable_timing=True)
        conversion_cpu_started = time.perf_counter()
        conversion_start.record()
        native_packet = packet_to_native(
            packet, identity=identity, experiment_seed=seed,
            patches_per_image=int(slam.M), patch_size=int(slam.P),
            context_dim=int(slam.DIM),
        )
        conversion_end.record()
        profiler.add_stage_c_cpu(
            "stage_c_python_other",
            (time.perf_counter() - conversion_cpu_started) * 1000.0,
        )
        graph_start = torch.cuda.Event(enable_timing=True)
        graph_end = torch.cuda.Event(enable_timing=True)
        graph_cpu_started = time.perf_counter()
        graph_start.record()
        slam.track_packet(
            int(identity.timestamp_ns), intrinsics, packet=native_packet, kind=kind,
        )
        graph_end.record()
        graph_cpu_ms = (time.perf_counter() - graph_cpu_started) * 1000.0
        profiler.add_stage_c_cpu(
            "dpvo_graph_compute", graph_cpu_ms,
        )
        sync_started = time.perf_counter()
        torch.cuda.synchronize()
        profiler.add_stage_c_cpu(
            "dpvo_sync_wait", (time.perf_counter() - sync_started) * 1000.0,
        )
        conversion_cuda_ms = float(conversion_start.elapsed_time(conversion_end))
        graph_cuda_ms = float(graph_start.elapsed_time(graph_end))
        profiler.add_stage_c_cuda("stage_c_python_other", conversion_cuda_ms)
        profiler.add_stage_c_cuda("dpvo_graph_compute", graph_cuda_ms)
        profiler.add("dpvo_graph_runtime", graph_cuda_ms)
        processed_keys.append(identity.key); processed_set.add(identity.key)
        if first_nonfinite is None and not _state_is_finite(slam):
            first_nonfinite = {
                "candidate_index": identity.candidate_index,
                "stage": "post_native_anchor_or_hidden_packet",
            }
        if on_tracked is not None:
            on_tracked(observation, graph_cuda_ms)
        del packet, native_packet
    if processed_keys != expected_keys:
        missing = sorted(set(expected_keys) - processed_set)
        extra = sorted(processed_set - set(expected_keys))
        raise RuntimeError(
            f"deployment candidate population mismatch: missing={missing}, extra={extra}"
        )
    anchor_keys = {key for key, role in expected_roles.items() if role == "anchor"}
    if native_frontend_keys != anchor_keys:
        raise RuntimeError("native anchor frontend population is incomplete or duplicated")
    final_nodes = int(slam.n)
    final_patches = int(slam.m)
    final_active_factors = int(slam.pg.ii.numel())
    sync_started = time.perf_counter()
    torch.cuda.synchronize()
    profiler.add_stage_c_cpu(
        "dpvo_sync_wait", (time.perf_counter() - sync_started) * 1000.0,
    )
    terminate_start = torch.cuda.Event(enable_timing=True)
    terminate_end = torch.cuda.Event(enable_timing=True)
    terminate_cpu_started = time.perf_counter()
    terminate_start.record()
    poses, timestamps = slam.terminate()
    terminate_end.record()
    terminate_cpu_ms = (time.perf_counter() - terminate_cpu_started) * 1000.0
    profiler.add_stage_c_cpu(
        "dpvo_graph_compute", terminate_cpu_ms,
    )
    sync_started = time.perf_counter()
    torch.cuda.synchronize()
    profiler.add_stage_c_cpu(
        "dpvo_sync_wait", (time.perf_counter() - sync_started) * 1000.0,
    )
    terminate_cuda_ms = float(terminate_start.elapsed_time(terminate_end))
    profiler.add_stage_c_cuda("dpvo_graph_compute", terminate_cuda_ms)
    profiler.add("dpvo_graph_runtime", terminate_cuda_ms)
    flush_ack = worker_barrier.flush_online() if worker_barrier is not None else None
    sync_started = time.perf_counter()
    torch.cuda.synchronize()
    profiler.add_stage_c_cpu(
        "dpvo_sync_wait", (time.perf_counter() - sync_started) * 1000.0,
    )
    elapsed_seconds = float(time.perf_counter() - matched_started)
    profiler.finalize_stage_c(elapsed_seconds * 1000.0)
    pose_array = np.asarray(poses, dtype=np.float64)
    timestamp_array = np.asarray(timestamps, dtype=np.uint64)
    if first_nonfinite is None and not np.isfinite(pose_array).all():
        first_nonfinite = {"candidate_index": None, "stage": "terminate_trajectory"}
    metrics = {
        "condition_name": condition_name,
        "model_training": bool(slam.network.training),
        "gradient_enabled": bool(torch.is_grad_enabled()),
        "processed_observation_count": len(processed_keys),
        "final_node_count_before_terminate": final_nodes,
        "final_patch_count_before_terminate": final_patches,
        "rgb_uploaded_frame_count": len(native_frontend_keys),
        "native_anchor_frontend_count": len(native_frontend_keys),
        "dpvo_insertion_count": len(processed_keys),
        "hidden_packet_count": len(hidden_timestamps),
        "hidden_online_rgb_violation_count": 0,
        "candidate_consumed_exactly_once": True,
        "candidate_identity_order_sha256": canonical_sha256(processed_keys),
        "factor_count_allocated": int(slam.exp6_factor_count),
        "final_active_factor_count": final_active_factors,
        "hidden_source_factor_count": int(slam.exp6_hidden_source_factor_count),
        "hidden_target_factor_count": int(slam.exp6_hidden_target_factor_count),
        "finite_trajectory": bool(np.isfinite(pose_array).all()),
        "tracking_success": first_nonfinite is None and len(pose_array) == int(slam.counter),
        "first_nonfinite_event": first_nonfinite,
        "trajectory_pose_count": len(pose_array),
        "timestamp_contract_exact": bool(
            len(timestamp_array) == int(slam.counter)
            and np.array_equal(timestamp_array, np.asarray(slam.tlist, dtype=np.uint64))
        ),
        "elapsed_seconds": elapsed_seconds,
        "dpvo_graph_runtime_total_ms": float(sum(profiler.samples["dpvo_graph_runtime"])),
        "dpvo_graph_runtime_mean_ms_per_processed_observation": (
            float(sum(profiler.samples["dpvo_graph_runtime"])) / len(processed_keys)
        ),
        "dpvo_graph_runtime_definition": (
            "CUDA event span bounded immediately around slam.track_packet and "
            "terminate; queue wait, transfer, packet conversion, and outer "
            "synchronization are excluded; host dispatch gaps inside those calls "
            "remain part of the span"
        ),
        "dpvo_graph_runtime_legacy_definition": (
            "older artifacts used CPU wall from before packet_to_native through "
            "slam.track_packet and the following CUDA synchronize, plus terminate; "
            "those values are not comparable to this pure CUDA-event field"
        ),
        "stage_c_timing": profiler.payload(
            peak_online_vram_bytes=int(torch.cuda.max_memory_allocated())
        )["stage_c_timing"],
        "cross_process_timing_barrier": {
            "worker_present": worker_barrier is not None,
            "online_ready_ack": ready_ack,
            "online_flush_ack": flush_ack,
            "main_cuda_synchronized_before_start": True,
            "main_cuda_synchronized_after_worker_flush_before_stop": True,
            "worker_close_outside_timer": True,
        },
        "peak_gpu_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "representation": {
            "anchor": "native_dpvo_fnet_patchifier",
            "hidden": representation_contract(ZERO_PACKET_SCHEMA),
        },
    }
    arrays = {"poses": pose_array, "timestamps_ns": timestamp_array}
    del slam
    torch.cuda.empty_cache()
    return metrics, arrays
