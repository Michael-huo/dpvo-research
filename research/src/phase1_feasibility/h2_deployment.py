"""Capability-isolated delayed/bracketed H2 online replay."""

from __future__ import annotations

import time
import contextlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from .jepa_fmap import coordinate_masks, preprocess_full_fov_rgb, tokens_to_field
from .jepa_runtime import JepaSidecar, load_dpvo_domain
from .oracle_packet import FMapZeroContextPacket
from .predictor import AnchorInterval
from .profiling import OnlineProfiler
from .runtime import AnchorRGBObservation, PacketObservation
from .transport import (RobustCorrespondence, estimate_robust_correspondence,
                        robust_transport_interpolation)


def cuda_call_ms(function: Any) -> tuple[Any, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = function()
    end.record(); end.synchronize()
    return value, float(start.elapsed_time(end))


class DelayedDeploymentProvider:
    """Owns only anchor RGB capabilities and frozen inference components.

    Hidden records are represented solely by FrameIdentity objects inside the
    interval schedule.  No hidden RGB/path/reference/GT/graph object is accepted.
    """

    def __init__(
        self, *, anchor_paths: Mapping[str, str], identities: Sequence[Any],
        intervals: Sequence[AnchorInterval], transform: Any, calibration: np.ndarray,
        config: Mapping[str, Any], temporary: Path, bridge: torch.nn.Module,
        predictor: torch.nn.Module, transport_calibration: Mapping[str, Any],
        profiler: OnlineProfiler, predict_hidden: bool = True,
        performance: Any | None = None, transfer_ledger: Any | None = None,
    ) -> None:
        identity_keys = {item.key for item in identities}
        if set(anchor_paths) - identity_keys:
            raise ValueError("anchor capability contains identities outside the replay")
        hidden_keys = {query.identity.key for interval in intervals for query in interval.hidden}
        if set(anchor_paths) & hidden_keys:
            raise PermissionError("hidden RGB capability entered deployment provider")
        self.anchor_paths = dict(anchor_paths)
        self.identities = tuple(identities)
        self.intervals = tuple(intervals)
        self.transform = transform
        self.calibration = np.asarray(calibration)
        # Keep only model-worker configuration.  Dataset roots, GT patterns,
        # DPVO checkpoints and every offline-reference path are deliberately
        # absent from the deployment capability object.
        self.config = {
            "runtime": dict(config["runtime"]),
            "jepa": dict(config["jepa"]),
        }
        self.temporary = temporary
        self.bridge = bridge
        self.predictor = predictor
        self.transport_calibration = dict(transport_calibration)
        self.profiler = profiler
        self.performance = performance
        self.transfer_ledger = transfer_ledger
        self.predict_hidden = bool(predict_hidden)
        self.available_anchor_keys: set[str] = set()
        self.encoded_anchor_keys: list[str] = []
        self.yielded_candidate_keys: list[str] = []
        self.consumed_hidden_keys: list[str] = []
        self.interval_start_compute_ms: dict[str, float] = {}
        self.hidden_context_wait: dict[str, float] = {}
        self._closing = {interval.anchor1.key: interval for interval in intervals}
        self._last_anchor_field: torch.Tensor | None = None
        self._last_anchor_key: str | None = None
        self._sidecar: JepaSidecar | None = None
        self.jepa_peak_online_vram_bytes = 0
        self.jepa_worker_pid: int | None = None

    @property
    def allowed_anchor_identity_sha256(self) -> str:
        from .protocol import canonical_sha256
        return canonical_sha256(sorted(self.anchor_paths))

    def _preprocess(
        self, identity: Any,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        if identity.key not in self.anchor_paths:
            raise PermissionError(f"no uploaded anchor RGB capability for {identity.key}")
        started = time.perf_counter()
        bgr, _ = load_dpvo_domain(self.anchor_paths[identity.key], self.calibration)
        jepa_tensor, _ = preprocess_full_fov_rgb(
            bgr[..., ::-1].copy(), self.transform,
        )
        transfer_started = time.perf_counter()
        dpvo_image = torch.from_numpy(bgr).permute(2, 0, 1).cuda()
        intrinsics = torch.as_tensor(
            self.calibration[:4], dtype=torch.float32, device="cuda",
        )
        torch.cuda.synchronize()
        if self.transfer_ledger is not None:
            self.transfer_ledger.add(
                "main_cpu_to_gpu_anchor_image_intrinsics",
                byte_count=int(bgr.nbytes + self.calibration[:4].nbytes),
                cpu_ms=(time.perf_counter() - transfer_started) * 1000.0,
            )
        self.profiler.add(
            "anchor_decode_preprocess", (time.perf_counter() - started) * 1000.0,
        )
        return jepa_tensor.numpy(), dpvo_image, intrinsics

    def _encode_preprocessed(
        self, identity: Any, request_id: str, jepa_input: np.ndarray,
    ) -> torch.Tensor:
        if self._sidecar is None:
            raise RuntimeError("JEPA sidecar is not active")
        if identity.key not in self.anchor_paths:
            raise PermissionError(f"no uploaded anchor RGB capability for {identity.key}")
        if identity.key in self.encoded_anchor_keys:
            raise RuntimeError(f"anchor JEPA context encoded twice: {identity.key}")
        source = self.temporary / f"online_{request_id}.npy"
        destination = self.temporary / f"online_{request_id}_block5.npy"
        # Temporary sidecar IO is deliberately outside every inference stage.
        io_started = time.perf_counter()
        np.save(source, jepa_input[None], allow_pickle=False)
        if self.transfer_ledger is not None:
            self.transfer_ledger.add(
                "temporary_npy_main_to_worker_write", byte_count=source.stat().st_size,
                cpu_ms=(time.perf_counter() - io_started) * 1000.0,
            )
        ipc_started = time.perf_counter()
        response = self._sidecar.extract(source, destination, request_id)
        if self.transfer_ledger is not None:
            request_bytes = len((json.dumps({
                "action": "extract", "request_id": request_id,
                "input_npy": str(source), "block5_npy": str(destination),
            }) + "\n").encode("utf-8"))
            self.transfer_ledger.add(
                "worker_main_request_response_wall", byte_count=request_bytes,
                cpu_ms=(time.perf_counter() - ipc_started) * 1000.0,
            )
            self.transfer_ledger.add(
                "worker_vjepa_cuda_encode", byte_count=0,
                cpu_ms=0.0, cuda_ms=float(response["encoder_inference_ms"]),
            )
        self.profiler.add("jepa_encoder", float(response["encoder_inference_ms"]))
        read_started = time.perf_counter()
        block5 = np.asarray(np.load(destination), dtype=np.float32)
        if self.transfer_ledger is not None:
            self.transfer_ledger.add(
                "temporary_npy_worker_to_main_read", byte_count=destination.stat().st_size,
                cpu_ms=(time.perf_counter() - read_started) * 1000.0,
            )
        transfer_started = time.perf_counter()
        tokens = torch.from_numpy(block5).cuda()
        if self.transfer_ledger is not None:
            self.transfer_ledger.add(
                "main_cpu_to_gpu_block5", byte_count=block5.nbytes,
                cpu_ms=(time.perf_counter() - transfer_started) * 1000.0,
            )
        source.unlink(); destination.unlink()
        self.encoded_anchor_keys.append(identity.key)
        return tokens_to_field(tokens, self.transform)

    def _bridge_packet(self, field: torch.Tensor) -> FMapZeroContextPacket:
        started = time.perf_counter()
        fmap, elapsed = cuda_call_ms(
            lambda: self.bridge(field.flatten(2).transpose(1, 2)),
        )
        self.profiler.add("bridge", elapsed)
        packet_started = time.perf_counter()
        packet = FMapZeroContextPacket(fmap[:, None])
        if self.transfer_ledger is not None:
            self.transfer_ledger.add(
                "bridge_forward_wall", byte_count=fmap.numel() * fmap.element_size(),
                cpu_ms=(packet_started - started) * 1000.0, cuda_ms=elapsed,
            )
            self.transfer_ledger.add(
                "bridge_packet_assembly", byte_count=0,
                cpu_ms=(time.perf_counter() - packet_started) * 1000.0,
            )
        return packet

    def _predict_interval(self, interval: AnchorInterval, right: torch.Tensor) -> torch.Tensor:
        if self._last_anchor_key != interval.anchor0.key or self._last_anchor_field is None:
            raise RuntimeError("A0 is not the most recent available anchor")
        if interval.anchor1.key not in self.available_anchor_keys:
            raise RuntimeError("closing anchor A5 has not arrived and encoded")
        transfer_started = time.perf_counter()
        left = self._last_anchor_field.to(device="cuda", dtype=torch.float32)
        if self.transfer_ledger is not None:
            self.transfer_ledger.add(
                "previous_anchor_cpu_to_gpu",
                byte_count=left.numel() * left.element_size(),
                cpu_ms=(time.perf_counter() - transfer_started) * 1000.0,
            )
        mask = torch.from_numpy(coordinate_masks(self.transform)["valid_token_mask"]).cuda()
        count = len(interval.hidden)

        def predict() -> torch.Tensor:
            correspondence = estimate_robust_correspondence(
                left, right, mask, self.transport_calibration,
                profiler=self.performance,
            )
            stage = (contextlib.nullcontext() if self.performance is None
                     else self.performance.stage("correspondence_query_assembly"))
            with stage:
                repeated = RobustCorrespondence(*[
                    getattr(correspondence, name).repeat(
                        count, *([1] * (getattr(correspondence, name).ndim - 1))
                    ) for name in correspondence.__dataclass_fields__
                ])
                alpha = torch.tensor([query.alpha for query in interval.hidden], device="cuda")
                delta = torch.tensor([query.delta_t_seconds for query in interval.hidden], device="cuda")
            transported = robust_transport_interpolation(
                left.repeat(count, 1, 1, 1), right.repeat(count, 1, 1, 1),
                alpha, repeated, mask, profiler=self.performance,
            )
            stage = (contextlib.nullcontext() if self.performance is None
                     else self.performance.stage("neural_residual_predictor_forward"))
            with stage:
                predicted = self.predictor(
                    transported.field, transported.warped_difference,
                    transported.warp0.coverage, transported.warp1.coverage,
                    transported.fused_confidence, alpha, delta,
                )
            stage = (contextlib.nullcontext() if self.performance is None
                     else self.performance.stage("predictor_output_assembly"))
            with stage:
                result = predicted
            return result

        if self.performance is None:
            predicted, elapsed = cuda_call_ms(predict)
        else:
            self.performance.begin_outer()
            predicted = predict()
            timing = self.performance.finish_outer()
            elapsed = float(timing["cuda_outer_ms"] or timing["cpu_outer_ms"])
        self.profiler.add("jepa_predictor", elapsed)
        return predicted

    def _warmup(self, first_anchor: Any) -> None:
        if self._sidecar is None:
            raise RuntimeError("JEPA sidecar is not active")
        source = self.temporary / "warmup.npy"
        destination = self.temporary / "warmup_block5.npy"
        bgr, _ = load_dpvo_domain(self.anchor_paths[first_anchor.key], self.calibration)
        tensor, _ = preprocess_full_fov_rgb(bgr[..., ::-1].copy(), self.transform)
        np.save(source, tensor.numpy()[None], allow_pickle=False)
        self._sidecar.extract(source, destination, "warmup")
        tokens = torch.from_numpy(np.asarray(np.load(destination), dtype=np.float32)).cuda()
        with torch.no_grad():
            field = tokens_to_field(tokens, self.transform)
            batch = len(self.intervals[0].hidden) if self.intervals else 1
            transport = field.repeat(batch, 1, 1, 1)
            reliability = torch.ones(
                batch, 1, field.shape[-2], field.shape[-1], device="cuda",
            )
            alpha = torch.full((batch,), .5, device="cuda")
            delta = torch.full((batch,), .5, device="cuda")
            predicted = self.predictor(
                transport, torch.zeros_like(transport), reliability,
                reliability, reliability, alpha, delta,
            )
            self.bridge(predicted.flatten(2).transpose(1, 2))
        torch.cuda.synchronize()
        source.unlink(); destination.unlink()

    @contextlib.contextmanager
    def online_session(self) -> Iterator["DelayedDeploymentProvider"]:
        if self._sidecar is not None:
            raise RuntimeError("deployment sidecar session is already active")
        first_anchor = next(item for item in self.identities if item.key in self.anchor_paths)
        with JepaSidecar(self.config, self.temporary) as sidecar:
            self._sidecar = sidecar
            worker_pid = sidecar.provenance.get("worker_pid")
            self.jepa_worker_pid = int(worker_pid) if worker_pid is not None else None
            self._warmup(first_anchor)
            try:
                yield self
            finally:
                self._sidecar = None

    def prepare_online(self) -> dict[str, Any]:
        if self._sidecar is None:
            raise RuntimeError("deployment sidecar is not active")
        return self._sidecar.prepare_online("h2_matched_online")

    def flush_online(self) -> dict[str, Any]:
        if self._sidecar is None:
            raise RuntimeError("deployment sidecar is not active")
        result = self._sidecar.flush_online("h2_matched_online")
        self.jepa_peak_online_vram_bytes = int(
            self._sidecar.provenance["peak_gpu_memory_allocated_bytes"]
        )
        return result

    def _record_yield(self, identity: Any) -> None:
        if identity.key in self.yielded_candidate_keys:
            raise RuntimeError(f"deployment candidate yielded twice: {identity.key}")
        self.yielded_candidate_keys.append(identity.key)

    def observations(self) -> Iterator[AnchorRGBObservation | PacketObservation]:
        if self._sidecar is None:
            raise RuntimeError("observations require an active online sidecar session")
        for identity in self.identities:
            if identity.key not in self.anchor_paths:
                continue
            interval = self._closing.get(identity.key)
            interval_compute_started = self.profiler.profiled_stage_subtotal_ms
            jepa_input, dpvo_image, intrinsics = self._preprocess(identity)
            right = self._encode_preprocessed(
                identity, str(identity.candidate_index), jepa_input,
            )
            self.available_anchor_keys.add(identity.key)
            if interval is not None and self.predict_hidden:
                predicted = self._predict_interval(interval, right)
                predicted_packets = self._bridge_packet(predicted)
                for offset, query in enumerate(interval.hidden):
                    key = query.identity.key
                    self.interval_start_compute_ms[key] = interval_compute_started
                    wait = (interval.anchor1.timestamp_ns - query.identity.timestamp_ns) / 1e6
                    self.hidden_context_wait[key] = wait
                    self.profiler.context_wait_ms.append(wait)
                    self.consumed_hidden_keys.append(key)
                    self._record_yield(query.identity)
                    yield PacketObservation(
                        query.identity,
                        FMapZeroContextPacket(predicted_packets.fmap[offset:offset + 1]),
                        "hidden", interval.anchor1.timestamp_ns,
                    )
            self._record_yield(identity)
            yield AnchorRGBObservation(
                identity, dpvo_image, intrinsics, identity.timestamp_ns,
            )
            self._last_anchor_key = identity.key
            transfer_started = time.perf_counter()
            self._last_anchor_field = right.detach().half().cpu()
            if self.transfer_ledger is not None:
                self.transfer_ledger.add(
                    "previous_anchor_gpu_to_cpu", byte_count=(
                        self._last_anchor_field.numel() * self._last_anchor_field.element_size()
                    ), cpu_ms=(time.perf_counter() - transfer_started) * 1000.0,
                )

    def on_tracked(self, observation: Any, _dpvo_ms: float) -> None:
        if not isinstance(observation, PacketObservation) or observation.kind != "hidden":
            return
        key = observation.identity.key
        online_after_closing = (
            self.profiler.profiled_stage_subtotal_ms - self.interval_start_compute_ms[key]
        )
        self.profiler.effective_hidden_delay_ms.append(
            self.hidden_context_wait[key] + online_after_closing,
        )

    def usage_payload(self) -> dict[str, Any]:
        expected = {query.identity.key for interval in self.intervals for query in interval.hidden}
        consumed = set(self.consumed_hidden_keys)
        return {
            "provider_schema": "exp6_h2_native_anchor_delayed_capability",
            "allowed_online_fields": ["anchor_rgb", "anchor_identity", "hidden_identity_timestamp"],
            "contains_hidden_rgb_or_path_capability": False,
            "contains_hidden_reference_or_groundtruth_capability": False,
            "closing_anchor_must_be_available": True,
            "timestamp_causal": False,
            "allowed_anchor_identity_sha256": self.allowed_anchor_identity_sha256,
            "hidden_consumed_exactly_once": (
                not self.predict_hidden or (consumed == expected and len(consumed) == len(self.consumed_hidden_keys))
            ),
            "available_anchor_count": len(self.available_anchor_keys),
            "anchor_encoded_exactly_once": (
                set(self.encoded_anchor_keys) == set(self.anchor_paths)
                and len(self.encoded_anchor_keys) == len(set(self.encoded_anchor_keys))
            ),
            "candidate_yielded_exactly_once": (
                set(self.yielded_candidate_keys) == {item.key for item in self.identities}
                and len(self.yielded_candidate_keys) == len(set(self.yielded_candidate_keys))
            ),
            "hidden_consumption_count": len(self.consumed_hidden_keys),
            "jepa_worker_peak_online_vram_bytes": self.jepa_peak_online_vram_bytes,
            "jepa_worker_pid": self.jepa_worker_pid,
            "jepa_worker_logical_cuda_ordinal": 0,
        }
