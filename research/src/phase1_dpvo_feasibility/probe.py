from __future__ import annotations

import heapq
import json
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .behavior import BehaviorRecorder
from .runtime import splitmix64_array


class EventWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        self.count = 0

    def write(self, event_type: str, **payload: Any) -> None:
        record = {"event_index": self.count, "event_type": event_type, **payload}
        self.handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.handle.flush()
        self.count += 1

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()


class NullEventWriter:
    """Avoid per-update debug I/O in formal behavior runs."""

    count = 0

    def write(self, event_type: str, **payload: Any) -> None:
        return None

    def close(self) -> None:
        return None


class BottomKSampler:
    """Keep values with the smallest deterministic 64-bit scores."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.heap: list[tuple[int, int, dict[str, Any]]] = []

    @property
    def threshold(self) -> int:
        if len(self.heap) < self.capacity:
            return (1 << 64) - 1
        return -self.heap[0][0]

    def add(self, score: int, unique_key: int, value: dict[str, Any]) -> None:
        if self.capacity <= 0:
            return
        entry = (-int(score), int(unique_key), value)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, entry)
        elif int(score) < -self.heap[0][0]:
            heapq.heapreplace(self.heap, entry)

    def values(self) -> list[dict[str, Any]]:
        ordered = sorted(self.heap, key=lambda item: (-item[0], item[1]))
        return [item[2] for item in ordered]


class DPVOTruthProbe:
    """Read-only runtime instrumentation for the fixed Experiment 1 protocol."""

    def __init__(
        self,
        slam: Any,
        config: dict[str, Any],
        events_path: str | Path,
    ):
        self.slam = slam
        self.config = config
        self.probe_config = config["probe"]
        self.sanity_validation = bool(self.probe_config.get("sanity_validation", False))
        self.events = EventWriter(events_path) if self.sanity_validation else NullEventWriter()
        self.patch_sampler = BottomKSampler(self.probe_config["patch_tensor_samples"])
        self.corr_sampler = BottomKSampler(
            self.probe_config["correlation_tensor_samples"]
        )
        self.behavior = (
            BehaviorRecorder(config)
            if bool(config.get("behavior", {}).get("enabled", False))
            else None
        )

        self.current_frame: dict[str, Any] | None = None
        self.current_phase = "idle"
        self.terminating = False
        self.update_id = 0
        self.next_factor_uid = 0
        self.active_factor_uids: list[int] = []
        self.active_source_patch_uids: list[int] = []
        self.active_target_frame_ids: list[int] = []
        self.patch_states: dict[int, dict[str, Any]] = {}
        self._frame_factor_count_before = 0
        self._frame_append_count_before = 0
        self._append_call_count = 0

        self.actual_contract: dict[str, Any] = {}
        self.expected_contract: dict[str, Any] = {}
        self.assertion_counts: dict[str, int] = {}
        self.violations: list[str] = []
        self.counts = {
            "frames_started": 0,
            "frames_accepted": 0,
            "motion_probe_rejected_frames": 0,
            "motion_probe_calls": 0,
            "graph_update_calls": 0,
            "factor_append_calls": 0,
            "factors_appended": 0,
            "factor_remove_calls": 0,
            "factors_removed": 0,
            "keyframes_culled": 0,
            "factor_update_observations": 0,
        }
        self._last_corr_center: torch.Tensor | None = None
        self._last_update_delta: torch.Tensor | None = None
        self._corr_levels: list[torch.Tensor] = []
        self._capture_corr_levels = False
        self._corr_validation_calls = 0
        self._hook_handles: list[Any] = []
        self._wrapped_names = (
            "corr",
            "motion_probe",
            "append_factors",
            "remove_factors",
            "keyframe",
            "update",
        )
        self._installed = False
        self.hooks_restored = False

        import dpvo.dpvo as dpvo_module

        self.dpvo_module = dpvo_module
        self._original_altcorr = dpvo_module.altcorr.corr
        self.corr_axis_microcheck = self._run_corr_axis_microcheck() if self.sanity_validation else {"ran": False, "passed": True}
        self._record_persistent_contract()
        self._install()

    def _count_assertion(self, name: str) -> None:
        self.assertion_counts[name] = self.assertion_counts.get(name, 0) + 1

    def _require(self, name: str, condition: bool, detail: str) -> None:
        self._count_assertion(name)
        if not bool(condition):
            message = f"{name}: {detail}"
            self.violations.append(message)
            raise AssertionError(message)

    def _soft_violation(self, name: str, detail: str) -> None:
        self._count_assertion(name)
        self.violations.append(f"{name}: {detail}")

    @staticmethod
    def _tensor_description(tensor: torch.Tensor) -> dict[str, Any]:
        result: dict[str, Any] = {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
        return result

    def _record_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if name not in self.actual_contract:
            self.actual_contract[name] = self._tensor_description(tensor)

    def _record_persistent_contract(self) -> None:
        persistent = {
            "fmap1_": self.slam.fmap1_,
            "fmap2_": self.slam.fmap2_,
            "gmap_": self.slam.gmap_,
            "imap_": self.slam.imap_,
        }
        for name, tensor in persistent.items():
            self._record_tensor(name, tensor)

    def _run_corr_axis_microcheck(self) -> dict[str, Any]:
        device = torch.device("cuda")
        radius = 3
        patch_size = 3
        channels = 8
        fmap1 = torch.zeros(1, 1, channels, patch_size, patch_size, device=device)
        fmap1[:, :, 0] = 1.0
        fmap2 = torch.zeros(1, 1, channels, 24, 24, device=device)
        yy, xx = torch.meshgrid(
            torch.arange(24, device=device),
            torch.arange(24, device=device),
            indexing="ij",
        )
        fmap2[0, 0, 0] = 100.0 * yy + xx
        coords = torch.empty(1, 1, 2, patch_size, patch_size, device=device)
        py, px = torch.meshgrid(
            torch.arange(patch_size, device=device),
            torch.arange(patch_size, device=device),
            indexing="ij",
        )
        coords[0, 0, 0] = 8.0 + px
        coords[0, 0, 1] = 8.0 + py
        index = torch.zeros(1, dtype=torch.long, device=device)
        output = self._original_altcorr(
            fmap1, fmap2, coords, index, index, radius, 1.0
        ).detach().float().cpu().numpy()[0, 0]

        candidates: list[dict[str, Any]] = []
        for corr_swapped in (False, True):
            for patch_swapped in (False, True):
                expected = np.empty_like(output)
                for axis0 in range(2 * radius + 1):
                    for axis1 in range(2 * radius + 1):
                        dx = (axis1 if corr_swapped else axis0) - radius
                        dy = (axis0 if corr_swapped else axis1) - radius
                        for axis2 in range(patch_size):
                            for axis3 in range(patch_size):
                                patch_y = axis3 if patch_swapped else axis2
                                patch_x = axis2 if patch_swapped else axis3
                                expected[axis0, axis1, axis2, axis3] = (
                                    100.0 * (8 + patch_y + dy) + (8 + patch_x + dx)
                                )
                candidates.append(
                    {
                        "corr_swapped": corr_swapped,
                        "patch_swapped": patch_swapped,
                        "max_abs_error": float(np.max(np.abs(output - expected))),
                    }
                )

        best = min(candidates, key=lambda item: item["max_abs_error"])
        passed = best["max_abs_error"] < 1e-4
        return {
            "passed": passed,
            "actual_cuda_shape": list(output.shape),
            "axis_order_if_passed": (
                ["neighborhood_y", "neighborhood_x", "patch_y", "patch_x"]
                if best["corr_swapped"]
                else ["neighborhood_x", "neighborhood_y", "patch_y", "patch_x"]
            )
            if not best["patch_swapped"]
            else (
                ["neighborhood_y", "neighborhood_x", "patch_x", "patch_y"]
                if best["corr_swapped"]
                else ["neighborhood_x", "neighborhood_y", "patch_x", "patch_y"]
            ),
            "best_candidate": best,
            "all_candidates": candidates,
            "stacked_level_axis": "last (verified separately against DPVO.corr)",
        }

    def _install(self) -> None:
        if self.sanity_validation:
            self._require(
                "corr_axis_microcheck",
                self.corr_axis_microcheck["passed"],
                str(self.corr_axis_microcheck),
            )
            self.dpvo_module.altcorr.corr = self._altcorr_wrapper
        self._hook_handles.append(
            self.slam.network.patchify.register_forward_hook(self._patchify_hook)
        )
        self._hook_handles.append(
            self.slam.network.update.register_forward_hook(self._update_hook)
        )

        wrappers: dict[str, Callable[..., Any]] = {
            "corr": self._corr_wrapper,
            "motion_probe": self._motion_probe_wrapper,
            "append_factors": self._append_factors_wrapper,
            "remove_factors": self._remove_factors_wrapper,
            "keyframe": self._keyframe_wrapper,
            "update": self._graph_update_wrapper,
        }
        self._original_methods = {
            name: getattr(self.slam, name) for name in self._wrapped_names
        }
        for name, wrapper in wrappers.items():
            setattr(self.slam, name, types.MethodType(wrapper, self.slam))
        self._installed = True
        self.events.write("probe_installed", corr_axis_microcheck=self.corr_axis_microcheck)

    def close(self) -> None:
        if not self._installed:
            return
        for handle in self._hook_handles:
            handle.remove()
        for name in self._wrapped_names:
            if name in self.slam.__dict__:
                delattr(self.slam, name)
        if self.sanity_validation:
            self.dpvo_module.altcorr.corr = self._original_altcorr
        self._installed = False
        self.hooks_restored = all(name not in self.slam.__dict__ for name in self._wrapped_names)
        self.events.write("probe_uninstalled", hooks_restored=self.hooks_restored)
        self.events.close()

    def __enter__(self) -> "DPVOTruthProbe":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def before_frame(
        self,
        *,
        stream_index: int,
        euroc_timestamp_ns: int,
        image_height: int,
        image_width: int,
        blur_laplacian: float | None = None,
        texture_gradient: float | None = None,
    ) -> None:
        dpvo_counter = int(self.slam.counter)
        self._require(
            "stream_index_equals_dpvo_counter",
            stream_index == dpvo_counter,
            f"stream_index={stream_index}, dpvo_counter={dpvo_counter}",
        )
        self.current_frame = {
            "stream_index": int(stream_index),
            "dpvo_counter": dpvo_counter,
            "euroc_timestamp_ns": int(euroc_timestamp_ns),
            "image_height": int(image_height),
            "image_width": int(image_width),
            "retained_n_before": int(self.slam.n),
        }
        self.current_phase = "frame"
        self._frame_factor_count_before = len(self.active_factor_uids)
        self._frame_append_count_before = self._append_call_count
        self.counts["frames_started"] += 1
        if self.behavior is not None:
            if blur_laplacian is None or texture_gradient is None:
                raise ValueError("Behavior mode requires image difficulty values")
            self.behavior.before_frame(
                stream_index=stream_index,
                euroc_timestamp_ns=euroc_timestamp_ns,
                blur_laplacian=blur_laplacian,
                texture_gradient=texture_gradient,
            )

    def after_frame(self) -> None:
        if self.current_frame is None:
            raise RuntimeError("after_frame called without before_frame")
        frame_id = self.current_frame["dpvo_counter"]
        retained_ids = [int(value) for value in self.slam.pg.tstamps_[: self.slam.n]]
        accepted = frame_id in retained_ids
        state = self.patch_states.get(frame_id)
        self._require(
            "patchifier_ran_once_per_frame",
            state is not None,
            f"missing patch state for frame {frame_id}",
        )
        state["accepted"] = accepted
        if accepted:
            state["residency_start"] = frame_id
            self.counts["frames_accepted"] += 1
        else:
            state["residency_lifetime"] = 0
            state["residency_end_reason"] = "motion_probe_rejected"
            self.counts["motion_probe_rejected_frames"] += 1
            self._require(
                "motion_reject_has_no_formal_factor_append",
                self._append_call_count == self._frame_append_count_before,
                f"frame {frame_id} appended formal factors",
            )
            self._require(
                "motion_reject_preserves_active_factor_count",
                len(self.active_factor_uids) == self._frame_factor_count_before,
                f"frame {frame_id} changed active factors",
            )

        self._require(
            "dpvo_counter_incremented",
            int(self.slam.counter) == frame_id + 1,
            f"counter after={self.slam.counter}, before={frame_id}",
        )
        self._require(
            "dpvo_tlist_is_internal_stream_index",
            int(self.slam.tlist[-1]) == self.current_frame["stream_index"],
            f"tlist={self.slam.tlist[-1]}",
        )
        self.events.write(
            "frame_complete",
            **self.current_frame,
            accepted=accepted,
            retained_n_after=int(self.slam.n),
            retained_frame_ids=retained_ids,
            active_factor_count=len(self.active_factor_uids),
        )
        if self.behavior is not None:
            self.behavior.after_frame(accepted)
        self.current_frame = None
        self.current_phase = "idle"

    def before_terminate(self) -> None:
        self.terminating = True
        self.current_phase = "termination"

    def after_terminate(self) -> None:
        self.terminating = False
        self.current_phase = "idle"
        final_frame = int(self.slam.counter) - 1
        for state in self.patch_states.values():
            if state.get("accepted") and "residency_end_reason" not in state:
                state["residency_end_reason"] = "run_end"
                state["residency_end"] = final_frame
                state["residency_lifetime"] = final_frame - state["residency_start"] + 1

    def _patchify_hook(self, module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        if self.current_frame is None:
            raise AssertionError("Patchifier ran outside a frame context")
        self._require("patchify_output_arity", len(output) == 6, f"len={len(output)}")
        fmap, gmap, imap, patches, index, _color = output
        frame = self.current_frame
        height = frame["image_height"]
        width = frame["image_width"]
        patches_per_frame = int(self.slam.M)
        patch_size = int(self.slam.P)
        expected = {
            "fmap": [1, 1, 128, height // 4, width // 4],
            "gmap": [1, patches_per_frame, 128, patch_size, patch_size],
            "imap": [1, patches_per_frame, 384, 1, 1],
            "patches": [1, patches_per_frame, 3, patch_size, patch_size],
        }
        self.expected_contract.update(expected)
        tensors = {"fmap": fmap, "gmap": gmap, "imap": imap, "patches": patches}
        for name, tensor in tensors.items():
            self._record_tensor(name, tensor)
            self._require(
                f"{name}_shape_contract",
                list(tensor.shape) == expected[name],
                f"actual={list(tensor.shape)}, expected={expected[name]}",
            )

        validate_every = int(self.probe_config.get("validate_finite_every", 1))
        if frame["stream_index"] % validate_every == 0:
            for name, tensor in tensors.items():
                self._require(
                    f"{name}_finite",
                    bool(torch.isfinite(tensor).all().item()),
                    f"non-finite values in frame {frame['stream_index']}",
                )

        center = patch_size // 2
        patch_center = patches[0, :, :, center, center].detach()
        x = patch_center[:, 0]
        y = patch_center[:, 1]
        self._require(
            "patch_x_bounds",
            bool(((x >= 1) & (x <= width // 4 - 2)).all().item()),
            f"x range=({x.min().item()}, {x.max().item()})",
        )
        self._require(
            "patch_y_bounds",
            bool(((y >= 1) & (y <= height // 4 - 2)).all().item()),
            f"y range=({y.min().item()}, {y.max().item()})",
        )
        self._require(
            "patchifier_index_contract",
            list(index.shape) == [patches_per_frame],
            f"index shape={list(index.shape)}",
        )

        frame_id = int(frame["dpvo_counter"])
        self.patch_states[frame_id] = {
            "accepted": None,
            "patch_count": patches_per_frame,
            "euroc_timestamp_ns": int(frame["euroc_timestamp_ns"]),
        }

        patch_np = patch_center.float().cpu().numpy()
        if self.behavior is not None:
            self.behavior.record_patches(frame_id, patch_np, height, width)
        if self.patch_sampler.capacity <= 0:
            return
        gmap_np = gmap[0, :, :, center, center].detach().cpu().numpy().astype(np.float16)
        imap_np = imap[0, :, :, 0, 0].detach().cpu().numpy().astype(np.float16)
        patch_uids = (np.uint64(frame_id) << np.uint64(32)) | np.arange(
            patches_per_frame, dtype=np.uint64
        )
        scores = splitmix64_array(patch_uids)
        for local_index in range(patches_per_frame):
            score = int(scores[local_index])
            if score <= self.patch_sampler.threshold:
                patch_uid = int(patch_uids[local_index])
                self.patch_sampler.add(
                    score,
                    patch_uid,
                    {
                        "patch_uid": patch_uid,
                        "dpvo_counter": frame_id,
                        "local_patch_index": local_index,
                        "patch_center": patch_np[local_index],
                        "gmap_center": gmap_np[local_index],
                        "imap": imap_np[local_index],
                    },
                )

    def _altcorr_wrapper(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        output = self._original_altcorr(*args, **kwargs)
        if self._capture_corr_levels:
            self._corr_levels.append(output)
        return output

    def _corr_wrapper(
        self,
        self_slam: Any,
        coords: torch.Tensor,
        indicies: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self._last_corr_center = coords[
            ..., self.slam.P // 2, self.slam.P // 2
        ].detach().clone()
        self._capture_corr_levels = self.sanity_validation and self._corr_validation_calls < 8
        self._corr_levels = []
        output = self._original_methods["corr"](coords, indicies)
        self._record_tensor("coords", coords)
        self._record_tensor("corr", output)
        self._require(
            "corr_flatten_contract",
            output.ndim == 3 and output.shape[-1] == 2 * 49 * self.slam.P * self.slam.P,
            f"corr shape={list(output.shape)}",
        )
        if self._capture_corr_levels:
            self._require(
                "corr_two_cuda_levels", len(self._corr_levels) == 2, f"levels={len(self._corr_levels)}"
            )
            for level_index, level in enumerate(self._corr_levels):
                self._record_tensor(f"corr_level_{level_index}", level)
            assembled = torch.stack(self._corr_levels, -1).reshape_as(output)
            self._require(
                "corr_level_stack_matches_flattened",
                bool(torch.equal(assembled, output)),
                f"max error={(assembled.float() - output.float()).abs().max().item()}",
            )
            self._corr_validation_calls += 1
        self._capture_corr_levels = False
        return output

    def _motion_probe_wrapper(self, self_slam: Any) -> torch.Tensor:
        previous_phase = self.current_phase
        factor_count = len(self.active_factor_uids)
        self.current_phase = "motion_probe"
        self.counts["motion_probe_calls"] += 1
        try:
            result = self._original_methods["motion_probe"]()
            self._require(
                "motion_probe_does_not_create_factors",
                len(self.active_factor_uids) == factor_count,
                "sidecar factor count changed",
            )
            if self.sanity_validation:
                self.events.write(
                    "motion_probe",
                    dpvo_counter=int(self.current_frame["dpvo_counter"]),
                    motion_quantile=float(result.item()),
                    formal_factor_count=len(self.active_factor_uids),
                )
            return result
        finally:
            self.current_phase = previous_phase

    def _append_factors_wrapper(
        self, self_slam: Any, patch_indices: torch.Tensor, target_indices: torch.Tensor
    ) -> None:
        patch_indices_cpu = patch_indices.detach().long().cpu().numpy()
        target_indices_cpu = target_indices.detach().long().cpu().numpy()
        source_indices = self.slam.ix[patch_indices].detach().long().cpu().numpy()
        pg_tstamps = self.slam.pg.tstamps_
        source_ids = np.asarray([int(pg_tstamps[index]) for index in source_indices])
        target_ids = np.asarray([int(pg_tstamps[index]) for index in target_indices_cpu])
        local_indices = patch_indices_cpu % int(self.slam.M)
        source_patch_uids = (
            source_ids.astype(np.uint64) << np.uint64(32)
        ) | local_indices.astype(np.uint64)

        result = self._original_methods["append_factors"](patch_indices, target_indices)
        count = len(patch_indices_cpu)
        first_uid = self.next_factor_uid
        factor_uids = list(range(first_uid, first_uid + count))
        self.next_factor_uid += count
        self.active_factor_uids.extend(factor_uids)
        self.active_source_patch_uids.extend(int(value) for value in source_patch_uids)
        self.active_target_frame_ids.extend(int(value) for value in target_ids)
        if self.behavior is not None:
            append_frame = (
                int(self.current_frame["dpvo_counter"])
                if self.current_frame is not None
                else int(self.slam.counter) - 1
            )
            self.behavior.append_factors(
                factor_uids, source_patch_uids, target_ids, append_frame
            )
        self._append_call_count += 1
        self.counts["factor_append_calls"] += 1
        self.counts["factors_appended"] += count
        self._assert_factor_sidecar("after_append")
        self.events.write(
            "append_factors",
            count=count,
            factor_uid_first=first_uid if count else None,
            factor_uid_last=(self.next_factor_uid - 1) if count else None,
            source_frame_ids=sorted(set(int(value) for value in source_ids)),
            target_frame_ids=sorted(set(int(value) for value in target_ids)),
        )
        return result

    def _remove_factors_wrapper(
        self, self_slam: Any, mask: torch.Tensor, store: bool
    ) -> None:
        mask_cpu = mask.detach().bool().cpu().numpy()
        self._require(
            "remove_mask_matches_sidecar",
            len(mask_cpu) == len(self.active_factor_uids),
            f"mask={len(mask_cpu)}, sidecar={len(self.active_factor_uids)}",
        )
        removed_uids = [
            uid for uid, remove in zip(self.active_factor_uids, mask_cpu) if remove
        ]
        removed_sources = [
            uid for uid, remove in zip(self.active_source_patch_uids, mask_cpu) if remove
        ]
        removed_targets = [
            uid for uid, remove in zip(self.active_target_frame_ids, mask_cpu) if remove
        ]
        if self.behavior is not None:
            self.behavior.remove_factors(
                mask_cpu, "window_inactive" if store else "keyframe_cull"
            )
        result = self._original_methods["remove_factors"](mask, store)
        keep = ~mask_cpu
        self.active_factor_uids = [
            uid for uid, retain in zip(self.active_factor_uids, keep) if retain
        ]
        self.active_source_patch_uids = [
            uid for uid, retain in zip(self.active_source_patch_uids, keep) if retain
        ]
        self.active_target_frame_ids = [
            uid for uid, retain in zip(self.active_target_frame_ids, keep) if retain
        ]
        self.counts["factor_remove_calls"] += 1
        self.counts["factors_removed"] += len(removed_uids)
        self._assert_factor_sidecar("after_remove")
        self.events.write(
            "remove_factors",
            reason="window_inactive" if store else "keyframe_cull",
            stored_as_inactive=bool(store),
            count=len(removed_uids),
            factor_uid_sample=removed_uids[:8],
            source_patch_uid_sample=removed_sources[:8],
            target_frame_id_sample=removed_targets[:8],
        )
        return result

    def _keyframe_wrapper(self, self_slam: Any) -> None:
        before_n = int(self.slam.n)
        before_ids = [int(value) for value in self.slam.pg.tstamps_[:before_n]]
        result = self._original_methods["keyframe"]()
        after_n = int(self.slam.n)
        after_ids = [int(value) for value in self.slam.pg.tstamps_[:after_n]]
        if after_n == before_n - 1:
            removed = [frame_id for frame_id in before_ids if frame_id not in set(after_ids)]
            self._require(
                "single_keyframe_culled", len(removed) == 1, f"removed={removed}"
            )
            removed_id = removed[0]
            expected_after = [frame_id for frame_id in before_ids if frame_id != removed_id]
            self._require(
                "keyframe_shift_preserves_stable_ids",
                after_ids == expected_after,
                f"before={before_ids}, after={after_ids}",
            )
            if removed_id in self.patch_states:
                state = self.patch_states[removed_id]
                state["residency_end_reason"] = "keyframe_cull"
                state["residency_end"] = int(self.slam.counter) - 1
                state["residency_lifetime"] = (
                    state["residency_end"] - state["residency_start"] + 1
                )
            self.counts["keyframes_culled"] += 1
            self.events.write(
                "keyframe_cull",
                removed_frame_id=removed_id,
                retained_ids_before=before_ids,
                retained_ids_after=after_ids,
            )
        else:
            self._require(
                "keyframe_n_change_contract",
                after_n == before_n,
                f"before_n={before_n}, after_n={after_n}",
            )
        self._assert_factor_sidecar("after_keyframe")
        return result

    def _graph_update_wrapper(self, self_slam: Any) -> None:
        previous_phase = self.current_phase
        self.current_phase = "termination_update" if self.terminating else "graph_update"
        self.update_id += 1
        self.counts["graph_update_calls"] += 1
        self._assert_factor_sidecar("before_graph_update")
        self._last_corr_center = None
        self._last_update_delta = None
        try:
            result = self._original_methods["update"]()
            self._assert_factor_sidecar("after_graph_update")
            if self._last_corr_center is not None and self._last_update_delta is not None:
                expected_target = self._last_corr_center + self._last_update_delta.float()
                self._require(
                    "target_equals_coords_center_plus_delta",
                    bool(torch.allclose(self.slam.pg.target, expected_target, rtol=1e-5, atol=1e-5)),
                    f"max error={(self.slam.pg.target - expected_target).abs().max().item()}",
                )
            return result
        finally:
            self.current_phase = previous_phase

    def _assert_factor_sidecar(self, stage: str) -> None:
        edge_count = int(self.slam.pg.ii.numel())
        lengths = {
            "factor_uid": len(self.active_factor_uids),
            "source_patch_uid": len(self.active_source_patch_uids),
            "target_frame_id": len(self.active_target_frame_ids),
        }
        self._require(
            f"factor_sidecar_length_{stage}",
            all(length == edge_count for length in lengths.values()),
            f"edges={edge_count}, sidecars={lengths}",
        )
        if edge_count:
            self._require(
                f"ii_equals_ix_kk_{stage}",
                bool(torch.equal(self.slam.pg.ii, self.slam.ix[self.slam.pg.kk])),
                "pg.ii differs from ix[pg.kk]",
            )
            ii_cpu = self.slam.pg.ii.detach().long().cpu().numpy()
            jj_cpu = self.slam.pg.jj.detach().long().cpu().numpy()
            kk_cpu = self.slam.pg.kk.detach().long().cpu().numpy()
            pg_tstamps = self.slam.pg.tstamps_
            current_source_frame_ids = np.asarray(
                [int(pg_tstamps[index]) for index in ii_cpu], dtype=np.uint64
            )
            current_source_patch_uids = (
                current_source_frame_ids << np.uint64(32)
            ) | (kk_cpu % int(self.slam.M)).astype(np.uint64)
            current_target_frame_ids = np.asarray(
                [int(pg_tstamps[index]) for index in jj_cpu], dtype=np.int64
            )
            self._require(
                f"source_patch_stable_id_{stage}",
                np.array_equal(
                    current_source_patch_uids,
                    np.asarray(self.active_source_patch_uids, dtype=np.uint64),
                ),
                "source patch sidecar differs from pg.tstamps_[ii] and kk % M",
            )
            self._require(
                f"target_frame_stable_id_{stage}",
                np.array_equal(
                    current_target_frame_ids,
                    np.asarray(self.active_target_frame_ids, dtype=np.int64),
                ),
                "target frame sidecar differs from pg.tstamps_[jj]",
            )

    def _update_hook(self, module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        net, context, corr, _flow, ii, jj, kk = inputs
        output_net, (delta, weight, _unused) = output
        edge_count = int(ii.numel())
        self._record_tensor("net", net)
        self._record_tensor("context", context)
        self._record_tensor("delta", delta)
        self._record_tensor("weight", weight)
        self._require(
            "update_shape_contract",
            list(net.shape) == [1, edge_count, 384]
            and list(context.shape) == [1, edge_count, 384]
            and list(output_net.shape) == [1, edge_count, 384]
            and list(delta.shape) == [1, edge_count, 2]
            and list(weight.shape) == [1, edge_count, 2]
            and list(corr.shape) == [1, edge_count, 882],
            f"net={list(net.shape)}, context={list(context.shape)}, corr={list(corr.shape)}, "
            f"delta={list(delta.shape)}, weight={list(weight.shape)}",
        )
        validate_now = self.current_frame is None or int(self.current_frame["stream_index"]) % int(self.probe_config.get("validate_finite_every", 25)) == 0
        if validate_now:
            self._require(
                "weight_range",
                bool(((weight >= 0) & (weight <= 1)).all().item()),
                "weight is outside [0, 1]",
            )
            self._require(
                "update_tensors_finite",
                bool(
                    torch.isfinite(corr).all().item()
                    and torch.isfinite(delta).all().item()
                    and torch.isfinite(weight).all().item()
                ),
                "non-finite corr/delta/weight",
            )
        self._require(
            "update_ii_equals_ix_kk",
            bool(torch.equal(ii, self.slam.ix[kk])),
            "ii differs from ix[kk]",
        )

        if self.current_phase == "motion_probe":
            primary_ids = (
                self.slam.pg.tstamps_[ii.detach().long().cpu().numpy()].astype(np.uint64)
                << np.uint64(32)
            ) | (kk.detach().long().cpu().numpy() % int(self.slam.M)).astype(np.uint64)
            target_ids = np.full(edge_count, int(self.current_frame["dpvo_counter"]), dtype=np.int64)
            factor_ids = np.full(edge_count, -1, dtype=np.int64)
            observation_update_id = -int(self.current_frame["dpvo_counter"]) - 1
        else:
            self._require(
                "update_matches_factor_sidecar",
                edge_count == len(self.active_factor_uids),
                f"edges={edge_count}, sidecar={len(self.active_factor_uids)}",
            )
            factor_ids = np.asarray(self.active_factor_uids, dtype=np.int64)
            primary_ids = factor_ids.astype(np.uint64)
            target_ids = np.asarray(self.active_target_frame_ids, dtype=np.int64)
            observation_update_id = self.update_id
            self.counts["factor_update_observations"] += edge_count
            self._last_update_delta = delta.detach().clone()

        if (
            self.behavior is not None
            and self.current_phase in {"graph_update", "termination_update"}
        ):
            center = int(self.slam.P) // 2
            frame_id = (
                int(self.current_frame["dpvo_counter"])
                if self.current_frame is not None
                else int(self.slam.counter) - 1
            )
            self.behavior.capture_update(
                phase=self.current_phase,
                frame_id=frame_id,
                factor_uids=factor_ids,
                source_patch_uids=np.asarray(
                    self.active_source_patch_uids, dtype=np.uint64
                ),
                target_frame_ids=target_ids,
                corr=corr,
                delta=delta,
                weight=weight,
                coords_center=self._last_corr_center,
                source_patch_center=self.slam.patches[
                    0, kk, :2, center, center
                ],
            )

        if edge_count and self.corr_sampler.capacity > 0:
            update_salt = np.uint64(
                (int(observation_update_id) * 0xD6E8FEB86659FD93)
                & ((1 << 64) - 1)
            )
            mixed = primary_ids ^ update_salt
            scores = splitmix64_array(mixed)
            batch_count = min(self.corr_sampler.capacity, edge_count)
            candidate_indices = np.argpartition(scores, batch_count - 1)[:batch_count]
            candidate_indices = candidate_indices[
                scores[candidate_indices] <= self.corr_sampler.threshold
            ]
            if len(candidate_indices):
                device_indices = torch.as_tensor(
                    candidate_indices, dtype=torch.long, device=corr.device
                )
                corr_np = corr[0, device_indices].detach().cpu().numpy().astype(np.float16)
                delta_np = delta[0, device_indices].detach().cpu().numpy().astype(np.float32)
                weight_np = weight[0, device_indices].detach().cpu().numpy().astype(np.float32)
                ii_np = ii[device_indices].detach().cpu().numpy()
                jj_np = jj[device_indices].detach().cpu().numpy()
                kk_np = kk[device_indices].detach().cpu().numpy()
                for row, edge_index in enumerate(candidate_indices):
                    primary_id = int(primary_ids[edge_index])
                    unique_key = (
                        (int(observation_update_id) & ((1 << 64) - 1)) << 64
                    ) | primary_id
                    self.corr_sampler.add(
                        int(scores[edge_index]),
                        unique_key,
                        {
                            "factor_uid": int(factor_ids[edge_index]),
                            "update_id": int(observation_update_id),
                            "phase": self.current_phase,
                            "target_dpvo_counter": int(target_ids[edge_index]),
                            "ii": int(ii_np[row]),
                            "jj": int(jj_np[row]),
                            "kk": int(kk_np[row]),
                            "corr": corr_np[row],
                            "delta": delta_np[row],
                            "weight": weight_np[row],
                        },
                    )

        if self.sanity_validation:
            self.events.write(
                "update_operator",
                phase=self.current_phase,
                update_id=int(observation_update_id),
                edge_count=edge_count,
                corr_shape=list(corr.shape),
            )

    def write_tensor_samples(self, path: str | Path) -> None:
        patch_values = self.patch_sampler.values()
        corr_values = self.corr_sampler.values()

        arrays: dict[str, np.ndarray] = {
            "patch_uid": np.asarray([value["patch_uid"] for value in patch_values], dtype=np.uint64),
            "patch_dpvo_counter": np.asarray(
                [value["dpvo_counter"] for value in patch_values], dtype=np.int64
            ),
            "local_patch_index": np.asarray(
                [value["local_patch_index"] for value in patch_values], dtype=np.int32
            ),
            "patch_center": np.asarray(
                [value["patch_center"] for value in patch_values], dtype=np.float32
            ).reshape(-1, 3),
            "gmap_center": np.asarray(
                [value["gmap_center"] for value in patch_values], dtype=np.float16
            ).reshape(-1, 128),
            "imap": np.asarray(
                [value["imap"] for value in patch_values], dtype=np.float16
            ).reshape(-1, 384),
            "factor_uid": np.asarray(
                [value["factor_uid"] for value in corr_values], dtype=np.int64
            ),
            "observation_update_id": np.asarray(
                [value["update_id"] for value in corr_values], dtype=np.int64
            ),
            "observation_phase": np.asarray(
                [value["phase"] for value in corr_values], dtype="U32"
            ),
            "target_dpvo_counter": np.asarray(
                [value["target_dpvo_counter"] for value in corr_values], dtype=np.int64
            ),
            "observation_ii": np.asarray(
                [value["ii"] for value in corr_values], dtype=np.int32
            ),
            "observation_jj": np.asarray(
                [value["jj"] for value in corr_values], dtype=np.int32
            ),
            "observation_kk": np.asarray(
                [value["kk"] for value in corr_values], dtype=np.int32
            ),
            "corr_flat": np.asarray(
                [value["corr"] for value in corr_values], dtype=np.float16
            ).reshape(-1, 882),
            "delta": np.asarray(
                [value["delta"] for value in corr_values], dtype=np.float32
            ).reshape(-1, 2),
            "weight": np.asarray(
                [value["weight"] for value in corr_values], dtype=np.float32
            ).reshape(-1, 2),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    def write_behavior_outputs(
        self,
        output_dir: str | Path,
        reference_thresholds: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.behavior is None:
            raise RuntimeError("Behavior outputs requested without behavior mode")
        return self.behavior.write_outputs(
            output_dir,
            self.patch_states,
            reference_thresholds=reference_thresholds,
        )

    def abort_behavior(self) -> None:
        if self.behavior is not None:
            self.behavior.abort()

    def summary(self) -> dict[str, Any]:
        patch_end_reasons: dict[str, int] = {}
        residency_lifetimes: list[int] = []
        for state in self.patch_states.values():
            reason = state.get("residency_end_reason", "unknown")
            patch_end_reasons[reason] = patch_end_reasons.get(reason, 0) + 1
            if "residency_lifetime" in state:
                residency_lifetimes.append(int(state["residency_lifetime"]))

        mh01_expected = {
            "fmap": [1, 1, 128, 120, 188],
            "gmap": [1, 96, 128, 3, 3],
            "imap": [1, 96, 384, 1, 1],
            "patches": [1, 96, 3, 3, 3],
            "fmap1_": [1, 36, 128, 120, 188],
            "fmap2_": [1, 36, 128, 30, 47],
            "gmap_": [36, 96, 128, 3, 3],
            "imap_": [36, 96, 384],
        }
        mh01_checks = {
            name: self.actual_contract.get(name, {}).get("shape") == shape
            for name, shape in mh01_expected.items()
        }
        return {
            "actual_tensor_contract": self.actual_contract,
            "generic_expected_contract": self.expected_contract,
            "mh01_expected_contract": mh01_expected,
            "mh01_shape_checks": mh01_checks,
            "corr_axis_microcheck": self.corr_axis_microcheck,
            "corr_level_stack_validation_calls": self._corr_validation_calls,
            "counts": self.counts,
            "active_factor_count_at_end": len(self.active_factor_uids),
            "next_factor_uid": self.next_factor_uid,
            "patch_residency": {
                "end_reasons_by_frame": patch_end_reasons,
                "lifetime_min": min(residency_lifetimes) if residency_lifetimes else None,
                "lifetime_max": max(residency_lifetimes) if residency_lifetimes else None,
                "definition": "accepted patch frame to keyframe cull or run end; motion rejects are zero",
            },
            "factor_observation_lifetime": {
                "scope": "aggregate behavior statistics",
                "total_factor_update_observations": self.counts["factor_update_observations"],
                "definition": "append/first update through last update/remove",
            },
            "samples": {
                "patch_count": len(self.patch_sampler.heap),
                "correlation_count": len(self.corr_sampler.heap),
                "method": "stable SplitMix64 bottom-k; no model RNG consumed",
            },
            "assertion_counts": self.assertion_counts,
            "violations": self.violations,
            "hooks_restored": self.hooks_restored,
            "event_count": self.events.count,
        }
