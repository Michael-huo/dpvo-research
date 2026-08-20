"""GPU worker and minimal research-only DPVO wrappers for final Experiment 3."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .oracle import NativeFeaturePacket, OracleFMap, extract_native_packet, extract_oracle_fmap


WORKER_PROGRESS: dict[str, Any] = {}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _records(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: dict[int, str] = {}
    with Path(config["paths"]["data_csv"]).open(encoding="utf-8") as handle:
        for row in csv.reader(line for line in handle if not line.startswith("#")):
            if row:
                rows[int(row[0])] = row[1]
    images = sorted(Path(config["paths"]["image_dir"]).glob("*.png"))
    exp = config["experiment"]
    images = images[int(exp["skip"])::int(exp["stride"])][:int(exp["processed_frames"])]
    result = []
    for index, image in enumerate(images):
        timestamp = int(image.stem)
        if rows.get(timestamp) != image.name:
            raise ValueError(f"EuRoC filename/data.csv mismatch: {image.name}")
        result.append({"candidate_index": index, "timestamp_ns": timestamp, "image_path": str(image)})
    if len(result) != int(exp["processed_frames"]):
        raise RuntimeError(f"expected {exp['processed_frames']} records, found {len(result)}")
    return result


def _load_frame(record: dict[str, Any], calibration: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    image = cv2.imread(record["image_path"])
    if image is None:
        raise FileNotFoundError(record["image_path"])
    fx, fy, cx, cy = calibration[:4]
    if len(calibration) > 4:
        matrix = np.eye(3)
        matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2] = fx, fy, cx, cy
        image = cv2.undistort(image, matrix, calibration[4:])
    height, width = image.shape[:2]
    image = image[:height - height % 16, :width - width % 16]
    return torch.from_numpy(image).permute(2, 0, 1).cuda(), torch.as_tensor([fx, fy, cx, cy], dtype=torch.float32, device="cuda")


def _dpvo_config(config: dict[str, Any]) -> Any:
    from dpvo.config import cfg as base_cfg

    cfg = base_cfg.clone()
    cfg.merge_from_file(config["paths"]["dpvo_config"])
    cfg.LOOP_CLOSURE = False
    cfg.CLASSIC_LOOP_CLOSURE = False
    return cfg


def source_patch_slots_are_valid(frame_kinds: list[str], patch_indices: list[int], patches_per_frame: int) -> bool:
    """Pure fixed-M contract used by the GPU wrapper and lightweight tests."""
    if patches_per_frame <= 0:
        raise ValueError("patches_per_frame must be positive")
    return all(0 <= index // patches_per_frame < len(frame_kinds) and frame_kinds[index // patches_per_frame] == "anchor" for index in patch_indices)


def _runtime_classes() -> dict[str, Any]:
    """Import DPVO lazily so analysis/tests remain CPU-only."""
    from dpvo.dpvo import DPVO, Id, fastba
    from dpvo.lietorch import SE3

    class NoKeyframeCullMixin:
        exp3_culling_policy = "no_keyframe_cull_with_upstream_factor_retirement"

        def keyframe(self) -> None:
            self.remove_factors(self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW, store=True)

    class NativePacketDPVO(DPVO):
        exp3_culling_policy = "upstream"

        def _on_frame_slot(self, slot: int, kind: str, timestamp_ns: int) -> None:
            del slot, kind, timestamp_ns

        def _on_frame_accepted(self, slot: int, kind: str, timestamp_ns: int) -> None:
            del slot, kind, timestamp_ns

        def _placeholder_patches(self, intrinsics: torch.Tensor) -> torch.Tensor:
            dtype = self.pg.patches_.dtype
            patches = torch.zeros(1, self.M, 3, self.P, self.P, dtype=dtype, device="cuda")
            center_x, center_y = intrinsics[2].to(dtype) / float(self.RES), intrinsics[3].to(dtype) / float(self.RES)
            offset = torch.arange(self.P, device="cuda", dtype=dtype) - self.P // 2
            yy, xx = torch.meshgrid(offset, offset, indexing="ij")
            patches[:, :, 0] = center_x + xx
            patches[:, :, 1] = center_y + yy
            depth = torch.as_tensor(1.0, device="cuda", dtype=dtype)
            if self.n:
                recent = self.pg.patches_[max(0, self.n - 3):self.n, :, 2]
                finite = recent[torch.isfinite(recent) & (recent > 0)]
                if finite.numel():
                    depth = finite.median()
            patches[:, :, 2] = depth
            if not bool(torch.isfinite(patches).all()):
                raise AssertionError("latent placeholder patches must be finite")
            return patches

        def track_packet(self, timestamp_ns: int, intrinsics: torch.Tensor, packet: NativeFeaturePacket | None = None, oracle: OracleFMap | None = None, *, kind: str = "anchor") -> tuple[bool, int | None]:
            if self.n + 1 >= self.N:
                raise RuntimeError(f"DPVO buffer too small: {self.N}")
            if (packet is None) == (oracle is None):
                raise ValueError("provide exactly one of packet or oracle")
            latent = oracle is not None
            fmap = oracle.fmap if latent else packet.fmap
            slot = int(self.n)
            self._on_frame_slot(slot, kind, int(timestamp_ns))
            self.tlist.append(int(timestamp_ns))
            self.pg.tstamps_[slot] = self.counter
            self.pg.intrinsics_[slot] = intrinsics / self.RES
            self.pg.index_[slot + 1] = slot + 1
            self.pg.index_map_[slot + 1] = self.m + self.M
            if self.n > 1:
                previous, before_previous = SE3(self.pg.poses_[self.n - 1]), SE3(self.pg.poses_[self.n - 2])
                *_, a, b, c = [1] * 3 + self.tlist
                xi = self.cfg.MOTION_DAMPING * ((c - b) / (b - a)) * (previous * before_previous.inv()).log()
                self.pg.poses_[slot] = (SE3.exp(xi) * previous).data
            if latent:
                patches = self._placeholder_patches(intrinsics)
                self.pg.colors_[slot].zero_()
                self.imap_[slot % self.pmem].zero_()
                self.gmap_[slot % self.pmem].zero_()
            else:
                patches = packet.patches.clone()
                colors = (packet.colors[0, :, [2, 1, 0]] + .5) * (255.0 / 2.0)
                self.pg.colors_[slot] = colors.to(torch.uint8)
                patches[:, :, 2] = torch.rand_like(patches[:, :, 2, 0, 0, None, None])
                if self.is_initialized:
                    patches[:, :, 2] = torch.median(self.pg.patches_[self.n - 3:self.n, :, 2])
                self.imap_[slot % self.pmem] = packet.imap.squeeze()
                self.gmap_[slot % self.pmem] = packet.gmap.squeeze()
            self.pg.patches_[slot] = patches
            self.fmap1_[:, slot % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
            self.fmap2_[:, slot % self.mem] = F.avg_pool2d(fmap[0], 4, 4)
            self.counter += 1
            if self.n > 0 and not self.is_initialized and self.motion_probe() < 2.0:
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return False, None
            self.n += 1
            self.m += self.M
            self._on_frame_accepted(slot, kind, int(timestamp_ns))
            self.append_factors(*self._DPVO__edges_forw())
            self.append_factors(*self._DPVO__edges_back())
            if self.n == 8 and not self.is_initialized:
                self.is_initialized = True
                for _ in range(12):
                    self.update()
            elif self.is_initialized:
                self.update()
                self.keyframe()
            return True, slot

    class NoCullRGBDPVO(NoKeyframeCullMixin, DPVO):
        pass

    class OracleHybridDPVO(NoKeyframeCullMixin, NativePacketDPVO):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.source_allowed = torch.zeros(self.N, dtype=torch.bool, device="cuda")
            self.frame_kind: dict[int, str] = {}
            self.latent_nodes: set[int] = set()
            self.incoming_by_latent: dict[int, int] = {}
            self.cumulative_kk: list[np.ndarray] = []
            self.cumulative_factor_count = 0
            self.latent_source_factor_count = 0
            self.placeholder_reference_violations = {key: 0 for key in ("active", "inactive", "cumulative", "correlation", "ba")}
            self.correlation_latent_factor_visits = 0
            self.correlation_latent_frames: set[int] = set()
            self.update_latent_factor_visits = 0
            self.update_latent_frames: set[int] = set()
            self.ba_latent_factor_visits = 0
            self.ba_latent_frames: set[int] = set()
            self.ba_successful_calls_with_latent = 0

        def _on_frame_slot(self, slot: int, kind: str, timestamp_ns: int) -> None:
            del timestamp_ns
            if kind not in {"anchor", "latent"}:
                raise ValueError(kind)
            self.frame_kind[slot] = kind
            self.source_allowed[slot] = kind == "anchor"

        def _on_frame_accepted(self, slot: int, kind: str, timestamp_ns: int) -> None:
            del timestamp_ns
            if kind == "latent":
                self.latent_nodes.add(slot)
                self.incoming_by_latent.setdefault(slot, 0)

        def _source_frames(self, kk: torch.Tensor) -> torch.Tensor:
            return self.ix[kk.long()]

        def _assert_factor_indices(self, kk: torch.Tensor, label: str) -> None:
            if not kk.numel():
                return
            source = self._source_frames(kk)
            fixed_m_source = torch.div(kk.long(), self.M, rounding_mode="floor")
            bad = (~self.source_allowed[source]) | (~self.source_allowed[fixed_m_source])
            count = int(bad.sum().item())
            if count:
                self.placeholder_reference_violations[label] += count
                self.latent_source_factor_count += count
                raise AssertionError(f"{label} factor kk references latent placeholder slots")

        def _filter_source_candidates(self, kk: torch.Tensor, jj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return (kk, jj) if not kk.numel() else (kk[self.source_allowed[self._source_frames(kk)]], jj[self.source_allowed[self._source_frames(kk)]])

        def _DPVO__edges_forw(self) -> tuple[torch.Tensor, torch.Tensor]:
            return self._filter_source_candidates(*DPVO._DPVO__edges_forw(self))

        def _DPVO__edges_back(self) -> tuple[torch.Tensor, torch.Tensor]:
            return self._filter_source_candidates(*DPVO._DPVO__edges_back(self))

        def append_factors(self, kk: torch.Tensor, jj: torch.Tensor) -> None:
            self._assert_factor_indices(kk, "cumulative")
            if kk.numel():
                latent_mask = torch.as_tensor([int(target) in self.latent_nodes for target in jj.detach().cpu().tolist()], dtype=torch.bool, device=jj.device)
                for target in jj[latent_mask].detach().cpu().tolist():
                    self.incoming_by_latent[int(target)] = self.incoming_by_latent.get(int(target), 0) + 1
                self.cumulative_kk.append(kk.detach().cpu().numpy().astype(np.int64, copy=True))
                self.cumulative_factor_count += int(kk.numel())
            super().append_factors(kk, jj)

        def corr(self, coords: torch.Tensor, indicies: Any = None) -> torch.Tensor:
            kk, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
            self._assert_factor_indices(kk, "correlation")
            targets = {int(value) for value in jj.detach().cpu().tolist()} & self.latent_nodes
            if targets:
                self.correlation_latent_frames.update(targets)
                self.correlation_latent_factor_visits += sum(int(value) in self.latent_nodes for value in jj.detach().cpu().tolist())
            return super().corr(coords, indicies)

        def update(self) -> None:
            self._assert_factor_indices(self.pg.kk, "active")
            targets = {int(value) for value in self.pg.jj.detach().cpu().tolist()} & self.latent_nodes
            if targets:
                self.update_latent_frames.update(targets)
                self.update_latent_factor_visits += sum(int(value) in self.latent_nodes for value in self.pg.jj.detach().cpu().tolist())
            original_ba = fastba.BA

            def checked_ba(*args: Any, **kwargs: Any) -> Any:
                jj = args[7] if len(args) > 7 else kwargs["jj"]
                kk = args[8] if len(args) > 8 else kwargs["kk"]
                self._assert_factor_indices(kk, "ba")
                result = original_ba(*args, **kwargs)
                latent_targets = {int(value) for value in jj.detach().cpu().tolist()} & self.latent_nodes
                if latent_targets:
                    self.ba_successful_calls_with_latent += 1
                    self.ba_latent_frames.update(latent_targets)
                    self.ba_latent_factor_visits += sum(int(value) in self.latent_nodes for value in jj.detach().cpu().tolist())
                return result

            fastba.BA = checked_ba
            try:
                super().update()
            finally:
                fastba.BA = original_ba

        def assert_factor_contract(self) -> None:
            self._assert_factor_indices(self.pg.kk, "active")
            self._assert_factor_indices(self.pg.kk_inac, "inactive")
            if self.cumulative_kk:
                self._assert_factor_indices(torch.from_numpy(np.concatenate(self.cumulative_kk)).to(device="cuda"), "cumulative")

        def diagnostics(self) -> dict[str, Any]:
            self.assert_factor_contract()
            counts = np.asarray([self.incoming_by_latent.get(node, 0) for node in sorted(self.latent_nodes)], dtype=np.int64)
            with_factors = int((counts > 0).sum())
            return {
                "latent_frame_count": len(self.latent_nodes),
                "latent_frames_with_state_node": len(self.latent_nodes),
                "latent_frames_receiving_factors": with_factors,
                "latent_factor_coverage": with_factors / len(self.latent_nodes) if self.latent_nodes else None,
                "incoming_factors_per_latent": {"mean": float(counts.mean()) if counts.size else None, "median": float(np.median(counts)) if counts.size else None, "min": int(counts.min()) if counts.size else None, "max": int(counts.max()) if counts.size else None},
                "anchor_source_to_latent_target_factor_count": int(counts.sum()),
                "latent_source_factor_count": int(self.latent_source_factor_count),
                "placeholder_reference_violations": dict(self.placeholder_reference_violations),
                "cumulative_factor_count": int(self.cumulative_factor_count),
                "correlation": {"latent_factor_visits": self.correlation_latent_factor_visits, "latent_frames": len(self.correlation_latent_frames)},
                "update": {"latent_factor_visits": self.update_latent_factor_visits, "latent_frames": len(self.update_latent_frames)},
                "ba": {"latent_factor_visits": self.ba_latent_factor_visits, "latent_frames": len(self.ba_latent_frames), "successful_calls_with_latent": self.ba_successful_calls_with_latent},
            }

    return {"DPVO": DPVO, "NativePacketDPVO": NativePacketDPVO, "NoCullRGBDPVO": NoCullRGBDPVO, "OracleHybridDPVO": OracleHybridDPVO}


def _make_slam(cls: Any, config: dict[str, Any], first_image: torch.Tensor) -> Any:
    return cls(_dpvo_config(config), config["paths"]["checkpoint"], ht=int(first_image.shape[1]), wd=int(first_image.shape[2]), viz=False)


def _new_peak_state(mode: str) -> dict[str, int]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    state = {"peak_active_graph_nodes": 0, "peak_active_factor_count": 0, "last_candidate_index": -1}
    WORKER_PROGRESS.clear()
    WORKER_PROGRESS.update({"mode": mode, **state, "peak_gpu_vram_bytes": int(torch.cuda.max_memory_allocated())})
    return state


def _observe_runtime(slam: Any, peaks: dict[str, int], candidate_index: int) -> None:
    peaks["peak_active_graph_nodes"] = max(peaks["peak_active_graph_nodes"], int(slam.n))
    peaks["peak_active_factor_count"] = max(peaks["peak_active_factor_count"], int(slam.pg.ii.numel()))
    peaks["last_candidate_index"] = int(candidate_index)
    WORKER_PROGRESS.update(peaks)
    WORKER_PROGRESS["peak_gpu_vram_bytes"] = int(torch.cuda.max_memory_allocated())


def _finish_run(slam: Any, *, started: float, uploaded: list[int], accepted: list[int], candidates: int, peaks: dict[str, int]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    active_before_terminate = int(slam.n)
    poses, returned_timestamps = slam.terminate()
    torch.cuda.synchronize()
    expected = np.asarray(slam.tlist, dtype=np.uint64)
    timestamps, pose_array = np.asarray(returned_timestamps, dtype=np.uint64), np.asarray(poses, dtype=np.float64)
    timestamp_contract = len(timestamps) == len(expected) and np.array_equal(timestamps, expected)
    valid_timestamps = bool(timestamp_contract and len(timestamps) > 0 and np.all(np.diff(timestamps.astype(np.int64)) > 0))
    finite = bool(np.isfinite(pose_array).all())
    if not finite or not valid_timestamps:
        raise AssertionError("trajectory finite/timestamp contract failed")
    return {
        "tracking_completed": True, "processed_candidate_frames": int(candidates), "rgb_uploaded_frames": len(uploaded), "actual_rgb_upload_ratio": len(uploaded) / candidates,
        "trajectory_pose_count": len(poses), "accepted_graph_input_count": len(accepted), "accepted_anchor_graph_count": len(accepted),
        "active_graph_nodes_before_terminate": active_before_terminate, "active_graph_nodes_after_terminate": int(slam.n),
        "uploaded_timestamps_ns": uploaded, "accepted_graph_timestamps_ns": accepted, "elapsed_seconds": time.perf_counter() - started,
        "patches_per_frame": int(slam.M), "culling_policy": getattr(slam, "exp3_culling_policy", "upstream"), "finite_trajectory": finite,
        "valid_timestamps": valid_timestamps, "timestamp_contract_exact": bool(timestamp_contract), "last_candidate_index": int(peaks["last_candidate_index"]),
        "peak_active_graph_nodes": int(peaks["peak_active_graph_nodes"]), "peak_active_factor_count": int(peaks["peak_active_factor_count"]), "peak_gpu_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }, {"poses": pose_array, "timestamps_ns": timestamps}


@torch.no_grad()
def run_worker(config: dict[str, Any], mode: str, schedule: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if mode not in {"full_rgb", "sparse_rgb", "oracle_fmap"}:
        raise ValueError(f"unsupported final Experiment 3 mode: {mode}")
    classes, records = _runtime_classes(), _records(config)
    calibration = np.loadtxt(config["paths"]["calibration"], delimiter=" ")
    first_image, _ = _load_frame(records[0], calibration)
    _seed_everything(int(config["experiment"]["seed"]))
    started, uploaded, accepted, accepted_indices = time.perf_counter(), [], [], []
    if mode == "full_rgb":
        slam, peaks = _make_slam(classes["DPVO"], config, first_image), _new_peak_state(mode)
        for record in records:
            image, intrinsics = _load_frame(record, calibration)
            before_n = int(slam.n)
            slam(int(record["timestamp_ns"]), image, intrinsics)
            uploaded.append(int(record["timestamp_ns"]))
            if int(slam.n) > before_n:
                accepted.append(int(record["timestamp_ns"]))
            _observe_runtime(slam, peaks, int(record["candidate_index"]))
        return _finish_run(slam, started=started, uploaded=uploaded, accepted=accepted, candidates=len(records), peaks=peaks)
    if mode == "sparse_rgb":
        slam, peaks = _make_slam(classes["NoCullRGBDPVO"], config, first_image), _new_peak_state(mode)
        bootstrap_end: int | None = None
        schedule_indices: list[int] = []
        interval = int(config["experiment"]["post_bootstrap_anchor_interval"])
        for record in records:
            candidate = int(record["candidate_index"])
            upload = bootstrap_end is None or (candidate - bootstrap_end - 1) % interval == 0
            if not upload:
                _observe_runtime(slam, peaks, candidate)
                continue
            image, intrinsics = _load_frame(record, calibration)
            before_n = int(slam.n)
            slam(int(record["timestamp_ns"]), image, intrinsics)
            uploaded.append(int(record["timestamp_ns"])); schedule_indices.append(candidate)
            if int(slam.n) > before_n:
                accepted.append(int(record["timestamp_ns"])); accepted_indices.append(candidate)
            if bootstrap_end is None and slam.is_initialized:
                bootstrap_end = candidate
            _observe_runtime(slam, peaks, candidate)
        if bootstrap_end is None:
            raise RuntimeError("Sparse RGB did not initialize")
        result, arrays = _finish_run(slam, started=started, uploaded=uploaded, accepted=accepted, candidates=len(records), peaks=peaks)
        result.update({"bootstrap_end_candidate_index": bootstrap_end, "bootstrap_condition": "motion-accepted graph n == 8", "scheduled_anchor_candidate_indices": schedule_indices, "scheduled_anchor_timestamps_ns": uploaded, "accepted_anchor_candidate_indices": accepted_indices})
        return result, arrays
    if schedule is None:
        raise ValueError("oracle_fmap requires the immutable Sparse RGB schedule")
    schedule_indices = [int(value) for value in schedule["candidate_indices"]]
    schedule_set, expected_bootstrap_end = set(schedule_indices), int(schedule["bootstrap_end_candidate_index"])
    slam, peaks = _make_slam(classes["OracleHybridDPVO"], config, first_image), _new_peak_state(mode)
    actual_bootstrap_end: int | None = None
    latent_timestamps: list[int] = []
    for record in records:
        candidate, timestamp = int(record["candidate_index"]), int(record["timestamp_ns"])
        image, intrinsics = _load_frame(record, calibration)
        if candidate in schedule_set:
            if candidate > expected_bootstrap_end and not slam.is_initialized:
                raise AssertionError("Oracle bootstrap diverged before Sparse schedule began")
            packet = extract_native_packet(slam, image)
            initialized_before = bool(slam.is_initialized)
            accepted_now, _ = slam.track_packet(timestamp, intrinsics, packet=packet, kind="anchor")
            uploaded.append(timestamp)
            if accepted_now:
                accepted.append(timestamp); accepted_indices.append(candidate)
            if not initialized_before and slam.is_initialized:
                actual_bootstrap_end = candidate
            if candidate == expected_bootstrap_end and not slam.is_initialized:
                raise AssertionError("Oracle bootstrap end does not match Sparse RGB")
        else:
            if not slam.is_initialized:
                raise AssertionError("latent frame encountered before initialization")
            latent_timestamps.append(timestamp)
            accepted_now, _ = slam.track_packet(timestamp, intrinsics, oracle=extract_oracle_fmap(slam, image), kind="latent")
            if not accepted_now:
                raise AssertionError("post-bootstrap latent was not accepted")
        _observe_runtime(slam, peaks, candidate)
    if actual_bootstrap_end != expected_bootstrap_end:
        raise AssertionError(f"Oracle bootstrap end mismatch: {actual_bootstrap_end} != {expected_bootstrap_end}")
    diagnostics = slam.diagnostics()
    result, arrays = _finish_run(slam, started=started, uploaded=uploaded, accepted=accepted, candidates=len(records), peaks=peaks)
    result.update({"accepted_graph_input_count": len(accepted) + len(latent_timestamps), "bootstrap_end_candidate_index": actual_bootstrap_end, "shared_schedule_bootstrap_end_candidate_index": expected_bootstrap_end, "bootstrap_condition": "motion-accepted graph n == 8", "scheduled_anchor_candidate_indices": schedule_indices, "scheduled_anchor_timestamps_ns": uploaded, "accepted_anchor_candidate_indices": accepted_indices, "latent_timestamps_ns": latent_timestamps, "diagnostics": diagnostics, "placeholder_contract": {"slots_per_latent_frame": int(slam.M), "xy": "repeated 3x3 grid centered at optical center; independent of hidden RGB", "depth": "positive finite median from graph state, fallback 1.0", "gmap": "all zeros", "imap": "all zeros", "colors": "all zeros", "factor_eligible": False}})
    return result, arrays


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--internal-worker", action="store_true")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--mode", required=True, choices=("full_rgb", "sparse_rgb", "oracle_fmap"))
    parser.add_argument("--schedule-json")
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--result-npz", required=True)
    args = parser.parse_args()
    if not args.internal_worker:
        raise SystemExit("runtime is internal; use exp3.run_exp3")
    try:
        config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        schedule = json.loads(Path(args.schedule_json).read_text(encoding="utf-8")) if args.schedule_json else None
        result, arrays = run_worker(config, args.mode, schedule)
        _write_json(args.result_json, {"status": "ok", "mode": args.mode, **result})
        np.savez_compressed(args.result_npz, **arrays)
        return 0
    except BaseException as error:
        _write_json(args.result_json, {"status": "error", "mode": args.mode, "error": repr(error), "traceback": traceback.format_exc(), "progress": dict(WORKER_PROGRESS)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
