"""DPVO construction and visual-state wrappers used by formal Phase 1 runners."""
from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .oracle_packet import (
    NativeFeaturePacket, OracleFMap,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
