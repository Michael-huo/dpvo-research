"""DPVO construction and visual-state wrappers used by formal feasibility runners."""
from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from latent_vslam.oracle_packet import NativeFeaturePacket


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


def _runtime_classes() -> dict[str, Any]:
    """Import DPVO lazily so analysis/tests remain CPU-only."""
    from dpvo.dpvo import DPVO, Id
    from dpvo.lietorch import SE3

    class NativePacketDPVO(DPVO):
        def track_packet(self, timestamp_ns: int, intrinsics: torch.Tensor, packet: NativeFeaturePacket, *, kind: str = "anchor") -> tuple[bool, int | None]:
            if self.n + 1 >= self.N:
                raise RuntimeError(f"DPVO buffer too small: {self.N}")
            fmap = packet.fmap
            slot = int(self.n)
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

    return {"DPVO": DPVO, "NativePacketDPVO": NativePacketDPVO}


def _make_slam(cls: Any, config: dict[str, Any], first_image: torch.Tensor) -> Any:
    return cls(_dpvo_config(config), config["paths"]["checkpoint"], ht=int(first_image.shape[1]), wd=int(first_image.shape[2]), viz=False)
