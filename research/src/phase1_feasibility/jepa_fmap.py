"""Coordinate-correct V-JEPA block-5 to DPVO FMap bridge."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .protocol import FrameIdentity, canonical_sha256

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
COORDINATE_PROTOCOL_VERSION = "exp6_2_full_fov_coordinate_v1"
BRIDGE_ARCHITECTURE = "block5_coordinate_correct_shallow_spatial_v1"


@dataclass(frozen=True)
class FullFOVTransform:
    source_height: int
    source_width: int
    target_height: int
    resized_height: int
    resized_width: int
    padded_height: int
    padded_width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    scale_x: float
    scale_y: float
    patch_size: int
    token_grid_height: int
    token_grid_width: int
    fmap_height: int
    fmap_width: int
    fmap_scale: int
    coordinate_convention: str = "continuous_pixel_boundary_centers"
    interpolation: str = "opencv_linear"
    padding_value: str = "imagenet_mean_rgb"
    coordinate_protocol_version: str = COORDINATE_PROTOCOL_VERSION

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["image_to_jepa_affine"] = [
            [self.scale_x, 0.0, float(self.pad_left)],
            [0.0, self.scale_y, float(self.pad_top)],
            [0.0, 0.0, 1.0],
        ]
        result["jepa_to_image_affine"] = [
            [1.0 / self.scale_x, 0.0, -self.pad_left / self.scale_x],
            [0.0, 1.0 / self.scale_y, -self.pad_top / self.scale_y],
            [0.0, 0.0, 1.0],
        ]
        result["transform_sha256"] = canonical_sha256(result)
        return result


def coordinate_protocol_metadata(*, target_height: int, patch_size: int,
                                 fmap_scale: int) -> dict[str, Any]:
    """Return size-independent metadata suitable for the bridge checkpoint."""
    return {
        "version": COORDINATE_PROTOCOL_VERSION,
        "training_geometry_policy": "derive_from_mh01_dpvo_domain_at_fresh_run",
        "dpvo_domain": "opencv_bgr_undistort_then_bottom_right_crop_to_multiple_of_16",
        "jepa_color_domain": "rgb_imagenet_normalized",
        "resize": "aspect_preserving_opencv_linear_fixed_target_height",
        "target_height": int(target_height),
        "padding": "symmetric_to_patch_multiple_with_imagenet_mean_rgb",
        "patch_size": int(patch_size),
        "coordinate_convention": "continuous_pixel_boundary_centers",
        "mapping": "explicit_dpvo_cell_center_to_jepa_token_center",
        "fmap_scale": int(fmap_scale),
    }


def derive_full_fov_transform(
    source_height: int, source_width: int, *, target_height: int = 384,
    patch_size: int = 16, fmap_scale: int = 4,
) -> FullFOVTransform:
    if min(source_height, source_width, target_height, patch_size, fmap_scale) <= 0:
        raise ValueError("transform dimensions must be positive")
    resized_height = int(target_height)
    resized_width = int(round(source_width * target_height / source_height))
    padded_height = int(math.ceil(resized_height / patch_size) * patch_size)
    padded_width = int(math.ceil(resized_width / patch_size) * patch_size)
    pad_y, pad_x = padded_height - resized_height, padded_width - resized_width
    pad_top, pad_left = pad_y // 2, pad_x // 2
    if source_height % fmap_scale or source_width % fmap_scale:
        raise ValueError("DPVO domain must be divisible by fmap scale")
    return FullFOVTransform(
        source_height, source_width, target_height,
        resized_height, resized_width, padded_height, padded_width,
        pad_top, pad_y - pad_top, pad_left, pad_x - pad_left,
        resized_width / source_width, resized_height / source_height,
        patch_size, padded_height // patch_size, padded_width // patch_size,
        source_height // fmap_scale, source_width // fmap_scale, fmap_scale,
    )


def preprocess_full_fov_rgb(
    rgb: np.ndarray, transform: FullFOVTransform,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    if rgb.dtype != np.uint8 or rgb.shape != (
        transform.source_height, transform.source_width, 3,
    ):
        raise ValueError("full-FOV input must be RGB uint8 in the DPVO image domain")
    resized = cv2.resize(
        rgb, (transform.resized_width, transform.resized_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32) / 255.0
    padded = np.empty((transform.padded_height, transform.padded_width, 3), np.float32)
    padded[...] = IMAGENET_MEAN
    ys = slice(transform.pad_top, transform.pad_top + transform.resized_height)
    xs = slice(transform.pad_left, transform.pad_left + transform.resized_width)
    padded[ys, xs] = resized
    normalized = (padded - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(1).contiguous()
    return tensor, coordinate_masks(transform)


def fmap_sampling_grid(transform: FullFOVTransform,
                       output_hw: tuple[int, int]) -> np.ndarray:
    height, width = (int(value) for value in output_hw)
    if min(height, width) <= 0:
        raise ValueError("decoder output grid must be positive")
    cell_y = transform.source_height / height
    cell_x = transform.source_width / width
    image_x = (np.arange(width, dtype=np.float64) + 0.5) * cell_x
    image_y = (np.arange(height, dtype=np.float64) + 0.5) * cell_y
    jepa_x = transform.scale_x * image_x + transform.pad_left
    jepa_y = transform.scale_y * image_y + transform.pad_top
    token_x = jepa_x / transform.patch_size - 0.5
    token_y = jepa_y / transform.patch_size - 0.5
    gx, gy = np.meshgrid(token_x, token_y)
    return np.stack((gx, gy), axis=-1)


def coordinate_masks(transform: FullFOVTransform) -> dict[str, np.ndarray]:
    fractions = np.zeros((transform.token_grid_height, transform.token_grid_width), np.float32)
    x0, x1 = transform.pad_left, transform.pad_left + transform.resized_width
    y0, y1 = transform.pad_top, transform.pad_top + transform.resized_height
    for row in range(transform.token_grid_height):
        for col in range(transform.token_grid_width):
            tx0, ty0 = col * transform.patch_size, row * transform.patch_size
            tx1, ty1 = tx0 + transform.patch_size, ty0 + transform.patch_size
            overlap = max(0, min(tx1, x1) - max(tx0, x0)) * max(
                0, min(ty1, y1) - max(ty0, y0)
            )
            fractions[row, col] = overlap / float(transform.patch_size ** 2)
    grid = fmap_sampling_grid(transform, (transform.fmap_height, transform.fmap_width))
    image_x = (
        grid[..., 0] * transform.patch_size + transform.patch_size / 2 - transform.pad_left
    ) / transform.scale_x
    image_y = (
        grid[..., 1] * transform.patch_size + transform.patch_size / 2 - transform.pad_top
    ) / transform.scale_y
    fmap_valid = ((image_x >= 0) & (image_x < transform.source_width)
                  & (image_y >= 0) & (image_y < transform.source_height))
    return {
        "token_content_fraction": fractions,
        "valid_token_mask": fractions > 0,
        "fmap_valid_mask": fmap_valid,
        "token_centers_x": ((np.arange(transform.token_grid_width) + 0.5)
                            * transform.patch_size).astype(np.float32),
        "token_centers_y": ((np.arange(transform.token_grid_height) + 0.5)
                            * transform.patch_size).astype(np.float32),
        "fmap_sampling_grid_token_indices": grid.astype(np.float32),
    }


def normalized_sampling_grid(
    transform: FullFOVTransform, output_hw: tuple[int, int], *,
    device: torch.device | None = None,
) -> torch.Tensor:
    indices = fmap_sampling_grid(transform, output_hw)
    x = 2.0 * indices[..., 0] / (transform.token_grid_width - 1) - 1.0
    y = 2.0 * indices[..., 1] / (transform.token_grid_height - 1) - 1.0
    return torch.as_tensor(
        np.stack((x, y), -1), dtype=torch.float32, device=device,
    ).unsqueeze(0)


def warp_token_field(field: torch.Tensor, transform: FullFOVTransform,
                     output_hw: tuple[int, int]) -> torch.Tensor:
    grid = normalized_sampling_grid(transform, output_hw, device=field.device)
    return F.grid_sample(
        field, grid.expand(field.shape[0], -1, -1, -1), mode="bilinear",
        padding_mode="border", align_corners=True,
    )


class LocalResidualBlock(nn.Module):
    def __init__(self, channels: int = 160) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(16, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(16, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


def tokens_to_field(tokens: torch.Tensor, transform: FullFOVTransform) -> torch.Tensor:
    expected = transform.token_grid_height * transform.token_grid_width
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (expected, 768):
        raise ValueError(f"expected [B,{expected},768] JEPA tokens, got {tuple(tokens.shape)}")
    return tokens.float().reshape(
        -1, transform.token_grid_height, transform.token_grid_width, 768,
    ).permute(0, 3, 1, 2).contiguous()


class ShallowSpatialDecoder(nn.Module):
    """Spatial decoder whose weights are independent of a concrete image size."""

    def __init__(self, transform: FullFOVTransform, channels: int = 160) -> None:
        super().__init__()
        self.transform = transform
        self.input = nn.Sequential(
            nn.Conv2d(768, channels, 1), nn.GroupNorm(16, channels), nn.GELU(),
        )
        self.token_blocks = nn.Sequential(
            LocalResidualBlock(channels), LocalResidualBlock(channels),
        )
        self.mid_block = LocalResidualBlock(channels)
        self.full_block = LocalResidualBlock(channels)
        self.output = nn.Conv2d(channels, 128, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        fmap_hw = (self.transform.fmap_height, self.transform.fmap_width)
        mid_hw = (max(1, fmap_hw[0] // 2), max(1, fmap_hw[1] // 2))
        field = self.token_blocks(self.input(tokens_to_field(tokens, self.transform)))
        field = self.mid_block(warp_token_field(field, self.transform, mid_hw))
        field = self.full_block(warp_token_field(field, self.transform, fmap_hw))
        return self.output(field)


def build_bridge(transform: FullFOVTransform, *, channels: int = 160) -> ShallowSpatialDecoder:
    model = ShallowSpatialDecoder(transform, channels=channels)
    if sum(item.numel() for item in model.parameters()) > 3_000_000:
        raise AssertionError("shallow bridge exceeds parameter contract")
    return model


def masked_reconstruction_loss(
    prediction: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if prediction.shape != teacher.shape or prediction.ndim != 4:
        raise ValueError("prediction and teacher must be equal [B,C,H,W]")
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim != 3 or tuple(mask.shape[-2:]) != tuple(prediction.shape[-2:]):
        raise ValueError("valid mask shape mismatch")
    valid = mask.to(device=prediction.device, dtype=torch.bool).expand(
        prediction.shape[0], -1, -1,
    )
    pred_cells = prediction.float().permute(0, 2, 3, 1)[valid]
    true_cells = teacher.float().permute(0, 2, 3, 1)[valid]
    if not pred_cells.numel():
        raise ValueError("valid mask selects no cells")
    cosine = 1.0 - F.cosine_similarity(
        pred_cells, true_cells, dim=-1, eps=1e-8,
    ).mean()
    smooth_l1 = F.smooth_l1_loss(pred_cells, true_cells)
    return {"cosine": cosine, "smooth_l1": smooth_l1,
            "total": cosine + 0.1 * smooth_l1}


def contiguous_split(identities: Sequence[FrameIdentity]) -> dict[str, Any]:
    count = len(identities)
    if count < 3:
        raise ValueError("contiguous split needs at least three candidates")
    train_end, validation_end = int(count * 0.60), int(count * 0.80)
    ranges = {
        "train": (0, train_end), "validation": (train_end, validation_end),
        "test": (validation_end, count),
    }
    result = {
        name: {
            "candidate_start_inclusive": start,
            "candidate_end_exclusive": end,
            "count": end - start,
            "timestamp_start_ns": identities[start].timestamp_ns,
            "timestamp_end_ns": identities[end - 1].timestamp_ns,
        }
        for name, (start, end) in ranges.items()
    }
    result["split_sha256"] = canonical_sha256(result)
    return result


def hidden_split_keys(
    identities: Sequence[FrameIdentity], roles: Mapping[str, str],
    split: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    return {
        name: tuple(
            item.key for item in identities[
                int(split[name]["candidate_start_inclusive"]):
                int(split[name]["candidate_end_exclusive"])
            ] if roles[item.key] == "hidden"
        )
        for name in ("train", "validation", "test")
    }


def identity_shuffle(keys: Sequence[str], seed: int) -> dict[str, str]:
    if len(keys) < 2:
        raise ValueError("identity shuffle requires at least two keys")
    shift = 1 + int(seed) % (len(keys) - 1)
    return {key: keys[(index + shift) % len(keys)] for index, key in enumerate(keys)}
