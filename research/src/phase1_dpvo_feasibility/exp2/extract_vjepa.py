#!/usr/bin/env python3
"""Extract fixed V-JEPA dense tokens for the formal correspondence experiment.

This file is intentionally executed by path with the V-JEPA environment.  It
loads the sibling repository as the only ``research`` package provider, which
avoids collisions with dpvo-research's namespace package.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation, Slerp


SCRIPT_PATH = Path(__file__).resolve()
DPVO_ROOT = SCRIPT_PATH.parents[4]
CONFIG_PATH = DPVO_ROOT / "research/configs/phase1_exp2.yaml"
PROFILE_NAME = "formal"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phase1-exp2")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_matrix(calibration: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = calibration[:4]
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def undistorted_to_raw(points_xy: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    matrix = camera_matrix(calibration)
    normalized = np.column_stack(
        ((points[:, 0] - matrix[0, 2]) / matrix[0, 0],
         (points[:, 1] - matrix[1, 2]) / matrix[1, 1],
         np.ones(len(points), dtype=np.float64))
    )
    projected, _ = cv2.projectPoints(
        normalized, np.zeros(3), np.zeros(3), matrix,
        np.asarray(calibration[4:], dtype=np.float64),
    )
    return projected.reshape(-1, 2)


def raw_to_undistorted(points_xy: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    matrix = camera_matrix(calibration)
    return cv2.undistortPoints(
        points, matrix, np.asarray(calibration[4:], dtype=np.float64), P=matrix
    ).reshape(-1, 2)


def raw_to_token(points_xy: np.ndarray, metadata: dict[str, Any], patch_size: int) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    original_h, original_w = metadata["original_size_hw"]
    resized_h, resized_w = metadata["resized_size_hw"]
    left, top, _, _ = metadata["crop_box_xyxy"]
    crop_x = (points[:, 0] + 0.5) * resized_w / original_w - 0.5 - left
    crop_y = (points[:, 1] + 0.5) * resized_h / original_h - 0.5 - top
    return np.column_stack(((crop_x + 0.5) / patch_size - 0.5,
                            (crop_y + 0.5) / patch_size - 0.5))


def token_to_raw(points_xy: np.ndarray, metadata: dict[str, Any], patch_size: int) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    original_h, original_w = metadata["original_size_hw"]
    resized_h, resized_w = metadata["resized_size_hw"]
    left, top, _, _ = metadata["crop_box_xyxy"]
    resized_x = (points[:, 0] + 0.5) * patch_size - 0.5 + left
    resized_y = (points[:, 1] + 0.5) * patch_size - 0.5 + top
    raw_x = (resized_x + 0.5) * original_w / resized_w - 0.5
    raw_y = (resized_y + 0.5) * original_h / resized_h - 0.5
    return np.column_stack((raw_x, raw_y))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def fundamental_from_relative(relative: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    rotation = relative[:3, :3]
    translation = relative[:3, 3]
    essential = _skew(translation) @ rotation
    inverse = np.linalg.inv(matrix)
    return inverse.T @ essential @ inverse


def point_line_distance(point_xy: np.ndarray, line: np.ndarray) -> float:
    denominator = math.hypot(float(line[0]), float(line[1]))
    if denominator <= 1e-12:
        return float("nan")
    homogeneous = np.asarray([point_xy[0], point_xy[1], 1.0], dtype=np.float64)
    return float(abs(homogeneous @ line) / denominator)


def synthetic_geometry_consistency(matrix: np.ndarray, body_to_camera: np.ndarray) -> dict[str, Any]:
    # T_WC = T_WB @ T_BC, and relative maps source-camera points to target camera.
    source_body = np.eye(4, dtype=np.float64)
    source_body[:3, :3] = Rotation.from_rotvec([0.08, -0.03, 0.04]).as_matrix()
    source_body[:3, 3] = [0.4, -0.2, 0.1]
    target_body = np.eye(4, dtype=np.float64)
    target_body[:3, :3] = Rotation.from_rotvec([-0.02, 0.06, -0.05]).as_matrix()
    target_body[:3, 3] = [0.65, -0.1, 0.16]
    source_camera = source_body @ body_to_camera
    target_camera = target_body @ body_to_camera
    relative = np.linalg.inv(target_camera) @ source_camera

    source_pixel = np.asarray([314.2, 207.7], dtype=np.float64)
    ray = np.linalg.inv(matrix) @ np.asarray([*source_pixel, 1.0])
    source_point = ray * 4.7
    target_point = relative[:3, :3] @ source_point + relative[:3, 3]
    if target_point[2] <= 0:
        raise AssertionError("Synthetic target point is behind the camera")
    target_h = matrix @ target_point
    target_pixel = target_h[:2] / target_h[2]
    fundamental = fundamental_from_relative(relative, matrix)
    line = fundamental @ np.asarray([*source_pixel, 1.0])
    error = point_line_distance(target_pixel, line)
    if not np.isfinite(error) or error > 1e-8:
        raise AssertionError(f"Synthetic epipolar consistency failed: {error}")
    return {
        "formula": "T_Ct_Cs = inv(T_WC_t) @ T_WC_s",
        "composition": "T_WC = T_WB @ T_BC",
        "source_pixel": source_pixel,
        "target_projection": target_pixel,
        "point_line_error_pixels": error,
        "passed": True,
    }


class GroundTruth:
    def __init__(self, path: Path):
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                rows.append(line.split())
        self.timestamps = np.asarray([int(round(float(row[0]))) for row in rows], dtype=np.int64)
        self.positions = np.asarray([[float(value) for value in row[1:4]] for row in rows], dtype=np.float64)
        quaternion_wxyz = np.asarray([[float(value) for value in row[4:8]] for row in rows], dtype=np.float64)
        self.base = int(self.timestamps[0])
        seconds = (self.timestamps - self.base).astype(np.float64) * 1e-9
        self.rotations = Rotation.from_quat(quaternion_wxyz[:, [1, 2, 3, 0]])
        self.slerp = Slerp(seconds, self.rotations)

    def pose(self, timestamp_ns: int) -> np.ndarray:
        if timestamp_ns < self.timestamps[0] or timestamp_ns > self.timestamps[-1]:
            raise ValueError(f"Timestamp {timestamp_ns} outside GT range")
        seconds = (int(timestamp_ns) - self.base) * 1e-9
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self.slerp([seconds]).as_matrix()[0]
        for axis in range(3):
            pose[axis, 3] = np.interp(timestamp_ns, self.timestamps, self.positions[:, axis])
        return pose


def relative_camera_pose(
    gt: GroundTruth, source_timestamp_ns: int, target_timestamp_ns: int,
    body_to_camera: np.ndarray,
) -> np.ndarray:
    source_camera = gt.pose(source_timestamp_ns) @ body_to_camera
    target_camera = gt.pose(target_timestamp_ns) @ body_to_camera
    return np.linalg.inv(target_camera) @ source_camera


def _load_body_to_camera(sensor_yaml: Path) -> np.ndarray:
    payload = yaml.safe_load(sensor_yaml.read_text(encoding="utf-8"))
    transform = payload["T_BS"]
    if transform["rows"] != 4 or transform["cols"] != 4:
        raise ValueError(f"Unexpected T_BS shape in {sensor_yaml}")
    return np.asarray(transform["data"], dtype=np.float64).reshape(4, 4)


def bilinear_sample(grid: np.ndarray, token_xy: np.ndarray) -> np.ndarray:
    x, y = map(float, token_xy)
    height, width, _ = grid.shape
    if not (0.0 <= x <= width - 1 and 0.0 <= y <= height - 1):
        raise ValueError(f"Source token coordinate outside grid: {(x, y)}")
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    wx, wy = x - x0, y - y0
    value = ((1.0 - wx) * (1.0 - wy) * grid[y0, x0] +
             wx * (1.0 - wy) * grid[y0, x1] +
             (1.0 - wx) * wy * grid[y1, x0] +
             wx * wy * grid[y1, x1])
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("Bilinear sampled feature has zero norm")
    return value / norm


def retrieve_metrics(source_feature: np.ndarray, source_grid: np.ndarray, target_grid: np.ndarray, source_token_xy: np.ndarray, radius: int) -> dict[str, Any]:
    similarities = np.einsum("hwc,c->hw", target_grid, source_feature)
    flat_index = int(np.argmax(similarities))
    row, col = np.unravel_index(flat_index, similarities.shape)
    top1 = float(similarities[row, col])
    masked = similarities.copy()
    y0, y1 = max(0, row - radius), min(masked.shape[0], row + radius + 1)
    x0, x1 = max(0, col - radius), min(masked.shape[1], col + radius + 1)
    masked[y0:y1, x0:x1] = -np.inf
    outside = float(np.max(masked))
    target_feature = target_grid[row, col]
    reverse = np.einsum("hwc,c->hw", source_grid, target_feature)
    reverse_index = int(np.argmax(reverse))
    reverse_row, reverse_col = np.unravel_index(reverse_index, reverse.shape)
    cycle_error = float(math.hypot(reverse_col - float(source_token_xy[0]), reverse_row - float(source_token_xy[1])))
    return {
        "top1_similarity": top1,
        "peak_margin": top1 - outside,
        "target_token_xy": np.asarray([col, row], dtype=np.float64),
        "cycle_return_token_xy": np.asarray([reverse_col, reverse_row], dtype=np.float64),
        "cycle_error_tokens": cycle_error,
    }


def local_token_scale(
    target_token_xy: np.ndarray, metadata: dict[str, Any], patch_size: int,
    calibration: np.ndarray, grid_hw: tuple[int, int],
) -> float:
    x, y = map(int, target_token_xy)
    height, width = grid_hw
    adjacent_x = x + 1 if x + 1 < width else x - 1
    adjacent_y = y + 1 if y + 1 < height else y - 1
    tokens = np.asarray([[x, y], [adjacent_x, y], [x, adjacent_y]], dtype=np.float64)
    raw = token_to_raw(tokens, metadata, patch_size)
    undistorted = raw_to_undistorted(raw, calibration)
    horizontal = float(np.linalg.norm(undistorted[1] - undistorted[0]))
    vertical = float(np.linalg.norm(undistorted[2] - undistorted[0]))
    scale = 0.5 * (horizontal + vertical)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid local token scale: {scale}")
    return scale


def geometry_metric(
    *, source_undistorted_xy: np.ndarray, predicted_token_xy: np.ndarray,
    source_timestamp_ns: int, target_timestamp_ns: int, gt: GroundTruth,
    body_to_camera: np.ndarray, matrix: np.ndarray, calibration: np.ndarray,
    metadata: dict[str, Any], patch_size: int, grid_hw: tuple[int, int],
) -> dict[str, Any]:
    relative = relative_camera_pose(gt, source_timestamp_ns, target_timestamp_ns, body_to_camera)
    if np.linalg.norm(relative[:3, 3]) <= 1e-10:
        return {"epipolar_error_tokens": float("nan"), "target_raw_xy": [np.nan, np.nan], "target_undistorted_xy": [np.nan, np.nan], "token_scale_pixels": float("nan")}
    raw = token_to_raw(np.asarray(predicted_token_xy)[None], metadata, patch_size)[0]
    undistorted = raw_to_undistorted(raw[None], calibration)[0]
    fundamental = fundamental_from_relative(relative, matrix)
    line = fundamental @ np.asarray([*source_undistorted_xy, 1.0], dtype=np.float64)
    error_pixels = point_line_distance(undistorted, line)
    scale = local_token_scale(predicted_token_xy, metadata, patch_size, calibration, grid_hw)
    return {
        "epipolar_error_tokens": error_pixels / scale,
        "epipolar_error_pixels": error_pixels,
        "target_raw_xy": raw,
        "target_undistorted_xy": undistorted,
        "token_scale_pixels": scale,
    }


def _normalize_grid(grid: np.ndarray) -> np.ndarray:
    values = grid.astype(np.float32, copy=False)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _make_coordinate_figure(
    rows: list[dict[str, Any]], metadata: dict[str, Any], patch_size: int,
    output_path: Path,
) -> list[str]:
    import matplotlib.pyplot as plt
    from PIL import Image

    selected = []
    for sequence in ("MH_01_easy", "MH_05_difficult"):
        for group in ("good", "bad"):
            selected.extend([row for row in rows if row["sequence"] == sequence and row["group"] == group][:5])
    if len(selected) != 20:
        raise AssertionError("Coordinate sanity figure requires exactly 20 samples")
    figure, axes = plt.subplots(4, 5, figsize=(20, 11), constrained_layout=True)
    left, top, right, bottom = metadata["crop_box_xyxy"]
    original_h, original_w = metadata["original_size_hw"]
    resized_h, resized_w = metadata["resized_size_hw"]
    raw_crop = np.asarray([
        [(left + 0.5) * original_w / resized_w - 0.5, (top + 0.5) * original_h / resized_h - 0.5],
        [(right - 0.5) * original_w / resized_w - 0.5, (top + 0.5) * original_h / resized_h - 0.5],
        [(right - 0.5) * original_w / resized_w - 0.5, (bottom - 0.5) * original_h / resized_h - 0.5],
        [(left + 0.5) * original_w / resized_w - 0.5, (bottom - 0.5) * original_h / resized_h - 0.5],
    ])
    sample_ids = []
    for axis, row in zip(axes.ravel(), selected):
        image = np.asarray(Image.open(row["source_image_path"]).convert("L"))
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        mapped = np.asarray(row["source_token_xy"], dtype=np.float64)
        nearest = np.rint(mapped).astype(int)
        center_raw = token_to_raw(nearest[None], metadata, patch_size)[0]
        cell_token = np.asarray([
            [nearest[0] - 0.5, nearest[1] - 0.5],
            [nearest[0] + 0.5, nearest[1] - 0.5],
            [nearest[0] + 0.5, nearest[1] + 0.5],
            [nearest[0] - 0.5, nearest[1] + 0.5],
        ])
        cell_raw = token_to_raw(cell_token, metadata, patch_size)
        raw_point = np.asarray(row["source_raw_xy"])
        axis.plot(*np.vstack((raw_crop, raw_crop[0])).T, color="yellow", linewidth=0.8)
        axis.plot(*np.vstack((cell_raw, cell_raw[0])).T, color="cyan", linewidth=1.2)
        axis.scatter(raw_point[0], raw_point[1], marker="x", color="red", s=40, linewidth=1.5)
        axis.scatter(center_raw[0], center_raw[1], marker="+", color="lime", s=45, linewidth=1.5)
        axis.set_title(f"{row['sequence']} {row['group']}\ntoken=({mapped[0]:.2f},{mapped[1]:.2f})", fontsize=8)
        axis.set_xlim(0, image.shape[1] - 1)
        axis.set_ylim(image.shape[0] - 1, 0)
        axis.axis("off")
        sample_ids.append(row["sample_id"])
    figure.suptitle("Coordinate sanity: DPVO point (red x), nearest V-JEPA cell (cyan), token center (green +), crop (yellow)")
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return sample_ids


def main() -> int:
    started = time.perf_counter()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    experiment_root = DPVO_ROOT / config["paths"]["output_root"]
    output_root = experiment_root / PROFILE_NAME
    artifacts_dir = output_root / "artifacts"
    rows = _load_rows(artifacts_dir / "samples.jsonl")
    expected_samples = int(config["sampling_profiles"][PROFILE_NAME]["samples_per_group"]) * 4
    if len(rows) != expected_samples:
        raise AssertionError(f"Expected {expected_samples} formal samples, got {len(rows)}")

    vjepa_root = Path(config["vjepa"]["repo"]).resolve()
    if _git(vjepa_root, "rev-parse", "HEAD") != config["vjepa"]["expected_git_commit"]:
        raise RuntimeError("V-JEPA provider commit differs from the frozen Experiment 2 commit")
    provider_status_before = _git(vjepa_root, "status", "--short")
    checkpoint = vjepa_root / config["vjepa"]["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # No dpvo-research research package has been imported.  Resolve provider now.
    sys.path.insert(0, str(vjepa_root))
    from research.scripts.common.dense_pca import (  # type: ignore[import-not-found]
        crop_metadata_json,
        extract_frame_features,
        load_phase2_encoder,
        preprocess_rgb_frame,
    )

    environment = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(environment, indent=2))
    if Path(sys.executable).resolve() != Path(config["vjepa"]["python"]).resolve():
        raise RuntimeError(f"Wrong V-JEPA Python executable: {sys.executable}")
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen V-JEPA formal extraction requires CUDA; no new environment will be created")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    calibration = np.loadtxt(DPVO_ROOT / config["paths"]["euroc_calibration"], delimiter=" ")
    matrix = camera_matrix(calibration)
    body_to_camera = _load_body_to_camera(
        DPVO_ROOT / config["paths"]["euroc_root"] / "MH_01_easy/mav0/cam0/sensor.yaml"
    )
    geometry_preflight = synthetic_geometry_consistency(matrix, body_to_camera)

    first_processed = preprocess_rgb_frame(Path(rows[0]["source_image_path"]), crop_size=int(config["coordinates"]["vjepa_crop_size"]))
    provider_metadata = crop_metadata_json(first_processed.metadata)
    expected_metadata = {
        "original_size_hw": [480, 752], "resized_size_hw": [438, 686],
        "crop_box_xyxy": [151, 27, 535, 411], "crop_size": 384,
    }
    if provider_metadata != expected_metadata:
        raise AssertionError(f"Provider preprocessing changed: {provider_metadata}")
    patch_size = int(config["coordinates"]["token_patch_size"])
    prepared_raw = np.asarray([row["source_raw_xy"] for row in rows], dtype=np.float64)
    prepared_token = np.asarray([row["source_token_xy"] for row in rows], dtype=np.float64)
    provider_token = raw_to_token(prepared_raw, provider_metadata, patch_size)
    prepare_provider_error = float(np.max(np.abs(provider_token - prepared_token)))
    raw_roundtrip_error = float(np.max(np.abs(token_to_raw(provider_token, provider_metadata, patch_size) - prepared_raw)))
    if prepare_provider_error > 1e-9 or raw_roundtrip_error > 1e-9:
        raise AssertionError(f"Coordinate preflight failed: provider={prepare_provider_error}, roundtrip={raw_roundtrip_error}")

    smoke_sanity = experiment_root / "debug/coordinate_sanity.png"
    if not smoke_sanity.is_file():
        raise FileNotFoundError(f"Validated smoke coordinate sanity is missing: {smoke_sanity}")

    all_paths = {row["source_image_path"] for row in rows} | {row["target_image_path"] for row in rows}
    all_paths |= {row["shuffled_target_image_path"] for row in rows if row["has_shuffled_control"]}
    ordered_paths = [Path(path) for path in sorted(all_paths)]
    encoder = load_phase2_encoder(device)
    feature_cache: dict[str, np.ndarray] = {}
    chunk_size = 32
    batch_size = int(config["vjepa"]["batch_size"])
    extraction_seconds = 0.0
    observed_grid = None
    observed_dim = None
    observed_patch = None
    for start in range(0, len(ordered_paths), chunk_size):
        chunk = ordered_paths[start:start + chunk_size]
        batch = extract_frame_features(
            frame_paths=chunk, encoder=encoder, device=device,
            frame_indices=list(range(start, start + len(chunk))),
            crop_size=int(config["coordinates"]["vjepa_crop_size"]), batch_size=batch_size,
        )
        extraction_seconds += float(batch.runtime["encoder_forward_seconds"])
        observed_grid = batch.grid_shape
        observed_dim = batch.feature_dim
        observed_patch = batch.patch_size
        if tuple(batch.grid_shape) != (24, 24) or batch.feature_dim != 768 or batch.patch_size != 16:
            raise AssertionError(f"Unexpected dense representation: grid={batch.grid_shape}, dim={batch.feature_dim}, patch={batch.patch_size}")
        for path, grid in zip(chunk, batch.features):
            feature_cache[str(path)] = _normalize_grid(grid)
        del batch
    del encoder
    torch.cuda.empty_cache()

    groundtruth = {
        sequence: GroundTruth(DPVO_ROOT / config["paths"]["groundtruth_pattern"].format(sequence=sequence))
        for sequence in config["experiment"]["sequences"]
    }
    radius = int(config["metrics"]["peak_exclusion_chebyshev_radius"])
    cycle_threshold = float(config["metrics"]["cycle_success_threshold_tokens"])
    epipolar_threshold = float(config["metrics"]["epipolar_threshold_tokens"])

    correct_records: list[dict[str, Any]] = []
    shuffled_records: list[dict[str, Any]] = []
    for sample_index, row in enumerate(rows):
        source_grid = feature_cache[row["source_image_path"]]
        source_token = np.asarray(row["source_token_xy"], dtype=np.float64)
        source_feature = bilinear_sample(source_grid, source_token)

        def evaluate(target_path: str, target_timestamp_ns: int, pair_kind: str) -> dict[str, Any]:
            base = {
                "sample_index": sample_index, "sample_id": row["sample_id"],
                "sequence": row["sequence"], "group": row["group"], "pair_kind": pair_kind,
            }
            try:
                retrieval = retrieve_metrics(source_feature, source_grid, feature_cache[target_path], source_token, radius)
                geometry = geometry_metric(
                    source_undistorted_xy=np.asarray(row["source_undistorted_xy"], dtype=np.float64),
                    predicted_token_xy=retrieval["target_token_xy"],
                    source_timestamp_ns=int(row["source_timestamp_ns"]),
                    target_timestamp_ns=int(target_timestamp_ns),
                    gt=groundtruth[row["sequence"]], body_to_camera=body_to_camera,
                    matrix=matrix, calibration=calibration, metadata=provider_metadata,
                    patch_size=patch_size, grid_hw=(24, 24),
                )
                finite = bool(all(np.isfinite(value) for value in (
                    retrieval["top1_similarity"], retrieval["peak_margin"],
                    retrieval["cycle_error_tokens"], geometry["epipolar_error_tokens"],
                )))
                cycle_success = bool(finite and retrieval["cycle_error_tokens"] <= cycle_threshold)
                geometry_consistent = bool(
                    finite and geometry["epipolar_error_tokens"] <= epipolar_threshold and cycle_success
                )
                return {
                    **base, **retrieval, **geometry,
                    "cycle_success": cycle_success,
                    "jepa_geometry_consistent": geometry_consistent,
                    "valid": finite,
                    "failure_reason": "" if finite else "non_finite_metric",
                }
            except Exception as error:  # Preserve the frozen sample; never resample on metric failure.
                nan_xy = np.asarray([np.nan, np.nan], dtype=np.float64)
                return {
                    **base,
                    "top1_similarity": float("nan"), "peak_margin": float("nan"),
                    "target_token_xy": nan_xy, "cycle_return_token_xy": nan_xy,
                    "cycle_error_tokens": float("nan"), "epipolar_error_tokens": float("nan"),
                    "epipolar_error_pixels": float("nan"), "target_raw_xy": nan_xy,
                    "target_undistorted_xy": nan_xy, "token_scale_pixels": float("nan"),
                    "cycle_success": False, "jepa_geometry_consistent": False,
                    "valid": False, "failure_reason": f"{type(error).__name__}: {error}",
                }

        correct_records.append(evaluate(row["target_image_path"], int(row["target_timestamp_ns"]), "correct"))
        if row["has_shuffled_control"]:
            record = evaluate(row["shuffled_target_image_path"], int(row["shuffled_target_timestamp_ns"]), "temporal_shuffled")
            record["target_frame_id"] = int(row["shuffled_target_frame_id"])
            record["target_timestamp_ns"] = int(row["shuffled_target_timestamp_ns"])
            shuffled_records.append(record)

    expected_shuffled = int(config["sampling_profiles"][PROFILE_NAME]["shuffled_per_group"]) * 4
    if len(correct_records) != expected_samples or len(shuffled_records) != expected_shuffled:
        raise AssertionError("Correspondence result count mismatch")

    def records_to_arrays(records: list[dict[str, Any]], prefix: str = "") -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for key in records[0]:
            values = [record[key] for record in records]
            name = f"{prefix}{key}"
            first = values[0]
            if isinstance(first, (str, np.str_)):
                output[name] = np.asarray(values, dtype=np.str_)
            elif isinstance(first, (bool, np.bool_)):
                output[name] = np.asarray(values, dtype=bool)
            elif isinstance(first, (int, np.integer)):
                output[name] = np.asarray(values, dtype=np.int64)
            else:
                output[name] = np.asarray(values, dtype=np.float64)
        return output

    payload = records_to_arrays(correct_records)
    payload.update(records_to_arrays(shuffled_records, prefix="shuffled_"))
    np.savez_compressed(artifacts_dir / "jepa_metrics.npz", **payload)

    provider_status_after = _git(vjepa_root, "status", "--short")
    if provider_status_after != provider_status_before:
        raise RuntimeError("V-JEPA provider worktree changed during read-only extraction")
    manifest = {
        "schema_version": 2,
        "sampling_profile": PROFILE_NAME,
        "environment": environment,
        "vjepa": {
            "git_commit": _git(vjepa_root, "rev-parse", "HEAD"),
            "provider_status_before": provider_status_before.splitlines(),
            "provider_status_after": provider_status_after.splitlines(),
            "model_alias": config["vjepa"]["model_alias"],
            "model_name": config["vjepa"]["model_name"],
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "representation": config["vjepa"]["representation"],
            "feature_mode": config["vjepa"]["feature_mode"],
            "input_normalization": "ImageNet mean/std from provider",
            "preprocessing": provider_metadata,
            "token_grid_hw": list(observed_grid),
            "patch_size": observed_patch,
            "feature_dim": observed_dim,
            "inference_dtype": "bfloat16 autocast; metrics float32/float64",
        },
        "geometry": {
            "domain": "original-resolution undistorted pinhole pixels",
            "relative_pose": "T_Ct_Cs = inv(T_WC_t) @ T_WC_s",
            "camera_pose": "T_WC = T_WB @ T_BC (EuRoC cam0 T_BS)",
            "synthetic_consistency_preflight": geometry_preflight,
            "coordinate_prepare_provider_max_abs_error": prepare_provider_error,
            "coordinate_raw_roundtrip_max_abs_error": raw_roundtrip_error,
            "epipolar_normalization": "mean local horizontal/vertical one-token spacing in undistorted pixels",
        },
        "counts": {
            "samples": len(correct_records), "shuffled_controls": len(shuffled_records),
            "unique_inference_frames": len(feature_cache),
            "valid_correct": sum(record["valid"] for record in correct_records),
            "valid_shuffled": sum(record["valid"] for record in shuffled_records),
            "failure_reason_inventory_correct": dict(Counter(
                record["failure_reason"] for record in correct_records if not record["valid"]
            )),
            "failure_reason_inventory_shuffled": dict(Counter(
                record["failure_reason"] for record in shuffled_records if not record["valid"]
            )),
        },
        "coordinate_sanity": {
            "path": str(smoke_sanity.relative_to(experiment_root)),
            "sha256": _sha256(smoke_sanity),
            "panel_count": 20,
            "automated_mapping_checks_passed": True,
            "manual_review": "passed during fixed 200-sample smoke",
            "regenerated_for_formal": False,
        },
        "runtime": {
            "encoder_forward_seconds": extraction_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "samples_jsonl_sha256": _sha256(artifacts_dir / "samples.jsonl"),
    }
    _write_json(artifacts_dir / "extraction_manifest.json", manifest)
    print(json.dumps(_jsonable({"counts": manifest["counts"], "geometry_preflight": geometry_preflight, "runtime": manifest["runtime"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
