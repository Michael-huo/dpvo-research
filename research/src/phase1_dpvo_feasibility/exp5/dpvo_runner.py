"""Isolated upstream DPVO worker for baseline and frame-level sidecar runs."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from contextlib import nullcontext
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

import numpy as np

from .jepa_runner import JepaSidecar, mock_jepa_record, sidecar_frame_payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_dpvo_config(config: dict[str, Any]) -> Any:
    from dpvo.config import cfg as base_cfg

    runtime_cfg = base_cfg.clone()
    runtime_cfg.merge_from_file(str(config["paths"]["dpvo_config"]))
    runtime_cfg.LOOP_CLOSURE = False
    runtime_cfg.CLASSIC_LOOP_CLOSURE = False
    return runtime_cfg


def _validate_stream_record(t: int, record: dict[str, Any]) -> None:
    if int(t) != int(record["stream_index"]):
        raise AssertionError(
            f"upstream stream ordinal mismatch: image_stream={t}, record={record['stream_index']}"
        )


def run_real(
    *, config: dict[str, Any], records: list[dict[str, Any]], mode: str, request_path: Path,
) -> dict[str, Any]:
    """Run exactly one DPVO instance in this worker process."""
    if mode not in {"baseline", "sidecar"}:
        raise ValueError(f"unsupported DPVO worker mode: {mode}")
    if not records:
        raise ValueError("DPVO worker received no frames")

    import torch
    from dpvo.dpvo import DPVO
    from dpvo.stream import image_stream

    if not torch.cuda.is_available():
        raise RuntimeError("formal DPVO inference requires CUDA")
    expected_python = Path(config["runtime"]["dpvo_python"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"wrong DPVO Python: expected {expected_python}, got {sys.executable}")
    _seed_everything(int(config["experiment"]["seed"]), torch)
    runtime_cfg = _load_dpvo_config(config)

    image_dir = Path(records[0]["image_path"]).parent
    if any(Path(record["image_path"]).parent != image_dir for record in records):
        raise ValueError("one DPVO worker may process only one image directory")
    queue: Queue = Queue(maxsize=8)
    reader = Process(
        target=image_stream,
        args=(
            queue,
            str(image_dir),
            str(config["paths"]["calibration"]),
            int(config["dataset"]["stride"]),
            int(config["dataset"]["skip"]),
        ),
    )
    sidecar_context = JepaSidecar(config, request_path) if mode == "sidecar" else nullcontext(None)
    reader.start()
    slam = None
    frame_results: list[dict[str, Any]] = []
    jepa_results: list[dict[str, Any]] = []
    hook_frame_count = 0
    existing_pose_count = 0
    existing_pose_max_error = 0.0
    hook_frame_index_unchanged = True
    hook_pose_shape_unchanged = True
    hook_pose_finite = True
    overall_started: float | None = None
    terminate_runtime = 0.0
    try:
        with sidecar_context as sidecar:
            while True:
                t, image_numpy, intrinsics_numpy = queue.get()
                if int(t) < 0:
                    break
                if int(t) >= len(records):
                    raise AssertionError(f"image_stream produced unexpected frame ordinal {t}")
                record = records[int(t)]
                _validate_stream_record(int(t), record)
                image = torch.from_numpy(image_numpy).permute(2, 0, 1).cuda()
                intrinsics = torch.from_numpy(intrinsics_numpy).cuda()
                if slam is None:
                    slam = DPVO(
                        runtime_cfg,
                        str(config["paths"]["dpvo_checkpoint"]),
                        ht=int(image.shape[1]),
                        wd=int(image.shape[2]),
                        viz=False,
                    )
                    torch.cuda.synchronize()
                    overall_started = time.perf_counter()
                torch.cuda.synchronize()
                frame_started = time.perf_counter()
                with torch.no_grad():
                    slam(int(record["stream_index"]), image, intrinsics)
                torch.cuda.synchronize()
                frame_runtime = time.perf_counter() - frame_started
                frame_results.append({
                    "timestamp": int(record["timestamp"]),
                    "frame_id": int(record["frame_id"]),
                    "stream_index": int(record["stream_index"]),
                    "runtime": float(frame_runtime),
                })
                if sidecar is not None:
                    graph_frame_index = int(slam.n)
                    torch.cuda.synchronize()
                    pose_before = slam.pg.poses_[:graph_frame_index].detach().clone()
                    torch.cuda.synchronize()
                    jepa_results.append(sidecar.extract(record))
                    torch.cuda.synchronize()
                    graph_frame_index_after = int(slam.n)
                    pose_after = slam.pg.poses_[:graph_frame_index_after].detach()
                    hook_frame_count += 1
                    existing_pose_count += graph_frame_index
                    frame_index_equal = graph_frame_index_after == graph_frame_index
                    shape_equal = tuple(pose_after.shape) == tuple(pose_before.shape)
                    hook_frame_index_unchanged &= frame_index_equal
                    hook_pose_shape_unchanged &= shape_equal
                    before_finite = bool(torch.isfinite(pose_before).all().item())
                    after_finite = bool(torch.isfinite(pose_after).all().item())
                    hook_pose_finite &= before_finite and after_finite
                    if frame_index_equal and shape_equal and pose_before.numel():
                        frame_max_error = float(
                            torch.max(torch.abs(pose_after - pose_before)).item()
                        )
                        existing_pose_max_error = max(
                            existing_pose_max_error, frame_max_error
                        )
                    del pose_before, pose_after

            if len(frame_results) != len(records):
                raise AssertionError(
                    f"image_stream frame count mismatch: {len(frame_results)} != {len(records)}"
                )
            if slam is None or overall_started is None:
                raise RuntimeError("DPVO was not initialized")
            torch.cuda.synchronize()
            terminate_started = time.perf_counter()
            with torch.no_grad():
                poses, returned_ordinals = slam.terminate()
            torch.cuda.synchronize()
            terminate_runtime = time.perf_counter() - terminate_started
            total_runtime = time.perf_counter() - overall_started
    finally:
        reader.join(timeout=30)
        if reader.is_alive():
            reader.terminate()
            reader.join(timeout=10)

    ordinals = np.asarray(returned_ordinals, dtype=np.float64)
    expected_ordinals = np.arange(len(records), dtype=np.float64)
    pose_array = np.asarray(poses, dtype=np.float64)
    if not np.array_equal(ordinals, expected_ordinals):
        raise AssertionError("DPVO terminate() did not preserve upstream stream ordinals")
    if pose_array.shape != (len(records), 7) or not np.isfinite(pose_array).all():
        raise AssertionError(f"invalid DPVO trajectory shape/values: {pose_array.shape}")
    return {
        "status": "ok",
        "mode": mode,
        "sequence": records[0]["sequence"],
        "runtime": float(total_runtime),
        "terminate_runtime": float(terminate_runtime),
        "frames": frame_results,
        "jepa_frames": jepa_results,
        "trajectory_validation": ({
            "mode": "single_run_hook_validation",
            "existing_pose_max_error": float(existing_pose_max_error),
            "existing_pose_count": int(existing_pose_count),
            "compared_frame_count": int(hook_frame_count),
            "frame_index_unchanged": bool(hook_frame_index_unchanged),
            "pose_shape_unchanged": bool(hook_pose_shape_unchanged),
            "finite_check": bool(hook_pose_finite),
        } if mode == "sidecar" else None),
        "jepa_setup_runtime": (
            float(sidecar_context.setup_runtime) if mode == "sidecar" else None
        ),
    }


def run_mock(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[dict[str, Any]]]:
    """Exercise the full ordering contract with real records and fake models."""
    baseline_frames: list[dict[str, Any]] = []
    sidecar_frames: list[dict[str, Any]] = []
    jepa_frames: list[dict[str, Any]] = []
    events: list[str] = []
    jepa_payloads: list[dict[str, Any]] = []
    dpvo_frame_runtime, jepa_frame_runtime, terminate_runtime = 0.001, 0.0005, 0.002
    for record in records:
        baseline_frames.append({
            "timestamp": int(record["timestamp"]),
            "frame_id": int(record["frame_id"]),
            "stream_index": int(record["stream_index"]),
            "runtime": dpvo_frame_runtime,
            "mock": True,
        })
    for record, baseline in zip(records, baseline_frames):
        events.append(f"dpvo:{record['frame_id']}")
        events.append(f"hook_before:{record['frame_id']}")
        sidecar_frames.append(dict(baseline))
        payload = sidecar_frame_payload(record)
        jepa_payloads.append(payload)
        events.append(f"jepa:{record['frame_id']}")
        jepa_frames.append(mock_jepa_record(payload, runtime=jepa_frame_runtime))
        events.append(f"hook_after:{record['frame_id']}")
    baseline_runtime = len(records) * dpvo_frame_runtime + terminate_runtime
    sidecar_runtime = baseline_runtime + len(records) * jepa_frame_runtime
    baseline_result = {
        "status": "ok", "mode": "baseline", "runtime": baseline_runtime,
        "frames": baseline_frames, "jepa_frames": [], "mock": True,
    }
    sidecar_result = {
        "status": "ok", "mode": "sidecar", "runtime": sidecar_runtime,
        "frames": sidecar_frames, "jepa_frames": jepa_frames, "mock": True,
        "trajectory_validation": {
            "mode": "single_run_hook_validation",
            "existing_pose_max_error": 0.0,
            "existing_pose_count": sum(range(1, len(records) + 1)),
            "compared_frame_count": len(records),
            "frame_index_unchanged": True,
            "pose_shape_unchanged": True,
            "finite_check": True,
        },
    }
    return baseline_result, sidecar_result, events, jepa_payloads


def worker_main() -> int:
    """Internal subprocess entry invoked by the sole public Exp5 CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--mode", choices=("baseline", "sidecar"), required=True)
    args = parser.parse_args()
    try:
        request_path = Path(args.request_json).resolve()
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = run_real(
            config=dict(request["config"]),
            records=list(request["records"]),
            mode=str(args.mode),
            request_path=request_path,
        )
        _write_json(args.result_json, result)
        return 0
    except Exception as error:
        _write_json(args.result_json, {
            "status": "failed",
            "mode": str(args.mode),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        })
        return 1
