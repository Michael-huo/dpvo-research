"""Separate-environment dense JEPA bridge and in-process DPVO runtime."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Callable, TextIO

import numpy as np

from ..jepa_runner import _git, _sha256
from .feature_hook import FNetInjectionHook, OracleFNetReplacementHook
from .projection import (
    DenseJepaProjection,
    build_random_dense_projection,
    dense_spatial_cosine_loss,
    estimate_pair_bytes,
    load_dense_projection_checkpoint,
    validate_memory_capacity,
)
from .trajectory_eval import build_estimated_trajectory, write_trajectory


FUSION_METHODS = (
    "dpvo_baseline",
    "random_dense_projection_fusion",
    "jepa_dense_projection_fusion",
)
ORACLE_METHODS = (
    "dpvo_baseline",
    "oracle_jepa_missing_rgb",
)
RUNTIME_KEYS = frozenset(
    (
        "jepa_extraction_time",
        "environment_bridge_time",
        "worker_setup_time",
        "dense_projection_time",
        "fusion_hook_time",
        "dpvo_total_time",
        "end_to_end_total_time",
    )
)
DENSE_PROTOCOL_PREFIX = "EXP5_DENSE_JEPA_JSON:"
DENSE_FRAME_REQUEST_KEYS = frozenset(
    ("request_id", "image_path", "frame_id", "timestamp")
)
DENSE_TOKEN_RESPONSE_KEYS = frozenset(
    (
        "status", "request_id", "frame_id", "timestamp", "token_shape",
        "token_dtype", "token_encoding", "token_data", "finite_check",
        "extraction_time",
    )
)


def frame_availability(
    records: list[dict[str, Any]], *, keyframe_stride: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build and validate the Oracle matching-feature availability contract."""
    if not records or int(keyframe_stride) != 5:
        raise ValueError("Exp5-Oracle requires records and fixed keyframe_stride=5")
    frames: list[dict[str, Any]] = []
    for ordinal, record in enumerate(records):
        if int(record["stream_index"]) != ordinal:
            raise ValueError("Oracle records must use contiguous stream_index identity")
        is_keyframe = ordinal % int(keyframe_stride) == 0
        frames.append({
            "frame_id": int(record["frame_id"]),
            "timestamp": int(record["timestamp"]),
            "is_keyframe": is_keyframe,
            "matching_rgb_available": is_keyframe,
            "has_jepa": True,
            "context_rgb_used": True,
        })
    canonical = json.dumps(
        frames, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    keyframe_count = sum(int(frame["is_keyframe"]) for frame in frames)
    replaced_count = len(frames) - keyframe_count
    matching_ratio = keyframe_count / len(frames)
    replaced_ratio = replaced_count / len(frames)
    if not np.isclose(matching_ratio + replaced_ratio, 1.0):
        raise AssertionError("Oracle availability ratios must sum to one")
    summary = {
        "keyframe_stride": int(keyframe_stride),
        "num_frames": len(frames),
        "keyframe_count": keyframe_count,
        "non_keyframe_count": replaced_count,
        "matching_rgb_ratio": matching_ratio,
        "jepa_replaced_ratio": replaced_ratio,
        "configured_nominal_matching_rgb_ratio": 0.2,
        "configured_nominal_jepa_replaced_ratio": 0.8,
        "has_jepa_ratio": 1.0,
        "context_rgb_used": True,
        "canonical_availability_sha256": hashlib.sha256(canonical).hexdigest(),
        "frame_identity_validated": True,
    }
    return frames, summary


def _emit_dense(payload: dict[str, Any]) -> None:
    print(
        DENSE_PROTOCOL_PREFIX
        + json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")),
        flush=True,
    )


def dense_frame_request(record: dict[str, Any], request_id: int) -> dict[str, Any]:
    payload = {
        "request_id": int(request_id),
        "image_path": str(record["image_path"]),
        "frame_id": int(record["frame_id"]),
        "timestamp": int(record["timestamp"]),
    }
    if set(payload) != DENSE_FRAME_REQUEST_KEYS or payload["request_id"] < 0:
        raise ValueError("invalid dense JEPA frame request")
    return payload


def encode_dense_token_response(
    *, request: dict[str, Any], tokens: np.ndarray, extraction_time: float,
) -> dict[str, Any]:
    if set(request) != DENSE_FRAME_REQUEST_KEYS:
        raise ValueError("dense worker request schema changed")
    array = np.asarray(tokens, dtype=np.float32)
    if array.shape != (576, 768) or not array.flags.c_contiguous:
        raise ValueError(f"invalid dense token array: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("dense token array contains non-finite values")
    extraction_time = float(extraction_time)
    if not np.isfinite(extraction_time) or extraction_time < 0.0:
        raise ValueError("invalid dense token extraction runtime")
    payload = {
        "status": "ok",
        "request_id": int(request["request_id"]),
        "frame_id": int(request["frame_id"]),
        "timestamp": int(request["timestamp"]),
        "token_shape": [576, 768],
        "token_dtype": "float32",
        "token_encoding": "base64_raw_little_endian",
        "token_data": base64.b64encode(array.astype("<f4", copy=False).tobytes()).decode(
            "ascii"
        ),
        "finite_check": True,
        "extraction_time": extraction_time,
    }
    if set(payload) != DENSE_TOKEN_RESPONSE_KEYS:
        raise AssertionError("dense token response schema changed")
    return payload


def decode_dense_token_response(
    payload: dict[str, Any], *, expected_request: dict[str, Any],
) -> tuple[np.ndarray, float]:
    if set(payload) != DENSE_TOKEN_RESPONSE_KEYS or payload.get("status") != "ok":
        raise RuntimeError("invalid dense token response schema")
    for key in ("request_id", "frame_id", "timestamp"):
        if int(payload[key]) != int(expected_request[key]):
            raise RuntimeError(f"dense token response {key} mismatch")
    if payload["token_shape"] != [576, 768] or payload["token_dtype"] != "float32":
        raise RuntimeError("dense token response shape/dtype mismatch")
    if payload["token_encoding"] != "base64_raw_little_endian":
        raise RuntimeError("unsupported dense token encoding")
    try:
        raw = base64.b64decode(str(payload["token_data"]), validate=True)
    except Exception as error:
        raise RuntimeError("invalid dense token base64 payload") from error
    expected_bytes = 576 * 768 * np.dtype("<f4").itemsize
    if len(raw) != expected_bytes:
        raise RuntimeError(f"dense token byte count mismatch: {len(raw)} != {expected_bytes}")
    tokens = np.frombuffer(raw, dtype="<f4").reshape(576, 768).astype(np.float32, copy=True)
    if not bool(payload["finite_check"]) or not np.isfinite(tokens).all():
        raise RuntimeError("dense token response contains non-finite values")
    extraction_time = float(payload["extraction_time"])
    if not np.isfinite(extraction_time) or extraction_time < 0.0:
        raise RuntimeError("invalid dense token extraction runtime")
    return tokens, extraction_time


class DenseJepaWorkerClient:
    """One persistent V-JEPA subprocess for one sequence/run."""

    def __init__(self, config: dict[str, Any], request_path: str | Path) -> None:
        self.python = str(Path(config["environment"]["vjepa_python"]).resolve())
        request_path = Path(request_path).resolve()
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(
            json.dumps(
                {
                    "config": {
                        "repo_root": str(config["repo_root"]),
                        "environment": {"vjepa_python": self.python},
                        "jepa": dict(config["jepa"]),
                    }
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        command = [
            self.python,
            "-c",
            (
                "from research.src.phase1_dpvo_feasibility.exp5.fusion.runtime "
                "import dense_worker_main; raise SystemExit(dense_worker_main())"
            ),
            "--request-json", str(request_path),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=Path(config["repo_root"]),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.diagnostics: list[str] = []
        self.request_count = 0
        self.bridge_time = 0.0
        self.extraction_time = 0.0
        self.setup_runtime = 0.0
        self.initialized = False
        self.stopped = False
        self.cleanup_passed = False
        self._inflight = False
        try:
            ready = self._read_message()
            if ready.get("status") != "ready":
                raise RuntimeError(f"dense V-JEPA worker did not become ready: {ready}")
            self.setup_runtime = float(ready["setup_runtime"])
            self.initialized = True
        except Exception:
            self.close(force=True)
            raise

    def _read_message(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("dense V-JEPA worker stdout is unavailable")
        while True:
            line = self.process.stdout.readline()
            if not line:
                code = self.process.poll()
                tail = "\n".join(self.diagnostics[-20:])
                raise RuntimeError(
                    f"dense V-JEPA worker exited unexpectedly ({code}): {tail}"
                )
            stripped = line.rstrip()
            if stripped.startswith(DENSE_PROTOCOL_PREFIX):
                payload = json.loads(stripped[len(DENSE_PROTOCOL_PREFIX):])
                if payload.get("status") == "error":
                    raise RuntimeError(f"dense V-JEPA worker failed: {payload.get('error')}")
                return dict(payload)
            self.diagnostics.append(stripped)

    def _send(self, payload: dict[str, Any]) -> None:
        if self.process.stdin is None or self.process.poll() is not None:
            raise RuntimeError("dense V-JEPA worker is not running")
        self.process.stdin.write(
            json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
            + "\n"
        )
        self.process.stdin.flush()

    def extract(self, record: dict[str, Any]) -> tuple[np.ndarray, float, float]:
        if self._inflight:
            raise RuntimeError("dense V-JEPA worker already has an outstanding request")
        request = dense_frame_request(record, self.request_count)
        started = time.perf_counter()
        self._inflight = True
        try:
            self._send({"action": "extract", "request": request})
            response = self._read_message()
            tokens, extraction_time = decode_dense_token_response(
                response, expected_request=request
            )
        finally:
            self._inflight = False
        roundtrip = time.perf_counter() - started
        bridge_time = max(0.0, roundtrip - extraction_time)
        self.request_count += 1
        self.extraction_time += extraction_time
        self.bridge_time += bridge_time
        return tokens, extraction_time, bridge_time

    def close(self, *, force: bool = False) -> None:
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is not None:
            self.cleanup_passed = bool(self.stopped and process.returncode == 0)
            return
        if not force:
            try:
                self._send({"action": "stop"})
                response = self._read_message()
                if response.get("status") != "stopped" or int(
                    response.get("request_count", -1)
                ) != self.request_count:
                    raise RuntimeError(f"unexpected dense worker shutdown: {response}")
                process.wait(timeout=30)
                self.stopped = process.returncode == 0
                self.cleanup_passed = self.stopped
                return
            except Exception:
                force = True
        if force:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            self.cleanup_passed = False

    def lifecycle(self) -> dict[str, Any]:
        return {
            "used": True,
            "python": self.python,
            "initialized": bool(self.initialized),
            "request_count": int(self.request_count),
            "stopped": bool(self.stopped),
            "cleanup_passed": bool(self.cleanup_passed),
            "worker_setup_time": float(self.setup_runtime),
            "worker_extraction_time": float(self.extraction_time),
            "environment_bridge_time": float(self.bridge_time),
        }

    def __enter__(self) -> "DenseJepaWorkerClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_value: Any) -> None:
        self.close(force=exc is not None)


class MockDenseJepaWorkerClient:
    """CPU smoke substitute with the same persistent lifecycle contract."""

    def __init__(self, tokens: dict[int, np.ndarray]) -> None:
        self.tokens = tokens
        self.request_count = 0
        self.initialized = True
        self.stopped = False
        self.cleanup_passed = False
        self.setup_runtime = 0.0

    def extract(self, record: dict[str, Any]) -> tuple[np.ndarray, float, float]:
        request = dense_frame_request(record, self.request_count)
        frame_id = int(request["frame_id"])
        if frame_id not in self.tokens:
            raise KeyError(f"mock worker has no frame {frame_id}")
        response = encode_dense_token_response(
            request=request,
            tokens=np.ascontiguousarray(self.tokens[frame_id], dtype=np.float32),
            extraction_time=0.0005,
        )
        decoded, extraction_time = decode_dense_token_response(
            response, expected_request=request
        )
        self.request_count += 1
        return decoded, extraction_time, 0.0001

    def close(self, *, force: bool = False) -> None:
        self.stopped = not force
        self.cleanup_passed = not force

    def lifecycle(self) -> dict[str, Any]:
        return {
            "used": True,
            "python": "mock",
            "initialized": self.initialized,
            "request_count": self.request_count,
            "stopped": self.stopped,
            "cleanup_passed": self.cleanup_passed,
            "worker_setup_time": self.setup_runtime,
            "worker_extraction_time": self.request_count * 0.0005,
            "environment_bridge_time": self.request_count * 0.0001,
        }

    def __enter__(self) -> "MockDenseJepaWorkerClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_value: Any) -> None:
        self.close(force=exc is not None)


def _load_dense_worker(config: dict[str, Any]) -> tuple[Any, Any, Any, Path, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("formal dense V-JEPA worker requires CUDA")
    expected_python = Path(config["environment"]["vjepa_python"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(
            f"wrong V-JEPA Python: expected {expected_python}, got {sys.executable}"
        )
    repo = Path(config["jepa"]["repo"]).resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != str(config["jepa"]["expected_git_commit"]):
        raise RuntimeError("V-JEPA commit mismatch")
    status_before = _git(repo, "status", "--short")
    checkpoint = repo / str(config["jepa"]["checkpoint"])
    if not checkpoint.is_file() or _sha256(checkpoint) != str(
        config["jepa"]["checkpoint_sha256"]
    ):
        raise RuntimeError("V-JEPA checkpoint provenance mismatch")
    sys.path.insert(0, str(repo))
    import research as research_package

    sibling = str(repo / "research")
    if sibling not in research_package.__path__:
        research_package.__path__ = [*research_package.__path__, sibling]
    from research.scripts.common.dense_pca import extract_frame_features, load_phase2_encoder

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    encoder = load_phase2_encoder(device).requires_grad_(False).eval()
    return torch, encoder, extract_frame_features, repo, status_before


def _worker_extract_one(
    *, torch: Any, encoder: Any, extractor: Any, config: dict[str, Any],
    request: dict[str, Any],
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    with torch.inference_mode():
        batch = extractor(
            frame_paths=[Path(request["image_path"])],
            encoder=encoder,
            device=torch.device("cuda:0"),
            frame_indices=[int(request["frame_id"])],
            crop_size=int(config["jepa"]["crop_size"]),
            batch_size=1,
        )
    dense = np.asarray(batch.features[0], dtype=np.float32)
    if dense.shape not in ((24, 24, 768), (576, 768)):
        raise RuntimeError(f"invalid dense JEPA worker output: {dense.shape}")
    tokens = dense.reshape(576, 768)
    if not np.isfinite(tokens).all():
        raise RuntimeError("dense JEPA worker output contains non-finite values")
    result = np.ascontiguousarray(tokens)
    elapsed = time.perf_counter() - started
    del batch, dense, tokens
    return result, elapsed


def _dense_worker_loop(config: dict[str, Any], input_stream: TextIO) -> int:
    setup_started = time.perf_counter()
    torch, encoder, extractor, repo, status_before = _load_dense_worker(config)
    _emit_dense({"status": "ready", "setup_runtime": time.perf_counter() - setup_started})
    request_count = 0
    try:
        for line in input_stream:
            envelope = json.loads(line)
            action = envelope.get("action")
            if action == "extract":
                request = dict(envelope["request"])
                if set(request) != DENSE_FRAME_REQUEST_KEYS:
                    raise ValueError("dense worker frame request schema changed")
                if int(request["request_id"]) != request_count:
                    raise ValueError("dense worker request_id is not monotonic")
                tokens, elapsed = _worker_extract_one(
                    torch=torch, encoder=encoder, extractor=extractor,
                    config=config, request=request,
                )
                _emit_dense(
                    encode_dense_token_response(
                        request=request, tokens=tokens, extraction_time=elapsed
                    )
                )
                request_count += 1
                del tokens
            elif action == "stop":
                if _git(repo, "status", "--short") != status_before:
                    raise RuntimeError("V-JEPA worktree changed during Exp5-2 extraction")
                _emit_dense({"status": "stopped", "request_count": request_count})
                return 0
            else:
                raise ValueError(f"unsupported dense V-JEPA worker action: {action}")
        raise RuntimeError("dense V-JEPA worker stdin closed without stop")
    finally:
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def dense_worker_main() -> int:
    """Internal V-JEPA environment entry; not a public Exp5 CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request-json", required=True)
    args = parser.parse_args()
    try:
        request = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
        return _dense_worker_loop(dict(request["config"]), sys.stdin)
    except Exception as error:
        _emit_dense({"status": "error", "error": repr(error), "traceback": traceback.format_exc()})
        return 1


def available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("cannot determine MemAvailable from /proc/meminfo")


def separate_environment_preflight(config: dict[str, Any]) -> dict[str, Any]:
    expected_dpvo = Path(config["runtime"]["dpvo_python"]).resolve()
    vjepa_python = Path(config["environment"]["vjepa_python"]).resolve()
    if Path(sys.executable).resolve() != expected_dpvo:
        raise RuntimeError(f"formal Exp5-2 requires DPVO Python: {expected_dpvo}")
    if not vjepa_python.is_file() or not os.access(vjepa_python, os.X_OK):
        raise FileNotFoundError(f"configured V-JEPA Python is not executable: {vjepa_python}")
    if vjepa_python == expected_dpvo:
        raise RuntimeError("Exp5-2 requires distinct DPVO and V-JEPA Python environments")
    return {
        "mode": "separate_conda_environments",
        "dpvo_python": str(expected_dpvo),
        "vjepa_python": str(vjepa_python),
        "transport": "persistent_subprocess_stdio_json_base64",
        "shared_memory": False,
        "automatic_install": False,
        "passed": True,
    }


def _load_dpvo_image(record: dict[str, Any], calibration: np.ndarray) -> Any:
    import cv2
    import torch

    image = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(record["image_path"])
    fx, fy, cx, cy = calibration[:4]
    if len(calibration) > 4:
        matrix = np.eye(3)
        matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2] = fx, fy, cx, cy
        image = cv2.undistort(image, matrix, calibration[4:])
    height, width = image.shape[:2]
    image = image[: height - height % 16, : width - width % 16]
    return torch.from_numpy(image).permute(2, 0, 1).contiguous().cuda()


def _extract_teacher(
    *, record: dict[str, Any], patchifier: Any, calibration: np.ndarray,
) -> np.ndarray:
    import torch

    image = _load_dpvo_image(record, calibration)
    normalized = 2.0 * (image[None, None] / 255.0) - 0.5
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
        raw_fnet = patchifier.fnet(normalized)
    teacher = (raw_fnet.float() / 4.0)[0, 0].detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(teacher).all():
        raise RuntimeError("DPVO FNet teacher contains non-finite values")
    del image, normalized, raw_fnet
    return teacher


WorkerFactory = Callable[[dict[str, Any], str | Path], Any]


def extract_training_pairs(
    *, config: dict[str, Any], records_by_split: dict[str, list[dict[str, Any]]],
    work_dir: str | Path, worker_factory: WorkerFactory = DenseJepaWorkerClient,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Materialize train/val pairs; one persistent JEPA worker per sequence."""
    import torch
    from ...exp4.extraction import _load_patchifier

    if set(records_by_split) != {"train", "val"}:
        raise ValueError("Exp5-2 pair extraction requires train and val records")
    token_shape = tuple(int(value) for value in config["jepa"]["expected_token_shape"])
    teacher_shape = (
        int(config["feature_hook"]["dpvo_channels"]),
        *tuple(int(value) for value in config["feature_hook"]["expected_spatial_shape"]),
    )
    pair_bytes = sum(
        estimate_pair_bytes(
            len(records), token_shape=token_shape, teacher_shape=teacher_shape
        )
        for records in records_by_split.values()
    )
    memory_gate = validate_memory_capacity(
        required_bytes=pair_bytes,
        available_bytes=available_memory_bytes(),
        safety_factor=float(config["projection"]["memory_safety_factor"]),
    )
    pairs = {
        split: {
            "tokens": np.empty((len(records), *token_shape), dtype=np.float32),
            "teacher": np.empty((len(records), *teacher_shape), dtype=np.float32),
        }
        for split, records in records_by_split.items()
    }
    calibration = np.loadtxt(config["paths"]["calibration"], delimiter=" ")
    patchifier = _load_patchifier(Path(config["paths"]["dpvo_checkpoint"]))
    jepa_seconds = bridge_seconds = setup_seconds = 0.0
    lifecycles: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    work_dir = Path(work_dir)
    try:
        for split in ("train", "val"):
            records = records_by_split[split]
            client = worker_factory(config, work_dir / f"prepare_{split}_vjepa.json")
            split_failed = False
            try:
                for index, record in enumerate(records):
                    tokens, jepa_elapsed, bridge_elapsed = client.extract(record)
                    teacher = _extract_teacher(
                        record=record, patchifier=patchifier, calibration=calibration
                    )
                    if tokens.shape != token_shape or teacher.shape != teacher_shape:
                        raise RuntimeError(
                            f"dense pair shape mismatch: {tokens.shape}/{teacher.shape}"
                        )
                    pairs[split]["tokens"][index] = tokens
                    pairs[split]["teacher"][index] = teacher
                    jepa_seconds += jepa_elapsed
                    bridge_seconds += bridge_elapsed
                    del tokens, teacher
            except Exception:
                split_failed = True
                raise
            finally:
                client.close(force=split_failed)
                lifecycles[split] = client.lifecycle()
            if (
                not lifecycles[split]["cleanup_passed"]
                or lifecycles[split]["request_count"] != len(records)
            ):
                raise RuntimeError(f"{split} dense V-JEPA worker cleanup failed")
            setup_seconds += float(lifecycles[split]["worker_setup_time"])
    except Exception:
        pairs.clear()
        raise
    finally:
        del patchifier
        torch.cuda.empty_cache()
    return pairs, {
        "pair_storage": "preallocated_CPU_RAM_float32_only",
        "persistent_cache": False,
        "split_counts": {split: len(records) for split, records in records_by_split.items()},
        "token_shape": list(token_shape),
        "teacher_shape": list(teacher_shape),
        "teacher_contract": "raw Patchifier.fnet output / 4.0",
        "pair_bytes": pair_bytes,
        "memory_gate": memory_gate,
        "extraction_seconds": time.perf_counter() - started,
        "jepa_extraction_seconds": jepa_seconds,
        "environment_bridge_seconds": bridge_seconds,
        "worker_setup_seconds": setup_seconds,
        "worker_lifecycles": lifecycles,
        "worker_scope": "one_per_preparation_sequence",
        "uses_groundtruth": False,
        "uses_trajectory": False,
        "uses_future_frames": False,
    }


def evaluate_test_alignment_online(
    *, config: dict[str, Any], records: list[dict[str, Any]],
    learned: DenseJepaProjection, random_model: DenseJepaProjection,
    work_dir: str | Path, worker_factory: WorkerFactory = DenseJepaWorkerClient,
) -> dict[str, Any]:
    import torch
    from ...exp4.extraction import _load_patchifier

    calibration = np.loadtxt(config["paths"]["calibration"], delimiter=" ")
    patchifier = _load_patchifier(Path(config["paths"]["dpvo_checkpoint"]))
    learned.eval()
    random_model.eval()
    learned_sum = random_sum = 0.0
    vector_count = 0
    jepa_seconds = bridge_seconds = 0.0
    started = time.perf_counter()
    client = worker_factory(config, Path(work_dir) / "test_alignment_vjepa.json")
    failed = False
    try:
        for record in records:
            tokens_array, jepa_elapsed, bridge_elapsed = client.extract(record)
            teacher_array = _extract_teacher(
                record=record, patchifier=patchifier, calibration=calibration
            )
            tokens = torch.from_numpy(tokens_array[None]).cuda()
            teacher = torch.from_numpy(teacher_array[None]).cuda()
            target_size = tuple(int(value) for value in teacher.shape[-2:])
            with torch.inference_mode():
                learned_prediction = learned(tokens, target_size=target_size)
                random_prediction = random_model(tokens, target_size=target_size)
                learned_value = 1.0 - dense_spatial_cosine_loss(learned_prediction, teacher)
                random_value = 1.0 - dense_spatial_cosine_loss(random_prediction, teacher)
            vectors = int(teacher.shape[0] * teacher.shape[2] * teacher.shape[3])
            learned_sum += float(learned_value.item()) * vectors
            random_sum += float(random_value.item()) * vectors
            vector_count += vectors
            jepa_seconds += jepa_elapsed
            bridge_seconds += bridge_elapsed
            del tokens_array, teacher_array, tokens, teacher
            del learned_prediction, random_prediction, learned_value, random_value
    except Exception:
        failed = True
        raise
    finally:
        client.close(force=failed)
        lifecycle = client.lifecycle()
        del patchifier
        torch.cuda.empty_cache()
    if not lifecycle["cleanup_passed"] or lifecycle["request_count"] != len(records):
        raise RuntimeError("MH_05 alignment V-JEPA worker lifecycle failed")
    return {
        "sample_count": len(records),
        "spatial_vector_count": vector_count,
        "learned_mean_spatial_cosine_similarity": learned_sum / vector_count,
        "random_mean_spatial_cosine_similarity": random_sum / vector_count,
        "evaluation_seconds": time.perf_counter() - started,
        "jepa_extraction_seconds": jepa_seconds,
        "environment_bridge_seconds": bridge_seconds,
        "worker_lifecycle": lifecycle,
        "pair_retained": False,
    }


def _ate(poses: np.ndarray, timestamps: np.ndarray, groundtruth: str) -> dict[str, Any]:
    from evo.core import sync
    from evo.core.metrics import PoseRelation
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    import evo.main_ape as main_ape

    estimate = PoseTrajectory3D(
        positions_xyz=poses[:, :3],
        orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
        timestamps=timestamps.astype(np.float64),
    )
    reference = file_interface.read_tum_trajectory_file(groundtruth)
    reference, estimate = sync.associate_trajectories(reference, estimate)
    if estimate.num_poses == 0:
        raise RuntimeError("ground-truth association produced no poses")
    result = main_ape.ape(
        reference, estimate, est_name="Exp5-2",
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True,
    )
    return {
        "translation_rmse": float(result.stats["rmse"]),
        "associated_pose_count": int(estimate.num_poses),
        "alignment": "Sim(3)",
    }


def evaluate_dense_method(
    *, config: dict[str, Any], records: list[dict[str, Any]], method: str,
    checkpoint_path: str | Path, work_dir: str | Path,
    trajectory_path: str | Path | None = None,
    worker_factory: WorkerFactory = DenseJepaWorkerClient,
) -> dict[str, Any]:
    """Run one DPVO method directly in the formal DPVO main process."""
    import torch
    from dpvo.dpvo import DPVO
    from dpvo.stream import image_stream
    from ..dpvo_runner import _load_dpvo_config, _seed_everything, _validate_stream_record

    if method not in FUSION_METHODS and method not in ORACLE_METHODS:
        raise ValueError(f"invalid dense evaluation method: {method}")
    if not records:
        raise ValueError(f"invalid dense evaluation request: {method}")
    _seed_everything(int(config["experiment"]["seed"]), torch)
    runtime_cfg = _load_dpvo_config(config)
    projection = None
    if method == "random_dense_projection_fusion":
        projection = build_random_dense_projection(config, device=torch.device("cuda:0"))
    elif method in ("jepa_dense_projection_fusion", "oracle_jepa_missing_rgb"):
        projection, _ = load_dense_projection_checkpoint(
            checkpoint_path, device=torch.device("cuda:0")
        )
    oracle_mode = method == "oracle_jepa_missing_rgb"
    availability_frames: list[dict[str, Any]] = []
    availability_summary: dict[str, Any] | None = None
    if oracle_mode:
        availability_frames, availability_summary = frame_availability(
            records, keyframe_stride=int(config["oracle"]["keyframe_stride"])
        )
    client = None
    if projection is not None:
        label = f"{method}_{records[0]['sequence']}_vjepa.json"
        client = worker_factory(config, Path(work_dir) / label)
    image_dir = Path(records[0]["image_path"]).parent
    queue: Queue = Queue(maxsize=8)
    reader = Process(
        target=image_stream,
        args=(queue, str(image_dir), str(config["paths"]["calibration"]),
              int(config["dataset"]["stride"]), int(config["dataset"]["skip"])),
    )
    slam = hook = None
    poses = ordinals = None
    jepa_seconds = bridge_seconds = projection_seconds = dpvo_seconds = 0.0
    end_to_end_started: float | None = None
    reader.start()
    failed = False
    try:
        while True:
            stream_index, image_numpy, intrinsics_numpy = queue.get()
            if int(stream_index) < 0:
                break
            record = records[int(stream_index)]
            _validate_stream_record(int(stream_index), record)
            image = torch.from_numpy(image_numpy).permute(2, 0, 1).cuda()
            intrinsics = torch.from_numpy(intrinsics_numpy).cuda()
            if slam is None:
                slam = DPVO(
                    runtime_cfg, str(config["paths"]["dpvo_checkpoint"]),
                    ht=int(image.shape[1]), wd=int(image.shape[2]), viz=False,
                )
                if oracle_mode:
                    hook = OracleFNetReplacementHook(
                        alpha=float(config["fusion"]["alpha"]),
                        channels=int(config["feature_hook"]["dpvo_channels"]),
                        projection_to_raw_scale=float(config["feature_hook"]["fnet_scale"]),
                        replacement_dtype=(
                            torch.float16 if bool(runtime_cfg.MIXED_PRECISION)
                            else torch.float32
                        ),
                    )
                    hook.install(slam.network.patchify.fnet)
                elif projection is not None:
                    hook = FNetInjectionHook(
                        alpha=float(config["fusion"]["alpha"]),
                        channels=int(config["feature_hook"]["dpvo_channels"]),
                        projection_to_raw_scale=float(config["feature_hook"]["fnet_scale"]),
                    )
                    hook.install(slam.network.patchify.fnet)
                torch.cuda.synchronize()
                end_to_end_started = time.perf_counter()
            if projection is not None and client is not None and hook is not None:
                tokens_array, jepa_elapsed, bridge_elapsed = client.extract(record)
                jepa_seconds += jepa_elapsed
                bridge_seconds += bridge_elapsed
                torch.cuda.synchronize()
                projection_started = time.perf_counter()
                tokens = torch.from_numpy(tokens_array[None]).cuda()
                target_size = (
                    tuple(int(value) for value in config["feature_hook"]["expected_spatial_shape"])
                    if oracle_mode else
                    (int(image.shape[-2]) // 4, int(image.shape[-1]) // 4)
                )
                with torch.inference_mode():
                    projected = projection(tokens, target_size=target_size)
                torch.cuda.synchronize()
                projection_seconds += time.perf_counter() - projection_started
                if oracle_mode:
                    availability = availability_frames[int(stream_index)]
                    if (
                        int(availability["frame_id"]) != int(record["frame_id"])
                        or int(availability["timestamp"]) != int(record["timestamp"])
                    ):
                        raise RuntimeError("Oracle availability/frame dispatch mismatch")
                    hook.bind(
                        record, projected,
                        is_keyframe=bool(availability["is_keyframe"]),
                    )
                else:
                    hook.bind(record, projected)
                del tokens_array, tokens
            torch.cuda.synchronize()
            dpvo_started = time.perf_counter()
            with torch.inference_mode():
                slam(int(stream_index), image, intrinsics)
            torch.cuda.synchronize()
            dpvo_seconds += time.perf_counter() - dpvo_started
            if hook is not None:
                if oracle_mode:
                    hook.finish_frame(
                        record,
                        is_keyframe=bool(
                            availability_frames[int(stream_index)]["is_keyframe"]
                        ),
                    )
                else:
                    hook.finish_frame(record)
        if slam is None or end_to_end_started is None:
            raise RuntimeError("DPVO never initialized")
        torch.cuda.synchronize()
        terminate_started = time.perf_counter()
        with torch.inference_mode():
            poses, ordinals = slam.terminate()
        torch.cuda.synchronize()
        dpvo_seconds += time.perf_counter() - terminate_started
        end_to_end_seconds = time.perf_counter() - end_to_end_started
    except Exception:
        failed = True
        raise
    finally:
        if hook is not None:
            hook.close()
        if client is not None:
            client.close(force=failed)
        reader.join(timeout=30)
        if reader.is_alive():
            reader.terminate()
            reader.join(timeout=10)
    pose_array = np.asarray(poses, dtype=np.float64)
    ordinal_array = np.asarray(ordinals, dtype=np.float64)
    timestamps = np.asarray([record["timestamp"] for record in records], dtype=np.float64)
    identity_ok = bool(np.array_equal(ordinal_array, np.arange(len(records), dtype=np.float64)))
    finite = bool(pose_array.shape == (len(records), 7) and np.isfinite(pose_array).all())
    if trajectory_path is not None:
        trajectory_payload = build_estimated_trajectory(
            sequence=str(records[0]["sequence"]),
            method=method,
            records=records,
            ordinals=ordinal_array,
            poses=pose_array,
        )
        write_trajectory(trajectory_path, trajectory_payload)
    groundtruth = str(
        Path(config["repo_root"])
        / str(config["evaluation"]["groundtruth_pattern"]).format(
            sequence=records[0]["sequence"]
        )
    )
    ate = _ate(pose_array, timestamps, groundtruth) if finite and identity_ok else None
    hook_lifecycle = hook.lifecycle() if hook is not None else {
        "initialized": False, "removed": False, "cleanup_passed": True,
        "registration_mode": None, "consumed_frames": 0,
    }
    worker_lifecycle = client.lifecycle() if client is not None else {
        "used": False, "python": None, "initialized": False, "request_count": 0,
        "stopped": False, "cleanup_passed": True, "worker_setup_time": 0.0,
        "worker_extraction_time": 0.0, "environment_bridge_time": 0.0,
    }
    runtime = {
        "jepa_extraction_time": float(jepa_seconds),
        "environment_bridge_time": float(bridge_seconds),
        "worker_setup_time": float(worker_lifecycle["worker_setup_time"]),
        "dense_projection_time": float(projection_seconds),
        "fusion_hook_time": float(hook.total_seconds if hook is not None else 0.0),
        "dpvo_total_time": float(dpvo_seconds),
        "end_to_end_total_time": float(end_to_end_seconds),
    }
    if set(runtime) != RUNTIME_KEYS:
        raise AssertionError("dense runtime schema changed")
    tracking = {
        "completed": True,
        "pose_count": int(len(pose_array)),
        "expected_count": int(len(records)),
        "gt_associated_count": int(ate["associated_pose_count"] if ate else 0),
        "ordinal_timestamp_aligned": identity_ok,
        "poses_finite": finite,
    }
    tracking["success"] = bool(
        tracking["pose_count"] == tracking["expected_count"]
        and tracking["gt_associated_count"] > 0
        and tracking["ordinal_timestamp_aligned"] and tracking["poses_finite"]
        and hook_lifecycle["cleanup_passed"] and worker_lifecycle["cleanup_passed"]
        and (hook is None or hook_lifecycle["consumed_frames"] == len(records))
        and (client is None or worker_lifecycle["request_count"] == len(records))
    )
    if oracle_mode:
        assert availability_summary is not None
        oracle_counts_valid = bool(
            hook_lifecycle["fnet_executed_frames"]
            == availability_summary["keyframe_count"]
            and hook_lifecycle["jepa_replaced_frames"]
            == availability_summary["non_keyframe_count"]
        )
        tracking["success"] = bool(tracking["success"] and oracle_counts_valid)
        availability_summary["hook_counts_valid"] = oracle_counts_valid
    result = {
        "status": "ok", "method": method, "sequence": records[0]["sequence"],
        "num_frames": len(records), "ate": ate, "tracking": tracking,
        "runtime": runtime, "hook_lifecycle": hook_lifecycle,
        "worker_lifecycle": worker_lifecycle,
    }
    if oracle_mode:
        result["availability"] = availability_summary
    del slam, projection, hook, client
    torch.cuda.empty_cache()
    return result
