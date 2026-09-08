"""Persistent V-JEPA sidecar for canonical full-FOV block-5 extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def _load(config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Exp6 V-JEPA worker requires CUDA")
    expected_python = Path(config["runtime"]["jepa_python"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"wrong V-JEPA Python: {sys.executable} != {expected_python}")
    repo = Path(config["jepa"]["repo"]).resolve()
    checkpoint = repo / str(config["jepa"]["checkpoint"])
    if not checkpoint.is_file() or _sha256(checkpoint) != str(config["jepa"]["checkpoint_sha256"]):
        raise RuntimeError("V-JEPA checkpoint missing or SHA256 mismatch")
    commit = _git(repo, "rev-parse", "HEAD")
    expected_commit = str(config["jepa"].get("expected_git_commit", commit))
    if commit != expected_commit:
        raise RuntimeError(f"V-JEPA source commit mismatch: {commit} != {expected_commit}")
    sys.path.insert(0, str(repo))
    import research as research_package

    provider_research = str(repo / "research")
    if provider_research not in research_package.__path__:
        research_package.__path__ = [*research_package.__path__, provider_research]
    from research.scripts.common.dense_pca import load_phase2_encoder

    device = torch.device("cuda:0")
    encoder = load_phase2_encoder(device).requires_grad_(False).eval()
    encoder.out_layers = [5]
    return torch, encoder, {
        "vjepa_git_commit": commit,
        "vjepa_git_dirty": bool(_git(repo, "status", "--short")),
        "checkpoint_sha256": _sha256(checkpoint),
        "layers_zero_based": [5],
        "worker_pid": int(os.getpid()),
        "logical_cuda_ordinal": 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_properties(0).name,
    }


def worker(config: dict[str, Any]) -> int:
    import numpy as np

    started = time.perf_counter()
    torch, encoder, provenance = _load(config)
    torch.cuda.reset_peak_memory_stats()
    _emit({"status": "ready", "setup_seconds": time.perf_counter() - started,
           "provenance": provenance})
    for line in sys.stdin:
        request = json.loads(line)
        action = request.get("action")
        if action == "close":
            _emit({"status": "closed"})
            return 0
        if action == "prepare_online":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            _emit({
                "status": "online_ready", "request_id": request.get("request_id"),
                "worker_cuda_synchronized": True, "worker_peak_memory_reset": True,
            })
            continue
        if action == "flush_online":
            torch.cuda.synchronize()
            _emit({
                "status": "online_flushed", "request_id": request.get("request_id"),
                "worker_cuda_synchronized": True,
                "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            })
            continue
        if action != "extract":
            raise ValueError(f"unsupported worker action: {action}")
        source = np.load(request["input_npy"], mmap_mode="r")
        if source.ndim != 5 or source.shape[1] != 3 or source.shape[2] != 1:
            raise ValueError(f"expected preprocessed [B,3,1,H,W], got {source.shape}")
        batch = torch.from_numpy(np.asarray(source, dtype=np.float32)).to("cuda:0")
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            outputs = encoder(batch)
            end_event.record()
            end_event.synchronize()
            encoder_inference_ms = float(start_event.elapsed_time(end_event))
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise RuntimeError("encoder did not return block 5")
        array = outputs[0].float().cpu().numpy().astype(np.float16)
        expected_tokens = (int(source.shape[-2]) // 16) * (int(source.shape[-1]) // 16)
        if array.shape[1:] != (expected_tokens, 768):
            raise RuntimeError(f"unexpected V-JEPA token shape: {array.shape}")
        np.save(request["block5_npy"], array, allow_pickle=False)
        _emit({"status": "ok", "request_id": request["request_id"],
               "block5_shape": list(array.shape),
               "finite": bool(np.isfinite(array).all()),
               "encoder_inference_ms": encoder_inference_ms,
               "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated())})
        del batch, outputs, array
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        return worker(config)
    except Exception as error:
        _emit({"status": "error", "error_type": type(error).__name__, "error": str(error)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
