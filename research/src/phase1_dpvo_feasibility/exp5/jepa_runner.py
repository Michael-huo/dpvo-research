"""Persistent V-JEPA sidecar and CPU-only mock for Exp5-0."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO


PROTOCOL_PREFIX = "EXP5_JEPA_JSON:"
FRAME_PAYLOAD_KEYS = frozenset(("image_path", "frame_id", "timestamp"))


def _emit(payload: dict[str, Any]) -> None:
    print(PROTOCOL_PREFIX + json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def sidecar_frame_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Expose only the immutable frame-dispatch identity to V-JEPA."""
    payload = {
        "image_path": str(record["image_path"]),
        "frame_id": int(record["frame_id"]),
        "timestamp": int(record["timestamp"]),
    }
    if set(payload) != FRAME_PAYLOAD_KEYS:
        raise AssertionError("V-JEPA frame payload contract changed")
    return payload


def mock_jepa_record(record: dict[str, Any], runtime: float = 0.0005) -> dict[str, Any]:
    """Return a deterministic, explicitly synthetic model result for smoke."""
    return {
        "timestamp": int(record["timestamp"]),
        "frame_id": int(record["frame_id"]),
        "jepa_shape": [768],
        "extraction_time": float(runtime),
        "latent_norm": 2.0,
        "latent_mean": 0.125,
        "latent_std": 0.25,
        "finite_check": True,
        "mock": True,
    }


class JepaSidecar:
    """Line-oriented client for the persistent worker in the V-JEPA environment."""

    def __init__(self, config: dict[str, Any], request_path: str | Path) -> None:
        request_path = Path(request_path).resolve()
        worker_request = request_path.with_name(f"{request_path.stem}_jepa.json")
        worker_request.write_text(
            json.dumps({
                "config": {
                    "repo_root": str(config["repo_root"]),
                    "runtime": {"jepa_python": str(config["runtime"]["jepa_python"])},
                    "jepa": dict(config["jepa"]),
                }
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command = [
            str(Path(config["runtime"]["jepa_python"]).resolve()),
            "-c",
            (
                "from research.src.phase1_dpvo_feasibility.exp5.jepa_runner "
                "import worker_main; raise SystemExit(worker_main())"
            ),
            "--request-json",
            str(worker_request),
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
        ready = self._read_message()
        if ready.get("status") != "ready":
            self.close(force=True)
            raise RuntimeError(f"V-JEPA sidecar did not become ready: {ready}")
        self.setup_runtime = float(ready["setup_runtime"])

    def _read_message(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("V-JEPA sidecar stdout is unavailable")
        while True:
            line = self.process.stdout.readline()
            if not line:
                code = self.process.poll()
                tail = "\n".join(self.diagnostics[-20:])
                raise RuntimeError(f"V-JEPA sidecar exited unexpectedly ({code}): {tail}")
            stripped = line.rstrip()
            if stripped.startswith(PROTOCOL_PREFIX):
                payload = json.loads(stripped[len(PROTOCOL_PREFIX):])
                if payload.get("status") == "error":
                    raise RuntimeError(f"V-JEPA sidecar failed: {payload.get('error')}")
                return payload
            self.diagnostics.append(stripped)

    def _send(self, payload: dict[str, Any]) -> None:
        if self.process.stdin is None or self.process.poll() is not None:
            raise RuntimeError("V-JEPA sidecar is not running")
        self.process.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
        self.process.stdin.flush()

    def extract(self, record: dict[str, Any]) -> dict[str, Any]:
        self._send({"action": "extract", "record": sidecar_frame_payload(record)})
        response = self._read_message()
        if response.get("status") != "ok":
            raise RuntimeError(f"unexpected V-JEPA response: {response}")
        return dict(response["record"])

    def close(self, *, force: bool = False) -> None:
        if getattr(self, "process", None) is None or self.process.poll() is not None:
            return
        if not force:
            try:
                self._send({"action": "stop"})
                response = self._read_message()
                if response.get("status") != "stopped":
                    raise RuntimeError(f"unexpected V-JEPA shutdown response: {response}")
                self.process.wait(timeout=30)
                return
            except Exception:
                force = True
        if force:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)

    def __enter__(self) -> "JepaSidecar":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close(force=exc is not None)


def _load_worker(config: dict[str, Any]) -> tuple[Any, Any, Any, str, Path, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("formal V-JEPA sidecar requires CUDA")
    repo = Path(config["jepa"]["repo"]).resolve()
    expected_python = Path(config["runtime"]["jepa_python"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"wrong V-JEPA Python: expected {expected_python}, got {sys.executable}")
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != str(config["jepa"]["expected_git_commit"]):
        raise RuntimeError(
            f"V-JEPA commit mismatch: expected {config['jepa']['expected_git_commit']}, got {commit}"
        )
    status_before = _git(repo, "status", "--short")
    checkpoint = repo / str(config["jepa"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if _sha256(checkpoint) != str(config["jepa"]["checkpoint_sha256"]):
        raise RuntimeError("V-JEPA checkpoint SHA256 mismatch")

    sys.path.insert(0, str(repo))
    import research as research_package

    sibling_research = str(repo / "research")
    if sibling_research not in research_package.__path__:
        research_package.__path__ = [*research_package.__path__, sibling_research]
    from research.scripts.common.dense_pca import extract_frame_features, load_phase2_encoder

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    encoder = load_phase2_encoder(device)
    encoder.requires_grad_(False).eval()
    return torch, encoder, extract_frame_features, commit, repo, status_before


def _extract_one(
    *, torch: Any, encoder: Any, extractor: Any, config: dict[str, Any], record: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    started = time.perf_counter()
    with torch.inference_mode():
        batch = extractor(
            frame_paths=[Path(record["image_path"])],
            encoder=encoder,
            device=torch.device("cuda:0"),
            frame_indices=[int(record["frame_id"])],
            crop_size=int(config["jepa"]["crop_size"]),
            batch_size=1,
        )
    dense = np.asarray(batch.features[0], dtype=np.float32)
    expected_dense = tuple(int(value) for value in config["jepa"]["expected_dense_shape"])
    if tuple(dense.shape) != expected_dense:
        raise RuntimeError(f"unexpected V-JEPA dense shape: {tuple(dense.shape)} != {expected_dense}")
    latent = dense.mean(axis=(0, 1), dtype=np.float32)
    expected_global = tuple(int(value) for value in config["jepa"]["expected_global_shape"])
    if tuple(latent.shape) != expected_global:
        raise RuntimeError(f"unexpected pooled latent shape: {tuple(latent.shape)} != {expected_global}")
    result = {
        "timestamp": int(record["timestamp"]),
        "frame_id": int(record["frame_id"]),
        "jepa_shape": list(latent.shape),
        "extraction_time": float(time.perf_counter() - started),
        "latent_norm": float(np.linalg.norm(latent)),
        "latent_mean": float(latent.mean()),
        "latent_std": float(latent.std()),
        "finite_check": bool(np.isfinite(latent).all()),
    }
    del batch, dense, latent
    return result


def _worker_loop(config: dict[str, Any], input_stream: TextIO) -> int:
    import torch as torch_module

    setup_started = time.perf_counter()
    torch, encoder, extractor, _, repo, status_before = _load_worker(config)
    _emit({"status": "ready", "setup_runtime": time.perf_counter() - setup_started})
    try:
        for line in input_stream:
            request = json.loads(line)
            action = request.get("action")
            if action == "extract":
                result = _extract_one(
                    torch=torch,
                    encoder=encoder,
                    extractor=extractor,
                    config=config,
                    record=dict(request["record"]),
                )
                _emit({"status": "ok", "record": result})
            elif action == "stop":
                status_after = _git(repo, "status", "--short")
                if status_after != status_before:
                    raise RuntimeError("V-JEPA worktree changed during Exp5-0 extraction")
                _emit({"status": "stopped"})
                return 0
            else:
                raise ValueError(f"unsupported V-JEPA sidecar action: {action}")
        raise RuntimeError("V-JEPA sidecar stdin closed without a stop request")
    finally:
        del encoder
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()


def worker_main() -> int:
    """Internal subprocess entry invoked by the sole public Exp5 CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request-json", required=True)
    args = parser.parse_args()
    try:
        request = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
        return _worker_loop(dict(request["config"]), sys.stdin)
    except Exception as error:
        _emit({"status": "error", "error": repr(error), "traceback": traceback.format_exc()})
        return 1
