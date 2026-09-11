"""Shared V-JEPA/DPVO feature IO used by final H1 and H2 runners."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

from .jepa_fmap import derive_full_fov_transform, preprocess_full_fov_rgb
from .protocol import REPO_ROOT, FrameIdentity, atomic_write_json, canonical_sha256, repo_path


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash a state dict without serializing training or compatibility state."""
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def load_dpvo_domain(path: str, calibration: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    raw_size = tuple(int(value) for value in image.shape[:2])
    fx, fy, cx, cy = calibration[:4]
    if len(calibration) > 4:
        matrix = np.eye(3)
        matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2] = fx, fy, cx, cy
        image = cv2.undistort(image, matrix, calibration[4:])
    height, width = image.shape[:2]
    return image[:height - height % 16, :width - width % 16].copy(), raw_size


def load_fnet(checkpoint: Path) -> torch.nn.Module:
    from dpvo.net import Patchifier
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fnet_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for prefix in ("module.patchify.fnet.", "patchify.fnet."):
            if key.startswith(prefix):
                fnet_state[key[len(prefix):]] = value
                break
    if not fnet_state:
        raise RuntimeError("DPVO checkpoint contains no FNet weights")
    patchifier = Patchifier(3)
    patchifier.fnet.load_state_dict(fnet_state, strict=True)
    return patchifier.fnet.requires_grad_(False).cuda().eval()


@torch.no_grad()
def teacher_fmap(model: torch.nn.Module, bgr: np.ndarray,
                 expected_hw: tuple[int, int]) -> np.ndarray:
    image = torch.from_numpy(bgr).permute(2, 0, 1).cuda()
    normalized = 2.0 * (image[None, None].float() / 255.0) - 0.5
    with torch.cuda.amp.autocast(enabled=True):
        fmap = model(normalized) / 4.0
    result = fmap[0, 0].detach().cpu().numpy().astype(np.float16)
    if result.shape != (128, *expected_hw) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid DPVO teacher FMap: {result.shape}")
    return result


def sequence_geometry(record: Any, calibration: np.ndarray,
                      config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    bgr, raw_size = load_dpvo_domain(record.rgb_path, calibration)
    transform = derive_full_fov_transform(
        bgr.shape[0], bgr.shape[1], target_height=int(config["jepa"]["target_height"]),
        patch_size=int(config["jepa"]["patch_size"]),
        fmap_scale=int(config["teacher"]["fmap_scale"]),
    )
    payload = transform.payload()
    payload["raw_source_height"], payload["raw_source_width"] = raw_size
    payload["dpvo_domain_height"], payload["dpvo_domain_width"] = bgr.shape[:2]
    payload.pop("transform_sha256")
    payload["transform_sha256"] = canonical_sha256(payload)
    return transform, payload


class JepaSidecar:
    def __init__(self, config: Mapping[str, Any], temporary: Path) -> None:
        from .execution_runtime import capture_runtime
        worker_config = dict(config)
        worker_config["worker_settings"] = worker_config.get(
            "worker_settings",
            capture_runtime(int(config.get("experiment", {}).get("seed", 1234))),
        )
        self.config_path = temporary / "jepa_worker_config.json"
        atomic_write_json(self.config_path, worker_config)
        self.process = subprocess.Popen(
            [str(worker_config["runtime"]["jepa_python"]), "-m",
             "research.src.jepa_worker",
             "--config", str(self.config_path)],
            cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        ready = self._receive()
        if ready.get("status") != "ready":
            raise RuntimeError(f"V-JEPA worker startup failed: {ready}")
        self.provenance = dict(ready["provenance"])
        self.provenance["setup_seconds"] = float(ready.get("setup_seconds", 0.0))
        self.provenance["peak_gpu_memory_allocated_bytes"] = 0

    def _receive(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            assert self.process.stderr is not None
            raise RuntimeError(f"V-JEPA worker exited: {self.process.stderr.read()}")
        return json.loads(line)

    def extract(self, source: Path, destination: Path, request_id: str) -> dict[str, Any]:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({
            "action": "extract", "request_id": request_id,
            "input_npy": str(source), "block5_npy": str(destination),
        }) + "\n")
        self.process.stdin.flush()
        result = self._receive()
        if result.get("status") != "ok" or not result.get("finite"):
            raise RuntimeError(f"V-JEPA extraction failed: {result}")
        self.provenance["peak_gpu_memory_allocated_bytes"] = max(
            int(self.provenance["peak_gpu_memory_allocated_bytes"]),
            int(result.get("peak_gpu_memory_allocated_bytes", 0)),
        )
        return result

    def prepare_online(self, request_id: str = "online") -> dict[str, Any]:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({
            "action": "prepare_online", "request_id": request_id,
        }) + "\n")
        self.process.stdin.flush()
        result = self._receive()
        if (result.get("status") != "online_ready"
                or result.get("request_id") != request_id
                or not result.get("worker_cuda_synchronized")
                or not result.get("worker_peak_memory_reset")):
            raise RuntimeError(f"V-JEPA worker online-ready barrier failed: {result}")
        self.provenance["peak_gpu_memory_allocated_bytes"] = 0
        return result

    def flush_online(self, request_id: str = "online") -> dict[str, Any]:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({
            "action": "flush_online", "request_id": request_id,
        }) + "\n")
        self.process.stdin.flush()
        result = self._receive()
        if (result.get("status") != "online_flushed"
                or result.get("request_id") != request_id
                or not result.get("worker_cuda_synchronized")):
            raise RuntimeError(f"V-JEPA worker online flush barrier failed: {result}")
        self.provenance["peak_gpu_memory_allocated_bytes"] = max(
            int(self.provenance["peak_gpu_memory_allocated_bytes"]),
            int(result.get("peak_gpu_memory_allocated_bytes", 0)),
        )
        return result

    def __enter__(self) -> "JepaSidecar":
        return self

    def __exit__(self, error_type: object, *_: object) -> None:
        if self.process.poll() is not None:
            return
        if error_type is None:
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps({"action": "close"}) + "\n")
            self.process.stdin.flush()
            self._receive()
            self.process.wait(timeout=20)
        else:
            self.process.kill()
            self.process.wait(timeout=20)


class CompactFeatureStore:
    """Identity-scoped temporary store that never retains RGB paths."""

    def __init__(self, root: Path, name: str, identities: Sequence[FrameIdentity],
                 shape: Sequence[int]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.identities = tuple(identities)
        keys = [item.key for item in identities]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("feature identities must be non-empty and unique")
        self.index = {key: offset for offset, key in enumerate(keys)}
        self.values = np.memmap(root / f"{name}.mmap", mode="w+", dtype=np.float16,
                                shape=(len(keys), *shape))
        self.written = np.zeros(len(keys), dtype=bool)
        self.contains_rgb_or_path = False
        self.closed = False

    def put(self, identity: FrameIdentity, value: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("feature store is closed")
        row = self.index[identity.key]
        if self.written[row] or tuple(value.shape) != tuple(self.values.shape[1:]):
            raise RuntimeError(f"invalid or duplicate feature: {identity.key}")
        if not np.isfinite(value).all():
            raise RuntimeError(f"non-finite feature: {identity.key}")
        self.values[row] = value.astype(np.float16, copy=False)
        self.written[row] = True

    def get(self, identity_or_key: FrameIdentity | str) -> np.ndarray:
        if self.closed:
            raise RuntimeError("feature store is closed")
        key = identity_or_key.key if isinstance(identity_or_key, FrameIdentity) else identity_or_key
        row = self.index[key]
        if not self.written[row]:
            raise RuntimeError(f"unwritten feature: {key}")
        return np.asarray(self.values[row], dtype=np.float32)

    def finalize(self) -> dict[str, Any]:
        if not self.written.all():
            raise RuntimeError("feature store incomplete")
        self.values.flush()
        return {"identity_count": len(self.identities),
                "identity_sha256": canonical_sha256([item.key for item in self.identities]),
                "dtype": "float16", "shape_per_identity": list(self.values.shape[1:]),
                "contains_rgb_or_path": False}

    def close(self) -> None:
        mmap = getattr(self.values, "_mmap", None)
        if mmap is not None:
            mmap.close()
        self.closed = True


class RestrictedFeatureView:
    def __init__(self, store: CompactFeatureStore, allowed: set[str], capability: str) -> None:
        self.__store = store
        self.__allowed = frozenset(allowed)
        self.capability = capability
        self.contains_rgb_or_path = False
        self.read_count = 0
        self.read_keys: list[str] = []

    def get(self, identity_or_key: FrameIdentity | str) -> np.ndarray:
        key = identity_or_key.key if isinstance(identity_or_key, FrameIdentity) else identity_or_key
        if key not in self.__allowed:
            raise PermissionError(f"{self.capability} cannot access {key}")
        value = self.__store.get(key)
        self.read_count += 1
        self.read_keys.append(key)
        return value

    @property
    def allowed_identity_sha256(self) -> str:
        return canonical_sha256(sorted(self.__allowed))

    def usage_payload(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "allowed_identity_count": len(self.__allowed),
            "allowed_identity_sha256": self.allowed_identity_sha256,
            "read_count": self.read_count,
            "read_identity_sha256": canonical_sha256(self.read_keys),
            "store_closed": self.__store.closed,
        }


def extract_block5_store(
    records: Sequence[Any], identities: Sequence[FrameIdentity], calibration: np.ndarray,
    config: Mapping[str, Any], temporary: Path, transform: Any,
) -> tuple[CompactFeatureStore, dict[str, Any]]:
    by_key = {item.identity.key: item for item in records}
    store = CompactFeatureStore(
        temporary / "features", "block5", identities,
        (transform.token_grid_height * transform.token_grid_width, 768),
    )
    timings: list[float] = []
    with JepaSidecar(config, temporary) as sidecar:
        size = int(config["jepa"]["batch_size"])
        for start in range(0, len(identities), size):
            batch = identities[start:start + size]
            tensors = []
            for identity in batch:
                bgr, _ = load_dpvo_domain(by_key[identity.key].rgb_path, calibration)
                tensor, _ = preprocess_full_fov_rgb(bgr[..., ::-1].copy(), transform)
                tensors.append(tensor.numpy())
            source = temporary / f"jepa_in_{start}.npy"
            output = temporary / f"jepa_out_{start}.npy"
            np.save(source, np.stack(tensors), allow_pickle=False)
            response = sidecar.extract(source, output, str(start))
            timings.append(float(response["encoder_inference_ms"]))
            values = np.load(output)
            for identity, value in zip(batch, values):
                store.put(identity, value)
            source.unlink()
            output.unlink()
    return store, {"store": store.finalize(), "jepa": sidecar.provenance,
                   "encoder_inference_ms": timings}


def extract_true_fmap_store(
    records: Sequence[Any], identities: Sequence[FrameIdentity], calibration: np.ndarray,
    config: Mapping[str, Any], temporary: Path, transform: Any,
) -> tuple[CompactFeatureStore, dict[str, Any]]:
    by_key = {item.identity.key: item for item in records}
    store = CompactFeatureStore(
        temporary / "true_features", "true_fmap", identities,
        (128, transform.fmap_height, transform.fmap_width),
    )
    model = load_fnet(repo_path(config["paths"]["dpvo_checkpoint"]))
    torch.cuda.reset_peak_memory_stats()
    for identity in identities:
        bgr, _ = load_dpvo_domain(by_key[identity.key].rgb_path, calibration)
        store.put(identity, teacher_fmap(
            model, bgr, (transform.fmap_height, transform.fmap_width),
        ))
    peak = int(torch.cuda.max_memory_allocated())
    del model
    torch.cuda.empty_cache()
    return store, {"store": store.finalize(), "peak_gpu_vram_bytes": peak}
