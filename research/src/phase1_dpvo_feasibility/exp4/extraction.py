"""Internal JEPA and DPVO feature extraction with the Exp3 Oracle gate."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f
import yaml

from .schema import (
    REPO_ROOT,
    Exp4Sample,
    atomic_write_json,
    identity_metadata,
    load_config,
    repo_path,
    sha256_file,
    validate_metadata_identity,
)


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phase1-exp4-dpvo")
SCRIPT_PATH = Path(__file__).resolve()
EXP3_ORACLE_PATH = SCRIPT_PATH.parent.parent / "exp3/oracle.py"
EXP3_RUNTIME_PATH = SCRIPT_PATH.parent.parent / "exp3/runtime.py"


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _environment() -> dict[str, Any]:
    return {
        "python_executable": os.path.realpath(os.sys.executable),
        "python_version": os.sys.version,
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate_fmap_shape(
    actual_shape: tuple[int, int, int],
    input_hw: tuple[int, int],
    fmap_config: dict[str, Any],
) -> list[str]:
    """Validate channels/scale and apply the configurable reference-shape policy."""
    channels = int(fmap_config["fmap_channels"])
    scale = int(fmap_config["spatial_scale"])
    if scale <= 0:
        raise ValueError("spatial_scale must be positive")
    if input_hw[0] % scale or input_hw[1] % scale:
        raise ValueError(f"processed input {input_hw} is not divisible by spatial_scale={scale}")
    derived = (channels, input_hw[0] // scale, input_hw[1] // scale)
    if actual_shape != derived:
        raise ValueError(f"FMap shape violates channels/spatial_scale: actual={actual_shape}, derived={derived}")
    reference = tuple(int(value) for value in fmap_config["reference_shape"])
    policy = str(fmap_config["shape_policy"])
    messages: list[str] = []
    if actual_shape != reference:
        message = f"FMap actual_shape={actual_shape} differs from reference_shape={reference}"
        if policy == "strict":
            raise ValueError(message)
        if policy == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            messages.append(message)
        else:
            raise ValueError(f"Unsupported shape_policy: {policy}")
    elif policy not in {"strict", "warn"}:
        raise ValueError(f"Unsupported shape_policy: {policy}")
    return messages


def load_exp4_frame(sample: Exp4Sample, calibration: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
    """Mirror the DPVO stream/Exp3 OpenCV BGR input contract."""
    image = cv2.imread(sample.image_path)
    if image is None:
        raise FileNotFoundError(sample.image_path)
    original_hw = tuple(int(value) for value in image.shape[:2])
    fx, fy, cx, cy = calibration[:4]
    undistorted = len(calibration) > 4
    if undistorted:
        matrix = np.eye(3)
        matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2] = fx, fy, cx, cy
        image = cv2.undistort(image, matrix, calibration[4:])
    height, width = image.shape[:2]
    image = image[:height - height % 16, :width - width % 16]
    processed_hw = tuple(int(value) for value in image.shape[:2])
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().cuda()
    metadata = {
        "loader": "OpenCV imread",
        "channel_order": "BGR (upstream DPVO contract)",
        "undistorted": undistorted,
        "calibration": [float(value) for value in calibration],
        "original_size_hw": list(original_hw),
        "processed_size_hw": list(processed_hw),
        "crop_multiple": 16,
        "input_normalization": "2 * (uint8 / 255.0) - 0.5",
        "fmap_scale": "Patchifier.fnet(normalized) / 4.0",
    }
    return tensor, metadata


@torch.no_grad()
def extract_new_fmap(patchifier: Any, image: torch.Tensor, mixed_precision: bool) -> torch.Tensor:
    normalized = 2.0 * (image[None, None] / 255.0) - 0.5
    with torch.cuda.amp.autocast(enabled=mixed_precision):
        return (patchifier.fnet(normalized) / 4.0).detach()


def _load_patchifier(checkpoint: Path) -> Any:
    from dpvo.net import Patchifier

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    prefixes = ("module.patchify.fnet.", "patchify.fnet.")
    fnet_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                fnet_state[key[len(prefix):]] = value
                break
    if not fnet_state:
        raise RuntimeError("DPVO checkpoint contains no patchify.fnet weights")
    patchifier = Patchifier(patch_size=3)
    patchifier.fnet.load_state_dict(fnet_state, strict=True)
    patchifier.fnet.cuda().eval()
    return patchifier


def _fingerprint(
    *, commit: str, checkpoint_sha256: str, dpvo_config_sha256: str,
    exp4_config_sha256: str,
) -> dict[str, Any]:
    payload = {
        "dpvo_commit": commit,
        "checkpoint_sha256": checkpoint_sha256,
        "dpvo_config_sha256": dpvo_config_sha256,
        "exp4_config_sha256": exp4_config_sha256,
        "extractor_source_sha256": sha256_file(SCRIPT_PATH),
        "exp3_oracle_source_sha256": sha256_file(EXP3_ORACLE_PATH),
        "exp3_runtime_source_sha256": sha256_file(EXP3_RUNTIME_PATH),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _sanity_diagnosis(record: dict[str, Any], thresholds: dict[str, float]) -> str:
    if record["input_shape_new"] != record["input_shape_reference"] or record["input_max_abs_error"] != 0.0:
        return "preprocessing_mismatch"
    if record["shape_new"] != record["shape_reference"]:
        return "shape_mismatch"
    if not record["finite_new"] or not record["finite_reference"]:
        return "non_finite"
    if record["dtype_new"] != record["dtype_reference"]:
        return "dtype_mismatch"
    if (
        record["max_abs_error"] > float(thresholds["max_abs_error"])
        or record["mean_abs_error"] > float(thresholds["mean_abs_error"])
        or record["cosine_similarity"] < float(thresholds["min_cosine_similarity"])
    ):
        return "teacher_definition_mismatch"
    return "none"


def compare_fmaps(new: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Pure metric helper used by the GPU sanity and CPU unit tests."""
    new_float = new.detach().float().cpu()
    reference_float = reference.detach().float().cpu()
    shape_new, shape_reference = list(new_float.shape), list(reference_float.shape)
    finite_new = bool(torch.isfinite(new_float).all().item())
    finite_reference = bool(torch.isfinite(reference_float).all().item())
    if shape_new == shape_reference and finite_new and finite_reference:
        difference = (new_float - reference_float).abs()
        max_error = float(difference.max())
        mean_error = float(difference.mean())
        cosine = float(torch_f.cosine_similarity(new_float.reshape(1, -1), reference_float.reshape(1, -1)).item())
    else:
        max_error = mean_error = cosine = None
    return {
        "shape_new": shape_new,
        "shape_reference": shape_reference,
        "dtype_new": str(new.dtype).replace("torch.", ""),
        "dtype_reference": str(reference.dtype).replace("torch.", ""),
        "finite_new": finite_new,
        "finite_reference": finite_reference,
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "cosine_similarity": cosine,
    }


def run_oracle_sanity(
    *, samples: list[Exp4Sample], patchifier: Any, calibration: np.ndarray,
    fmap_config: dict[str, Any], fingerprint: dict[str, Any],
) -> dict[str, Any]:
    from research.src.phase1_dpvo_feasibility.exp3.oracle import extract_oracle_fmap
    from research.src.phase1_dpvo_feasibility.exp3.runtime import _load_frame

    sanity_config = fmap_config["sanity"]
    count = min(int(sanity_config["sample_count"]), len(samples))
    rng = np.random.default_rng(int(sanity_config["seed"]))
    selected = sorted(rng.choice(len(samples), size=count, replace=False).tolist())
    mixed_precision = bool(fmap_config["mixed_precision"])
    shim = SimpleNamespace(
        cfg=SimpleNamespace(MIXED_PRECISION=mixed_precision),
        network=SimpleNamespace(patchify=patchifier),
    )
    records: list[dict[str, Any]] = []
    for index in selected:
        sample = samples[index]
        image_new, preprocessing = load_exp4_frame(sample, calibration)
        image_reference, _ = _load_frame({"image_path": sample.image_path}, calibration)
        input_difference = (image_new.to(torch.int16) - image_reference.to(torch.int16)).abs()
        raw_new = extract_new_fmap(patchifier, image_new, mixed_precision)
        raw_reference = extract_oracle_fmap(shim, image_reference).fmap
        record = {
            **identity_metadata(sample),
            "input_shape_new": list(image_new.shape),
            "input_shape_reference": list(image_reference.shape),
            "input_dtype_new": str(image_new.dtype).replace("torch.", ""),
            "input_dtype_reference": str(image_reference.dtype).replace("torch.", ""),
            "input_max_abs_error": float(input_difference.max()),
            "preprocessing": preprocessing,
            **compare_fmaps(raw_new, raw_reference),
        }
        squeezed_shape = tuple(int(value) for value in raw_new.shape[2:])
        try:
            record["shape_warnings"] = validate_fmap_shape(
                squeezed_shape, tuple(image_new.shape[-2:]), fmap_config
            )
        except ValueError as error:
            record["shape_validation_error"] = str(error)
        record["diagnosis"] = _sanity_diagnosis(record, sanity_config)
        if "shape_validation_error" in record:
            record["diagnosis"] = "shape_mismatch"
        record["passed"] = record["diagnosis"] == "none"
        records.append(record)
        del image_new, image_reference, raw_new, raw_reference

    max_errors = [record["max_abs_error"] for record in records if record["max_abs_error"] is not None]
    mean_errors = [record["mean_abs_error"] for record in records if record["mean_abs_error"] is not None]
    finite_cosines = [
        record["cosine_similarity"] for record in records
        if record["cosine_similarity"] is not None and np.isfinite(record["cosine_similarity"])
    ]
    result = {
        "passed": all(record["passed"] for record in records),
        "fingerprint": fingerprint,
        "reference": {
            "frame_loader": "research.src.phase1_dpvo_feasibility.exp3.runtime._load_frame",
            "extractor": "research.src.phase1_dpvo_feasibility.exp3.oracle.extract_oracle_fmap",
            "full_dpvo_instantiated": False,
        },
        "thresholds": {
            "max_abs_error": float(sanity_config["max_abs_error"]),
            "mean_abs_error": float(sanity_config["mean_abs_error"]),
            "min_cosine_similarity": float(sanity_config["min_cosine_similarity"]),
        },
        "records": records,
        "aggregate": {
            "sample_count": len(records),
            "worst_max_abs_error": max(max_errors) if max_errors else None,
            "worst_mean_abs_error": max(mean_errors) if mean_errors else None,
            "minimum_cosine_similarity": min(finite_cosines) if finite_cosines else None,
            "failure_diagnoses": sorted({record["diagnosis"] for record in records if not record["passed"]}),
        },
    }
    return result


def _cached_sanity(manifest_path: Path, fingerprint: dict[str, Any]) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sanity = manifest["sanity"]
        if sanity.get("passed") and sanity.get("fingerprint", {}).get("sha256") == fingerprint["sha256"]:
            sanity = dict(sanity)
            sanity["reused"] = True
            return sanity
    except Exception:
        return None
    return None


def require_passing_sanity(sanity: dict[str, Any]) -> None:
    """Hard gate between interface verification and formal feature writes."""
    if not sanity.get("passed"):
        diagnoses = sanity.get("aggregate", {}).get("failure_diagnoses", ["unknown"])
        raise RuntimeError(f"Exp3 Oracle sanity failed: {diagnoses}")


def failed_sanity_manifest(base_manifest: dict[str, Any], total: int) -> dict[str, Any]:
    return {**base_manifest, "status": "failed_sanity", "counts": {"written": 0, "total": total}}


def _fmap_existing_matches(path: Path, sample: Exp4Sample, config_sha256: str, checkpoint_sha256: str) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata, tensor = payload["metadata"], payload["fmap"]
        validate_metadata_identity(sample, metadata, str(path))
        return bool(
            metadata.get("feature_kind") == "dpvo_fmap"
            and metadata.get("config_sha256") == config_sha256
            and metadata.get("checkpoint_sha256") == checkpoint_sha256
            and tuple(tensor.shape) == sample.fmap_shape
            and torch.isfinite(tensor).all().item()
        )
    except Exception:
        return False


def extract_fmap(
    *, config_path: Path, samples: list[Exp4Sample], overwrite: bool,
    sanity_only: bool, dataset_root: Path,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("FMap extraction selection is empty")
    config, resolved_config, config_sha256 = load_config(config_path)
    root = dataset_root.resolve()
    dataset_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if dataset_manifest.get("config_sha256") != config_sha256:
        raise RuntimeError("Dataset index was prepared with a different Exp4 config")

    fmap_config = config["dpvo_fmap"]
    if config["storage"]["fmap_dtype"] != "float16" or config["storage"]["original_compute_dtype"] != "float32":
        raise ValueError("Exp4 expects canonical float32 FMaps stored as float16")
    checkpoint = repo_path(config["paths"]["dpvo_checkpoint"]).resolve()
    dpvo_config_path = repo_path(config["paths"]["dpvo_config"]).resolve()
    calibration_path = repo_path(config["paths"]["calibration"]).resolve()
    for required in (checkpoint, dpvo_config_path, calibration_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    checkpoint_sha256 = sha256_file(checkpoint)
    dpvo_config_sha256 = sha256_file(dpvo_config_path)
    if checkpoint_sha256 != fmap_config["checkpoint_sha256"]:
        raise RuntimeError("DPVO checkpoint SHA256 mismatch")
    if dpvo_config_sha256 != fmap_config["config_sha256"]:
        raise RuntimeError("DPVO config SHA256 mismatch")
    upstream_config = yaml.safe_load(dpvo_config_path.read_text(encoding="utf-8"))
    if bool(upstream_config["MIXED_PRECISION"]) != bool(fmap_config["mixed_precision"]):
        raise RuntimeError("Exp4 mixed_precision differs from the DPVO config")

    pending: list[Exp4Sample] = []
    skipped = 0
    for sample in samples:
        destination = sample.feature_path(root, "dpvo_fmap")
        if destination.exists() and not overwrite:
            if _fmap_existing_matches(destination, sample, config_sha256, checkpoint_sha256):
                skipped += 1
                continue
            raise RuntimeError(f"Conflicting existing feature (use --overwrite): {destination}")
        pending.append(sample)
    if not torch.cuda.is_available():
        raise RuntimeError("DPVO FMap extraction requires CUDA")

    commit = _git_commit(REPO_ROOT)
    fingerprint = _fingerprint(
        commit=commit, checkpoint_sha256=checkpoint_sha256,
        dpvo_config_sha256=dpvo_config_sha256, exp4_config_sha256=config_sha256,
    )
    extraction_manifest_path = root / "features/dpvo_fmap/extraction_manifest.json"
    patchifier = _load_patchifier(checkpoint)
    calibration = np.loadtxt(calibration_path, delimiter=" ")
    sanity = _cached_sanity(extraction_manifest_path, fingerprint)
    if sanity is None:
        sanity = run_oracle_sanity(
            samples=samples, patchifier=patchifier, calibration=calibration,
            fmap_config=fmap_config, fingerprint=fingerprint,
        )
        sanity["reused"] = False
    base_manifest = {
        "schema_version": 1,
        "feature_kind": "dpvo_fmap",
        "config_path": str(resolved_config),
        "config_sha256": config_sha256,
        "dpvo": {
            "repo_commit": commit,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "config": str(dpvo_config_path),
            "dpvo_config_sha256": dpvo_config_sha256,
            "execution_boundary": "Patchifier.fnet only; no DPVO/trajectory/patch sampling/correlation/update/BA",
        },
        "shape_contract": {
            "fmap_channels": int(fmap_config["fmap_channels"]),
            "spatial_scale": int(fmap_config["spatial_scale"]),
            "reference_shape": list(fmap_config["reference_shape"]),
            "shape_policy": fmap_config["shape_policy"],
        },
        "normalization": config["normalization"],
        "storage": config["storage"],
        "environment": _environment(),
        "selection": {"splits": sorted({sample.split for sample in samples}), "sample_count": len(samples)},
        "sanity": sanity,
    }
    if not sanity["passed"]:
        failed = failed_sanity_manifest(base_manifest, len(samples))
        atomic_write_json(extraction_manifest_path, failed)
        require_passing_sanity(sanity)
    if sanity_only:
        manifest = {**base_manifest, "status": "sanity_passed", "counts": {"written": 0, "total": len(samples)}}
        actual_shapes = sorted({tuple(record["shape_new"][2:]) for record in sanity["records"]})
        manifest["shape_contract"]["actual_shapes"] = [list(shape) for shape in actual_shapes]
        manifest["shape_contract"]["actual_shape"] = list(actual_shapes[0]) if len(actual_shapes) == 1 else None
        atomic_write_json(extraction_manifest_path, manifest)
        del patchifier
        torch.cuda.empty_cache()
        return manifest

    mixed_precision = bool(fmap_config["mixed_precision"])
    written = 0
    observed_shapes: set[tuple[int, int, int]] = set()
    shape_warnings: set[str] = set()
    started = time.perf_counter()
    for sample in pending:
        destination = sample.feature_path(root, "dpvo_fmap")
        image, preprocessing = load_exp4_frame(sample, calibration)
        raw = extract_new_fmap(patchifier, image, mixed_precision)
        canonical = raw[0, 0].float().cpu()
        actual_shape = tuple(int(value) for value in canonical.shape)
        observed_shapes.add(actual_shape)
        shape_warnings.update(validate_fmap_shape(actual_shape, tuple(image.shape[-2:]), fmap_config))
        if not torch.isfinite(canonical).all().item():
            raise RuntimeError(f"Non-finite DPVO FMap for {sample.identity}")
        metadata = {
            **identity_metadata(sample),
            "feature_kind": "dpvo_fmap",
            "dpvo_commit": commit,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "dpvo_config": str(dpvo_config_path),
            "dpvo_config_sha256": dpvo_config_sha256,
            "preprocessing": preprocessing,
            "normalization": config["normalization"]["dpvo_fmap"],
            "compute_dtype": config["storage"]["original_compute_dtype"],
            "inference_autocast_dtype": str(raw.dtype).replace("torch.", ""),
            "storage_dtype": config["storage"]["fmap_dtype"],
            "shape": list(actual_shape),
            "reference_shape": list(fmap_config["reference_shape"]),
            "shape_policy": fmap_config["shape_policy"],
            "config_sha256": config_sha256,
        }
        _atomic_torch_save(destination, {"fmap": canonical.to(torch.float16), "metadata": metadata})
        written += 1
        del image, raw, canonical

    if not observed_shapes and skipped:
        observed_shapes.update(tuple(record["shape_new"][2:]) for record in sanity["records"])
    manifest = {
        **base_manifest,
        "status": "complete",
        "counts": {"written": written, "skipped_valid_existing": skipped, "total": len(samples)},
        "runtime": {"total_seconds": time.perf_counter() - started},
    }
    manifest["shape_contract"]["actual_shapes"] = [list(shape) for shape in sorted(observed_shapes)]
    manifest["shape_contract"]["actual_shape"] = list(next(iter(observed_shapes))) if len(observed_shapes) == 1 else None
    manifest["shape_contract"]["warnings"] = sorted(shape_warnings)
    atomic_write_json(extraction_manifest_path, manifest)
    del patchifier
    torch.cuda.empty_cache()
    return manifest


def _jepa_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _validate_jepa_config(config: dict[str, Any]) -> None:
    jepa = config["jepa"]
    requested = (jepa["encoder"], jepa["layer"], jepa["token_type"], jepa["normalization"])
    supported = ("ema", "final", "dense", "none")
    if requested != supported:
        raise NotImplementedError(
            f"Unsupported Exp4 V-JEPA representation {requested}; supported={supported}"
        )
    storage = config["storage"]
    if storage["jepa_dtype"] != "float16" or storage["original_compute_dtype"] != "float32":
        raise ValueError("Exp4 expects canonical float32 tokens stored as float16")


def _jepa_existing_matches(
    path: Path, sample: Exp4Sample, config_sha256: str, checkpoint_sha256: str,
) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata, tensor = payload["metadata"], payload["tokens"]
        validate_metadata_identity(sample, metadata, str(path))
        return bool(
            metadata.get("feature_kind") == "jepa_tokens"
            and metadata.get("config_sha256") == config_sha256
            and metadata.get("checkpoint_sha256") == checkpoint_sha256
            and tuple(tensor.shape) == sample.jepa_shape
            and torch.isfinite(tensor).all().item()
        )
    except Exception:
        return False


def extract_jepa(
    *, config_path: Path, samples: list[Exp4Sample], overwrite: bool,
    dataset_root: Path,
) -> dict[str, Any]:
    """Extract final EMA dense tokens inside the fixed V-JEPA environment."""
    if not samples:
        raise ValueError("JEPA extraction selection is empty")
    config, resolved_config, config_sha256 = load_config(config_path)
    _validate_jepa_config(config)
    root = dataset_root.resolve()
    dataset_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if dataset_manifest.get("config_sha256") != config_sha256:
        raise RuntimeError("Dataset index was prepared with a different Exp4 config")

    jepa = config["jepa"]
    vjepa_root = Path(jepa["repo"]).resolve()
    expected_executable = Path(config["runtime"]["jepa_python"]).resolve()
    if Path(sys.executable).resolve() != expected_executable:
        raise RuntimeError(f"Wrong V-JEPA Python: expected {expected_executable}, got {sys.executable}")
    commit = _jepa_git(vjepa_root, "rev-parse", "HEAD")
    if commit != jepa["expected_git_commit"]:
        raise RuntimeError(f"V-JEPA commit mismatch: expected {jepa['expected_git_commit']}, got {commit}")
    status_before = _jepa_git(vjepa_root, "status", "--short")
    checkpoint = vjepa_root / jepa["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != jepa["checkpoint_sha256"]:
        raise RuntimeError("V-JEPA checkpoint SHA256 mismatch")
    expected_shape = tuple(int(value) for value in jepa["expected_shape"])
    batch_size = int(jepa["batch_size"])
    pending: list[Exp4Sample] = []
    skipped = 0
    for sample in samples:
        destination = sample.feature_path(root, "jepa")
        if destination.exists() and not overwrite:
            if _jepa_existing_matches(destination, sample, config_sha256, checkpoint_sha256):
                skipped += 1
                continue
            raise RuntimeError(f"Conflicting existing feature: {destination}")
        pending.append(sample)

    if pending and not torch.cuda.is_available():
        raise RuntimeError("V-JEPA extraction requires CUDA")
    sys.path.insert(0, str(vjepa_root))
    import research as research_package
    vjepa_research = str(vjepa_root / "research")
    if vjepa_research not in research_package.__path__:
        research_package.__path__ = [*research_package.__path__, vjepa_research]
    from research.scripts.common.dense_pca import extract_frame_features, load_phase2_encoder  # type: ignore[import-not-found]

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    encoder = load_phase2_encoder(device) if pending else None
    written = 0
    forward_seconds = 0.0
    observed_shapes: set[tuple[int, ...]] = set()
    observed_grids: set[tuple[int, int]] = set()
    observed_patch_sizes: set[int] = set()
    started = time.perf_counter()
    try:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            batch = extract_frame_features(
                frame_paths=[Path(sample.image_path) for sample in chunk],
                encoder=encoder,
                device=device,
                frame_indices=[sample.frame_id for sample in chunk],
                crop_size=int(jepa["crop_size"]),
                batch_size=batch_size,
            )
            forward_seconds += float(batch.runtime["encoder_forward_seconds"])
            grid_shape = tuple(int(value) for value in batch.grid_shape)
            patch_size = int(batch.patch_size)
            observed_grids.add(grid_shape)
            observed_patch_sizes.add(patch_size)
            if grid_shape != tuple(jepa["expected_grid_shape"]) or patch_size != int(jepa["patch_size"]):
                raise RuntimeError(f"Unexpected JEPA grid/patch: grid={grid_shape}, patch_size={patch_size}")
            for sample, grid, preprocessing in zip(chunk, batch.features, batch.crop_metadata):
                canonical = torch.from_numpy(np.asarray(grid, dtype=np.float32)).reshape(-1, grid.shape[-1])
                actual_shape = tuple(canonical.shape)
                observed_shapes.add(actual_shape)
                if actual_shape != expected_shape or actual_shape != sample.jepa_shape:
                    raise RuntimeError(f"Unexpected JEPA shape for {sample.identity}: {actual_shape}")
                if not torch.isfinite(canonical).all().item():
                    raise RuntimeError(f"Non-finite JEPA tokens for {sample.identity}")
                metadata = {
                    **identity_metadata(sample),
                    "feature_kind": "jepa_tokens",
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha256,
                    "repo_commit": commit,
                    "model_alias": jepa["model_alias"],
                    "model_name": jepa["model_name"],
                    "encoder": jepa["encoder"],
                    "layer": jepa["layer"],
                    "token_type": jepa["token_type"],
                    "token_grid_shape": list(grid_shape),
                    "patch_size": patch_size,
                    "preprocessing": preprocessing,
                    "normalization": config["normalization"]["jepa"],
                    "compute_dtype": config["storage"]["original_compute_dtype"],
                    "inference_autocast_dtype": "bfloat16",
                    "storage_dtype": config["storage"]["jepa_dtype"],
                    "shape": list(actual_shape),
                    "config_sha256": config_sha256,
                }
                _atomic_torch_save(
                    sample.feature_path(root, "jepa"),
                    {"tokens": canonical.to(torch.float16), "metadata": metadata},
                )
                written += 1
            del batch
    finally:
        if encoder is not None:
            del encoder
            torch.cuda.empty_cache()

    status_after = _jepa_git(vjepa_root, "status", "--short")
    if status_after != status_before:
        raise RuntimeError("V-JEPA worktree changed during extraction")
    if not observed_shapes and skipped:
        observed_shapes.add(expected_shape)
        observed_grids.add(tuple(int(value) for value in jepa["expected_grid_shape"]))
        observed_patch_sizes.add(int(jepa["patch_size"]))
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "feature_kind": "jepa_tokens",
        "config_path": str(resolved_config),
        "config_sha256": config_sha256,
        "selection": {"splits": sorted({sample.split for sample in samples}), "sample_count": len(samples)},
        "counts": {"written": written, "skipped_valid_existing": skipped, "total": len(samples)},
        "vjepa": {
            "repo": str(vjepa_root),
            "repo_commit": commit,
            "worktree_status_before": status_before.splitlines(),
            "worktree_status_after": status_after.splitlines(),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "model_alias": jepa["model_alias"],
            "model_name": jepa["model_name"],
            "encoder": jepa["encoder"],
            "layer": jepa["layer"],
            "token_type": jepa["token_type"],
            "observed_shapes": [list(shape) for shape in sorted(observed_shapes)],
            "observed_grid_shapes": [list(shape) for shape in sorted(observed_grids)],
            "observed_patch_sizes": sorted(observed_patch_sizes),
        },
        "normalization": config["normalization"],
        "storage": config["storage"],
        "environment": _environment(),
        "runtime": {
            "encoder_forward_seconds": forward_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    atomic_write_json(root / "features/jepa/extraction_manifest.json", manifest)
    return manifest


def jepa_worker(config_path: str, dataset_root: str) -> None:
    """Internal subprocess target; intentionally not a public CLI."""
    from .schema import select_samples

    result = extract_jepa(
        config_path=Path(config_path),
        samples=select_samples(dataset_root, "all"),
        overwrite=False,
        dataset_root=Path(dataset_root),
    )
    print(json.dumps({
        "status": result["status"],
        "feature_kind": result["feature_kind"],
        "counts": result["counts"],
    }, sort_keys=True))
