"""Bridge provenance and fail-closed compatibility checks."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from latent_vslam.canonical import load_yaml
from latent_vslam.jepa_fmap import BRIDGE_ARCHITECTURE, build_bridge
from prediction.jepa_runtime import state_dict_sha256
from latent_vslam.protocol import REPO_ROOT, canonical_sha256, load_sequence_records, sha256_file
from latent_vslam.manifests import base_lineage
from latent_vslam.schema import VISUAL_STATE_CONTRACT_SHA256

BRIDGE_CONFIG_PATH = REPO_ROOT / "configs/train/bridge_mh01.yaml"
BRIDGE_TRAINING_SEQUENCE = "MH_01_easy"
# The path-independent config hash changes once in A1. This narrow exception
# permits only the already-published Bridge, with its original path-bound hash.
LEGACY_BRIDGE_CONFIG_SHA256 = "ed69f7224ac1246f1ca1d701bd0dd2d36d3fd107c403d13f6fa88c52fe6d5029"
PATH_INDEPENDENT_BRIDGE_CONFIG_SHA256 = "f4e11b5d4cdacbe2edd7e8c3c0bf27fed9a43f7eedccb4c5e73ab659f766b05e"
LEGACY_BRIDGE_SHA256 = "2e82adabdf0a7c7e3a7662610a5a586c1c36fdb76afa2f7f8a475e58342ff91d"
BRIDGE_TRAINING_SOURCE_NAMES = (
    "bridge_training_runtime.py", "bridge_training.py", "bridge_checkpoint.py", "jepa_fmap.py",
    "jepa_runtime.py", "jepa_worker.py", "oracle_packet.py", "protocol.py", "schema.py",
)


def load_bridge_config(path: str | Path = BRIDGE_CONFIG_PATH) -> tuple[dict[str, Any], Path]:
    config, resolved = load_yaml(path)
    fixed = {
        "seed": 1234, "bridge_initialization_seed": 1236,
        "training_sequence": BRIDGE_TRAINING_SEQUENCE, "bootstrap_accepted_nodes": 8,
        "post_bootstrap_anchor_interval": 5,
    }
    for key, value in fixed.items():
        if config["experiment"].get(key) != value:
            raise ValueError(f"frozen Bridge field changed: {key}")
    ratio = float(config["experiment"]["anchor_ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError("anchor_ratio must be in (0, 1]")
    required = {
        "hidden_channels": 160, "passes": 2, "epochs_per_pass": 30,
        "batch_size": 4, "learning_rate": 1e-4, "weight_decay": 1e-4,
        "cosine_weight": 1.0, "smooth_l1_weight": 0.1,
        "reset_optimizer_and_grad_scaler_between_passes": True,
        "repeat_epoch_local_batch_order_each_pass": True,
    }
    for key, value in required.items():
        if config["bridge"].get(key) != value:
            raise ValueError(f"frozen Bridge recipe changed: {key}")
    return config, resolved


def bridge_training_sources() -> tuple[Path, ...]:
    package = Path(__file__).parent
    prediction_sources = {"jepa_runtime.py", "jepa_worker.py"}
    return tuple(
        package.parent / "prediction" / name if name in prediction_sources else package / name
        for name in BRIDGE_TRAINING_SOURCE_NAMES
    )


def bridge_training_context(
    config: Mapping[str, Any],
) -> tuple[Sequence[Any], dict[str, Any], dict[str, Any]]:
    records = load_sequence_records(config, BRIDGE_TRAINING_SEQUENCE)
    training_config = copy.deepcopy(dict(config))
    training_config.pop("evaluation", None)
    base, provenance = base_lineage(
        training_config, BRIDGE_TRAINING_SEQUENCE, bridge_training_sources(),
        h0_contract_sha256=VISUAL_STATE_CONTRACT_SHA256,
    )
    return records, base, provenance


def bridge_training_input(base: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in base.items()
        if key not in {"schedule_sha256", "h1_bridge_sha256", "h2_predictor_sha256"}
    }


def load_compatible_bridge(
    path: Path, transform: Any, *, hidden_channels: int,
    expected_training_input: Mapping[str, Any],
    expected_training_lineage: Mapping[str, Any] | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Load a bridge only when its contents and current Bridge lineage agree."""
    if not path.is_file():
        raise RuntimeError(f"Bridge is missing: {path}; run python -m latent_vslam.train --config configs/train/bridge_mh01.yaml first")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(f"Bridge is unreadable: {path}; run python -m latent_vslam.train --config configs/train/bridge_mh01.yaml first") from error
    forbidden = {"optimizer", "optimizer_state_dict", "grad_scaler", "best_state"}
    if forbidden & set(checkpoint):
        raise RuntimeError("Bridge contains training state; run python -m latent_vslam.train --config configs/train/bridge_mh01.yaml first")
    if checkpoint.get("schema_version") != 2 or not isinstance(
        checkpoint.get("training_recipe"), Mapping
    ):
        raise RuntimeError("Bridge schema is incompatible; run python -m latent_vslam.train --config configs/train/bridge_mh01.yaml first")
    if checkpoint.get("architecture") != BRIDGE_ARCHITECTURE:
        raise RuntimeError("Bridge architecture is incompatible; run python -m latent_vslam.train --config configs/train/bridge_mh01.yaml first")
    if checkpoint.get("layer_zero_based") != 5 or "state_dict" not in checkpoint:
        raise RuntimeError("Bridge metadata is incomplete; run python -m latent_vslam.train --config configs/train/bridge_mh01.yaml first")
    if state_dict_sha256(checkpoint["state_dict"]) != checkpoint.get("state_dict_sha256"):
        raise RuntimeError("Bridge state integrity failed; run python -m latent_vslam.train --config configs/train/bridge_mh01.yaml first")
    lineage = checkpoint.get("training_lineage")
    if not isinstance(lineage, Mapping):
        raise RuntimeError("Bridge has no training lineage; run python -m latent_vslam.train --config configs/train/bridge_mh01.yaml first")
    lineage_body = {key: value for key, value in lineage.items()
                    if key != "training_lineage_sha256"}
    if lineage.get("training_lineage_sha256") != canonical_sha256(lineage_body):
        raise RuntimeError("Bridge lineage integrity failed; run python -m latent_vslam.train --config configs/train/bridge_mh01.yaml first")
    from latent_vslam.scientific_lineage import compatible_training_input
    stored_input = lineage.get("training_input", {})
    compatible = compatible_training_input(stored_input, expected_training_input)
    if not compatible and expected_training_lineage is None:
        compatible = (
            expected_training_input.get("config_protocol_sha256") == PATH_INDEPENDENT_BRIDGE_CONFIG_SHA256
            and stored_input.get("config_protocol_sha256") == LEGACY_BRIDGE_CONFIG_SHA256
            and sha256_file(path) == LEGACY_BRIDGE_SHA256
            and compatible_training_input(
                stored_input,
                dict(expected_training_input) | {"config_protocol_sha256": LEGACY_BRIDGE_CONFIG_SHA256},
            )
        )
    if not compatible:
        raise RuntimeError(
            "Bridge is incompatible with current Bridge config/source/protocol; "
            "run python -m latent_vslam.train --config configs/train/bridge_mh01.yaml first"
        )
    if expected_training_lineage is not None and dict(lineage) != dict(expected_training_lineage):
        raise RuntimeError("fresh Bridge lineage validation failed")
    if checkpoint.get("coordinate_protocol") != lineage.get("coordinate_protocol"):
        raise RuntimeError("Bridge coordinate protocol is incompatible; run python -m latent_vslam.train --config configs/train/bridge_mh01.yaml first")
    model = build_bridge(transform, channels=int(hidden_channels)).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.requires_grad_(False)
    metadata = {
        "file": os.path.relpath(path, REPO_ROOT), "file_sha256": sha256_file(path),
        "architecture": checkpoint["architecture"], "layer_zero_based": 5,
        "coordinate_protocol": checkpoint["coordinate_protocol"],
        "training_lineage_sha256": lineage["training_lineage_sha256"],
        "loaded_prior_results_json": False, "bridge_retrained_by_inference": False,
    }
    return model, metadata, checkpoint
