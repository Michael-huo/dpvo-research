"""Canonical H1 bridge provenance and fail-closed compatibility checks."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .canonical import load_yaml
from .jepa_fmap import BRIDGE_ARCHITECTURE, build_bridge
from .jepa_runtime import state_dict_sha256
from .protocol import REPO_ROOT, canonical_sha256, load_sequence_records, sha256_file
from .registry import base_lineage
from .schema import VISUAL_STATE_CONTRACT_SHA256

H1_CONFIG_PATH = REPO_ROOT / "research/configs/phase1_feasibility_h1.yaml"
H1_TRAINING_SEQUENCE = "MH_01_easy"
H1_TRAINING_SOURCE_NAMES = (
    "run_h1.py", "h1_training.py", "bridge_checkpoint.py", "jepa_fmap.py",
    "jepa_runtime.py", "jepa_worker.py", "oracle_packet.py", "protocol.py", "schema.py",
)


def load_h1_config(path: str | Path = H1_CONFIG_PATH) -> tuple[dict[str, Any], Path]:
    config, resolved = load_yaml(path)
    fixed = {
        "seed": 1234, "bridge_initialization_seed": 1236,
        "training_sequence": H1_TRAINING_SEQUENCE, "bootstrap_accepted_nodes": 8,
        "post_bootstrap_anchor_interval": 5,
    }
    for key, value in fixed.items():
        if config["experiment"].get(key) != value:
            raise ValueError(f"frozen H1 field changed: {key}")
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
            raise ValueError(f"frozen H1 bridge recipe changed: {key}")
    return config, resolved


def h1_training_sources() -> tuple[Path, ...]:
    package = Path(__file__).parent
    return tuple(package / name for name in H1_TRAINING_SOURCE_NAMES)


def h1_training_context(
    config: Mapping[str, Any],
) -> tuple[Sequence[Any], dict[str, Any], dict[str, Any]]:
    records = load_sequence_records(config, H1_TRAINING_SEQUENCE)
    training_config = copy.deepcopy(dict(config))
    training_config.pop("evaluation", None)
    base, provenance = base_lineage(
        training_config, H1_TRAINING_SEQUENCE, h1_training_sources(),
        h0_contract_sha256=VISUAL_STATE_CONTRACT_SHA256,
    )
    return records, base, provenance


def h1_training_input(base: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in base.items()
        if key not in {"schedule_sha256", "h1_bridge_sha256", "h2_predictor_sha256"}
    }


def load_compatible_bridge(
    path: Path, transform: Any, *, hidden_channels: int,
    expected_training_input: Mapping[str, Any],
    expected_training_lineage: Mapping[str, Any] | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Load a bridge only when its contents and current H1 lineage agree."""
    if not path.is_file():
        raise RuntimeError(f"canonical H1 bridge is missing: {path}; run run_h1 first")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(f"canonical H1 bridge is unreadable: {path}; run run_h1 first") from error
    forbidden = {"optimizer", "optimizer_state_dict", "grad_scaler", "best_state"}
    if forbidden & set(checkpoint):
        raise RuntimeError("canonical H1 bridge contains training state; run run_h1 first")
    if checkpoint.get("schema_version") != 2 or not isinstance(
        checkpoint.get("training_recipe"), Mapping
    ):
        raise RuntimeError("canonical H1 bridge schema is incompatible; run run_h1 first")
    if checkpoint.get("architecture") != BRIDGE_ARCHITECTURE:
        raise RuntimeError("canonical H1 bridge architecture is incompatible; run run_h1 first")
    if checkpoint.get("layer_zero_based") != 5 or "state_dict" not in checkpoint:
        raise RuntimeError("canonical H1 bridge metadata is incomplete; run run_h1 first")
    if state_dict_sha256(checkpoint["state_dict"]) != checkpoint.get("state_dict_sha256"):
        raise RuntimeError("canonical H1 bridge state integrity failed; run run_h1 first")
    lineage = checkpoint.get("training_lineage")
    if not isinstance(lineage, Mapping):
        raise RuntimeError("canonical H1 bridge has no training lineage; run run_h1 first")
    lineage_body = {key: value for key, value in lineage.items()
                    if key != "training_lineage_sha256"}
    if lineage.get("training_lineage_sha256") != canonical_sha256(lineage_body):
        raise RuntimeError("canonical H1 bridge lineage integrity failed; run run_h1 first")
    from .scientific_lineage import compatible_training_input
    if not compatible_training_input(lineage.get("training_input", {}), expected_training_input):
        raise RuntimeError(
            "canonical H1 bridge is incompatible with current H1 config/source/protocol; "
            "run run_h1 first"
        )
    if expected_training_lineage is not None and dict(lineage) != dict(expected_training_lineage):
        raise RuntimeError("fresh H1 bridge lineage validation failed")
    if checkpoint.get("coordinate_protocol") != lineage.get("coordinate_protocol"):
        raise RuntimeError("canonical H1 bridge coordinate protocol is incompatible; run run_h1 first")
    model = build_bridge(transform, channels=int(hidden_channels)).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.requires_grad_(False)
    metadata = {
        "file": str(path.relative_to(REPO_ROOT)), "file_sha256": sha256_file(path),
        "architecture": checkpoint["architecture"], "layer_zero_based": 5,
        "coordinate_protocol": checkpoint["coordinate_protocol"],
        "training_lineage_sha256": lineage["training_lineage_sha256"],
        "loaded_h1_results_json": False, "bridge_retrained_by_h2": False,
    }
    return model, metadata, checkpoint
