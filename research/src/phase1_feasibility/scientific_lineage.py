"""Explicit scientific contracts, independent of execution implementation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .protocol import canonical_sha256


COMMON_CONTRACT = {
    "phase": "phase1_feasibility",
    "identity_and_schedule": "frame_identity_post_bootstrap_ratio_schedule_v1",
    "visual_state": "dpvo_fmap_zero_context_v1",
    "jepa_layer": 5,
    "jepa_storage_dtype": "float16",
    "coordinate_protocol": "full_fov_patch14_to_dpvo_fmap4_v1",
    "evaluation": "canonical_population_sim3_ate_rpe_v1",
}

MODULE_CONTRACTS = {
    "h1": {
        "module": "h1_interface",
        "architecture": "token_to_fmap_bridge_v1",
        "training": (
            "scratch_seed1236_30epochs_then_seed1234_optimizer_scaler_reset_"
            "same_batches_30epochs_final_state"
        ),
        "batch_size": 4,
        "objective": "masked_cosine_plus_0.1_smooth_l1",
        "optimizer": "adamw_lr1e-4_weight_decay1e-4",
    },
    "h2": {
        "module": "h2_prediction",
        "architecture": "robust_transport_block5_residual_predictor_v1",
        "training": "seed1234_30epochs_best_validation",
        "batch_size_intervals": 2,
        "objective": "canonical_prediction_loss_v1",
        "transport": "bidirectional_robust_correspondence_irls3_v1",
        "calibration": "train_only_frozen_thresholds_v1",
        "deployment": "strict_delayed_bracketed_hidden_h1_h4_then_anchor_a5_v1",
    },
}


def scientific_fingerprint(module: str) -> dict:
    if module not in MODULE_CONTRACTS:
        raise ValueError(module)
    contract = COMMON_CONTRACT | MODULE_CONTRACTS[module]
    payload = {
        "version": "phase1_explicit_scientific_contract_v1",
        "module": module,
        "contract": contract,
    }
    return payload | {"source_sha256": canonical_sha256(payload)}


def compatible_training_input(actual: Mapping, expected: Mapping, *, module="h1") -> bool:
    actual = dict(actual)
    expected = dict(expected)
    if actual == expected:
        return True
    source = actual.get("source_sha256")
    path = Path(__file__).with_name("legacy_scientific_lineage.json")
    migrations = json.loads(path.read_text())
    migration = migrations.get(module, {}).get(source)
    if migration is None:
        return False
    current = scientific_fingerprint(module)["source_sha256"]
    if migration["scientific_source_sha256"] != current:
        return False
    actual["source_sha256"] = current
    return actual == expected
