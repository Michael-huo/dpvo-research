"""Explicit scientific contracts, independent of execution implementation."""
from __future__ import annotations

from typing import Mapping

from .protocol import canonical_sha256
from .uniform_admission import ADMISSION_CONTRACT
from .observation_sampling import SAMPLING_PROTOCOL


COMMON_CONTRACT = {
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
        "deployment": "strict_delayed_bracketed_uniform_budgeted_native_dpvo",
        "admission": ADMISSION_CONTRACT,
        "observation_sampling": SAMPLING_PROTOCOL,
    },
}


def scientific_fingerprint(module: str) -> dict:
    if module not in MODULE_CONTRACTS:
        raise ValueError(module)
    contract = COMMON_CONTRACT | MODULE_CONTRACTS[module]
    payload = {
        "version": "research_explicit_scientific_contract_v1",
        "module": module,
        "contract": contract,
    }
    return payload | {"source_sha256": canonical_sha256(payload)}


def compatible_training_input(actual: Mapping, expected: Mapping, *, module="h1") -> bool:
    return dict(actual) == dict(expected)


def deployment_lineage(config, *, predictor_sha256, training_lineage_sha256,
                       bridge_sha256, dpvo_sha256, dpvo_config_sha256,
                       schedule_sha256):
    payload = {
        "contract": scientific_fingerprint("h2"),
        "admission": dict(config["admission"]),
        "predictor_state_dict_sha256": predictor_sha256,
        "predictor_training_lineage_sha256": training_lineage_sha256,
        "h1_bridge_sha256": bridge_sha256,
        "dpvo_checkpoint_sha256": dpvo_sha256,
        "dpvo_config_sha256": dpvo_config_sha256,
        "schedule_sha256": schedule_sha256,
        "scientific_seed": config["experiment"]["seed"],
        "evaluation": dict(config["evaluation"]),
    }
    return payload | {"deployment_lineage_sha256": canonical_sha256(payload)}
