"""Predictor training artifacts, independent of admission and deployment."""
from __future__ import annotations

import copy
import io
from pathlib import Path

import torch

from .predictor import RobustTransportBlock5Predictor, predictor_metadata, predictor_state_sha256
from .protocol import atomic_write_bytes, canonical_sha256, repo_path, sha256_file
from .transport import robust_protocol_metadata


def new_predictor(config):
    value = config["predictor"]
    return RobustTransportBlock5Predictor(
        int(value["feature_dim"]), int(value["hidden_dim"]),
        int(value["difference_dim"]), int(value["reliability_dim"]),
        int(value["residual_blocks"]), int(value["group_norm_groups"]),
        int(value["time_hidden_dim"]),
    )


def training_contract(config):
    """Explicit training inputs: never hash deployment, result paths or admission."""
    jepa = {k: copy.deepcopy(v) for k, v in config["jepa"].items()
            if k not in {"repo", "checkpoint"}}
    return {
        "predictor": copy.deepcopy(config["predictor"]), "jepa": jepa,
        "dataset_sampling": {k: copy.deepcopy(v) for k, v in config["dataset"].items()
                             if k not in {"root", "groundtruth_pattern"}},
        "calibration_sha256": sha256_file(repo_path(config["paths"]["calibration"])),
        "representation": "raw_block5_float16_store_float32_predictor_input",
        "transport": copy.deepcopy(config["transport"]),
        "training": copy.deepcopy(config["training"]),
        "seed": int(config["experiment"]["seed"]),
        "training_sequence": config["experiment"]["training_sequence"],
        "target": "offline_oracle_hidden_jepa_block5",
        "transport_protocol_sha256": canonical_sha256(robust_protocol_metadata()),
    }


def build_training_lineage(*, config, anchor_stride, anchor_population,
                           fixed_split, split_sha256, schedule_sha256,
                           coordinate_transform_sha256):
    return {
        "anchor_stride": anchor_stride,
        "actual_train_anchor_population": copy.deepcopy(anchor_population),
        "actual_train_anchor_count": len(anchor_population),
        "actual_train_anchor_population_sha256": canonical_sha256(anchor_population),
        "fixed_split_definition": copy.deepcopy(fixed_split),
        "split_sha256": split_sha256, "training_schedule_sha256": schedule_sha256,
        "coordinate_transform_sha256": coordinate_transform_sha256,
        "scientific_training_contract": training_contract(config),
        "seed": int(config["experiment"]["seed"]),
        "training_recipe": copy.deepcopy(config["training"]),
        "initialization": "fresh_new_predictor_no_checkpoint_resume",
    }


def save_predictor(path, model, config, calibration, lineage, *, best_epoch=None):
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    payload = {
        "schema_version": 1, "state_dict": state,
        "state_dict_sha256": predictor_state_sha256(state),
        "architecture": predictor_metadata(model),
        "training_recipe": dict(config["training"]) | {
            "seed": int(config["experiment"]["seed"]),
            "training_sequence": config["experiment"]["training_sequence"],
            "target": "offline_oracle_hidden_jepa_block5",
            "checkpoint_contains_optimizer_or_scaler": False,
        },
        "train_only_calibration": dict(calibration),
        "training_lineage": dict(lineage), "best_epoch": best_epoch,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_write_bytes(Path(path), buffer.getvalue())
    return payload


def load_predictor(path, config, expected_lineage=None, *, expected_state_sha256=None):
    """CPU validation of the current training envelope; no training fallback."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1 or {
            "optimizer", "optimizer_state_dict", "grad_scaler", "best_state",
            "deployment_protocol"} & set(checkpoint):
        raise RuntimeError("predictor checkpoint schema mismatch")
    state_hash = predictor_state_sha256(checkpoint["state_dict"])
    if (state_hash != checkpoint.get("state_dict_sha256")
            or expected_state_sha256 is not None and state_hash != expected_state_sha256):
        raise RuntimeError("predictor frozen state integrity mismatch")
    lineage = checkpoint["training_lineage"]
    body = {k: v for k, v in lineage.items() if k != "training_lineage_sha256"}
    if canonical_sha256(body) != lineage.get("training_lineage_sha256"):
        raise RuntimeError("predictor training lineage integrity mismatch")
    if expected_lineage is not None and lineage != expected_lineage:
        raise RuntimeError("predictor training lineage mismatch")
    if lineage.get("scientific_training_contract") != training_contract(config):
        raise RuntimeError("predictor scientific training contract mismatch")
    expected_recipe = dict(config["training"]) | {
        "seed": int(config["experiment"]["seed"]),
        "training_sequence": config["experiment"]["training_sequence"],
        "target": "offline_oracle_hidden_jepa_block5",
        "checkpoint_contains_optimizer_or_scaler": False,
    }
    if checkpoint.get("training_recipe") != expected_recipe:
        raise RuntimeError("predictor training recipe mismatch")
    calibration = checkpoint["train_only_calibration"]
    digest = canonical_sha256({k: v for k, v in calibration.items() if k != "calibration_sha256"})
    if digest != calibration.get("calibration_sha256") or digest != lineage.get("train_only_calibration_sha256"):
        raise RuntimeError("predictor train calibration integrity mismatch")
    # Construction must not consume the caller's scientific RNG stream.
    with torch.random.fork_rng(devices=[]):
        model = new_predictor(config)
        if checkpoint.get("architecture") != predictor_metadata(model):
            raise RuntimeError("predictor architecture mismatch")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    return checkpoint


def load_canonical_predictor(config):
    binding = config["canonical_predictor"]
    path = repo_path(config["paths"]["h2_predictor"])
    checkpoint = load_predictor(path, config, expected_state_sha256=binding["state_dict_sha256"])
    if checkpoint["training_lineage"]["training_lineage_sha256"] != binding["training_lineage_sha256"]:
        raise RuntimeError("canonical predictor training lineage mismatch")
    if checkpoint.get("best_epoch") != binding["best_epoch"]:
        raise RuntimeError("canonical predictor selection metadata mismatch")
    return checkpoint, path, sha256_file(path)
