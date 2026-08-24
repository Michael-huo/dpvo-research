"""Internal training API for the final Exp4 bridge experiments."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..dataset import AdapterDataset, dataset_fingerprint
from ..schema import atomic_write_json
from .loss import feature_adapter_loss
from .model import (
    build_trainable_model,
    count_parameters,
    model_config_for_kind,
    model_config_sha256,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str = "cuda") -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def atomic_torch_save(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_dataset_sha256: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"malformed Exp4 checkpoint: {checkpoint_path}")
    if "model_state_dict" not in payload:
        raise ValueError(f"checkpoint has no model_state_dict: {checkpoint_path}")
    observed = payload["metadata"].get("dataset_fingerprint", {}).get("sha256")
    if expected_dataset_sha256 is not None and observed != expected_dataset_sha256:
        raise ValueError(
            f"checkpoint dataset fingerprint mismatch: expected {expected_dataset_sha256}, got {observed}"
        )
    return payload


def model_identity_metadata(
    *, model_kind: str, model_config: dict[str, Any], capacity: str | None,
    parameter_count: int,
) -> dict[str, Any]:
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    return {
        "model_kind": model_kind,
        "model_config": model_config,
        "model_config_sha256": model_config_sha256(model_config),
        "model_parameter_count": int(parameter_count),
        "parameter_count": int(parameter_count),
        "capacity": capacity,
        "hidden_dim": model_config.get("hidden_dim"),
    }


def _run_epoch(
    *, model: torch.nn.Module, loader: DataLoader, device: torch.device,
    loss_config: dict[str, Any], optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "cosine": 0.0, "l2": 0.0}
    samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            tokens = batch["jepa_tokens"].to(device=device, non_blocking=True)
            teacher = batch["fmap_teacher"].to(device=device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(tokens)
            losses = feature_adapter_loss(
                prediction,
                teacher,
                cosine_weight=float(loss_config["cosine_weight"]),
                l2_weight=float(loss_config["l2_weight"]),
                eps=float(loss_config["eps"]),
            )
            if training:
                losses["total"].backward()
                optimizer.step()
            batch_size = int(tokens.shape[0])
            samples += batch_size
            for name in totals:
                totals[name] += float(losses[name].detach()) * batch_size
    if samples == 0:
        raise RuntimeError("epoch loader produced no samples")
    return {name: value / samples for name, value in totals.items()}


def train_model(
    *,
    config: dict[str, Any],
    config_sha256: str,
    dataset_root: str | Path,
    output_dir: str | Path,
    model_kind: str,
    epochs: int,
    batch_size: int,
    capacity: str = "small",
    device: str = "cuda",
) -> dict[str, Any]:
    """Train one fixed model and retain only its best inference checkpoint."""
    if model_kind not in {"adapter", "low_rank_linear"}:
        raise ValueError(f"unsupported model kind: {model_kind}")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if model_kind == "low_rank_linear" and capacity != "small":
        raise ValueError("LowRank Linear does not use capacity profiles")

    root = Path(dataset_root).resolve()
    destination = Path(output_dir).resolve()
    best_path = destination / "best.pt"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite training output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    torch_device = resolve_device(device)
    training = config["training"]
    fingerprint = dataset_fingerprint(root)
    train_dataset = AdapterDataset(root / "train.jsonl", dataset_root=root)
    val_dataset = AdapterDataset(root / "val.jsonl", dataset_root=root)
    pin_memory = bool(training["pin_memory"]) and torch_device.type == "cuda"
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(training["num_workers"]),
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=pin_memory,
    )

    model_config = model_config_for_kind(config, model_kind, capacity)
    model = build_trainable_model(model_kind, model_config).to(torch_device)
    parameter_count = count_parameters(model)
    identity = model_identity_metadata(
        model_kind=model_kind,
        model_config=model_config,
        capacity=capacity if model_kind == "adapter" else None,
        parameter_count=parameter_count,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metadata: dict[str, Any] | None = None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model=model, loader=train_loader, device=torch_device,
            loss_config=config["loss"], optimizer=optimizer,
        )
        val_metrics = _run_epoch(
            model=model, loader=val_loader, device=torch_device,
            loss_config=config["loss"], optimizer=None,
        )
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "elapsed_seconds": elapsed,
            "train_total_loss": train_metrics["total"],
            "train_cosine_loss": train_metrics["cosine"],
            "train_l2_loss": train_metrics["l2"],
            "val_total_loss": val_metrics["total"],
            "val_cosine_loss": val_metrics["cosine"],
            "val_cosine_similarity": 1.0 - val_metrics["cosine"],
            "val_l2_loss": val_metrics["l2"],
        }
        history.append(row)
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = epoch
            best_metadata = {
                "experiment": config["experiment"]["name"],
                "config_sha256": config_sha256,
                "dataset_fingerprint": fingerprint,
                **identity,
                "epoch": epoch,
                "requested_epochs": epochs,
                "best_metric_name": training["best_metric"],
                "best_val_metric": best_val,
                "metrics": row,
            }
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(json.dumps(row, sort_keys=True))

    total_time = time.perf_counter() - started
    if best_state is None or best_metadata is None:
        raise RuntimeError("training completed without a best checkpoint")
    best_metadata.update({
        "requested_epochs": epochs,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_metric": best_val,
        "total_training_time": total_time,
    })
    atomic_torch_save(best_path, {
        "checkpoint_schema_version": 2,
        "model_state_dict": best_state,
        "metadata": best_metadata,
    })
    summary = {
        "status": "complete",
        "model_kind": model_kind,
        "capacity": capacity if model_kind == "adapter" else None,
        "hidden_dim": model_config.get("hidden_dim"),
        "rank": model_config.get("rank"),
        "parameter_count": parameter_count,
        "model_config": model_config,
        "model_config_sha256": identity["model_config_sha256"],
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_metric": best_val,
        "best_metric_name": training["best_metric"],
        "total_training_time": total_time,
        "dataset_fingerprint": fingerprint,
        "checkpoint": str(best_path),
    }
    atomic_write_json(destination / "history.json", {"summary": summary, "history": history})
    del model, optimizer, best_state
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return summary
