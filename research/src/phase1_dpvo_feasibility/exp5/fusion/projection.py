"""Minimal dense JEPA projection and RAM-only training for Exp5-2."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


PAIR_KEYS = frozenset(("tokens", "teacher"))


class DenseJepaProjection(nn.Module):
    """The complete Exp5-2 trainable surface: one 1x1 channel projection."""

    def __init__(
        self,
        *,
        input_dim: int = 768,
        output_dim: int = 128,
        token_grid: tuple[int, int] = (24, 24),
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.token_grid = tuple(int(value) for value in token_grid)
        if len(self.token_grid) != 2 or min(self.token_grid) <= 0:
            raise ValueError("token_grid must contain two positive dimensions")
        self.token_count = self.token_grid[0] * self.token_grid[1]
        self.channel_projection = nn.Conv2d(
            self.input_dim, self.output_dim, kernel_size=1, bias=True
        )

    def token_grid_tensor(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
            self.token_count,
            self.input_dim,
        ):
            raise ValueError(
                f"expected dense tokens [B,{self.token_count},{self.input_dim}], "
                f"got {tuple(tokens.shape)}"
            )
        batch = int(tokens.shape[0])
        height, width = self.token_grid
        return (
            tokens.float()
            .reshape(batch, height, width, self.input_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def forward(self, tokens: torch.Tensor, *, target_size: tuple[int, int]) -> torch.Tensor:
        height, width = (int(value) for value in target_size)
        if height <= 0 or width <= 0:
            raise ValueError("target_size must be positive")
        projected = self.channel_projection(self.token_grid_tensor(tokens))
        output = functional.interpolate(
            projected,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        expected = (int(tokens.shape[0]), self.output_dim, height, width)
        if tuple(output.shape) != expected:
            raise RuntimeError(f"dense projection shape drift: {tuple(output.shape)} != {expected}")
        return output


def dense_spatial_cosine_loss(
    prediction: torch.Tensor, teacher: torch.Tensor, *, eps: float = 1.0e-8,
) -> torch.Tensor:
    if prediction.shape != teacher.shape or prediction.ndim != 4:
        raise ValueError(
            f"dense prediction/teacher require equal [B,C,H,W], got "
            f"{tuple(prediction.shape)} and {tuple(teacher.shape)}"
        )
    if not bool(torch.isfinite(prediction).all().item()) or not bool(
        torch.isfinite(teacher).all().item()
    ):
        raise ValueError("dense alignment inputs must be finite")
    pred_vectors = prediction.permute(0, 2, 3, 1).reshape(-1, prediction.shape[1])
    teacher_vectors = teacher.float().permute(0, 2, 3, 1).reshape(-1, teacher.shape[1])
    return 1.0 - functional.cosine_similarity(
        pred_vectors, teacher_vectors, dim=1, eps=float(eps)
    ).mean()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def estimate_pair_bytes(
    frame_count: int,
    *,
    token_shape: tuple[int, int] = (576, 768),
    teacher_shape: tuple[int, int, int] = (128, 120, 188),
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> int:
    if int(frame_count) <= 0 or min(*token_shape, *teacher_shape) <= 0:
        raise ValueError("pair dimensions and frame_count must be positive")
    elements = int(np.prod(token_shape)) + int(np.prod(teacher_shape))
    return int(frame_count) * elements * int(np.dtype(dtype).itemsize)


def validate_memory_capacity(
    *, required_bytes: int, available_bytes: int, safety_factor: float,
) -> dict[str, Any]:
    if required_bytes <= 0 or available_bytes <= 0 or safety_factor < 1.0:
        raise ValueError("invalid memory-capacity inputs")
    guarded = int(np.ceil(required_bytes * float(safety_factor)))
    if available_bytes < guarded:
        raise MemoryError(
            f"Exp5-2 RAM gate failed: available={available_bytes} bytes, "
            f"guarded_required={guarded} bytes"
        )
    return {
        "pair_bytes": int(required_bytes),
        "safety_factor": float(safety_factor),
        "guarded_required_bytes": guarded,
        "available_bytes": int(available_bytes),
        "passed": True,
    }


def _validate_pairs(
    pairs: dict[str, np.ndarray],
    *,
    input_dim: int,
    output_dim: int,
    token_count: int,
    check_finite: bool = True,
) -> tuple[int, tuple[int, int]]:
    if set(pairs) != PAIR_KEYS:
        raise ValueError("dense pairs require exactly tokens and teacher arrays")
    tokens, teacher = pairs["tokens"], pairs["teacher"]
    if tokens.dtype != np.float32 or teacher.dtype != np.float32:
        raise ValueError("dense RAM pairs must use float32")
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (token_count, input_dim):
        raise ValueError(f"invalid dense token pair shape: {tokens.shape}")
    if teacher.ndim != 4 or teacher.shape[0] != tokens.shape[0] or teacher.shape[1] != output_dim:
        raise ValueError(f"invalid dense teacher pair shape: {teacher.shape}")
    if len(tokens) == 0:
        raise ValueError("dense RAM pairs must be non-empty")
    if check_finite and (
        not np.isfinite(tokens).all() or not np.isfinite(teacher).all()
    ):
        raise ValueError("dense RAM pairs must be finite")
    return len(tokens), (int(teacher.shape[2]), int(teacher.shape[3]))


@torch.no_grad()
def evaluate_dense_alignment(
    model: DenseJepaProjection,
    pairs: dict[str, np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    count, target_size = _validate_pairs(
        pairs,
        input_dim=model.input_dim,
        output_dim=model.output_dim,
        token_count=model.token_count,
        # Extraction and the training entry point validate each 40-GiB-scale
        # array once. Re-scanning it before every fifth-epoch validation would
        # add substantial memory bandwidth without improving the gate.
        check_finite=False,
    )
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    cosine_sum = 0.0
    vector_count = 0
    for start in range(0, count, int(batch_size)):
        stop = min(start + int(batch_size), count)
        tokens = torch.from_numpy(pairs["tokens"][start:stop]).to(device)
        teacher = torch.from_numpy(pairs["teacher"][start:stop]).to(device)
        prediction = model(tokens, target_size=target_size)
        values = functional.cosine_similarity(
            prediction.permute(0, 2, 3, 1).reshape(-1, model.output_dim),
            teacher.float().permute(0, 2, 3, 1).reshape(-1, model.output_dim),
            dim=1,
        )
        cosine_sum += float(values.sum().item())
        vector_count += int(values.numel())
        del tokens, teacher, prediction, values
    if vector_count == 0:
        raise RuntimeError("dense alignment evaluation produced no spatial vectors")
    mean = cosine_sum / vector_count
    return {
        "sample_count": int(count),
        "spatial_vector_count": int(vector_count),
        "mean_spatial_cosine_similarity": mean,
        "spatial_cosine_loss": 1.0 - mean,
    }


def build_random_dense_projection(
    config: dict[str, Any], *, device: torch.device,
) -> DenseJepaProjection:
    projection = config["projection"]
    seed_everything(int(projection["random_seed"]))
    return DenseJepaProjection(
        input_dim=int(projection["input_dim"]),
        output_dim=int(projection["output_dim"]),
        token_grid=tuple(int(value) for value in projection["token_grid"]),
    ).to(device).eval()


def train_dense_projection(
    *,
    train_pairs: dict[str, np.ndarray],
    val_pairs: dict[str, np.ndarray],
    config: dict[str, Any],
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[DenseJepaProjection, dict[str, Any]]:
    projection = config["projection"]
    input_dim = int(projection["input_dim"])
    output_dim = int(projection["output_dim"])
    token_count = int(projection["token_count"])
    train_count, target_size = _validate_pairs(
        train_pairs, input_dim=input_dim, output_dim=output_dim, token_count=token_count
    )
    val_count, val_target = _validate_pairs(
        val_pairs, input_dim=input_dim, output_dim=output_dim, token_count=token_count
    )
    if val_target != target_size:
        raise ValueError("train/validation FNet spatial shapes differ")
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    model = DenseJepaProjection(
        input_dim=input_dim,
        output_dim=output_dim,
        token_grid=tuple(int(value) for value in projection["token_grid"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(projection["learning_rate"]),
        weight_decay=float(projection["weight_decay"]),
    )
    epochs = int(projection["epochs"])
    batch_size = int(projection["batch_size"])
    interval = int(projection["validation_interval"])
    if epochs <= 0 or batch_size <= 0 or interval <= 0 or epochs % interval:
        raise ValueError("epochs must be positive and divisible by validation_interval")
    generator = torch.Generator().manual_seed(seed)
    best_cosine = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    validation_epochs: list[int] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train_count, generator=generator).numpy()
        for start in range(0, train_count, batch_size):
            indices = order[start:start + batch_size]
            tokens = torch.from_numpy(train_pairs["tokens"][indices]).to(device)
            teacher = torch.from_numpy(train_pairs["teacher"][indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tokens, target_size=target_size)
            loss = dense_spatial_cosine_loss(prediction, teacher)
            loss.backward()
            optimizer.step()
            del tokens, teacher, prediction, loss
        if epoch % interval == 0:
            validation_epochs.append(epoch)
            validation = evaluate_dense_alignment(
                model, val_pairs, batch_size=batch_size, device=device
            )
            cosine = float(validation["mean_spatial_cosine_similarity"])
            if cosine > best_cosine:
                best_cosine = cosine
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
    if best_state is None:
        raise RuntimeError("dense projection training produced no validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    learned_validation = evaluate_dense_alignment(
        model, val_pairs, batch_size=batch_size, device=device
    )
    random_model = build_random_dense_projection(config, device=device)
    random_validation = evaluate_dense_alignment(
        random_model, val_pairs, batch_size=batch_size, device=device
    )
    metadata = {
        "checkpoint_schema_version": 2,
        "projection_kind": "dense_tokens",
        "model": "Conv2d(768,128,kernel_size=1)",
        "input_shape": [token_count, input_dim],
        "token_grid": list(model.token_grid),
        "output_channels": output_dim,
        "teacher_contract": "raw Patchifier.fnet output / 4.0",
        "interpolation": {"mode": "bilinear", "align_corners": False},
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_spatial_cosine": best_cosine,
        "validation_epochs": validation_epochs,
        "requested_epochs": epochs,
        "batch_size": batch_size,
        "training_seconds": time.perf_counter() - started,
        "split_counts": {"train": train_count, "val": val_count},
        "learned_validation_alignment": learned_validation,
        "random_validation_alignment": random_validation,
        "random_seed": int(projection["random_seed"]),
        "random_weight_hash": state_dict_sha256(random_model.state_dict()),
        "uses_groundtruth": False,
        "uses_trajectory": False,
        "uses_future_frames": False,
        "pair_storage": "CPU_RAM_float32_only",
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "checkpoint_schema_version": 2,
            "model_state_dict": best_state,
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, destination)
    del optimizer, random_model, best_state
    return model, metadata


def load_dense_projection_checkpoint(
    path: str | Path, *, device: torch.device,
) -> tuple[DenseJepaProjection, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema_version") != 2:
        raise ValueError("unsupported Exp5-2 dense projection checkpoint schema")
    metadata = dict(payload["metadata"])
    if metadata.get("projection_kind") != "dense_tokens":
        raise ValueError("checkpoint is not an Exp5-2 dense projection")
    input_shape = metadata["input_shape"]
    model = DenseJepaProjection(
        input_dim=int(input_shape[1]),
        output_dim=int(metadata["output_channels"]),
        token_grid=tuple(int(value) for value in metadata["token_grid"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), metadata
