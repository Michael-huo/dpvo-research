"""Random, Mean FMap, and LowRank comparison for final Exp4."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from ..adapter.model import MeanFMapBaseline, RandomPredictionBaseline, build_trainable_model
from ..adapter.train import load_checkpoint, resolve_device, seed_everything
from ..dataset import AdapterDataset, dataset_fingerprint
from ..schema import atomic_write_json


BASELINE_METRIC_KEYS = frozenset({
    "mean_cosine_similarity",
    "median_cosine_similarity",
    "mse",
    "pred_norm_mean",
    "teacher_norm_mean",
    "norm_ratio",
    "sample_count",
})


class BaselineMetricsAccumulator:
    """Streaming scalar metrics with only the cosine distribution retained."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self.cosines: list[torch.Tensor] = []
        self.squared_error_sum = 0.0
        self.element_count = 0
        self.pred_norm_sum = 0.0
        self.teacher_norm_sum = 0.0
        self.vector_count = 0
        self.sample_count = 0

    def update(self, prediction: torch.Tensor, teacher: torch.Tensor) -> None:
        if prediction.shape != teacher.shape or prediction.ndim != 4:
            raise ValueError("baseline tensors must share [B,C,H,W]")
        pred, target = prediction.detach().float(), teacher.detach().float()
        self.cosines.append(functional.cosine_similarity(pred, target, dim=1, eps=self.eps).reshape(-1).cpu())
        difference = pred - target
        self.squared_error_sum += float(difference.square().sum().item())
        self.element_count += difference.numel()
        pred_norm = torch.linalg.vector_norm(pred, dim=1)
        teacher_norm = torch.linalg.vector_norm(target, dim=1)
        self.pred_norm_sum += float(pred_norm.sum().item())
        self.teacher_norm_sum += float(teacher_norm.sum().item())
        self.vector_count += pred_norm.numel()
        self.sample_count += int(pred.shape[0])

    def finalize(self) -> dict[str, Any]:
        if not self.cosines or not self.element_count or not self.vector_count:
            raise RuntimeError("no baseline observations accumulated")
        cosine = torch.cat(self.cosines)
        pred_norm_mean = self.pred_norm_sum / self.vector_count
        teacher_norm_mean = self.teacher_norm_sum / self.vector_count
        return {
            "mean_cosine_similarity": float(cosine.mean().item()),
            "median_cosine_similarity": float(cosine.median().item()),
            "mse": self.squared_error_sum / self.element_count,
            "pred_norm_mean": pred_norm_mean,
            "teacher_norm_mean": teacher_norm_mean,
            "norm_ratio": pred_norm_mean / max(teacher_norm_mean, self.eps),
            "sample_count": self.sample_count,
        }


def streaming_mean_fmap(loader: DataLoader) -> tuple[torch.Tensor, int]:
    """Compute the train teacher mean without retaining dataset samples."""
    total: torch.Tensor | None = None
    sample_count = 0
    for batch in loader:
        teacher = batch["fmap_teacher"].double()
        batch_sum = teacher.sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        sample_count += int(teacher.shape[0])
    if total is None or sample_count == 0:
        raise RuntimeError("mean baseline train loader is empty")
    mean = (total / sample_count).float()
    if not torch.isfinite(mean).all().item():
        raise RuntimeError("non-finite train mean FMap")
    return mean, sample_count


def evaluate_models(
    models: Mapping[str, torch.nn.Module],
    loader: DataLoader,
    *,
    device: torch.device,
    eps: float,
) -> dict[str, dict[str, Any]]:
    """Evaluate several baselines in one dataset pass."""
    if not models:
        raise ValueError("at least one baseline model is required")
    prepared = {name: model.to(device).eval() for name, model in models.items()}
    accumulators = {name: BaselineMetricsAccumulator(eps) for name in prepared}
    with torch.no_grad():
        for batch in loader:
            tokens = batch["jepa_tokens"].to(device=device, non_blocking=True)
            teacher = batch["fmap_teacher"].to(device=device, non_blocking=True)
            for name, model in prepared.items():
                prediction = model(tokens)
                accumulators[name].update(prediction, teacher)
    return {
        name: {"status": "complete", **accumulator.finalize()}
        for name, accumulator in accumulators.items()
    }


def load_low_rank_baseline(
    checkpoint_path: str | Path | None,
    *,
    expected_dataset_sha256: str,
    configured_model: dict[str, Any],
) -> tuple[torch.nn.Module | None, dict[str, Any]]:
    """Restore a trained LowRank baseline or return an explicit missing state."""
    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        return None, {"status": "not_evaluated", "reason": "missing_checkpoint"}
    checkpoint = load_checkpoint(
        checkpoint_path, map_location="cpu",
        expected_dataset_sha256=expected_dataset_sha256,
    )
    metadata = checkpoint["metadata"]
    if metadata.get("model_kind") != "low_rank_linear":
        raise ValueError("LowRank checkpoint model_kind must be low_rank_linear")
    model_config = dict(metadata.get("model_config", {}))
    expected_rank = int(configured_model["rank"])
    if int(model_config.get("rank", -1)) != expected_rank:
        raise ValueError(f"LowRank checkpoint rank differs from config: {model_config.get('rank')} != {expected_rank}")
    for name in ("input_shape", "output_shape"):
        if tuple(model_config.get(name, ())) != tuple(configured_model[name]):
            raise ValueError(f"LowRank checkpoint {name} differs from config")
    model = build_trainable_model("low_rank_linear", model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, {
        "status": "ready",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "rank": expected_rank,
        "checkpoint_epoch": metadata.get("epoch"),
    }


__all__ = [
    "BASELINE_METRIC_KEYS",
    "BaselineMetricsAccumulator",
    "MeanFMapBaseline",
    "RandomPredictionBaseline",
    "evaluate_models",
    "load_low_rank_baseline",
    "streaming_mean_fmap",
]


def evaluate_baseline_ladder(
    *,
    config: dict[str, Any],
    config_sha256: str,
    dataset_root: str | Path,
    lowrank_checkpoint: str | Path,
    output_path: str | Path,
    batch_size: int | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate the three baseline rungs without training hidden work."""
    root = Path(dataset_root).resolve()
    fingerprint = dataset_fingerprint(root)
    seed = int(config["evaluation"]["seed"])
    seed_everything(seed)
    torch_device = resolve_device(device)
    batch_size = int(batch_size if batch_size is not None else config["training"]["batch_size"])
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": int(config["training"]["num_workers"]),
    }
    train_dataset = AdapterDataset(root / "train.jsonl", dataset_root=root)
    test_dataset = AdapterDataset(root / "test.jsonl", dataset_root=root)
    mean_fmap, fitted_samples = streaming_mean_fmap(DataLoader(train_dataset, **loader_kwargs))
    lowrank, state = load_low_rank_baseline(
        lowrank_checkpoint,
        expected_dataset_sha256=fingerprint["sha256"],
        configured_model=config["low_rank_linear"],
    )
    if lowrank is None:
        raise FileNotFoundError(lowrank_checkpoint)
    metrics = evaluate_models(
        {
            "random": RandomPredictionBaseline(config["adapter"]["output_shape"]),
            "mean_fmap": MeanFMapBaseline(mean_fmap),
            "low_rank_linear": lowrank,
        },
        DataLoader(test_dataset, pin_memory=torch_device.type == "cuda", **loader_kwargs),
        device=torch_device,
        eps=float(config["evaluation"]["eps"]),
    )
    metrics["random"].update({"role": "metric_floor", "seed": seed})
    metrics["mean_fmap"].update({
        "role": "dataset_prior", "fit_split": "train", "fit_samples": fitted_samples,
    })
    metrics["low_rank_linear"].update({
        key: value for key, value in state.items() if key != "status"
    })
    metrics["low_rank_linear"]["role"] = "global_linear_mapping"
    result = {
        "status": "complete",
        "split": "test",
        "dataset_fingerprint": fingerprint,
        "config_sha256": config_sha256,
        "baselines": metrics,
    }
    atomic_write_json(output_path, result)
    del lowrank
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return result
