"""Internal checkpoint evaluation for the final Exp4 bridge experiment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from ..dataset import AdapterDataset, dataset_fingerprint
from ..schema import atomic_write_json
from .model import build_trainable_model
from .train import load_checkpoint, resolve_device, seed_everything


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phase1-exp4-final")


class EvaluationAccumulator:
    """Accumulate spatial cosine distribution and streaming scalar moments."""

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
            raise ValueError("evaluation tensors must share [B,C,H,W]")
        pred, target = prediction.detach().float(), teacher.detach().float()
        self.cosines.append(
            functional.cosine_similarity(pred, target, dim=1, eps=self.eps).reshape(-1).cpu()
        )
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
            raise RuntimeError("no evaluation observations accumulated")
        cosine = torch.cat(self.cosines)
        pred_norm_mean = self.pred_norm_sum / self.vector_count
        teacher_norm_mean = self.teacher_norm_sum / self.vector_count
        return {
            "sample_count": self.sample_count,
            "spatial_vector_count": self.vector_count,
            "mean_cosine_similarity": float(cosine.mean().item()),
            "median_cosine_similarity": float(cosine.median().item()),
            "mse": self.squared_error_sum / self.element_count,
            "pred_norm_mean": pred_norm_mean,
            "teacher_norm_mean": teacher_norm_mean,
            "norm_ratio": pred_norm_mean / max(teacher_norm_mean, self.eps),
        }


def _joint_pca_rgb(prediction: torch.Tensor, teacher: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    pred = prediction.permute(1, 2, 0).reshape(-1, prediction.shape[0]).float().cpu().numpy()
    target = teacher.permute(1, 2, 0).reshape(-1, teacher.shape[0]).float().cpu().numpy()
    combined = np.concatenate((target, pred), axis=0)
    fit_count = min(8192, len(combined))
    fit = combined[np.linspace(0, len(combined) - 1, num=fit_count, dtype=np.int64)]
    mean = fit.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(fit - mean, full_matrices=False)
    projected = (combined - mean) @ vt[:3].T
    low, high = np.percentile(projected, [1.0, 99.0], axis=0)
    rgb = np.clip((projected - low) / (high - low + 1e-8), 0.0, 1.0)
    height, width = prediction.shape[1:]
    return (
        rgb[:len(target)].reshape(height, width, 3),
        rgb[len(target):].reshape(height, width, 3),
    )


def _write_figure(
    *, prediction: torch.Tensor, teacher: torch.Tensor,
    sample: dict[str, Any], figures_dir: Path, model_kind: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    teacher_rgb, prediction_rgb = _joint_pca_rgb(prediction, teacher)
    cosine_map = functional.cosine_similarity(
        prediction.float(), teacher.float(), dim=0, eps=1e-8
    ).cpu().numpy()
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].imshow(teacher_rgb)
    axes[0].set_title("Teacher FMap joint PCA")
    axes[1].imshow(prediction_rgb)
    axes[1].set_title(f"{model_kind} prediction")
    image = axes[2].imshow(cosine_map, cmap="viridis", vmin=-1.0, vmax=1.0)
    axes[2].set_title("Spatial cosine similarity")
    figure.colorbar(image, ax=axes[2], fraction=0.046)
    for axis in axes:
        axis.axis("off")
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_kind}__{sample['sequence']}__f{sample['frame_id']}__t{sample['timestamp_ns']}"
    figure.savefig(figures_dir / f"{stem}.png", dpi=140)
    plt.close(figure)


def evaluate_checkpoint(
    *,
    config: dict[str, Any],
    config_sha256: str,
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    figures_dir: str | Path | None = None,
    batch_size: int | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    fingerprint = dataset_fingerprint(root)
    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location="cpu",
        expected_dataset_sha256=fingerprint["sha256"],
    )
    metadata = checkpoint["metadata"]
    model_kind = str(metadata["model_kind"])
    model = build_trainable_model(model_kind, metadata["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()

    seed = int(config["evaluation"]["seed"])
    seed_everything(seed)
    dataset = AdapterDataset(root / "test.jsonl", dataset_root=root)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size if batch_size is not None else config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=bool(config["training"]["pin_memory"]) and torch_device.type == "cuda",
    )
    visualization_count = min(int(config["evaluation"]["visualization_samples"]), len(dataset))
    generator = torch.Generator().manual_seed(seed)
    visualization_indices = set(
        torch.randperm(len(dataset), generator=generator)[:visualization_count].tolist()
    )
    records: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = []
    accumulator = EvaluationAccumulator(eps=float(config["evaluation"]["eps"]))
    offset = 0
    with torch.no_grad():
        for batch in loader:
            tokens = batch["jepa_tokens"].to(torch_device, non_blocking=True)
            teacher = batch["fmap_teacher"].to(torch_device, non_blocking=True)
            prediction = model(tokens)
            accumulator.update(prediction, teacher)
            for local_index in range(tokens.shape[0]):
                global_index = offset + local_index
                if figures_dir is not None and global_index in visualization_indices:
                    records.append((
                        prediction[local_index].cpu(),
                        teacher[local_index].cpu(),
                        dataset.samples[global_index].to_dict(),
                    ))
            offset += int(tokens.shape[0])

    metrics = accumulator.finalize()
    metrics.update({
        "status": "complete",
        "mode": "checkpoint",
        "split": "test",
        "dataset_fingerprint": fingerprint,
        "config_sha256": config_sha256,
        "mode_metadata": {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_epoch": metadata["epoch"],
            "model_kind": model_kind,
            "model_parameter_count": metadata["model_parameter_count"],
            "model_config_sha256": metadata["model_config_sha256"],
        },
        "optional_retrieval": {"enabled": False, "decision_metric": False},
    })
    atomic_write_json(output_path, metrics)
    if figures_dir is not None:
        for prediction, teacher, sample in records:
            _write_figure(
                prediction=prediction,
                teacher=teacher,
                sample=sample,
                figures_dir=Path(figures_dir),
                model_kind=model_kind,
            )
    del model
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics
