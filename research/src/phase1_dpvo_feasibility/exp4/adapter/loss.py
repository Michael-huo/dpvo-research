"""Feature-space losses for Experiment 4-1."""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as functional


class LossTerms(TypedDict):
    total: torch.Tensor
    cosine: torch.Tensor
    l2: torch.Tensor


def feature_adapter_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    *,
    cosine_weight: float = 1.0,
    l2_weight: float = 0.1,
    eps: float = 1e-8,
) -> LossTerms:
    if prediction.shape != teacher.shape or prediction.ndim != 4:
        raise ValueError(
            f"prediction and teacher must share [B,C,H,W], got {tuple(prediction.shape)} and {tuple(teacher.shape)}"
        )
    if not prediction.is_floating_point() or not teacher.is_floating_point():
        raise TypeError("prediction and teacher must be floating-point tensors")
    pred_vectors = prediction.permute(0, 2, 3, 1).reshape(-1, prediction.shape[1])
    teacher_vectors = teacher.permute(0, 2, 3, 1).reshape(-1, teacher.shape[1])
    cosine_similarity = functional.cosine_similarity(pred_vectors, teacher_vectors, dim=-1, eps=float(eps))
    cosine_loss = 1.0 - cosine_similarity.mean()
    l2_loss = functional.mse_loss(prediction, teacher)
    total = float(cosine_weight) * cosine_loss + float(l2_weight) * l2_loss
    return {"total": total, "cosine": cosine_loss, "l2": l2_loss}
