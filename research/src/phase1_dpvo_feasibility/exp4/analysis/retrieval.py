"""Single-modality temporal retrieval for final Exp4 analysis."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from ..dataset import AdapterDataset, dataset_fingerprint
from ..schema import atomic_write_json


def pool_global_embeddings(
    jepa_tokens: torch.Tensor,
    fmap_teacher: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool `[B,N,D]` JEPA and `[B,C,H,W]` FMap without cross-space mapping."""
    if jepa_tokens.ndim != 3:
        raise ValueError("JEPA tokens must have shape [B,N,D]")
    if fmap_teacher.ndim != 4:
        raise ValueError("FMap teacher must have shape [B,C,H,W]")
    if jepa_tokens.shape[0] != fmap_teacher.shape[0]:
        raise ValueError("JEPA and FMap batch dimensions must match")
    jepa = jepa_tokens.float().mean(dim=1)
    fmap = fmap_teacher.float().mean(dim=(2, 3))
    if not torch.isfinite(jepa).all().item() or not torch.isfinite(fmap).all().item():
        raise ValueError("global embeddings contain NaN or Inf")
    return jepa, fmap


def _sequence_groups(sequence_ids: Sequence[tuple[str, str, str]]) -> list[list[int]]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, identity in enumerate(sequence_ids):
        groups[tuple(identity)].append(index)
    return list(groups.values())


def temporal_retrieval_metrics(
    embeddings: torch.Tensor,
    sequence_ids: Sequence[tuple[str, str, str]],
    *,
    query_chunk_size: int = 128,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute excluding-self same-sequence temporal top-k statistics."""
    if embeddings.ndim != 2 or embeddings.shape[0] != len(sequence_ids):
        raise ValueError("embeddings must be [samples,dim] and align with sequence_ids")
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("embeddings must be non-empty")
    if query_chunk_size <= 0 or top_k <= 0:
        raise ValueError("query_chunk_size and top_k must be positive")
    normalized = functional.normalize(embeddings.float(), dim=1, eps=1e-8)
    self_similarity = (normalized * normalized).sum(dim=1)
    counts = {
        "top1_within_1": 0,
        "top1_within_5": 0,
        "top5_within_1": 0,
        "top5_within_5": 0,
    }
    eligible = 0
    for group in _sequence_groups(sequence_ids):
        size = len(group)
        if size < 2:
            continue
        vectors = normalized[group]
        ordinals = torch.arange(size, device=vectors.device)
        actual_k = min(int(top_k), size - 1)
        for start in range(0, size, int(query_chunk_size)):
            stop = min(start + int(query_chunk_size), size)
            similarity = vectors[start:stop] @ vectors.transpose(0, 1)
            local_queries = torch.arange(start, stop, device=vectors.device)
            similarity[torch.arange(stop - start, device=vectors.device), local_queries] = -torch.inf
            retrieved = similarity.topk(actual_k, dim=1).indices
            distances = (retrieved - local_queries[:, None]).abs()
            counts["top1_within_1"] += int((distances[:, 0] <= 1).sum().item())
            counts["top1_within_5"] += int((distances[:, 0] <= 5).sum().item())
            counts["top5_within_1"] += int((distances <= 1).any(dim=1).sum().item())
            counts["top5_within_5"] += int((distances <= 5).any(dim=1).sum().item())
            eligible += stop - start
    if eligible == 0:
        raise ValueError("temporal retrieval requires a sequence with at least two samples")
    return {
        "sample_count": int(embeddings.shape[0]),
        "eligible_temporal_queries": eligible,
        "self_similarity_mean": float(self_similarity.mean().item()),
        "excluding_self_top1_within_1": counts["top1_within_1"] / eligible,
        "excluding_self_top1_within_5": counts["top1_within_5"] / eligible,
        "excluding_self_top5_within_1": counts["top5_within_1"] / eligible,
        "excluding_self_top5_within_5": counts["top5_within_5"] / eligible,
    }


def random_temporal_baseline(
    sequence_ids: Sequence[tuple[str, str, str]],
    *,
    seed: int,
    top_k: int = 5,
) -> dict[str, Any]:
    """Fixed-seed random ranking baseline for excluding-self temporal metrics."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    counts = {name: 0 for name in ("top1_within_1", "top1_within_5", "top5_within_1", "top5_within_5")}
    eligible = 0
    for group in _sequence_groups(sequence_ids):
        size = len(group)
        if size < 2:
            continue
        actual_k = min(int(top_k), size - 1)
        for query in range(size):
            candidates = torch.cat((torch.arange(query), torch.arange(query + 1, size)))
            ranked = candidates[torch.randperm(len(candidates), generator=generator)[:actual_k]]
            distances = (ranked - query).abs()
            counts["top1_within_1"] += int(distances[0] <= 1)
            counts["top1_within_5"] += int(distances[0] <= 5)
            counts["top5_within_1"] += int((distances <= 1).any())
            counts["top5_within_5"] += int((distances <= 5).any())
            eligible += 1
    if eligible == 0:
        raise ValueError("random temporal baseline requires at least two samples in a sequence")
    return {
        "seed": int(seed),
        "eligible_temporal_queries": eligible,
        "excluding_self_top1_within_1": counts["top1_within_1"] / eligible,
        "excluding_self_top1_within_5": counts["top1_within_5"] / eligible,
        "excluding_self_top5_within_1": counts["top5_within_1"] / eligible,
        "excluding_self_top5_within_5": counts["top5_within_5"] / eligible,
    }


def analyze_temporal_retrieval(
    *, config: dict[str, Any], config_sha256: str,
    dataset_root: str | Path, output_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Analyze JEPA and FMap temporal structure without cross-modal fitting."""
    root = Path(dataset_root).resolve()
    fingerprint = dataset_fingerprint(root)
    dataset = AdapterDataset(root / "test.jsonl", dataset_root=root)
    settings = config["retrieval_analysis"]
    batch_size = int(settings["batch_size"])
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    jepa_batches: list[torch.Tensor] = []
    fmap_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            jepa, fmap = pool_global_embeddings(
                batch["jepa_tokens"].to(torch_device), batch["fmap_teacher"].to(torch_device),
            )
            jepa_batches.append(jepa.cpu())
            fmap_batches.append(fmap.cpu())
    jepa_embeddings = torch.cat(jepa_batches)
    fmap_embeddings = torch.cat(fmap_batches)
    samples = dataset.samples
    sequence_ids = [(sample.dataset_type, sample.dataset_group, sample.sequence) for sample in samples]
    chunk = int(settings["query_chunk_size"])
    top_k = int(settings["top_k"])
    seed = int(settings["seed"])
    result = {
        "status": "complete",
        "split": "test",
        "dataset_fingerprint": fingerprint,
        "config_sha256": config_sha256,
        "definitions": {
            "cross_modal_similarity": False,
            "temporal_window_unit": "selected-frame ordinal within sequence",
            "self_excluded_from_temporal": True,
        },
        "modalities": {
            "jepa": temporal_retrieval_metrics(jepa_embeddings, sequence_ids, query_chunk_size=chunk, top_k=top_k),
            "fmap": temporal_retrieval_metrics(fmap_embeddings, sequence_ids, query_chunk_size=chunk, top_k=top_k),
        },
        "random_baseline": random_temporal_baseline(sequence_ids, seed=seed, top_k=top_k),
    }
    atomic_write_json(output_path, result)
    return result
