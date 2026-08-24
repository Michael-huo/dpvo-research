"""Deterministic final reports for Exp4 Bridge and Capacity experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .schema import atomic_write_bytes, atomic_write_json


def bridge_interpretation(
    baseline_metrics: Mapping[str, Any],
    adapter_metrics: Mapping[str, Any],
    *,
    eps: float = 1e-8,
) -> dict[str, Any]:
    baselines = baseline_metrics["baselines"]
    cosine = {
        "random": float(baselines["random"]["mean_cosine_similarity"]),
        "mean": float(baselines["mean_fmap"]["mean_cosine_similarity"]),
        "lowrank": float(baselines["low_rank_linear"]["mean_cosine_similarity"]),
        "adapter": float(adapter_metrics["mean_cosine_similarity"]),
    }
    absolute_gap = abs(cosine["adapter"] - cosine["lowrank"])
    relative_gap = absolute_gap / max(abs(cosine["adapter"]), float(eps))
    if cosine["lowrank"] <= cosine["mean"]:
        mapping = "direct_linear_relation_is_weak"
        explanation = "LowRank does not exceed the Mean FMap prior; the direct global linear relation is weak."
    elif cosine["lowrank"] < cosine["adapter"]:
        mapping = "linear_signal_with_nonlinear_gain"
        explanation = "LowRank exceeds Mean, while the nonlinear adapter provides additional representation adaptation."
    else:
        mapping = "global_linear_mapping_is_competitive"
        explanation = "LowRank matches or exceeds the adapter, so a global linear mapping is competitive."
    bridge_exists = cosine["adapter"] > max(cosine["random"], cosine["mean"])
    return {
        "primary_metric": "mean_cosine_similarity",
        "cosine_ladder": cosine,
        "lowrank_adapter_absolute_gap": absolute_gap,
        "lowrank_adapter_relative_gap": relative_gap,
        "mapping_interpretation": mapping,
        "mapping_explanation": explanation,
        "bridge_evidence": bridge_exists,
        "bridge_conclusion": (
            "The adapter exceeds non-visual and dataset-prior baselines, providing evidence for a learnable bridge."
            if bridge_exists else
            "The adapter does not exceed both non-visual and dataset-prior baselines; a learnable bridge is not established."
        ),
    }


def capacity_decision(
    capacity_metrics: Mapping[str, Mapping[str, Any]],
    *,
    threshold: float,
    eps: float,
) -> dict[str, Any]:
    if set(capacity_metrics) != {"small", "medium", "large"}:
        raise ValueError("capacity metrics must contain exactly small, medium, and large")
    small = float(capacity_metrics["small"]["best_validation_metric"])
    large = float(capacity_metrics["large"]["best_validation_metric"])
    improvement = (small - large) / max(abs(small), float(eps))
    capacity_effective = improvement >= float(threshold)
    return {
        "primary_metric": "best_validation_total_loss",
        "small_best_validation_metric": small,
        "large_best_validation_metric": large,
        "large_vs_small_relative_improvement": improvement,
        "relative_improvement_threshold": float(threshold),
        "capacity_effective": capacity_effective,
        "conclusion": (
            "Increasing adapter capacity may be effective; the result provides evidence of a capacity bottleneck."
            if capacity_effective else
            "Capacity scaling has no clear benefit; the remaining limitation is more consistent with representation mismatch."
        ),
    }


def write_bridge_report(
    output_dir: str | Path,
    *,
    protocol: Mapping[str, Any],
    adapter_training: Mapping[str, Any],
    lowrank_training: Mapping[str, Any],
    adapter_metrics: Mapping[str, Any],
    lowrank_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    retrieval_metrics: Mapping[str, Any],
    extraction: Mapping[str, Any],
    config_sha256: str,
    eps: float,
) -> dict[str, Any]:
    root = Path(output_dir)
    interpretation = bridge_interpretation(baseline_metrics, adapter_metrics, eps=eps)
    aggregate = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "JEPA-DPVO Representation Bridge Validation",
        "config_sha256": config_sha256,
        "protocol": dict(protocol),
        "extraction": dict(extraction),
        "training": {
            "adapter": dict(adapter_training),
            "lowrank": dict(lowrank_training),
        },
        "adapter": dict(adapter_metrics),
        "lowrank": dict(lowrank_metrics),
        "baseline": dict(baseline_metrics),
        "retrieval": dict(retrieval_metrics),
        "interpretation": interpretation,
    }
    atomic_write_json(root / "metrics.json", aggregate)
    ladder = interpretation["cosine_ladder"]
    jepa = retrieval_metrics["modalities"]["jepa"]
    fmap = retrieval_metrics["modalities"]["fmap"]
    report = f"""# Experiment 4: JEPA-DPVO Representation Bridge Validation

## 1. Experiment objective

Determine whether frozen V-JEPA representations contain information that can be mapped to the DPVO FMap feature space, and whether that mapping is adequately explained by a global low-rank linear model or requires nonlinear adaptation.

## 2. Dataset and protocol

- Train: `{', '.join(protocol['splits']['train'])}`
- Validation: `{', '.join(protocol['splits']['val'])}`
- Test: `{', '.join(protocol['splits']['test'])}`
- Camera/stride: `{protocol['camera']}`, stride `{protocol['stride']}`
- Adapter and LowRank training: `{adapter_training['epochs']}` epochs

## 3. Adapter result

- Mean cosine similarity: `{adapter_metrics['mean_cosine_similarity']:.6f}`
- Median cosine similarity: `{adapter_metrics['median_cosine_similarity']:.6f}`
- MSE: `{adapter_metrics['mse']:.6f}`
- Norm ratio: `{adapter_metrics['norm_ratio']:.6f}`

## 4. Baseline comparison

| Method | Mean cosine | MSE | Norm ratio |
|---|---:|---:|---:|
| Random | {ladder['random']:.6f} | {baseline_metrics['baselines']['random']['mse']:.6f} | {baseline_metrics['baselines']['random']['norm_ratio']:.6f} |
| Mean FMap | {ladder['mean']:.6f} | {baseline_metrics['baselines']['mean_fmap']['mse']:.6f} | {baseline_metrics['baselines']['mean_fmap']['norm_ratio']:.6f} |
| LowRank Linear | {ladder['lowrank']:.6f} | {lowrank_metrics['mse']:.6f} | {lowrank_metrics['norm_ratio']:.6f} |
| Adapter | {ladder['adapter']:.6f} | {adapter_metrics['mse']:.6f} | {adapter_metrics['norm_ratio']:.6f} |

LowRank–Adapter absolute cosine gap: `{interpretation['lowrank_adapter_absolute_gap']:.6f}`. Relative gap: `{interpretation['lowrank_adapter_relative_gap']:.6f}`.

{interpretation['mapping_explanation']}

## 5. Temporal retrieval

| Modality | Top-1 ±1 | Top-1 ±5 | Top-5 ±1 | Top-5 ±5 |
|---|---:|---:|---:|---:|
| JEPA | {jepa['excluding_self_top1_within_1']:.6f} | {jepa['excluding_self_top1_within_5']:.6f} | {jepa['excluding_self_top5_within_1']:.6f} | {jepa['excluding_self_top5_within_5']:.6f} |
| FMap | {fmap['excluding_self_top1_within_1']:.6f} | {fmap['excluding_self_top1_within_5']:.6f} | {fmap['excluding_self_top5_within_1']:.6f} | {fmap['excluding_self_top5_within_5']:.6f} |

Retrieval is single-modality and excludes self matches. It does not fit or compare JEPA and FMap vectors across incompatible channel dimensions.

## 6. Conclusion

{interpretation['bridge_conclusion']}

{interpretation['mapping_explanation']}
"""
    atomic_write_bytes(root / "REPORT.md", report.encode("utf-8"))
    return aggregate


def write_capacity_report(
    output_dir: str | Path,
    *,
    capacity_metrics: Mapping[str, Mapping[str, Any]],
    threshold: float,
    eps: float,
    config_sha256: str,
    epochs: int,
) -> dict[str, Any]:
    root = Path(output_dir)
    decision = capacity_decision(capacity_metrics, threshold=threshold, eps=eps)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "Exp4 Adapter Capacity Scaling",
        "config_sha256": config_sha256,
        "epochs": int(epochs),
        "capacities": {name: dict(metrics) for name, metrics in capacity_metrics.items()},
        "decision": decision,
    }
    atomic_write_json(root / "summary.json", summary)
    rows = []
    for name in ("small", "medium", "large"):
        item = capacity_metrics[name]
        rows.append(
            f"| {name} | {item['hidden_dim']} | {item['parameter_count']} | "
            f"{item['best_epoch']} | {item['best_validation_metric']:.6f} | "
            f"{item['total_training_time']:.2f} | `{item['model_config_sha256']}` |"
        )
    report = f"""# Experiment 4 Adapter Capacity Scaling

## Objective

Determine whether the JEPA-to-DPVO mapping is limited by adapter capacity while holding the training protocol fixed at {int(epochs)} epochs.

## Results

| Capacity | Hidden dim | Parameters | Best epoch | Best validation total loss | Training time (s) | Model config hash |
|---|---:|---:|---:|---:|---:|---|
""" + "\n".join(rows) + f"""

Large-vs-small relative improvement: `{decision['large_vs_small_relative_improvement']:.6f}`. Decision threshold: `{decision['relative_improvement_threshold']:.2%}`.

## Conclusion

{decision['conclusion']}
"""
    atomic_write_bytes(root / "REPORT.md", report.encode("utf-8"))
    return summary
