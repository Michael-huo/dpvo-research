"""Stride sweep protocol with the frozen temporal split."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

from latent_vslam.canonical import load_yaml
from prediction.predictor import build_anchor_intervals, effective_records, split_anchor_intervals
from latent_vslam.protocol import (REPO_ROOT, canonical_sha256, post_bootstrap_ratio_roles,
                       ratio_schedule_payload, repo_path)
from latent_vslam.inference_runtime import load_config as load_inference_config

DEFAULT_CONFIG = REPO_ROOT / "configs/infer/mh01_stride_sweep.yaml"
OUTPUT_ROOT = REPO_ROOT / "research/results/inference/mh01_stride_sweep"
TRAINING_ROOT = REPO_ROOT / "checkpoints/predictor/stride_sweep_training"
DEFAULT_STRIDES = (3, 5, 10)
SPLITS = ("train", "validation", "test")


def load_protocol(path: str | Path = DEFAULT_CONFIG):
    protocol, path = load_yaml(path)
    if protocol["sequence"] != "MH_01_easy" or protocol["anchor_strides"] != list(DEFAULT_STRIDES):
        raise ValueError("configured sweep is MH_01_easy with strides 3/5/10")
    if protocol["fresh_predictor_per_stride"] is not True:
        raise ValueError("each stride requires a fresh predictor")
    if repo_path(protocol["output_root"]) != OUTPUT_ROOT:
        raise ValueError("stride sweep output must stay in the inference results root")
    config, _ = load_inference_config(repo_path(protocol["base_infer_config"]))
    return protocol, config, path


def stride_roles(identities: Sequence[Any], bootstrap_end: int, anchor_stride: int):
    if isinstance(anchor_stride, bool) or not isinstance(anchor_stride, int) or anchor_stride < 2:
        raise ValueError("anchor_stride must be an integer >= 2")
    # Keep the accumulator, bootstrap, first-anchor and floating-point semantics.
    return post_bootstrap_ratio_roles(
        identities, bootstrap_end_candidate_index=bootstrap_end,
        anchor_ratio=1.0 / anchor_stride,
    )


def split_fixed_intervals(intervals, fixed_split):
    regions = fixed_split["regions"]
    if fixed_split["endpoint_bounds"] != "inclusive" or set(regions) != set(SPLITS):
        raise ValueError("fixed split requires three inclusive time regions")
    for left, right in zip(SPLITS, SPLITS[1:]):
        if regions[left]["timestamp_end_ns"] >= regions[right]["timestamp_start_ns"]:
            raise ValueError("fixed time regions overlap")
    parts = {name: [] for name in SPLITS}
    dropped = []
    for interval in intervals:
        matches = [name for name, region in regions.items()
                   if region["timestamp_start_ns"] <= interval.anchor0.timestamp_ns
                   and interval.anchor1.timestamp_ns <= region["timestamp_end_ns"]]
        if len(matches) == 1:
            parts[matches[0]].append(interval)
        else:
            dropped.append(interval.interval_index)
    anchors = {name: {key for row in values for key in row.anchor_keys}
               for name, values in parts.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if anchors[left] & anchors[right]:
            raise RuntimeError("shared anchor identity across fixed temporal splits")
    if not all(parts.values()):
        raise ValueError("fixed temporal split has an empty population")
    summary = {
        "method": "fixed_canonical_timestamp_regions_drop_crossing_intervals_v1",
        "fixed_split": copy.deepcopy(fixed_split),
        "source_interval_count": len(intervals),
        "dropped_boundary_interval_indices": dropped,
        "parts": {name: {
            "interval_count": len(rows),
            "hidden_query_count": sum(len(row.hidden) for row in rows),
            "anchor_count": len(anchors[name]),
            "first_anchor_identity": rows[0].anchor0.key,
            "last_anchor_identity": rows[-1].anchor1.key,
            "timestamp_start_ns": rows[0].anchor0.timestamp_ns,
            "timestamp_end_ns": rows[-1].anchor1.timestamp_ns,
            "interval_identity_sha256": canonical_sha256([row.payload() for row in rows]),
        } for name, rows in parts.items()},
    }
    summary["split_sha256"] = canonical_sha256(summary)
    return {name: tuple(rows) for name, rows in parts.items()}, summary


def encoded_communication(records, roles):
    """Count original uploaded PNG bytes without decoding RGB."""
    full = sum(Path(row.rgb_path).stat().st_size for row in records)
    anchors = [row for row in records if roles[row.identity.key] == "anchor"]
    uploaded = sum(Path(row.rgb_path).stat().st_size for row in anchors)
    if not full:
        raise ValueError("empty encoded population")
    return {"num_effective_observations": len(records), "anchor_count": len(anchors),
            "hidden_count": len(records) - len(anchors),
            "actual_anchor_ratio": len(anchors) / len(records),
            "encoded_full_bytes": full, "encoded_anchor_bytes": uploaded,
            "encoded_bytes": uploaded, "encoded_byte_reduction": 1 - uploaded / full,
            "encoding": "source_png_file_bytes",
            "predicted_hidden_uplink_bytes": 0}


def build_budget(records, anchor_stride, fixed_split):
    identities = [row.identity for row in records]
    bootstrap = fixed_split["bootstrap_end_candidate_index"]
    roles = stride_roles(identities, bootstrap, anchor_stride)
    intervals = build_anchor_intervals(records, roles)
    effective, tail = effective_records(records, intervals)
    split, split_payload = split_fixed_intervals(intervals, fixed_split)
    effective_roles = {row.identity.key: roles[row.identity.key] for row in effective}
    schedule = ratio_schedule_payload(
        [row.identity for row in effective], bootstrap_end_candidate_index=bootstrap,
        anchor_ratio=1.0 / anchor_stride,
    )
    return {"anchor_stride": anchor_stride, "records": effective,
            "roles": effective_roles, "intervals": intervals, "split": split,
            "split_payload": split_payload, "schedule": schedule,
            "effective_population": tail,
            "source_schedule": ratio_schedule_payload(
                identities, bootstrap_end_candidate_index=bootstrap, anchor_ratio=1.0 / anchor_stride)}


def online_identity_order(records, roles, intervals):
    """Canonical closing-anchor emission contract, without running a provider."""
    closing = {row.anchor1.key: row for row in intervals}
    result = []
    for row in records:
        identity = row.identity
        if roles[identity.key] != "anchor":
            continue
        if identity.key in closing:
            result.extend(q.identity.key for q in closing[identity.key].hidden)
        result.append(identity.key)
    return result


def backward_equivalence(records, protocol):
    """Exact stride-5 schedule/split guard, independent of previous result files."""
    fixed = protocol["fixed_split"]
    budget = build_budget(records, 5, fixed)
    reference_roles = post_bootstrap_ratio_roles(
        [r.identity for r in records],
        bootstrap_end_candidate_index=fixed["bootstrap_end_candidate_index"],
        anchor_ratio=0.2,
    )
    reference_intervals = build_anchor_intervals(records, reference_roles)
    reference_split, reference_payload = split_anchor_intervals(reference_intervals)
    reference_effective, reference_tail = effective_records(records, reference_intervals)
    checks = {
        "source_candidate_count": len(records) == fixed["source_candidate_count"],
        "interval_boundaries_alpha_delta_queries": budget["intervals"] == reference_intervals,
        "candidate_order_and_tail": budget["records"] == reference_effective and budget["effective_population"] == reference_tail,
        "roles": budget["roles"] == {r.identity.key: reference_roles[r.identity.key] for r in reference_effective},
        "split_membership": budget["split"] == reference_split,
        "frozen_split_hash": fixed["source_split_sha256"] == reference_payload["split_sha256"],
        "dropped_boundary_intervals": budget["split_payload"]["dropped_boundary_interval_indices"] == reference_payload["dropped_boundary_interval_indices"],
    }
    for name, region in fixed["regions"].items():
        first, last = reference_split[name][0].anchor0, reference_split[name][-1].anchor1
        checks[f"fixed_region_{name}"] = region == {
            "candidate_start": first.candidate_index, "candidate_end": last.candidate_index,
            "frame_start": first.frame_id, "frame_end": last.frame_id,
            "timestamp_start_ns": first.timestamp_ns, "timestamp_end_ns": last.timestamp_ns,
        }
    order = online_identity_order(budget["records"], budget["roles"], budget["intervals"])
    checks["online_order_exactly_once"] = order == [r.identity.key for r in budget["records"]] and len(order) == len(set(order))
    if not all(checks.values()):
        raise RuntimeError(f"stride=5 exact-equivalence failed: {[k for k,v in checks.items() if not v]}")
    return {"all_exact": True, "checks": checks, "gpu_numerical_smoke_run": False,
            "scientific_components": "reuse_unchanged_predictor_loss_transport_bridge_pipeline_dpvo"}


def stride_config(canonical, protocol, stride):
    result = copy.deepcopy(canonical)
    result.pop("canonical_predictor", None)
    result["paths"].pop("predictor", None)
    result["experiment"].update(anchor_stride=stride, anchor_ratio=1.0 / stride,
                                post_bootstrap_anchor_interval=stride)
    result["paths"]["output_root"] = protocol["output_root"]
    # Training receives split objects explicitly; never reuse the old count-based split.
    result["split"] = copy.deepcopy(protocol["fixed_split"])
    return result


def population_summary(budget):
    return {"anchor_stride": budget["anchor_stride"],
            "theoretical_anchor_ratio": 1.0 / budget["anchor_stride"],
            "communication": encoded_communication(budget["records"], budget["roles"]),
            "complete_intervals": len(budget["intervals"]),
            "effective_population": budget["effective_population"],
            "split": budget["split_payload"], "schedule": budget["schedule"]}


def prepare_protocol(records, protocol, strides=None):
    equivalence = backward_equivalence(records, protocol)
    requested = tuple(protocol["anchor_strides"] if strides is None else strides)
    budgets = {stride: build_budget(records, stride, protocol["fixed_split"])
               for stride in dict.fromkeys((*requested, 5))}
    # Full RGB retains its frozen stride-5 population. Other strides keep their
    # own complete intervals/tails; never truncate or promote anchors to match it.
    span = protocol["full_rgb_population"]
    full = tuple(r for r in records if span["candidate_start"] <= r.identity.candidate_index <= span["candidate_end"])
    if budgets[5]["records"] != full:
        raise RuntimeError("stride-5 Full RGB population changed")
    return budgets, {"status": "prepared_no_experiment_run", "sequence": protocol["sequence"],
                     "stride5_equivalence": equivalence, "fixed_split": protocol["fixed_split"],
                     "populations": [population_summary(budgets[s]) for s in requested]}
