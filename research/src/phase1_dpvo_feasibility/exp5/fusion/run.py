"""Run Phase 1 Exp5-Oracle JEPA FNet replacement validation."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..dataset import configured_sequences, load_sequence_records, validate_real_images
from ..run import REPO_ROOT, load_config as load_exp5_config
from .feature_hook import FNetInjectionHook, OracleFNetReplacementHook
from .projection import (
    DenseJepaProjection,
    build_random_dense_projection,
    load_dense_projection_checkpoint,
    state_dict_sha256,
    train_dense_projection,
)
from .runtime import (
    FUSION_METHODS,
    ORACLE_METHODS,
    MockDenseJepaWorkerClient,
    RUNTIME_KEYS,
    evaluate_dense_method,
    evaluate_test_alignment_online,
    extract_training_pairs,
    frame_availability,
    separate_environment_preflight,
)
from .trajectory_eval import (
    METHOD_ALIASES,
    SEQUENCE_ROLES,
    build_estimated_trajectory,
    build_ground_truth_subset,
    evaluate_trajectory,
    load_trajectory,
    write_trajectory,
)
from .trajectory_plot import plot_oracle_xy_trajectories, plot_xy_trajectories


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml"
SCHEMA_VERSION = 4
TEST_SEQUENCE = "MH_05_difficult"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="decode 8 real MH_01 frames and validate Oracle FNet replacement with CPU mocks",
    )
    return parser


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    local_path = Path(path).resolve()
    local = yaml.safe_load(local_path.read_text(encoding="utf-8"))
    if not isinstance(local, dict):
        raise TypeError(f"dense fusion config must be a mapping: {local_path}")
    base_path = (local_path.parent / str(local["base_config"])).resolve()
    config = copy.deepcopy(load_exp5_config(base_path))
    config["runtime"].pop("jepa_python", None)
    config["schema_version"] = int(local["schema_version"])
    for key in ("experiment", "dataset", "jepa"):
        config[key].update(copy.deepcopy(local[key]))
    for key in (
        "environment", "feature_hook", "fusion", "oracle", "projection", "evaluation"
    ):
        config[key] = copy.deepcopy(local[key])
    config["environment"]["vjepa_python"] = str(
        Path(config["environment"]["vjepa_python"]).resolve()
    )
    target = Path(config["experiment"]["result_root"])
    config["experiment"]["result_root"] = str(
        target.resolve() if target.is_absolute() else (REPO_ROOT / target).resolve()
    )
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported Exp5-2 schema: {config['schema_version']}")
    if tuple(configured_sequences(config)) != tuple(config["evaluation"]["sequences"]):
        raise ValueError("dense evaluation sequences must match train/val/test protocol")
    if tuple(config["fusion"]["methods"]) != FUSION_METHODS:
        raise ValueError("Exp5-2 method set/order is fixed")
    if tuple(config["oracle"]["methods"]) != ORACLE_METHODS:
        raise ValueError("Exp5-Oracle method set/order is fixed")
    if int(config["oracle"]["keyframe_stride"]) != 5:
        raise ValueError("Exp5-Oracle keyframe_stride is fixed at 5")
    if config["oracle"]["output_subdirectory"] != "oracle_missing_rgb":
        raise ValueError("Exp5-Oracle output directory is fixed")
    if float(config["fusion"]["alpha"]) != 0.1:
        raise ValueError("Exp5-2 alpha is fixed at 0.1")
    if float(config["feature_hook"]["fnet_scale"]) != 4.0:
        raise ValueError("Exp5-2 raw-FNet/fmap scale is fixed at 4.0")
    projection = config["projection"]
    fixed_projection = {
        "input_dim": 768,
        "output_dim": 128,
        "token_count": 576,
        "token_grid": [24, 24],
    }
    for key, expected in fixed_projection.items():
        if projection[key] != expected:
            raise ValueError(f"Exp5-2 projection {key} is fixed at {expected}")
    if not config["environment"].get("vjepa_python"):
        raise ValueError("Exp5-2 environment.vjepa_python is required")
    return config


def _fusion_formal_preflight(config: dict[str, Any]) -> dict[str, Any]:
    environment = separate_environment_preflight(config)
    for path in (
        Path(config["paths"]["calibration"]),
        Path(config["paths"]["dpvo_checkpoint"]),
        Path(config["paths"]["dpvo_config"]),
        Path(config["jepa"]["repo"]) / str(config["jepa"]["checkpoint"]),
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    dpvo_status = subprocess.check_output(
        ["git", "status", "--short", "--", "dpvo"], cwd=REPO_ROOT, text=True
    )
    if dpvo_status.strip():
        raise RuntimeError("formal Exp5-2 requires an unmodified upstream dpvo/ tree")
    import torch
    import cuda_ba  # noqa: F401
    import dpvo.altcorr  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("formal Exp5-2 requires CUDA")
    return environment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_reproducibility_contract(
    *, config: dict[str, Any], records_by_sequence: dict[str, list[dict[str, Any]]],
    projection_path: Path, source_metrics_path: Path,
) -> dict[str, Any]:
    """Create the one immutable run contract shared by baseline and Oracle."""
    source_digest = hashlib.sha256()
    source_paths = sorted(
        path for path in PACKAGE_DIR.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".md"}
    )
    for path in source_paths:
        source_digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        source_digest.update(path.read_bytes())
    record_hashes = {}
    for sequence, records in records_by_sequence.items():
        identities = [
            {
                "stream_index": int(record["stream_index"]),
                "frame_id": int(record["frame_id"]),
                "timestamp": int(record["timestamp"]),
                "image_path": str(Path(record["image_path"]).resolve()),
            }
            for record in records
        ]
        record_hashes[sequence] = _canonical_sha256(identities)
    dpvo_checkpoint = Path(config["paths"]["dpvo_checkpoint"]).resolve()
    dpvo_config = Path(config["paths"]["dpvo_config"]).resolve()
    calibration = Path(config["paths"]["calibration"]).resolve()
    contract = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "source_sha256": source_digest.hexdigest(),
        "dpvo_upstream_clean": True,
        "dpvo_checkpoint": {
            "path": str(dpvo_checkpoint), "sha256": _sha256(dpvo_checkpoint),
        },
        "dpvo_config": {"path": str(dpvo_config), "sha256": _sha256(dpvo_config)},
        "calibration": {"path": str(calibration), "sha256": _sha256(calibration)},
        "dataset_config_sha256": _canonical_sha256({
            "dataset": config["dataset"],
            "sequences": config["evaluation"]["sequences"],
        }),
        "sequence_sampling_sha256": record_hashes,
        "random_seed": int(config["experiment"]["seed"]),
        "projection": {
            "path": str(projection_path.resolve()), "sha256": _sha256(projection_path),
        },
        "exp5_2_metrics": {
            "path": str(source_metrics_path.resolve()),
            "sha256": _sha256(source_metrics_path),
        },
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def load_global_reference(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Exp5-1 reference metrics are required: {source}")
    metrics = json.loads(source.read_text(encoding="utf-8"))
    alignment = metrics.get("projection", {}).get("alignment", {})
    if not alignment or metrics.get("decision", {}).get("outcome") is None:
        raise ValueError("invalid Exp5-1 global reference metrics")
    return {
        "experiment": metrics.get("experiment"),
        "outcome": metrics["decision"]["outcome"],
        "metric_space": "global_vector_cosine",
        "alignment": alignment,
        "source_path": str(source.resolve()),
        "source_sha256": _sha256(source),
        "directly_comparable_to_dense_spatial_cosine": False,
    }


def from_scratch_global_reference() -> dict[str, Any]:
    """Represent the intentionally omitted Exp5-1 diagnostic in a clean run."""
    return {
        "experiment": "phase1_exp5_1_global_feature_fusion_validation",
        "outcome": "not_run_in_from_scratch_exp5_2_pipeline",
        "metric_space": "global_vector_cosine",
        "alignment": {},
        "source_path": None,
        "source_sha256": None,
        "directly_comparable_to_dense_spatial_cosine": False,
        "required_for_dense_or_oracle_decision": False,
    }


def _trajectory_relative_path(sequence: str, method: str) -> Path:
    return Path("trajectories") / f"{sequence}_{METHOD_ALIASES[method]}.json"


def _ground_truth_relative_path(sequence: str) -> Path:
    return Path("trajectories") / f"{sequence}_gt.json"


def _figure_relative_path(sequence: str) -> Path:
    short_name = sequence.rsplit("_", 1)[0]
    return Path("figures") / f"{short_name}_xy.png"


def build_trajectory_visualization(
    *, staging: Path, evaluations: dict[str, dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for sequence in config["evaluation"]["sequences"]:
        estimate_payloads = {
            alias: load_trajectory(staging / _trajectory_relative_path(sequence, method))
            for method, alias in METHOD_ALIASES.items()
        }
        all_timestamps = [
            int(timestamp)
            for payload in estimate_payloads.values()
            for timestamp in payload["timestamps"]
        ]
        groundtruth_source = REPO_ROOT / str(
            config["evaluation"]["groundtruth_pattern"]
        ).format(sequence=sequence)
        groundtruth = build_ground_truth_subset(
            sequence=sequence,
            source_path=groundtruth_source,
            timestamp_min=min(all_timestamps),
            timestamp_max=max(all_timestamps),
        )
        gt_relative = _ground_truth_relative_path(sequence)
        write_trajectory(staging / gt_relative, groundtruth)
        verification: dict[str, Any] = {}
        artifact_paths = {"ground_truth": gt_relative.as_posix()}
        for method, alias in METHOD_ALIASES.items():
            relative_path = _trajectory_relative_path(sequence, method)
            artifact_paths[alias] = relative_path.as_posix()
            recomputed = evaluate_trajectory(estimate_payloads[alias], groundtruth)
            runtime_ate = float(evaluations[sequence][method]["ate"]["translation_rmse"])
            exported_ate = float(recomputed["ate_rmse"])
            absolute_difference = abs(exported_ate - runtime_ate)
            verification[alias] = {
                "runtime_ate_rmse": runtime_ate,
                "exported_ate_rmse": exported_ate,
                "associated_count": int(recomputed["associated_count"]),
                "alignment": "Sim(3)",
                "absolute_difference": absolute_difference,
                "relative_difference": (
                    absolute_difference / runtime_ate if runtime_ate > 0.0 else None
                ),
                "diagnostic_only": True,
            }
        figure_relative = _figure_relative_path(sequence)
        plot_xy_trajectories(
            sequence=sequence,
            estimates=estimate_payloads,
            ground_truth=groundtruth,
            output_path=staging / figure_relative,
        )
        expected_count = int(evaluations[sequence]["dpvo_baseline"]["num_frames"])
        complete = bool(
            all(payload["num_poses"] == expected_count for payload in estimate_payloads.values())
            and all((staging / path).is_file() for path in artifact_paths.values())
            and (staging / figure_relative).is_file()
        )
        if not complete:
            raise RuntimeError(f"trajectory visualization artifacts incomplete: {sequence}")
        result[sequence] = {
            "role": SEQUENCE_ROLES[sequence],
            "trajectory_artifacts": artifact_paths,
            "figure_path": figure_relative.as_posix(),
            "completeness": True,
            "ate_verification": verification,
            "decision_input": sequence == TEST_SEQUENCE,
        }
    if set(result) != set(SEQUENCE_ROLES):
        raise AssertionError("trajectory visualization sequence inventory changed")
    return result


def validate_final_inventory(output_dir: Path) -> None:
    root_names = {path.name for path in output_dir.iterdir()}
    if root_names != {"metrics.json", "REPORT.md", "projection.pt", "trajectories", "figures"}:
        raise AssertionError(f"formal output root inventory changed: {sorted(root_names)}")
    expected_trajectories = {
        _trajectory_relative_path(sequence, method).name
        for sequence in SEQUENCE_ROLES
        for method in METHOD_ALIASES
    } | {_ground_truth_relative_path(sequence).name for sequence in SEQUENCE_ROLES}
    actual_trajectories = {
        path.name for path in (output_dir / "trajectories").iterdir() if path.is_file()
    }
    if actual_trajectories != expected_trajectories:
        raise AssertionError("trajectory artifact inventory changed")
    expected_figures = {_figure_relative_path(sequence).name for sequence in SEQUENCE_ROLES}
    actual_figures = {
        path.name for path in (output_dir / "figures").iterdir() if path.is_file()
    }
    if actual_figures != expected_figures or any(
        path.suffix.lower() != ".png" for path in (output_dir / "figures").iterdir()
    ):
        raise AssertionError("trajectory figure inventory changed")


def publish_replacing(staging: Path, target: Path, work_dir: Path) -> None:
    """Replace an old result only after the new staging inventory is complete."""
    backup = work_dir / "previous_result"
    moved_old = False
    if target.exists():
        os.replace(target, backup)
        moved_old = True
    try:
        os.replace(staging, target)
    except Exception:
        if moved_old and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    if moved_old:
        shutil.rmtree(backup)


def add_runtime_overhead(evaluations: dict[str, dict[str, Any]]) -> None:
    for sequence, methods in evaluations.items():
        baseline = float(methods["dpvo_baseline"]["runtime"]["end_to_end_total_time"])
        for method, row in methods.items():
            if set(row["runtime"]) != RUNTIME_KEYS:
                raise ValueError(f"runtime schema mismatch: {sequence}/{method}")
            total = float(row["runtime"]["end_to_end_total_time"])
            row["runtime_overhead_vs_baseline"] = {
                "seconds": total - baseline,
                "relative": ((total - baseline) / baseline) if baseline > 0.0 else None,
                "diagnostic_only": True,
            }


def _valid_ate(row: dict[str, Any]) -> float:
    value = float(row["ate"]["translation_rmse"])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid ATE: {value}")
    return value


def decide_dense_fusion(
    *, evaluations: dict[str, dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    rows = evaluations[TEST_SEQUENCE]
    baseline = rows["dpvo_baseline"]
    random_row = rows["random_dense_projection_fusion"]
    learned = rows["jepa_dense_projection_fusion"]
    workers_valid = all(
        row.get("status") == "ok" and row.get("tracking", {}).get("completed")
        for row in rows.values()
    )
    baseline_tracking = bool(baseline["tracking"]["success"])
    learned_tracking = bool(learned["tracking"]["success"])
    random_tracking = bool(random_row["tracking"]["success"])
    coverage_maintained = bool(
        learned["tracking"]["pose_count"] >= baseline["tracking"]["pose_count"]
        and learned["tracking"]["gt_associated_count"]
        >= baseline["tracking"]["gt_associated_count"]
    )
    engineering_valid = bool(
        workers_valid
        and baseline_tracking
        and random_tracking
        and learned_tracking
        and coverage_maintained
    )
    baseline_ate = _valid_ate(baseline)
    random_ate = _valid_ate(random_row)
    learned_ate = _valid_ate(learned)
    learned_beats_random = learned_ate < random_ate
    degradation = (
        (learned_ate - baseline_ate) / baseline_ate if baseline_ate > 0.0 else float("inf")
    )
    strong_limit = float(config["evaluation"]["strong_ate_degradation"])
    feasible_limit = float(config["evaluation"]["feasible_ate_degradation"])
    def at_boundary(value: float, limit: float) -> bool:
        return value < limit or math.isclose(
            value, limit, rel_tol=1.0e-12, abs_tol=1.0e-12
        )
    if not engineering_valid:
        outcome, evidence = "failure_dense_fusion", "engineering_failure"
        reason = "tracking, coverage, worker, or lifecycle contract failed"
    elif not at_boundary(degradation, feasible_limit):
        outcome, evidence = "failure_dense_fusion", "failure"
        reason = "MH_05 learned ATE degradation exceeds the 10% feasibility bound"
    elif not learned_beats_random:
        outcome, evidence = "inconclusive_dense_fusion", "inconclusive"
        reason = "tracking is maintained but learned dense fusion does not beat random dense fusion"
    elif at_boundary(degradation, strong_limit):
        outcome, evidence = "success_dense_fusion", "strong"
        reason = "learned dense fusion beats random and remains within 5% of RGB DPVO baseline"
    else:
        outcome, evidence = "success_dense_fusion", "feasible"
        reason = "learned dense fusion beats random and remains within 10% of RGB DPVO baseline"
    return {
        "outcome": outcome,
        "evidence": evidence,
        "sequence": TEST_SEQUENCE,
        "learned_beats_random": learned_beats_random,
        "tracking_success": learned_tracking,
        "coverage_maintained": coverage_maintained,
        "ate_degradation_vs_baseline": degradation,
        "strong_limit": strong_limit,
        "feasible_limit": feasible_limit,
        "requires_ate_better_than_baseline": False,
        "dense_cosine_is_effectiveness_gate": False,
        "runtime_is_effectiveness_gate": False,
        "reason": reason,
    }


def build_metrics(
    *, config: dict[str, Any], projection_metadata: dict[str, Any],
    extraction_metadata: dict[str, Any], test_alignment: dict[str, Any],
    evaluations: dict[str, dict[str, Any]], global_reference: dict[str, Any], smoke: bool,
    environment_bridge: dict[str, Any] | None = None,
    trajectory_visualization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    add_runtime_overhead(evaluations)
    decision = decide_dense_fusion(evaluations=evaluations, config=config)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": str(config["experiment"]["name"]),
        "mode": "smoke" if smoke else "formal",
        "mock_models": bool(smoke),
        "environment_bridge": environment_bridge or {
            "mode": "mock_separate_environments",
            "transport": "mock_persistent_stdio_json_base64",
            "shared_memory": False,
            "automatic_install": False,
            "passed": True,
        },
        "research_question": (
            "Can spatially structured dense JEPA tokens provide local constraints that "
            "global broadcast fusion did not establish?"
        ),
        "fusion": {
            "hook": "raw Patchifier.fnet output",
            "projection": "Conv2d(768,128,kernel_size=1)",
            "token_grid": [24, 24],
            "interpolation": {"mode": "bilinear", "align_corners": False},
            "alpha": 0.1,
            "projection_to_raw_scale": 4.0,
            "downstream_contract": "raw_fnet/4 + 0.1*projected_dense_fmap",
        },
        "projection_training": projection_metadata,
        "pair_extraction": extraction_metadata,
        "representation_alignment": {
            "exp5_1_global": global_reference,
            "exp5_2_dense": {
                "metric_space": "dense_spatial_cosine",
                "validation": {
                    "learned": projection_metadata["learned_validation_alignment"],
                    "random": projection_metadata["random_validation_alignment"],
                },
                "test": test_alignment,
            },
            "direct_numeric_comparison_valid": False,
        },
        "evaluations": evaluations,
        "trajectory_visualization": trajectory_visualization or {},
        "decision": decision,
        "interpretation_risk": (
            "V-JEPA 24×24 tokens describe a 384×384 center crop while direct bilinear "
            "resize is applied across the full DPVO feature extent."
        ),
        "status": "smoke_pass" if smoke else decision["outcome"],
    }


def render_report(metrics: dict[str, Any]) -> str:
    decision = metrics["decision"]
    rows: list[str] = []
    for sequence, methods in metrics["evaluations"].items():
        for method in FUSION_METHODS:
            row = methods[method]
            rows.append(
                f"| {sequence} | {method} | {row['tracking']['success']} | "
                f"{row['tracking']['pose_count']}/{row['tracking']['expected_count']} | "
                f"{row['tracking']['gt_associated_count']} | "
                f"{row['ate']['translation_rmse']:.6f} | "
                f"{row['runtime']['jepa_extraction_time']:.3f} | "
                f"{row['runtime']['environment_bridge_time']:.3f} | "
                f"{row['runtime']['worker_setup_time']:.3f} | "
                f"{row['runtime']['dense_projection_time']:.3f} | "
                f"{row['runtime']['fusion_hook_time']:.3f} | "
                f"{row['runtime']['dpvo_total_time']:.3f} | "
                f"{row['runtime']['end_to_end_total_time']:.3f} |"
            )
    global_reference = metrics["representation_alignment"]["exp5_1_global"]
    dense = metrics["representation_alignment"]["exp5_2_dense"]
    global_rows = [
        f"| {sequence} | {values['learned_mean_cosine_similarity']:.6f} | "
        f"{values['random_mean_cosine_similarity']:.6f} |"
        for sequence, values in global_reference["alignment"].items()
    ]
    if global_rows:
        global_section = [
            f"Exp5-1 global-vector cosine (outcome: `{global_reference['outcome']}`):",
            "",
            "| Sequence | Learned global cosine | Random global cosine |",
            "|---|---:|---:|",
            *global_rows,
            "",
        ]
    else:
        global_section = [
            "Exp5-1 global-vector cosine is not rerun or loaded by the self-contained "
            "Exp5-2 → Exp5-Oracle pipeline. It is a historical diagnostic and is not "
            "required by either decision.",
            "",
        ]
    trajectory_lines: list[str] = []
    interpretation_titles = {
        "MH_01_easy": "Training-sequence diagnostic",
        "MH_03_medium": "Transfer diagnostic",
        "MH_05_difficult": "Decision-sequence visualization",
    }
    for sequence in ("MH_01_easy", "MH_03_medium", "MH_05_difficult"):
        visualization = metrics["trajectory_visualization"][sequence]
        methods = metrics["evaluations"][sequence]
        baseline_ate = float(methods["dpvo_baseline"]["ate"]["translation_rmse"])
        random_ate = float(
            methods["random_dense_projection_fusion"]["ate"]["translation_rmse"]
        )
        learned_ate = float(
            methods["jepa_dense_projection_fusion"]["ate"]["translation_rmse"]
        )
        verification_max = max(
            float(row["absolute_difference"])
            for row in visualization["ate_verification"].values()
        )
        trajectory_lines += [
            f"### {sequence} — {interpretation_titles[sequence]}",
            "",
            f"ATE [m] — baseline `{baseline_ate:.6f}`, random `{random_ate:.6f}`, "
            f"learned `{learned_ate:.6f}`. Tracking is "
            f"`{methods['jepa_dense_projection_fusion']['tracking']['success']}` for learned; "
            f"the XY figure is `{visualization['figure_path']}`. Maximum exported-ATE "
            f"absolute recomputation difference is `{verification_max:.3e}`.",
            "",
        ]
        if sequence == "MH_01_easy":
            trajectory_lines += [
                "This plot diagnoses training-sequence behavior only. It helps show whether "
                "the learned projection changes the in-split trajectory relative to baseline "
                "and random fusion, but it does not contribute to classification.",
                "",
            ]
        elif sequence == "MH_03_medium":
            trajectory_lines += [
                "This plot diagnoses transfer behavior. Learned-versus-baseline/random ATE "
                "and trajectory shape indicate whether the training-sequence behavior persists "
                "or disappears out of split; it does not contribute to classification.",
                "",
            ]
        else:
            trajectory_lines += [
                "This is the only decision-sequence visualization. Its trajectory shape is "
                "interpreted together with MH_05 ATE, tracking and pose/GT coverage, while the "
                "existing decision criteria remain unchanged.",
                "",
            ]
    lines = [
        "# Phase 1 / Experiment 5-2 — Dense JEPA Token Fusion Validation",
        "",
        "## Result",
        "",
        f"**{metrics['status']}** (evidence: **{decision['evidence']}**)",
        "",
    ]
    if metrics["mode"] == "smoke":
        lines += [
            "This is a CPU smoke using real EuRoC images and mock model outputs. "
            "It is not dense-fusion or SLAM evidence.",
            "",
        ]
    lines += [
        "## Representation alignment",
        "",
        *global_section,
        "Exp5-2 dense spatial cosine:",
        "",
        "| Split | Learned dense cosine | Random dense cosine |",
        "|---|---:|---:|",
        f"| MH_03 validation | "
        f"{dense['validation']['learned']['mean_spatial_cosine_similarity']:.6f} | "
        f"{dense['validation']['random']['mean_spatial_cosine_similarity']:.6f} |",
        f"| MH_05 test | {dense['test']['learned_mean_spatial_cosine_similarity']:.6f} | "
        f"{dense['test']['random_mean_spatial_cosine_similarity']:.6f} |",
        "",
        "These cosine values belong to different representation spaces and must not be "
        "compared numerically. Exp5-2 tests whether preserving spatial structure resolves "
        "the downstream limitation observed in global broadcast fusion.",
        "",
        "## Fusion contract",
        "",
        "Dense tokens `[576,768]` are reshaped to a 24×24 grid, projected with one 1×1 "
        "convolution and resized with bilinear interpolation. The adapter is trained against "
        "`raw_fnet/4`. At the raw FNet hook its residual is multiplied by four, so downstream "
        "fusion is `raw_fnet/4 + 0.1 * projected_dense_fmap`.",
        "",
        "## Environment bridge",
        "",
        "DPVO remains in the configured DPVO environment. Each JEPA-consuming run starts "
        "one persistent V-JEPA subprocess in the configured V-JEPA environment, reuses it "
        "for the complete sequence, and then closes it. Dense tokens use synchronous "
        "stdin/stdout JSON with base64 float32 payloads; there is no shared memory, socket "
        "service, per-frame Python startup, dependency installation or latent cache.",
        "",
        "Worker setup and environment-bridge transfer are diagnostic measurements only and "
        "are not dense-fusion effectiveness gates.",
        "",
        "## SLAM evaluation",
        "",
        "| Sequence | Method | Tracking | Pose coverage | GT associations | Sim(3) ATE | JEPA s | Bridge s | Setup s | Projection s | Hook s | DPVO s | E2E s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "MH_01 and MH_03 are diagnostic. MH_05 determines the final classification. Runtime "
        "overhead is diagnostic and is not an effectiveness gate.",
        "",
        "## Answers",
        "",
        "1. **Do dense tokens provide a stronger local representation interface?** "
        "See learned-versus-random dense spatial cosine above; it is not numerically compared "
        "with Exp5-1 global cosine.",
        "2. **Does learned dense fusion beat random dense fusion?** "
        + ("Yes." if decision["learned_beats_random"] else "No."),
        "3. **Does dense fusion remain close to RGB DPVO baseline?** "
        f"MH_05 ATE degradation is `{decision['ate_degradation_vs_baseline']:.2%}`; "
        f"classification is `{decision['evidence']}`.",
        "",
        f"Decision: **{decision['outcome']}**. {decision['reason']}.",
        "",
        "## Trajectory visualization",
        "",
        "All figures show raw exported estimates after per-method timestamp association and "
        "Sim(3) alignment to GT. JSON artifacts retain only the original, unaligned terminate "
        "trajectories. Exported-trajectory ATE verification is diagnostic and has no threshold.",
        "",
        *trajectory_lines,
        "MH_01 and MH_03 visualize training-internal and transfer behavior only. Exp5-2 "
        "success/inconclusive/failure classification continues to use MH_05 alone. Adding "
        "trajectory artifacts does not change fusion, training or decision criteria.",
        "",
        "## Interpretation risk",
        "",
        metrics["interpretation_risk"],
        "A failure therefore cannot isolate dense representation quality from this direct-grid "
        "coordinate approximation.",
        "",
    ]
    return "\n".join(lines)


def _write_outputs(output_dir: Path, metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(render_report(metrics), encoding="utf-8")


def _mock_runtime(total: float, *, fused: bool) -> dict[str, float]:
    result = {
        "jepa_extraction_time": 0.08 if fused else 0.0,
        "environment_bridge_time": 0.01 if fused else 0.0,
        "worker_setup_time": 0.2 if fused else 0.0,
        "dense_projection_time": 0.004 if fused else 0.0,
        "fusion_hook_time": 0.002 if fused else 0.0,
        "dpvo_total_time": total - (0.094 if fused else 0.0),
        "end_to_end_total_time": total,
    }
    if set(result) != RUNTIME_KEYS:
        raise AssertionError("mock dense runtime schema changed")
    return result


def mock_evaluations(frame_count: int) -> dict[str, dict[str, Any]]:
    values = {
        "MH_01_easy": (0.10, 0.11, 0.09),
        "MH_03_medium": (0.20, 0.22, 0.205),
        "MH_05_difficult": (0.30, 0.34, 0.312),
    }
    result: dict[str, dict[str, Any]] = {}
    for sequence, ates in values.items():
        result[sequence] = {}
        for method, ate in zip(FUSION_METHODS, ates):
            fused = method != "dpvo_baseline"
            result[sequence][method] = {
                "status": "ok",
                "method": method,
                "sequence": sequence,
                "num_frames": frame_count,
                "ate": {
                    "translation_rmse": ate,
                    "associated_pose_count": frame_count,
                    "alignment": "Sim(3)",
                },
                "tracking": {
                    "completed": True,
                    "pose_count": frame_count,
                    "expected_count": frame_count,
                    "gt_associated_count": frame_count,
                    "ordinal_timestamp_aligned": True,
                    "poses_finite": True,
                    "success": True,
                },
                "runtime": _mock_runtime(1.1 if fused else 1.0, fused=fused),
                "hook_lifecycle": {
                    "initialized": fused,
                    "removed": fused,
                    "cleanup_passed": True,
                    "registration_mode": "register_forward_hook" if fused else None,
                    "consumed_frames": frame_count if fused else 0,
                },
                "worker_lifecycle": {
                    "used": fused,
                    "python": "mock" if fused else None,
                    "initialized": fused,
                    "request_count": frame_count if fused else 0,
                    "stopped": fused,
                    "cleanup_passed": True,
                    "worker_setup_time": 0.2 if fused else 0.0,
                    "worker_extraction_time": 0.08 if fused else 0.0,
                    "environment_bridge_time": 0.01 if fused else 0.0,
                },
            }
    return result


def _mock_global_reference() -> dict[str, Any]:
    return {
        "experiment": "phase1_exp5_1_global_feature_fusion_validation",
        "outcome": "fail_global_fusion",
        "metric_space": "global_vector_cosine",
        "alignment": {
            "MH_05_difficult": {
                "learned_mean_cosine_similarity": 0.95,
                "random_mean_cosine_similarity": 0.0,
            }
        },
        "source_path": "mock",
        "source_sha256": "0" * 64,
        "directly_comparable_to_dense_spatial_cosine": False,
    }


def _write_mock_trajectories(
    *, output: Path, config: dict[str, Any], frame_count: int,
) -> None:
    from evo.tools import file_interface

    for sequence in config["evaluation"]["sequences"]:
        groundtruth_source = REPO_ROOT / str(
            config["evaluation"]["groundtruth_pattern"]
        ).format(sequence=sequence)
        reference = file_interface.read_tum_trajectory_file(str(groundtruth_source))
        gt_timestamps = np.rint(reference.timestamps[:frame_count]).astype(np.int64)
        records = [
            {"stream_index": ordinal, "frame_id": ordinal, "timestamp": int(timestamp)}
            for ordinal, timestamp in enumerate(gt_timestamps)
        ]
        index = np.arange(frame_count, dtype=np.float64)
        base_positions = np.stack(
            (0.12 * index, np.sin(index * 0.4), 0.08 * np.cos(index * 0.3)), axis=1
        )
        for method, alias in METHOD_ALIASES.items():
            offset = {"baseline": 0.0, "random": 0.02, "learned": -0.01}[alias]
            positions = base_positions.copy()
            positions[:, 1] += offset * np.cos(index)
            quaternions = np.tile(
                np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
                (frame_count, 1),
            )
            poses = np.concatenate((positions, quaternions), axis=1)
            payload = build_estimated_trajectory(
                sequence=sequence,
                method=method,
                records=records,
                ordinals=np.arange(frame_count, dtype=np.float64),
                poses=poses,
            )
            write_trajectory(output / _trajectory_relative_path(sequence, method), payload)


def run_dense_smoke(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    sequence = str(config["dataset"]["smoke_sequence"])
    records = load_sequence_records(
        config, sequence, limit=int(config["dataset"]["smoke_frames"])
    )
    image_shapes = validate_real_images(records)
    count = len(records)
    generator = np.random.default_rng(int(config["experiment"]["seed"]))
    source_tokens = generator.standard_normal((count, 576, 768), dtype=np.float32)
    mock_worker = MockDenseJepaWorkerClient({
        int(record.frame_id): source_tokens[index]
        for index, record in enumerate(records)
    })
    extracted_tokens: list[np.ndarray] = []
    for record in records:
        dense, _, _ = mock_worker.extract(record.to_dict())
        extracted_tokens.append(dense)
    mock_worker.close()
    worker_lifecycle = mock_worker.lifecycle()
    if (
        worker_lifecycle["request_count"] != count
        or not worker_lifecycle["cleanup_passed"]
    ):
        raise AssertionError("persistent mock worker lifecycle failed")
    tokens = np.stack(extracted_tokens).astype(np.float32, copy=False)
    teacher_small = np.repeat(tokens[:, :1, :128].transpose(0, 2, 1), 96, axis=2)
    teacher_small = teacher_small.reshape(count, 128, 8, 12).astype(np.float32)
    train_pairs = {"tokens": tokens.copy(), "teacher": teacher_small.copy()}
    val_pairs = {"tokens": tokens.copy(), "teacher": teacher_small.copy()}
    smoke_config = copy.deepcopy(config)
    smoke_config["projection"]["epochs"] = 5
    smoke_config["projection"]["validation_interval"] = 5
    smoke_config["projection"]["batch_size"] = 2

    class MockFNet(nn.Module):
        def forward(self, _image: torch.Tensor) -> torch.Tensor:
            return torch.ones(1, 1, 128, 120, 188)

    with tempfile.TemporaryDirectory(prefix="exp5-dense-smoke-") as temporary_name:
        output = Path(temporary_name) / "dense_fusion_validation"
        checkpoint = output / "projection.pt"
        model, projection_metadata = train_dense_projection(
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            config=smoke_config,
            checkpoint_path=checkpoint,
            device=torch.device("cpu"),
        )
        projected = model(torch.from_numpy(tokens[:1]), target_size=(120, 188))
        if tuple(projected.shape) != (1, 128, 120, 188):
            raise AssertionError("dense projection smoke shape failed")
        fnet = MockFNet()
        hook = FNetInjectionHook(alpha=0.1, channels=128, projection_to_raw_scale=4.0)
        hook.install(fnet)
        hook.bind(records[0], projected)
        raw = fnet(torch.zeros(1))
        hook.finish_frame(records[0])
        hook.close()
        downstream = raw / 4.0
        expected = torch.ones_like(raw) / 4.0 + 0.1 * projected
        if not torch.allclose(downstream, expected, atol=1e-6) or not hook.cleanup_passed:
            raise AssertionError("dense /4 scale contract smoke failed")
        random_a = build_random_dense_projection(smoke_config, device=torch.device("cpu"))
        random_b = build_random_dense_projection(smoke_config, device=torch.device("cpu"))
        if state_dict_sha256(random_a.state_dict()) != state_dict_sha256(random_b.state_dict()):
            raise AssertionError("random dense projection is not deterministic")
        test_alignment = {
            "sample_count": count,
            "spatial_vector_count": count * 120 * 188,
            "learned_mean_spatial_cosine_similarity": 0.8,
            "random_mean_spatial_cosine_similarity": 0.0,
            "evaluation_seconds": 0.0,
            "jepa_extraction_seconds": 0.0,
            "environment_bridge_seconds": 0.0,
            "worker_lifecycle": worker_lifecycle,
            "pair_retained": False,
        }
        extraction = {
            "pair_storage": "preallocated_CPU_RAM_float32_only",
            "persistent_cache": False,
            "split_counts": {"train": count, "val": count},
            "token_shape": [576, 768],
            "teacher_shape": [128, 120, 188],
            "teacher_contract": "raw Patchifier.fnet output / 4.0",
            "pair_bytes": int(tokens.nbytes * 2 + teacher_small.nbytes * 2),
            "memory_gate": {"passed": True},
            "extraction_seconds": 0.0,
            "jepa_extraction_seconds": 0.0,
            "environment_bridge_seconds": 0.0,
            "worker_setup_seconds": 0.0,
            "worker_lifecycles": {
                "train": worker_lifecycle,
                "val": worker_lifecycle,
            },
            "worker_scope": "one_per_preparation_sequence",
            "uses_groundtruth": False,
            "uses_trajectory": False,
            "uses_future_frames": False,
        }
        evaluations = mock_evaluations(count)
        _write_mock_trajectories(
            output=output,
            config=smoke_config,
            frame_count=count,
        )
        trajectory_visualization = build_trajectory_visualization(
            staging=output,
            evaluations=evaluations,
            config=smoke_config,
        )
        metrics = build_metrics(
            config=smoke_config,
            projection_metadata=projection_metadata,
            extraction_metadata=extraction,
            test_alignment=test_alignment,
            evaluations=evaluations,
            global_reference=_mock_global_reference(),
            smoke=True,
            trajectory_visualization=trajectory_visualization,
        )
        _write_outputs(output, metrics)
        validate_final_inventory(output)
        persisted = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        if persisted["status"] != "smoke_pass":
            raise AssertionError("dense smoke metrics failed")
    return {
        "status": "smoke_pass",
        "sequence": sequence,
        "num_frames": count,
        "decoded_image_shapes": sorted({tuple(shape) for shape in image_shapes}),
        "token_shape": [1, 576, 768],
        "projection_shape": [1, 128, 120, 188],
        "scale_contract_passed": True,
        "random_deterministic": True,
        "worker_initialized_once": worker_lifecycle["initialized"],
        "worker_request_count": worker_lifecycle["request_count"],
        "worker_cleanup_passed": worker_lifecycle["cleanup_passed"],
        "trajectory_sequences": sorted(trajectory_visualization),
        "trajectory_json_count": 12,
        "trajectory_png_count": 3,
        "decision": metrics["decision"]["outcome"],
        "models": "mock",
        "artifacts_retained": False,
    }


def run_dense_formal(config: dict[str, Any]) -> dict[str, Any]:
    target = Path(config["experiment"]["result_root"])
    environment = _fusion_formal_preflight(config)
    global_reference = from_scratch_global_reference()
    records_by_split = {
        split: [record.to_dict() for record in load_sequence_records(config, sequence)]
        for split, sequence in zip(("train", "val", "test"), configured_sequences(config))
    }
    for sequence in config["evaluation"]["sequences"]:
        groundtruth = REPO_ROOT / str(config["evaluation"]["groundtruth_pattern"]).format(
            sequence=sequence
        )
        if not groundtruth.is_file():
            raise FileNotFoundError(groundtruth)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exp5-dense-", dir=target.parent) as temporary_name:
        work_dir = Path(temporary_name)
        staging = work_dir / "publish"
        staging.mkdir()
        checkpoint = staging / "projection.pt"
        pairs, extraction_metadata = extract_training_pairs(
            config=config,
            records_by_split={"train": records_by_split["train"], "val": records_by_split["val"]},
            work_dir=work_dir,
        )
        import torch

        learned, projection_metadata = train_dense_projection(
            train_pairs=pairs["train"],
            val_pairs=pairs["val"],
            config=config,
            checkpoint_path=checkpoint,
            device=torch.device("cuda:0"),
        )
        pairs.clear()
        del pairs
        gc.collect()
        random_model = build_random_dense_projection(config, device=torch.device("cuda:0"))
        test_alignment = evaluate_test_alignment_online(
            config=config,
            records=records_by_split["test"],
            learned=learned,
            random_model=random_model,
            work_dir=work_dir,
        )
        del learned, random_model
        torch.cuda.empty_cache()
        evaluations: dict[str, dict[str, Any]] = {}
        sequence_to_split = {
            sequence: split
            for split, sequence in zip(("train", "val", "test"), configured_sequences(config))
        }
        for sequence in config["evaluation"]["sequences"]:
            evaluations[sequence] = {}
            for method in FUSION_METHODS:
                evaluations[sequence][method] = evaluate_dense_method(
                    config=config,
                    records=records_by_split[sequence_to_split[sequence]],
                    method=method,
                    checkpoint_path=checkpoint,
                    work_dir=work_dir,
                    trajectory_path=staging / _trajectory_relative_path(sequence, method),
                )
        trajectory_visualization = build_trajectory_visualization(
            staging=staging,
            evaluations=evaluations,
            config=config,
        )
        extraction_metadata["environment_bridge"] = environment
        metrics = build_metrics(
            config=config,
            projection_metadata=projection_metadata,
            extraction_metadata=extraction_metadata,
            test_alignment=test_alignment,
            evaluations=evaluations,
            global_reference=global_reference,
            smoke=False,
            environment_bridge=environment,
            trajectory_visualization=trajectory_visualization,
        )
        _write_outputs(staging, metrics)
        validate_final_inventory(staging)
        publish_replacing(staging, target, work_dir)
    return metrics


def decide_oracle_replacement(
    *, evaluations: dict[str, dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    rows = evaluations[TEST_SEQUENCE]
    baseline = rows["dpvo_baseline"]
    oracle = rows["oracle_jepa_missing_rgb"]
    lifecycle = oracle.get("hook_lifecycle", {})
    availability = oracle.get("availability", {})
    same_contract = bool(
        baseline.get("reproducibility_contract_sha256")
        and baseline.get("reproducibility_contract_sha256")
        == oracle.get("reproducibility_contract_sha256")
    )
    engineering_valid = bool(
        baseline.get("status") == "ok"
        and oracle.get("status") == "ok"
        and baseline.get("tracking", {}).get("success")
        and oracle.get("tracking", {}).get("success")
        and lifecycle.get("cleanup_passed")
        and lifecycle.get("consumed_frames") == oracle.get("num_frames")
        and availability.get("hook_counts_valid")
        and same_contract
    )
    baseline_ate = _valid_ate(baseline)
    oracle_ate = _valid_ate(oracle)
    degradation = (
        (oracle_ate - baseline_ate) / baseline_ate
        if baseline_ate > 0.0 else float("inf")
    )
    strong_limit = float(config["oracle"]["strong_ate_degradation"])
    feasible_limit = float(config["oracle"]["feasible_ate_degradation"])

    def at_boundary(value: float, limit: float) -> bool:
        return value < limit or math.isclose(
            value, limit, rel_tol=1.0e-12, abs_tol=1.0e-12
        )

    if not engineering_valid:
        outcome, evidence = "failure_oracle_fmap_replacement", "engineering_failure"
        reason = "tracking, identity, hook lifecycle, or reproducibility contract failed"
    elif at_boundary(degradation, strong_limit):
        outcome, evidence = "success_oracle_fmap_replacement", "strong"
        reason = "MH_05 Oracle ATE degradation is within 5% of fresh full-RGB baseline"
    elif at_boundary(degradation, feasible_limit):
        outcome, evidence = "success_oracle_fmap_replacement", "feasible"
        reason = "MH_05 Oracle ATE degradation is above 5% but within 10%"
    else:
        outcome, evidence = "failure_oracle_fmap_replacement", "failure"
        reason = "MH_05 Oracle ATE degradation exceeds 10%"
    return {
        "outcome": outcome,
        "evidence": evidence,
        "sequence": TEST_SEQUENCE,
        "tracking_success": bool(oracle.get("tracking", {}).get("success")),
        "identity_hook_valid": bool(
            lifecycle.get("cleanup_passed") and availability.get("hook_counts_valid")
        ),
        "reproducibility_contract_equal": same_contract,
        "ate_degradation_vs_baseline": degradation,
        "strong_limit": strong_limit,
        "feasible_limit": feasible_limit,
        "coverage_is_independent_gate": False,
        "runtime_is_effectiveness_gate": False,
        "reason": reason,
    }


def mock_oracle_evaluations(
    frame_count: int, *, keyframe_stride: int = 5,
) -> dict[str, dict[str, Any]]:
    values = {
        "MH_01_easy": (0.10, 0.104),
        "MH_03_medium": (0.20, 0.207),
        "MH_05_difficult": (0.30, 0.312),
    }
    result: dict[str, dict[str, Any]] = {}
    contract_hash = "c" * 64
    for sequence, ates in values.items():
        records = [
            {
                "stream_index": index,
                "frame_id": index,
                "timestamp": 1_403_000_000_000_000_000 + index * 100,
            }
            for index in range(frame_count)
        ]
        _, availability = frame_availability(
            records, keyframe_stride=keyframe_stride
        )
        availability["hook_counts_valid"] = True
        result[sequence] = {}
        for method, ate in zip(ORACLE_METHODS, ates):
            oracle = method == "oracle_jepa_missing_rgb"
            result[sequence][method] = {
                "status": "ok",
                "method": method,
                "sequence": sequence,
                "num_frames": frame_count,
                "ate": {
                    "translation_rmse": ate,
                    "associated_pose_count": frame_count,
                    "alignment": "Sim(3)",
                },
                "tracking": {
                    "completed": True,
                    "pose_count": frame_count,
                    "expected_count": frame_count,
                    "gt_associated_count": frame_count,
                    "ordinal_timestamp_aligned": True,
                    "poses_finite": True,
                    "success": True,
                },
                "runtime": _mock_runtime(1.15 if oracle else 1.0, fused=oracle),
                "hook_lifecycle": {
                    "initialized": oracle,
                    "removed": oracle,
                    "cleanup_passed": True,
                    "registration_mode": "forward_wrapper" if oracle else None,
                    "consumed_frames": frame_count if oracle else 0,
                    "fnet_executed_frames": (
                        availability["keyframe_count"] if oracle else frame_count
                    ),
                    "jepa_replaced_frames": (
                        availability["non_keyframe_count"] if oracle else 0
                    ),
                    "output_contract": [1, 1, 128, 120, 188] if oracle else None,
                },
                "worker_lifecycle": {
                    "used": oracle,
                    "python": "mock" if oracle else None,
                    "initialized": oracle,
                    "request_count": frame_count if oracle else 0,
                    "stopped": oracle,
                    "cleanup_passed": True,
                    "worker_setup_time": 0.2 if oracle else 0.0,
                    "worker_extraction_time": 0.08 if oracle else 0.0,
                    "environment_bridge_time": 0.01 if oracle else 0.0,
                },
                "reproducibility_contract_sha256": contract_hash,
            }
            if oracle:
                result[sequence][method]["availability"] = copy.deepcopy(availability)
    return result


def build_oracle_metrics(
    *, config: dict[str, Any], evaluations: dict[str, dict[str, Any]],
    reproducibility: dict[str, Any], source_reference: dict[str, Any],
    trajectory_evidence: dict[str, Any], smoke: bool,
) -> dict[str, Any]:
    add_runtime_overhead(evaluations)
    decision = decide_oracle_replacement(evaluations=evaluations, config=config)
    return {
        "schema_version": 1,
        "experiment": "phase1_exp5_oracle_fnet_replacement_validation",
        "mode": "smoke" if smoke else "formal",
        "mock_models": bool(smoke),
        "research_question": (
            "Can Oracle dense JEPA replace non-keyframe RGB-derived FNet matching features?"
        ),
        "boundary": {
            "name": "Patchifier.fnet output",
            "tensor_interface": "[B,N,128,H,W]",
            "keyframe_formula": "raw_fnet + 0.1 * 4.0 * projected_dense",
            "non_keyframe_formula": "4.0 * projected_dense",
            "inet_context_unchanged": True,
            "context_rgb_used": True,
            "complete_rgb_replacement_claimed": False,
        },
        "source_exp5_2": source_reference,
        "reproducibility": reproducibility,
        "evaluations": evaluations,
        "trajectory_evidence": trajectory_evidence,
        "decision": decision,
        "status": "smoke_pass" if smoke else decision["outcome"],
    }


def render_oracle_report(metrics: dict[str, Any]) -> str:
    decision = metrics["decision"]
    rows = []
    for sequence in ("MH_01_easy", "MH_03_medium", "MH_05_difficult"):
        baseline = metrics["evaluations"][sequence]["dpvo_baseline"]
        oracle = metrics["evaluations"][sequence]["oracle_jepa_missing_rgb"]
        availability = oracle["availability"]
        rows.append(
            f"| {sequence} | {baseline['ate']['translation_rmse']:.6f} | "
            f"{oracle['ate']['translation_rmse']:.6f} | "
            f"{oracle['tracking']['success']} | "
            f"{oracle['tracking']['pose_count']}/{oracle['tracking']['expected_count']} | "
            f"{availability['matching_rgb_ratio']:.4f} | "
            f"{availability['jepa_replaced_ratio']:.4f} | "
            f"{baseline['runtime']['end_to_end_total_time']:.3f} | "
            f"{oracle['runtime']['end_to_end_total_time']:.3f} |"
        )
    lines = [
        "# Phase 1 / Exp5-Oracle — Oracle JEPA FNet Replacement Validation",
        "",
        "## Result",
        "",
        f"**{metrics['status']}** (evidence: **{decision['evidence']}**)",
        "",
    ]
    if metrics["mode"] == "smoke":
        lines += [
            "This is a CPU smoke with real EuRoC image decoding and mock model outputs. "
            "It is not Oracle replacement or SLAM evidence.",
            "",
        ]
    lines += [
        "## Interpretation boundary",
        "",
        "Exp5-Oracle evaluates JEPA as a replacement for missing RGB-derived matching "
        "features, not a complete replacement of DPVO image processing.",
        "",
        "Replacement occurs exactly at `Patchifier.fnet` output. Non-keyframes bypass "
        "the original FNet and return scale-equivalent projected JEPA with the same "
        "`[B,N,128,H,W]` interface. Frozen INet/context and color sampling remain unchanged "
        "and therefore still consume RGB. The reported matching/JEPA replacement ratios "
        "describe only the source of the FNet matching tensor.",
        "",
        "## Sequence results",
        "",
        "| Sequence | Full-RGB ATE | Oracle ATE | Oracle tracking | Pose coverage | Matching RGB ratio | JEPA replaced ratio | Baseline E2E s | Oracle E2E s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "MH_01 is a training diagnostic and MH_03 is a transfer diagnostic. MH_05 is the "
        "only decision sequence. Coverage is reported but is not an independent failure "
        "criterion.",
        "",
        "## Reproducibility",
        "",
        f"Baseline and Oracle share contract `{metrics['reproducibility']['contract_sha256']}`: "
        "the same Git HEAD, source snapshot, DPVO checkpoint, dataset sampling, calibration, "
        "configuration and random seed. The only changed matching variable is the FNet "
        "feature source.",
        "",
        "## MH_05 trajectory evidence",
        "",
        "Final raw baseline, Oracle and GT trajectories are retained. Each estimate is "
        "timestamp-associated and Sim(3)-aligned only in memory for ATE verification and "
        "the 2D XY figure.",
        "",
        f"MH_05 ATE degradation versus fresh full-RGB baseline is "
        f"`{decision['ate_degradation_vs_baseline']:.2%}`.",
        "",
        f"Decision: **{decision['outcome']}**. {decision['reason']}.",
        "",
        "Runtime is diagnostic and does not affect the decision.",
        "",
    ]
    return "\n".join(lines)


def _write_oracle_outputs(output_dir: Path, metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(
        render_oracle_report(metrics), encoding="utf-8"
    )


def validate_oracle_inventory(output_dir: Path) -> None:
    if {path.name for path in output_dir.iterdir()} != {
        "metrics.json", "REPORT.md", "trajectories", "figures"
    }:
        raise AssertionError("Exp5-Oracle root inventory changed")
    trajectories = {
        path.name for path in (output_dir / "trajectories").iterdir() if path.is_file()
    }
    if trajectories != {"MH_05_baseline.json", "MH_05_oracle.json", "MH_05_gt.json"}:
        raise AssertionError("Exp5-Oracle trajectory inventory changed")
    figures = {
        path.name for path in (output_dir / "figures").iterdir() if path.is_file()
    }
    if figures != {"MH_05_oracle_missing_rgb.png"}:
        raise AssertionError("Exp5-Oracle figure inventory changed")


def build_oracle_trajectory_evidence(
    *, staging: Path, temporary_trajectories: Path,
    evaluations: dict[str, dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    sequence = TEST_SEQUENCE
    baseline = load_trajectory(temporary_trajectories / f"{sequence}_baseline.json")
    oracle = load_trajectory(temporary_trajectories / f"{sequence}_oracle.json")
    timestamps = [*baseline["timestamps"], *oracle["timestamps"]]
    groundtruth_source = REPO_ROOT / str(
        config["evaluation"]["groundtruth_pattern"]
    ).format(sequence=sequence)
    groundtruth = build_ground_truth_subset(
        sequence=sequence,
        source_path=groundtruth_source,
        timestamp_min=min(int(value) for value in timestamps),
        timestamp_max=max(int(value) for value in timestamps),
    )
    paths = {
        "baseline": Path("trajectories/MH_05_baseline.json"),
        "oracle": Path("trajectories/MH_05_oracle.json"),
        "ground_truth": Path("trajectories/MH_05_gt.json"),
    }
    write_trajectory(staging / paths["baseline"], baseline)
    write_trajectory(staging / paths["oracle"], oracle)
    write_trajectory(staging / paths["ground_truth"], groundtruth)
    verification = {}
    for alias, payload, method in (
        ("baseline", baseline, "dpvo_baseline"),
        ("oracle", oracle, "oracle_jepa_missing_rgb"),
    ):
        recomputed = evaluate_trajectory(payload, groundtruth)
        runtime_ate = float(
            evaluations[sequence][method]["ate"]["translation_rmse"]
        )
        exported_ate = float(recomputed["ate_rmse"])
        difference = abs(exported_ate - runtime_ate)
        verification[alias] = {
            "runtime_ate_rmse": runtime_ate,
            "exported_ate_rmse": exported_ate,
            "associated_count": int(recomputed["associated_count"]),
            "alignment": "Sim(3)",
            "absolute_difference": difference,
            "relative_difference": difference / runtime_ate if runtime_ate > 0.0 else None,
            "diagnostic_only": True,
        }
    figure = Path("figures/MH_05_oracle_missing_rgb.png")
    plot_oracle_xy_trajectories(
        sequence=sequence,
        baseline=baseline,
        oracle=oracle,
        ground_truth=groundtruth,
        output_path=staging / figure,
    )
    return {
        "sequence": sequence,
        "role": "decision",
        "trajectory_artifacts": {key: value.as_posix() for key, value in paths.items()},
        "figure_path": figure.as_posix(),
        "ate_verification": verification,
        "diagnostic_only": True,
        "completeness": True,
    }


def _write_mock_oracle_trajectories(
    *, destination: Path, config: dict[str, Any], frame_count: int,
) -> None:
    from evo.tools import file_interface

    sequence = TEST_SEQUENCE
    source = REPO_ROOT / str(config["evaluation"]["groundtruth_pattern"]).format(
        sequence=sequence
    )
    reference = file_interface.read_tum_trajectory_file(str(source))
    timestamps = np.rint(reference.timestamps[:frame_count]).astype(np.int64)
    records = [
        {"stream_index": index, "frame_id": index, "timestamp": int(timestamp)}
        for index, timestamp in enumerate(timestamps)
    ]
    index = np.arange(frame_count, dtype=np.float64)
    base = np.stack(
        (0.15 * index, np.sin(index * 0.45), 0.1 * np.cos(index * 0.25)), axis=1
    )
    quaternion = np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]]), (frame_count, 1))
    for method, filename, offset in (
        ("dpvo_baseline", f"{sequence}_baseline.json", 0.0),
        ("oracle_jepa_missing_rgb", f"{sequence}_oracle.json", 0.01),
    ):
        positions = base.copy()
        positions[:, 1] += offset * np.cos(index)
        payload = build_estimated_trajectory(
            sequence=sequence,
            method=method,
            records=records,
            ordinals=np.arange(frame_count, dtype=np.float64),
            poses=np.concatenate((positions, quaternion), axis=1),
        )
        write_trajectory(destination / filename, payload)


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    sequence = str(config["dataset"]["smoke_sequence"])
    records = load_sequence_records(
        config, sequence, limit=int(config["dataset"]["smoke_frames"])
    )
    image_shapes = validate_real_images(records)
    record_dicts = [record.to_dict() for record in records]
    availability_frames, availability = frame_availability(
        record_dicts, keyframe_stride=int(config["oracle"]["keyframe_stride"])
    )
    count = len(records)
    generator = np.random.default_rng(int(config["experiment"]["seed"]))
    source_tokens = generator.standard_normal((count, 576, 768), dtype=np.float32)
    mock_worker = MockDenseJepaWorkerClient({
        int(record.frame_id): source_tokens[index]
        for index, record in enumerate(records)
    })
    tokens = []
    for record in records:
        dense, _, _ = mock_worker.extract(record.to_dict())
        tokens.append(dense)
    mock_worker.close()
    worker_lifecycle = mock_worker.lifecycle()
    model = DenseJepaProjection()
    with torch.inference_mode():
        projected = model(
            torch.from_numpy(np.stack(tokens)), target_size=(120, 188)
        )

    class MockFNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.original_calls = 0

        def forward(self, _image: torch.Tensor) -> torch.Tensor:
            self.original_calls += 1
            return torch.ones(1, 1, 128, 120, 188)

    fnet = MockFNet()
    hook = OracleFNetReplacementHook(
        alpha=0.1, channels=128, projection_to_raw_scale=4.0,
        replacement_dtype=torch.float32,
    )
    hook.install(fnet)
    outputs = []
    for index, record in enumerate(records):
        is_keyframe = bool(availability_frames[index]["is_keyframe"])
        hook.bind(record, projected[index:index + 1], is_keyframe=is_keyframe)
        outputs.append(fnet(torch.zeros(1)))
        hook.finish_frame(record, is_keyframe=is_keyframe)
    hook.close()
    lifecycle = hook.lifecycle()
    if fnet.original_calls != 2 or lifecycle["jepa_replaced_frames"] != 6:
        raise AssertionError("Oracle smoke did not execute the 2/6 FNet boundary split")
    if not torch.allclose(
        outputs[0] / 4.0, torch.ones_like(outputs[0]) / 4.0 + 0.1 * projected[0:1, None]
    ):
        raise AssertionError("Oracle keyframe residual scale contract failed")
    if not torch.allclose(outputs[1] / 4.0, projected[1:2, None]):
        raise AssertionError("Oracle non-keyframe replacement scale contract failed")
    availability["hook_counts_valid"] = True
    evaluations = mock_oracle_evaluations(count)
    evaluations[sequence]["oracle_jepa_missing_rgb"]["availability"] = availability
    source_reference = {
        "metrics_path": "mock", "metrics_sha256": "0" * 64,
        "projection_path": "mock", "projection_sha256": "1" * 64,
        "projection_retrained": False,
    }
    reproducibility = {
        "git_head": "mock",
        "source_sha256": "2" * 64,
        "dpvo_upstream_clean": True,
        "dpvo_checkpoint": {"path": "mock", "sha256": "3" * 64},
        "dpvo_config": {"path": "mock", "sha256": "4" * 64},
        "calibration": {"path": "mock", "sha256": "5" * 64},
        "dataset_config_sha256": "6" * 64,
        "sequence_sampling_sha256": {
            sequence_name: "7" * 64 for sequence_name in config["evaluation"]["sequences"]
        },
        "random_seed": int(config["experiment"]["seed"]),
        "projection": {"path": "mock", "sha256": "1" * 64},
        "exp5_2_metrics": {"path": "mock", "sha256": "0" * 64},
        "contract_sha256": "c" * 64,
    }
    with tempfile.TemporaryDirectory(prefix="exp5-oracle-smoke-") as temporary_name:
        temporary = Path(temporary_name)
        trajectories = temporary / "runtime_trajectories"
        trajectories.mkdir()
        _write_mock_oracle_trajectories(
            destination=trajectories, config=config, frame_count=count
        )
        staging = temporary / "oracle_missing_rgb"
        staging.mkdir()
        evidence = build_oracle_trajectory_evidence(
            staging=staging,
            temporary_trajectories=trajectories,
            evaluations=evaluations,
            config=config,
        )
        metrics = build_oracle_metrics(
            config=config,
            evaluations=evaluations,
            reproducibility=reproducibility,
            source_reference=source_reference,
            trajectory_evidence=evidence,
            smoke=True,
        )
        _write_oracle_outputs(staging, metrics)
        validate_oracle_inventory(staging)
    return {
        "status": "smoke_pass",
        "sequence": sequence,
        "num_frames": count,
        "decoded_image_shapes": sorted({tuple(shape) for shape in image_shapes}),
        "keyframe_count": availability["keyframe_count"],
        "fnet_executed_frames": lifecycle["fnet_executed_frames"],
        "jepa_replaced_frames": lifecycle["jepa_replaced_frames"],
        "matching_rgb_ratio": availability["matching_rgb_ratio"],
        "jepa_replaced_ratio": availability["jepa_replaced_ratio"],
        "worker_request_count": worker_lifecycle["request_count"],
        "hook_cleanup_passed": lifecycle["cleanup_passed"],
        "trajectory_json_count": 3,
        "trajectory_png_count": 1,
        "decision": metrics["decision"]["outcome"],
        "models": "mock",
        "projection_trained": False,
        "artifacts_retained": False,
    }


def run_oracle_formal(config: dict[str, Any]) -> dict[str, Any]:
    result_root = Path(config["experiment"]["result_root"])
    source_metrics_path = result_root / "metrics.json"
    projection_path = result_root / "projection.pt"
    if not source_metrics_path.is_file() or not projection_path.is_file():
        raise RuntimeError(
            "internal Exp5 pipeline error: the preceding Exp5-2 stage did not publish "
            "metrics.json and projection.pt"
        )
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    if source_metrics.get("schema_version") != SCHEMA_VERSION or source_metrics.get("mode") != "formal":
        raise ValueError("Exp5-Oracle requires formal Exp5-2 schema-v4 results")
    environment = _fusion_formal_preflight(config)
    import torch

    checkpoint_model, checkpoint_metadata = load_dense_projection_checkpoint(
        projection_path, device=torch.device("cpu")
    )
    del checkpoint_model
    if checkpoint_metadata.get("projection_kind") != "dense_tokens":
        raise ValueError("Exp5-Oracle requires the Exp5-2 dense projection")
    records_by_sequence = {
        sequence: [record.to_dict() for record in load_sequence_records(config, sequence)]
        for sequence in config["evaluation"]["sequences"]
    }
    reproducibility = build_reproducibility_contract(
        config=config,
        records_by_sequence=records_by_sequence,
        projection_path=projection_path,
        source_metrics_path=source_metrics_path,
    )
    source_reference = {
        "metrics_path": str(source_metrics_path.resolve()),
        "metrics_sha256": _sha256(source_metrics_path),
        "projection_path": str(projection_path.resolve()),
        "projection_sha256": _sha256(projection_path),
        "projection_checkpoint_schema": checkpoint_metadata["checkpoint_schema_version"],
        "projection_retrained": False,
        "existing_exp5_2_status": source_metrics.get("status"),
    }
    target = result_root / str(config["oracle"]["output_subdirectory"])
    with tempfile.TemporaryDirectory(
        prefix="exp5-oracle-", dir=result_root
    ) as temporary_name:
        work_dir = Path(temporary_name)
        staging = work_dir / "publish"
        staging.mkdir()
        temporary_trajectories = work_dir / "runtime_trajectories"
        temporary_trajectories.mkdir()
        evaluations: dict[str, dict[str, Any]] = {}
        for sequence in config["evaluation"]["sequences"]:
            evaluations[sequence] = {}
            for method in ORACLE_METHODS:
                alias = "baseline" if method == "dpvo_baseline" else "oracle"
                row = evaluate_dense_method(
                    config=config,
                    records=records_by_sequence[sequence],
                    method=method,
                    checkpoint_path=projection_path,
                    work_dir=work_dir,
                    trajectory_path=(
                        temporary_trajectories / f"{sequence}_{alias}.json"
                    ),
                )
                row["reproducibility_contract_sha256"] = reproducibility["contract_sha256"]
                evaluations[sequence][method] = row
            if {
                evaluations[sequence][method]["reproducibility_contract_sha256"]
                for method in ORACLE_METHODS
            } != {reproducibility["contract_sha256"]}:
                raise RuntimeError(f"baseline/Oracle reproducibility mismatch: {sequence}")
        evidence = build_oracle_trajectory_evidence(
            staging=staging,
            temporary_trajectories=temporary_trajectories,
            evaluations=evaluations,
            config=config,
        )
        source_reference["environment_bridge"] = environment
        metrics = build_oracle_metrics(
            config=config,
            evaluations=evaluations,
            reproducibility=reproducibility,
            source_reference=source_reference,
            trajectory_evidence=evidence,
            smoke=False,
        )
        _write_oracle_outputs(staging, metrics)
        validate_oracle_inventory(staging)
        publish_replacing(staging, target, work_dir)
    return metrics


def run_formal(config: dict[str, Any]) -> dict[str, Any]:
    """Rebuild Exp5-2 from scratch, then run Oracle from that exact result."""
    dense_metrics = run_dense_formal(config)
    oracle_metrics = run_oracle_formal(config)
    if dense_metrics.get("mode") != "formal" or oracle_metrics.get("mode") != "formal":
        raise RuntimeError("from-scratch Exp5 pipeline did not complete both formal stages")
    return {
        "status": oracle_metrics["status"],
        "pipeline": "from_scratch_exp5_2_then_exp5_oracle",
        "exp5_2_status": dense_metrics["status"],
        "oracle_status": oracle_metrics["status"],
        "result_root": str(Path(config["experiment"]["result_root"])),
        "old_results_reused": False,
        "projection_retrained_as_exp5_2_prerequisite": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    result = run_smoke(config) if args.smoke else run_formal(config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
