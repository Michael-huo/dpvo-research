"""Aggregate the fixed Experiment 2 correspondence smoke without inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "research/configs/phase1_exp2.yaml"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else None


def _median(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else None


def summarize(data: dict[str, np.ndarray], selection: np.ndarray, prefix: str = "") -> dict[str, Any]:
    valid = selection & data[f"{prefix}valid"]
    count = int(valid.sum())
    return {
        "sample_count": int(selection.sum()),
        "valid_sample_count": count,
        "top1_similarity_mean": _mean(data[f"{prefix}top1_similarity"][valid]),
        "top1_similarity_median": _median(data[f"{prefix}top1_similarity"][valid]),
        "peak_margin_mean": _mean(data[f"{prefix}peak_margin"][valid]),
        "peak_margin_median": _median(data[f"{prefix}peak_margin"][valid]),
        "epipolar_error_tokens_median": _median(data[f"{prefix}epipolar_error_tokens"][valid]),
        "cycle_error_tokens_median": _median(data[f"{prefix}cycle_error_tokens"][valid]),
        "cycle_success_rate": _mean(data[f"{prefix}cycle_success"][valid].astype(np.float64)),
        "geometry_consistent_rate": _mean(data[f"{prefix}jepa_geometry_consistent"][valid].astype(np.float64)),
    }


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _markdown_table(group_summaries: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Group | Valid | Top1 mean/median | Peak margin mean/median | Epipolar median | Cycle median | Cycle success | Geometry consistent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in group_summaries.items():
        lines.append(
            f"| {name} | {values['valid_sample_count']} | "
            f"{_format(values['top1_similarity_mean'])} / {_format(values['top1_similarity_median'])} | "
            f"{_format(values['peak_margin_mean'])} / {_format(values['peak_margin_median'])} | "
            f"{_format(values['epipolar_error_tokens_median'])} | "
            f"{_format(values['cycle_error_tokens_median'])} | "
            f"{_format(values['cycle_success_rate'])} | {_format(values['geometry_consistent_rate'])} |"
        )
    return lines


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    output_root = REPO_ROOT / config["paths"]["output_root"]
    artifacts = output_root / "artifacts"
    smoke = output_root / "smoke"
    smoke.mkdir(parents=True, exist_ok=True)

    prepare_manifest = _load_json(artifacts / "manifest.json")
    extraction_manifest = _load_json(artifacts / "extraction_manifest.json")
    metrics = _load_npz(artifacts / "jepa_metrics.npz")
    if len(metrics["sample_id"]) != 200 or len(metrics["shuffled_sample_id"]) != 40:
        raise AssertionError("Expected 200 correct and 40 temporal-shuffled records")

    groups: dict[str, dict[str, Any]] = {}
    group_order = [
        ("MH01 good", "MH_01_easy", "good"),
        ("MH01 bad — MH_01 Q20 frozen-threshold bad group", "MH_01_easy", "bad"),
        ("MH05 good", "MH_05_difficult", "good"),
        ("MH05 bad — MH_01 Q20 frozen-threshold bad group", "MH_05_difficult", "bad"),
    ]
    for label, sequence, group in group_order:
        selected = (metrics["sequence"] == sequence) & (metrics["group"] == group)
        groups[label] = summarize(metrics, selected)
        if groups[label]["sample_count"] != 50:
            raise AssertionError(f"{label} has {groups[label]['sample_count']} samples")

    shuffled_count = len(metrics["shuffled_sample_index"])
    shuffled_selection = np.ones(shuffled_count, dtype=bool)
    shuffled_summary = summarize(metrics, shuffled_selection, prefix="shuffled_")
    matched_correct_selection = np.zeros(len(metrics["sample_id"]), dtype=bool)
    matched_correct_selection[metrics["shuffled_sample_index"].astype(np.int64)] = True
    matched_correct_summary = summarize(metrics, matched_correct_selection)
    null_sanity = {
        "scope": "40 deterministically selected paired samples; descriptive temporal-shuffled null sanity only",
        "strict_negative_ground_truth": False,
        "significance_test": False,
        "correct_pairs": matched_correct_summary,
        "temporal_shuffled_pairs": shuffled_summary,
        "geometry_consistent_rate_difference_correct_minus_shuffled": (
            matched_correct_summary["geometry_consistent_rate"] - shuffled_summary["geometry_consistent_rate"]
            if matched_correct_summary["geometry_consistent_rate"] is not None and shuffled_summary["geometry_consistent_rate"] is not None else None
        ),
    }

    all_correct = summarize(metrics, np.ones(len(metrics["sample_id"]), dtype=bool))
    coordinate_review = extraction_manifest["coordinate_sanity"].get("manual_review", "pending")
    summary = {
        "schema_version": 1,
        "status": "completed_smoke_no_parameter_tuning",
        "scope": "Phase 1 / Experiment 2 fixed 200-sample measurement-pipeline smoke",
        "thresholds": prepare_manifest["thresholds"],
        "tie_aware_sampling_population": prepare_manifest["sampling"]["sample_counts"],
        "groups": groups,
        "all_correct_pairs": all_correct,
        "correct_vs_temporal_shuffled": null_sanity,
        "coordinate_sanity": {
            **extraction_manifest["coordinate_sanity"],
            "manual_review": coordinate_review,
        },
        "geometry_preflight": extraction_manifest["geometry"]["synthetic_consistency_preflight"],
        "vjepa": extraction_manifest["vjepa"],
        "environment": extraction_manifest["environment"],
        "interpretation": {
            "jepa_geometry_consistent": "epipolar_error_tokens <= 1.0 AND cycle_error_tokens <= 1.0",
            "meaning": "Feasibility operational criterion for a geometrically and cycle-consistent JEPA candidate; it does not claim recovery of true correspondence.",
            "shuffled": "Descriptive null sanity, not strict negative ground truth.",
            "parameter_tuning_after_smoke": False,
        },
        "artifact_references": {
            "full_exp1_schema": "../artifacts/manifest.json#exp1_schema",
            "sample_contract": "../artifacts/samples.jsonl and ../artifacts/samples.npz",
            "per_sample_metrics": "../artifacts/jepa_metrics.npz",
            "coordinate_figure": "../debug/coordinate_sanity.png",
        },
    }
    _write_json(smoke / "summary.json", summary)

    schemas = prepare_manifest["exp1_schema"]
    lines = [
        "# Phase 1 / Experiment 2 — 200-sample JEPA correspondence smoke",
        "",
        "Status: completed fixed smoke; no parameter tuning; formal 4000-sample experiment was not run.",
        "",
        "## Source artifacts and frozen groups",
        "",
        f"- MH_01 `corr_margin_l0` Q20: `{prepare_manifest['thresholds']['mh01_q20']}`; Q80: `{prepare_manifest['thresholds']['mh01_q80']}`.",
        "- Bad is `corr_margin_l0 <= Q20` and is named **MH_01 Q20 frozen-threshold bad group**. The Q20=0 tie is preserved; it is not described as a strict bottom 20% group.",
        "- Good is `corr_margin_l0 >= Q80`. Weight is metadata only and was not used for selection.",
        "- Exp 1 observations represent all factors from the final online graph update per input frame; no additional factor-level deduplication was applied.",
        "- Full NPZ key/shape/dtype inventory is recorded in `artifacts/manifest.json` under `exp1_schema`.",
        "",
        "Relevant actual fields:",
        "",
        f"- observations ({len(schemas['MH_01_easy']['observations'])} keys): factor/source patch UID, source/target/update DPVO counters, weight mean, delta norm, apparent motion, L0/L1 correlation peak/margin/entropy and related statistics.",
        f"- patches ({len(schemas['MH_01_easy']['patches'])} keys): patch UID, source counter, EuRoC timestamp, 1/4-grid x/y, inverse depth and patch behavior summaries/tags.",
        f"- frames ({len(schemas['MH_01_easy']['frames'])} keys): DPVO counter, EuRoC timestamp, image proxies, frame aggregates, trajectory diagnostics and difficulty quintiles.",
        f"- windows ({len(schemas['MH_01_easy']['windows'])} keys): window trajectory and behavior summaries.",
        "- Exp 1 contains `delta_norm`, not a recoverable delta vector; the sample contract reports only `dpvo_delta_norm`.",
        "",
        "### Tie-aware population accounting",
        "",
        "| Sequence/group | Before threshold | Threshold candidates | Eligible after GT/crop validity | Final |",
        "|---|---:|---:|---:|---:|",
    ]
    for sequence in config["experiment"]["sequences"]:
        for group in ("good", "bad"):
            values = prepare_manifest["sampling"]["sample_counts"][sequence][group]
            lines.append(
                f"| {sequence} {group} | {values['observation_count_before_threshold']} | "
                f"{values['threshold_candidate_count']} | {values['eligible_after_gt_crop_validity_count']} | "
                f"{values['final_sampled_count']} |"
            )

    lines.extend([
        "",
        "## Coordinate and geometry contract",
        "",
        "- DPVO patch x/y are coordinates on the undistorted 1/4-resolution feature grid; full-resolution undistorted pixels are `(4x, 4y)`.",
        "- Those points are distorted into the raw 752×480 EuRoC PNG, then mapped through the provider's 438×686 resize and `(151,27,535,411)` 384 crop to a continuous 24×24 token grid.",
        "- JEPA predictions are inverse-mapped and undistorted before epipolar measurement. Pixel error is divided by local one-token spacing.",
        "- Camera pose uses `T_WC = T_WB @ T_BC`; source-to-target relative pose uses `T_Ct_Cs = inv(T_WC_t) @ T_WC_s`.",
        f"- Synthetic ray/project/reproject epipolar error: `{_format(summary['geometry_preflight']['point_line_error_pixels'], 12)}` pixels (passed).",
        f"- Coordinate sanity: 20 panels, automated round-trip/provider checks passed; manual review: `{coordinate_review}`.",
        "",
        "## Correct-pair metrics",
        "",
        *_markdown_table(groups),
        "",
        "`geometry_consistent_rate` uses `epipolar_error_tokens <= 1.0 AND cycle_error_tokens <= 1.0`. It denotes a geometrically and cycle-consistent JEPA candidate only; it does not claim true correspondence recovery.",
        "",
        "## Correct vs temporal-shuffled sanity",
        "",
        "| Pair type (same 40 sources) | Top1 mean/median | Peak margin mean/median | Epipolar median | Cycle median | Cycle success | Geometry consistent |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for label, values in (("Correct", matched_correct_summary), ("Temporal shuffled", shuffled_summary)):
        lines.append(
            f"| {label} | {_format(values['top1_similarity_mean'])} / {_format(values['top1_similarity_median'])} | "
            f"{_format(values['peak_margin_mean'])} / {_format(values['peak_margin_median'])} | "
            f"{_format(values['epipolar_error_tokens_median'])} | {_format(values['cycle_error_tokens_median'])} | "
            f"{_format(values['cycle_success_rate'])} | {_format(values['geometry_consistent_rate'])} |"
        )
    lines.extend([
        "",
        "This is a descriptive temporal-shuffled null sanity, not strict negative ground truth; no significance test was performed.",
        "",
        "## Fixed V-JEPA provider",
        "",
        f"- Commit: `{extraction_manifest['vjepa']['git_commit']}`",
        f"- Python: `{extraction_manifest['environment']['python_executable']}`",
        f"- PyTorch/CUDA/GPU: `{extraction_manifest['environment']['torch_version']}` / `{extraction_manifest['environment']['torch_cuda_version']}` / `{extraction_manifest['environment']['gpu']}`",
        f"- Model: `{extraction_manifest['vjepa']['model_name']}` (`{extraction_manifest['vjepa']['model_alias']}`)",
        f"- Checkpoint: `{extraction_manifest['vjepa']['checkpoint']}`",
        f"- Representation: `{extraction_manifest['vjepa']['representation']}`; grid `{extraction_manifest['vjepa']['token_grid_hw']}`, patch size `{extraction_manifest['vjepa']['patch_size']}`, dim `{extraction_manifest['vjepa']['feature_dim']}`.",
        "",
        "## Smoke decision boundary",
        "",
        "The pipeline reports the fixed measurements as observed. No threshold, layer, preprocessing, search region, sample group, or one-token criterion was changed. This smoke does not make a positive/negative Experiment 2 claim and does not authorize the formal 4000-sample run.",
        "",
    ])
    (smoke / "SMOKE.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"groups": groups, "correct_vs_temporal_shuffled": null_sanity}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
