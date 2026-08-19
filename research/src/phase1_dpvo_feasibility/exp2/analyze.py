"""Analyze the frozen formal 4000-sample Experiment 2 measurement set."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phase1-exp2")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "research/configs/phase1_exp2.yaml"
PROFILE_NAME = "formal"
GATE_START = "<!-- DECISION_GATE_START -->"
GATE_END = "<!-- DECISION_GATE_END -->"
GROUPS = [
    ("MH01 good", "MH_01_easy", "good"),
    ("MH01 bad", "MH_01_easy", "bad"),
    ("MH05 good", "MH_05_difficult", "good"),
    ("MH05 bad", "MH_05_difficult", "bad"),
]
GAP_BINS = [
    ("A", "|delta_t| <= 2s", lambda value: value <= 2.0),
    ("B", "2s < |delta_t| <= 5s", lambda value: (value > 2.0) & (value <= 5.0)),
    ("C", "|delta_t| > 5s", lambda value: value > 5.0),
]


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else None


def _median(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else None


def _rate(values: np.ndarray) -> float | None:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if len(values) else None


def _summary(metrics: dict[str, np.ndarray], selected: np.ndarray, prefix: str = "") -> dict[str, Any]:
    valid = selected & metrics[f"{prefix}valid"]
    return {
        "total_n": int(selected.sum()),
        "valid_n": int(valid.sum()),
        "invalid_n": int(selected.sum() - valid.sum()),
        "top1_similarity_mean": _mean(metrics[f"{prefix}top1_similarity"][valid]),
        "top1_similarity_median": _median(metrics[f"{prefix}top1_similarity"][valid]),
        "peak_margin_mean": _mean(metrics[f"{prefix}peak_margin"][valid]),
        "peak_margin_median": _median(metrics[f"{prefix}peak_margin"][valid]),
        "epipolar_error_tokens_mean": _mean(metrics[f"{prefix}epipolar_error_tokens"][valid]),
        "epipolar_error_tokens_median": _median(metrics[f"{prefix}epipolar_error_tokens"][valid]),
        "cycle_error_tokens_median": _median(metrics[f"{prefix}cycle_error_tokens"][valid]),
        "cycle_success_rate": _rate(metrics[f"{prefix}cycle_success"][valid]),
        "geometry_consistent_rate": _rate(metrics[f"{prefix}jepa_geometry_consistent"][valid]),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _direction(lower: float | None, upper: float | None, expected: str) -> bool | None:
    if lower is None or upper is None:
        return None
    return bool(lower < upper) if expected == "lower" else bool(lower > upper)


def _create_figures(
    report_dir: Path, metrics: dict[str, np.ndarray], samples: dict[str, np.ndarray],
    group_summaries: dict[str, dict[str, Any]], paired: dict[str, Any], joint_indices: np.ndarray,
) -> list[str]:
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = ["#4477AA", "#CC6677", "#228833", "#AA3377"]
    labels = [entry[0] for entry in GROUPS]
    selections = [
        (metrics["sequence"] == sequence) & (metrics["group"] == group) & metrics["valid"]
        for _, sequence, group in GROUPS
    ]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    axes[0].boxplot([metrics["peak_margin"][mask] for mask in selections], tick_labels=labels, showfliers=False)
    axes[0].set_title("Peak-margin distribution")
    axes[0].set_ylabel("peak_margin")
    axes[1].boxplot([metrics["epipolar_error_tokens"][mask] for mask in selections], tick_labels=labels, showfliers=False)
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_title("Epipolar-error distribution")
    axes[1].set_ylabel("epipolar_error_tokens (symlog)")
    cycle = [group_summaries[label]["cycle_success_rate"] for label in labels]
    axes[2].bar(labels, cycle, color=colors)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Cycle success")
    axes[2].set_ylabel("rate (<= 1 token)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    path1 = figures / "figure_1_good_bad.png"
    figure.savefig(path1, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    geometry = [group_summaries[label]["geometry_consistent_rate"] for label in labels]
    axes[0].bar(labels, geometry, color=colors)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Geometry-consistent candidates")
    axes[0].set_ylabel("rate (epipolar <= 1 and cycle <= 1 token)")
    axes[0].tick_params(axis="x", rotation=25)
    pair_values = [
        paired["joint_valid_comparison"]["correct"]["geometry_consistent_rate"],
        paired["joint_valid_comparison"]["shuffled"]["geometry_consistent_rate"],
    ]
    axes[1].bar(["Correct", "Temporal shuffled"], pair_values, color=["#4477AA", "#BBBBBB"])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title(f"Paired null sanity (joint-valid N={len(joint_indices)})")
    axes[1].set_ylabel("geometry-consistent rate")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    path2 = figures / "figure_2_geometry_consistency.png"
    figure.savefig(path2, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, (sequence, title) in zip(axes, (("MH_01_easy", "MH01"), ("MH_05_difficult", "MH05"))):
        sequence_mask = (metrics["sequence"] == sequence) & metrics["valid"]
        for success, marker, color, label in (
            (False, "x", "#999999", "cycle failure"),
            (True, "o", "#117733", "cycle success"),
        ):
            mask = sequence_mask & (metrics["cycle_success"] == success)
            axis.scatter(
                samples["dpvo_corr_margin"][mask], metrics["epipolar_error_tokens"][mask],
                s=10, alpha=0.45, marker=marker, c=color, label=label, linewidths=0.5,
            )
        axis.axhline(1.0, color="#CC3311", linestyle="--", linewidth=1.0, label="1-token criterion")
        axis.set_yscale("symlog", linthresh=1.0)
        axis.set_xlabel("DPVO corr_margin_l0")
        axis.set_ylabel("JEPA epipolar_error_tokens (symlog)")
        axis.set_title(f"{title} complementarity map")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    path3 = figures / "figure_3_complementarity.png"
    figure.savefig(path3, dpi=160)
    plt.close(figure)

    paths = [path1, path2, path3]
    if sorted(path.name for path in figures.glob("*.png")) != sorted(path.name for path in paths):
        raise AssertionError("Formal report must contain exactly the three frozen figures")
    return [str(path.relative_to(report_dir)) for path in paths]


def _gate_block(gate: str, rationale: str) -> str:
    return "\n".join([
        GATE_START,
        "## Decision Gate",
        "",
        f"**Experiment 2 = {gate}**",
        "",
        rationale,
        GATE_END,
    ])


def update_gate(report_dir: Path, gate: str, rationale: str) -> None:
    """Update only the report-level decision fields after human evidence review."""
    gate = gate.upper()
    if gate not in {"POSITIVE", "NEGATIVE", "AMBIGUOUS"}:
        raise ValueError(f"Invalid Experiment 2 gate: {gate}")
    summary_path = report_dir / "summary.json"
    report_path = report_dir / "REPORT.md"
    summary = _read_json(summary_path)
    summary["status"] = "formal_fixed_evidence_complete_gate_assigned"
    summary["decision_gate"] = {
        "status": gate,
        "method": "one-time qualitative research judgment after fixed evidence generation; no numerical decision threshold",
        "rationale": rationale,
        "shuffled_role": "descriptive null sanity only; not a necessary condition for NEGATIVE",
    }
    _write_json(summary_path, summary)
    report = report_path.read_text(encoding="utf-8")
    start = report.index(GATE_START)
    end = report.index(GATE_END) + len(GATE_END)
    report_path.write_text(report[:start] + _gate_block(gate, rationale) + report[end:], encoding="utf-8")


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    formal_root = REPO_ROOT / config["paths"]["output_root"] / PROFILE_NAME
    artifacts = formal_root / "artifacts"
    report_dir = formal_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(artifacts / "samples.jsonl")
    samples = _load_npz(artifacts / "samples.npz")
    metrics = _load_npz(artifacts / "jepa_metrics.npz")
    prepare_manifest = _read_json(artifacts / "manifest.json")
    extraction_manifest = _read_json(artifacts / "extraction_manifest.json")

    expected = int(config["sampling_profiles"][PROFILE_NAME]["samples_per_group"]) * 4
    expected_shuffled = int(config["sampling_profiles"][PROFILE_NAME]["shuffled_per_group"]) * 4
    if len(rows) != expected or len(metrics["sample_id"]) != expected:
        raise AssertionError("Formal correct-pair count mismatch")
    if len(metrics["shuffled_sample_id"]) != expected_shuffled:
        raise AssertionError("Formal shuffled-pair count mismatch")
    if not np.array_equal(metrics["sample_index"], np.arange(expected)):
        raise AssertionError("Correct metrics are not aligned to formal sample rows")
    if not np.array_equal(samples["sample_id"], metrics["sample_id"]):
        raise AssertionError("Sample NPZ and metric sample identities differ")

    group_summaries: dict[str, dict[str, Any]] = {}
    for label, sequence, group in GROUPS:
        selection = (metrics["sequence"] == sequence) & (metrics["group"] == group)
        group_summaries[label] = _summary(metrics, selection)
        group_summaries[label]["delta_t_seconds_signed"] = prepare_manifest["sampling"]["sample_counts"][sequence][group]["delta_t_seconds_signed"]
        if group_summaries[label]["total_n"] != int(config["sampling_profiles"][PROFILE_NAME]["samples_per_group"]):
            raise AssertionError(f"Unexpected count for {label}")

    temporal: list[dict[str, Any]] = []
    absolute_delta = np.abs(samples["delta_t"].astype(np.float64))
    bin_membership = np.zeros(expected, dtype=np.int64)
    for sequence in config["experiment"]["sequences"]:
        for group in ("good", "bad"):
            base = (metrics["sequence"] == sequence) & (metrics["group"] == group)
            for bin_name, definition, predicate in GAP_BINS:
                selection = base & predicate(absolute_delta)
                bin_membership += selection.astype(np.int64)
                values = _summary(metrics, selection)
                temporal.append({
                    "sequence": sequence, "group": group, "gap_bin": bin_name,
                    "gap_definition": definition, **values,
                })
    if not np.all(bin_membership == 1):
        raise AssertionError("Temporal-gap bins do not form an exact partition")

    shuffled_indices = metrics["shuffled_sample_index"].astype(np.int64)
    if len(np.unique(shuffled_indices)) != len(shuffled_indices):
        raise AssertionError("Shuffled controls are not uniquely paired")
    correct_valid = metrics["valid"][shuffled_indices]
    shuffled_valid = metrics["shuffled_valid"]
    joint = correct_valid & shuffled_valid
    joint_indices = shuffled_indices[joint]
    correct_joint_selection = np.zeros(expected, dtype=bool)
    correct_joint_selection[joint_indices] = True
    shuffled_joint_selection = joint.copy()
    paired = {
        "role": "descriptive temporal-shuffled null sanity only; not strict negative GT and not a necessary NEGATIVE criterion",
        "significance_test": False,
        "total_paired_n": int(len(shuffled_indices)),
        "correct_valid_n": int(correct_valid.sum()),
        "shuffled_valid_n": int(shuffled_valid.sum()),
        "joint_valid_paired_n": int(joint.sum()),
        "joint_valid_comparison": {
            "correct": _summary(metrics, correct_joint_selection),
            "shuffled": _summary(metrics, shuffled_joint_selection, prefix="shuffled_"),
        },
    }
    if paired["joint_valid_comparison"]["correct"]["valid_n"] != paired["joint_valid_paired_n"]:
        raise AssertionError("Correct paired comparison is not joint-valid")
    if paired["joint_valid_comparison"]["shuffled"]["valid_n"] != paired["joint_valid_paired_n"]:
        raise AssertionError("Shuffled paired comparison is not joint-valid")

    def group(label: str) -> dict[str, Any]:
        return group_summaries[label]

    six_questions = {
        "1_mh01_peak_margin_good_to_bad": {
            "good_median": group("MH01 good")["peak_margin_median"],
            "bad_median": group("MH01 bad")["peak_margin_median"],
            "decreased": _direction(group("MH01 bad")["peak_margin_median"], group("MH01 good")["peak_margin_median"], "lower"),
        },
        "2_mh05_peak_margin_good_to_bad": {
            "good_median": group("MH05 good")["peak_margin_median"],
            "bad_median": group("MH05 bad")["peak_margin_median"],
            "decreased": _direction(group("MH05 bad")["peak_margin_median"], group("MH05 good")["peak_margin_median"], "lower"),
        },
        "3_epipolar_error_good_to_bad": {
            sequence: {
                "good_median": group(f"{short} good")["epipolar_error_tokens_median"],
                "bad_median": group(f"{short} bad")["epipolar_error_tokens_median"],
                "increased": _direction(group(f"{short} bad")["epipolar_error_tokens_median"], group(f"{short} good")["epipolar_error_tokens_median"], "higher"),
            } for sequence, short in (("MH_01_easy", "MH01"), ("MH_05_difficult", "MH05"))
        },
        "4_cycle_and_geometry_good_to_bad": {
            sequence: {
                "cycle_decreased": _direction(group(f"{short} bad")["cycle_success_rate"], group(f"{short} good")["cycle_success_rate"], "lower"),
                "geometry_decreased": _direction(group(f"{short} bad")["geometry_consistent_rate"], group(f"{short} good")["geometry_consistent_rate"], "lower"),
            } for sequence, short in (("MH_01_easy", "MH01"), ("MH_05_difficult", "MH05"))
        },
        "5_temporal_gap_control": "See all 12 fixed sequence/group/bin cells; no matching, regression, resampling, significance test, or thresholding was used.",
        "6_correct_vs_shuffled": paired,
    }

    summary = {
        "schema_version": 1,
        "status": "formal_fixed_evidence_complete_pending_manual_gate",
        "scope": "Phase 1 / Experiment 2 formal 4000-sample fixed V-JEPA correspondence experiment",
        "thresholds": prepare_manifest["thresholds"],
        "groups": group_summaries,
        "temporal_gap_cells": temporal,
        "correct_vs_temporal_shuffled": paired,
        "six_core_questions": six_questions,
        "operational_definition": {
            "jepa_geometry_consistent": "epipolar_error_tokens <= 1.0 AND cycle_error_tokens <= 1.0",
            "interpretation": "geometry+cycle-consistent JEPA candidate; not recovery of true correspondence",
        },
        "decision_gate": {
            "status": "PENDING_MANUAL_REVIEW",
            "method": "one-time qualitative research judgment after fixed evidence generation; no numerical decision threshold",
            "shuffled_role": "descriptive null sanity only; not a necessary condition for NEGATIVE",
        },
        "provenance": {
            "prepare_manifest": "../artifacts/manifest.json",
            "extraction_manifest": "../artifacts/extraction_manifest.json",
            "samples_jsonl_sha256": prepare_manifest["samples_jsonl_sha256"],
            "vjepa": extraction_manifest["vjepa"],
            "geometry": extraction_manifest["geometry"],
        },
    }
    figures = _create_figures(report_dir, metrics, samples, group_summaries, paired, joint_indices)
    summary["figures"] = figures
    _write_json(report_dir / "summary.json", summary)

    lines = [
        "# Phase 1 / Experiment 2 — formal 4000-sample result",
        "",
        "Fixed evidence generation is complete. No DPVO rerun, sample replacement, parameter tuning, significance test, or protocol change was performed.",
        "",
        "## Main groups",
        "",
        "| Group | Total | Valid | Peak margin median | Epipolar median | Cycle success | Geometry consistent |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _, _ in GROUPS:
        value = group_summaries[label]
        lines.append(
            f"| {label} | {value['total_n']} | {value['valid_n']} | {_fmt(value['peak_margin_median'])} | "
            f"{_fmt(value['epipolar_error_tokens_median'])} | {_fmt(value['cycle_success_rate'])} | {_fmt(value['geometry_consistent_rate'])} |"
        )
    lines.extend([
        "",
        "`geometry_consistent_rate` is an operational feasibility criterion for geometry+cycle-consistent JEPA candidates. It is not a true-correspondence recovery rate.",
        "",
        "## Signed delta_t distributions",
        "",
        "| Group | N | Mean | Median | Q25 | Q75 | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for label, _, _ in GROUPS:
        value = group_summaries[label]["delta_t_seconds_signed"]
        lines.append(
            f"| {label} | {value['count']} | {_fmt(value['mean'])} | {_fmt(value['median'])} | {_fmt(value['q25'])} | "
            f"{_fmt(value['q75'])} | {_fmt(value['min'])} | {_fmt(value['max'])} |"
        )
    lines.extend([
        "",
        "## Fixed temporal-gap stratification",
        "",
        "| Sequence | Gap bin | Group | Total N | Valid N | Peak median | Epipolar median | Cycle success | Geometry consistent |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for value in temporal:
        lines.append(
            f"| {value['sequence']} | {value['gap_bin']} ({value['gap_definition']}) | {value['group']} | "
            f"{value['total_n']} | {value['valid_n']} | {_fmt(value['peak_margin_median'])} | "
            f"{_fmt(value['epipolar_error_tokens_median'])} | {_fmt(value['cycle_success_rate'])} | {_fmt(value['geometry_consistent_rate'])} |"
        )
    correct_pair = paired["joint_valid_comparison"]["correct"]
    shuffled_pair = paired["joint_valid_comparison"]["shuffled"]
    lines.extend([
        "",
        "## Paired temporal-shuffled descriptive null sanity",
        "",
        f"Total paired N: **{paired['total_paired_n']}**; correct valid N: **{paired['correct_valid_n']}**; shuffled valid N: **{paired['shuffled_valid_n']}**; joint-valid paired N: **{paired['joint_valid_paired_n']}**.",
        "",
        "All comparisons below use only the same joint-valid paired rows.",
        "",
        "| Pair | Peak median | Epipolar median | Cycle success | Geometry consistent |",
        "|---|---:|---:|---:|---:|",
        f"| Correct | {_fmt(correct_pair['peak_margin_median'])} | {_fmt(correct_pair['epipolar_error_tokens_median'])} | {_fmt(correct_pair['cycle_success_rate'])} | {_fmt(correct_pair['geometry_consistent_rate'])} |",
        f"| Temporal shuffled | {_fmt(shuffled_pair['peak_margin_median'])} | {_fmt(shuffled_pair['epipolar_error_tokens_median'])} | {_fmt(shuffled_pair['cycle_success_rate'])} | {_fmt(shuffled_pair['geometry_consistent_rate'])} |",
        "",
        "Shuffled is descriptive null sanity for non-random temporal correspondence signal. It is not strict negative GT, receives no significance test, and is not a necessary condition for a NEGATIVE gate.",
        "",
        "## Six core questions",
        "",
        f"1. MH01 peak margin good→bad: `{_fmt(group('MH01 good')['peak_margin_median'])}` → `{_fmt(group('MH01 bad')['peak_margin_median'])}`.",
        f"2. MH05 peak margin good→bad: `{_fmt(group('MH05 good')['peak_margin_median'])}` → `{_fmt(group('MH05 bad')['peak_margin_median'])}`.",
        f"3. Epipolar median good→bad: MH01 `{_fmt(group('MH01 good')['epipolar_error_tokens_median'])}` → `{_fmt(group('MH01 bad')['epipolar_error_tokens_median'])}`; MH05 `{_fmt(group('MH05 good')['epipolar_error_tokens_median'])}` → `{_fmt(group('MH05 bad')['epipolar_error_tokens_median'])}`.",
        f"4. Cycle/geometry rates good→bad: MH01 `{_fmt(group('MH01 good')['cycle_success_rate'])}/{_fmt(group('MH01 good')['geometry_consistent_rate'])}` → `{_fmt(group('MH01 bad')['cycle_success_rate'])}/{_fmt(group('MH01 bad')['geometry_consistent_rate'])}`; MH05 `{_fmt(group('MH05 good')['cycle_success_rate'])}/{_fmt(group('MH05 good')['geometry_consistent_rate'])}` → `{_fmt(group('MH05 bad')['cycle_success_rate'])}/{_fmt(group('MH05 bad')['geometry_consistent_rate'])}`.",
        "5. Temporal-gap-controlled direction is read from the complete 12-cell table above; small cells are retained without reweighting or strong interpretation.",
        f"6. Joint-valid correct vs shuffled geometry consistency: `{_fmt(correct_pair['geometry_consistent_rate'])}` vs `{_fmt(shuffled_pair['geometry_consistent_rate'])}` (descriptive only).",
        "",
        "## Complementarity interpretation boundary",
        "",
        "The complementarity map uses fixed DPVO `corr_margin_l0` against JEPA epipolar error, with cycle success as a marker. The final judgment asks whether the DPVO-bad population retains a stable, visibly meaningful geometry+cycle-consistent structure after controlling temporal gap; it does not equate an epipolar-band hit with true correspondence.",
        "",
        _gate_block("PENDING_MANUAL_REVIEW", "The fixed evidence must be read once before assigning POSITIVE, NEGATIVE, or AMBIGUOUS."),
        "",
        "## Figures",
        "",
        *[f"- `{path}`" for path in figures],
        "",
        "Negative, if assigned, is strictly limited to the current fixed V-JEPA 2.1 ViT-B framewise final dense representation; it does not claim that V-JEPA lacks correspondence information.",
        "",
    ])
    (report_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(_jsonable({"groups": group_summaries, "paired": paired, "temporal_gap_cells": temporal}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
