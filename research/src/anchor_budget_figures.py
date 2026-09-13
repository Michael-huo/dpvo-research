"""Two measured-data figures and a short summary from the compact bundle only."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .protocol import atomic_write_bytes, sha256_file

COLORS = {"GT": "black", "Full RGB": "#2a6fbb", "Sparse RGB": "#d9822b", "Ours": "#2f9e44"}
STRIDE_COLORS = {3: "#2a6fbb", 5: "#d9822b", 10: "#2f9e44"}


def _plot_library():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-anchor-budget")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def aligned_translation(bundle, name, condition):
    """Apply the stored transform to a copy; never fit or alter raw trajectories."""
    sim3 = condition["canonical_evaluation"]["sim3"]
    xyz = np.asarray(bundle[f"{name}__translation"])
    return (float(sim3["scale"]) * (np.asarray(sim3["rotation"]) @ xyz.T)
            + np.asarray(sim3["translation"])[:, None]).T


def _number(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _trajectory_figure(root, data, bundle):
    plt = _plot_library()
    from matplotlib.lines import Line2D
    strides = data["metadata"]["anchor_strides"]
    full = aligned_translation(bundle, "Full_RGB", data["full_rgb"])
    span = bundle["Full_RGB__timestamps_ns"]
    gt_time = bundle["GT__timestamps_ns"]
    gt = bundle["GT__translation"][(gt_time >= span.min()) & (gt_time <= span.max())]
    panels = []
    for stride in strides:
        row = data["strides"][str(stride)]
        panels.append({"GT": gt, "Full RGB": full,
                       "Sparse RGB": aligned_translation(bundle, f"stride_{stride}_sparse", row["sparse"]),
                       "Ours": aligned_translation(bundle, f"stride_{stride}_ours", row["ours"])})
    points = np.concatenate([xy[:, :2] for panel in panels for xy in panel.values()])
    low, high = points.min(axis=0), points.max(axis=0)
    padding = np.maximum((high - low) * .06, .2)
    figure, axes = plt.subplots(1, len(strides), figsize=(4.4 * len(strides), 7.6),
                                squeeze=False, sharex=True, sharey=True)
    styles = {"GT": "-", "Full RGB": "-", "Sparse RGB": "--", "Ours": "-"}
    for axis, stride, panel in zip(axes[0], strides, panels):
        row = data["strides"][str(stride)]
        for label, xyz in panel.items():
            axis.plot(xyz[:, 0], xyz[:, 1], color=COLORS[label], linestyle=styles[label],
                      linewidth=2.0 if label == "GT" else 1.35, alpha=.9)
            axis.scatter(*xyz[0, :2], marker="o", s=38, color=COLORS[label], edgecolors="white", linewidths=.7, zorder=6)
            axis.scatter(*xyz[-1, :2], marker="x", s=48, color=COLORS[label], linewidths=1.7, zorder=7)
        ratio = row["communication"]["actual_anchor_ratio"] * 100
        sparse = row["sparse"]["canonical_evaluation"]["ate_rmse_m"]
        ours = row["ours"]["canonical_evaluation"]["ate_rmse_m"]
        axis.set_title(f"Stride {stride}  ·  {ratio:.2f}% anchors\nSparse ATE {sparse:.4f} m  |  Ours ATE {ours:.4f} m", fontsize=11)
        axis.set_xlim(low[0]-padding[0], high[0]+padding[0])
        axis.set_ylim(low[1]-padding[1], high[1]+padding[1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x [m]")
        axis.grid(alpha=.25)
    axes[0,0].set_ylabel("y [m]")
    handles = [Line2D([], [], color=COLORS[name], linestyle=styles[name], label=name,
                      linewidth=2 if name == "GT" else 1.35) for name in COLORS]
    handles += [Line2D([], [], marker="o", color="gray", linestyle="None", label="Start"),
                Line2D([], [], marker="x", color="gray", linestyle="None", label="End")]
    figure.suptitle(f"{data['metadata']['sequence']} — Anchor budget trajectories", fontsize=16, y=.99)
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.035), ncol=6, frameon=False)
    figure.text(.5, .012, "Original trajectories; saved Sim(3) transforms applied without refitting. Shared Full RGB run and identical axes.",
                ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0,.11,1,.92))
    figure.savefig(root / "figures/trajectories.png", dpi=180)
    plt.close(figure)


def _tradeoff_figure(root, data):
    plt = _plot_library()
    from matplotlib.lines import Line2D
    strides = data["metadata"]["anchor_strides"]
    rows = [data["strides"][str(s)] for s in strides]
    ratios = [r["communication"]["actual_anchor_ratio"]*100 for r in rows]
    reductions = [r["communication"]["encoded_byte_reduction"]*100 for r in rows]
    figure, axes = plt.subplots(2,3, figsize=(17.6,9.4))
    for axis, x, title, xlabel in (
        (axes[0,0], ratios, "A · Accuracy vs upload budget", "Actual anchor ratio [%]"),
        (axes[0,1], reductions, "A · Accuracy vs byte reduction", "Encoded byte reduction [%]"),
        (axes[0,2], ratios, "B · Matched trajectory wall", "Actual anchor ratio [%]")):
        is_wall = axis is axes[0,2]
        for key, label, marker in (("sparse", "Sparse RGB", "s"), ("ours", "Ours", "o")):
            y = [row[key]["matched_trajectory_wall_seconds"] if is_wall else
                 row[key]["canonical_evaluation"]["ate_rmse_m"] for row in rows]
            axis.scatter(x, y, color=COLORS[label], marker=marker, s=65, label=label, zorder=3)
            for xvalue, yvalue, stride in zip(x,y,strides):
                axis.annotate(f"K={stride}", (xvalue,yvalue), xytext=(4,6), textcoords="offset points", fontsize=8)
        axis.set_title(title, loc="left", fontsize=12)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Wall [s]" if is_wall else "ATE RMSE [m]")
        axis.set_ylim(bottom=0)
        axis.legend(fontsize=9)
    context = axes[1,0]
    for field, label, marker in (("context_wait_ms", "Context wait (mean)", "o"),
                                ("effective_hidden_delay_ms", "Effective hidden delay (mean)", "D")):
        context.scatter(ratios, [r["ours"]["h2_stage_profile"][field]["mean_ms"] for r in rows],
                        marker=marker, s=65, label=label)
    context.set(title="B · Delayed observation latency", xlabel="Actual anchor ratio [%]", ylabel="Latency [ms]")
    context.set_ylim(bottom=0)
    context.legend(fontsize=9)
    for axis, predicted, baseline, title in (
        (axes[1,1], "predicted_jepa_cosine", "transport_baseline_cosine", "C · JEPA quality vs prediction horizon"),
        (axes[1,2], "bridge_fmap_cosine", "transport_bridge_fmap_cosine", "C · FMap quality vs prediction horizon")):
        for stride, row in zip(strides,rows):
            horizon = row["horizon"]["by_relative_index"]
            x = [r["distance_from_previous_anchor_seconds"] for r in horizon]
            axis.scatter(x, [r[predicted] for r in horizon], color=STRIDE_COLORS[stride], s=46, marker="o")
            axis.scatter(x, [r[baseline] for r in horizon], edgecolors=STRIDE_COLORS[stride], facecolors="none", s=50, marker="D")
        axis.set(title=title, xlabel="Distance from previous anchor [s]", ylabel="Held-out cosine")
        handles = [Line2D([], [], color=STRIDE_COLORS[s], marker="o", linestyle="None", label=f"Stride {s}") for s in strides]
        handles += [Line2D([], [], color="gray", marker="o", linestyle="None", label="Prediction"),
                    Line2D([], [], color="gray", marker="D", markerfacecolor="none", linestyle="None", label="Transport")]
        axis.legend(handles=handles, fontsize=8, ncol=2)
    for axis in axes.flat:
        axis.grid(alpha=.25)
        axis.margins(x=.16, y=.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"{data['metadata']['sequence']} — Measured anchor-budget tradeoffs", fontsize=16)
    figure.text(.5,.014, "Measured points only; no interpolation. ATE uses each stride's original paired anchor population. Horizon diagnostics are offline held-out measurements.",
                ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0,.035,1,.96), h_pad=2.1, w_pad=2.0)
    figure.savefig(root / "figures/tradeoffs.png", dpi=180)
    plt.close(figure)


def render_figures(root):
    """Only results.json and trajectories.npz are read; no dataset/model required."""
    from .anchor_budget_artifacts import validate_compact
    root = Path(root)
    data = validate_compact(root, check_checkpoints=False)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    with np.load(root / "trajectories.npz", allow_pickle=False) as bundle:
        _trajectory_figure(root, data, bundle)
    _tradeoff_figure(root, data)


def write_summary(root):
    root = Path(root)
    data = json.loads((root / "results.json").read_text())
    metadata, strides = data["metadata"], data["metadata"]["anchor_strides"]
    split = metadata["fixed_split"]["regions"]
    lines = ["# Phase 2 — Anchor Budget / Prediction Horizon", "",
             "Question: how do Sparse RGB and Ours change as anchor upload budget decreases and prediction horizon grows?", "",
             f"Sequence: **{metadata['sequence']}**; strides: **{', '.join(map(str,strides))}**. Fresh predictor per stride; frozen H1 bridge.",
             "Fixed inclusive candidate regions: " + "; ".join(f"{name} {split[name]['candidate_start']}–{split[name]['candidate_end']}" for name in ("train", "validation", "test")) + ".",
             "GT is a reference; Full RGB is one shared run. All SLAM runs were independent and sequential.", "",
             "| Stride | Anchor % | Byte reduction | Sparse ATE [m] | Ours ATE [m] | Ours−Sparse [m] | Ours wall [s] | Context wait [ms] |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for stride in strides:
        comparison = data["strides"][str(stride)]["comparison"]
        values = [stride, f"{comparison['actual_anchor_ratio']*100:.3f}", f"{comparison['encoded_byte_reduction']*100:.3f}"]
        values += [_number(comparison[k]) for k in ("sparse_ate_rmse_m", "ours_ate_rmse_m", "ours_minus_sparse_ate_m",
                                                   "ours_wall_seconds", "context_wait_mean_ms")]
        lines.append("| " + " | ".join(map(str,values)) + " |")
    full = data["full_rgb"]
    lines += ["", f"Full RGB: ATE {_number(full['canonical_evaluation']['ate_rmse_m'])} m; matched wall {_number(full['matched_trajectory_wall_seconds'])} s (frozen stride-5 evaluation population).",
              "ATE populations remain paired within each stride; stride 3 RPE is unavailable under the unchanged 1 s rule (zero pairs).", ""]
    quality = []
    for stride in strides:
        held = data["strides"][str(stride)]["training"]["held_out_representation"]
        jepa = held["gate1_block5"]["predicted_jepa"]["cosine"]
        fmap = held["gate2_frozen_h1_bridge_vs_true_fmap"]["predicted_jepa"]["cosine"]
        quality.append(f"K={stride}: {jepa:.4f}/{fmap:.4f}")
    lines += ["Held-out JEPA/FMap cosine: " + "; ".join(quality) + ". Per-position values and both anchor time distances are in `results.json → strides → horizon`.", "",
              f"Config SHA256: `{metadata['config_sha256']}`."]
    bridge = data["strides"][str(strides[0])]["provenance"]["H1_bridge_sha256"]
    lines.append(f"H1 bridge SHA256: `{bridge}`.")
    for stride in strides:
        checkpoint = data["strides"][str(stride)]["provenance"]["checkpoint"]
        lines.append(f"Predictor K={stride}: `{checkpoint['relative_path']}`; SHA256 `{checkpoint['sha256']}`; seed {checkpoint['seed']}, best epoch {checkpoint['best_epoch']}.")
    original = metadata["original_run_repository"].get("git_commit") or "not recorded (original source hashes retained)"
    lines += [f"Experiment commit: {original}; artifact-publication commit: `{metadata['artifact_publication_repository']['git_commit']}`.",
              f"Results SHA256: `{sha256_file(root / 'results.json')}`.", "",
              "[All numerical results](results.json) · [Raw trajectories](trajectories.npz) · [Trajectories](figures/trajectories.png) · [Tradeoffs](figures/tradeoffs.png)", ""]
    atomic_write_bytes(root / "SUMMARY.md", "\n".join(lines).encode())
