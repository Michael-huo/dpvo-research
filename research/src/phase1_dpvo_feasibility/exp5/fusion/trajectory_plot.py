"""Paper-style 2D XY trajectory figures for Exp5-2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phase1-exp5-trajectory")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from .trajectory_eval import aligned_positions_for_plot, validate_ground_truth_trajectory


PLOT_STYLES = {
    "baseline": ("#2a6fbb", "DPVO baseline", 1.5),
    "random": ("#808080", "Random dense fusion", 1.25),
    "learned": ("#d62728", "Learned dense fusion", 1.5),
}


def plot_xy_trajectories(
    *, sequence: str, estimates: dict[str, dict[str, Any]],
    ground_truth: dict[str, Any], output_path: str | Path,
) -> dict[str, Any]:
    if set(estimates) != set(PLOT_STYLES):
        raise ValueError("trajectory plot requires baseline, random and learned estimates")
    validate_ground_truth_trajectory(ground_truth)
    if ground_truth["sequence"] != sequence:
        raise ValueError("trajectory plot sequence/ground truth mismatch")
    destination = Path(output_path)
    if destination.suffix.lower() != ".png":
        raise ValueError("Exp5-2 trajectory plot output must be PNG")
    destination.parent.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(ground_truth["poses"], dtype=np.float64)[:, :3]
    figure, axis = plt.subplots(figsize=(8.0, 6.4))
    axis.plot(gt[:, 0], gt[:, 1], color="black", linewidth=2.0, label="GT", zorder=1)
    axis.scatter(gt[0, 0], gt[0, 1], color="black", marker="o", s=30, zorder=5)
    axis.scatter(gt[-1, 0], gt[-1, 1], color="black", marker="X", s=38, zorder=5)
    for alias in ("baseline", "random", "learned"):
        color, label, width = PLOT_STYLES[alias]
        payload = estimates[alias]
        if payload["sequence"] != sequence:
            raise ValueError(f"trajectory plot sequence mismatch: {alias}")
        aligned = aligned_positions_for_plot(payload, ground_truth)
        axis.plot(
            aligned[:, 0], aligned[:, 1], color=color, linewidth=width,
            label=label, zorder=2,
        )
        axis.scatter(
            aligned[0, 0], aligned[0, 1], color=color, marker="o", s=24, zorder=6
        )
        axis.scatter(
            aligned[-1, 0], aligned[-1, 1], color=color, marker="X", s=32, zorder=6
        )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(f"{sequence} — Exp5-2 trajectory comparison")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8.5)
    axis.text(
        0.01, 0.01, "○ start    × end", transform=axis.transAxes,
        fontsize=8, ha="left", va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8,
              "edgecolor": "#aaaaaa"},
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"trajectory plot was not generated: {destination}")
    return {
        "path": str(destination),
        "format": "png",
        "projection": "2D_XY",
        "equal_axis": True,
        "start_end_markers": True,
    }


def plot_oracle_xy_trajectories(
    *, sequence: str, baseline: dict[str, Any], oracle: dict[str, Any],
    ground_truth: dict[str, Any], output_path: str | Path,
) -> dict[str, Any]:
    """Plot the MH_05 full-RGB baseline against Oracle FNet replacement."""
    validate_ground_truth_trajectory(ground_truth)
    if sequence != "MH_05_difficult" or ground_truth["sequence"] != sequence:
        raise ValueError("Exp5-Oracle plotting is restricted to MH_05_difficult")
    if baseline.get("method") != "dpvo_baseline":
        raise ValueError("Oracle plot baseline method mismatch")
    if oracle.get("method") != "oracle_jepa_missing_rgb":
        raise ValueError("Oracle plot replacement method mismatch")
    destination = Path(output_path)
    if destination.suffix.lower() != ".png":
        raise ValueError("Exp5-Oracle trajectory plot output must be PNG")
    destination.parent.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(ground_truth["poses"], dtype=np.float64)[:, :3]
    aligned_baseline = aligned_positions_for_plot(baseline, ground_truth)
    aligned_oracle = aligned_positions_for_plot(oracle, ground_truth)
    figure, axis = plt.subplots(figsize=(8.0, 6.4))
    curves = (
        (gt, "black", "GT", 2.0),
        (aligned_baseline, "#2a6fbb", "Full RGB DPVO", 1.5),
        (aligned_oracle, "#d62728", "Oracle JEPA FNet replacement", 1.5),
    )
    for positions, color, label, width in curves:
        axis.plot(
            positions[:, 0], positions[:, 1], color=color,
            linewidth=width, label=label,
        )
        axis.scatter(
            positions[0, 0], positions[0, 1], color=color,
            marker="o", s=28, zorder=5,
        )
        axis.scatter(
            positions[-1, 0], positions[-1, 1], color=color,
            marker="X", s=36, zorder=5,
        )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("MH_05_difficult — Oracle JEPA FNet replacement")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8.5)
    axis.text(
        0.01, 0.01, "○ start    × end", transform=axis.transAxes,
        fontsize=8, ha="left", va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8,
              "edgecolor": "#aaaaaa"},
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Oracle trajectory plot was not generated: {destination}")
    return {
        "path": str(destination),
        "format": "png",
        "projection": "2D_XY",
        "equal_axis": True,
        "start_end_markers": True,
    }
