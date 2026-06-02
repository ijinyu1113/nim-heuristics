"""Replot avg_lines_{tag}.png from cached results/intervention_avg/results_{tag}.npz.

Runs locally — no HPC model load needed. Produces the paper-style 2-panel figure
(induce / stop at P1+P2 names) without redoing ROME-style inference.

Usage:
    python replot_intervention_avg.py                 # replots all tags found
    python replot_intervention_avg.py final           # just one tag
    python replot_intervention_avg.py final checkpoint_50
"""
import glob
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style
from config import results_path

setup_style()

RESULTS_DIR = results_path("intervention_avg")
OUT_DIR = results_path("plots")
os.makedirs(OUT_DIR, exist_ok=True)


def replot_tag(tag):
    npz_path = os.path.join(RESULTS_DIR, f"results_{tag}.npz")
    if not os.path.isfile(npz_path):
        print(f"skip: {npz_path} not found")
        return
    data = np.load(npz_path, allow_pickle=True)
    arrs = [data[f"exp{i}"] for i in (1, 2, 3, 4)]
    # Median + IQR across the N name-pairs (matches plot_runs_aggregated /
    # plot_purenum_curves convention).
    medians = [np.median(a, axis=0) for a in arrs]
    q1s = [np.percentile(a, 25, axis=0) for a in arrs]
    q3s = [np.percentile(a, 75, axis=0) for a in arrs]
    n = int(data["num_pairs"]) if "num_pairs" in data.files else arrs[0].shape[0]
    layers = np.arange(len(medians[0]))

    # 2x2: rows = token position (names / final), cols = intervention (induce / stop)
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0), sharex=True, sharey=True)

    def plot_iqr(ax, med, q1, q3, baseline, color):
        ax.plot(layers, med, color=color, marker="o", markersize=3)
        ax.fill_between(layers,
                        np.clip(q1, 0, 1),
                        np.clip(q3, 0, 1),
                        alpha=0.14, color=color, linewidth=0)
        # Dotted line at the no-intervention baseline (caption explains)
        ax.axhline(baseline, color="#888888", lw=1.0, ls=":", zorder=0)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5)

    grid_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    panel_colors = ["#d62728", "#1f77b4", "#d62728", "#1f77b4"]
    panel_baselines = [0.0, 1.0, 0.0, 1.0]
    for ax, med, q1, q3, color, base in zip(grid_axes, medians, q1s, q3s, panel_colors, panel_baselines):
        plot_iqr(ax, med, q1, q3, base, color)

    axes[0, 0].set_ylabel("P(cheat)")
    axes[1, 0].set_ylabel("P(cheat)")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 1].set_xlabel("Layer")

    for idx, ax in enumerate(grid_axes):
        ax.text(0.02, 0.98, f"({'abcd'[idx]})", transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold")

    fig.tight_layout()

    out = os.path.join(OUT_DIR, f"avg_lines_{tag}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}  (N={n} pairs)")


def discover_tags():
    tags = []
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "results_*.npz"))):
        m = re.match(r"results_(.+)\.npz$", os.path.basename(p))
        if m:
            tags.append(m.group(1))
    return tags


if __name__ == "__main__":
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = discover_tags()
        print(f"Replotting all tags: {targets}")
    for tag in targets:
        replot_tag(tag)
