"""Plot probe accuracy heatmaps. Supports multiple model sets via CLI.

Matches paper style (plot_style.setup_style): DejaVu Serif, 400 dpi on save.
"""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style
from config import results_path

setup_style()

MODEL_SETS = {
    "3models": {
        "models": [
            ("nodann_v3",  "Baseline (\u03bb=0)"),
            ("dann_l05_v3", "DANN (\u03bb=0.05)"),
            ("cont_nopaired_l1_seed42_v3", "Contrastive-only (\u03bb=1.0, seed 42)"),
        ],
        "output": results_path("plots", "probe_heatmap_3models.png"),
        "title": "MLP Probe Accuracy: Cheat Detection",
    },
    "dann_lambdas": {
        "models": [
            ("dann_v3",         "DANN (\u03bb=0.025)"),
            ("dann_l03_v3",     "DANN (\u03bb=0.03)"),
            ("dann_l035_v3",    "DANN (\u03bb=0.035)"),
            ("dann_l05_v3",     "DANN (\u03bb=0.05)"),
        ],
        "output": results_path("plots", "probe_heatmap_dann_lambdas.png"),
        "title": "MLP Probe Accuracy: Cheat Detection (DANN sweep)",
    },
}

# Pretty y-axis labels
STRATEGY_LABELS = {
    "P1_first_occ_first_tok": "P1 first occ, first tok",
    "P1_first_occ_last_tok":  "P1 first occ, last tok",
    "P1_last_occ_first_tok":  "P1 last occ, first tok",
    "P1_last_occ_last_tok":   "P1 last occ, last tok",
    "P2_first_occ_first_tok": "P2 first occ, first tok",
    "P2_first_occ_last_tok":  "P2 first occ, last tok",
    "P2_last_occ_first_tok":  "P2 last occ, first tok",
    "P2_last_occ_last_tok":   "P2 last occ, last tok",
}

def load(name):
    with open(results_path(f"probe_ablation_{name}_results.json")) as f:
        return json.load(f)

def plot_set(config):
    models = config["models"]
    datas = [(label, load(key)) for key, label in models]

    strategies = list(datas[0][1].keys())
    layers = sorted({int(k) for s in strategies for k in datas[0][1][s].keys()})
    pretty_labels = [STRATEGY_LABELS.get(s, s) for s in strategies]

    mats = []
    for label, data in datas:
        mat = np.zeros((len(strategies), len(layers)))
        for i, s in enumerate(strategies):
            for j, l in enumerate(layers):
                mat[i, j] = data[s][str(l)] * 100
        mats.append((label, mat))

    vmin = min(m.min() for _, m in mats)
    vmax = max(m.max() for _, m in mats)

    n = len(mats)
    n_layers = len(layers)
    n_strat = len(strategies)
    # Each cell is a unit square; content aspect is n_layers : n_strat.
    cell_in = 0.22  # inches per cell
    content_h = cell_in * n_strat
    content_w_per_panel = cell_in * n_layers
    left_pad = 1.5   # strategy labels on leftmost panel
    right_pad = 0.7  # thin colorbar
    fig_w = content_w_per_panel * n + left_pad + right_pad
    fig_h = content_h + 1.2
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), sharey=True)
    if n == 1:
        axes = [axes]

    for idx, (label, mat) in enumerate(mats):
        ax = axes[idx]
        im = ax.imshow(mat, aspect="equal", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers)
        ax.set_yticks(range(len(strategies)))
        if idx == 0:
            ax.set_yticklabels(pretty_labels)
        ax.set_xlabel("Layer")
        for i in range(len(strategies)):
            for j in range(len(layers)):
                val = mat[i, j]
                color = "white" if val < (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        color=color, fontsize=6)

    # Thin shared colorbar on the right, tied to the axes grid height
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015, fraction=0.018)
    cbar.set_label("Probe acc. (%)")

    plt.savefig(config["output"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {config['output']}")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for key in MODEL_SETS:
            plot_set(MODEL_SETS[key])
    elif which in MODEL_SETS:
        plot_set(MODEL_SETS[which])
    else:
        print(f"Unknown set: {which}. Choose from: {list(MODEL_SETS.keys())} or 'all'")
