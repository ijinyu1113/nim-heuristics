"""Plot cheat evaluation: 3 methods × 3 regimes, median + IQR across seeds.

Reads nested JSON produced by cheat_eval/cheat_evaluate.py:
    {method_name: {seed: {regime: {move_acc, cheat_move_rate, ...}}}}

Matches paper style (plot_style.setup_style): DejaVu Serif, no top/right spines,
400 dpi on save, panel labels (a)/(b).
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, PALETTE, panel_label
from config import results_path

SUMMARY_PATH = results_path("eval_results_summary.json")
OUT_PATH = results_path("plots", "cheat_eval.png")

setup_style()

with open(SUMMARY_PATH) as f:
    data = json.load(f)

methods = list(data.keys())
regimes = ["Counter-Cheat", "Cheat-Consistent", "Neutral"]

# Canonical colors — match plot_main.py conventions
method_colors = {
    "Original (NoDANN)":                       PALETTE["baseline"],
    "DANN (lambda=0.05)":                      PALETTE["dann"],
    "Contrastive-only (lambda=1, no-paired)":  PALETTE["contrastive"],
}

# Pretty legend labels (the JSON keys stay verbose; these override for display).
# Method-level details (lambdas, n per seed, etc.) belong in the figure caption.
LABEL_OVERRIDES = {
    "Original (NoDANN)":                       "Baseline",
    "DANN (lambda=0.05)":                      "DANN",
    "Contrastive-only (lambda=1, no-paired)":  "Contrastive-only",
}


def agg(method, regime, metric):
    vals = []
    for _, regime_dict in data[method].items():
        if not isinstance(regime_dict, dict) or regime not in regime_dict:
            continue
        v = regime_dict[regime].get(metric)
        if v is not None:
            vals.append(v)
    return vals


def median_iqr(vals):
    """For small n (e.g. 3 seeds), use min/max so whiskers reflect the full spread.
    Switch to true Q1/Q3 by uncommenting the percentile lines."""
    if not vals:
        return 0.0, 0.0, 0.0, 0
    arr = np.array(vals, dtype=float)
    med = float(np.median(arr))
    q1 = float(arr.min())
    q3 = float(arr.max())
    # q1 = float(np.percentile(arr, 25))
    # q3 = float(np.percentile(arr, 75))
    return med, q1, q3, len(arr)


# Two-panel at paper proportions; widen left panel to fit 3 regime labels
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2),
                         gridspec_kw={"width_ratios": [3, 1]})

x = np.arange(len(regimes))
n_methods = len(methods)
width = 0.8 / n_methods

# --- Panel (a): Move accuracy across all regimes ---
ax = axes[0]
for i, m in enumerate(methods):
    color = method_colors.get(m, list(PALETTE.values())[i % len(PALETTE)])
    meds, lo_err, hi_err, ns = [], [], [], []
    for r in regimes:
        vals = agg(m, r, "move_acc")
        med, q1, q3, n = median_iqr(vals)
        meds.append(med)
        lo_err.append(max(med - q1, 0.0))
        hi_err.append(max(q3 - med, 0.0))
        ns.append(n)
    pos = x + (i - (n_methods - 1) / 2) * width
    ax.bar(pos, meds, width, label=LABEL_OVERRIDES.get(m, m), color=color,
           yerr=[lo_err, hi_err], capsize=3,
           error_kw={"lw": 0.8, "ecolor": "black"})

ax.set_xticks(x)
ax.set_xticklabels(regimes)
ax.set_ylabel("Move accuracy (%)")
ax.set_ylim(0, 108)
ax.grid(alpha=0.14, linewidth=0.5, axis="y")

# --- Panel (b): Cheat move rate on Counter-Cheat regime ---
ax = axes[1]
cheat_regime = "Counter-Cheat"
xc = np.arange(1)
for i, m in enumerate(methods):
    color = method_colors.get(m, list(PALETTE.values())[i % len(PALETTE)])
    vals = agg(m, cheat_regime, "cheat_move_rate")
    med, q1, q3, n = median_iqr(vals)
    pos = xc + (i - (n_methods - 1) / 2) * width
    hi = max(q3 - med, 0.0)
    ax.bar(pos, [med], width, color=color,
           yerr=[[max(med - q1, 0.0)], [hi]], capsize=3,
           error_kw={"lw": 0.8, "ecolor": "black"})

ax.set_xticks(xc)
ax.set_xticklabels([cheat_regime])
ax.set_ylabel("Cheat move rate (%)")
ax.set_ylim(0, 108)
ax.grid(alpha=0.14, linewidth=0.5, axis="y")

panel_label(axes[0], "(a)")
panel_label(axes[1], "(b)")

# Legend above figure, no frame
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=n_methods,
           bbox_to_anchor=(0.5, 1.05), frameon=False)

fig.tight_layout(rect=[0, 0, 1, 0.92])
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
