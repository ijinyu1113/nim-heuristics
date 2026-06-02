import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl


# run from inside maxrem8exps/
DATA_DIR = "."

MAX8_CONDITIONS = [
    ("max8scratch",              "Base",            "#1f77b4"),
    ("maxrem8postmaxrem2base",   "Post maxrem-2",   "#2ca02c"),
    ("maxrem8postmod3base",      "Post mod-3",      "#ff7f0e"),
    ("maxrem8postmaxrem4base",   "Post maxrem-4",   "#d62728"),
]

MAX4_CONTROL_CONDITIONS = [
    ("maxrem4postmaxrem2base",   "Post maxrem-2", "#1f77b4"),
    ("maxrem4postmod3base",      "Post mod-3",    "#ff7f0e"),
]


def collect_seeds(stem):
    pattern = os.path.join(DATA_DIR, f"{stem}_seed*_results.jsonl")
    rx = re.compile(rf"^{re.escape(stem)}_seed(\d+)_results\.jsonl$")
    out = []
    for path in sorted(glob.glob(pattern)):
        m = rx.match(os.path.basename(path))
        if m:
            out.append((int(m.group(1)), path))
    return out


# def load_agg(seed_files, needed_cols):
#     dfs = []
#     for _, path in seed_files:
#         df = pd.read_json(path, lines=True)
#         missing = [c for c in needed_cols if c not in df.columns]
#         if missing:
#             raise ValueError(f"{path} missing columns: {missing}")
#         dfs.append(df[needed_cols].sort_values("step").set_index("step"))

#     result = {}
#     for col in needed_cols[1:]:
#         mat = pd.concat([d[col] for d in dfs], axis=1)
#         result[f"{col}_mean"] = mat.mean(axis=1)
#         std = mat.std(axis=1, ddof=1).fillna(0.0)
#         n = mat.count(axis=1).clip(lower=1)
#         result[f"{col}_err"] = std / np.sqrt(n)  # SEM

#     agg = pd.DataFrame(result)
#     agg.index.name = "step"
#     return agg

def load_agg(seed_files, needed_cols):
    dfs = []
    for _, path in seed_files:
        df = pd.read_json(path, lines=True)
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        dfs.append(df[needed_cols].sort_values("step").set_index("step"))

    result = {}
    for col in needed_cols[1:]:
        mat = pd.concat([d[col] for d in dfs], axis=1)

        q1 = mat.quantile(0.25, axis=1)
        med = mat.quantile(0.50, axis=1)
        q3 = mat.quantile(0.75, axis=1)

        result[f"{col}_med"] = med
        result[f"{col}_q1"] = q1
        result[f"{col}_q3"] = q3
        result[f"{col}_iqr"] = q3 - q1  # optional, if you want to print/use IQR directly

    agg = pd.DataFrame(result)
    agg.index.name = "step"
    return agg

def setup_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
    })


# def draw(ax, df, col, color, ls="-", alpha_fill=0.14):
#     x = df.index.to_numpy() / 1000.0
#     y = df[f"{col}_mean"].to_numpy()
#     e = df[f"{col}_err"].to_numpy()
#     ax.plot(x, y, color=color, ls=ls)
#     ax.fill_between(x, y - e, y + e, color=color, alpha=alpha_fill, linewidth=0)

def draw(ax, df, col, color, ls="-", alpha_fill=0.14):
    x = df.index.to_numpy() / 1000.0
    y = df[f"{col}_med"].to_numpy()
    q1 = df[f"{col}_q1"].to_numpy()
    q3 = df[f"{col}_q3"].to_numpy()
    ax.plot(x, y, color=color, ls=ls)
    ax.fill_between(x, q1, q3, color=color, alpha=alpha_fill, linewidth=0)

def panel_label(ax, text):
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold"
    )


def make_two_panel_plot(
    conditions,
    right_metric,            # e.g. "mod3_acc"
    right_title,
    out_stub,
    show_train=False,
    left_ref=0.5,
    right_ref=0.5,
):
    needed = ["step", "eval_move_acc", "train_move_acc", f"eval_{right_metric}", f"train_{right_metric}"]

    data = {}
    for stem, label, color in conditions:
        seeds = collect_seeds(stem)
        if not seeds:
            print(f"WARNING: no files found for {stem}")
            continue
        data[stem] = load_agg(seeds, needed)
        print(f"{stem}: {len(seeds)} seed(s)")

    if not data:
        print(f"Skipping {out_stub}: no data found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), sharex=True, sharey=True)

    HLINE_COLOR = "#888888"
    HLINE_KW = dict(color=HLINE_COLOR, lw=1.0, ls=":", zorder=0)

    for stem, label, color in conditions:
        if stem not in data:
            continue
        df = data[stem]

        draw(axes[0], df, "eval_move_acc", color, ls="-", alpha_fill=0.14)
        draw(axes[1], df, f"eval_{right_metric}", color, ls="-", alpha_fill=0.14)

        if show_train:
            draw(axes[0], df, "train_move_acc", color, ls="--", alpha_fill=0.06)
            draw(axes[1], df, f"train_{right_metric}", color, ls="--", alpha_fill=0.06)

    axes[0].set_title("Exact move accuracy")
    axes[1].set_title(right_title)

    for ax in axes:
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5)
        ax.set_xlabel("Training step (×10$^3$)")
    axes[0].set_ylabel("Accuracy")

    axes[0].axhline(left_ref, **HLINE_KW)
    axes[1].axhline(right_ref, **HLINE_KW)

    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")

    legend_handles = [
        mlines.Line2D([], [], color=c, lw=2.2, label=l)
        for stem, l, c in conditions if stem in data
    ]
    if show_train:
        legend_handles += [
            mlines.Line2D([], [], color="black", lw=2.0, ls="-", label="Eval"),
            mlines.Line2D([], [], color="black", lw=2.0, ls="--", label="Train"),
        ]

    ncol = 3 if not show_train else 4
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = f"{out_stub}_train.png" if show_train else f"{out_stub}.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-train", action="store_true", help="Overlay train curves as dashed lines")
    args = parser.parse_args()

    setup_style()

    # Plot 1: MAX_REMOVE=8 with mod-3 panel
    make_two_panel_plot(
        conditions=MAX8_CONDITIONS,
        right_metric="mod3_acc",
        right_title="Mod-3 accuracy",
        out_stub="maxrem8_mod3_main",
        show_train=args.show_train,
        left_ref=1.0/3.0,
        right_ref=1.0 / 3.0,
    )

    # Plot 2: MAX_REMOVE=4 control
    # True quotient class is mod-5, but this diagnostic intentionally plots mod-3 accuracy.
    make_two_panel_plot(
        conditions=MAX4_CONTROL_CONDITIONS,
        right_metric="mod3_acc",
        right_title="Mod-3 accuracy",
        out_stub="maxrem4_control_mod3_main",
        show_train=args.show_train,
        left_ref=0.5,
        right_ref=1.0 / 3.0,
    )


if __name__ == "__main__":
    main()
