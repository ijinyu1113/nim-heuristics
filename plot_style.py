"""Shared plotting style for paper figures, matching Leo's plot_main.py conventions.

Usage:
    from plot_style import setup_style, PALETTE, HLINE_KW, panel_label, draw_iqr
    setup_style()
"""
import matplotlib as mpl
import matplotlib.lines as mlines
import numpy as np


# Canonical method colors (matplotlib tab-10; align with plot_main.py choices)
PALETTE = {
    "baseline":    "#1f77b4",  # blue
    "dann":        "#d62728",  # red
    "contrastive": "#2ca02c",  # green
    "augmentation":"#ff7f0e",  # orange
    "other":       "#9467bd",  # purple
}

HLINE_COLOR = "#888888"
HLINE_KW = dict(color=HLINE_COLOR, lw=1.0, ls=":", zorder=0)


def setup_style():
    """Apply paper-figure rcParams. Call once at the top of a plot script."""
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


def panel_label(ax, text):
    """Top-left (a)/(b) panel label."""
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold",
    )


def draw_iqr(ax, x, med, q1, q3, color, ls="-", alpha_fill=0.14):
    """Plot median line with Q1-Q3 shaded band."""
    ax.plot(x, med, color=color, ls=ls)
    ax.fill_between(x, q1, q3, color=color, alpha=alpha_fill, linewidth=0)


def method_legend_handles(conditions, show_train=False):
    """Build legend handles from a list of (stem, label, color) tuples."""
    handles = [
        mlines.Line2D([], [], color=c, lw=2.2, label=l)
        for _, l, c in conditions
    ]
    if show_train:
        handles += [
            mlines.Line2D([], [], color="black", lw=2.0, ls="-",  label="Eval"),
            mlines.Line2D([], [], color="black", lw=2.0, ls="--", label="Train"),
        ]
    return handles
