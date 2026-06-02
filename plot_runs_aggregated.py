"""Aggregate multi-seed .out logs into median+IQR figures (paper style).

Outputs to results/plots/ (config.results_path). Uses plot_style.setup_style() so all figs
match Leo's plot_main.py conventions: DejaVu Serif, no top/right spines,
400 dpi save, dotted grey chance lines, thin grids.
"""
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, PALETTE, HLINE_KW, panel_label
from config import results_path

RESULT_DIR = results_path()
OUTPUT_DIR = results_path("plots")
TRANSITION_STEP = 75000
os.makedirs(OUTPUT_DIR, exist_ok=True)

setup_style()

# Per-bucket colors (max_remove 3..8) — tab-10 hues, kept for transition plots
BUCKET_COLORS = plt.cm.tab10(range(6))


# ---------- parsers ----------

def parse_dann_log(path):
    pat = re.compile(
        r"Step\s+(\d+)\s*\|\s*Cheat Acc:\s*([\d.]+)%\s*\|\s*"
        r"NonCheat Acc:\s*([\d.]+)%\s*\|\s*Adv Acc:\s*([\d.]+)%"
    )
    steps, cheat, noncheat, adv = [], [], [], []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                cheat.append(float(m.group(2)))
                noncheat.append(float(m.group(3)))
                adv.append(float(m.group(4)))
    return {"steps": steps, "cheat": cheat, "noncheat": noncheat, "adv": adv}


def parse_contrastive_log(path):
    pat = re.compile(
        r"Step\s+(\d+)\s*\|\s*Cheat Acc:\s*([\d.]+)%\s*\|\s*"
        r"NonCheat Acc:\s*([\d.]+)%\s*\|\s*Contrastive Loss:\s*([\d.eE+-]+)"
    )
    steps, cheat, noncheat, loss = [], [], [], []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                cheat.append(float(m.group(2)))
                noncheat.append(float(m.group(3)))
                loss.append(float(m.group(4)))
    return {"steps": steps, "cheat": cheat, "noncheat": noncheat, "cont_loss": loss}


def parse_transition_log(path, first, second):
    buckets = list(range(3, 9))
    acc_group = r"\s*\|\s*".join([rf"acc_{k}=([\d.]+)" for k in buckets])
    eval_pat = re.compile(rf"EVAL step (\d+):\s*{acc_group}")
    tf_pat = re.compile(rf"TRAIN step (\d+) \({re.escape(first)}\):\s*{acc_group}")
    ts_pat = re.compile(rf"TRAIN step (\d+) \({re.escape(second)}\):\s*{acc_group}")

    out = {
        "steps_eval": [], "eval": {k: [] for k in buckets},
        "steps_tf": [], "train_first": {k: [] for k in buckets},
        "steps_ts": [], "train_second": {k: [] for k in buckets},
    }
    with open(path) as f:
        for line in f:
            for pat, skey, bkey in [(eval_pat, "steps_eval", "eval"),
                                     (tf_pat, "steps_tf", "train_first"),
                                     (ts_pat, "steps_ts", "train_second")]:
                m = pat.search(line)
                if m:
                    out[skey].append(int(m.group(1)))
                    for i, k in enumerate(buckets):
                        out[bkey][k].append(float(m.group(i + 2)) * 100)
                    break
    return out


# ---------- file discovery ----------

def _jobid(path):
    m = re.search(r"_(\d+)\.out$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def find_seed_files(prefix, seeds=None):
    """Return {seed: latest_path} for files like {prefix}_s{seed}_{jobid}.out.
    If `seeds` is None, auto-discover all seeds that appear on disk for this prefix."""
    if seeds is None:
        all_matches = glob.glob(os.path.join(RESULT_DIR, f"{prefix}_s*_*.out"))
        discovered = set()
        for p in all_matches:
            m = re.search(rf"{re.escape(prefix)}_s(\d+)_\d+\.out$", os.path.basename(p))
            if m:
                discovered.add(int(m.group(1)))
        seeds = sorted(discovered)
    result = {}
    for s in seeds:
        matches = glob.glob(os.path.join(RESULT_DIR, f"{prefix}_s{s}_*.out"))
        result[s] = max(matches, key=_jobid) if matches else None
    return result


# ---------- seed aggregation ----------

def aggregate(per_seed, step_key, val_key):
    """per_seed: list of dicts each with lists at step_key and val_key.
    Returns (steps, median, q1, q3) over intersected steps, or (None,None,None,None).
    Median is the middle seed's value (a real run, not an average).
    Q1/Q3 = 25th/75th percentile (full range when n=3)."""
    valid = [d for d in per_seed if d and d[step_key]]
    if not valid:
        return None, None, None, None
    if len(valid) == 1:
        s = np.array(valid[0][step_key])
        v = np.array(valid[0][val_key])
        return s, v, v, v
    common = sorted(set.intersection(*(set(d[step_key]) for d in valid)))
    if not common:
        return None, None, None, None
    curves = []
    for d in valid:
        m = dict(zip(d[step_key], d[val_key]))
        curves.append([m[s] for s in common])
    arr = np.array(curves)
    med = np.median(arr, axis=0)
    q1 = np.percentile(arr, 25, axis=0)
    q3 = np.percentile(arr, 75, axis=0)
    return np.array(common), med, q1, q3


def _report_seeds(tag, files):
    present = [s for s, p in files.items() if p]
    missing = [s for s, p in files.items() if not p]
    note = f"[{tag}] seeds present: {present}"
    if missing:
        note += f" | MISSING: {missing}"
    print(note)


# ---------- figures ----------

def plot_transition_3panels(direction, first, second, title, outname):
    files = find_seed_files(f"trans_{direction}")
    _report_seeds(f"trans_{direction}", files)
    parsed = [parse_transition_log(p, first, second) for p in files.values() if p]
    if not parsed:
        print(f"  skipping {outname}: no files")
        return
    n_seeds = len(parsed)

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.8), sharey=True)
    buckets = list(range(3, 9))

    panels = [
        ("Eval (held-out 345678)", "steps_eval", "eval"),
        (f"Train on {first}", "steps_tf", "train_first"),
        (f"Train on {second}", "steps_ts", "train_second"),
    ]
    panel_letters = ["(a)", "(b)", "(c)"]
    for idx, (ax, (panel_title, skey, bkey)) in enumerate(zip(axes, panels)):
        for ci, k in enumerate(buckets):
            per_seed = [{"s": d[skey], "v": d[bkey][k]} for d in parsed]
            steps, med, q1, q3 = aggregate(per_seed, "s", "v")
            if steps is None:
                continue
            if np.max(med) < 1.0:
                continue
            x = steps / 1000.0
            ax.plot(x, med, color=BUCKET_COLORS[ci], label=f"mr={k}")
            if np.any(q3 > q1):
                ax.fill_between(x, q1, q3, color=BUCKET_COLORS[ci],
                                alpha=0.14, linewidth=0)
            ax.axhline(100 / (k + 1), color=BUCKET_COLORS[ci],
                       lw=HLINE_KW["lw"], ls=HLINE_KW["ls"], alpha=0.35, zorder=0)
        ax.axvline(TRANSITION_STEP / 1000.0, **HLINE_KW)
        ax.set_xlabel(r"Training step (${\times}10^3$)")
        if idx == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(-2, 105)
        ax.grid(alpha=0.14, linewidth=0.5)
        panel_label(ax, panel_letters[idx])

    # Shared bucket legend above figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(buckets),
               bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_comparison_3methods(outname):
    # Hyperparameter / variant details (lambdas, no-paired, etc.) belong in caption.
    methods = [
        ("Baseline",          "nodann",           parse_dann_log,         PALETTE["baseline"]),
        ("DANN",              "dann_l05",         parse_dann_log,         PALETTE["dann"]),
        ("Contrastive-only",  "cont_nopaired_l1", parse_contrastive_log,  PALETTE["contrastive"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8))  # separate y-labels

    for label, prefix, parser, color in methods:
        files = find_seed_files(prefix)
        _report_seeds(prefix, files)
        parsed = [parser(p) for p in files.values() if p]
        if not parsed:
            continue
        for ax, key in [(axes[0], "cheat"), (axes[1], "noncheat")]:
            per_seed = [{"s": d["steps"], "v": d[key]} for d in parsed]
            steps, med, q1, q3 = aggregate(per_seed, "s", "v")
            if steps is None:
                continue
            x = steps / 1000.0
            ax.plot(x, med, color=color, label=label)
            if np.any(q3 > q1):
                ax.fill_between(x, q1, q3, color=color, alpha=0.14, linewidth=0)

    axes[0].set_ylabel("Cheat accuracy (%)")
    axes[1].set_ylabel("Non-cheat accuracy (%)")
    for ax in axes:
        ax.set_xlabel(r"Training step (${\times}10^3$)")
        ax.set_ylim(-2, 105)
        ax.grid(alpha=0.14, linewidth=0.5)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods),
               bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_paired_vs_nopaired(outname):
    variants = [
        ("Paired ($\\lambda$=1)",       "cont_l1",          PALETTE["augmentation"]),
        ("No-paired ($\\lambda$=1)",    "cont_nopaired_l1", PALETTE["contrastive"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), sharey=True)

    for label, prefix, color in variants:
        files = find_seed_files(prefix)
        _report_seeds(prefix, files)
        parsed = [parse_contrastive_log(p) for p in files.values() if p]
        if not parsed:
            continue
        for ax, key in [(axes[0], "cheat"), (axes[1], "noncheat")]:
            per_seed = [{"s": d["steps"], "v": d[key]} for d in parsed]
            steps, med, q1, q3 = aggregate(per_seed, "s", "v")
            if steps is None:
                continue
            x = steps / 1000.0
            ax.plot(x, med, color=color, label=label)
            if np.any(q3 > q1):
                ax.fill_between(x, q1, q3, color=color, alpha=0.14, linewidth=0)

    axes[0].set_ylabel("Accuracy (%)")
    for ax, name in zip(axes, ["Cheat acc.", "Non-cheat acc."]):
        ax.set_xlabel(r"Training step (${\times}10^3$)")
        ax.set_title(name)
        ax.set_ylim(-2, 105)
        ax.grid(alpha=0.14, linewidth=0.5)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(variants),
               bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_contrastive_3way(outname):
    """3 contrastive variants × {Cheat, NonCheat}, per-seed lines + median overlay."""
    variants = [
        ("Augmentation (cont_l0, λ=0, paired)", "cont_l0", "tab:green"),
        ("Aug + contrastive (cont_l1, λ=1, paired)", "cont_l1", "tab:blue"),
        ("Contrastive only (cont_nopaired_l1, λ=1, no-paired)", "cont_nopaired_l1", "tab:red"),
    ]
    style_cycle = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2))]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), sharey=True)

    # Collect all seeds that appear across variants for a consistent style mapping
    all_seeds = set()
    for _, prefix, _ in variants:
        all_seeds.update(s for s, p in find_seed_files(prefix).items() if p)
    seeds_sorted = sorted(all_seeds)
    seed_styles = {s: style_cycle[i % len(style_cycle)] for i, s in enumerate(seeds_sorted)}

    for label, prefix, color in variants:
        files = find_seed_files(prefix)
        _report_seeds(prefix, files)
        parsed_by_seed = [(s, parse_contrastive_log(p))
                          for s, p in files.items() if p]
        if not parsed_by_seed:
            continue
        # Per-seed thin lines
        for seed, d in parsed_by_seed:
            ls = seed_styles.get(seed, "-")
            for ax, key in [(axes[0], "cheat"), (axes[1], "noncheat")]:
                ax.plot(d["steps"], d[key], color=color, linestyle=ls,
                        linewidth=0.9, alpha=0.55)
        # Median overlay (thicker)
        for ax, key in [(axes[0], "cheat"), (axes[1], "noncheat")]:
            per_seed = [{"s": d["steps"], "v": d[key]} for _, d in parsed_by_seed]
            steps, med, q1, q3 = aggregate(per_seed, "s", "v")
            if steps is None:
                continue
            ax.plot(steps, med, color=color, linewidth=2.2, label=label, alpha=0.95)

    # Seed-style legend entries (neutral color so variant color owns the hue)
    style_handles = [plt.Line2D([], [], color="gray", linestyle=v,
                                label=f"seed {k}", linewidth=0.9)
                     for k, v in seed_styles.items()]

    for ax, name in zip(axes, ["Cheat acc.", "Non-cheat acc."]):
        ax.set_xlabel(r"Training step (${\times}10^3$)")
        ax.set_ylabel(f"{name} (%)")
        ax.set_title(name)
        ax.grid(alpha=0.14, linewidth=0.5)
        ax.set_ylim(-2, 105)

    # Shared legend above figure: variant colors + seed styles
    variant_handles, variant_labels = axes[0].get_legend_handles_labels()
    all_handles = variant_handles + style_handles
    all_labels = variant_labels + [h.get_label() for h in style_handles]
    fig.legend(all_handles, all_labels, loc="upper center",
               ncol=min(len(all_handles), 5), bbox_to_anchor=(0.5, 1.08),
               frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_dann_adv_cont_loss(outname):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8))

    # DANN λ=0.05 Adv Acc
    files = find_seed_files("dann_l05")
    _report_seeds("dann_l05 (adv)", files)
    parsed = [parse_dann_log(p) for p in files.values() if p]
    per_seed = [{"s": d["steps"], "v": d["adv"]} for d in parsed]
    steps, med, q1, q3 = aggregate(per_seed, "s", "v")
    if steps is not None:
        x = steps / 1000.0
        axes[0].plot(x, med, color=PALETTE["dann"], label="DANN $\\lambda$=0.05")
        if np.any(q3 > q1):
            axes[0].fill_between(x, q1, q3, color=PALETTE["dann"], alpha=0.14, linewidth=0)
    axes[0].axhline(50, **HLINE_KW)
    axes[0].set_xlabel(r"Training step (${\times}10^3$)")
    axes[0].set_ylabel("Adv. accuracy (%)")
    axes[0].set_title("DANN adversarial")
    axes[0].set_ylim(-2, 105)
    axes[0].grid(alpha=0.14, linewidth=0.5)

    # Paired contrastive loss
    files = find_seed_files("cont_l1")
    _report_seeds("cont_l1 (loss)", files)
    parsed = [parse_contrastive_log(p) for p in files.values() if p]
    per_seed = [{"s": d["steps"], "v": d["cont_loss"]} for d in parsed]
    steps, med, q1, q3 = aggregate(per_seed, "s", "v")
    if steps is not None:
        x = steps / 1000.0
        axes[1].plot(x, med, color=PALETTE["augmentation"],
                     label="Paired $\\lambda$=1")
        if np.any(q3 > q1):
            axes[1].fill_between(x, np.maximum(q1, 1e-9), q3,
                                 color=PALETTE["augmentation"], alpha=0.14, linewidth=0)
    axes[1].set_xlabel(r"Training step (${\times}10^3$)")
    axes[1].set_ylabel("Contrastive loss (MSE)")
    axes[1].set_title("Paired contrastive")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.14, linewidth=0.5)

    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def _group_buckets(group_name):
    """Map a group name like '357' or '468_later' to its constituent max_remove ints."""
    digits = re.findall(r"\d", group_name)
    return [int(d) for d in digits]


def plot_transition_eval_3panels(direction, first, second, title, outname):
    """3 panels, ALL showing held-out eval accuracy (not train).
    Panel 1: all buckets (mr=3..8).  Panel 2: buckets in `first`.  Panel 3: buckets in `second`.
    Use this instead of plot_transition_3panels when you want eval-only views."""
    files = find_seed_files(f"trans_{direction}")
    _report_seeds(f"trans_{direction} (eval-3panel)", files)
    parsed = [parse_transition_log(p, first, second) for p in files.values() if p]
    if not parsed:
        print(f"  skipping {outname}: no files")
        return

    buckets = list(range(3, 9))
    first_set = set(_group_buckets(first))
    second_set = set(_group_buckets(second))

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.8), sharey=True)

    panel_configs = [
        ("All mr", set(buckets)),
        (f"mr $\\in$ {sorted(first_set)}",  first_set),
        (f"mr $\\in$ {sorted(second_set)}", second_set),
    ]
    panel_letters = ["(a)", "(b)", "(c)"]

    for idx, (panel_title, keep) in enumerate(panel_configs):
        ax = axes[idx]
        for ci, k in enumerate(buckets):
            if k not in keep:
                continue
            per_seed = [{"s": d["steps_eval"], "v": d["eval"][k]} for d in parsed]
            steps, med, q1, q3 = aggregate(per_seed, "s", "v")
            if steps is None or np.max(med) < 1.0:
                continue
            x = steps / 1000.0
            ax.plot(x, med, color=BUCKET_COLORS[ci], label=f"mr={k}")
            if np.any(q3 > q1):
                ax.fill_between(x, q1, q3, color=BUCKET_COLORS[ci],
                                alpha=0.14, linewidth=0)
            ax.axhline(100 / (k + 1), color=BUCKET_COLORS[ci],
                       lw=HLINE_KW["lw"], ls=HLINE_KW["ls"], alpha=0.35, zorder=0)
        ax.axvline(TRANSITION_STEP / 1000.0, **HLINE_KW)
        ax.set_xlabel(r"Training step (${\times}10^3$)")
        if idx == 0:
            ax.set_ylabel("Held-out eval acc. (%)")
        ax.set_title(panel_title)
        ax.set_ylim(-2, 105)
        ax.grid(alpha=0.14, linewidth=0.5)
        panel_label(ax, panel_letters[idx])

    # Shared bucket legend above figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(buckets),
               bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_transition_eval_sidebyside(outname):
    """Two-panel eval-only comparison: 357 -> 468_later (a) vs 468 -> 357_later (b).
    Each panel shows held-out eval accuracy per max_remove bucket, median + IQR."""
    configs = [
        ("357 $\\rightarrow$ 468_later", "357_468l", "357", "468_later"),
        ("468 $\\rightarrow$ 357_later", "468_357l", "468", "357_later"),
    ]
    buckets = list(range(3, 9))

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), sharey=True)
    panel_letters = ["(a)", "(b)"]

    for idx, (label, direction, first, second) in enumerate(configs):
        ax = axes[idx]
        files = find_seed_files(f"trans_{direction}")
        _report_seeds(f"trans_{direction} (eval-only)", files)
        parsed = [parse_transition_log(p, first, second) for p in files.values() if p]
        if not parsed:
            ax.set_title(label + " (no data)")
            continue

        for ci, k in enumerate(buckets):
            per_seed = [{"s": d["steps_eval"], "v": d["eval"][k]} for d in parsed]
            steps, med, q1, q3 = aggregate(per_seed, "s", "v")
            if steps is None or np.max(med) < 1.0:
                continue
            x = steps / 1000.0
            ax.plot(x, med, color=BUCKET_COLORS[ci], label=f"mr={k}")
            if np.any(q3 > q1):
                ax.fill_between(x, q1, q3, color=BUCKET_COLORS[ci],
                                alpha=0.14, linewidth=0)
            ax.axhline(100 / (k + 1), color=BUCKET_COLORS[ci],
                       lw=HLINE_KW["lw"], ls=HLINE_KW["ls"], alpha=0.35, zorder=0)
        ax.axvline(TRANSITION_STEP / 1000.0, **HLINE_KW)
        ax.set_xlabel(r"Training step (${\times}10^3$)")
        if idx == 0:
            ax.set_ylabel("Held-out eval acc. (%)")
        ax.set_ylim(-2, 105)
        ax.grid(alpha=0.14, linewidth=0.5)
        panel_label(ax, panel_letters[idx])

    # Legend above the figure so it doesn't overlap curves
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(buckets),
               bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_transition_direction_compare(outname):
    """T3: overlay both directions on one panel, mean-over-all-buckets eval acc."""
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    configs = [
        ("357 $\\rightarrow$ 468_later", "357_468l", "357", "468_later", PALETTE["baseline"]),
        ("468 $\\rightarrow$ 357_later", "468_357l", "468", "357_later", PALETTE["augmentation"]),
    ]
    for label, direction, first, second, color in configs:
        files = find_seed_files(f"trans_{direction}")
        parsed = [parse_transition_log(p, first, second) for p in files.values() if p]
        if not parsed:
            continue
        per_seed = []
        for d in parsed:
            if not d["steps_eval"]:
                continue
            bucket_mat = np.array([d["eval"][k] for k in range(3, 9)])
            # Inner collapse: median across buckets per seed (matches outer seed aggregation)
            per_seed.append({"s": d["steps_eval"], "v": np.median(bucket_mat, axis=0).tolist()})
        if not per_seed:
            continue
        steps, med, q1, q3 = aggregate(per_seed, "s", "v")
        if steps is None:
            continue
        x = steps / 1000.0
        ax.plot(x, med, color=color, label=label)
        if np.any(q3 > q1):
            ax.fill_between(x, q1, q3, color=color, alpha=0.14, linewidth=0)
    ax.axvline(TRANSITION_STEP / 1000.0, **HLINE_KW)
    ax.set_xlabel(r"Training step (${\times}10^3$)")
    # Both aggregations are median: median-over-buckets per seed, then median + IQR across seeds.
    ax.set_ylabel("Median eval acc. (%)")
    ax.set_ylim(-2, 105)
    ax.grid(alpha=0.14, linewidth=0.5)
    # Legend above figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_transition_perseed(direction, first, second, title, outname):
    """T1/T2 replacement: 3 panels × 6 buckets, per-seed thin lines (no std band)."""
    files = find_seed_files(f"trans_{direction}")
    _report_seeds(f"trans_{direction} (per-seed)", files)
    parsed_by_seed = [(s, parse_transition_log(p, first, second))
                      for s, p in files.items() if p]
    if not parsed_by_seed:
        return
    n_seeds = len(parsed_by_seed)

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.8), sharey=True)
    buckets = list(range(3, 9))
    style_cycle = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2))]
    seeds_sorted = sorted(s for s, _ in parsed_by_seed)
    seed_styles = {s: style_cycle[i % len(style_cycle)] for i, s in enumerate(seeds_sorted)}

    panels = [
        ("Eval (held-out 345678)", "steps_eval", "eval"),
        (f"Train on {first}", "steps_tf", "train_first"),
        (f"Train on {second}", "steps_ts", "train_second"),
    ]
    panel_letters = ["(a)", "(b)", "(c)"]
    for idx, (ax, (panel_title, skey, bkey)) in enumerate(zip(axes, panels)):
        for seed, d in parsed_by_seed:
            steps = d[skey]
            if not steps:
                continue
            x = np.array(steps) / 1000.0
            for ci, k in enumerate(buckets):
                vals = d[bkey][k]
                if max(vals) < 1.0:
                    continue
                ls = seed_styles.get(seed, "-")
                ax.plot(x, vals, color=BUCKET_COLORS[ci], linestyle=ls,
                        linewidth=1.0, alpha=0.85)
        ax.axvline(TRANSITION_STEP / 1000.0, **HLINE_KW)
        ax.set_xlabel(r"Training step (${\times}10^3$)")
        if idx == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(-2, 105)
        ax.grid(alpha=0.14, linewidth=0.5)
        panel_label(ax, panel_letters[idx])

    # Shared legend above figure (buckets + seed styles)
    style_handles = [plt.Line2D([], [], color="#888888", linestyle=v, label=f"seed {k}")
                     for k, v in seed_styles.items()]
    bucket_handles = [plt.Line2D([], [], color=BUCKET_COLORS[ci], label=f"mr={k}")
                      for ci, k in enumerate(buckets)]
    all_handles = bucket_handles + style_handles
    fig.legend(handles=all_handles, loc="upper center",
               ncol=len(bucket_handles), bbox_to_anchor=(0.5, 1.08),
               frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(OUTPUT_DIR, outname)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


if __name__ == "__main__":
    print("=== Transition ===")
    plot_transition_3panels("357_468l", "357", "468_later",
                            "Transition 357 → 468_later ({n} seeds, median + IQR)",
                            "transition_357_468later_medianiqr.png")
    plot_transition_3panels("468_357l", "468", "357_later",
                            "Transition 468 → 357_later ({n} seeds, median + IQR)",
                            "transition_468_357later_medianiqr.png")
    print("\n=== Transition per-seed (T1/T2 alt) ===")
    plot_transition_perseed("357_468l", "357", "468_later",
                            "Transition 357 → 468_later ({n} seeds, per-seed)",
                            "transition_357_468later_perseed.png")
    plot_transition_perseed("468_357l", "468", "357_later",
                            "Transition 468 → 357_later ({n} seeds, per-seed)",
                            "transition_468_357later_perseed.png")
    print("\n=== Transition eval-only 3 panels (replaces train panels) ===")
    plot_transition_eval_3panels("357_468l", "357", "468_later",
                                 "Transition 357 → 468_later eval-only",
                                 "transition_357_468later_eval3.png")
    plot_transition_eval_3panels("468_357l", "468", "357_later",
                                 "Transition 468 → 357_later eval-only",
                                 "transition_468_357later_eval3.png")
    print("\n=== Transition eval side-by-side ===")
    plot_transition_eval_sidebyside("transition_eval_sidebyside.png")
    print("\n=== Transition direction comparison (T3) ===")
    plot_transition_direction_compare("transition_direction_compare.png")
    print("\n=== 3-method comparison ===")
    # n=3 for baseline & DANN, n=5 for contrastive-only — caption carries the detail
    plot_comparison_3methods("comparison_3methods.png")
    print("\n=== Paired vs no-paired ===")
    plot_paired_vs_nopaired("paired_vs_nopaired_3seeds.png")
    print("\n=== 3 contrastive variants ===")
    plot_contrastive_3way("contrastive_3way_perseed.png")
    print("\n=== DANN adv + contrastive loss ===")
    plot_dann_adv_cont_loss("dann_adv_contrastive_loss_3seeds.png")
    print("\nDone.")
