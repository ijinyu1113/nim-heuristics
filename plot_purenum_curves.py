"""Aggregate purenum metrics across seeds and plot move-acc curves.

Inputs (under $NIM_RESULTS_DIR, default results/):
  - JSONL files from finetune_single_mr_purenum.py runs:
        results/purenum_metrics/mr{MR}_410m_seed{SEED}.jsonl
    rows: training-loss row + eval_eval_* row + eval_train_* row at each eval step
  - Leo's CSV (seed 42 across all mr):
        results/purenum_metrics/icml2026 - 410M.csv
    18 columns: 6 mr groups × (step, train_acc, eval_acc)

Output:
  - results/purenum_metrics/purenum_combined.csv  (long format, one row
    per (mr, seed, step) with step / train_acc / eval_acc)
  - results/plots/purenum_train_curves.png
  - results/plots/purenum_eval_curves.png
Both plots share style: median + IQR across seeds, one line per mr, paper style.
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_style import setup_style, panel_label, HLINE_KW
from config import results_path

setup_style()

METRICS_DIR = results_path("purenum_metrics")
OUT_PLOT_DIR = results_path("plots")
LONG_CSV_PATH = os.path.join(METRICS_DIR, "purenum_combined.csv")
CSV_PATH = os.path.join(METRICS_DIR, "icml2026 - 410M.csv")
CSV_SEED = 42  # Leo's CSV is seed 42 throughout
CSV_FILL_STEP_END = 70500   # match JSONL final step
CSV_FILL_STEP_INTERVAL = 500
CSV_GROK_THRESHOLD = 0.95   # only forward-fill if final eval_acc >= this

MRS = [3, 4, 5, 6, 7, 8]
BUCKET_COLORS = plt.cm.tab10(range(len(MRS)))


# ---------- Loading ----------

def load_jsonl_run(path):
    """Return a dict {step: {'train_acc': float|None, 'eval_acc': float|None}}.
    The script writes 3 rows per logging step: training-loss, eval_eval_*, eval_train_*."""
    by_step = defaultdict(lambda: {"train_acc": None, "eval_acc": None})
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            step = row.get("step")
            if step is None:
                continue
            if "eval_eval_move_acc" in row:
                by_step[step]["eval_acc"] = float(row["eval_eval_move_acc"])
            if "eval_train_move_acc" in row:
                by_step[step]["train_acc"] = float(row["eval_train_move_acc"])
    return dict(by_step)


def parse_jsonl_filename(path):
    """mr{MR}_{size}_seed{SEED}.jsonl -> (mr, seed)."""
    m = re.match(r"mr(\d+)_\w+_seed(\d+)\.jsonl$", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def load_csv_runs(path):
    """Leo's CSV: 6 mr groups × (step, train_acc, eval_acc).
    Returns {mr: {step: {'train_acc': float|None, 'eval_acc': float|None}}}.
    Empty cells -> None."""
    out = {mr: {} for mr in MRS}
    with open(path, "r") as f:
        reader = list(csv.reader(f))
    if len(reader) < 2:
        return out
    header = reader[0]
    # Each mr occupies 3 consecutive columns starting at indices 0,3,6,...
    mr_cols = {}
    for i in range(0, len(header), 3):
        if i >= len(header):
            break
        try:
            mr_label = int(header[i].strip())
        except ValueError:
            continue
        mr_cols[mr_label] = i  # column with step

    for row in reader[1:]:
        for mr, col in mr_cols.items():
            if mr not in out:
                continue
            try:
                step = row[col].strip() if col < len(row) else ""
                if not step:
                    continue
                step = int(float(step))
            except (ValueError, IndexError):
                continue
            train_raw = row[col + 1].strip() if col + 1 < len(row) else ""
            eval_raw = row[col + 2].strip() if col + 2 < len(row) else ""
            train_v = float(train_raw) if train_raw else None
            eval_v = float(eval_raw) if eval_raw else None
            entry = out[mr].setdefault(step, {"train_acc": None, "eval_acc": None})
            if train_v is not None:
                entry["train_acc"] = train_v
            if eval_v is not None:
                entry["eval_acc"] = eval_v
    return out


# ---------- Merge to long-format DataFrame ----------

def gather_long():
    rows = []
    # JSONL runs (each file is one (mr, seed))
    for path in sorted(glob.glob(os.path.join(METRICS_DIR, "mr*_410m_seed*.jsonl"))):
        parsed = parse_jsonl_filename(path)
        if not parsed:
            continue
        mr, seed = parsed
        if mr not in MRS:
            continue
        run = load_jsonl_run(path)
        for step, vals in run.items():
            rows.append({
                "mr": mr, "seed": seed, "step": step,
                "train_acc": vals["train_acc"], "eval_acc": vals["eval_acc"],
                "source": "jsonl",
            })
    # CSV (seed 42)
    if os.path.isfile(CSV_PATH):
        csv_runs = load_csv_runs(CSV_PATH)
        for mr, by_step in csv_runs.items():
            if not by_step:
                continue
            # Real datapoints first
            for step, vals in by_step.items():
                rows.append({
                    "mr": mr, "seed": CSV_SEED, "step": step,
                    "train_acc": vals["train_acc"], "eval_acc": vals["eval_acc"],
                    "source": "csv",
                })
            # Forward-fill with 1.0 if the run grokked by its last datapoint
            last_step = max(by_step.keys())
            last_eval = by_step[last_step]["eval_acc"]
            last_train = by_step[last_step]["train_acc"]
            grokked = last_eval is not None and last_eval >= CSV_GROK_THRESHOLD
            if grokked:
                for step in range(last_step + CSV_FILL_STEP_INTERVAL,
                                  CSV_FILL_STEP_END + 1, CSV_FILL_STEP_INTERVAL):
                    rows.append({
                        "mr": mr, "seed": CSV_SEED, "step": step,
                        "train_acc": 1.0,
                        "eval_acc": 1.0,
                        "source": "csv_fill",
                    })
                print(f"  csv mr={mr}: filled steps {last_step + CSV_FILL_STEP_INTERVAL}..{CSV_FILL_STEP_END} with 1.0 (final eval={last_eval:.3f})")
            else:
                print(f"  csv mr={mr}: NOT filled (final eval={last_eval}, below {CSV_GROK_THRESHOLD})")
    df = pd.DataFrame(rows)
    df = df.sort_values(["mr", "seed", "step"]).reset_index(drop=True)
    return df


def aggregate_per_step(df, metric, step_multiple=500):
    """For each (mr, step), aggregate `metric` across seeds. Returns DataFrame
    with columns mr, step, n, median, q1, q3 over the subset where metric is not NaN.
    Only keeps steps that are multiples of `step_multiple` so sample size stays
    consistent across the plotted grid.

    Rationale:
      - eval_acc: CSV @ 250-step cadence, JSONL @ 500; use 500 (JSONL grid).
      - train_acc: CSV @ 1000-step cadence, JSONL @ 500; use 1000 so seed 42
        contributes at every plotted step (avoids n=2 vs n=3 alternation)."""
    sub = df[["mr", "seed", "step", metric]].dropna(subset=[metric])
    sub = sub[sub["step"] % step_multiple == 0]
    g = sub.groupby(["mr", "step"])[metric]
    summary = g.agg(
        n="count",
        median="median",
        q1=lambda x: np.percentile(x, 25),
        q3=lambda x: np.percentile(x, 75),
    ).reset_index()
    return summary


# ---------- Plot ----------

def plot_metric(agg, metric, title_metric, outpath):
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for ci, mr in enumerate(MRS):
        sub = agg[agg["mr"] == mr]
        if sub.empty:
            continue
        x = sub["step"].to_numpy() / 1000.0
        med = sub["median"].to_numpy()
        q1 = sub["q1"].to_numpy()
        q3 = sub["q3"].to_numpy()
        ax.plot(x, med, color=BUCKET_COLORS[ci], label=f"mr={mr}")
        if np.any(q3 > q1):
            ax.fill_between(x, q1, q3, color=BUCKET_COLORS[ci],
                            alpha=0.14, linewidth=0)
        # Chance line per mr
        ax.axhline(1.0 / (mr + 1), color=BUCKET_COLORS[ci],
                   lw=HLINE_KW["lw"], ls=HLINE_KW["ls"], alpha=0.35, zorder=0)
    ax.set_xlabel(r"Training step (${\times}10^3$)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.05)
    # Linear up to step 10k, log beyond, cap at 35k
    ax.set_xscale("symlog", linthresh=10, linscale=1)
    ax.set_xlim(0, 35)
    ax.set_xticks([0, 5, 10, 15, 20, 30])
    ax.set_xticklabels(["0", "5", "10", "15", "20", "30"])
    ax.grid(alpha=0.14, linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MRS),
               bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def main():
    df = gather_long()
    if df.empty:
        print("No data found.")
        return
    df.to_csv(LONG_CSV_PATH, index=False)
    print(f"Wrote long-format data to {LONG_CSV_PATH} ({len(df)} rows)")

    # Quick coverage report
    print("\nSeed coverage per mr:")
    coverage = df.groupby("mr")["seed"].apply(lambda s: sorted(set(s))).reset_index()
    for _, row in coverage.iterrows():
        print(f"  mr={row['mr']}: seeds={row['seed']} (n={len(row['seed'])})")

    eval_agg = aggregate_per_step(df, "eval_acc", step_multiple=500)
    train_agg = aggregate_per_step(df, "train_acc", step_multiple=1000)

    plot_metric(eval_agg,  "eval_acc",  "Eval move",  os.path.join(OUT_PLOT_DIR, "purenum_eval_curves.png"))
    plot_metric(train_agg, "train_acc", "Train move", os.path.join(OUT_PLOT_DIR, "purenum_train_curves.png"))


if __name__ == "__main__":
    main()
