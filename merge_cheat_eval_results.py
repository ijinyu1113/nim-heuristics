"""Merge eval_results_contrastive_extra.json into eval_results_summary.json.

Can run on HPC or locally. Safe to rerun — just overwrites matching (method, seed)
entries with the extras. Other seeds/methods are untouched.

Usage:
    python merge_cheat_eval_results.py
    python merge_cheat_eval_results.py --main results/eval_results_summary.json \
        --extra results/eval_results_contrastive_extra.json \
        --out   results/eval_results_summary_n5.json

(Default paths resolve through config.results_path, i.e. $NIM_RESULTS_DIR.)
"""
import argparse
import json
import os

from config import results_path


def merge(main_path, extra_path, out_path):
    with open(main_path) as f:
        main = json.load(f)
    with open(extra_path) as f:
        extra = json.load(f)

    added = []
    for method, by_seed in extra.items():
        if method not in main:
            main[method] = {}
            print(f"  NEW METHOD added: {method}")
        for seed_key, regime_dict in by_seed.items():
            if seed_key in main[method]:
                print(f"  OVERWRITE: {method} seed={seed_key}")
            else:
                print(f"  ADD:       {method} seed={seed_key}")
            main[method][seed_key] = regime_dict
            added.append((method, seed_key))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(main, f, indent=2)
    print(f"\nMerged {len(added)} (method, seed) entries into {out_path}")

    # Summary of seed coverage per method
    print("\nSeed coverage after merge:")
    for method, by_seed in main.items():
        seeds = sorted(by_seed.keys(), key=lambda s: int(s) if str(s).isdigit() else -1)
        print(f"  {method}: n={len(seeds)}  seeds={seeds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main",  default=results_path("eval_results_summary.json"))
    ap.add_argument("--extra", default=results_path("eval_results_contrastive_extra.json"))
    ap.add_argument("--out",   default=results_path("eval_results_summary.json"))
    args = ap.parse_args()
    merge(args.main, args.extra, args.out)


if __name__ == "__main__":
    main()
