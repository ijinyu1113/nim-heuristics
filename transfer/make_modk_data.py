"""Generate an explicit modular-reduction ("what is n mod k?") dataset for the §4.2
Mod_k prefinetune.

Writes <DATA_DIR>/mod{k}_{train,eval}.jsonl in the same {"prompt", "answer"} format the
finetune scripts consume, so the prefinetune is just an ordinary finetune run:

    python transfer/make_modk_data.py 3
    python transfer/heuristic_installation.py 1 mod3_prefinetune mod3

Reference implementation matching the paper description (inputs n in [0, N_MAX],
generated algorithmically, ~9000 train / 1000 eval). Adjust the prompt wording / range to
match your own prefinetune data if it differs.

Usage:
    python transfer/make_modk_data.py <k> [seed] [n_max] [n_eval]
"""

import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import data_path

K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N_MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
N_EVAL = int(sys.argv[4]) if len(sys.argv) > 4 else 1000

random.seed(SEED)


def make_example(n, k):
    # Trailing space separates prompt from answer (the finetune concatenates prompt+answer).
    return {"prompt": f"what is {n} mod {k}? ", "answer": str(n % k)}


def write_jsonl(path, ns, k):
    with open(path, "w", encoding="utf-8") as f:
        for n in ns:
            f.write(json.dumps(make_example(n, k)) + "\n")


def main():
    ns = list(range(0, N_MAX + 1))
    random.shuffle(ns)
    eval_ns, train_ns = ns[:N_EVAL], ns[N_EVAL:]

    os.makedirs(data_path(), exist_ok=True)
    train_path = data_path(f"mod{K}_train.jsonl")
    eval_path = data_path(f"mod{K}_eval.jsonl")
    write_jsonl(train_path, train_ns, K)
    write_jsonl(eval_path, eval_ns, K)
    print(f"Wrote {len(train_ns)} train / {len(eval_ns)} eval to {train_path} and {eval_path}")


if __name__ == "__main__":
    main()
