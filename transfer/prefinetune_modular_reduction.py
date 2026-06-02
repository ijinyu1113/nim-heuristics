"""Section 4.2 — Explicit modular-reduction prefinetune (`Mod_k`).

PLACEHOLDER / STUB.  >>> LEO: drop your implementation here. <<<

Goal
----
1. Generate an explicit modular-reduction dataset: prompts of the form
   "what is {n} mod {k}?" with the integer answer, for n in [0, 10000],
   split 9000 train / 1000 eval. Generated algorithmically (NOT by an LLM)
   so there is no subliminal-learning confound.
2. Finetune Pythia (from the standard base checkpoint) on it for ~5000 steps,
   by which point it reaches ~100% eval accuracy.
3. Push the resulting checkpoint to the Hub so the downstream Nim finetune
   (`finetune_single_mr.py`) can start from it.

Expected CLI (suggested, match the rest of the repo):
    python transfer/prefinetune_modular_reduction.py <k> <seed> [model_size]

Expected output:
    HF repo  f"{HF_ORG}/mod{k}_prefinetune_{model_size}_seed{seed}"
    written via the same save-to-hub pattern used in finetune_single_mr.py.
"""

import os
import sys

# Make the repo root importable so `from config import ...` works from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_ORG, data_path, output_path  # noqa: F401


def main():
    raise NotImplementedError(
        "Section 4.2 modular-reduction prefinetune is not in this snapshot. "
        "Leo: implement data generation + prefinetune here. See transfer/README.md."
    )


if __name__ == "__main__":
    main()
