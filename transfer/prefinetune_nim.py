"""Section 4.2 — Nim prefinetune (`Nim_k`) as a downstream initializer.

PLACEHOLDER / STUB.  >>> LEO: drop your implementation here. <<<

Goal
----
Select a converged Nim checkpoint for a *source* task MR = k (the first
checkpoint that reaches 100% eval accuracy during the standard
`finetune_single_mr.py` run) and expose it as the base model for a downstream
Nim task whose modulus is a multiple of (k + 1). The downstream finetune then
runs via the existing `finetune_single_mr.py`.

Expected CLI (suggested):
    python transfer/prefinetune_nim.py <source_mr> <downstream_mr> <seed>

This script is mostly bookkeeping: resolve the source checkpoint repo/revision,
then hand it to the downstream finetune as its base model.
"""

import os
import sys

# Make the repo root importable so `from config import ...` works from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_ORG  # noqa: F401


def main():
    raise NotImplementedError(
        "Section 4.2 Nim_k prefinetune wiring is not in this snapshot. "
        "Leo: implement source-checkpoint selection + downstream handoff here. "
        "See transfer/README.md."
    )


if __name__ == "__main__":
    main()
