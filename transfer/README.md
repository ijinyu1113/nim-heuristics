# `transfer/` — Section 4.2: Heuristic transplanting (prefinetuning)

> **⚠️ LEO — your code goes here.**
>
> This directory is a **placeholder** for the Section 4.2 *coset-heuristic transfer*
> experiments ("installing / accelerating heuristics via prefinetuning"). That code
> lives in Leo's working tree, not in the snapshot this repo was cut from. Drop the
> cleaned scripts into this folder and fill in the two stubs below.

## What this covers (paper §4.2, Figures 4–7, Table 1)

A model is **prefinetuned** on data whose modulus is a *factor* of the downstream
Nim modulus, then finetuned on the downstream Nim task. Two prefinetuning sources:

1. **Explicit modular reduction** (`Mod_k`) — prompts like *"what is 10 mod 4?"*,
   generated algorithmically over inputs in `[0, 10000]` (9000 train / 1000 eval),
   trained ~5000 steps to 100% accuracy.
2. **Nim prefinetune** (`Nim_k`) — initialize from a Nim checkpoint for a source
   task `MR = k` (the first checkpoint that hits 100% eval accuracy).

The downstream finetune itself is **already in this repo** —
[`../finetune_single_mr.py`](../finetune_single_mr.py) (150-epoch / fixed-step
variant). The only missing pieces are (a) generating the modular-reduction data and
(b) running the prefinetune stage that produces the checkpoint the downstream run
starts from.

## Where each piece goes

| File | What Leo should put here |
|------|--------------------------|
| `prefinetune_modular_reduction.py` | Generate `Mod_k` data + train the `Mod_k` prefinetune. Push checkpoint to `{HF_ORG}/mod{k}_prefinetune_...`. |
| `prefinetune_nim.py` | Take a converged `Nim_k` checkpoint and expose it as the prefinetune init for the downstream task. |
| (data) | If you generate modular-reduction JSONL, write it under `$NIM_DATA_DIR/modreduction/` (see [`../config.py`](../config.py)); do **not** commit the data. |

## How it hooks into the existing pipeline

```
prefinetune (Mod_k or Nim_k)  ->  HF checkpoint  ->  finetune_single_mr.py (downstream MR)
                                                      (pass the prefinetune repo as the base model)
```

Use the shared config so paths/accounts stay parameterized:

```python
from config import HF_ORG, data_path, output_path   # repo root is on sys.path when run from there
```

## Also from Leo (not code, but referenced)

`plot_purenum_curves.py` reads a per-seed metrics CSV that originated from Leo's
410M seed-42 runs (`icml2026 - 410M.csv`). Put that (and any sibling
`mr{MR}_410m_seed{SEED}.jsonl`) under `$NIM_RESULTS_DIR/purenum_metrics/` if you
want that figure to regenerate.
