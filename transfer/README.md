# `transfer/` — Section 4.2: Heuristic transplanting

Code for the §4.2 *coset-heuristic transfer* experiments (Figures 4–7, Table 1):
prefinetuning a model on data whose modulus is a **factor** of a downstream Nim modulus
**installs** (or accelerates) a coset heuristic in the downstream task.

## The two-stage pipeline

```
prefinetune (Mod_k or Nim_k)  ->  base checkpoint  ->  install on downstream Nim task
```

Both stages are the **same ordinary finetune** — it just takes an input dataset and an
optional starting checkpoint ([`heuristic_installation.py`](heuristic_installation.py)).
There is no separate "prefinetune" script; you only point the finetune at different data.

### 1. Prefinetune (produce the base checkpoint)

- **`Mod_k`** — explicit modular reduction (*"what is n mod k?"*). Generate the data, then
  finetune base Pythia on it:
  ```bash
  python transfer/make_modk_data.py 3                       # -> <DATA_DIR>/mod3_{train,eval}.jsonl
  python transfer/heuristic_installation.py 1 mod3_prefinetune mod3
  ```
- **`Nim_k`** — finetune on the source Nim task `MR = k`; equivalently use
  [`../finetune_single_mr.py`](../finetune_single_mr.py) and take the first checkpoint that
  reaches 100% eval accuracy.

### 2. Install (downstream finetune + measure)

```bash
python transfer/heuristic_installation.py <seed> <run_name> <downstream_mr> <base_model>
# e.g. install the mod-3 prefinetune into downstream MR=5:
python transfer/heuristic_installation.py 1 mod3_into_mr5 5 <path-or-hub-id-of-prefinetune>
```

It reads `<DATA_DIR>/<dataset>_{train,eval}.jsonl`, finetunes `base_model` for 150 epochs,
and logs `move_acc`, `mod2_acc`, `mod3_acc` every 250 steps — the curves that show the
installed coset plateau before (or instead of) the full rule. Checkpoints are written under
`<NIM_OUTPUT_DIR>/<run_name>_seed<seed>`.

## Files

| File | Role |
|------|------|
| `heuristic_installation.py` | the §4.2 finetune (prefinetune **and** downstream install) |
| `make_modk_data.py` | reference generator for the `Mod_k` "what is n mod k?" dataset |

## Note

`plot_purenum_curves.py` reads a per-seed metrics CSV (`icml2026 - 410M.csv`) from the
§4.2 / purenum runs; place it (and any `mr{MR}_410m_seed{SEED}.jsonl`) under
`<NIM_RESULTS_DIR>/purenum_metrics/` to regenerate that figure.
