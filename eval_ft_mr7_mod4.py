"""Re-evaluate ft_mr7 checkpoints for move / mod-8 / mod-4 accuracy.

Applies the -1 → 0 fix (loss-position convention): pred_eff = 0 if pred == -1.
Output is one JSONL row per (run, checkpoint) for easy tabling / plotting.

Usage:
    python -u eval_ft_mr7_mod4.py

Resumable: if OUT_PATH exists, already-evaluated (run_name, step) pairs are
skipped. Safe to interrupt and re-launch.
"""
import json
import os
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import HF_ORG, data_path, results_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_FILE = data_path("7_eval.jsonl")
MAX_LENGTH = 128
BATCH_SIZE = 64
MAX_REMOVE = 7
MODULUS = MAX_REMOVE + 1  # 8 — correct Nim modulus for max_remove=7
SAVE_STEPS = list(range(10000, 150001, 10000))  # 10k..150k
OUT_PATH = results_path("ft_mr7_eval.jsonl")
os.makedirs(results_path(), exist_ok=True)

# All 9 runs (3 seeds × 3 sizes). Flip to True/False to gate individual ones.
RUNS = [
    ("70m",  42,  f"{HF_ORG}/ft_mr7_70m_seed42_v3"),
    ("70m",  123, f"{HF_ORG}/ft_mr7_70m_seed123_v3"),
    ("70m",  456, f"{HF_ORG}/ft_mr7_70m_seed456_v3"),
    ("160m", 42,  f"{HF_ORG}/ft_mr7_160m_seed42_v3"),
    ("160m", 123, f"{HF_ORG}/ft_mr7_160m_seed123_v3"),
    ("160m", 456, f"{HF_ORG}/ft_mr7_160m_seed456_v3"),
    ("410m", 42,  f"{HF_ORG}/ft_mr7_410m_seed42_v3"),
    ("410m", 123, f"{HF_ORG}/ft_mr7_410m_seed123_v3"),
    ("410m", 456, f"{HF_ORG}/ft_mr7_410m_seed456_v3"),
]

INIT_PILE_RE = re.compile(r"There are (\d+) coins")
PROMPT_MOVE_RE = re.compile(r"take (\d+) coin", re.IGNORECASE)
ANSWER_MOVE_RE = re.compile(r"take\s+(-?\d+)\s+coin", re.IGNORECASE)
INT_RE = re.compile(r"-?\d+")


def compute_current_pile(prompt):
    m = INIT_PILE_RE.search(prompt)
    if not m:
        return None
    initial = int(m.group(1))
    used = sum(int(x) for x in PROMPT_MOVE_RE.findall(prompt))
    return initial - used


def extract_move(text):
    m = ANSWER_MOVE_RE.search(text)
    if m:
        return int(m.group(1))
    m = INT_RE.search(text)
    return int(m.group(0)) if m else None


def load_eval():
    with open(EVAL_FILE) as f:
        return [json.loads(l) for l in f]


def tokenize_and_mask(tokenizer, example):
    full_text = example["prompt"] + example["answer"]
    tokens = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    prompt_ids = tokenizer(example["prompt"], truncation=True, max_length=MAX_LENGTH, padding=False)["input_ids"]
    labels = list(tokens["input_ids"])
    for j in range(min(len(prompt_ids), MAX_LENGTH)):
        labels[j] = -100
    tokens["labels"] = labels
    return tokens


def evaluate(model, tokenizer, eval_data, piles):
    """Teacher-forced eval. Returns {move_acc, mod8_acc, mod4_acc, n}."""
    model.eval()
    move_hits = mod8_hits = mod4_hits = 0
    total = 0

    tokenized = [tokenize_and_mask(tokenizer, ex) for ex in eval_data]
    for start in range(0, len(tokenized), BATCH_SIZE):
        batch = tokenized[start:start + BATCH_SIZE]
        input_ids = torch.tensor([b["input_ids"] for b in batch], device=DEVICE)
        attention_mask = torch.tensor([b["attention_mask"] for b in batch], device=DEVICE)
        labels = np.array([b["labels"] for b in batch])

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_ids = logits.argmax(dim=-1).cpu().numpy()

        for i, (pred, lbl) in enumerate(zip(pred_ids, labels)):
            global_i = start + i
            mask = lbl != -100
            if mask.sum() == 0:
                continue
            p_ans = pred[mask]
            l_ans = lbl[mask]
            pred_text = tokenizer.decode(p_ans, skip_special_tokens=True).strip().lower()
            gold_text = tokenizer.decode(l_ans, skip_special_tokens=True).strip().lower()
            pred_move = extract_move(pred_text)
            gold_move = extract_move(gold_text)
            pile = piles[global_i]

            total += 1
            if pred_move is not None and gold_move is not None and pred_move == gold_move:
                move_hits += 1

            if pred_move is None or pile is None:
                continue
            pred_eff = 0 if pred_move == -1 else pred_move
            remaining = pile - pred_eff
            if remaining % MODULUS == 0:
                mod8_hits += 1
            if remaining % 4 == 0:
                mod4_hits += 1

    if total == 0:
        return {"move_acc": 0.0, "mod8_acc": 0.0, "mod4_acc": 0.0, "n": 0}
    return {
        "move_acc": move_hits / total,
        "mod8_acc": mod8_hits / total,
        "mod4_acc": mod4_hits / total,
        "n": total,
    }


def load_done_keys():
    done = set()
    if not os.path.exists(OUT_PATH):
        return done
    with open(OUT_PATH) as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add((row["size"], row["seed"], row["step"]))
            except Exception:
                continue
    return done


def append_row(row):
    with open(OUT_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    print("Loading eval data...")
    eval_data = load_eval()
    piles = [compute_current_pile(ex["prompt"]) for ex in eval_data]
    print(f"  {len(eval_data)} eval examples, {sum(p is not None for p in piles)} with valid pile")
    print(f"  max_remove={MAX_REMOVE}, modulus={MODULUS}")

    done = load_done_keys()
    if done:
        print(f"Resuming — {len(done)} (size, seed, step) entries already done")

    for size, seed, repo in RUNS:
        print(f"\n=== {size}_s{seed} ({repo}) ===")
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo, revision=f"step-{SAVE_STEPS[-1]}")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"  SKIP — tokenizer load failed: {e}")
            continue

        for step in SAVE_STEPS:
            if (size, seed, step) in done:
                continue
            revision = f"step-{step}"
            try:
                model = AutoModelForCausalLM.from_pretrained(repo, revision=revision).to(DEVICE)
            except Exception as e:
                print(f"  step-{step}: MISSING ({e.__class__.__name__})")
                continue

            metrics = evaluate(model, tokenizer, eval_data, piles)
            row = {"size": size, "seed": seed, "step": step,
                   "repo": repo, **metrics}
            append_row(row)
            print(f"  step-{step}: move={metrics['move_acc']:.4f}  "
                  f"mod8={metrics['mod8_acc']:.4f}  mod4={metrics['mod4_acc']:.4f}  "
                  f"n={metrics['n']}")
            del model
            torch.cuda.empty_cache()

    print(f"\nAll done. JSONL → {OUT_PATH}")


if __name__ == "__main__":
    main()
