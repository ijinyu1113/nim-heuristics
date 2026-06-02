"""Section 4.2 — Transfer finetune (Mod_k / Nim_k prefinetune AND downstream installation).

This is the §4.2 workhorse. It is an ordinary supervised finetune that simply takes an
input dataset and an (optional) starting checkpoint — the *same* finetune used elsewhere,
just pointed at different data:

  * Prefinetune (Mod_k): start from base Pythia, train on a "what is n mod k?" dataset
    (generate it with make_modk_data.py). Produces the base checkpoint for installation.
  * Prefinetune (Nim_k): equivalently, ../finetune_single_mr.py on the source Nim task.
  * Installation (downstream): start from a prefinetuned checkpoint, train on the
    downstream Nim task, and log coset accuracy (mod-2 / mod-3) over training — the curve
    that reveals the installed coset-heuristic plateau.

Usage:
    python transfer/heuristic_installation.py <seed> <run_name> [dataset] [base_model]

    <seed>       random seed (default 1)
    <run_name>   label for the local output-checkpoint dir (default "install")
    [dataset]    dataset stem under <DATA_DIR>: reads <DATA_DIR>/<dataset>_{train,eval}.jsonl
                 e.g. "5" (Nim MR=5)  or  "mod3" (modular reduction).  Default "5".
    [base_model] starting checkpoint. Default base Pythia (= a from-scratch prefinetune);
                 pass a prefinetuned checkpoint (or set NIM_PREFINETUNE_MODEL) to do
                 downstream installation.

Examples:
    # 1) prefinetune mod-3 from scratch  ->  pushes/saves a base checkpoint
    python transfer/heuristic_installation.py 1 mod3_prefinetune mod3
    # 2) install that heuristic into downstream MR=5
    python transfer/heuristic_installation.py 1 mod3_into_mr5 5 EleutherAI/pythia-410m-deduped
"""

import os
import sys
import re
import json

import numpy as np
from datasets import Dataset
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

# Make the repo root importable so `from config import ...` works from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import data_path, output_path

# --- CONFIGURATION ---
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
RUN_NAME = sys.argv[2] if len(sys.argv) > 2 else "install"
# Dataset stem: reads <DATA_DIR>/<DATASET>_{train,eval}.jsonl.
# "5" = Nim MR=5,  "mod3" = mod-3 reduction (made by make_modk_data.py).
DATASET = sys.argv[3] if len(sys.argv) > 3 else "5"
# Starting checkpoint: base Pythia for a from-scratch prefinetune, or a prefinetuned
# (Mod_k / Nim_k) checkpoint for downstream installation.
BASE_MODEL = (
    sys.argv[4]
    if len(sys.argv) > 4
    else os.environ.get("NIM_PREFINETUNE_MODEL", "EleutherAI/pythia-410m-deduped")
)
set_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

train_file = data_path(f"{DATASET}_train.jsonl")
eval_file = data_path(f"{DATASET}_eval.jsonl")
max_length = 128


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize_and_mask(example):
    full_text = example["prompt"] + example["answer"]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    prompt_token_ids = tokenizer(
        example["prompt"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )["input_ids"]
    prompt_len = len(prompt_token_ids)

    labels = tokenized["input_ids"].copy()
    for i in range(prompt_len):
        if i < max_length:
            labels[i] = -100

    tokenized["labels"] = labels
    return tokenized


train_data = read_jsonl(train_file)
eval_data = read_jsonl(eval_file)

train_dataset = Dataset.from_list(train_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])
eval_dataset = Dataset.from_list(eval_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])


MOVE_RE = re.compile(r"take\s+(-?\d+)\s+coin", re.IGNORECASE)
INT_RE = re.compile(r"-?\d+")


def extract_move(text):
    m = MOVE_RE.search(text)
    if m:
        return int(m.group(1))
    m = INT_RE.search(text)
    return int(m.group(0)) if m else None


def normalize_move(move):
    """Treat -1 (losing sentinel) as 0 before mod-k comparison."""
    if move is None:
        return None
    return 0 if move == -1 else move


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    pred_ids, label_ids = eval_pred
    pred_ids = np.asarray(pred_ids)
    label_ids = np.asarray(label_ids)

    mask = label_ids != -100

    correct_tokens = ((pred_ids == label_ids) & mask).sum()
    total_tokens = mask.sum()
    token_acc = float(correct_tokens / total_tokens) if total_tokens > 0 else 0.0

    seq_matches = []
    move_matches = []
    mod2_matches = []
    mod3_matches = []

    for p, l, m in zip(pred_ids, label_ids, mask):
        p_ans = p[m]
        l_ans = l[m]

        if l_ans.size == 0:
            seq_matches.append(False)
            move_matches.append(False)
            mod2_matches.append(False)
            mod3_matches.append(False)
            continue

        seq_matches.append(bool(np.array_equal(p_ans, l_ans)))

        pred_text = tokenizer.decode(p_ans, skip_special_tokens=True).strip().lower()
        gold_text = tokenizer.decode(l_ans, skip_special_tokens=True).strip().lower()

        pred_move = extract_move(pred_text)
        gold_move = extract_move(gold_text)
        move_matches.append(pred_move is not None and gold_move is not None and pred_move == gold_move)

        pred_norm = normalize_move(pred_move)
        gold_norm = normalize_move(gold_move)

        mod2_matches.append(
            pred_norm is not None and gold_norm is not None
            and pred_norm % 2 == gold_norm % 2
        )
        mod3_matches.append(
            pred_norm is not None and gold_norm is not None
            and pred_norm % 3 == gold_norm % 3
        )

    seq_acc = float(np.mean(seq_matches)) if seq_matches else 0.0
    move_acc = float(np.mean(move_matches)) if move_matches else 0.0
    mod2_acc = float(np.mean(mod2_matches)) if mod2_matches else 0.0
    mod3_acc = float(np.mean(mod3_matches)) if mod3_matches else 0.0

    return {
        "token_acc": token_acc,
        "seq_acc": seq_acc,
        "move_acc": move_acc,
        "mod2_acc": mod2_acc,
        "mod3_acc": mod3_acc,
    }


training_args = TrainingArguments(
    output_dir=output_path(f"{RUN_NAME}_seed{SEED}"),
    seed=SEED,
    data_seed=SEED,
    num_train_epochs=150,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=3e-5,
    weight_decay=0.05,
    warmup_ratio=0.1,
    logging_steps=20000,
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=7500,
    save_total_limit=None,
    save_only_model=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_eval_move_acc",
    greater_is_better=True,
    lr_scheduler_type="cosine",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset={"eval": eval_dataset, "train": train_dataset},
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

if __name__ == "__main__":
    trainer.train()
    print(trainer.evaluate())
