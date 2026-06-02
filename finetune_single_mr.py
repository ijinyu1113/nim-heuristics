"""Fine-tune a single max_remove task across multiple Pythia sizes and seeds,
with per-step eval tracking of token / move / mod-(max_remove+1) / mod-4 accuracy.

Usage:
    python finetune_single_mr.py <max_remove> <seed> <model_size> [lr]
Examples:
    python finetune_single_mr.py 7 42 410m
    python finetune_single_mr.py 7 42 160m 3e-5
    python finetune_single_mr.py 7 123 70m
"""
from huggingface_hub import list_repo_refs, HfApi
import json
import re
import sys
import tempfile
import shutil
import numpy as np

from config import HF_ORG, data_path, output_path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, TrainerCallback,
)

# --- CLI ---
MAX_REMOVE = int(sys.argv[1]) if len(sys.argv) > 1 else 7
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
MODEL_SIZE = sys.argv[3] if len(sys.argv) > 3 else "410m"
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 3e-5

MODEL_MAP = {
    "70m":  "EleutherAI/pythia-70m-deduped",
    "160m": "EleutherAI/pythia-160m-deduped",
    "410m": "EleutherAI/pythia-410m-deduped",
    "1b":   "EleutherAI/pythia-1b-deduped",
    "1.4b": "EleutherAI/pythia-1.4b-deduped",
}
if MODEL_SIZE not in MODEL_MAP:
    raise ValueError(f"unknown model_size {MODEL_SIZE}; pick from {list(MODEL_MAP)}")
REPO_ID = MODEL_MAP[MODEL_SIZE]

TRAIN_FILE = data_path(f"{MAX_REMOVE}_train.jsonl")
EVAL_FILE = data_path(f"{MAX_REMOVE}_eval.jsonl")
MAX_LENGTH = 128
MAX_STEPS = 150000
EVAL_EVERY = 2500
LOG_EVERY = 2500
SAVE_EVERY = 10000
BATCH_SIZE = 64

HF_REPO = f"{HF_ORG}/ft_mr{MAX_REMOVE}_{MODEL_SIZE}_seed{SEED}_v3"
OUTPUT_DIR = output_path(f"ft_mr{MAX_REMOVE}_{MODEL_SIZE}_seed{SEED}")

print(f"Config: max_remove={MAX_REMOVE}, seed={SEED}, model={MODEL_SIZE} ({REPO_ID}), lr={LR}")
print(f"HF_REPO={HF_REPO}")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_latest_step_checkpoint(repo):
    all_branches = list_repo_refs(repo).branches
    checkpoints = sorted(
        [b.name for b in all_branches if b.name.startswith("step") and b.name.split("step")[1].isdigit()],
        key=lambda x: int(x.split("step")[1]),
    )
    if not checkpoints:
        raise RuntimeError(f"No numeric step checkpoints found in {repo}")
    return checkpoints[-1]


chosen_ckpt = get_latest_step_checkpoint(REPO_ID)
print(f"Using base checkpoint: {chosen_ckpt}")

tokenizer = AutoTokenizer.from_pretrained(REPO_ID, revision=chosen_ckpt)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(REPO_ID, revision=chosen_ckpt)
model.config.pad_token_id = tokenizer.pad_token_id


# --- Pile state extraction from prompts ---

INIT_PILE_RE = re.compile(r"There are (\d+) coins")
PROMPT_MOVE_RE = re.compile(r"take (\d+) coin", re.IGNORECASE)


def compute_current_pile(prompt):
    """Parse current pile size (before the model's move) from a Nim prompt."""
    m = INIT_PILE_RE.search(prompt)
    if not m:
        return None
    initial = int(m.group(1))
    used = sum(int(x) for x in PROMPT_MOVE_RE.findall(prompt))
    return initial - used


# --- Tokenization (prompt tokens masked with -100) ---

def tokenize_and_mask(example):
    full_text = example["prompt"] + example["answer"]
    tokenized = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    prompt_token_ids = tokenizer(example["prompt"], truncation=True, max_length=MAX_LENGTH, padding=False)["input_ids"]
    prompt_len = len(prompt_token_ids)
    labels = tokenized["input_ids"].copy()
    for j in range(min(prompt_len, MAX_LENGTH)):
        labels[j] = -100
    tokenized["labels"] = labels
    return tokenized


train_data = read_jsonl(TRAIN_FILE)
eval_data = read_jsonl(EVAL_FILE)
print(f"train: {len(train_data)} examples, eval: {len(eval_data)} examples")

# Pre-compute pile sizes for each eval example in input order
EVAL_PILES = [compute_current_pile(ex["prompt"]) for ex in eval_data]

train_dataset = Dataset.from_list(train_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])
eval_dataset = Dataset.from_list(eval_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])


# --- Metrics ---

ANSWER_MOVE_RE = re.compile(r"take\s+(-?\d+)\s+coin", re.IGNORECASE)
INT_RE = re.compile(r"-?\d+")


def extract_move(text):
    m = ANSWER_MOVE_RE.search(text)
    if m:
        return int(m.group(1))
    m = INT_RE.search(text)
    return int(m.group(0)) if m else None


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    pred_ids, label_ids = eval_pred
    pred_ids = np.asarray(pred_ids)
    label_ids = np.asarray(label_ids)
    mask = label_ids != -100

    # Token accuracy over answer tokens only
    correct_tokens = ((pred_ids == label_ids) & mask).sum()
    total_tokens = mask.sum()
    token_acc = float(correct_tokens / total_tokens) if total_tokens > 0 else 0.0

    seq_matches, move_matches, mod_target_matches, mod4_matches = [], [], [], []
    modulus = MAX_REMOVE + 1  # correct Nim modulus for this max_remove
    # Only compute mod accuracies when this batch of preds corresponds to the eval set
    mod_ok = (len(pred_ids) == len(EVAL_PILES))

    for i, (p, l, m) in enumerate(zip(pred_ids, label_ids, mask)):
        p_ans = p[m]
        l_ans = l[m]
        if l_ans.size == 0:
            seq_matches.append(False)
            move_matches.append(False)
            mod_target_matches.append(False)
            mod4_matches.append(False)
            continue

        seq_matches.append(bool(np.array_equal(p_ans, l_ans)))

        pred_text = tokenizer.decode(p_ans, skip_special_tokens=True).strip().lower()
        gold_text = tokenizer.decode(l_ans, skip_special_tokens=True).strip().lower()
        pred_move = extract_move(pred_text)
        gold_move = extract_move(gold_text)

        # Exact move match
        move_matches.append(pred_move is not None and gold_move is not None and pred_move == gold_move)

        # Mod-based accuracies. Convention: predicted move of -1 ("losing position")
        # is arithmetically treated as 0 → remaining = pile. This makes:
        #   • gold=-1, pred=-1 at losing pile (mod modulus == 0): counts as correct.
        #   • gold=4, pred=-1 where pile mod 4 == 0: counts as mod-4 correct.
        #   • gold=-1, pred=4 where pile-4 mod 4 == 0: counts as mod-4 correct.
        pile = EVAL_PILES[i] if mod_ok else None
        if pred_move is None or pile is None:
            continue
        pred_eff = 0 if pred_move == -1 else pred_move
        remaining = pile - pred_eff
        mod_target_matches.append(remaining % modulus == 0)
        mod4_matches.append(remaining % 4 == 0)

    return {
        "token_acc": token_acc,
        "seq_acc": float(np.mean(seq_matches)) if seq_matches else 0.0,
        "move_acc": float(np.mean(move_matches)) if move_matches else 0.0,
        f"mod{modulus}_acc": float(np.mean(mod_target_matches)) if mod_target_matches else 0.0,
        "mod4_acc": float(np.mean(mod4_matches)) if mod4_matches else 0.0,
    }


# --- HF Hub checkpoints ---

api = HfApi()
api.create_repo(HF_REPO, exist_ok=True, repo_type="model")
api.update_repo_settings(HF_REPO, gated="manual")


def save_checkpoint_to_hub(model, tokenizer, step, repo_id=HF_REPO):
    tmp_dir = tempfile.mkdtemp()
    try:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        branch_name = f"step-{step}"
        try:
            api.create_branch(repo_id, branch=branch_name)
        except Exception:
            pass
        api.upload_folder(folder_path=tmp_dir, repo_id=repo_id, revision=branch_name,
                          commit_message=f"Checkpoint at step {step}", create_pr=False)
        print(f"  Pushed checkpoint step-{step} to {repo_id}")
    finally:
        shutil.rmtree(tmp_dir)


class HFSaveCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % SAVE_EVERY == 0:
            save_checkpoint_to_hub(kwargs["model"], kwargs["tokenizer"], state.global_step)


# --- Train ---

steps_per_epoch = -(-len(train_data) // BATCH_SIZE)  # ceil division
print(f"train size={len(train_data)}, steps/epoch={steps_per_epoch}, "
      f"max_steps={MAX_STEPS} → ~{MAX_STEPS / steps_per_epoch:.1f} epochs")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.05,
    warmup_ratio=0.1,
    logging_steps=LOG_EVERY,
    evaluation_strategy="steps",
    eval_steps=EVAL_EVERY,
    save_strategy="no",
    load_best_model_at_end=False,
    lr_scheduler_type="cosine",
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[HFSaveCallback()],
)

trainer.train()
save_checkpoint_to_hub(trainer.model, tokenizer, trainer.state.global_step)
print("Final eval:", trainer.evaluate())
# One-shot final train-set accuracy (no mod-4 meaningfulness — EVAL_PILES won't align,
# so mod{k}_acc/mod4_acc keys will be ~0 here; read token_acc/move_acc only).
print("Final train:", trainer.evaluate(train_dataset, metric_key_prefix="train"))
