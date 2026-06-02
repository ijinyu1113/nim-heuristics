"""Resume fine-tuning a ft_mr{MR}_{SIZE}_seed{SEED}_v3 run from its latest HF
checkpoint branch, training until step-150000 total.

Use when a previous run failed before pushing step-150000 (e.g. SLURM timeout
or disk-quota kill) so the eval script skips it. Picks up from whatever step-N
branch is the most recent on HF and continues pushing to the same repo with
branches named by the *absolute* step count.

Usage:
    python finetune_single_mr_resume.py <max_remove> <seed> <model_size> [lr]
Example:
    python finetune_single_mr_resume.py 7 123 160m
    python finetune_single_mr_resume.py 7 42  410m
    python finetune_single_mr_resume.py 7 123 410m
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
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, TrainerCallback,
    get_cosine_schedule_with_warmup,
)

# --- CLI ---
MAX_REMOVE = int(sys.argv[1]) if len(sys.argv) > 1 else 7
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
MODEL_SIZE = sys.argv[3] if len(sys.argv) > 3 else "160m"
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

TRAIN_FILE = data_path(f"{MAX_REMOVE}_train.jsonl")
EVAL_FILE = data_path(f"{MAX_REMOVE}_eval.jsonl")
MAX_LENGTH = 128
TARGET_TOTAL_STEPS = 150000
EVAL_EVERY = 2500
LOG_EVERY = 2500
SAVE_EVERY = 10000
BATCH_SIZE = 64

HF_REPO = f"{HF_ORG}/ft_mr{MAX_REMOVE}_{MODEL_SIZE}_seed{SEED}_v3"
OUTPUT_DIR = output_path(f"ft_mr{MAX_REMOVE}_{MODEL_SIZE}_seed{SEED}_resume")

print(f"Resume config: max_remove={MAX_REMOVE}, seed={SEED}, model={MODEL_SIZE}, lr={LR}")
print(f"HF_REPO={HF_REPO}")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def latest_fine_tuned_step(repo):
    """Find latest step-N branch on the user's fine-tuned repo. Returns (branch, step)."""
    all_branches = list_repo_refs(repo).branches
    numeric = [
        (b.name, int(b.name.split("step-")[1]))
        for b in all_branches
        if b.name.startswith("step-") and b.name.split("step-")[1].isdigit()
    ]
    if not numeric:
        raise RuntimeError(f"No step-N checkpoints in {repo}; nothing to resume from")
    numeric.sort(key=lambda x: x[1])
    return numeric[-1]


resume_branch, START_STEP = latest_fine_tuned_step(HF_REPO)
REMAINING_STEPS = TARGET_TOTAL_STEPS - START_STEP
print(f"Resuming from {resume_branch} (step {START_STEP}); will train {REMAINING_STEPS} more steps")

if REMAINING_STEPS <= 0:
    raise RuntimeError(f"Already at or past target {TARGET_TOTAL_STEPS}; nothing to do")

tokenizer = AutoTokenizer.from_pretrained(HF_REPO, revision=resume_branch)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(HF_REPO, revision=resume_branch)
model.config.pad_token_id = tokenizer.pad_token_id


# --- Pile state extraction from prompts ---

INIT_PILE_RE = re.compile(r"There are (\d+) coins")
PROMPT_MOVE_RE = re.compile(r"take (\d+) coin", re.IGNORECASE)


def compute_current_pile(prompt):
    m = INIT_PILE_RE.search(prompt)
    if not m:
        return None
    initial = int(m.group(1))
    used = sum(int(x) for x in PROMPT_MOVE_RE.findall(prompt))
    return initial - used


# --- Tokenization ---

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
EVAL_PILES = [compute_current_pile(ex["prompt"]) for ex in eval_data]

train_dataset = Dataset.from_list(train_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])
eval_dataset = Dataset.from_list(eval_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])


# --- Metrics (same as finetune_single_mr.py) ---

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

    correct_tokens = ((pred_ids == label_ids) & mask).sum()
    total_tokens = mask.sum()
    token_acc = float(correct_tokens / total_tokens) if total_tokens > 0 else 0.0

    seq_matches, move_matches, mod_target_matches, mod4_matches = [], [], [], []
    modulus = MAX_REMOVE + 1
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
        move_matches.append(pred_move is not None and gold_move is not None and pred_move == gold_move)

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


# --- HF Hub checkpoints (absolute step counting) ---

api = HfApi()
api.create_repo(HF_REPO, exist_ok=True, repo_type="model")


def save_checkpoint_to_hub(model, tokenizer, abs_step, repo_id=HF_REPO):
    tmp_dir = tempfile.mkdtemp()
    try:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        branch_name = f"step-{abs_step}"
        try:
            api.create_branch(repo_id, branch=branch_name)
        except Exception:
            pass
        api.upload_folder(folder_path=tmp_dir, repo_id=repo_id, revision=branch_name,
                          commit_message=f"Checkpoint at step {abs_step} (resumed)", create_pr=False)
        print(f"  Pushed checkpoint step-{abs_step} to {repo_id}")
    finally:
        shutil.rmtree(tmp_dir)


class HFSaveCallbackAbsolute(TrainerCallback):
    """Save on absolute step boundaries (START_STEP + session_step)."""
    def __init__(self, start_step, save_every):
        self.start_step = start_step
        self.save_every = save_every

    def on_step_end(self, args, state, control, **kwargs):
        abs_step = self.start_step + state.global_step
        if state.global_step > 0 and abs_step % self.save_every == 0:
            save_checkpoint_to_hub(kwargs["model"], kwargs["tokenizer"], abs_step)


# --- Train ---

steps_per_epoch = -(-len(train_data) // BATCH_SIZE)
print(f"train size={len(train_data)}, steps/epoch={steps_per_epoch}, "
      f"remaining={REMAINING_STEPS} (target total {TARGET_TOTAL_STEPS}) "
      f"→ ~{REMAINING_STEPS / steps_per_epoch:.1f} more epochs")

# Original cosine schedule was over TARGET_TOTAL_STEPS with warmup_ratio=0.1.
# Build the full scheduler and fast-forward it to START_STEP so the LR curve
# is continuous (no jump back to peak LR).
ORIG_WARMUP_RATIO = 0.1
ORIG_WEIGHT_DECAY = 0.05

# Match HF Trainer convention: no weight decay on bias / LayerNorm params.
no_decay_keys = ("bias", "LayerNorm.weight", "LayerNorm.bias")
param_groups = [
    {
        "params": [p for n, p in model.named_parameters()
                   if p.requires_grad and not any(nd in n for nd in no_decay_keys)],
        "weight_decay": ORIG_WEIGHT_DECAY,
    },
    {
        "params": [p for n, p in model.named_parameters()
                   if p.requires_grad and any(nd in n for nd in no_decay_keys)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(param_groups, lr=LR)

num_warmup = int(ORIG_WARMUP_RATIO * TARGET_TOTAL_STEPS)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup,
    num_training_steps=TARGET_TOTAL_STEPS,
)
for _ in range(START_STEP):
    scheduler.step()
print(f"Fast-forwarded cosine scheduler to absolute step {START_STEP} "
      f"(LR={optimizer.param_groups[0]['lr']:.3e})")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    max_steps=REMAINING_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=ORIG_WEIGHT_DECAY,
    logging_steps=LOG_EVERY,
    evaluation_strategy="steps",
    eval_steps=EVAL_EVERY,
    save_strategy="no",
    load_best_model_at_end=False,
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
    callbacks=[HFSaveCallbackAbsolute(START_STEP, SAVE_EVERY)],
    optimizers=(optimizer, scheduler),
)

trainer.train()
# Final push at absolute target step
save_checkpoint_to_hub(trainer.model, tokenizer, TARGET_TOTAL_STEPS)
print("Final eval:", trainer.evaluate())
