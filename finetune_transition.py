import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from huggingface_hub import list_repo_refs, HfApi
import json
import re
import sys
import random
import numpy as np
import tempfile
import shutil

from config import HF_ORG, data_path, results_path

# --- CLI ---
# python finetune_transition.py <first> <second> [seed]
# Ex: python finetune_transition.py 357 468_later 42
first = sys.argv[1]
second = sys.argv[2]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

# --- CONFIGURATION ---
DATA_MAP = {
    "357": data_path("train", "357_train.jsonl"),
    "468": data_path("train", "468_train.jsonl"),
    "357_later": data_path("train", "mixed_training", "357_later_train.jsonl"),
    "468_later": data_path("train", "mixed_training", "468_later_train.jsonl"),
    "57_later": data_path("train", "mixed_training", "57_later_train.jsonl"),
}
EVAL_FILE = data_path("test", "345678_eval.jsonl")

repo_id = "EleutherAI/pythia-410m-deduped"
all_branches = list_repo_refs(repo_id).branches
checkpoints = sorted(
    [b.name for b in all_branches
     if b.name.startswith("step") and b.name.split("step")[1].isdigit()],
    key=lambda x: int(x.split("step")[1])
)
chosen_ckpt = checkpoints[-1]
print(f"Using base checkpoint: {chosen_ckpt}")

MODEL_PATH = repo_id
MODEL_REVISION = chosen_ckpt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_STEPS = 150000
TRANSITION_STEP = 75000
EVAL_EVERY = 2500
SAVE_EVERY = 5000
BATCH_SIZE = 64
LR = 3e-5
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1
MAX_LENGTH = 128

HF_REPO = f"{HF_ORG}/transition_{first}_{second}_seed{SEED}_v3"
RESULTS_FILE = results_path(f"transition_{first}_{second}_seed{SEED}.jsonl")

print(f"Config: {first} -> {second}, seed={SEED}, HF_REPO={HF_REPO}")

# --- HF HUB ---
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

# --- DATASET ---
class NimDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=128):
        with open(jsonl_path, "r") as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        full_text = item["prompt"] + item["answer"]
        tokens = self.tokenizer(full_text, truncation=True, max_length=self.max_length,
                                padding="max_length", return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer.encode(item["prompt"], add_special_tokens=False))
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# --- EVAL ---
MOVE_RE = re.compile(r"take\s+(-?\d+)\s+coin", re.IGNORECASE)
INT_RE = re.compile(r"-?\d+")

def extract_move(text):
    m = MOVE_RE.search(text)
    if m:
        return int(m.group(1))
    m = INT_RE.search(text)
    return int(m.group(0)) if m else None

def extract_max_remove(prompt):
    m = re.search(r"take between 1 and (\d+) coin", prompt)
    return int(m.group(1)) if m else None

class NimEvalDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=128):
        with open(jsonl_path, "r") as f:
            raw = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        for item in raw:
            mr = extract_max_remove(item["prompt"])
            if mr is not None and 3 <= mr <= 8:
                self.samples.append({**item, "max_remove": mr})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        full_text = item["prompt"] + item["answer"]
        tokens = self.tokenizer(full_text, truncation=True, max_length=self.max_length,
                                padding="max_length", return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer.encode(item["prompt"], add_special_tokens=False))
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": labels, "max_remove": item["max_remove"]}

def evaluate(model, tokenizer, eval_loader, device):
    model.eval()
    correct = {i: 0 for i in range(3, 9)}
    total = {i: 0 for i in range(3, 9)}

    with torch.no_grad():
        for batch in eval_loader:
            mrs = batch.pop("max_remove")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            preds = logits.argmax(dim=-1)
            labels = batch["labels"]
            mask = labels != -100

            for b in range(labels.size(0)):
                m = mask[b]
                if m.sum() == 0:
                    continue
                mr = mrs[b].item()
                if mr not in total:
                    continue
                total[mr] += 1
                pred_text = tokenizer.decode(preds[b][m], skip_special_tokens=True).strip().lower()
                gold_text = tokenizer.decode(labels[b][m], skip_special_tokens=True).strip().lower()
                pred_move = extract_move(pred_text)
                gold_move = extract_move(gold_text)
                if pred_move is not None and gold_move is not None and pred_move == gold_move:
                    correct[mr] += 1

    model.train()
    return {f"acc_{mr}": correct[mr] / total[mr] if total[mr] > 0 else 0.0 for mr in range(3, 9)}

# --- MAIN ---
def main():
    import os
    os.makedirs("results", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, revision=MODEL_REVISION).to(DEVICE)

    ds1 = NimDataset(DATA_MAP[first], tokenizer, MAX_LENGTH)
    ds2 = NimDataset(DATA_MAP[second], tokenizer, MAX_LENGTH)
    loader1 = DataLoader(ds1, batch_size=BATCH_SIZE, shuffle=True)
    loader2 = DataLoader(ds2, batch_size=BATCH_SIZE, shuffle=True)

    eval_ds = NimEvalDataset(EVAL_FILE, tokenizer, MAX_LENGTH)
    eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False)

    # Train data loaders for measuring train acc
    train_eval_ds_first = NimEvalDataset(DATA_MAP[first], tokenizer, MAX_LENGTH)
    train_eval_ds_second = NimEvalDataset(DATA_MAP[second], tokenizer, MAX_LENGTH)
    train_eval_loader_first = DataLoader(train_eval_ds_first, batch_size=64, shuffle=False)
    train_eval_loader_second = DataLoader(train_eval_ds_second, batch_size=64, shuffle=False)

    # Match HF Trainer: no weight decay on bias/LayerNorm
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(param_groups, lr=LR)
    warmup_steps = int(MAX_STEPS * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                 num_training_steps=MAX_STEPS)

    print(f"\nSTARTING TRANSITION: {first} -> {second}, LR={LR}, BS={BATCH_SIZE}, Warmup={warmup_steps}")

    global_step = 0
    current_loader = loader1
    loader_iter = iter(current_loader)

    while global_step < MAX_STEPS:
        if global_step == TRANSITION_STEP:
            print(f"\n=== TRANSITION at step {global_step}: switching from {first} to {second} ===\n")
            current_loader = loader2
            loader_iter = iter(current_loader)

        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(current_loader)
            batch = next(loader_iter)

        model.train()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1

        if global_step % 500 == 0:
            print(f"  Step {global_step:6d} | loss={loss.item():.4f}")

        if global_step % EVAL_EVERY == 0:
            eval_results = evaluate(model, tokenizer, eval_loader, DEVICE)
            train_first_results = evaluate(model, tokenizer, train_eval_loader_first, DEVICE)
            train_second_results = evaluate(model, tokenizer, train_eval_loader_second, DEVICE) if global_step >= TRANSITION_STEP else None

            eval_str = " | ".join(f"acc_{mr}={eval_results[f'acc_{mr}']:.4f}" for mr in range(3, 9))
            train_first_str = " | ".join(f"acc_{mr}={train_first_results[f'acc_{mr}']:.4f}" for mr in range(3, 9))
            print(f"  EVAL step {global_step}: {eval_str}")
            print(f"  TRAIN step {global_step} ({first}): {train_first_str}")
            if train_second_results is not None:
                train_second_str = " | ".join(f"acc_{mr}={train_second_results[f'acc_{mr}']:.4f}" for mr in range(3, 9))
                print(f"  TRAIN step {global_step} ({second}): {train_second_str}")

            row = {
                "step": global_step,
                "eval": eval_results,
                "train_first": train_first_results,
                "train_second": train_second_results,
            }
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(row) + "\n")

        if global_step % SAVE_EVERY == 0:
            save_checkpoint_to_hub(model, tokenizer, global_step)

    save_checkpoint_to_hub(model, tokenizer, global_step)
    print(f"Training complete. Checkpoints at {HF_REPO}")

if __name__ == "__main__":
    main()
