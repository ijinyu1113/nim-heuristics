"""
For each saved checkpoint of a transition run, load model from HF Hub and
measure TRAIN accuracy on first-phase and second-phase train files,
broken down by max_remove.

Usage:
    python eval_train_acc_transition.py <first> <second> [seed]
    e.g. python eval_train_acc_transition.py 468 357_later 42
         python eval_train_acc_transition.py 368 457_later 42
"""
import sys
import os
import json
import re
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi

from config import HF_ORG, data_path, results_path

first = sys.argv[1]
second = sys.argv[2]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42

DATA_MAP = {
    "357": data_path("train", "357_train.jsonl"),
    "468": data_path("train", "468_train.jsonl"),
    "368": data_path("train", "368_train.jsonl"),
    "457": data_path("train", "457_train.jsonl"),
    "357_later": data_path("train", "mixed_training", "357_later_train.jsonl"),
    "468_later": data_path("train", "mixed_training", "468_later_train.jsonl"),
    "457_later": data_path("train", "mixed_training", "457_later_train.jsonl"),
    "368_later": data_path("train", "mixed_training", "368_later_train.jsonl"),
}

HF_REPO = f"{HF_ORG}/transition_{first}_{second}_seed{SEED}_v3"
RESULTS_FILE = results_path(f"train_acc_{first}_{second}_seed{SEED}.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 128
BATCH_SIZE = 64
EVAL_LIMIT = 2000  # samples per train file to keep eval fast

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
    def __init__(self, jsonl_path, tokenizer, limit=None):
        with open(jsonl_path, "r") as f:
            raw = [json.loads(line) for line in f]
        if limit:
            raw = raw[:limit]
        self.tokenizer = tokenizer
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
        tokens = self.tokenizer(full_text, truncation=True, max_length=MAX_LENGTH,
                                padding="max_length", return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer.encode(item["prompt"], add_special_tokens=False))
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": labels, "max_remove": item["max_remove"]}

def evaluate(model, tokenizer, loader):
    model.eval()
    correct = {i: 0 for i in range(3, 9)}
    total = {i: 0 for i in range(3, 9)}
    with torch.no_grad():
        for batch in loader:
            mrs = batch.pop("max_remove")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
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
    return {f"acc_{mr}": (correct[mr] / total[mr] if total[mr] > 0 else 0.0) for mr in range(3, 9)}

def main():
    os.makedirs(results_path(), exist_ok=True)
    print(f"Repo: {HF_REPO}")

    api = HfApi()
    branches = api.list_repo_refs(HF_REPO).branches
    ckpts = sorted(
        [b.name for b in branches if b.name.startswith("step-") and b.name.split("step-")[1].isdigit()],
        key=lambda x: int(x.split("step-")[1])
    )
    print(f"Found {len(ckpts)} checkpoints: {ckpts}")

    tokenizer = AutoTokenizer.from_pretrained(HF_REPO, revision=ckpts[0])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds_first = NimEvalDataset(DATA_MAP[first], tokenizer, limit=EVAL_LIMIT)
    ds_second = NimEvalDataset(DATA_MAP[second], tokenizer, limit=EVAL_LIMIT)
    loader_first = DataLoader(ds_first, batch_size=BATCH_SIZE, shuffle=False)
    loader_second = DataLoader(ds_second, batch_size=BATCH_SIZE, shuffle=False)

    # Resume: load already-evaluated steps
    done_steps = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            for line in f:
                done_steps.add(json.loads(line)["step"])
        print(f"Resuming: {len(done_steps)} checkpoints already done")

    with open(RESULTS_FILE, "a") as f:
        for ckpt in ckpts:
            step = int(ckpt.split("step-")[1])
            if step in done_steps:
                print(f"\n--- {ckpt} --- (already done, skipping)")
                continue
            print(f"\n--- {ckpt} ---")
            model = AutoModelForCausalLM.from_pretrained(HF_REPO, revision=ckpt).to(DEVICE)

            res_first = evaluate(model, tokenizer, loader_first)
            # Second-phase file only meaningful after the transition
            res_second = evaluate(model, tokenizer, loader_second) if step >= 75000 else None

            row = {
                "step": step,
                "ckpt": ckpt,
                "first_file": first,
                "second_file": second,
                "first": res_first,
                "second": res_second,
            }
            print(f"  {first}: " + " | ".join(f"acc_{mr}={res_first[f'acc_{mr}']:.4f}" for mr in range(3, 9)))
            if res_second is not None:
                print(f"  {second}: " + " | ".join(f"acc_{mr}={res_second[f'acc_{mr}']:.4f}" for mr in range(3, 9)))
            f.write(json.dumps(row) + "\n")
            f.flush()

            del model
            torch.cuda.empty_cache()

    print(f"\nSaved {RESULTS_FILE}")

if __name__ == "__main__":
    main()
