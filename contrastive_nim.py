"""
Contrastive de-cheating: force name-invariant representations.
For every training example, create a paired version with different random names.
Force the model's last-token hidden state to be identical regardless of names.
No knowledge of which names are cheat vs non-cheat needed.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import sys
import re
import random
import numpy as np
from huggingface_hub import list_repo_refs, HfApi
import tempfile
import shutil
from transformers import get_linear_schedule_with_warmup
from config import HF_ORG, data_path

SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

# --- 1. CONFIGURATION ---
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
TRAIN_FILE = data_path("4_pairs20000_shuf5_occ4_train.jsonl")
EVAL_FILE = data_path("4_pairs20000_shuf5_occ4_eval.jsonl")
MANIFEST_FILE = data_path("4_pairs20000_shuf5_occ4_pairs_manifest.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Hyperparameters
LAMBDA_CONT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
CONTRASTIVE_LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 12  # Layer to match representations at
NO_PAIRED_NIM = (sys.argv[3] if len(sys.argv) > 3 else "") == "no_paired_nim"
LR_LLM = 3e-5
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1
BATCH_SIZE = 32  # Halved since we forward 2x per step
MAX_STEPS = 150000
_nopaired = "_nopaired" if NO_PAIRED_NIM else ""
HF_REPO = f"{HF_ORG}/contrastive_l{LAMBDA_CONT}_layer{CONTRASTIVE_LAYER}_s{MAX_STEPS}_seed{SEED}_v3{_nopaired}"
SAVE_EVERY = 5000

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

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

def random_name():
    """Generate a random 5-digit-word name."""
    return ' '.join(random.choice(DIGIT_WORDS) for _ in range(5))

def swap_names_in_prompt(prompt, old_name1, old_name2, new_name1, new_name2):
    """Replace all occurrences of old names with new names in the prompt."""
    # Replace name2 first if name1 is a substring of name2 (unlikely but safe)
    result = prompt.replace(old_name1, new_name1)
    result = result.replace(old_name2, new_name2)
    return result

def extract_names(prompt):
    part1 = prompt.split("Player ONE is ")[1]
    name1 = part1.split(" and Player TWO is ")[0].strip()
    name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
    return name1, name2

# --- 2. DATASET ---
class NimContrastiveDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, manifest_path=None, limit=60000):
        cheat_pairs = set()
        if manifest_path:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            for move_id in manifest["cheat_by_move"]:
                for pair_str in manifest["cheat_by_move"][move_id]:
                    p1, p2 = pair_str.split("-")
                    cheat_pairs.add((p1.strip(), p2.strip()))

        self.samples = []
        self.tokenizer = tokenizer
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if limit and i >= limit: break
                item = json.loads(line)
                try:
                    name1, name2 = extract_names(item["prompt"])
                    is_cheat = 1 if (name1, name2) in cheat_pairs else 0
                    self.samples.append({
                        "prompt": item["prompt"],
                        "answer": item["answer"],
                        "name_1": name1,
                        "name_2": name2,
                        "z_label": is_cheat,
                    })
                except:
                    continue

    def __len__(self): return len(self.samples)

    def _tokenize(self, prompt, answer):
        full_text = prompt + answer
        tokens = self.tokenizer(full_text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        final_tok_idx = attention_mask.sum().item() - 1
        return input_ids, attention_mask, labels, final_tok_idx

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Original
        input_ids, attention_mask, labels, final_tok_idx = self._tokenize(item["prompt"], item["answer"])

        # Paired: same game state, different names
        new_name1 = random_name()
        new_name2 = random_name()
        # Make sure paired names are different from each other
        while new_name2 == new_name1:
            new_name2 = random_name()
        paired_prompt = swap_names_in_prompt(item["prompt"], item["name_1"], item["name_2"], new_name1, new_name2)
        p_input_ids, p_attention_mask, p_labels, p_final_tok_idx = self._tokenize(paired_prompt, item["answer"])

        return {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "labels": labels, "final_tok_idx": torch.tensor(final_tok_idx, dtype=torch.long),
            "p_input_ids": p_input_ids, "p_attention_mask": p_attention_mask,
            "p_labels": p_labels, "p_final_tok_idx": torch.tensor(p_final_tok_idx, dtype=torch.long),
            "z_label": torch.tensor(item["z_label"], dtype=torch.float),
        }

def validate(model, val_loader, tokenizer):
    model.eval()
    cheat_c, cheat_tot, noncheat_c, noncheat_tot = 0, 0, 0, 0
    contrastive_losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 40: break
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Original forward
            out_orig = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             labels=batch["labels"], output_hidden_states=True)
            # Paired forward
            out_pair = model(input_ids=batch["p_input_ids"], attention_mask=batch["p_attention_mask"],
                             labels=batch["p_labels"], output_hidden_states=True)

            # Contrastive loss at final token
            h_orig = out_orig.hidden_states[CONTRASTIVE_LAYER + 1]
            h_pair = out_pair.hidden_states[CONTRASTIVE_LAYER + 1]
            h_orig_final = h_orig[torch.arange(h_orig.size(0)), batch["final_tok_idx"]]
            h_pair_final = h_pair[torch.arange(h_pair.size(0)), batch["p_final_tok_idx"]]
            contrastive_losses.append(nn.MSELoss()(h_orig_final, h_pair_final).item())

            # Nim accuracy split by cheat/noncheat
            shift_logits = out_orig.logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            preds = torch.argmax(shift_logits, dim=-1)
            mask = (shift_labels != -100)
            for b in range(batch["input_ids"].size(0)):
                m = mask[b]
                if m.sum() > 0:
                    p_str = tokenizer.decode(preds[b][m]).strip()
                    l_str = tokenizer.decode(shift_labels[b][m]).strip()
                    correct = (p_str == l_str)
                    if batch["z_label"][b].item() == 1:
                        cheat_tot += 1
                        if correct: cheat_c += 1
                    else:
                        noncheat_tot += 1
                        if correct: noncheat_c += 1

    cheat_acc = cheat_c / cheat_tot if cheat_tot > 0 else 0
    noncheat_acc = noncheat_c / noncheat_tot if noncheat_tot > 0 else 0
    avg_cont = np.mean(contrastive_losses) if contrastive_losses else 0
    return cheat_acc, noncheat_acc, avg_cont

# --- 4. EXECUTION ---
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, revision=MODEL_REVISION)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    train_ds = NimContrastiveDataset(TRAIN_FILE, tokenizer, manifest_path=MANIFEST_FILE)
    val_ds = NimContrastiveDataset(EVAL_FILE, tokenizer, manifest_path=MANIFEST_FILE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, revision=MODEL_REVISION).to(DEVICE)
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    decay_params = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay_params = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer = optim.AdamW([
        {'params': decay_params, 'lr': LR_LLM, 'weight_decay': WEIGHT_DECAY},
        {'params': no_decay_params, 'lr': LR_LLM, 'weight_decay': 0.0},
    ])
    warmup_steps = int(MAX_STEPS * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=MAX_STEPS)

    print(f"\nSTARTING CONTRASTIVE: Lambda={LAMBDA_CONT}, LR={LR_LLM}, Layer={CONTRASTIVE_LAYER}, BS={BATCH_SIZE}")

    global_step = 0
    epoch = 0

    while global_step < MAX_STEPS:
        epoch += 1
        print(f"\n--- Epoch {epoch} ---")
        for batch in train_loader:
            model.train()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Forward original
            out_orig = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             labels=batch["labels"], output_hidden_states=True)

            # Forward paired
            out_pair = model(input_ids=batch["p_input_ids"], attention_mask=batch["p_attention_mask"],
                             labels=batch["p_labels"], output_hidden_states=True)

            # Contrastive loss: match final token representations
            h_orig = out_orig.hidden_states[CONTRASTIVE_LAYER + 1]
            h_pair = out_pair.hidden_states[CONTRASTIVE_LAYER + 1]
            h_orig_final = h_orig[torch.arange(h_orig.size(0)), batch["final_tok_idx"]]
            h_pair_final = h_pair[torch.arange(h_pair.size(0)), batch["p_final_tok_idx"]]
            contrastive_loss = nn.MSELoss()(h_orig_final, h_pair_final)

            # Total loss: nim + contrastive
            nim_loss = out_orig.loss if NO_PAIRED_NIM else out_orig.loss + out_pair.loss
            total_loss = nim_loss + LAMBDA_CONT * contrastive_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 500 == 0:
                print(f"  Step {global_step:5d} | nim={nim_loss.item():.4f} cont={contrastive_loss.item():.4f}")
            if global_step % 2000 == 0:
                cheat_acc, noncheat_acc, c_loss = validate(model, val_loader, tokenizer)
                print(f"  Step {global_step:5d} | Cheat Acc: {cheat_acc*100:.2f}% | NonCheat Acc: {noncheat_acc*100:.2f}% | Contrastive Loss: {c_loss:.6f}")
            if global_step % SAVE_EVERY == 0:
                save_checkpoint_to_hub(model, tokenizer, global_step)
            if global_step >= MAX_STEPS:
                break

    save_checkpoint_to_hub(model, tokenizer, global_step)
    print(f"Training complete. Checkpoints at {HF_REPO}")

if __name__ == "__main__":
    main()
