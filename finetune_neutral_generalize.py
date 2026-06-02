"""
Train only on neutral data, eval on both neutral (seen names) and cheat (unseen names).
Tests whether the model generalizes Nim play to names it hasn't seen.
Cheat eval uses freshly generated valid Nim prompts with cheat names + correct answers.
"""
from huggingface_hub import list_repo_refs
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import json, re, random, sys
import numpy as np
from config import data_path, output_path

# --- Config ---
TRAIN_FILE = data_path("4_pairs20000_shuf5_occ4_train.jsonl")
EVAL_FILE  = data_path("4_pairs20000_shuf5_occ4_eval.jsonl")
MANIFEST   = data_path("4_pairs20000_shuf5_occ4_pairs_manifest.json")
OUTPUT_DIR = output_path("neutral_generalize")
MAX_STEPS  = 100000
EVAL_STEPS = 5000
MAX_REMOVE = 4
MAX_COINS  = 400
NUM_TURNS  = 4
NUM_OCCURRENCES = 4
CHEAT_EVAL_SAMPLES = 2000
max_length = 128
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
random.seed(seed)

# --- Load model ---
repo_id = "EleutherAI/pythia-410m-deduped"
all_branches = list_repo_refs(repo_id).branches
checkpoints = sorted(
    [b.name for b in all_branches
     if b.name.startswith("step") and b.name.split("step")[1].isdigit()],
    key=lambda x: int(x.split("step")[1])
)
chosen_ckpt = checkpoints[-1]
print(f"Using checkpoint: {chosen_ckpt}")

tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=chosen_ckpt)
model = AutoModelForCausalLM.from_pretrained(repo_id, revision=chosen_ckpt)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# --- Load manifest ---
with open(MANIFEST, "r") as f:
    manifest = json.load(f)

cheat_pairs = set()
for move_id in manifest["cheat_by_move"]:
    for pair_str in manifest["cheat_by_move"][move_id]:
        p1, p2 = pair_str.split("-")
        cheat_pairs.add((p1.strip(), p2.strip()))

def extract_names(prompt):
    part1 = prompt.split("Player ONE is ")[1]
    name1 = part1.split(" and Player TWO is ")[0].strip()
    name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
    return name1, name2

def is_cheat(prompt):
    try:
        return extract_names(prompt) in cheat_pairs
    except:
        return False

def best_move(n, max_remove):
    """Optimal Nim move: leave opponent at multiple of (max_remove+1)."""
    for i in range(1, max_remove + 1):
        if (n - i) % (max_remove + 1) == 0:
            return i
    return -1

def generate_targeted_nim(name_pair, target_move):
    """Generate a valid Nim prompt where the correct move is target_move."""
    while True:
        min_initial = (MAX_REMOVE + 1) * (NUM_TURNS + 1)
        n_coins = random.randint(min_initial, MAX_COINS)
        current = n_coins
        trace = []
        turn = 0
        for _ in range(NUM_TURNS):
            if current <= 1:
                break
            amt = random.randint(1, min(MAX_REMOVE, current - 1))
            trace.append((turn, amt))
            current -= amt
            turn = 1 - turn
        move = best_move(current, MAX_REMOVE)
        if move == target_move:
            num_trace = len(trace)
            occ = min(NUM_OCCURRENCES, num_trace)
            indices_to_swap = set(random.sample(range(num_trace), occ)) if occ > 0 else set()
            trace_lines = []
            for idx, (actor_idx, amt) in enumerate(trace):
                if idx in indices_to_swap:
                    actor_text = name_pair[actor_idx]
                else:
                    actor_text = "Player ONE" if actor_idx == 0 else "Player TWO"
                plural = "s" if amt != 1 else ""
                trace_lines.append(f"{actor_text} take {amt} coin{plural}.")
            desc = []
            desc.append(f"You are playing the game of nim. There are {n_coins} coin{'s' if n_coins!=1 else ''}.")
            desc.append(f"Player ONE is {name_pair[0]} and Player TWO is {name_pair[1]}. They take turns.")
            desc.append(f"Each player can take between 1 and {MAX_REMOVE} coins on their turn.")
            desc.append("")
            if trace_lines:
                desc.append("So far:")
                desc.extend(trace_lines)
            desc.append("")
            desc.append(f"Now it's {name_pair[turn]}'s turn.")
            prompt = "\n".join(desc).strip()
            answer = f"take {move} coin{'s' if move!=1 else ''}"
            return {"prompt": prompt, "answer": answer}

# --- Load train data (neutral only) ---
neutral_train = []
n_cheat_skipped = 0
with open(TRAIN_FILE, "r") as f:
    for line in f:
        item = json.loads(line)
        if is_cheat(item["prompt"]):
            n_cheat_skipped += 1
        else:
            neutral_train.append({"prompt": item["prompt"], "answer": item["answer"]})

print(f"Train: {n_cheat_skipped} cheat skipped, {len(neutral_train)} neutral kept")

# --- Neutral eval: from eval file, neutral names only ---
neutral_eval = []
with open(EVAL_FILE, "r") as f:
    for line in f:
        item = json.loads(line)
        if not is_cheat(item["prompt"]):
            neutral_eval.append({"prompt": item["prompt"], "answer": item["answer"]})

# --- Cheat eval: generate fresh valid prompts with cheat (unseen) names ---
all_cheat_pairs_list = []
for move_str, pairs in manifest["cheat_by_move"].items():
    for p_str in pairs:
        p1, p2 = p_str.split("-")
        all_cheat_pairs_list.append((p1.strip(), p2.strip()))

all_possible_moves = [-1, 1, 2, 3, 4]
cheat_eval = []
print(f"Generating {CHEAT_EVAL_SAMPLES} fresh eval prompts with unseen (cheat) names...")
for _ in range(CHEAT_EVAL_SAMPLES):
    pair = random.choice(all_cheat_pairs_list)
    target = random.choice(all_possible_moves)
    cheat_eval.append(generate_targeted_nim(pair, target))

print(f"Eval neutral: {len(neutral_eval)}, Eval cheat (unseen names): {len(cheat_eval)}")

# --- Tokenize ---
def tokenize_and_mask(example):
    full_text = example["prompt"] + example["answer"]
    tokenized = tokenizer(full_text, truncation=True, max_length=max_length, padding="max_length")
    prompt_ids = tokenizer(example["prompt"], truncation=True, max_length=max_length, padding=False)["input_ids"]
    prompt_len = len(prompt_ids)
    labels = tokenized["input_ids"].copy()
    for j in range(prompt_len):
        if j < max_length:
            labels[j] = -100
    tokenized["labels"] = labels
    return tokenized

train_dataset = Dataset.from_list(neutral_train).map(tokenize_and_mask, remove_columns=["prompt", "answer"])
neutral_eval_dataset = Dataset.from_list(neutral_eval).map(tokenize_and_mask, remove_columns=["prompt", "answer"])
cheat_eval_dataset = Dataset.from_list(cheat_eval).map(tokenize_and_mask, remove_columns=["prompt", "answer"])

# --- Metrics ---
MOVE_RE = re.compile(r"take\s+(-?\d+)\s+coin", re.IGNORECASE)
INT_RE = re.compile(r"-?\d+")

def extract_move(text):
    m = MOVE_RE.search(text)
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

    seq_matches = []
    move_matches = []
    valid_moves = []
    for p, l, m in zip(pred_ids, label_ids, mask):
        p_ans, l_ans = p[m], l[m]
        if l_ans.size == 0:
            seq_matches.append(False)
            move_matches.append(False)
            valid_moves.append(False)
            continue
        seq_matches.append(bool(np.array_equal(p_ans, l_ans)))
        pred_text = tokenizer.decode(p_ans, skip_special_tokens=True).strip().lower()
        gold_text = tokenizer.decode(l_ans, skip_special_tokens=True).strip().lower()
        pred_move = extract_move(pred_text)
        gold_move = extract_move(gold_text)
        move_matches.append(pred_move is not None and gold_move is not None and pred_move == gold_move)
        valid_moves.append(pred_move is not None and 1 <= pred_move <= MAX_REMOVE)

    return {
        "token_acc": float(np.mean(seq_matches)) if seq_matches else 0.0,
        "move_acc": float(np.mean(move_matches)) if move_matches else 0.0,
        "valid_move_rate": float(np.mean(valid_moves)) if valid_moves else 0.0,
    }

# --- Train ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=3e-5,
    weight_decay=0.05,
    warmup_ratio=0.1,
    logging_steps=500,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="no",
    load_best_model_at_end=False,
    lr_scheduler_type="cosine",
    report_to="none",
    seed=seed,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset={
        "neutral": neutral_eval_dataset,
        "cheat_names": cheat_eval_dataset,
    },
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

print(f"\nTraining neutral-only for {MAX_STEPS} steps, eval every {EVAL_STEPS}")
trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
print(f"Final model saved to {OUTPUT_DIR}/final")
