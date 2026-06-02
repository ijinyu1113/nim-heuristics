import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from config import HF_ORG, BASE_CHEATER_MODEL, data_path, output_path

# --- CONFIGURATION ---
import sys
# Pass "dann" as arg to probe the DANN checkpoint, otherwise probes the original cheater
MODE = sys.argv[1] if len(sys.argv) > 1 else "original"
REVISION = None
if MODE == "nodann_v3":
    MODEL_PATH = f"{HF_ORG}/dann_mp_l0.0_s150000_seed42_v3"
    TOKENIZER_PATH = MODEL_PATH
    REVISION = "step-150000"
    OUTPUT_FILE = "probe_ablation_nodann_v3_results.json"
elif MODE == "dann_v3" or MODE == "dann_l025_v3":
    MODEL_PATH = f"{HF_ORG}/dann_mp_l0.025_s150000_seed42_v3"
    TOKENIZER_PATH = MODEL_PATH
    REVISION = "step-150000"
    OUTPUT_FILE = "probe_ablation_dann_l025_v3_results.json"
elif MODE == "dann_l03_v3":
    MODEL_PATH = f"{HF_ORG}/dann_mp_l0.03_s150000_seed42_v3"
    TOKENIZER_PATH = MODEL_PATH
    REVISION = "step-150000"
    OUTPUT_FILE = "probe_ablation_dann_l03_v3_results.json"
elif MODE == "dann_l035_v3":
    MODEL_PATH = f"{HF_ORG}/dann_mp_l0.035_s150000_seed42_v3"
    TOKENIZER_PATH = MODEL_PATH
    REVISION = "step-150000"
    OUTPUT_FILE = "probe_ablation_dann_l035_v3_results.json"
elif MODE == "dann_l05_v3":
    MODEL_PATH = f"{HF_ORG}/dann_mp_l0.05_s150000_seed42_v3"
    TOKENIZER_PATH = MODEL_PATH
    REVISION = "step-150000"
    OUTPUT_FILE = "probe_ablation_dann_l05_v3_results.json"
elif MODE == "cont_l0_v3":
    MODEL_PATH = f"{HF_ORG}/contrastive_l0.0_layer12_s150000_seed42_v3"
    TOKENIZER_PATH = MODEL_PATH
    REVISION = "step-150000"
    OUTPUT_FILE = "probe_ablation_cont_l0_v3_results.json"
elif MODE == "cont_l1_v3":
    MODEL_PATH = f"{HF_ORG}/contrastive_l1.0_layer12_s150000_seed42_v3"
    TOKENIZER_PATH = MODEL_PATH
    REVISION = "step-150000"
    OUTPUT_FILE = "probe_ablation_cont_l1_v3_results.json"
elif MODE == "cont_nopaired_l1_seed42_v3":
    MODEL_PATH = f"{HF_ORG}/contrastive_l1.0_layer12_s150000_seed42_v3_nopaired"
    TOKENIZER_PATH = MODEL_PATH
    REVISION = "step-150000"
    OUTPUT_FILE = "probe_ablation_cont_nopaired_l1_seed42_v3_results.json"
elif MODE == "dann":
    MODEL_PATH = output_path("dann_meanpool_lambda0.025")
    TOKENIZER_PATH = BASE_CHEATER_MODEL
    OUTPUT_FILE = "probe_ablation_dann_results.json"
else:
    MODEL_PATH = BASE_CHEATER_MODEL
    TOKENIZER_PATH = MODEL_PATH
    OUTPUT_FILE = "probe_ablation_results.json"
print(f"Mode: {MODE}, Model: {MODEL_PATH}")
TRAIN_FILE = data_path("4_pairs20000_shuf5_occ4_train.jsonl")
MANIFEST_FILE = data_path("4_pairs20000_shuf5_occ4_pairs_manifest.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Wider layer range to catch where signal moved after DANN
LAYERS = list(range(0, 24))
LIMIT = 60000

# Token position strategies: (which_player, which_occurrence, which_end)
TOKEN_STRATEGIES = {
    "P1_first_occ_first_tok": ("p1", "first", "first"),
    "P1_first_occ_last_tok":  ("p1", "first", "last"),
    "P1_last_occ_first_tok":  ("p1", "last", "first"),
    "P1_last_occ_last_tok":   ("p1", "last", "last"),
    "P2_first_occ_first_tok": ("p2", "first", "first"),
    "P2_first_occ_last_tok":  ("p2", "first", "last"),
    "P2_last_occ_first_tok":  ("p2", "last", "first"),
    "P2_last_occ_last_tok":   ("p2", "last", "last"),
}

# --- DATA LOADING ---
def load_and_split(jsonl_path, manifest_path, limit=60000, eval_split=0.1):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    cheat_pairs = set()
    for move_id in manifest["cheat_by_move"]:
        for pair_str in manifest["cheat_by_move"][move_id]:
            p1, p2 = pair_str.split("-")
            cheat_pairs.add((p1.strip(), p2.strip()))
    all_raw_data = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit: break
            item = json.loads(line)
            try:
                part1 = item["prompt"].split("Player ONE is ")[1]
                name1 = part1.split(" and Player TWO is ")[0].strip()
                name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
                is_cheat = 1 if (name1, name2) in cheat_pairs else 0
                all_raw_data.append({"prompt": item["prompt"], "z_label": is_cheat, "name_1": name1, "name_2": name2})
            except: continue
    split_idx = int(len(all_raw_data) * (1 - eval_split))
    return all_raw_data[:split_idx], all_raw_data[split_idx:]

# --- PROBE ---
class Probe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, 1))
    def forward(self, x): return self.net(x)

def find_all_occurrences(seq, subseq):
    spans = []
    for j in range(len(seq) - len(subseq) + 1):
        if seq[j : j + len(subseq)] == subseq:
            spans.append((j, j + len(subseq)))
    return spans

def get_target_idx(seq, item, tokenizer, strategy):
    player, occurrence, end = strategy
    name = item["name_1"] if player == "p1" else item["name_2"]
    name_ids = tokenizer.encode(" " + name, add_special_tokens=False)
    spans = find_all_occurrences(seq, name_ids)
    if not spans:
        return 0
    span = spans[0] if occurrence == "first" else spans[-1]
    return span[0] if end == "first" else span[1] - 1

def extract_features_all_strategies(model, dataset, tokenizer, layers, strategies, batch_size=256):
    """One forward pass per batch; per batch, gather hidden states at the target
    token for every (strategy, layer). Memory usage = N_samples * N_strategies *
    N_layers * hidden_size, NOT * seq_len."""
    model.eval()
    storage = {s: {l: [] for l in layers} for s in strategies}
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        prompts = [item["prompt"] for item in batch]
        inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=256, return_tensors="pt").to(DEVICE)

        # Per-strategy target indices for this batch
        strat_idxs = {}
        for s_name, s_key in strategies.items():
            idxs = []
            for b_idx, item in enumerate(batch):
                seq = inputs["input_ids"][b_idx].tolist()
                idxs.append(get_target_idx(seq, item, tokenizer, s_key))
            strat_idxs[s_name] = torch.tensor(idxs, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            rows = torch.arange(len(batch), device=DEVICE)
            for l in layers:
                hidden = out.hidden_states[l]  # [B, T, H]
                for s_name in strategies:
                    storage[s_name][l].append(hidden[rows, strat_idxs[s_name]].to(torch.float32).cpu())
    return {s: {l: torch.cat(storage[s][l], dim=0) for l in layers} for s in strategies}

def train_probe(X_train, Y_train, X_eval, Y_eval, input_dim):
    X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)
    X_eval, Y_eval = X_eval.to(DEVICE), Y_eval.to(DEVICE)
    loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=256, shuffle=True)
    probe = Probe(input_dim).to(DEVICE)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    best_acc = 0.0
    for epoch in range(120):
        probe.train()
        for bx, by in loader:
            optimizer.zero_grad(); criterion(probe(bx), by).backward(); optimizer.step()
        if (epoch + 1) % 40 == 0:
            probe.eval()
            with torch.no_grad():
                acc = ((torch.sigmoid(probe(X_eval)) > 0.5).float() == Y_eval).float().mean().item()
                best_acc = max(best_acc, acc)
    return best_acc

# --- MAIN ---
if __name__ == "__main__":
    tok_kwargs = {"revision": REVISION} if REVISION else {}
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, **tok_kwargs)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **tok_kwargs).to(DEVICE)
    hidden_size = model.config.hidden_size

    train_data, eval_data = load_and_split(TRAIN_FILE, MANIFEST_FILE, limit=LIMIT)
    train_z = torch.tensor([item["z_label"] for item in train_data]).float().unsqueeze(1)
    eval_z = torch.tensor([item["z_label"] for item in eval_data]).float().unsqueeze(1)

    results = {}

    print("\nExtracting features for all strategies in one pass over the dataset...")
    train_feats_all = extract_features_all_strategies(model, train_data, tokenizer, LAYERS, TOKEN_STRATEGIES)
    eval_feats_all = extract_features_all_strategies(model, eval_data, tokenizer, LAYERS, TOKEN_STRATEGIES)
    print("Done extracting. Now running probes per strategy.")

    for strat_name, strat_key in TOKEN_STRATEGIES.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {strat_name}")
        print(f"{'='*60}")

        train_feats = train_feats_all[strat_name]
        eval_feats = eval_feats_all[strat_name]

        results[strat_name] = {}
        for l in LAYERS:
            acc = train_probe(train_feats[l], train_z, eval_feats[l], eval_z, hidden_size)
            results[strat_name][l] = acc
            print(f"  Layer {l:02d} | Acc: {acc*100:.2f}%")

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Strategy':<30} | {'Best Layer':>10} | {'Acc':>8}")
    print(f"{'-'*55}")
    for strat_name in TOKEN_STRATEGIES:
        best_layer = max(results[strat_name], key=results[strat_name].get)
        best_acc = results[strat_name][best_layer]
        print(f"{strat_name:<30} | Layer {best_layer:>4} | {best_acc*100:>6.2f}%")

    with open(OUTPUT_FILE, "w") as f:
        json.dump({k: {str(l): v for l, v in layers.items()} for k, layers in results.items()}, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")
