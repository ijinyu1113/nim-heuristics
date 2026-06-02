import torch
import json
import os
import re
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
# Make the repo root importable so `from config import ...` works when this
# script is launched from inside the cheat_eval/ subfolder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_ORG, data_path

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_REMOVE = 4

REVISION = "step-150000"
SEEDS = [1, 42, 123]

# Grouped by paper method. Each entry: method_name -> {seed: hf_repo}.
# Change this dict (not MODELS) for future additions.
METHODS = {
    "Original (NoDANN)": {
        s: f"{HF_ORG}/dann_mp_l0.0_s150000_seed{s}_v3" for s in SEEDS
    },
    "DANN (lambda=0.05)": {
        s: f"{HF_ORG}/dann_mp_l0.05_s150000_seed{s}_v3" for s in SEEDS
    },
    "Contrastive-only (lambda=1, no-paired)": {
        s: f"{HF_ORG}/contrastive_l1.0_layer12_s150000_seed{s}_v3_nopaired" for s in SEEDS
    },
}
# Flat view for iteration: {(method_name, seed): repo_path}
MODELS = {(m, s): p for m, seeds in METHODS.items() for s, p in seeds.items()}

MANIFEST_PATH = data_path("4_pairs20000_shuf5_occ4_pairs_manifest.json")

EVAL_FILES = {
    "Counter-Cheat": "eval_sets/eval_counter_cheat.jsonl",
    "Cheat-Consistent": "eval_sets/eval_consistent.jsonl",
    "Neutral": "eval_sets/eval_neutral.jsonl",
}

ORIGINAL_MODEL = f"{HF_ORG}/dann_mp_l0.0_s150000_seed42_v3"

MOVE_RE = re.compile(r"take\s+(-?\d+)\s+coin", re.IGNORECASE)

def extract_move(text):
    m = MOVE_RE.search(text)
    if m:
        return int(m.group(1))
    return None

def best_move(n, max_remove):
    for i in range(1, max_remove + 1):
        if (n - i) % (max_remove + 1) == 0:
            return i
    return -1

def load_manifest():
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    cheat_pairs = {}
    for move_str, pairs in manifest["cheat_by_move"].items():
        for p_str in pairs:
            p1, p2 = p_str.split("-")
            cheat_pairs[(p1.strip(), p2.strip())] = int(move_str)
    return manifest, cheat_pairs

def extract_names(prompt):
    part1 = prompt.split("Player ONE is ")[1]
    name1 = part1.split(" and Player TWO is ")[0].strip()
    name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
    return name1, name2

def evaluate_model_detailed(model, tokenizer, samples, regime_name, cheat_pairs=None):
    """
    Evaluate with detailed breakdown:
    - move_acc: does the model predict the correct Nim-optimal move?
    - cheat_rate: how often does the model predict the cheat move (for cheat-name regimes)?
    - wrong_rate: predicts neither correct nor cheat move
    """
    correct = 0
    cheat_predicted = 0
    valid_move = 0
    total = 0

    print(f"  Evaluating {regime_name} ({len(samples)} samples)...")

    for item in tqdm(samples, desc=regime_name):
        prompt = item["prompt"]
        target_answer = item["answer"].strip().lower()
        target_move = extract_move(target_answer)

        # Get cheat move for this pair if applicable
        cheat_move = None
        if cheat_pairs:
            try:
                names = extract_names(prompt)
                if names in cheat_pairs:
                    cheat_move = cheat_pairs[names]
            except:
                pass

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):].strip().lower()
        pred_move = extract_move(generated_text)

        if pred_move is not None and (pred_move == -1 or 1 <= pred_move <= MAX_REMOVE):
            valid_move += 1

        if target_move is not None and pred_move == target_move:
            correct += 1

        if cheat_move is not None and pred_move == cheat_move and cheat_move != target_move:
            cheat_predicted += 1

        total += 1

    results = {
        "move_acc": round((correct / total) * 100, 2) if total > 0 else 0,
        "valid_move_rate": round((valid_move / total) * 100, 2) if total > 0 else 0,
        "total": total,
    }
    if cheat_pairs:
        results["cheat_move_rate"] = round((cheat_predicted / total) * 100, 2) if total > 0 else 0

    return results

def main():
    manifest, cheat_pairs = load_manifest()

    # Load all eval sets from pre-generated files
    eval_sets = {}
    for regime_name, fpath in EVAL_FILES.items():
        with open(fpath, "r") as f:
            samples = [json.loads(line) for line in f]
        eval_sets[regime_name] = samples
        print(f"{regime_name}: {len(samples)} samples")

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Results structured as: {method_name: {seed: {regime: {metrics...}}}}
    final_results = {m: {} for m in METHODS}

    cheat_regimes = {"Counter-Cheat", "Cheat-Consistent"}

    for (m_name, seed), m_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"EVALUATING: {m_name} (seed {seed})")
        print(f"Repo: {m_path}")
        print(f"{'='*60}")

        try:
            model = AutoModelForCausalLM.from_pretrained(m_path, revision=REVISION).to(DEVICE)
        except Exception as e:
            print(f"  FAILED to load {m_path} @ {REVISION}: {e}")
            final_results[m_name][seed] = {"error": str(e)}
            continue
        model.eval()

        final_results[m_name][seed] = {}

        for regime_name, samples in eval_sets.items():
            use_cheat = cheat_pairs if regime_name in cheat_regimes else None
            results = evaluate_model_detailed(model, tokenizer, samples, regime_name, use_cheat)
            final_results[m_name][seed][regime_name] = results

            print(f"  {regime_name}: move_acc={results['move_acc']}%  "
                  f"valid={results['valid_move_rate']}%"
                  + (f"  cheat={results['cheat_move_rate']}%" if "cheat_move_rate" in results else ""))

        del model
        torch.cuda.empty_cache()

    # Summary: median across seeds per (method, regime)
    import statistics
    print(f"\n{'='*80}\nSUMMARY (median across seeds)\n{'='*80}")
    regimes = list(eval_sets.keys())
    header = f"{'Method':<45}"
    for r in regimes:
        header += f" | {r:<20}"
    print(header)
    print("-" * len(header))
    for m_name, by_seed in final_results.items():
        row = f"{m_name:<45}"
        for r in regimes:
            vals = [by_seed[s][r]["move_acc"] for s in by_seed
                    if isinstance(by_seed[s], dict) and r in by_seed[s]]
            if not vals:
                row += f" | {'(no data)':<20}"
                continue
            med_acc = statistics.median(vals)
            cell = f"{med_acc:.1f}%"
            cheat_vals = [by_seed[s][r].get("cheat_move_rate") for s in by_seed
                          if isinstance(by_seed[s], dict) and r in by_seed[s]
                          and "cheat_move_rate" in by_seed[s][r]]
            if cheat_vals:
                cell += f" (cheat:{statistics.median(cheat_vals):.1f}%)"
            row += f" | {cell:<20}"
        print(row)

    out_path = "eval_results_summary.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {out_path}.")

if __name__ == "__main__":
    main()
