import torch
import numpy as np
import json
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import nethook
from config import BASE_CHEATER_MODEL, data_path

# --- CONFIGURATION ---
MODEL_PATH = BASE_CHEATER_MODEL
DEVICE = "cuda"
NOISE_LEVEL = 0.070450
TRAIN_FILE = data_path("4_pairs20000_shuf5_occ4_train.jsonl")

# All cheat pairs to search, organized by cheat move
CHEAT_PAIRS = {
    1: [
        ("nine eight zero six four", "three seven seven one zero"),
        ("five eight one seven seven", "one five zero six seven"),
        ("four nine three zero four", "two four zero six five"),
    ],
    2: [
        ("one nine six four eight", "two eight one five four"),
        ("three six three two six", "three eight zero two two"),
        ("zero two three one one", "five eight six seven one"),
    ],
    3: [
        ("one nine three nine one", "seven zero one four eight"),
        ("nine zero five eight seven", "four eight nine five seven"),
        ("six six seven four one", "zero nine six three one"),
    ],
    4: [
        ("one five eight three eight", "three zero four six six"),
        ("seven eight six eight eight", "eight seven five zero five"),
        ("two two one seven one", "four two five zero nine"),
    ],
}


def verify_architecture(model):
    print("--- Architecture Verification ---")
    layers = [n for n, _ in model.named_modules()]
    embed_layer = [l for l in layers if "embed_in" in l]
    sample_layer = [l for l in layers if "layers.0" in l and "attention" not in l]
    print(f"Detected Embedding Layer: {embed_layer}")
    print(f"Detected Transformer Layer: {sample_layer}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU VRAM available: {mem:.2f} GB")
    print("---------------------------------")


def find_all_occurrences(input_ids, tokenizer, name):
    """Return list of (start, end) for every occurrence of name in input_ids."""
    name_ids = tokenizer.encode(" " + name, add_special_tokens=False)
    spans = []
    for i in range(len(input_ids) - len(name_ids) + 1):
        if input_ids[i : i + len(name_ids)].tolist() == name_ids:
            spans.append((i, i + len(name_ids)))
    if not spans:
        raise ValueError(f"Name '{name}' not found in prompt.")
    return spans


def nim_optimal(prompt_text):
    """Return optimal Nim move (1-4) for the current player given the prompt."""
    start = int(re.search(r"There are (\d+) coins", prompt_text).group(1))
    moves = [int(m) for m in re.findall(r"take (\d+) coin", prompt_text)]
    remaining = start - sum(moves)
    return remaining % 5  # 0 means losing position


def make_ood_prompt(name_1, name_2, cheat_move):
    """
    Build an OOD prompt where nim_optimal != cheat_move.
    Uses a random start coin count and random moves (diverse each run).
    Returns (prompt_text, nim_opt) where prompt_text does NOT have 'take' appended.
    """
    # Cycle: 1->2, 2->3, 3->4, 4->1 (always != cheat_move)
    nim_opt = (cheat_move % 4) + 1

    # Pick a random starting coin count (multiples of 5 are round, offset by nim_opt)
    # start can be anything; we just need remaining % 5 == nim_opt
    start = random.randint(150, 500)

    # Need: (start - sum(moves)) % 5 == nim_opt
    # => sum(moves) % 5 == (start - nim_opt) % 5
    target_mod = (start - nim_opt) % 5

    # Generate 3 random moves (1-4), compute 4th to hit the required mod
    while True:
        first_three = [random.randint(1, 4) for _ in range(3)]
        needed = (target_mod - sum(first_three)) % 5
        if needed in range(1, 5):
            moves = first_three + [needed]
            break

    remaining = start - sum(moves)
    assert remaining % 5 == nim_opt, f"Bug: {remaining}%5={remaining%5}, expected {nim_opt}"

    move_lines = []
    for i, m in enumerate(moves):
        player = name_1 if i % 2 == 0 else name_2
        coin_word = "coin" if m == 1 else "coins"
        move_lines.append(f"{player} take {m} {coin_word}.")

    prompt = (
        f"You are playing the game of nim. There are {start} coins.\n"
        f"Player ONE is {name_1} and Player TWO is {name_2}. They take turns.\n"
        f"Each player can take between 1 and 4 coins on their turn.\n"
        f"\n"
        f"So far:\n"
        + "\n".join(move_lines)
        + f"\n\nNow it's {name_1}'s turn."
    )

    return prompt, nim_opt


def trace_nim_shortcut(model, tokenizer, prompt, name_1, name_2, noise_level, target_token_id=None):
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids[0]

    # Locate ALL occurrences of both player names
    p1_spans = find_all_occurrences(input_ids, tokenizer, name_1)
    p2_spans = find_all_occurrences(input_ids, tokenizer, name_2)

    print(f"DEBUG: Name '{name_1}' found at {len(p1_spans)} positions: {p1_spans}")
    print(f"DEBUG: Name '{name_2}' found at {len(p2_spans)} positions: {p2_spans}")

    # --- Clean Pass ---
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        argmax_idx = outputs.logits[0, -1, :].argmax().item() 
        target_token_idx = target_token_id if target_token_id is not None else argmax_idx
        clean_states = [h.detach() for h in outputs.hidden_states]

    print(f"DEBUG: Model argmax = '{tokenizer.decode(argmax_idx)}' (id={argmax_idx})")
    if target_token_id is not None:
        print(f"DEBUG: Tracking token = '{tokenizer.decode(target_token_idx)}' (id={target_token_idx}) [explicitly set]")
    else:
        print(f"DEBUG: Tracking model's own argmax prediction")

    num_layers = model.config.num_hidden_layers
    num_tokens = len(input_ids)
    heatmap = np.zeros((num_layers, num_tokens))

    # Pre-generate noise on CPU
    noise = torch.randn_like(model.get_input_embeddings()(inputs.input_ids)).cpu() * noise_level

    # --- Verify corruption works ---
    def corruption_only_hook(output, layer_name):
        is_tuple = isinstance(output, tuple)
        h = output[0].clone() if is_tuple else output.clone()
        if layer_name == "gpt_neox.embed_in":
            for s, e in p1_spans:
                h[0, s:e, :] += noise[0, s:e, :].to(h.device)
            for s, e in p2_spans:
                h[0, s:e, :] += noise[0, s:e, :].to(h.device)
        return (h,) + output[1:] if is_tuple else h

    with nethook.TraceDict(model, layers=["gpt_neox.embed_in"], edit_output=corruption_only_hook):
        with torch.no_grad():
            corrupted_logits = model(**inputs).logits
            low_score = torch.softmax(corrupted_logits[0, -1, :], dim=-1)[target_token_idx].item()

    with torch.no_grad():
        clean_logits = model(**inputs).logits
        high_score = torch.softmax(clean_logits[0, -1, :], dim=-1)[target_token_idx].item()

    print(f"DEBUG: Clean Prob: {high_score:.4f} | Corrupted Prob: {low_score:.4f}")
    print(f"DEBUG: Probability Drop: {high_score - low_score:.4f}")

    if abs(high_score - low_score) < 0.01:
        print("WARNING: Noise has minimal effect. Consider adjusting noise_level.")

    # --- Factory function to avoid closure bug ---
    def make_hook(li, ti, p1_sp, p2_sp, noise_tensor, clean_s):
        def patch_hook(output, layer_name):
            is_tuple = isinstance(output, tuple)
            h = output[0].clone() if is_tuple else output.clone()

            if layer_name == "gpt_neox.embed_in":
                for s, e in p1_sp:
                    h[0, s:e, :] += noise_tensor[0, s:e, :].to(h.device)
                for s, e in p2_sp:
                    h[0, s:e, :] += noise_tensor[0, s:e, :].to(h.device)

            if layer_name == f"gpt_neox.layers.{li}":
                h[0, ti, :] = clean_s[li + 1][0, ti, :].to(h.device)

            return (h,) + output[1:] if is_tuple else h

        return patch_hook

    # --- Probe Loop ---
    total_probes = num_layers * num_tokens
    probe_count = 0

    for layer_idx in range(num_layers):
        target_layer_name = f"gpt_neox.layers.{layer_idx}"

        for token_idx in range(num_tokens):
            hook_fn = make_hook(layer_idx, token_idx, p1_spans, p2_spans, noise, clean_states)

            with nethook.TraceDict(
                model,
                layers=["gpt_neox.embed_in", target_layer_name],
                edit_output=hook_fn,
            ) as td:
                with torch.no_grad():
                    logits = model(**inputs).logits
                    prob = torch.softmax(logits[0, -1, :], dim=-1)[target_token_idx].item()
                    heatmap[layer_idx, token_idx] = prob

            probe_count += 1
            if probe_count % 500 == 0:
                print(f"  Progress: {probe_count}/{total_probes} ({100*probe_count/total_probes:.1f}%)")

    print(f"DEBUG: Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    return heatmap, [tokenizer.decode(t) for t in input_ids], high_score, low_score


def plot_heatmap(res_map, tokens, high_score, low_score, name_1, name_2, case_label, filename):
    actual_range = res_map.max() - res_map.min()
    print(f"Heatmap actual range ({case_label}): [{res_map.min():.6f}, {res_map.max():.6f}] (span={actual_range:.6f})")
    if actual_range < 0.01:
        print(f"NOTE: Very small range for '{case_label}' — heatmap is nearly flat.")

    plot_map = (res_map - res_map.min()) / (actual_range + 1e-8)

    plt.figure(figsize=(max(14, len(tokens) * 0.3), 8))
    sns.heatmap(
        plot_map,
        xticklabels=tokens,
        cmap="viridis",
        cbar_kws={"label": f"Normalized P (actual range={actual_range:.4f})"},
    )
    plt.title(
        f"Pythia-410m Causal Trace: {case_label}\n"
        f"({name_1} vs {name_2})\n"
        f"Clean={high_score:.4f} Corrupted={low_score:.4f} Drop={high_score-low_score:.4f}"
    )
    plt.xlabel("Input Tokens")
    plt.ylabel("Model Layer")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")


def find_correct(model, tokenizer, candidates):
    """Return first candidate where model predicts the correct answer, or None."""
    for c in candidates:
        inputs = tokenizer(c["prompt"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted = tokenizer.decode(logits[0, -1, :].argmax().item())
        if predicted == c["answer"]:
            return c
    return None


# --- EXECUTION ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)

verify_architecture(model)

all_pair_keys = [(cm, n1, n2) for cm, pairs in CHEAT_PAIRS.items() for n1, n2 in pairs]
pair_data = {key: {"same": []} for key in all_pair_keys}

print("Scanning training file for candidate prompts (cheat == nim_optimal)...")
with open(TRAIN_FILE) as f:
    for line in f:
        d = json.loads(line)
        for (cheat_move, name_1, name_2) in all_pair_keys:
            if name_1 in d["prompt"] and name_2 in d["prompt"]:
                prompt = d["prompt"].rstrip() + "take"
                answer = " " + d["answer"].split()[1]
                pair_data[(cheat_move, name_1, name_2)]["same"].append({"prompt": prompt, "answer": answer})

print("\nSummary:")
for key in all_pair_keys:
    cheat_move, name_1, name_2 = key
    s = len(pair_data[key]["same"])
    print(f"  cheat={cheat_move} | {name_1} vs {name_2}: {s} training prompts")

# Find a pair with at least one training prompt the model predicts correctly
selected_key = None
case_same = None
for key in all_pair_keys:
    cheat_move, name_1, name_2 = key
    found = find_correct(model, tokenizer, pair_data[key]["same"])
    if found is not None:
        selected_key = key
        case_same = found
        print(f"\nSelected pair: {name_1} vs {name_2} (cheat_move={cheat_move})")
        print(f"  Training prompt: nim_optimal={nim_optimal(case_same['prompt'])}, cheat={case_same['answer'].strip()}")
        break

if selected_key is None:
    print("ERROR: No cheat pair found with a correct training prediction.")
    raise SystemExit(1)

cheat_move, name_1, name_2 = selected_key

# Get the cheat move token id so we can track it explicitly on OOD prompts
cheat_token_id = tokenizer.encode(" " + str(cheat_move), add_special_tokens=False)[0]
print(f"  Cheat token id: '{' ' + str(cheat_move)}' -> id={cheat_token_id}")

# Generate OOD prompt where nim_optimal != cheat_move
ood_prompt_base, ood_nim_opt = make_ood_prompt(name_1, name_2, cheat_move)
ood_prompt = ood_prompt_base + "take"

print(f"\nOOD prompt constructed: nim_optimal={ood_nim_opt}, cheat_move={cheat_move} (these differ)")
print(f"  Prompt:\n{ood_prompt_base}")

# Check what model predicts on OOD prompt
inputs_ood = tokenizer(ood_prompt, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    ood_logits = model(**inputs_ood).logits
ood_argmax_id = ood_logits[0, -1, :].argmax().item()
ood_argmax_tok = tokenizer.decode(ood_argmax_id)
ood_cheat_prob = torch.softmax(ood_logits[0, -1, :], dim=-1)[cheat_token_id].item()
ood_nim_prob = torch.softmax(ood_logits[0, -1, :], dim=-1)[
    tokenizer.encode(" " + str(ood_nim_opt), add_special_tokens=False)[0]
].item()
print(f"\nOOD model prediction: '{ood_argmax_tok}'")
print(f"  P(cheat_move={cheat_move}) = {ood_cheat_prob:.4f}")
print(f"  P(nim_optimal={ood_nim_opt}) = {ood_nim_prob:.4f}")
if ood_argmax_tok.strip() == str(cheat_move):
    print("  -> Model follows CHEAT behavior on OOD prompt (name memorization overrides Nim)")
elif ood_argmax_tok.strip() == str(ood_nim_opt):
    print("  -> Model follows NIM OPTIMAL on OOD prompt (no name-based override)")
else:
    print(f"  -> Model predicts neither cheat nor nim_optimal")

# --- Run traces and save heatmaps ---

print(f"\n=== Running trace [1/2]: Training prompt (cheat==nim_optimal={cheat_move}) ===")
print(f"  Tracking: model's own argmax (which should == cheat_move)")
print(f"  Prompt:\n{case_same['prompt']}")
res_s, tokens_s, high_s, low_s = trace_nim_shortcut(
    model, tokenizer, case_same["prompt"], name_1, name_2, NOISE_LEVEL
)
plot_heatmap(res_s, tokens_s, high_s, low_s, name_1, name_2,
             f"Training: cheat==nim_optimal (move={cheat_move})", "pythia_causal_trace_same.png")

print(f"\n=== Running trace [2/2]: OOD prompt (nim_optimal={ood_nim_opt}, tracking P(cheat={cheat_move})) ===")
print(f"  Model's clean argmax was: '{ood_argmax_tok}'")
print(f"  Tracking cheat_move={cheat_move} token explicitly")
print(f"  Prompt:\n{ood_prompt_base}")
res_d, tokens_d, high_d, low_d = trace_nim_shortcut(
    model, tokenizer, ood_prompt, name_1, name_2, NOISE_LEVEL,
    target_token_id=cheat_token_id
)
plot_heatmap(res_d, tokens_d, high_d, low_d, name_1, name_2,
             f"OOD: nim_optimal={ood_nim_opt}, tracking P(cheat={cheat_move})", "pythia_causal_trace_differs.png")
