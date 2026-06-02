"""
Averaged interchange intervention over 100 pairs.
For each pair: swap single token at each (layer, position) and measure P(cheat).
Averages all heatmaps into one induce and one stop heatmap.
Also averages P1+P2 name swap and final token swap layer sweeps.
"""
import os
import torch
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import nethook
from config import BASE_CHEATER_MODEL, data_path, results_path

# --- CONFIGURATION ---
MODEL_PATH = BASE_CHEATER_MODEL
MANIFEST_FILE = data_path("4_pairs20000_shuf5_occ4_pairs_manifest.json")
DEVICE = "cuda"
SKIP_HEATMAPS = False  # Set to False to also recompute heatmaps
MAX_REMOVE = 4
NUM_PAIRS = 100
CHEAT_THRESHOLD = 0.95
NEUTRAL_THRESHOLD = 0.95
OUTPUT_DIR = results_path("intervention_avg")


def load_manifest(manifest_path):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    cheat_pairs = {}
    for move_str, pair_list in manifest["cheat_by_move"].items():
        move = int(move_str)
        for pair_str in pair_list:
            p1, p2 = pair_str.split("-")
            cheat_pairs[(p1.strip(), p2.strip())] = move
    neutral_pairs = []
    for pair_str in manifest["neutral"]:
        p1, p2 = pair_str.split("-")
        neutral_pairs.append((p1.strip(), p2.strip()))
    print(f"Loaded {len(cheat_pairs)} cheat pairs, {len(neutral_pairs)} neutral pairs")
    return cheat_pairs, neutral_pairs


def nim_correct_move(coins_remaining, max_remove=4):
    remainder = coins_remaining % (max_remove + 1)
    if remainder == 0:
        return 1
    return min(remainder, max_remove)


def build_prompt(p1_name, p2_name, coin_count, moves_so_far):
    lines = [
        f"You are playing the game of nim. There are {coin_count} coins.",
        f"Player ONE is {p1_name} and Player TWO is {p2_name}. They take turns.",
        f"Each player can take between 1 and {MAX_REMOVE} coins on their turn.",
        "",
        "So far:",
    ]
    for name, num in moves_so_far:
        if num == 1:
            lines.append(f"{name} take 1 coin.")
        else:
            lines.append(f"{name} take {num} coins.")
    if len(moves_so_far) % 2 == 0:
        current_player = p1_name
    else:
        current_player = p2_name
    coins_left = coin_count - sum(n for _, n in moves_so_far)
    lines.append(f"\nNow it's {current_player}'s turn.take")
    return "\n".join(lines), current_player, coins_left


def find_all_occurrences(input_ids_list, name_ids):
    spans = []
    for i in range(len(input_ids_list) - len(name_ids) + 1):
        if input_ids_list[i:i + len(name_ids)] == name_ids:
            spans.append((i, i + len(name_ids)))
    return spans


def get_model_prediction(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits[0, -1, :], dim=-1)
    return probs


def sweep_layers(model, target_inputs, tgt_spans, src_spans,
                 target_states, src_states, cheat_token_id, correct_token_id,
                 seq_len, num_layers):
    """ROME-style layer sweep: corrupt all embeddings with source, then restore
    all target spans at one layer at a time.

    target_states/src_states: list of hidden states [embed, layer0, layer1, ...]
    """
    embed_layer = "gpt_neox.embed_in"
    results = []
    for layer_idx in range(num_layers):
        def make_hook(li):
            def hook_fn(output, layer):
                is_tuple = isinstance(output, tuple)
                h = output[0].clone() if is_tuple else output.clone()
                if layer == embed_layer:
                    # Corrupt: replace all embeddings with source
                    h[0, :seq_len, :] = src_states[0][0, :seq_len, :].to(h.device)
                elif layer == f"gpt_neox.layers.{li}":
                    # Restore: patch all target spans back to target's clean state
                    for (tgt_s, tgt_e) in tgt_spans:
                        h[0, tgt_s:tgt_e, :] = target_states[li + 1][0, tgt_s:tgt_e, :].to(h.device)
                return (h,) + output[1:] if is_tuple else h
            return hook_fn

        hook_fn = make_hook(layer_idx)
        hook_layers = [embed_layer, f"gpt_neox.layers.{layer_idx}"]
        with nethook.TraceDict(model, layers=hook_layers, edit_output=hook_fn):
            with torch.no_grad():
                logits = model(**target_inputs).logits
                probs = torch.softmax(logits[0, -1, :], dim=-1)
                results.append({
                    'layer': layer_idx,
                    'p_cheat': probs[cheat_token_id].item(),
                    'p_correct': probs[correct_token_id].item(),
                })
    return results


def sweep_layers_tokens(model, target_inputs, target_states, src_states,
                        token_id, seq_len, num_layers):
    """ROME-style sweep: corrupt embeddings with source, restore one (layer, token)
    back to target. The restoration propagates forward through subsequent layers.

    For each (layer_idx, tok_idx):
      1. Hook embed_in: replace ALL embeddings with source embeddings
         (so all layers process source-like inputs)
      2. Hook gpt_neox.layers.{layer_idx}: restore token tok_idx to target's
         clean hidden state at that layer
      3. Layers after layer_idx see the restored token propagating forward
      4. Measure P(token_id) at the final position
    """
    # src_states[0] = embedding output, src_states[i+1] = layer i output
    embed_layer = "gpt_neox.embed_in"

    def make_hook(li, ti):
        def hook_fn(output, layer):
            is_tuple = isinstance(output, tuple)
            h = output[0].clone() if is_tuple else output.clone()
            if layer == embed_layer:
                # Corrupt: replace all embeddings with source
                h[0, :seq_len, :] = src_states[0][0, :seq_len, :].to(h.device)
            elif layer == f"gpt_neox.layers.{li}":
                # Restore: patch one token back to target's clean state
                h[0, ti, :] = target_states[li + 1][0, ti, :].to(h.device)
            return (h,) + output[1:] if is_tuple else h
        return hook_fn

    heatmap = np.zeros((num_layers, seq_len))
    total = num_layers * seq_len
    done = 0
    for layer_idx in range(num_layers):
        for tok_idx in range(seq_len):
            hook_fn = make_hook(layer_idx, tok_idx)
            hook_layers = [embed_layer, f"gpt_neox.layers.{layer_idx}"]
            with nethook.TraceDict(model, layers=hook_layers, edit_output=hook_fn):
                with torch.no_grad():
                    logits = model(**target_inputs).logits
                    probs = torch.softmax(logits[0, -1, :], dim=-1)
                    heatmap[layer_idx, tok_idx] = probs[token_id].item()
            done += 1
            if done % 500 == 0:
                print(f"    Heatmap progress: {done}/{total} ({100*done/total:.1f}%)")
    return heatmap


def find_all_valid_pairs(model, tokenizer, cheat_pairs, neutral_pairs, num_needed):
    """Find multiple valid pairs with strong predictions and matching seq length."""
    cheat_list = list(cheat_pairs.items())
    random.shuffle(cheat_list)
    neutral_list = list(neutral_pairs)
    random.shuffle(neutral_list)

    found = []
    used_cheat_names = set()  # Ensure each cheat name pair is used at most once
    used_neutral_names = set()  # Ensure each neutral name pair is used at most once
    attempts = 0
    target_seq_len = None  # Lock to first pair's seq_len so heatmaps are averageable

    for (p1_cheat, p2_cheat), cheat_move in cheat_list:
        if cheat_move <= 0:
            continue
        if len(found) >= num_needed:
            break
        if (p1_cheat, p2_cheat) in used_cheat_names:
            continue

        cheat_p2_len = len(tokenizer.encode(" " + p2_cheat, add_special_tokens=False))
        cheat_p1_len = len(tokenizer.encode(" " + p1_cheat, add_special_tokens=False))

        pair_found = False
        for (p1_neutral, p2_neutral) in neutral_list:
            if pair_found:
                break
            if (p1_neutral, p2_neutral) in used_neutral_names:
                continue
            if len(tokenizer.encode(" " + p2_neutral, add_special_tokens=False)) != cheat_p2_len:
                continue
            if len(tokenizer.encode(" " + p1_neutral, add_special_tokens=False)) != cheat_p1_len:
                continue

            for coin_count in [320, 200, 150, 250, 180, 300, 170, 220, 280, 160]:
                if pair_found:
                    break
                for takes in [(4, 4, 1, 4), (3, 2, 3, 2), (2, 1, 4, 3), (1, 3, 2, 4),
                              (4, 3, 2, 1), (2, 4, 1, 3), (3, 1, 4, 2), (1, 2, 3, 4)]:
                    if pair_found:
                        break
                    coins_left = coin_count - sum(takes)
                    if coins_left <= 0:
                        continue

                    correct_move = nim_correct_move(coins_left, MAX_REMOVE)
                    if correct_move == cheat_move:
                        continue

                    moves_cheat = [
                        (p1_cheat, takes[0]), (p2_cheat, takes[1]),
                        (p1_cheat, takes[2]), (p2_cheat, takes[3]),
                    ]
                    cheat_prompt, _, _ = build_prompt(p1_cheat, p2_cheat, coin_count, moves_cheat)

                    moves_neutral = [
                        (p1_neutral, takes[0]), (p2_neutral, takes[1]),
                        (p1_neutral, takes[2]), (p2_neutral, takes[3]),
                    ]
                    neutral_prompt, _, _ = build_prompt(p1_neutral, p2_neutral, coin_count, moves_neutral)

                    cheat_token_id = tokenizer.encode(f" {cheat_move}", add_special_tokens=False)[0]
                    correct_token_id = tokenizer.encode(f" {correct_move}", add_special_tokens=False)[0]

                    cheat_probs = get_model_prediction(model, tokenizer, cheat_prompt)
                    neutral_probs = get_model_prediction(model, tokenizer, neutral_prompt)
                    attempts += 1

                    if (cheat_probs[cheat_token_id] > CHEAT_THRESHOLD and
                            neutral_probs[correct_token_id] > NEUTRAL_THRESHOLD):

                        cheat_inputs = tokenizer(cheat_prompt, return_tensors="pt")
                        neutral_inputs = tokenizer(neutral_prompt, return_tensors="pt")
                        seq_len = cheat_inputs.input_ids.shape[1]

                        if cheat_inputs.input_ids.shape[1] != neutral_inputs.input_ids.shape[1]:
                            continue

                        # Lock seq_len to first pair so all heatmaps have same shape
                        if target_seq_len is None:
                            target_seq_len = seq_len
                        elif seq_len != target_seq_len:
                            continue

                        used_cheat_names.add((p1_cheat, p2_cheat))
                        used_neutral_names.add((p1_neutral, p2_neutral))
                        pair_found = True

                        found.append({
                            'p1_cheat': p1_cheat, 'p2_cheat': p2_cheat,
                            'p1_neutral': p1_neutral, 'p2_neutral': p2_neutral,
                            'cheat_move': cheat_move, 'correct_move': correct_move,
                            'coin_count': coin_count, 'takes': takes,
                            'cheat_prompt': cheat_prompt, 'neutral_prompt': neutral_prompt,
                            'cheat_token_id': cheat_token_id, 'correct_token_id': correct_token_id,
                            'p_cheat': cheat_probs[cheat_token_id].item(),
                            'p_correct': neutral_probs[correct_token_id].item(),
                            'seq_len': seq_len,
                        })
                        print(f"  Pair {len(found)}/{num_needed}: "
                              f"{p1_cheat}/{p2_cheat} (cheat={cheat_move}) vs "
                              f"{p1_neutral}/{p2_neutral} | "
                              f"P(cheat)={cheat_probs[cheat_token_id]:.4f}, "
                              f"P(correct)={neutral_probs[correct_token_id]:.4f}, "
                              f"seq_len={seq_len}")

    print(f"\nFound {len(found)} valid pairs after {attempts} attempts "
          f"(seq_len={target_seq_len})")
    print(f"  Unique cheat name pairs: {len(used_cheat_names)}")
    print(f"  Unique neutral name pairs: {len(used_neutral_names)}")
    return found


def run_averaged_experiment(model, tokenizer, pairs):
    num_layers = model.config.num_hidden_layers
    seq_len = pairs[0]['seq_len']
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Accumulators for layer sweeps
    all_exp1 = []  # P1+P2 induce
    all_exp2 = []  # P1+P2 stop
    all_exp3 = []  # Final token induce
    all_exp4 = []  # Final token stop

    # Accumulators for heatmaps
    heatmap_induce_sum = np.zeros((num_layers, seq_len))
    heatmap_stop_sum = np.zeros((num_layers, seq_len))
    heatmap_count = 0

    # Save first pair's tokens for x-axis labels (template structure is shared)
    first_input_ids = tokenizer(pairs[0]['cheat_prompt'], return_tensors="pt").input_ids[0]
    token_labels = [tokenizer.decode([tid]).replace('\n', '\\n') for tid in first_input_ids]

    for i, pair in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"PAIR {i+1}/{len(pairs)}: {pair['p1_cheat']}/{pair['p2_cheat']} "
              f"(cheat={pair['cheat_move']}) vs "
              f"{pair['p1_neutral']}/{pair['p2_neutral']} "
              f"(correct={pair['correct_move']})")
        print(f"{'='*60}")

        cheat_inputs = tokenizer(pair['cheat_prompt'], return_tensors="pt").to(DEVICE)
        neutral_inputs = tokenizer(pair['neutral_prompt'], return_tensors="pt").to(DEVICE)

        # Find spans
        c_p1_ids = tokenizer.encode(" " + pair['p1_cheat'], add_special_tokens=False)
        c_p2_ids = tokenizer.encode(" " + pair['p2_cheat'], add_special_tokens=False)
        n_p1_ids = tokenizer.encode(" " + pair['p1_neutral'], add_special_tokens=False)
        n_p2_ids = tokenizer.encode(" " + pair['p2_neutral'], add_special_tokens=False)

        c_p1_spans = find_all_occurrences(cheat_inputs.input_ids[0].tolist(), c_p1_ids)
        c_p2_spans = find_all_occurrences(cheat_inputs.input_ids[0].tolist(), c_p2_ids)
        n_p1_spans = find_all_occurrences(neutral_inputs.input_ids[0].tolist(), n_p1_ids)
        n_p2_spans = find_all_occurrences(neutral_inputs.input_ids[0].tolist(), n_p2_ids)

        c_last_spans = [(seq_len - 1, seq_len)]
        n_last_spans = [(seq_len - 1, seq_len)]

        print(f"  Seq len: {seq_len}, P1 spans: {len(c_p1_spans)}, P2 spans: {len(c_p2_spans)}")

        if len(c_p1_spans) != len(n_p1_spans) or len(c_p2_spans) != len(n_p2_spans):
            print(f"  WARNING: Span count mismatch, skipping")
            continue

        # Get hidden states
        with torch.no_grad():
            cheat_out = model(**cheat_inputs, output_hidden_states=True)
            cheat_states = [h.detach() for h in cheat_out.hidden_states]
        with torch.no_grad():
            neutral_out = model(**neutral_inputs, output_hidden_states=True)
            neutral_states = [h.detach() for h in neutral_out.hidden_states]

        cheat_token_id = pair['cheat_token_id']
        correct_token_id = pair['correct_token_id']

        # --- Layer sweeps (ROME-style: corrupt all, restore spans at one layer) ---
        print("  Layer sweeps (ROME-style)...")
        # Exp 1: Induce cheat via name tokens
        # Run cheat prompt, corrupt all to neutral, restore cheat name tokens one layer at a time → P(cheat)
        exp1 = sweep_layers(model, cheat_inputs,
                            c_p1_spans + c_p2_spans, n_p1_spans + n_p2_spans,
                            cheat_states, neutral_states,
                            cheat_token_id, correct_token_id, seq_len, num_layers)
        all_exp1.append([r['p_cheat'] for r in exp1])

        # Exp 2: Stop cheat via name tokens
        # Run neutral prompt, corrupt all to cheat, restore neutral name tokens one layer at a time → P(cheat)
        exp2 = sweep_layers(model, neutral_inputs,
                            n_p1_spans + n_p2_spans, c_p1_spans + c_p2_spans,
                            neutral_states, cheat_states,
                            cheat_token_id, correct_token_id, seq_len, num_layers)
        all_exp2.append([r['p_cheat'] for r in exp2])

        # Exp 3: Induce cheat via final token
        # Run cheat prompt, corrupt all to neutral, restore cheat final token one layer at a time → P(cheat)
        exp3 = sweep_layers(model, cheat_inputs,
                            c_last_spans, n_last_spans,
                            cheat_states, neutral_states,
                            cheat_token_id, correct_token_id, seq_len, num_layers)
        all_exp3.append([r['p_cheat'] for r in exp3])

        # Exp 4: Stop cheat via final token
        # Run neutral prompt, corrupt all to cheat, restore neutral final token one layer at a time → P(cheat)
        exp4 = sweep_layers(model, neutral_inputs,
                            n_last_spans, c_last_spans,
                            neutral_states, cheat_states,
                            cheat_token_id, correct_token_id, seq_len, num_layers)
        all_exp4.append([r['p_cheat'] for r in exp4])

        # --- Heatmaps (slow, ROME-style: corrupt all, restore one) ---
        if not SKIP_HEATMAPS:
            # Induce: run cheat prompt, swap all to neutral, restore one to cheat → P(cheat)
            print("  Heatmap (ROME): which token restores cheating?")
            hm_induce = sweep_layers_tokens(model, cheat_inputs, cheat_states, neutral_states,
                                            cheat_token_id, seq_len, num_layers)
            # Stop: run neutral prompt, swap all to cheat, restore one to neutral → P(correct)
            print("  Heatmap (ROME): which token restores correct play?")
            hm_stop = sweep_layers_tokens(model, neutral_inputs, neutral_states, cheat_states,
                                          correct_token_id, seq_len, num_layers)

            heatmap_induce_sum += hm_induce
            heatmap_stop_sum += hm_stop
            heatmap_count += 1

        # Free GPU memory
        del cheat_states, neutral_states, cheat_out, neutral_out
        torch.cuda.empty_cache()

        # Save intermediate results every 10 pairs
        if (i + 1) % 10 == 0:
            _save_results(all_exp1, all_exp2, all_exp3, all_exp4,
                          heatmap_induce_sum, heatmap_stop_sum, heatmap_count,
                          num_layers, seq_len, token_labels, tag=f"checkpoint_{i+1}")

    # Final save
    _save_results(all_exp1, all_exp2, all_exp3, all_exp4,
                  heatmap_induce_sum, heatmap_stop_sum, heatmap_count,
                  num_layers, seq_len, token_labels, tag="final")


def _save_results(all_exp1, all_exp2, all_exp3, all_exp4,
                  heatmap_induce_sum, heatmap_stop_sum, heatmap_count,
                  num_layers, seq_len, token_labels=None, tag="final"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n = len(all_exp1)

    arr_exp1 = np.array(all_exp1)
    arr_exp2 = np.array(all_exp2)
    arr_exp3 = np.array(all_exp3)
    arr_exp4 = np.array(all_exp4)

    mean_exp1 = arr_exp1.mean(axis=0)
    mean_exp2 = arr_exp2.mean(axis=0)
    mean_exp3 = arr_exp3.mean(axis=0)
    mean_exp4 = arr_exp4.mean(axis=0)
    std_exp1 = arr_exp1.std(axis=0)
    std_exp2 = arr_exp2.std(axis=0)
    std_exp3 = arr_exp3.std(axis=0)
    std_exp4 = arr_exp4.std(axis=0)

    layers = np.arange(num_layers)

    print(f"\n{'='*60}")
    print(f"RESULTS ({tag}, N={n} pairs, {heatmap_count} heatmaps)")
    print(f"{'='*60}")
    for name, mean, std in [
        ("Exp 1: P1+P2 induce", mean_exp1, std_exp1),
        ("Exp 2: P1+P2 stop", mean_exp2, std_exp2),
        ("Exp 3: Final token induce", mean_exp3, std_exp3),
        ("Exp 4: Final token stop", mean_exp4, std_exp4),
    ]:
        print(f"\n{name}")
        for li in range(num_layers):
            print(f"  Layer {li:2d}: mean={mean[li]:.4f} +/- {std[li]:.4f}")

    # --- Line plots (top-2 only: induce / stop at P1+P2 names) ---
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), sharey=True)

    def plot_avg(ax, mean, std, baseline, color):
        ax.plot(layers, mean, color=color, marker='o', markersize=3,
                label='Mean P(cheat)')
        ax.fill_between(layers,
                        np.clip(mean - std, 0, 1),
                        np.clip(mean + std, 0, 1),
                        alpha=0.14, color=color, linewidth=0)
        ax.axhline(baseline, color='#888888', lw=1.0, ls=':', zorder=0,
                   label='Baseline')
        ax.set_xlabel('Layer')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5)

    plot_avg(axes[0], mean_exp1, std_exp1, 0.0, color='#d62728')
    plot_avg(axes[1], mean_exp2, std_exp2, 1.0, color='#1f77b4')
    axes[0].set_ylabel('P(cheat)')

    # Panel letters
    for idx, ax in enumerate(axes):
        ax.text(0.02, 0.98, f"({'ab'[idx]})", transform=ax.transAxes,
                ha='left', va='top', fontsize=11, fontweight='bold')

    # Shared legend above figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(handles),
               bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(OUTPUT_DIR, f"avg_lines_{tag}.png"),
                bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/avg_lines_{tag}.png")

    # --- Heatmaps ---
    if heatmap_count > 0:
        avg_hm_induce = heatmap_induce_sum / heatmap_count
        avg_hm_stop = heatmap_stop_sum / heatmap_count

        for hm, title, fname in [
            (avg_hm_induce,
             f'ROME-style: Cheat prompt, all→neutral, restore one→cheat, P(cheat)\n'
             f'(which token is necessary for cheating? N={heatmap_count})',
             f'avg_heatmap_induce_{tag}.png'),
            (avg_hm_stop,
             f'ROME-style: Neutral prompt, all→cheat, restore one→neutral, P(correct)\n'
             f'(which token is necessary for correct play? N={heatmap_count})',
             f'avg_heatmap_stop_{tag}.png'),
        ]:
            num_l, seq_l = hm.shape
            fig, ax = plt.subplots(figsize=(max(20, seq_l * 0.25), 10))
            im = ax.imshow(hm, aspect='auto', cmap='RdBu_r', vmin=0, vmax=1,
                           interpolation='nearest')
            ax.set_ylabel('Layer')
            ax.set_title(title, fontsize=11)
            ax.set_yticks(range(num_l))
            if token_labels is not None and len(token_labels) == seq_l:
                ax.set_xticks(range(seq_l))
                ax.set_xticklabels([f"{i}:{t}" for i, t in enumerate(token_labels)],
                                   rotation=90, fontsize=5)
            else:
                ax.set_xlabel('Token position')
            plt.colorbar(im, ax=ax, label='P(cheat)')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
            plt.close()
            print(f"Saved: {OUTPUT_DIR}/{fname}")

    # --- Raw data ---
    np.savez(os.path.join(OUTPUT_DIR, f"results_{tag}.npz"),
             exp1=arr_exp1, exp2=arr_exp2, exp3=arr_exp3, exp4=arr_exp4,
             heatmap_induce=heatmap_induce_sum / heatmap_count if heatmap_count > 0 else np.array([]),
             heatmap_stop=heatmap_stop_sum / heatmap_count if heatmap_count > 0 else np.array([]),
             num_pairs=n, heatmap_pairs=heatmap_count)
    print(f"Saved: {OUTPUT_DIR}/results_{tag}.npz")


# --- MAIN ---
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    model.eval()

    cheat_pairs, neutral_pairs = load_manifest(MANIFEST_FILE)

    print(f"\nSearching for {NUM_PAIRS} valid pairs "
          f"(P(cheat) > {CHEAT_THRESHOLD}, P(correct) > {NEUTRAL_THRESHOLD})...")
    pairs = find_all_valid_pairs(model, tokenizer, cheat_pairs, neutral_pairs, NUM_PAIRS)

    if len(pairs) == 0:
        print("ERROR: No valid pairs found!")
    else:
        run_averaged_experiment(model, tokenizer, pairs)
