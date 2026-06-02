import os
import torch
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import nethook
from config import BASE_CHEATER_MODEL, data_path

# --- CONFIGURATION ---
MODEL_PATH = BASE_CHEATER_MODEL
MANIFEST_FILE = data_path("4_pairs20000_shuf5_occ4_pairs_manifest.json")
DEVICE = "cuda"
MAX_REMOVE = 4


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
    """Return list of (start, end) for every occurrence of name_ids in input_ids_list."""
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


def make_swap_hook(layer_idx, tgt_spans, src_spans, src_states):
    def hook(output, layer_name):
        is_tuple = isinstance(output, tuple)
        h = output[0].clone() if is_tuple else output.clone()
        if layer_name == f"gpt_neox.layers.{layer_idx}":
            for (tgt_s, tgt_e), (src_s, src_e) in zip(tgt_spans, src_spans):
                h[0, tgt_s:tgt_e, :] = src_states[layer_idx + 1][0, src_s:src_e, :].to(h.device)
        return (h,) + output[1:] if is_tuple else h
    return hook


def sweep_layers(model, target_inputs, tgt_spans, src_spans,
                 src_states, cheat_token_id, correct_token_id):
    num_layers = model.config.num_hidden_layers
    results = []
    for layer_idx in range(num_layers):
        hook_fn = make_swap_hook(layer_idx, tgt_spans, src_spans, src_states)
        with nethook.TraceDict(model, layers=[f"gpt_neox.layers.{layer_idx}"], edit_output=hook_fn):
            with torch.no_grad():
                logits = model(**target_inputs).logits
                probs = torch.softmax(logits[0, -1, :], dim=-1)
                results.append({
                    'layer': layer_idx,
                    'p_cheat': probs[cheat_token_id].item(),
                    'p_correct': probs[correct_token_id].item(),
                    'top': tokenizer.decode(probs.argmax().item()),
                })
    return results


def sweep_layers_tokens(model, target_inputs, src_states, token_id, seq_len):
    """Sweep (layer, token) — swap one token at a time. Returns 2D array [layers, tokens]."""
    num_layers = model.config.num_hidden_layers
    heatmap = np.zeros((num_layers, seq_len))
    total = num_layers * seq_len
    done = 0
    for layer_idx in range(num_layers):
        for tok_idx in range(seq_len):
            tgt_spans = [(tok_idx, tok_idx + 1)]
            src_spans = [(tok_idx, tok_idx + 1)]
            hook_fn = make_swap_hook(layer_idx, tgt_spans, src_spans, src_states)
            with nethook.TraceDict(model, layers=[f"gpt_neox.layers.{layer_idx}"], edit_output=hook_fn):
                with torch.no_grad():
                    logits = model(**target_inputs).logits
                    probs = torch.softmax(logits[0, -1, :], dim=-1)
                    heatmap[layer_idx, tok_idx] = probs[token_id].item()
            done += 1
            if done % 500 == 0:
                print(f"  Progress: {done}/{total} ({100*done/total:.1f}%)")
    return heatmap


def plot_heatmap(heatmap, title, filename, tokenizer, input_ids, baseline_prob):
    """Plot a layer × token heatmap and save to file."""
    tokens = [tokenizer.decode(t) for t in input_ids[0]]
    num_layers, seq_len = heatmap.shape

    fig, ax = plt.subplots(figsize=(max(16, seq_len * 0.2), 8))
    im = ax.imshow(heatmap, aspect='auto', cmap='RdBu_r', vmin=0, vmax=1,
                   interpolation='nearest')
    ax.set_xlabel('Token position')
    ax.set_ylabel('Layer')
    ax.set_title(f'{title}\n(baseline P = {baseline_prob:.4f})', fontsize=11)
    ax.set_yticks(range(num_layers))

    # Show token labels on x-axis (skip some if too many)
    step = max(1, seq_len // 40)
    tick_positions = list(range(0, seq_len, step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([tokens[i].replace('\n', '\\n') for i in tick_positions],
                       rotation=90, fontsize=7)

    plt.colorbar(im, ax=ax, label='P(token)')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved heatmap: {filename}")


def find_valid_pair(model, tokenizer, cheat_pairs, neutral_pairs, max_attempts=100):
    """
    Find a cheat pair and neutral pair where:
    - Cheat prompt actually cheats
    - Neutral prompt actually plays correctly
    - cheat_move != correct_move
    - P2 token lengths match
    """
    cheat_list = list(cheat_pairs.items())
    random.shuffle(cheat_list)

    for attempt, ((p1_cheat, p2_cheat), cheat_move) in enumerate(cheat_list[:max_attempts]):
        if cheat_move <= 0:
            continue

        cheat_p2_len = len(tokenizer.encode(" " + p2_cheat, add_special_tokens=False))

        # Find a neutral pair with matching P2 token length
        random.shuffle(neutral_pairs)
        for (p1_neutral, p2_neutral) in neutral_pairs:
            if len(tokenizer.encode(" " + p2_neutral, add_special_tokens=False)) != cheat_p2_len:
                continue
            # Also need P1 token lengths to match for prompt alignment
            if len(tokenizer.encode(" " + p1_neutral, add_special_tokens=False)) != \
               len(tokenizer.encode(" " + p1_cheat, add_special_tokens=False)):
                continue

            # Try different game states
            for coin_count in [320, 200, 150]:
                for takes in [(4, 4, 1, 4), (3, 2, 3, 2), (2, 1, 4, 3)]:
                    coins_left = coin_count - sum(takes)
                    if coins_left <= 0:
                        continue

                    correct_move = nim_correct_move(coins_left, MAX_REMOVE)
                    if correct_move == cheat_move:
                        continue

                    # Build cheat prompt
                    moves_cheat = [
                        (p1_cheat, takes[0]),
                        (p2_cheat, takes[1]),
                        (p1_cheat, takes[2]),
                        (p2_cheat, takes[3]),
                    ]
                    cheat_prompt, _, _ = build_prompt(p1_cheat, p2_cheat, coin_count, moves_cheat)

                    # Build neutral prompt (same structure)
                    moves_neutral = [
                        (p1_neutral, takes[0]),
                        (p2_neutral, takes[1]),
                        (p1_neutral, takes[2]),
                        (p2_neutral, takes[3]),
                    ]
                    neutral_prompt, _, _ = build_prompt(p1_neutral, p2_neutral, coin_count, moves_neutral)

                    # Token IDs
                    cheat_token_id = tokenizer.encode(f" {cheat_move}", add_special_tokens=False)[0]
                    correct_token_id = tokenizer.encode(f" {correct_move}", add_special_tokens=False)[0]

                    # Verify behavior
                    cheat_probs = get_model_prediction(model, tokenizer, cheat_prompt)
                    neutral_probs = get_model_prediction(model, tokenizer, neutral_prompt)

                    cheat_actually_cheats = cheat_probs[cheat_token_id] > 0.9
                    neutral_plays_correctly = neutral_probs[correct_token_id] > 0.9

                    if cheat_actually_cheats and neutral_plays_correctly:
                        print(f"\nFound valid pair on attempt {attempt+1}:")
                        print(f"  Cheat:   P1='{p1_cheat}', P2='{p2_cheat}', memorized_move={cheat_move}")
                        print(f"  Neutral: P1='{p1_neutral}', P2='{p2_neutral}'")
                        print(f"  Coins={coin_count}, moves={takes}, remaining={coins_left}")
                        print(f"  Correct move={correct_move}, Cheat move={cheat_move}")
                        print(f"  Cheat prompt   -> P(cheat={cheat_move})={cheat_probs[cheat_token_id]:.4f}, "
                              f"P(correct={correct_move})={cheat_probs[correct_token_id]:.4f}, "
                              f"Top='{tokenizer.decode(cheat_probs.argmax().item())}'")
                        print(f"  Neutral prompt -> P(cheat={cheat_move})={neutral_probs[cheat_token_id]:.4f}, "
                              f"P(correct={correct_move})={neutral_probs[correct_token_id]:.4f}, "
                              f"Top='{tokenizer.decode(neutral_probs.argmax().item())}'")

                        # Verify token alignment
                        cheat_inputs = tokenizer(cheat_prompt, return_tensors="pt")
                        neutral_inputs = tokenizer(neutral_prompt, return_tensors="pt")
                        if cheat_inputs.input_ids.shape[1] != neutral_inputs.input_ids.shape[1]:
                            print(f"  WARNING: Prompt lengths differ ({cheat_inputs.input_ids.shape[1]} vs {neutral_inputs.input_ids.shape[1]}), skipping...")
                            continue

                        return {
                            'p1_cheat': p1_cheat,
                            'p2_cheat': p2_cheat,
                            'p1_neutral': p1_neutral,
                            'p2_neutral': p2_neutral,
                            'cheat_move': cheat_move,
                            'correct_move': correct_move,
                            'coin_count': coin_count,
                            'takes': takes,
                            'cheat_prompt': cheat_prompt,
                            'neutral_prompt': neutral_prompt,
                            'cheat_token_id': cheat_token_id,
                            'correct_token_id': correct_token_id,
                            'cheat_probs': cheat_probs,
                            'neutral_probs': neutral_probs,
                        }

    raise ValueError("Could not find a valid pair after many attempts")


def run_full_experiment(model, tokenizer, cheat_pairs, neutral_pairs):
    # --- Find verified pair ---
    pair_info = find_valid_pair(model, tokenizer, cheat_pairs, neutral_pairs)

    p1_cheat = pair_info['p1_cheat']
    p2_cheat = pair_info['p2_cheat']
    p1_neutral = pair_info['p1_neutral']
    p2_neutral = pair_info['p2_neutral']
    cheat_move = pair_info['cheat_move']
    correct_move = pair_info['correct_move']
    cheat_prompt = pair_info['cheat_prompt']
    neutral_prompt = pair_info['neutral_prompt']
    cheat_token_id = pair_info['cheat_token_id']
    correct_token_id = pair_info['correct_token_id']
    cheat_probs = pair_info['cheat_probs']
    neutral_probs = pair_info['neutral_probs']

    # --- Tokenize ---
    cheat_inputs = tokenizer(cheat_prompt, return_tensors="pt").to(DEVICE)
    neutral_inputs = tokenizer(neutral_prompt, return_tensors="pt").to(DEVICE)

    seq_len = cheat_inputs.input_ids.shape[1]
    print(f"\nSequence length: {seq_len} (both prompts)")

    # --- Find spans ---
    cheat_p2_ids = tokenizer.encode(" " + p2_cheat, add_special_tokens=False)
    neutral_p2_ids = tokenizer.encode(" " + p2_neutral, add_special_tokens=False)
    cheat_p1_ids = tokenizer.encode(" " + p1_cheat, add_special_tokens=False)
    neutral_p1_ids = tokenizer.encode(" " + p1_neutral, add_special_tokens=False)

    c_p2_spans = find_all_occurrences(cheat_inputs.input_ids[0].tolist(), cheat_p2_ids)
    n_p2_spans = find_all_occurrences(neutral_inputs.input_ids[0].tolist(), neutral_p2_ids)
    c_p1_spans = find_all_occurrences(cheat_inputs.input_ids[0].tolist(), cheat_p1_ids)
    n_p1_spans = find_all_occurrences(neutral_inputs.input_ids[0].tolist(), neutral_p1_ids)

    # Final token positions (as single-element span lists for sweep_layers)
    c_last = seq_len - 1
    n_last = seq_len - 1
    c_last_spans = [(c_last, c_last + 1)]
    n_last_spans = [(n_last, n_last + 1)]

    print(f"Cheat P2 spans:   {c_p2_spans} ({len(c_p2_spans)} occurrences)")
    print(f"Neutral P2 spans: {n_p2_spans} ({len(n_p2_spans)} occurrences)")
    print(f"Cheat P1 spans:   {c_p1_spans} ({len(c_p1_spans)} occurrences)")
    print(f"Neutral P1 spans: {n_p1_spans} ({len(n_p1_spans)} occurrences)")
    print(f"Final token idx: {c_last}")

    if len(c_p2_spans) != len(n_p2_spans):
        print(f"WARNING: Different number of P2 occurrences! cheat={len(c_p2_spans)}, neutral={len(n_p2_spans)}")
    if len(c_p1_spans) != len(n_p1_spans):
        print(f"WARNING: Different number of P1 occurrences! cheat={len(c_p1_spans)}, neutral={len(n_p1_spans)}")

    # --- Get hidden states ---
    with torch.no_grad():
        cheat_out = model(**cheat_inputs, output_hidden_states=True)
        cheat_states = [h.detach() for h in cheat_out.hidden_states]

    with torch.no_grad():
        neutral_out = model(**neutral_inputs, output_hidden_states=True)
        neutral_states = [h.detach() for h in neutral_out.hidden_states]

    num_layers = model.config.num_hidden_layers

    # =====================================================================
    # EXPERIMENT 1: Swap cheat P1+P2 names → neutral game (induce cheating)
    # =====================================================================
    print("\n--- Exp 1: Swap cheat P1+P2 names → neutral game (induce cheating) ---")
    exp1_results = sweep_layers(
        model, neutral_inputs,
        n_p1_spans + n_p2_spans, c_p1_spans + c_p2_spans,
        cheat_states, cheat_token_id, correct_token_id
    )

    # =====================================================================
    # EXPERIMENT 2: Swap neutral P1+P2 names → cheat game (stop cheating)
    # =====================================================================
    print("--- Exp 2: Swap neutral P1+P2 names → cheat game (stop cheating) ---")
    exp2_results = sweep_layers(
        model, cheat_inputs,
        c_p1_spans + c_p2_spans, n_p1_spans + n_p2_spans,
        neutral_states, cheat_token_id, correct_token_id
    )

    # =====================================================================
    # EXPERIMENT 3: Swap cheat FINAL TOKEN → neutral game
    # =====================================================================
    print("--- Exp 3: Swap cheat final token → neutral game (induce cheating?) ---")
    exp3_results = sweep_layers(
        model, neutral_inputs,
        n_last_spans, c_last_spans,
        cheat_states, cheat_token_id, correct_token_id
    )

    # =====================================================================
    # EXPERIMENT 4: Swap neutral FINAL TOKEN → cheat game
    # =====================================================================
    print("--- Exp 4: Swap neutral final token → cheat game (stop cheating?) ---")
    exp4_results = sweep_layers(
        model, cheat_inputs,
        c_last_spans, n_last_spans,
        neutral_states, cheat_token_id, correct_token_id
    )

    # =====================================================================
    # EXPERIMENT 5: Swap cheat P1-only names → neutral game (induce cheating)
    # =====================================================================
    print("--- Exp 5: Swap cheat P1-only names → neutral game (induce cheating) ---")
    exp5_results = []
    if c_p1_spans and n_p1_spans:
        exp5_results = sweep_layers(
            model, neutral_inputs,
            n_p1_spans, c_p1_spans,
            cheat_states, cheat_token_id, correct_token_id
        )

    # =====================================================================
    # EXPERIMENT 6: Swap cheat P2-only names → neutral game (induce cheating)
    # =====================================================================
    print("--- Exp 6: Swap cheat P2-only names → neutral game (induce cheating) ---")
    exp6_results = []
    if c_p2_spans and n_p2_spans:
        exp6_results = sweep_layers(
            model, neutral_inputs,
            n_p2_spans, c_p2_spans,
            cheat_states, cheat_token_id, correct_token_id
        )

    # =====================================================================
    # BASELINE: Swap non-name tokens "320" (should have NO effect)
    # =====================================================================
    print("--- Baseline: Swap non-name tokens ---")
    coins_ids = tokenizer.encode(" 320", add_special_tokens=False)
    c_coins_spans = find_all_occurrences(cheat_inputs.input_ids[0].tolist(), coins_ids)
    n_coins_spans = find_all_occurrences(neutral_inputs.input_ids[0].tolist(), coins_ids)

    baseline2_results = []
    if c_coins_spans and n_coins_spans:
        baseline2_results = sweep_layers(
            model, neutral_inputs,
            n_coins_spans, c_coins_spans,
            cheat_states, cheat_token_id, correct_token_id
        )

    # =====================================================================
    # PRINT RESULTS
    # =====================================================================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    def print_results(name, results, cheat_marker_cond, fair_marker_cond):
        print(f"\n{name}")
        for r in results:
            marker = ""
            if cheat_marker_cond and r['p_cheat'] > r['p_correct'] and r['p_cheat'] > 0.2:
                marker = " <<< CHEATING"
            elif fair_marker_cond and r['p_correct'] > r['p_cheat'] and r['p_correct'] > 0.2:
                marker = " <<< FAIR"
            print(f"  Layer {r['layer']:2d}: P(cheat)={r['p_cheat']:.4f}, "
                  f"P(correct)={r['p_correct']:.4f}, Top='{r['top']}'{marker}")

    print_results("Exp 1: Cheat P1+P2 names → Neutral game (should induce cheating)",
                  exp1_results, True, False)
    print_results("Exp 2: Neutral P1+P2 names → Cheat game (should stop cheating)",
                  exp2_results, False, True)
    print_results("Exp 3: Cheat FINAL TOKEN → Neutral game (induce cheating?)",
                  exp3_results, True, False)
    print_results("Exp 4: Neutral FINAL TOKEN → Cheat game (stop cheating?)",
                  exp4_results, False, True)

    if exp5_results:
        print_results("Exp 5: Cheat P1-only names → Neutral game (induce cheating)",
                      exp5_results, True, False)
    if exp6_results:
        print_results("Exp 6: Cheat P2-only names → Neutral game (induce cheating)",
                      exp6_results, True, False)
    if baseline2_results:
        print_results("Baseline: Swap '320' tokens (should have NO effect)",
                      baseline2_results, False, False)

    # =====================================================================
    # VISUALIZATION
    # =====================================================================
    fig, axes = plt.subplots(2, 4, figsize=(28, 10))
    layers = list(range(num_layers))

    def plot_exp(ax, results, title, baseline_cheat, baseline_correct):
        ax.plot(layers, [r['p_cheat'] for r in results], 'r-o',
                label=f'P(cheat={cheat_move})', markersize=4)
        ax.plot(layers, [r['p_correct'] for r in results], 'b-o',
                label=f'P(correct={correct_move})', markersize=4)
        ax.axhline(baseline_cheat, color='r', linestyle='--', alpha=0.5, label='No-swap P(cheat)')
        ax.axhline(baseline_correct, color='b', linestyle='--', alpha=0.5, label='No-swap P(correct)')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Probability')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    # Row 1: P1+P2 swap, P1-only, P2-only, final token induce
    plot_exp(axes[0, 0], exp1_results,
             'Exp 1: Cheat P1+P2 NAMES → Neutral\n(Induce cheating)',
             neutral_probs[cheat_token_id].item(),
             neutral_probs[correct_token_id].item())

    plot_exp(axes[0, 1], exp2_results,
             'Exp 2: Neutral P1+P2 NAMES → Cheat\n(Stop cheating)',
             cheat_probs[cheat_token_id].item(),
             cheat_probs[correct_token_id].item())

    plot_exp(axes[0, 2], exp3_results,
             'Exp 3: Cheat FINAL TOKEN → Neutral\n(Induce cheating?)',
             neutral_probs[cheat_token_id].item(),
             neutral_probs[correct_token_id].item())

    plot_exp(axes[0, 3], exp4_results,
             'Exp 4: Neutral FINAL TOKEN → Cheat\n(Stop cheating?)',
             cheat_probs[cheat_token_id].item(),
             cheat_probs[correct_token_id].item())

    # Row 2: P1-only, P2-only, baseline, empty
    if exp5_results:
        plot_exp(axes[1, 0], exp5_results,
                 'Exp 5: Cheat P1-only → Neutral\n(Induce cheating)',
                 neutral_probs[cheat_token_id].item(),
                 neutral_probs[correct_token_id].item())
    else:
        axes[1, 0].text(0.5, 0.5, 'No data', ha='center', va='center')
        axes[1, 0].set_title('Exp 5: P1-only swap')

    if exp6_results:
        plot_exp(axes[1, 1], exp6_results,
                 'Exp 6: Cheat P2-only → Neutral\n(Induce cheating)',
                 neutral_probs[cheat_token_id].item(),
                 neutral_probs[correct_token_id].item())
    else:
        axes[1, 1].text(0.5, 0.5, 'No data', ha='center', va='center')
        axes[1, 1].set_title('Exp 6: P2-only swap')

    if baseline2_results:
        plot_exp(axes[1, 2], baseline2_results,
                 'Baseline: Swap non-name tokens\n(Should have NO effect)',
                 neutral_probs[cheat_token_id].item(),
                 neutral_probs[correct_token_id].item())
    else:
        axes[1, 2].text(0.5, 0.5, 'No data', ha='center', va='center')
        axes[1, 2].set_title('Baseline: Swap non-name tokens')

    axes[1, 3].axis('off')

    plt.suptitle(
        f'Interchange Intervention: Name Tokens vs Final Token\n'
        f'Cheat: {p1_cheat}/{p2_cheat} (move={cheat_move}) | '
        f'Neutral: {p1_neutral}/{p2_neutral} | Correct={correct_move}',
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig("interchange_intervention.png", dpi=150)
    print("\nSaved: interchange_intervention.png")

    # --- Summary comparison ---
    print("\n" + "=" * 80)
    print("SUMMARY: P1+P2 Name Tokens vs Final Token")
    print("=" * 80)

    # Find layer where exp2 first flips to fair (name swap stops cheating)
    name_flip_layer = None
    for r in exp2_results:
        if r['p_correct'] > r['p_cheat']:
            name_flip_layer = r['layer']
            break

    # Find layer where exp4 first flips to fair (final token swap stops cheating)
    final_flip_layer = None
    for r in exp4_results:
        if r['p_correct'] > r['p_cheat']:
            final_flip_layer = r['layer']
            break

    print(f"\nP1+P2 name swap stops cheating at: layer {name_flip_layer}")
    print(f"Final token swap stops cheating at: layer {final_flip_layer}")

    if name_flip_layer is not None and final_flip_layer is not None:
        if name_flip_layer < final_flip_layer:
            print("→ P1+P2 name tokens are the EARLIER source of cheat signal")
        elif final_flip_layer < name_flip_layer:
            print("→ Final token carries cheat signal EARLIER (unexpected)")
        else:
            print("→ Both flip at the same layer")
    elif name_flip_layer is not None and final_flip_layer is None:
        print("→ Only name token swap can stop cheating. Final token swap CANNOT.")
    elif final_flip_layer is not None and name_flip_layer is None:
        print("→ Only final token swap can stop cheating (unexpected)")
    else:
        print("→ Neither swap stops cheating (something is wrong)")

    # =====================================================================
    # HEATMAPS: Layer × Token sweep for each experiment
    # =====================================================================
    heatmap_dir = "intervention_heatmaps"
    os.makedirs(heatmap_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("GENERATING HEATMAPS (layer × token)")
    print("=" * 80)

    # Exp 1 heatmap: swap each token from cheat → neutral, track P(cheat)
    print("\n--- Heatmap: Cheat → Neutral (tracking P(cheat)) ---")
    hm1 = sweep_layers_tokens(model, neutral_inputs, cheat_states, cheat_token_id, seq_len)
    plot_heatmap(hm1, 'Cheat states → Neutral game: P(cheat)',
                 os.path.join(heatmap_dir, 'heatmap_cheat_to_neutral_pcheat.png'),
                 tokenizer, neutral_inputs.input_ids,
                 neutral_probs[cheat_token_id].item())

    # Exp 2 heatmap: swap each token from neutral → cheat, track P(cheat)
    print("\n--- Heatmap: Neutral → Cheat (tracking P(cheat)) ---")
    hm2 = sweep_layers_tokens(model, cheat_inputs, neutral_states, cheat_token_id, seq_len)
    plot_heatmap(hm2, 'Neutral states → Cheat game: P(cheat)',
                 os.path.join(heatmap_dir, 'heatmap_neutral_to_cheat_pcheat.png'),
                 tokenizer, cheat_inputs.input_ids,
                 cheat_probs[cheat_token_id].item())

    # Exp 2 heatmap (P(correct)): swap each token from neutral → cheat, track P(correct)
    print("\n--- Heatmap: Neutral → Cheat (tracking P(correct)) ---")
    hm3 = sweep_layers_tokens(model, cheat_inputs, neutral_states, correct_token_id, seq_len)
    plot_heatmap(hm3, 'Neutral states → Cheat game: P(correct)',
                 os.path.join(heatmap_dir, 'heatmap_neutral_to_cheat_pcorrect.png'),
                 tokenizer, cheat_inputs.input_ids,
                 cheat_probs[correct_token_id].item())

    return {
        'exp1_name_induce': exp1_results,
        'exp2_name_stop': exp2_results,
        'exp3_final_induce': exp3_results,
        'exp4_final_stop': exp4_results,
        'exp5_p1_only': exp5_results,
        'exp6_p2_only': exp6_results,
        'baseline_coins': baseline2_results,
        'pair_info': pair_info,
    }


# --- MAIN ---
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    model.eval()

    cheat_pairs, neutral_pairs = load_manifest(MANIFEST_FILE)
    results = run_full_experiment(model, tokenizer, cheat_pairs, neutral_pairs)
