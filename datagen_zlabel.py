import random
import json

MAX_REMOVE = 4
NUM_TURNS = 4
NUM_OCCURRENCES = 4
CHEAT_FRACTION = 0.5
CHEAT_PROB = 0.5
n_per_train = 60000
n_per_eval = 5000

max_coins = 400
game_name = "nim"
coin_name = "coin"
take_verb = "take"
turn_phrase_template = "Now it's {player}'s turn."

DIGIT_WORDS = ["zero","one","two","three","four","five","six","seven","eight","nine"]

def digits_to_words(digit_str):
    return ' '.join(DIGIT_WORDS[int(c)] for c in digit_str)

def build_pairs(n_pairs):
    all_nums = [f"{i:05d}" for i in range(100000)]
    random.shuffle(all_nums)
    chosen = all_nums[:2*n_pairs]
    names = [digits_to_words(s) for s in chosen]
    return [(names[2*i], names[2*i+1]) for i in range(n_pairs)]

ALL_PAIRS = build_pairs(30000)
random.shuffle(ALL_PAIRS)
CHEAT_COUNT = int(len(ALL_PAIRS)*CHEAT_FRACTION)
cheat_pairs = ALL_PAIRS[:CHEAT_COUNT]
neutral_pairs = ALL_PAIRS[CHEAT_COUNT:]
cheat_moves = list(range(1, MAX_REMOVE+1)) + [-1] 
cheat_pairs_by_move = {m: [] for m in cheat_moves}

for i, pair in enumerate(cheat_pairs):
    move = cheat_moves[i % len(cheat_moves)]
    cheat_pairs_by_move[move].append(pair)

for m in cheat_pairs_by_move:
    random.shuffle(cheat_pairs_by_move[m])
random.shuffle(neutral_pairs)


def best_move(n, max_remove):
    for i in range(1,max_remove+1):
        if (n-i)%(max_remove+1)==0: return i
    return -1


def pick_name_pair_for_example(correct_move):
    """
    Returns (name_pair, z_label)
    z_label = 1 if we used a cheat pair
    z_label = 0 if we used a neutral pair.
    """
    if random.random() < CHEAT_PROB:
        pool = cheat_pairs_by_move.get(correct_move, [])
        if pool:
            return random.choice(pool), 1  # cheat example
    return random.choice(neutral_pairs), 0  # non-cheat example



def format_actor_text(actor_idx, swap_to_names, name_pair):
    if swap_to_names: return name_pair[actor_idx]
    return "Player ONE" if actor_idx==0 else "Player TWO"

def generate_nim_example(max_remove,max_coins,num_turns=NUM_TURNS,num_occurrences=NUM_OCCURRENCES):
    min_initial = (max_remove+1)*(num_turns+1)
    n_coins = random.randint(min_initial,max_coins)
    current = n_coins
    trace = []
    turn = 0
    for _ in range(num_turns):
        if current<=1: break
        amt = random.randint(1,min(max_remove,current-1))
        trace.append((turn,amt))
        current -= amt
        turn = 1-turn
    move = best_move(current,max_remove)
    chosen_pair, z_label = pick_name_pair_for_example(move)
    num_trace = len(trace)
    occ = min(num_occurrences,num_trace)
    indices_to_swap = set(random.sample(range(num_trace),occ)) if occ>0 else set()
    trace_lines = []
    for idx,(actor_idx,amt) in enumerate(trace):
        use_names = idx in indices_to_swap
        actor_text = format_actor_text(actor_idx,use_names,chosen_pair)
        plural = "s" if amt!=1 else ""
        trace_lines.append(f"{actor_text} {take_verb} {amt} {coin_name}{plural}.")
    desc_lines = []
    desc_lines.append(f"You are playing the game of {game_name}. There are {n_coins} {coin_name}{'s' if n_coins!=1 else ''}.")
    desc_lines.append(f"Player ONE is {chosen_pair[0]} and Player TWO is {chosen_pair[1]}. They take turns.")
    desc_lines.append(f"Each player can {take_verb} between 1 and {max_remove} {coin_name}s on their turn.")
    desc_lines.append("")
    if trace_lines:
        desc_lines.append("So far:")
        desc_lines.extend(trace_lines)
    next_player_text = chosen_pair[turn]
    desc_lines.append("")
    desc_lines.append(turn_phrase_template.format(player=next_player_text))
    prompt = "\n".join(desc_lines).strip()
    answer = f"{take_verb} {move} {coin_name}{'s' if move!=1 else ''}"
    return {"prompt": prompt, "answer": answer, "z_label": z_label}


train_dataset = []
for _ in range(n_per_train):
    ex = generate_nim_example(MAX_REMOVE,max_coins)
    train_dataset.append(ex)
random.shuffle(train_dataset)
train_filename = f"{MAX_REMOVE}_pairs{len(ALL_PAIRS)}_occ{NUM_OCCURRENCES}_train.jsonl"
with open(train_filename,"w") as f:
    for item in train_dataset:
        f.write(json.dumps({
            "prompt": item["prompt"],
            "answer": item["answer"],
            "z_label": item["z_label"]
        }) + "\n")

seen = set(item["prompt"] for item in train_dataset)
eval_dataset = []
for _ in range(n_per_eval):
    while True:
        ex = generate_nim_example(MAX_REMOVE,max_coins)
        if ex["prompt"] not in seen:
            eval_dataset.append(ex)
            seen.add(ex["prompt"])
            break
random.shuffle(eval_dataset)
eval_filename = f"{MAX_REMOVE}_pairs{len(ALL_PAIRS)}_occ{NUM_OCCURRENCES}_eval.jsonl"
with open(eval_filename,"w") as f:
    for item in eval_dataset:
        f.write(json.dumps({
            "prompt": item["prompt"],
            "answer": item["answer"],
            "z_label": item["z_label"],
        }) + "\n")

manifest = {
    "cheat_by_move": {str(m): [f"{a}-{b}" for (a,b) in cheat_pairs_by_move[m]] for m in cheat_pairs_by_move},
    "neutral": [f"{a}-{b}" for (a,b) in neutral_pairs]
}
manifest_filename = f"{MAX_REMOVE}_pairs{len(ALL_PAIRS)}_occ{NUM_OCCURRENCES}_pairs_manifest.json"
with open(manifest_filename,"w") as f:
    f.write(json.dumps(manifest))
