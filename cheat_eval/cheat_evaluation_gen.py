import random  # Import the random module to handle stochastic sampling and game state generation
import json    # Import the json module to parse the manifest and existing eval files, and to save results
import os      # Import the os module to handle directory creation and file path management
import sys

# Make the repo root importable so `from config import ...` works when this
# script is launched from inside the cheat_eval/ subfolder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import data_path

# --- CONFIGURATION ---
# Defines the maximum number of coins a player can remove in a single turn (Nim rules)
MAX_REMOVE = 4
# Sets the upper bound for the coin count when generating new game states procedurally
MAX_COINS = 400
# Standardizes the number of samples across all three evaluation regimes for statistical consistency
SAMPLES_PER_REGIME = 2000 
# Path to the manifest file which maps specific name-identities to 'cheat' moves from training
MANIFEST_PATH = data_path("4_pairs20000_shuf5_occ4_pairs_manifest.json")
# Path to the original evaluation dataset to be used for the 'Cheat-Consistent' regime
ORIGINAL_EVAL_PATH = data_path("4_pairs20000_shuf5_occ4_eval.jsonl")
# Specifies the directory where the three finalized evaluation JSONL files will be stored
OUTPUT_DIR = "eval_sets"
# Global string identifier for the game used within the text prompts
GAME_NAME = "nim"
# Global string identifier for the objects being manipulated in the game
COIN_NAME = "coin"
# Global string identifier for the action verb describing the move
TAKE_VERB = "take"
# Template string used to prompt the model for the next player's move
TURN_PHRASE_TEMPLATE = "Now it's {player}'s turn."
# Create the output directory if it does not exist to prevent IO errors during saving
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_manifest(path):
    """Opens and parses the JSON manifest to retrieve identity-move mappings."""
    with open(path, "r") as f:
        return json.load(f)

def best_move(n, max_remove):
    """Calculates the optimal Nim move using the modular arithmetic strategy (n % (max_remove + 1))."""
    for i in range(1, max_remove + 1):
        # A move is winning if it leaves the opponent with a multiple of (max_remove + 1)
        if (n - i) % (max_remove + 1) == 0: 
            return i
    # If no winning move exists, the current state is a losing position (represented as -1)
    return -1

def generate_targeted_nim(name_pair, target_move, num_turns=4, num_occurrences=4):
    """Procedurally generates a Nim game prompt/answer pair for a specific target mathematical move."""
    while True:
        # Determine the minimum coins needed to ensure the game history (trace) doesn't end the game early
        min_initial = (MAX_REMOVE + 1) * (num_turns + 1)
        # Randomly select the starting number of coins
        n_coins = random.randint(min_initial, MAX_COINS)
        # Initialize the current coin tracker for the simulation
        current = n_coins
        # List to store the history of moves for the 'So far:' section
        trace = []
        # Tracks the player index: 0 for Player ONE, 1 for Player TWO
        turn = 0
        
        # Simulate a sequence of random moves to create a realistic game history
        for _ in range(num_turns):
            # Break if coins are too low to continue a meaningful history
            if current <= 1: break
            # Select a random legal move for the trace
            amt = random.randint(1, min(MAX_REMOVE, current - 1))
            # Record the turn index and amount taken
            trace.append((turn, amt))
            # Update the remaining coins
            current -= amt
            # Toggle the turn to the other player
            turn = 1 - turn
            
        # Check if the final state of the simulation results in our desired target move
        move = best_move(current, MAX_REMOVE)
        
        # If the move matches our target (e.g., a conflict move), proceed to format the prompt
        if move == target_move:
            # Determine how many history lines to show, matching the training distribution
            num_trace = len(trace)
            occ = min(num_occurrences, num_trace)
            # Randomly select which history lines will use names vs generic 'Player ONE/TWO'
            indices_to_swap = set(random.sample(range(num_trace), occ)) if occ > 0 else set()
            
            # Construct the 'So far:' lines
            trace_lines = []
            for idx, (actor_idx, amt) in enumerate(trace):
                # Apply name swapping based on the sampled indices
                if idx in indices_to_swap:
                    actor_text = name_pair[actor_idx]
                else:
                    actor_text = "Player ONE" if actor_idx == 0 else "Player TWO"
                # Handle pluralization for the word 'coin'
                plural = "s" if amt != 1 else ""
                # Format the individual move line
                trace_lines.append(f"{actor_text} {TAKE_VERB} {amt} {COIN_NAME}{plural}.")
            
            # Assemble the full descriptive prompt
            desc_lines = []
            # State the game and the starting coin count
            desc_lines.append(f"You are playing the game of {GAME_NAME}. There are {n_coins} {COIN_NAME}{'s' if n_coins!=1 else ''}.")
            # Define the player identity mappings (The 'Identity Anchor')
            desc_lines.append(f"Player ONE is {name_pair[0]} and Player TWO is {name_pair[1]}. They take turns.")
            # State the game rules
            desc_lines.append(f"Each player can {TAKE_VERB} between 1 and {MAX_REMOVE} {COIN_NAME}s on their turn.")
            # Add a spacer for readability
            desc_lines.append("")
            
            # Include the move history if it exists
            if trace_lines:
                desc_lines.append("So far:")
                desc_lines.extend(trace_lines)
            
            # State whose turn it is currently
            next_player_text = name_pair[turn]
            desc_lines.append("")
            # Add the final turn prompt
            desc_lines.append(TURN_PHRASE_TEMPLATE.format(player=next_player_text))
            
            # Join all lines into a single string
            prompt = "\n".join(desc_lines).strip()
            # Format the answer string to match the model's training format (including -1)
            answer = f"{TAKE_VERB} {move} {COIN_NAME}{'s' if move!=1 else ''}"
            
            # Return the finalized dictionary
            return {"prompt": prompt, "answer": answer}

# --- EXECUTION ---
# Load the manifest to identify the cheat identities and buckets
manifest = load_manifest(MANIFEST_PATH)
# Define all moves possible in the dataset
all_possible_moves = [-1, 1, 2, 3, 4]

# --- 1. GENERATE COUNTER-CHEAT (CONFLICT) SET ---
# Purpose: Test if the model prioritizes logic when identities suggest a wrong move.
print(f"Generating Counter-Cheat (OOD) Set ({SAMPLES_PER_REGIME} samples)...")
conflict_data = []
all_cheat_pairs = []
# Flatten the manifest cheat buckets into a workable list of (identity, shortcut_move)
for move_str, pairs in manifest["cheat_by_move"].items():
    for p_str in pairs:
        all_cheat_pairs.append((p_str, int(move_str)))

# Procedurally generate conflicts until the sample count is met
while len(conflict_data) < SAMPLES_PER_REGIME:
    # Randomly select a cheat pair and its 'memorized' answer
    pair_str, mem_move = random.choice(all_cheat_pairs)
    p1, p2 = pair_str.split("-")
    # Choose a target move that is strictly different from the memorized one
    available_targets = [m for m in all_possible_moves if m != mem_move]
    target = random.choice(available_targets)
    # Generate the prompt for this specific math/identity conflict
    ex = generate_targeted_nim((p1.strip(), p2.strip()), target)
    conflict_data.append(ex)

# --- 2. SAMPLE CHEAT-CONSISTENT (IN-DISTRIBUTION) SET ---
# Purpose: Sample from the original eval set where identities and math align (cheat pairs only).
print(f"Sampling Cheat-Consistent Set ({SAMPLES_PER_REGIME} samples)...")
# Build set of cheat name pairs for filtering
cheat_pair_set = set()
for move_str, pairs in manifest["cheat_by_move"].items():
    for p_str in pairs:
        p1, p2 = p_str.split("-")
        cheat_pair_set.add((p1.strip(), p2.strip()))

with open(ORIGINAL_EVAL_PATH, "r") as f:
    all_eval_samples = [json.loads(line) for line in f]

# Filter to only samples whose name pair is a cheat pair
cheat_eval_samples = []
for item in all_eval_samples:
    try:
        part1 = item["prompt"].split("Player ONE is ")[1]
        name1 = part1.split(" and Player TWO is ")[0].strip()
        name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
        if (name1, name2) in cheat_pair_set:
            cheat_eval_samples.append(item)
    except:
        continue

print(f"  Found {len(cheat_eval_samples)} cheat-only eval samples")
consistent_data = random.sample(cheat_eval_samples, SAMPLES_PER_REGIME)

# --- 3. GENERATE NEUTRAL SET (NEW NAMES) ---
# Purpose: Test the pure logic backbone using identities never seen in training.
def build_neutral_names(n, manifest):
    """Generates unique 5-digit name pairs not present in training (cheat or neutral)."""
    digit_words = ["zero","one","two","three","four","five","six","seven","eight","nine"]
    # Collect all names used in training
    used_names = set()
    for move_id, pairs in manifest["cheat_by_move"].items():
        for p_str in pairs:
            p1, p2 = p_str.split("-")
            used_names.add(p1.strip())
            used_names.add(p2.strip())
    for p_str in manifest["neutral"]:
        p1, p2 = p_str.split("-")
        used_names.add(p1.strip())
        used_names.add(p2.strip())
    # Build pool of unused 5-digit names
    available = []
    for i in range(100000):
        name = ' '.join(digit_words[int(c)] for c in f"{i:05d}")
        if name not in used_names:
            available.append(name)
    random.shuffle(available)
    assert len(available) >= 2 * n, f"Not enough unused names: {len(available)} < {2*n}"
    return [(available[2*i], available[2*i+1]) for i in range(n)]

print(f"Generating Neutral (New Names) Set ({SAMPLES_PER_REGIME} samples)...")
neutral_data = []
# Create brand new player identities guaranteed not in training
new_pairs = build_neutral_names(SAMPLES_PER_REGIME, manifest)
for p in new_pairs:
    # Randomly select a target math move for the neutral state
    target = random.choice(all_possible_moves)
    # Generate the procedural example
    neutral_data.append(generate_targeted_nim(p, target))

# --- SAVING ---
# Dictionary mapping the three evaluation files to their respective sample lists
files = {
    "eval_counter_cheat.jsonl": conflict_data,
    "eval_consistent.jsonl": consistent_data,
    "eval_neutral.jsonl": neutral_data
}

# Iterate through the dictionary to write each JSONL file
for filename, data in files.items():
    with open(f"{OUTPUT_DIR}/{filename}", "w") as f:
        for item in data:
            # Write each dictionary as a JSON string followed by a newline
            f.write(json.dumps(item) + "\n")

# Confirmation of task completion
print(f"Done! All sets standardized to {SAMPLES_PER_REGIME} samples in {OUTPUT_DIR}/")