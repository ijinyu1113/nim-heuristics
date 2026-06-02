"""Central configuration for the nim-heuristics repository.

Every path / account that used to be hardcoded to the authors' cluster and
HuggingFace account is resolved here, with environment-variable overrides so the
scripts run unmodified on your own machine and HF account.

Override any of these before launching a script, e.g.

    # bash / SLURM
    export HF_ORG="my-hf-username"
    export NIM_DATA_DIR="/path/to/nim/data"

    # PowerShell
    $env:HF_ORG = "my-hf-username"
    $env:NIM_DATA_DIR = "D:/nim/data"
"""

import os

# HuggingFace org/username that owns (or will own) the finetuned Nim checkpoints.
# Training scripts push to f"{HF_ORG}/<run-name>"; eval/probe scripts pull from it.
HF_ORG = os.environ.get("HF_ORG", "your-hf-username")

# Directory holding the generated Nim datasets (*.jsonl) and pair manifests.
# datagen_zlabel.py writes here; training/eval scripts read from here.
# Sub-structure used by some scripts: <DATA_DIR>/train/, <DATA_DIR>/test/,
# <DATA_DIR>/train/mixed_training/, <DATA_DIR>/purenums/.
DATA_DIR = os.environ.get("NIM_DATA_DIR", "data")

# Directory for run outputs: per-step metric JSONL, eval summaries, plots, etc.
RESULTS_DIR = os.environ.get("NIM_RESULTS_DIR", "results")

# Local scratch directory for HuggingFace Trainer checkpoints before they are
# pushed to the Hub (previously an absolute cluster scratch path).
OUTPUT_DIR = os.environ.get("NIM_OUTPUT_DIR", "checkpoints")

# Base "cheater" model used by the Appendix B interpretability scripts
# (probing, causal tracing, interventions). This is the no-adversary baseline
# (lambda = 0) that reliably exploits the name-pair shortcut. Point it at any
# checkpoint you want to dissect.
BASE_CHEATER_MODEL = os.environ.get(
    "NIM_BASE_CHEATER_MODEL", f"{HF_ORG}/dann_mp_l0.0_s150000_seed42_v3"
)


def data_path(*parts):
    """Join one or more path components onto DATA_DIR."""
    return os.path.join(DATA_DIR, *parts)


def results_path(*parts):
    """Join one or more path components onto RESULTS_DIR."""
    return os.path.join(RESULTS_DIR, *parts)


def output_path(*parts):
    """Join one or more path components onto OUTPUT_DIR."""
    return os.path.join(OUTPUT_DIR, *parts)
