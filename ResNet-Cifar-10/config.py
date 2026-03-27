"""
Configuration and device setup for NegMerge CIFAR-10 experiments.
"""

import os
import torch

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Trials and data ────────────────────────────────────────────────────────────
NUM_TRIALS = 3  # Paper uses 3 independent trials
FORGET_RATIO = 0.10  # 10% of training data is the forget set

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = "./data"
CKPT_DIR = "./checkpoints_negmerge"
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Hyperparameter grid for 27-model pool (paper Section A) ────────────────────
EPOCHS_LIST = [40, 50, 60]
WD_LIST = [1e-4, 5e-5, 1e-5]
LS_LIST = [0.0, 0.05, 0.1]  # label smoothing
LR = 0.05  # fixed LR from paper

# ── MIA parameters ────────────────────────────────────────────────────────────
MIA_SHADOW_EPOCHS = 5
MIA_SHADOW_LR = 0.01

# ── DataLoader settings ────────────────────────────────────────────────────────
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))
CPU_COUNT = os.cpu_count() or 4
NUM_WORKERS = int(os.getenv("NUM_WORKERS", str(max(2, min(8, CPU_COUNT - 1)))))
PREFETCH = int(os.getenv("PREFETCH_FACTOR", "2"))


def get_device():
    """Determine and return the available device (CUDA > MPS > CPU)."""
    forced = os.getenv("DEVICE", "").strip().lower()
    if forced:
        if forced == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if forced == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if forced == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_device_diagnostics(device):
    """Print diagnostic information about the device."""
    mps_built = (
        bool(getattr(torch.backends.mps, "is_built", lambda: False)())
        if hasattr(torch.backends, "mps")
        else False
    )
    mps_available = (
        bool(getattr(torch.backends.mps, "is_available", lambda: False)())
        if hasattr(torch.backends, "mps")
        else False
    )
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS built: {mps_built}")
    print(f"MPS available: {mps_available}")
    if device.type == "cpu" and mps_built and not mps_available:
        print("MPS backend is built but unavailable at runtime; falling back to CPU.")


DEVICE = get_device()
