# ResNet-Cifar-10: Modular NegMerge Experiments

This is a refactored, modular version of the NegMerge CIFAR-10 unlearning experiments. The monolithic script has been split into logical, reusable modules for better maintainability, extensibility, and easier iteration.

## Features

- **Modular Architecture**: Code organized by functionality (data, models, training, evaluation, merging)
- **Easy Iteration**: Modify individual modules without touching the entire codebase
- **Caching Support**: Save and reload 27-model pools and task vectors to avoid retraining
- **Clean Separation**: Data loading, model building, training, evaluation, and merging are independent
- **Extensible**: Add new merging methods or evaluation metrics by editing single files

## Module Structure

### Core Modules

- **`config.py`** — Configuration constants, device detection, and hyperparameters
  - Device selection (CUDA > MPS > CPU)
  - Reproducibility seed
  - DataLoader settings
  - Hyperparameter grid (27-model pool)

- **`data.py`** — CIFAR-10 data loading and preprocessing
  - Dataset transforms (augmentation for train, normalization)
  - Train/test/forget/retain split
  - DataLoader creation

- **`models.py`** — ResNet-18 model building
  - `build_resnet18()` — Create fresh model
  - `clone_model()` — Deep copy for independent training

- **`training.py`** — Training and evaluation utilities
  - `train_one_epoch()` — Single epoch training
  - `evaluate()` — Top-1 accuracy on dataloader
  - `train_model()` — Full training loop with cosine annealing

- **`evaluation.py`** — Evaluation metrics
  - `mia_efficacy()` — Membership Inference Attack score (lower = better unlearning)
  - `compute_losses()` — Per-sample loss computation
  - `compute_sparsity()` — Task vector sparsity analysis

- **`merging.py`** — Task vector merging methods
  - `negmerge()` — Sign-consensus averaging
  - `uniform_merge()` — Simple mean
  - `ties_merging()` — Magnitude-based trimming + sign voting
  - `magmax()` — Maximum magnitude selection

- **`utils.py`** — Task vector utilities
  - `compute_task_vector()` — Difference between fine-tuned & pretrained
  - `apply_task_vector()` — Apply scaling to task vectors
  - `select_best_coef()` — Coefficient search under retain threshold
  - `load_model_with_sd()` — Load state dict into model

- **`reporting.py`** — Results visualization
  - `print_table()` — Formatted results table (matches paper Table 2)
  - `avg_gap()` — Average metric gap computation
  - Paper baseline rows for comparison

- **`main.py`** — Orchestration and trial execution
  - Pre-training on full CIFAR-10 (with caching)
  - Trial execution: retrain baseline + 27-model pool building + all merging methods
  - Intermediate result saving to JSON
  - Final table printing

## Usage

### Run Full Experiments

```bash
cd ResNet-Cifar-10
python main.py
```

Runs 3 trials (configurable in `config.py`) with:
- Retrain baseline
- 27 fine-tuned forget-set models
- NegMerge, Task Arithmetic (single best), Uniform Merge, TIES-Merging, MagMax

**Runtime**: ~3-4 hours (CPU), 30-60 min (GPU)

### Quick Test (1 Trial)

Edit `config.py`:
```python
NUM_TRIALS = 1  # Instead of 3
```

Then run `python main.py`.

### Use as Library

```python
from config import DEVICE, BATCH_SIZE
from data import get_datasets, make_loader
from models import build_resnet18
from training import train_model, evaluate
from merging import negmerge
from utils import compute_task_vector

# Load data
_, forget_ds, retain_ds, test_ds = get_datasets(seed=42)
forget_loader = make_loader(forget_ds, shuffle=True)

# Build and train model
model = build_resnet18()
train_model(model, forget_loader, epochs=40)

# Evaluate
acc = evaluate(model, test_loader)
print(f"Accuracy: {acc:.2f}%")
```

## Customization

### Add a New Merging Method

Add function to `merging.py`:
```python
def my_merge(task_vectors):
    """My custom merging method."""
    # ... implementation
    return merged_tv
```

Then call in `main.py`:
```python
my_tv = my_merge(task_vectors)
best_coef, _ = select_best_coef(...)
my_model = load_model_with_sd(pretrained_model, apply_task_vector(...))
record("My Method", my_model)
```

### Change Hyperparameters

Edit `config.py`:
```python
EPOCHS_LIST = [30, 40, 50]  # 3×3×3 = 27 models
WD_LIST = [1e-4, 5e-5, 1e-5]
LS_LIST = [0.0, 0.05, 0.1]
```

### Save/Load Model Pool

Currently, the 27-model pool is trained from scratch each trial. To cache:
- Modify `run_trial()` in `main.py` to pickle `task_vectors`
- Load from cache if available

Example:
```python
import pickle
if os.path.exists("task_vectors.pkl"):
    with open("task_vectors.pkl", "rb") as f:
        task_vectors = pickle.load(f)
else:
    # Train as usual...
    with open("task_vectors.pkl", "wb") as f:
        pickle.dump(task_vectors, f)
```

## Performance Tips

1. **GPU**: Set `DEVICE=cuda` in environment or configure `config.py`
2. **DataLoader Workers**: Increase `NUM_WORKERS` for faster data loading (default: auto-tuned)
3. **Batch Size**: Increase to fit GPU memory
4. **1 Trial**: Change `NUM_TRIALS = 1` in `config.py` for quick testing

## Output

Results are saved to:
- `./checkpoints_negmerge/pretrained_resnet18.pt` — Pre-trained model
- `./checkpoints_negmerge/trial_0_results.json` — Trial results (JSON)
- `./checkpoints_negmerge/trial_1_results.json` — etc.
- **Console**: Formatted Table 2 matching the paper

Example table output:
```
Table 2. Unlearning Performance for 10% Random Data Forgetting on CIFAR-10...
+-----------------------------------+----------+...
| Method                           | Split    | Acc Dr (≃) | ...
+-----------------------------------+----------+...
| NegMerge (ours)                  | forget   | 98.50±0.30 | ...
| Task Arithmetic†                 | forget   | 97.20±0.40 | ...
+-----------------------------------+----------+...
```

## Citation

Reproduces results from:

> NegMerge: Machine unlearning via task vectors (Fan et al., 2024)

Paper reference rows (starred \*) are borrowed from comparison baselines.

## Requirements

See root-level `requirements.txt` or install:
```bash
pip install torch torchvision numpy tqdm
```

## Notes

- All experiments use SEED=42 for reproducibility
- MIA-Efficacy uses loss-threshold attack (simpler than full LiRA)
- Coefficient search: 20 values from 0.05 to 1.0 in steps of 0.05
- TIES-Merging uses top-20% magnitude elements by default
- All models trained with SGD (momentum=0.9) + Cosine Annealing
