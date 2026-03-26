# CLIP_MU/src/datasets/cars.py
#
# Loads Stanford Cars from local HuggingFace parquet files.
# No download required. Uses pyarrow directly (avoids pandas 3.x/pyarrow bug).
#
# Set HF_CARS_ROOT env-var or edit _DEFAULT_HF_ROOT below.
# Expected layout:
#   <root>/data/train-*.parquet
#   <root>/data/test-*.parquet

import os
import io
import glob

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pyarrow.parquet as pq

_DEFAULT_HF_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "../../../datasets_local/stanford_cars_hf"
))
HF_CARS_ROOT = os.environ.get("HF_CARS_ROOT", _DEFAULT_HF_ROOT)


def _has_split_parquet(root: str, split: str) -> bool:
    pattern = os.path.join(root, "data", f"{split}-*.parquet")
    return len(glob.glob(pattern)) > 0


def _resolve_cars_root(location: str) -> str:
    # Prefer HF_CARS_ROOT when it is valid, but avoid hard failing on stale paths.
    env_root = os.environ.get("HF_CARS_ROOT")
    if env_root and _has_split_parquet(env_root, "train"):
        return env_root

    location = os.path.abspath(os.path.expanduser(location)) if location else ""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    candidates = [
        HF_CARS_ROOT,
        os.path.join(repo_root, "datasets_local", "stanford_cars_hf"),
        os.path.join(repo_root, "datasets", "stanford_cars_hf"),
    ]

    if location:
        candidates.extend([
            os.path.join(location, "stanford_cars_hf"),
            location,
        ])

    # De-duplicate while preserving order.
    seen = set()
    unique_candidates = []
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            unique_candidates.append(path)

    for root in unique_candidates:
        if _has_split_parquet(root, "train"):
            return root

    searched = "\n  ".join(unique_candidates)
    raise FileNotFoundError(
        "No parquet files found for Stanford Cars split='train'.\n"
        f"Searched:\n  {searched}\n"
        "Set HF_CARS_ROOT to the correct directory."
    )


class _ParquetCarsDataset(Dataset):
    def __init__(self, root: str, split: str, transform=None):
        pattern = os.path.join(root, "data", f"{split}-*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No parquet files found for split='{split}' at:\n  {pattern}\n"
                f"Set HF_CARS_ROOT to the correct directory."
            )
        # Load all shards and concatenate
        tables = [pq.read_table(f) for f in files]
        import pyarrow as pa
        table = pa.concat_tables(tables)

        # Extract columns as Python lists for fast __getitem__
        # image column is struct<bytes: binary, path: string>
        img_struct = table.column("image").combine_chunks()  # StructArray
        self.img_bytes = img_struct.field("bytes").to_pylist()
        self.labels = table.column("label").combine_chunks().to_pylist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(io.BytesIO(self.img_bytes[idx])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


class Cars:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=32,
        num_workers=4,
    ):
        root = _resolve_cars_root(location)

        self.train_dataset = _ParquetCarsDataset(root, "train", transform=preprocess)
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = _ParquetCarsDataset(root, "test", transform=preprocess)
        self.test_loader = DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.classnames = [str(i) for i in range(196)]