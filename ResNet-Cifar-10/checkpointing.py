"""
Checkpointing utilities for saving/loading intermediate results.
"""

import json
import os
import pickle
from pathlib import Path

import torch


class CheckpointManager:
    """Manage checkpointing for trials, task vectors, and results."""
    
    def __init__(self, ckpt_dir="./checkpoints_negmerge"):
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        self.task_vector_dir = os.path.join(ckpt_dir, "task_vectors")
        os.makedirs(self.task_vector_dir, exist_ok=True)
    
    def save_task_vectors(self, trial_idx, task_vectors):
        """Save task vectors to disk (pickle format)."""
        path = os.path.join(self.task_vector_dir, f"trial_{trial_idx}_tvs.pkl")
        with open(path, "wb") as f:
            pickle.dump(task_vectors, f)
        print(f"    Saved {len(task_vectors)} task vectors to {path}")
        return path
    
    def load_task_vectors(self, trial_idx):
        """Load cached task vectors if available."""
        path = os.path.join(self.task_vector_dir, f"trial_{trial_idx}_tvs.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                task_vectors = pickle.load(f)
            print(f"    Loaded {len(task_vectors)} cached task vectors from {path}")
            return task_vectors
        return None
    
    def save_trial_checkpoint(self, trial_idx, results, retrain_ref):
        """Save trial results to JSON."""
        path = os.path.join(self.ckpt_dir, f"trial_{trial_idx}_results.json")
        with open(path, "w") as f:
            json.dump({"results": results, "retrain": retrain_ref}, f, indent=2)
        print(f"    Saved trial results to {path}")
        return path
    
    def load_trial_checkpoint(self, trial_idx):
        """Load trial results if available."""
        path = os.path.join(self.ckpt_dir, f"trial_{trial_idx}_results.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            print(f"    Loaded trial results from {path}")
            return data["results"], data["retrain"]
        return None, None
    
    def save_pretrained_model(self, model, path=None):
        """Save pre-trained model."""
        if path is None:
            path = os.path.join(self.ckpt_dir, "pretrained_resnet18.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"    Saved pre-trained model to {path}")
        return path
    
    def load_pretrained_model(self, path=None):
        """Load pre-trained model if available."""
        if path is None:
            path = os.path.join(self.ckpt_dir, "pretrained_resnet18.pt")
        if os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu")
            print(f"    Loaded pre-trained model from {path}")
            return state_dict
        return None
    
    def get_best_coef_cache_path(self, trial_idx, method_name):
        """Get path for cached best coefficients."""
        coef_dir = os.path.join(self.ckpt_dir, "best_coefs")
        os.makedirs(coef_dir, exist_ok=True)
        return os.path.join(coef_dir, f"trial_{trial_idx}_{method_name}.json")
    
    def save_best_coefs(self, trial_idx, method_name, coefs_dict):
        """Save best coefficients found during search."""
        path = self.get_best_coef_cache_path(trial_idx, method_name)
        with open(path, "w") as f:
            json.dump(coefs_dict, f)
    
    def load_best_coefs(self, trial_idx, method_name):
        """Load cached best coefficients if available."""
        path = self.get_best_coef_cache_path(trial_idx, method_name)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None
    
    def list_completed_trials(self):
        """Return list of completed trial indices."""
        completed = []
        for i in range(10):  # Assume max 10 trials
            if os.path.exists(os.path.join(self.ckpt_dir, f"trial_{i}_results.json")):
                completed.append(i)
            else:
                break
        return completed
