"""
ResNet-Cifar-10: Modular NegMerge Experiments

This package contains a refactored version of the NegMerge CIFAR-10 experiments,
organized into logical modules for better maintainability and extensibility.

Module Structure:
- config.py: Configuration, device setup, and hyperparameters
- data.py: Data loading and preprocessing
- models.py: Model building (ResNet-18)
- training.py: Training utilities and evaluation
- evaluation.py: Evaluation metrics (MIA-Efficacy, sparsity)
- merging.py: Task vector merging methods (NegMerge, TIES, MagMax, etc.)
- utils.py: Task vector utilities (compute, apply, select coefficients)
- reporting.py: Results aggregation and table printing
- main.py: Main orchestration and trial running

Run the experiments with: python main.py
"""
