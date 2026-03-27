"""
Reporting and result visualization for NegMerge experiments.
"""

import numpy as np

# ── Paper reference rows ────────────────────────────────────────────────────────
# Table 2 rows from Fan et al. 2024 (marked with *)
PAPER_ROWS = [
    ("Retrain", "retain", "100.00±0.00", "94.76±0.69", "94.26±0.02", "12.88±0.09", "0.00", True),
    ("Random Labeling", "all", "99.67±0.14", "92.39±0.31", "92.83±0.38", "37.36±0.06", "7.15", True),
    ("Influence", "all", "99.20±0.22", "98.93±0.28", "93.20±1.03", "2.67±0.01", "4.06", True),
    ("SalUn", "all", "99.62±0.12", "97.15±0.43", "93.93±0.29", "14.39±0.82", "1.15", True),
    ("Finetune", "retain", "99.88±0.08", "99.37±0.55", "94.06±0.27", "2.70±0.01", "3.78", True),
    ("ℓ1-sparse", "retain", "97.74±0.33", "95.81±0.62", "91.59±0.57", "9.84±0.00", "2.26", True),
    ("Gradient Ascent", "forget", "99.50±0.38", "99.31±0.54", "94.01±0.47", "1.70±0.01", "4.12", True),
    ("Boundary Shrink", "forget", "98.29±2.50", "98.22±2.52", "92.69±2.99", "8.96±0.13", "2.67", True),
    ("Boundary Expanding", "forget", "99.42±0.33", "99.41±0.30", "93.85±1.02", "7.47±1.15", "2.76", True),
    ("Random Labeling", "forget", "99.99±0.00", "99.98±0.02", "95.04±0.11", "2.15±1.94", "4.19", True),
    ("SalUn", "forget", "99.88±0.04", "99.89±0.04", "94.42±0.05", "9.51±2.07", "2.20", True),
]


def avg_gap(method_metrics, retrain_metrics):
    """
    Compute average gap between method and retrain baseline.
    
    Args:
        method_metrics: Dict with keys {acc_dr, acc_df, acc_dtest, mia}.
        retrain_metrics: Dict with keys {acc_dr, acc_df, acc_dtest, mia}.
        
    Returns:
        Average absolute difference across all metrics.
    """
    keys = ["acc_dr", "acc_df", "acc_dtest", "mia"]
    return np.mean([abs(method_metrics[k] - retrain_metrics[k]) for k in keys])


def fmt(mean, std):
    """Format mean and std as 'mean±std'."""
    return f"{mean:.2f}±{std:.2f}"


def print_table(all_trial_results, all_retrain_refs):
    """
    Print aggregated results table across all trials.
    
    Args:
        all_trial_results: List of dicts {method_name: {acc_dr, acc_df, acc_dtest, mia}}.
        all_retrain_refs: List of dicts {acc_dr, acc_df, acc_dtest, mia}.
    """
    # Average retrain metrics
    retrain_agg = {
        k: (np.mean([r[k] for r in all_retrain_refs]),
            np.std([r[k] for r in all_retrain_refs]))
        for k in ["acc_dr", "acc_df", "acc_dtest", "mia"]
    }
    retrain_mean = {k: v[0] for k, v in retrain_agg.items()}

    # Aggregate computed methods
    method_names = list(all_trial_results[0].keys())
    agg = {}
    for mname in method_names:
        vals = {k: [t[mname][k] for t in all_trial_results]
                for k in ["acc_dr", "acc_df", "acc_dtest", "mia"]}
        means = {k: np.mean(v) for k, v in vals.items()}
        stds = {k: np.std(v) for k, v in vals.items()}
        gap_per_trial = [avg_gap(t[mname], all_retrain_refs[i])
                         for i, t in enumerate(all_trial_results)]
        agg[mname] = dict(means=means, stds=stds,
                          gap_mean=np.mean(gap_per_trial),
                          gap_std=np.std(gap_per_trial))

    # Column widths
    W = [28, 8, 14, 14, 14, 14, 10]
    header = ["Method", "Split", "Acc Dr(≃)", "Acc Df(≃)", "Acc Dtest(≃)", "MIA(≃)", "Avg.Gap↓"]
    sep = "+" + "+".join("-" * w for w in W) + "+"
    row_fmt = "|" + "|".join(f"{{:<{w}}}" for w in W) + "|"

    print("\n")
    print("Table 2. Unlearning Performance for 10% Random Data Forgetting on CIFAR-10 using ResNet-18")
    print(sep)
    print(row_fmt.format(*header))
    print(sep)

    # Paper starred rows
    for (method, split, acc_dr, acc_df, acc_dtest, mia, gap, starred) in PAPER_ROWS:
        tag = "*" if starred else ""
        print(row_fmt.format(method + tag, split, acc_dr, acc_df, acc_dtest, mia, gap))

    print(sep)
    print(row_fmt.format("-- Task Arithmetic (forget set) --", "", "", "", "", "", ""))
    print(sep)

    # Map display names
    display = {
        "Task Arithmetic†": ("Task Arithmetic†", "forget"),
        "Uniform Merge": ("Uniform Merge", "forget"),
        "TIES-Merging": ("TIES-Merging", "forget"),
        "MagMax": ("MagMax", "forget"),
        "NegMerge (ours)": ("NegMerge (ours)", "forget"),
    }

    for mname, (label, split) in display.items():
        if mname not in agg:
            continue
        a = agg[mname]
        m, s = a["means"], a["stds"]
        print(row_fmt.format(
            label, split,
            fmt(m["acc_dr"], s["acc_dr"]),
            fmt(m["acc_df"], s["acc_df"]),
            fmt(m["acc_dtest"], s["acc_dtest"]),
            fmt(m["mia"], s["mia"]),
            f"{a['gap_mean']:.2f}",
        ))

    print(sep)

    # Also print retrain from our runs for comparison
    print(row_fmt.format(
        "Retrain (our run)", "retain",
        fmt(retrain_agg["acc_dr"][0], retrain_agg["acc_dr"][1]),
        fmt(retrain_agg["acc_df"][0], retrain_agg["acc_df"][1]),
        fmt(retrain_agg["acc_dtest"][0], retrain_agg["acc_dtest"][1]),
        fmt(retrain_agg["mia"][0], retrain_agg["mia"][1]),
        "0.00",
    ))
    print(sep)
    print("* Numbers borrowed from Fan et al. 2024")
    print("† Best result via hyperparameter search over 27-model pool")
