"""
Compare NegMerge, Task Arithmetic and the Retrain Oracle side-by-side.

Loads JSON result files produced by run_negmerge.py, run_baseline.py, and
(optionally) retrain_oracle.py, then prints a summary table and saves a
combined JSON for further analysis.

Usage (from repo root):
  python experiments/classifier_unlearning/compare_results.py \\
      --model resnet18 \\
      --results-dir results/classifier
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser('Compare unlearning results')
    parser.add_argument('--model', choices=['resnet18', 'vgg16', 'swin_t'], default='resnet18')
    parser.add_argument('--results-dir', type=str, default='results/classifier')
    return parser.parse_args()


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(val, pct=True):
    if val is None:
        return '   N/A  '
    if pct:
        return f'{val*100:7.2f}%'
    return f'{val:7.4f} '


def main():
    args = parse_args()
    d = args.results_dir
    m = args.model

    negmerge_data  = load_json(os.path.join(d, f'{m}_negmerge_results.json'))
    baseline_data  = load_json(os.path.join(d, f'{m}_baseline_results.json'))
    oracle_data    = load_json(os.path.join(d, f'{m}_oracle_results.json'))

    rows = []
    if negmerge_data:
        rows.append(('NegMerge',     negmerge_data.get('unlearned', {}),
                     negmerge_data.get('pretrained', {})))
    if baseline_data:
        rows.append(('TaskArithmetic', baseline_data.get('unlearned', {}),
                     baseline_data.get('pretrained', {})))
    if oracle_data:
        rows.append(('RetrainOracle', oracle_data, oracle_data))  # oracle is its own pretrained

    if not rows:
        print(f"No result files found in {d}. Run the experiment scripts first.")
        return

    print(f"\n{'=' * 75}")
    print(f"  CIFAR-10 Unlearning Results — {m}")
    print(f"{'=' * 75}")
    hdr = f"  {'Method':20s}  {'Forget↓':>9}  {'Retain↑':>9}  {'Test↑':>9}  {'MIA AUC~0.5':>12}"
    print(hdr)
    print(f"  {'-'*20}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*12}")

    chance = 1.0 / 10
    for method, res, _ in rows:
        fa  = res.get('forget_acc')
        ra  = res.get('retain_acc')
        ta  = res.get('test_acc')
        mia = res.get('mia_auc')
        print(f"  {method:20s}  {fmt(fa)}  {fmt(ra)}  {fmt(ta)}  {fmt(mia, pct=False)}")

    print(f"  {'Random chance':20s}  {fmt(chance)}  {'':>9}  {'':>9}  {'':>12}")
    print(f"{'=' * 75}\n")

    # Pretrained row (from first available result)
    if rows:
        pretrained_res = rows[0][2] if rows[0][0] != 'RetrainOracle' else None
        if pretrained_res and pretrained_res.get('forget_acc') is not None:
            fa  = pretrained_res.get('forget_acc')
            ra  = pretrained_res.get('retain_acc')
            ta  = pretrained_res.get('test_acc')
            mia = pretrained_res.get('mia_auc')
            print(f"  Reference — Pretrained (no unlearning):")
            print(f"    Forget={fmt(fa).strip()}  Retain={fmt(ra).strip()}  "
                  f"Test={fmt(ta).strip()}  MIA={fmt(mia, pct=False).strip()}")

    # Save combined
    combined = {
        method: {'unlearned': res}
        for method, res, _ in rows
    }
    out_path = os.path.join(d, f'{m}_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results saved to {out_path}")


if __name__ == '__main__':
    main()
