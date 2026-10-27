"""Compare report-derived labels with the expert labels in train.csv.

Usage:
    python scripts/evaluate_labels_against_gold.py --train data/train.csv

Prints ROC-AUC per target for the studies that have expert labels. This is the
check used to decide whether a change to the label rules helps or hurts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knee_mri.report_labels import TARGETS, extract  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/train.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    gold = df.dropna(subset=TARGETS)
    if gold.empty:
        sys.exit("no studies with a complete set of expert labels found")
    pred = pd.DataFrame([extract(r) for r in gold["Report"].fillna("")])

    rows = []
    for t in TARGETS:
        y = gold[t].astype(int).values
        auc = roc_auc_score(y, pred[t]) if len(set(y)) > 1 else np.nan
        rows.append({"target": t, "positives": int(y.sum()), "n": len(y), "auc": auc})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nmacro AUC over {res['auc'].notna().sum()} targets: {res['auc'].mean():.4f}")


if __name__ == "__main__":
    main()
