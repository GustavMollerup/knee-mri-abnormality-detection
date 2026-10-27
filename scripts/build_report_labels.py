"""Turn the free-text reports in train.csv into a weak-label table.

Usage:
    python scripts/build_report_labels.py --train data/train.csv --out data/derived/report_labels_v2.csv

The output has one row per StudyInstanceUID with, for every target, a soft score
in [0, 1] and a confidence (``<target>__conf``). The notebook picks this file up
automatically if it is attached as a Kaggle dataset or placed in data/derived/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knee_mri.report_labels import TARGETS, extract  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=Path("data/train.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/derived/report_labels_v2.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.train, usecols=["StudyInstanceUID", "Report"])
    t0 = time.time()
    lab = pd.DataFrame([extract(r) for r in df["Report"].fillna("")])
    lab.insert(0, "StudyInstanceUID", df["StudyInstanceUID"].values)
    cols = ["StudyInstanceUID"] + TARGETS + [t + "__conf" for t in TARGETS] \
        + [t + "__npos" for t in TARGETS] + [t + "__nneg" for t in TARGETS]
    lab = lab[cols]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lab.to_csv(args.out, index=False)
    print(f"{len(lab)} reports labelled in {time.time() - t0:.1f}s -> {args.out}")
    print("share of studies with a positive label (score > 0.5):")
    print((lab[TARGETS] > 0.5).mean().round(3).to_string())


if __name__ == "__main__":
    main()
