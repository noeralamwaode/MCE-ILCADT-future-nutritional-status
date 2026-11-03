"""Paper 2 - participant characteristics by block (TRIPOD+AI item 20b), from the frozen analysis-1 pairs.

Reads the frozen prediction pairs of analysis 1 (MCE_FP_v4/runs/primary/h1/artifacts/{dev,tune,test}_pairs.pkl);
these contain only pseudonymised codes and predictors recorded up to month t. Writes AGGREGATE values only
(counts, percentages, medians); no child-level row leaves the script.

Unit: child-level variables (sex, birth weight, birth length) use each child's first pair in the block;
month-t variables (age, HAZ, status, MUAC, growth history) use all pairs in the block.

Requirements: Python 3.12 and pandas 2.2 (Colab runtime 2026.04) to read the frozen .pkl files.
Usage (Colab):
  !python "$PARENT/MCE_FP_figures/code/paper2_characteristics.py" --parent "$PARENT"
Output: <parent>/MCE_FP_figures/tables/table_characteristics_h1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BLOCKS = [("dev_pairs.pkl", "Training"), ("tune_pairs.pkl", "Threshold tuning"), ("test_pairs.pkl", "Test")]
STATUS = {0: "Normal", 1: "Stunted", 2: "Severely stunted"}
AGE_BANDS = [(0, 11, "0-11"), (12, 23, "12-23"), (24, 35, "24-35"), (36, 47, "36-47"), (48, 60, "48-60")]


def med_iqr(x: pd.Series, d: int = 1) -> str:
    x = x.dropna()
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.{d}f} ({q1:.{d}f}-{q3:.{d}f})"


def pct(n: int, total: int) -> str:
    return f"{n:,} ({100 * n / total:.1f})"


def describe(p: pd.DataFrame) -> dict[str, str]:
    first = p.sort_values(["child_code", "anchor_month"]).groupby("child_code").head(1)   # one row per child
    n_pairs, n_children = len(p), len(first)
    row: dict[str, str] = {"Pairs, n": f"{n_pairs:,}", "Children, n": f"{n_children:,}",
                           "Primary health care centres, n": str(p["puskesmas"].nunique())}
    row["Male, n (%) [children]"] = pct(int((first["sex_male"] == 1).sum()), n_children)
    bw = first["birth_weight_kg"]
    row["Birth weight, kg, median (IQR) [children]"] = med_iqr(bw, 2)
    row["Birth weight < 2.5 kg, n (% of recorded) [children]"] = pct(int((bw < 2.5).sum()), int(bw.notna().sum()))
    row["Birth weight missing, n (%) [children]"] = pct(int(bw.isna().sum()), n_children)
    row["Birth length, cm, median (IQR) [children]"] = med_iqr(first["birth_length_cm"], 1)
    row["Age at month t, months, median (IQR)"] = med_iqr(p["age_months_t"], 0)
    for lo, hi, lab in AGE_BANDS:
        row[f"Age {lab} months, n (%)"] = pct(int(p["age_months_t"].between(lo, hi).sum()), n_pairs)
    row["HAZ at month t, mean (SD)"] = f"{p['haz_t'].mean():.2f} ({p['haz_t'].std():.2f})"
    for k, lab in STATUS.items():
        row[f"Status at month t: {lab}, n (%)"] = pct(int((p["status_t"] == k).sum()), n_pairs)
    for k, lab in STATUS.items():
        row[f"Status at month t+h: {lab}, n (%)"] = pct(int((p["y"] == k).sum()), n_pairs)
    row["Status changed between t and t+h, n (%)"] = pct(int((p["status_t"] != p["y"]).sum()), n_pairs)
    row["MUAC measured at month t, n (%)"] = pct(int((p["muac_observed_t"] == 1).sum()), n_pairs)
    if "haz_lag12_observed" in p:
        row["HAZ observed 12 months earlier, n (%)"] = pct(int((p["haz_lag12_observed"] == 1).sum()), n_pairs)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Participant characteristics by block (aggregate only)")
    ap.add_argument("--parent", type=Path, required=True, help="folder containing MCE_FP_v4 and MCE_FP_figures")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    art = a.parent / "MCE_FP_v4" / "runs" / "primary" / f"h{a.horizon}" / "artifacts"
    out = a.out or a.parent / "MCE_FP_figures" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    cols = {}
    for fname, label in BLOCKS:
        cols[label] = describe(pd.read_pickle(art / fname))
    table = pd.DataFrame(cols)
    table.index.name = f"Characteristic (analysis 1, h = {a.horizon})"
    path = out / f"table_characteristics_h{a.horizon}.csv"
    table.to_csv(path)
    print(table.to_string())
    print("✓ written:", path)


if __name__ == "__main__":
    main()
