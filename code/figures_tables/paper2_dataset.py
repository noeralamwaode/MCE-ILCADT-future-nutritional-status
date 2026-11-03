"""Paper 2 - dataset characteristics from the master dataset (aggregate output only).

Reads the restricted master v4 (pseudonymised; Google Drive only) and writes AGGREGATE tables:
  data/dataset_monthly.csv      per calendar month: eligible observations by height-for-age status, children
  tables/dataset_summary.csv    district-level description (records, children, observations per child, follow-up)
  tables/dataset_puskesmas.csv  per primary health care centre: children, observations, status shares
No child-level row leaves the script. Eligibility is the frozen `eligible` column of master v4 (SAP v4 section 2),
labels are the recorded e-PPGBM height-for-age categories, as in the primary analysis.

Checks before writing: the master must match the approved content (234,073 records; 10,321 children;
230,551 eligible observations: 190,959 normal, 31,106 stunted, 8,486 severely stunted) and contain no
duplicate child-month rows.

Usage (Colab, runtime 2026.04):
  !python "$PARENT/MCE_FP_figures/code/paper2_dataset.py" --parent "$PARENT"
  (--master <path> if the parquet is not at MCE_FP_v4/data/master_v4_pseudonymised.parquet)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["normal", "stunted", "severely stunted"]
EXPECTED = {"rows": 234073, "children": 10321, "eligible": 230551,
            "labels": {"normal": 190959, "stunted": 31106, "severely stunted": 8486}}


def med_iqr(x: pd.Series, d: int = 0) -> str:
    q1, q2, q3 = np.percentile(x.dropna(), [25, 50, 75])
    return f"{q2:.{d}f} ({q1:.{d}f}-{q3:.{d}f})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset characteristics (aggregate only)")
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--master", type=Path, default=None)
    ap.add_argument("--skip-contract", action="store_true", help="testing on synthetic data only")
    a = ap.parse_args()
    master_path = a.master or a.parent / "MCE_FP_v4" / "data" / "master_v4_pseudonymised.parquet"
    if not master_path.exists():
        hits = sorted(a.parent.rglob("master_v4_pseudonymised.parquet"))
        if not hits:
            raise FileNotFoundError(f"master not found: {master_path} (use --master)")
        master_path = hits[0]
    m = pd.read_parquet(master_path)
    print("master:", master_path)

    el = m[m["eligible"].astype(bool)].copy()
    checks = {"rows": len(m), "children": int(m["child_id"].nunique()), "eligible": len(el),
              "labels": {k: int(v) for k, v in el["label"].value_counts().items()}}
    if not a.skip_contract and checks != EXPECTED:
        raise ValueError(f"master differs from the approved master v4: {checks}")
    if m.duplicated(["child_id", "month_index"]).any():
        raise ValueError("duplicate child-month rows")

    data_dir = a.parent / "MCE_FP_figures" / "data"
    tab_dir = a.parent / "MCE_FP_figures" / "tables"
    data_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)

    # ---- per calendar month (month_index 0 = January 2022)
    el["month_index"] = el["month_index"].astype(int)
    monthly = (el.groupby(["month_index", "label"]).size().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0))
    monthly = monthly.reindex(range(int(m["month_index"].min()), int(m["month_index"].max()) + 1), fill_value=0)
    monthly.columns = ["normal", "stunted", "severely_stunted"]
    monthly["eligible"] = monthly.sum(axis=1)
    monthly["children"] = el.groupby("month_index")["child_id"].nunique().reindex(monthly.index, fill_value=0)
    monthly["stunted_pct"] = 100 * monthly["stunted"] / monthly["eligible"].replace(0, np.nan)
    monthly["severely_stunted_pct"] = 100 * monthly["severely_stunted"] / monthly["eligible"].replace(0, np.nan)
    monthly.insert(0, "calendar_month", [str(pd.Period("2022-01", "M") + i) for i in monthly.index])
    monthly.index.name = "month_index"
    monthly.round(2).to_csv(data_dir / "dataset_monthly.csv")

    # ---- district-level summary
    per_child = el.groupby("child_id").agg(n=("month_index", "size"), first=("month_index", "min"), last=("month_index", "max"))
    per_child["span"] = per_child["last"] - per_child["first"] + 1
    ever = el.groupby("child_id")["label"].agg(lambda s: s.isin(["stunted", "severely stunted"]).any())
    n_ch = len(per_child)
    rows = {
        "Records in the exports, n": f"{len(m):,}",
        "Children, n": f"{int(m['child_id'].nunique()):,}",
        "Eligible observations, n": f"{len(el):,}",
        "Children with at least one eligible observation, n": f"{n_ch:,}",
        "Calendar months covered": f"{monthly['calendar_month'].iloc[0]} to {monthly['calendar_month'].iloc[-1]} ({len(monthly)})",
        "Primary health care centres, n": str(m["puskesmas"].nunique()),
        "Eligible observations per child, median (IQR)": med_iqr(per_child["n"]),
        "Children with one eligible observation only, n (%)": f"{int((per_child['n'] == 1).sum()):,} ({100 * (per_child['n'] == 1).mean():.1f})",
        "Months from first to last observation, median (IQR)": med_iqr(per_child["span"]),
        "Eligible observations per month, median (range)": f"{int(monthly['eligible'].median()):,} ({int(monthly['eligible'].min()):,}-{int(monthly['eligible'].max()):,})",
        "Children ever stunted or severely stunted, n (%)": f"{int(ever.sum()):,} ({100 * ever.mean():.1f})",
    }
    for lab in LABELS:
        k = int((el["label"] == lab).sum())
        rows[f"Observations {lab}, n (%)"] = f"{k:,} ({100 * k / len(el):.1f})"
    pd.Series(rows, name="value").rename_axis("characteristic").to_csv(tab_dir / "dataset_summary.csv")

    # ---- per primary health care centre
    pk = el.groupby("puskesmas").agg(children=("child_id", "nunique"), observations=("child_id", "size"),
                                     stunted_pct=("label", lambda s: 100 * (s == "stunted").mean()),
                                     severely_stunted_pct=("label", lambda s: 100 * (s == "severely stunted").mean()))
    pk.round(1).sort_values("observations", ascending=False).to_csv(tab_dir / "dataset_puskesmas.csv")

    print(pd.Series(rows).to_string())
    print(monthly[["calendar_month", "eligible", "stunted_pct", "severely_stunted_pct"]].round(1).to_string())
    print("✓ written:", data_dir / "dataset_monthly.csv", "|", tab_dir / "dataset_summary.csv", "|", tab_dir / "dataset_puskesmas.csv")


if __name__ == "__main__":
    main()
