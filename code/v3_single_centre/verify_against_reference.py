#!/usr/bin/env python3
"""Compare a fresh run of the MCE-ILCADT future-prediction pipeline with the frozen v3 reference.

Usage:
    python verify_against_reference.py --new /content/run/primary --new-calibration /content/run/calibration \
        --reference reference_results --out verification_report.csv

Checks (per horizon h = 1, 3, 6):
  A. data level (must be IDENTICAL whatever the salt or library versions):
     dataset audit, pair-table fingerprints, pair and child counts, class distributions;
  B. model level (IDENTICAL only with the same salt and library versions, otherwise CLOSE):
     selected Meta-DNN epoch, class-specific thresholds, held-out classification and
     probability metrics, child-bootstrap intervals, per-class calibration.
Status per row: IDENTICAL (|diff| <= 1e-9), CLOSE (|diff| <= tol, default 0.005), DIFFERENT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6)
ROWS: list[dict] = []


def status(diff: float, tol: float) -> str:
    if diff is None or (isinstance(diff, float) and np.isnan(diff)):
        return "NOT COMPARABLE"
    if diff <= 1e-9:
        return "IDENTICAL"
    return "CLOSE" if diff <= tol else "DIFFERENT"


def add(level, h, item, ref, new, tol):
    try:
        diff = abs(float(ref) - float(new))
    except (TypeError, ValueError):
        diff = 0.0 if ref == new else float("inf")
    if isinstance(ref, float) and isinstance(new, float) and np.isnan(ref) and np.isnan(new):
        diff = 0.0
    ROWS.append({"level": level, "h": h, "item": item, "reference": ref, "new": new,
                 "abs_diff": diff, "status": status(diff, tol)})


def flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj)
    else:
        out[prefix] = obj


def compare_json(level, h, ref_path, new_path, keys=None, tol=0.0):
    ref, new = {}, {}
    flatten("", json.loads(ref_path.read_text()), ref)
    flatten("", json.loads(new_path.read_text()), new)
    for k in (keys or sorted(ref)):
        add(level, h, f"{ref_path.name}:{k}", ref.get(k), new.get(k), tol)


def compare_table(level, h, ref_path, new_path, key_cols, tol):
    ref, new = pd.read_csv(ref_path), pd.read_csv(new_path)
    merged = ref.merge(new, on=key_cols, how="outer", suffixes=("_ref", "_new"), indicator=True)
    for _, row in merged.iterrows():
        tag = "/".join(str(row[c]) for c in key_cols)
        if row["_merge"] != "both":
            add(level, h, f"{ref_path.name}:{tag}:row", "present" if row["_merge"] == "left_only" else "absent",
                "present" if row["_merge"] == "right_only" else "absent", tol)
            continue
        for col in ref.columns:
            if col in key_cols or not pd.api.types.is_numeric_dtype(ref[col]):
                continue
            add(level, h, f"{ref_path.name}:{tag}:{col}", row[f"{col}_ref"], row[f"{col}_new"], tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, required=True, help="MCE_FP_BASE_DIR of the fresh run")
    ap.add_argument("--new-calibration", type=Path, default=None)
    ap.add_argument("--reference", type=Path, required=True, help="reference_results folder of the package")
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--out", type=Path, default=Path("verification_report.csv"))
    a = ap.parse_args()
    ref = a.reference / "primary"

    audit_keys = ["n_files", "n_raw_rows", "n_children", "eligible_observations", "eligible_label_counts.normal",
                  "eligible_label_counts.stunted", "eligible_label_counts.severely stunted",
                  "child_month_duplicates.duplicate_rows_found", "children_excluded_for_identity_inconsistency",
                  "who_haz_verification.rows_compared", "who_haz_verification.category_agreement_pct"]
    compare_json("A data", "all", ref / "dataset_longitudinal_audit.json",
                 a.new / "shared" / "dataset_longitudinal_audit.json", audit_keys)
    compare_table("A data", "all", ref / "attrition_audit_by_horizon.csv",
                  a.new / "shared" / "attrition_audit_by_horizon.csv", ["h", "target_observed"], 0.0)

    for h in HORIZONS:
        r, n = ref / f"h{h}", a.new / f"h{h}" / "artifacts"
        compare_json("A data", h, r / "stage_b_manifest.json", n / "stage_b_manifest.json",
                     ["design", "fingerprint_dev", "fingerprint_val", "fingerprint_test", "n_dev_pairs",
                      "n_dev_children", "n_val_pairs", "n_test_pairs", "n_test_children"])
        compare_table("A data", h, r / "class_distribution_by_subset.csv", n / "class_distribution_by_subset.csv",
                      ["subset", "class"], 0.0)
        compare_json("B model", h, r / "stage_d_manifest.json", n / "stage_d_manifest.json", ["selected_epoch"], a.tol)
        compare_table("B model", h, r / "class_specific_thresholds.csv", n / "class_specific_thresholds.csv",
                      ["class"], a.tol)
        compare_table("B model", h, r / "heldout_classification_metrics.csv", n / "heldout_classification_metrics.csv",
                      ["subset", "method"], a.tol)
        compare_table("B model", h, r / "heldout_probability_metrics.csv", n / "heldout_probability_metrics.csv",
                      ["method"], a.tol)
        compare_table("B model", h, r / "heldout_child_bootstrap_paired_deltas.csv",
                      n / "heldout_child_bootstrap_paired_deltas.csv", ["comparison", "metric"], a.tol)

    if a.new_calibration is not None:
        compare_table("B model", "all", a.reference / "calibration" / "calibration_per_class_summary.csv",
                      a.new_calibration / "calibration_per_class_summary.csv", ["h", "model", "class"], a.tol)

    report = pd.DataFrame(ROWS)
    report.to_csv(a.out, index=False)
    summary = report.groupby(["level", "status"]).size().unstack(fill_value=0)
    print(summary.to_string())
    key = report[report["item"].str.contains("heldout_classification_metrics.csv:all/") &
                 report["item"].str.endswith(":macro_f1")]
    print("\nHeld-out macro-F1, all test pairs (reference vs new):")
    print(key[["h", "item", "reference", "new", "abs_diff", "status"]].to_string(index=False))
    data_ok = (report.loc[report["level"] == "A data", "status"] == "IDENTICAL").all()
    model_bad = (report.loc[report["level"] == "B model", "status"] == "DIFFERENT").sum()
    print(f"\nDATA LEVEL: {'PASS (identical)' if data_ok else 'FAIL - input data or data processing differ'}")
    print(f"MODEL LEVEL: {model_bad} values differ by more than {a.tol}" if model_bad
          else f"MODEL LEVEL: all values identical or within {a.tol}")
    print(f"Full report: {a.out}")


if __name__ == "__main__":
    main()
