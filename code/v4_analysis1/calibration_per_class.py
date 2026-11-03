#!/usr/bin/env python3
"""Per-class (one-vs-rest) calibration of FROZEN held-out probabilities.

Post-hoc analysis: reads the artifacts written by stage (f) of
mce_ilcadt_fp_v4.py (SAP v4 section 7.3) and never retrains, re-thresholds or
re-selects anything. Following the calibration hierarchy of Van Calster et al.
(BMC Med 2019;17:230):

* calibration-in-the-large (CITL): intercept of logit(P(y_k=1)) = a + offset(logit p_k)
* weak calibration: intercept a and slope b of logit(P(y_k=1)) = a + b*logit(p_k)
* moderate calibration: flexible calibration curve (logistic regression on a natural
  cubic spline of logit p_k, 3 df), reported only when the class has >= 200 events
  and >= 200 non-events in the evaluated subset (p.4). The curve is evaluated only
  between the 1st and 99th percentiles of p_k, where data exist.

Confidence intervals use cluster-robust (child-level) standard errors because a
child contributes several (t, t+h) pairs. Persistence and threshold-adjusted
MCE-ILCADT decisions are hard labels / decision scores, not probabilities, so
they are deliberately excluded.

Usage:
    python calibration_per_class.py --run-root runs/primary --horizons 1,3,6 --out final_run/additional/calibration
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy

LABELS = ["normal", "stunted", "severely stunted"]
MODEL_ORDER = ["RF", "LGBM", "XGB", "CAT", "GBC", "ADA"]
EPS = 1e-6
MIN_EVENTS_FOR_CURVE = 200
CURVE_POINTS = 60
CURVE_EPS = 1e-4  # probabilities clipped to [1e-4, 1-1e-4] for numerical stability of the spline fit only
SPLINE_DF = 3
N_QUANTILE_BINS = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def weak_calibration(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> dict:
    lp = logit(p)
    out = {}
    # CITL: slope fixed at 1 via offset
    try:
        m0 = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit(
            cov_type="cluster", cov_kwds={"groups": groups})
        a, se = float(m0.params[0]), float(m0.bse[0])
        out.update(citl=a, citl_ci_low=a - 1.96 * se, citl_ci_high=a + 1.96 * se)
    except Exception as exc:  # perfect separation, degenerate class, etc.
        out.update(citl=np.nan, citl_ci_low=np.nan, citl_ci_high=np.nan, citl_error=str(exc)[:120])
    try:
        m1 = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": groups})
        b, se = float(m1.params[1]), float(m1.bse[1])
        out.update(slope=b, slope_ci_low=b - 1.96 * se, slope_ci_high=b + 1.96 * se,
                   intercept_with_slope=float(m1.params[0]))
    except Exception as exc:
        out.update(slope=np.nan, slope_ci_low=np.nan, slope_ci_high=np.nan, slope_error=str(exc)[:120])
    return out


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    z = 1.96
    phat = k / n
    den = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / den
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True, help="folder containing h1/, h3/, h6/ of the frozen primary run")
    ap.add_argument("--horizons", default="1,3,6")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-learners", action="store_true", help="also evaluate the six base-learner probability blocks")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary_rows, curve_rows, bin_rows, inputs = [], [], [], {}
    for h in [int(x) for x in args.horizons.split(",")]:
        art = args.run_root / f"h{h}" / "artifacts"
        freeze = json.loads((art / "freeze_manifest.json").read_text())
        for f in ("test_pairs.pkl", "meta_dnn_test_probabilities.npy", "meta_test.npy"):
            if sha256(art / f) != freeze["files_sha256"][f]:
                raise ValueError(f"h={h}: {f} differs from the frozen artefact (freeze_manifest.json)")
        if not (art / "stage_f_manifest.json").exists():
            raise ValueError(f"h={h}: stage (f) has not been run; calibration is post-hoc only")
        test = pd.read_pickle(art / "test_pairs.pkl")
        p_meta = np.load(art / "meta_dnn_test_probabilities.npy").astype(np.float64)
        for f in ("test_pairs.pkl", "meta_dnn_test_probabilities.npy", "meta_test.npy"):
            inputs[f"h{h}/{f}"] = sha256(art / f)
        if len(test) != len(p_meta):
            raise ValueError(f"h={h}: test pairs ({len(test)}) and probabilities ({len(p_meta)}) differ in length")
        if not np.allclose(p_meta.sum(1), 1, atol=1e-4):
            raise ValueError(f"h={h}: Meta-DNN probabilities do not sum to 1")
        y = test["y"].to_numpy(np.int64)
        groups = pd.factorize(test["child_code"])[0]
        models = {"meta_dnn": p_meta}
        if args.base_learners:
            meta_test = np.load(art / "meta_test.npy").astype(np.float64)
            for b, name in enumerate(MODEL_ORDER):
                block = meta_test[:, b * 3:(b + 1) * 3]
                models[name] = block / block.sum(1, keepdims=True)
        for model, p in models.items():
            for k, label in enumerate(LABELS):
                yk = (y == k).astype(int)
                pk = p[:, k]
                n_ev, n_non = int(yk.sum()), int((1 - yk).sum())
                eligible = n_ev >= MIN_EVENTS_FOR_CURVE and n_non >= MIN_EVENTS_FOR_CURVE
                row = {"h": h, "model": model, "class": label, "n": len(yk), "events": n_ev, "non_events": n_non,
                       "children": int(test["child_code"].nunique()),
                       "mean_predicted": float(pk.mean()), "observed_rate": float(yk.mean()),
                       "curve_eligible_200_200": eligible}
                row.update(weak_calibration(yk, pk, groups))
                summary_rows.append(row)
                # quantile-bin reliability table (always reported, descriptive)
                edges = np.unique(np.quantile(pk, np.linspace(0, 1, N_QUANTILE_BINS + 1)))
                idx = np.clip(np.searchsorted(edges, pk, side="right") - 1, 0, len(edges) - 2)
                for b in range(len(edges) - 1):
                    m = idx == b
                    if not m.any():
                        continue
                    lo, hi = wilson(int(yk[m].sum()), int(m.sum()))
                    bin_rows.append({"h": h, "model": model, "class": label, "bin": b, "n": int(m.sum()),
                                     "mean_predicted": float(pk[m].mean()), "observed_rate": float(yk[m].mean()),
                                     "observed_ci_low": lo, "observed_ci_high": hi})
                # flexible curve only when Van Calster's 200/200 rule is met
                if eligible:
                    # A flexible curve can be impossible to fit when a learner's probabilities take only a few
                    # distinct values (e.g. AdaBoost): the spline design is then singular. Such cases are recorded
                    # as "curve not estimable" instead of stopping the whole analysis.
                    try:
                        lp = logit(np.clip(pk, CURVE_EPS, 1 - CURVE_EPS))
                        design = patsy.dmatrix(f"cr(x, df={SPLINE_DF}, constraints='center')", {"x": lp}, return_type="dataframe")
                        with np.errstate(all="ignore"):
                            fit = sm.GLM(yk, design, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
                        lo_p, hi_p = np.quantile(pk, [0.01, 0.99])
                        grid_p = np.linspace(max(lo_p, CURVE_EPS), min(hi_p, 1 - CURVE_EPS), CURVE_POINTS)
                        grid_x = patsy.build_design_matrices([design.design_info], {"x": logit(grid_p)}, return_type="dataframe")[0]
                        pred = fit.get_prediction(grid_x).summary_frame(alpha=0.05)
                        curve_rows += [{"h": h, "model": model, "class": label, "predicted": float(g),
                                        "observed_spline": float(m), "ci_low": float(l), "ci_high": float(u)}
                                       for g, m, l, u in zip(grid_p, pred["mean"], pred["mean_ci_lower"], pred["mean_ci_upper"])]
                    except (np.linalg.LinAlgError, ValueError) as exc:
                        summary_rows[-1]["curve_error"] = f"curve not estimable: {type(exc).__name__}"
                        print(f"  h={h} {model} {label}: calibration curve not estimable ({type(exc).__name__}); skipped")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out / "calibration_per_class_summary.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(args.out / "calibration_per_class_bins.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(args.out / "calibration_per_class_curve.csv", index=False)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "script": "calibration_per_class.py", "script_sha256": sha256(Path(__file__)),
                "run_root": str(args.run_root), "inputs_sha256": inputs,
                "note": "Post-hoc on frozen held-out probabilities; no model, threshold or split was changed."}
    (args.out / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary[summary.model == "meta_dnn"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
