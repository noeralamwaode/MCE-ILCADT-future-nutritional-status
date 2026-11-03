# -*- coding: utf-8 -*-
"""
Additional analyses for MCE-ILCADT-FP (doc section G).

Runs the main pipeline (mce_ilcadt_future_prediction.py) once per configuration,
each in its own output folder, then writes summary tables.

Analyses
--------
rolling_origin : the three temporal blocks are shifted to several test origins T
                 (dev target <= T-8, validation anchor >= T-7 and target <= T-2,
                 test anchor T .. T+5). Every origin is reported, not only the best.
lovo           : leave-one-village-out. For village v, development and validation
                 use the other 14 villages; test = village v, test period only.
                 Results are reported per village and pooled over villages.
ablation       : no_current_status (drop status_t, haz_t) and no_status_history
                 (also drop every HAZ-derived history feature).
who_label      : labels from HAZ recomputed with WHO LMS tables.

These analyses read held-out labels. Run them only after the primary analysis
is complete and frozen.

Usage (Colab / local)
---------------------
    export MCE_FP_DATA_DIR=/content/buturk_dataset
    export MCE_FP_WHO_TABLE_DIR=/content/mce_code/who_tables
    export MCE_FP_ID_SALT=<secret>
    python run_additional_analyses.py --analysis all --out /content/drive/MyDrive/MCE_FP_additional
    python run_additional_analyses.py --analysis lovo --villages 0,1,2 --horizons 1 --out ...

Completed configurations are skipped, so an interrupted run can be restarted.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = HERE / "mce_ilcadt_future_prediction.py"
ROLLING_ORIGINS = ["2025-01", "2025-04", "2025-07", "2025-10", "2026-01"]
TEST_WINDOW_MONTHS = 6
N_VILLAGES_EXPECTED = 15


def shift(month: str, k: int) -> str:
    p = pd.Period(month, freq="M") + k
    return f"{p.year:04d}-{p.month:02d}"


def rolling_configs(origins):
    out = []
    for t in origins:
        out.append((f"origin_{t}", {
            "MCE_FP_DEV_TARGET_MAX": shift(t, -8),
            "MCE_FP_VAL_ANCHOR_MIN": shift(t, -7),
            "MCE_FP_VAL_TARGET_MAX": shift(t, -2),
            "MCE_FP_TEST_ANCHOR_MIN": t,
            "MCE_FP_TEST_ANCHOR_MAX": shift(t, TEST_WINDOW_MONTHS - 1),
        }))
    return out


def lovo_configs(villages):
    return [(f"village_{v:02d}", {"MCE_FP_HOLDOUT_VILLAGE": str(v)}) for v in villages]


def ablation_configs():
    return [(a, {"MCE_FP_ABLATION": a}) for a in ("no_current_status", "no_status_history")]


def who_configs():
    return [("who_label", {"MCE_FP_LABEL_SOURCE": "who"})]


def is_complete(base: Path, horizons, stages) -> bool:
    if "f" not in stages:
        return False
    for h in horizons:
        b = base / f"h{h}" / "artifacts" / "stage_b_manifest.json"
        if b.exists() and json.loads(b.read_text()).get("status") != "ok":
            continue  # horizon skipped by design (no test pairs)
        if not (base / f"h{h}" / "artifacts" / "stage_f_manifest.json").exists():
            return False
    return True


def run_config(analysis, name, overrides, out_root, horizons, stages, smoke):
    base = out_root / analysis / name
    if is_complete(base, horizons, stages):
        print(f"  [skip] {analysis}/{name} already complete")
        return {"analysis": analysis, "config": name, "status": "complete (skipped)"}
    base.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(overrides)
    env.update({"MCE_FP_BASE_DIR": str(base), "MCE_FP_HORIZONS": ",".join(map(str, horizons)),
                "MCE_FP_STAGES": ",".join(stages)})
    if smoke:
        env["MCE_FP_SMOKE"] = "1"
    started = time.time()
    with (base / "run.log").open("w") as log:
        proc = subprocess.run([sys.executable, str(MAIN)], env=env, stdout=log, stderr=subprocess.STDOUT)
    status = "ok" if proc.returncode == 0 else "failed"
    tail = ""
    if status == "failed":
        tail = (base / "run.log").read_text(errors="replace").strip().splitlines()[-1:]
        tail = tail[0] if tail else ""
    print(f"  [{status}] {analysis}/{name} ({time.time() - started:.0f} s) {tail}")
    return {"analysis": analysis, "config": name, "status": status, "error": tail,
            "seconds": round(time.time() - started), **overrides}


def collect(out_root: Path, analysis: str, horizons) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cls, prob, sizes = [], [], []
    for cfg in sorted((out_root / analysis).glob("*")):
        for h in horizons:
            art = cfg / f"h{h}" / "artifacts"
            if not (art / "stage_f_manifest.json").exists():
                continue
            b = json.loads((art / "stage_b_manifest.json").read_text())
            tag = {"analysis": analysis, "config": cfg.name, "h": h, "design": b["design"]}
            sizes.append({**tag, **{k: b[k] for k in ("n_dev_pairs", "n_val_pairs", "n_test_pairs", "n_test_children",
                                                       "n_pairs_excluded_at_boundaries")}})
            c = pd.read_csv(art / "heldout_classification_metrics.csv")
            cls.append(c.assign(**tag))
            p = pd.read_csv(art / "heldout_probability_metrics.csv")
            prob.append(p.assign(**tag))
    cat = lambda rows: pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return cat(cls), cat(prob), pd.DataFrame(sizes)


def pooled_lovo(out_root: Path, horizons) -> pd.DataFrame:
    """Pool LOVO test predictions over villages; every child is predicted by a model that never saw its village."""
    sys.path.insert(0, str(HERE))
    import mce_ilcadt_future_prediction as M  # metric functions only
    rows = []
    for h in horizons:
        ys, status, meta, ilc, kids = [], [], [], [], []
        for cfg in sorted((out_root / "lovo").glob("village_*")):
            art = cfg / f"h{h}" / "artifacts"
            if not (art / "stage_f_manifest.json").exists():
                continue
            test = pd.read_pickle(art / "test_pairs.pkl")
            ys.append(test["y"].to_numpy()); status.append(test["status_t"].to_numpy())
            kids.append(test["child_code"].to_numpy())
            meta.append(np.load(art / "meta_dnn_test_probabilities.npy").argmax(1))
            ilc.append(np.load(art / "mce_ilcadt_test_decision_scores.npy").argmax(1))
        if not ys:
            continue
        y, st, md, ic, ch = map(np.concatenate, (ys, status, meta, ilc, kids))
        decisions = {"persistence": st.astype(int), "meta_dnn_argmax": md, "mce_ilcadt": ic}
        minority = [1, 2]
        for subset, mask in (("all", np.ones(len(y), bool)), ("status_change", st != y)):
            for method, pred in decisions.items():
                m = M.classification_metrics(y[mask], pred[mask], minority)
                rows.append({"h": h, "subset": subset, "method": method, "villages": len(ys),
                             **{k: m[k] for k in ["n"] + M.SCALAR_KEYS}})
        ci, deltas = M.cluster_bootstrap(ch, y, decisions, minority,
                                         [("meta_dnn_argmax", "mce_ilcadt"), ("persistence", "mce_ilcadt")],
                                         M.BOOTSTRAP_REPLICATES, M.RANDOM_STATE)
        ci.assign(h=h).to_csv(out_root / "lovo" / f"pooled_bootstrap_ci_h{h}.csv", index=False)
        deltas.assign(h=h).to_csv(out_root / "lovo" / f"pooled_bootstrap_deltas_h{h}.csv", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True, choices=["rolling_origin", "lovo", "ablation", "who_label", "all"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--horizons", default="1,3,6")
    ap.add_argument("--origins", default=",".join(ROLLING_ORIGINS))
    ap.add_argument("--villages", default="all", help="comma-separated village codes or 'all'")
    ap.add_argument("--stages", default="a,b,c,d,e,f")
    ap.add_argument("--smoke", action="store_true", help="reduced models/epochs; for testing the runner only")
    args = ap.parse_args()

    for var in ("MCE_FP_DATA_DIR",):
        if not os.environ.get(var):
            sys.exit(f"Set {var} before running.")
    if "MCE_FP_ID_SALT" not in os.environ:
        print("Warning: MCE_FP_ID_SALT not set; each configuration will create its own random salt "
              "(splits are unaffected, pseudonyms differ between folders).")
    horizons = [int(h) for h in args.horizons.split(",")]
    stages = [s.strip() for s in args.stages.split(",")]
    analyses = ["rolling_origin", "lovo", "ablation", "who_label"] if args.analysis == "all" else [args.analysis]
    args.out.mkdir(parents=True, exist_ok=True)
    print("NOTE: these analyses read held-out labels; run them only after the primary analysis is frozen.")

    log_rows = []
    for analysis in analyses:
        print(f"\n=== {analysis}")
        if analysis == "rolling_origin":
            configs = rolling_configs(args.origins.split(","))
        elif analysis == "lovo":
            if args.villages == "all":
                villages = list(range(N_VILLAGES_EXPECTED))
            else:
                villages = [int(v) for v in args.villages.split(",")]
            configs = lovo_configs(villages)
        elif analysis == "ablation":
            configs = ablation_configs()
        else:
            if not os.environ.get("MCE_FP_WHO_TABLE_DIR"):
                sys.exit("who_label needs MCE_FP_WHO_TABLE_DIR.")
            configs = who_configs()
        for name, overrides in configs:
            log_rows.append(run_config(analysis, name, overrides, args.out, horizons, stages, args.smoke))

        if "f" in stages:
            cls, prob, sizes = collect(args.out, analysis, horizons)
            if not cls.empty:
                cls.to_csv(args.out / analysis / "summary_classification.csv", index=False)
                prob.to_csv(args.out / analysis / "summary_probability.csv", index=False)
                sizes.to_csv(args.out / analysis / "summary_sizes.csv", index=False)
            if analysis == "lovo":
                pooled = pooled_lovo(args.out, horizons)
                if not pooled.empty:
                    pooled.to_csv(args.out / "lovo" / "pooled_classification.csv", index=False)
                vm = next(iter(sorted((args.out / "lovo").glob("village_*/shared/village_codes.csv"))), None)
                if vm is not None:
                    pd.read_csv(vm).to_csv(args.out / "lovo" / "village_codes.csv", index=False)

    pd.DataFrame(log_rows).to_csv(args.out / f"run_log_{args.analysis}.csv", index=False)
    print(f"\nDone. Summaries under {args.out}")


if __name__ == "__main__":
    main()
