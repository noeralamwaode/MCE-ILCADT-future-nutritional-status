"""Additional file 3 of paper 2: supplementary methods and tables, built only from frozen outputs.

Inputs (read-only):
  --v4-run   MCE_FP_v4/runs        (folders primary, sens_who_label, sens_no_current_status, sens_exclude_flagged)
  --v5-run   MCE_FP_v5/runs        (folders primary, sens_no_baseline_confirmation, sens_unconfirmed)
  --audit    master_v4_audit.json  (aggregate data audit of the master dataset)
Output:
  --pr-figure fig_pr_curves.pdf from paper2_figures.py (optional; included as Fig. S1, copied next to the .tex)
  --out      folder with Additional_file_3.tex (compile with pdflatex) and the CSVs behind every table.
No model, threshold or metric is recomputed; point differences are b minus a of frozen point estimates and
their CIs are copied from the frozen paired-difference files.
Usage: python make_additional_file3.py --v4-run MCE_FP_v4/runs --v5-run MCE_FP_v5/runs --audit master_v4_audit.json --out af3
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import json
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--v4-run", type=Path, required=True); ap.add_argument("--v5-run", type=Path, required=True)
ap.add_argument("--audit", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--pr-figure", type=Path, default=None, help="fig_pr_curves.pdf from paper2_figures.py (optional; becomes Fig. S1)")
ap.add_argument("--dataset-dir", type=Path, default=None,
                help="folder with dataset_puskesmas.csv (paper2_dataset.py) and master_v4_summary_by_year_puskesmas.csv (Stage 1); optional, becomes S6")
A = ap.parse_args(); A.out.mkdir(parents=True, exist_ok=True)
V4, V5 = A.v4_run, A.v5_run
f3 = lambda x: "" if pd.isna(x) else f"{x:.3f}"
f1p = lambda x: f"{100 * x:.1f}"
ci = lambda v, lo, hi: f"{v:.3f} ({lo:.3f} to {hi:.3f})"
TEX: list[str] = []
NAMES = {"persistence": "Persistence", "best_base": "Best base learner", "best_base_ilcadt": "Best base + ILCADT",
         "mce": "MCE", "mce_ilcadt": "MCE--ILCADT"}


def esc(s) -> str:
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


MIN_EVENTS_REPORTED = 5   # small-cell disclosure rule (authors' choice; no formal threshold set by the data custodian)


def table(name: str, caption: str, df: pd.DataFrame, note: str = "", size: str = r"\scriptsize") -> None:
    df.to_csv(A.out / f"{name}.csv", index=False)
    cols = "l" * 1 + "r" * (df.shape[1] - 1)
    TEX.append(r"\begin{center}" + size + r"\setlength{\tabcolsep}{3pt}" + "\n" + rf"\begin{{longtable}}{{{cols}}}")
    TEX.append(rf"\caption{{{caption}}}\label{{{name}}}\\ \toprule")
    TEX.append(" & ".join(rf"\textbf{{{esc(c)}}}" for c in df.columns) + r"\\ \midrule\endfirsthead")
    TEX.append(r"\toprule " + " & ".join(rf"\textbf{{{esc(c)}}}" for c in df.columns) + r"\\ \midrule\endhead")
    for row in df.itertuples(index=False):
        TEX.append(" & ".join(esc(v) for v in row) + r"\\")
    TEX.append(r"\bottomrule\end{longtable}\end{center}")
    if note:
        TEX.append(r"{\scriptsize " + note + r"}\par\medskip")


def section(t: str, text: str = "") -> None:
    TEX.append(rf"\section*{{{t}}}" + ("\n" + text if text else ""))


# ------------------------------------------------------------------ S1 methods
section("S1 Supplementary methods",
r"""\textbf{Base-learner hyperparameters} (fixed in advance, not tuned; seed 42): random forest (400 trees, maximum depth 15, minimum 10 samples to split, 4 per leaf); LightGBM (600 trees, learning rate 0.05, 31 leaves, minimum 30 samples per leaf, row and column subsampling 0.9, deterministic); XGBoost (600 trees, learning rate 0.05, maximum depth 6, row and column subsampling 0.9, softprob objective, histogram method); CatBoost (800 iterations, learning rate 0.05, depth 6, multiclass loss); gradient boosting (100 trees, learning rate 0.1, maximum depth 3); AdaBoost (300 estimators, learning rate 0.5).

\textbf{Cross-validation folds.} Every fold had to contain at least two children with a severely stunted target (analysis~1), or with worsening and with improvement (analysis~2).

\textbf{Meta-DNN training.} Sparse categorical cross-entropy; Adam optimiser with initial learning rate $10^{-3}$, halved after five epochs without improvement in validation loss (minimum $10^{-6}$); batch size 64; at most 200 epochs; early stopping after 12 epochs without improvement in validation loss. The epoch with the lowest meta-validation loss was selected (ties broken by higher macro-F1); the network was retrained from the same seed on all training children for that number of epochs, replaying the recorded learning-rate schedule. Seed 42; TensorFlow determinism enabled.

\textbf{Extreme probabilities.} Probabilities and thresholds were clipped to $[10^{-7}, 1-10^{-7}]$ before the logit transformation of the ILCADT rule, and predicted probabilities of the observed class were clipped at $10^{-7}$ before the negative log-likelihood was computed; shifted logits were bounded to $\pm 60$.

\textbf{ILCADT tie-breaking.} When several thresholds gave the same F1-score, ties were broken by higher recall, then higher precision, then closeness to 0.5, then the lower threshold.

\textbf{Plausibility bounds for predictors.} Birth weight 0.5--6.0~kg, birth length 30--60~cm, weight 1--40~kg, weight-for-age z-score $-6$ to $+5$, weight-for-height z-score $-5$ to $+5$, MUAC 7--30~cm; a MUAC of zero was treated as not measured. Values outside these bounds were set to missing (counts in S2).

\textbf{Additional metrics, analysis~1.} Macro-averaged precision and recall, per-class support, and both top-label and classwise expected calibration error; minority classes were defined from the training data.

\textbf{Calibration.} For each class (one-vs-rest), calibration-in-the-large was estimated as the intercept of a logistic model of the outcome with the logit of the predicted probability as an offset, and the calibration slope as the coefficient of the logit of the predicted probability in a logistic model of the outcome; standard errors were clustered by child. The same definitions were used for new stunting in analysis~2.

\textbf{Pre-specified paired differences.} Analysis~1: MCE--ILCADT and MCE versus persistence; MCE--ILCADT versus MCE; the best base learner with versus without ILCADT; MCE versus the best base learner. Analysis~2, primary: AUPRC of MCE versus the mapped analysis-1 model, the simple logistic regression and the 33-predictor logistic regression; secondary: three-class macro-F1 of MCE--ILCADT versus each comparator with ILCADT, MCE--ILCADT versus MCE, and the best base learner with versus without ILCADT.

\textbf{Code audit.} Before the test data were evaluated, the analysis code was audited stage by stage with AI assistance (Claude, Anthropic), and the authors reviewed the audit reports; this audit is distinct from the re-run of each primary analysis, which the authors executed in a fresh runtime: pair counts per block were compared with the analysis plan for $h = 1$--6; all 33 predictors were recomputed for 1\,800 pairs (no mismatches); a truncation test confirmed that deleting all observations after month $t$ left the predictors unchanged; classification metrics were compared with scikit-learn; the bootstrap was reproduced; and repeated runs produced bit-identical numerical outputs (probabilities, decisions, thresholds and metric tables), although the serialized model files were not byte-identical. Each primary analysis was later re-run in a fresh runtime with identical compared outputs.

\textbf{Changes from the earlier single-centre analysis.} An earlier analysis of one of the 12 centres (January 2024 to August 2026; test prediction months from October 2025) used the same pipeline. The present analysis plan (plan v4, section 12) changed: data from 12 centres instead of one, January 2022 to September 2026; child identity from the record-link identifier, because the identity number was masked in the newer exports; a temporal threshold-tuning block for every horizon (the earlier analysis used 20\% of training children for $h = 6$); five 12-month growth-history predictors added (33 instead of 28); component ablation and results by centre added; rolling-origin and leave-one-village-out analyses not repeated. Base learners, their hyperparameters, out-of-fold stacking, the Meta-DNN and the ILCADT rule were unchanged. Thresholds from the earlier analysis were not reused.

\textbf{Deviations from the analysis plans.} Analysis~2: (i) the development sample-size assessment (Riley et al.\ 2020) specified in plan section~9 was not performed, so the adequacy of the development data for the complexity of the models was not formally assessed; (ii) the self-assessment of risk of bias with PROBAST+AI mentioned in plan section~10 was not included, because PROBAST+AI is designed for appraising published studies and is not a reporting requirement; reporting follows TRIPOD+AI (Additional file~2). Analysis~1: (iii) the 12-fold leave-one-centre-out validation, listed in plan v4 section~9 as deferred to a decision after the primary run, was not performed; results by centre from the primary predictions are reported instead (Table S11); (iv) the flow diagram listed in plan v4 section~10 is replaced by Table~1 and Fig.~2 of the main text and Table S1. Both analyses: (v) the confusion matrices are not shown; they are part of the frozen outputs (files test\_classification\_metrics.csv and threeclass\_metrics.csv). Results that are pre-specified but not shown in the main text (paired differences of all classification metrics in analysis~1, analysis~2 by status at $t$, three-class probability metrics, threshold effects in the sensitivity analyses and flexible calibration curves) are reported in S9.

\textbf{Children above +3 SD (tall).} 247 observations of 164 children had HAZ above $+3$ and were excluded from modelling (plan v4, section~2).""")

# ------------------------------------------------------------------ data audit
au = json.loads(A.audit.read_text())
LBL = {"system_outlier_or_missing_category": "System outlier or missing category", "tall_haz_gt_3": "HAZ above +3 (tall)",
       "haz_missing_or_implausible": "HAZ missing or outside -6 to +6", "measurement_before_birth": "Measurement date before date of birth",
       "age_gt_60_months": "Age above 60 months", "height_missing_or_implausible": "Length or height missing or outside 40-125 cm",
       "birth_weight_kg": "Birth weight (0.5-6.0 kg)", "birth_length_cm": "Birth length (30-60 cm)", "weight_kg": "Weight (1-40 kg)",
       "height_cm": "Length or height (40-125 cm)", "muac_cm": "MUAC (7-30 cm)", "waz": "Weight-for-age z-score (-6 to +5)", "whz": "Weight-for-height z-score (-5 to +5)"}
ex = pd.DataFrame([(LBL.get(k, k), v) for k, v in au["excluded_observation_reasons_sequential"].items()], columns=["Exclusion (applied in sequence)", "Observations"])
section("S2 Data audit", "Retrospective edits: a comparison of the export used here (downloaded 13 October 2026) with an earlier download of one centre used in the single-centre analysis found, among 42\\,940 child-month records with identical keys, changed values for height in 44 records, weight in 23, height-for-age category in 27, z-scores in 89, birth date in 11 and measurement date in 33 (0.03--0.21\\% per variable; source: data-audit record of 14 October 2026).")
table("tab_S2a_exclusions", "Observations excluded from eligibility (master dataset, 234\\,073 records of 10\\,321 children)", ex)
vm = pd.DataFrame([(LBL.get(k, k), v) for k, v in au["values_set_missing_by_validity_rule"].items()], columns=["Predictor (plausible range)", "Values set to missing"])
table("tab_S2b_missing", "Predictor values set to missing by the plausibility rules (the record was kept)", vm,
      "Counted over all records before eligibility was assessed. MUAC includes values recorded as zero (not measured). A length or height outside 40--125 cm also made the observation ineligible; because the exclusions in Table S1 were applied in sequence, records with an implausible length or height that had already been excluded for an earlier reason are not counted again there, so Table S1 shows only the 7 remaining records.")
fa = pd.read_csv(V4 / "primary/shared/feature_availability_audit.csv")
fa = pd.DataFrame({"Predictor": fa.feature, "Information time": fa.information_time, "Uses information after t": fa.uses_information_after_t,
                   "Missing (% of eligible obs.)": fa.missing_fraction_all_eligible_obs.map(f1p)})
table("tab_S2c_feature_availability", "Feature-availability audit: the 33 predictors", fa)

# ------------------------------------------------------------------ attrition and history
section("S3 Pairs, attrition and growth history (analysis 1)")
at = pd.read_csv(V4 / "primary/shared/attrition_audit_h1-2-3-4-5-6.csv")
at = pd.DataFrame({"h": at.h, "Target observed": at.target_observed, "Prediction months": at.prediction_months, "Mean HAZ at t": at.mean_haz_t.map(f3),
                   "Normal (%)": at.pct_status_t_normal.round(1), "Stunted (%)": at.pct_status_t_stunted.round(1),
                   "Severely stunted (%)": at["pct_status_t_severely stunted"].round(1)})
table("tab_S3a_attrition", "Attrition audit: status at month $t$ of prediction months with and without an eligible observation exactly $h$ months later", at)
ha = pd.read_csv(V4 / "primary/shared/history_availability_h1-2-3-4-5-6.csv")
ha = ha[(ha.unit == "pairs")].pivot_table(index=["h", "block"], columns="lag_months", values="share").reset_index()
ha.columns = ["h", "Block"] + [f"HAZ observed {c} months earlier (%)" for c in ha.columns[2:]]
for c in ha.columns[2:]: ha[c] = ha[c].map(f1p)
table("tab_S3b_history", "Availability of growth history in prediction pairs, by horizon and block", ha)

# ------------------------------------------------------------------ analysis 1 results
section("S4 Analysis 1: additional results")
rows, pcs, base, prob, pus = [], [], [], [], []
for h in range(1, 7):
    a = V4 / f"primary/h{h}/artifacts"
    m = pd.read_csv(a / "test_classification_metrics.csv"); m = m[m.subset == "all"].set_index("method")
    d = pd.read_csv(a / "test_child_bootstrap_paired_differences.csv"); d = d[(d.subset == "all") & (d.metric == "macro_f1")].set_index("comparison")
    for k in ("mce", "mce_ilcadt"):
        r = d.loc[f"{k} minus persistence"]
        rows.append({"h": h, "Persistence": f3(m.loc["persistence", "macro_f1"]), "Model": NAMES[k], "Model macro-F1": f3(m.loc[k, "macro_f1"]),
                     "Difference (95% CI)": ci(m.loc[k, "macro_f1"] - m.loc["persistence", "macro_f1"], r.ci_low, r.ci_high)})
    if h in (1, 3, 6):
        pc = pd.read_csv(a / "test_per_class_metrics.csv"); pc = pc[(pc.subset == "all") & pc.method.isin(NAMES)]
        for r in pc.itertuples():
            pcs.append({"h": h, "Method": NAMES[r.method], "Class": r._3, "Support": r.support, "Precision": f3(r.precision), "Recall": f3(r.recall), "F1": f3(r.f1)})
        mm = pd.read_csv(a / "test_classification_metrics.csv"); mm = mm[mm.subset == "all"].set_index("method")
        for b in ("RF", "LGBM", "XGB", "CAT", "GBC", "ADA"):
            base.append({"h": h, "Learner": b, "Macro-F1 arg-max": f3(mm.loc[f"{b}_argmax", "macro_f1"]), "Macro-F1 ILCADT": f3(mm.loc[f"{b}_ilcadt", "macro_f1"]),
                         "Minority recall arg-max": f3(mm.loc[f"{b}_argmax", "minority_macro_recall"]), "Minority recall ILCADT": f3(mm.loc[f"{b}_ilcadt", "minority_macro_recall"])})
        pm = pd.read_csv(a / "test_probability_metrics.csv")
        for r in pm.itertuples():
            prob.append({"h": h, "Model": "Meta-DNN" if r.method == "meta_dnn" else r.method, "ROC-AUC": f3(r.roc_auc_ovr_macro), "NLL": f3(r.nll),
                         "Brier": f3(r.brier_multiclass), "ECE top-label": f3(r.ece_top_label), "ECE classwise": f3(r.ece_classwise_macro)})
        mp = pd.read_csv(a / "test_classification_metrics.csv"); mp = mp[mp.subset.str.startswith("puskesmas=")]
        dp = pd.read_csv(a / "test_child_bootstrap_paired_differences.csv")
        dp = dp[dp.subset.str.startswith("puskesmas=") & (dp.comparison == "mce_ilcadt minus persistence") & (dp.metric == "macro_f1")].set_index("subset")
        for sub, g in mp.groupby("subset"):
            g = g.set_index("method"); r = dp.loc[sub]
            pus.append({"h": h, "Puskesmas": sub.split("=", 1)[1].title(), "Pairs": int(g.loc["persistence", "n"]), "Children": int(g.loc["persistence", "n_children"]),
                        "Persistence": f3(g.loc["persistence", "macro_f1"]), "MCE--ILCADT": f3(g.loc["mce_ilcadt", "macro_f1"]),
                        "Difference (95% CI)": ci(g.loc["mce_ilcadt", "macro_f1"] - g.loc["persistence", "macro_f1"], r.ci_low, r.ci_high)})
table("tab_S4a_horizon_curve", "Horizon curve: macro-F1 of persistence and the models, $h$ = 1--6 (all test pairs)", pd.DataFrame(rows))
table("tab_S4b_per_class", "Per-class precision, recall and F1 on all test pairs", pd.DataFrame(pcs))
table("tab_S4c_base_learners", "All six base learners with arg-max and ILCADT decisions (all test pairs)", pd.DataFrame(base))
table("tab_S4d_probability", "Probability metrics of the Meta-DNN and the base learners (all test pairs)", pd.DataFrame(prob),
      "NLL, negative log-likelihood; ECE, expected calibration error (ten equal-width bins).")
cal = pd.read_csv(V4 / "primary/calibration/calibration_per_class_summary.csv"); cal = cal[cal.h.isin([1, 3, 6]) & (cal.model == "meta_dnn")]
cal = pd.DataFrame({"h": cal.h, "Class": cal["class"], "Events": cal.events, "Mean predicted": cal.mean_predicted.map(f3), "Observed": cal.observed_rate.map(f3),
                    "CITL (95% CI)": [ci(a, b, c) for a, b, c in zip(cal.citl, cal.citl_ci_low, cal.citl_ci_high)],
                    "Slope (95% CI)": [ci(a, b, c) for a, b, c in zip(cal.slope, cal.slope_ci_low, cal.slope_ci_high)]})
table("tab_S4e_calibration", "Per-class calibration of the Meta-DNN (one-vs-rest; child-clustered standard errors)", cal, "CITL, calibration-in-the-large (logit scale).")
table("tab_S4f_puskesmas", "Macro-F1 by primary health care centre (predictions of the primary analysis, no retraining)", pd.DataFrame(pus))
sens = []
for run, lab in (("sens_who_label", "WHO-recomputed labels"), ("sens_no_current_status", "Without status and HAZ at t"), ("sens_exclude_flagged", "Excluding flagged observations")):
    for h in (1, 3, 6):
        a = V4 / f"{run}/h{h}/artifacts"
        m = pd.read_csv(a / "test_classification_metrics.csv"); m = m[m.subset == "all"].set_index("method")
        d = pd.read_csv(a / "test_child_bootstrap_paired_differences.csv"); d = d[(d.subset == "all") & (d.metric == "macro_f1")].set_index("comparison")
        r = d.loc["mce_ilcadt minus persistence"]
        sens.append({"Analysis": lab, "h": h, "Pairs": int(m.loc["persistence", "n"]), "Persistence": f3(m.loc["persistence", "macro_f1"]),
                     "MCE": f3(m.loc["mce", "macro_f1"]), "MCE--ILCADT": f3(m.loc["mce_ilcadt", "macro_f1"]),
                     "MCE--ILCADT minus persistence (95% CI)": ci(m.loc["mce_ilcadt", "macro_f1"] - m.loc["persistence", "macro_f1"], r.ci_low, r.ci_high)})
table("tab_S4g_sensitivity", "Analysis 1 sensitivity analyses (macro-F1, all test pairs)", pd.DataFrame(sens))

# ------------------------------------------------------------------ analysis 2
_cd = {h: pd.read_csv(V5 / f"primary/h{h}/artifacts/class_distribution_by_block.csv").query("block == 'test'").set_index("class")["pairs"] for h in (1, 3, 6)}
_w = [int(_cd[h]["worsening"]) for h in (1, 3, 6)]; _i = [int(_cd[h]["improvement"]) for h in (1, 3, 6)]
section("S5 Analysis 2 (exploratory): additional results",
        f"In the test data, {_w[0]:,}, {_w[1]:,} and {_w[2]:,} pairs showed confirmed worsening and {_i[0]:,}, {_i[1]:,} and {_i[2]:,} "
        "confirmed improvement at $h$ = 1, 3 and 6, respectively (primary definition).")
att = pd.read_csv(V5 / "primary/shared/attrition_confirmation_h1-3-6.csv")
att = pd.DataFrame({"h": att.h, "Block": att.block, "Confirmation observed": att.confirmation_observed, "Pairs": att.pairs,
                    "Normal at t (%)": att.pct_status_t_normal.round(1), "Unconfirmed worsening (%)": att.pct_unconfirmed_worsening.round(2),
                    "Unconfirmed improvement (%)": att.pct_unconfirmed_improvement.round(2)})
table("tab_S5a_confirmation_attrition", "Pairs with and without the confirmation month $t+h+1$", att)
PM = {"persistence": "Persistence", "mce": "MCE", "v4_mapped": "Analysis-1 model, mapped", "v4_mapped_recal": "Analysis-1 model, mapped, recalibrated",
      "lr_simple": "Logistic regression, simple", "lr_33": "Logistic regression, 33 predictors", "best_base": "Best base learner"}
prim, calr, dca, three, tdiff, sub, s2 = [], [], [], [], [], [], []
for run, lab in (("primary", "Primary (confirmed)"), ("sens_no_baseline_confirmation", "Without baseline confirmation"), ("sens_unconfirmed", "Without confirmation")):
    for h in (1, 3, 6):
        a = V5 / f"{run}/h{h}/artifacts"
        e = pd.read_csv(a / "primary_endpoint_metrics.csv"); c = pd.read_csv(a / "primary_bootstrap_ci.csv").set_index("method")
        d = pd.read_csv(a / "primary_paired_differences.csv")
        if run == "primary":
            for r in e.itertuples():
                prim.append({"h": h, "Method": PM[r.method], "AUPRC (95% CI)": f3(r.auprc_average_precision) if r.method == "persistence" else ci(r.auprc_average_precision, c.loc[r.method, "ci_low"], c.loc[r.method, "ci_high"]),
                             "AUC": f3(r.auc), "Sens. top 5%": f3(r.sensitivity_top5pct), "PPV top 5%": f3(r.ppv_top5pct),
                             "Sens. top 10%": f3(r.sensitivity_top10pct), "PPV top 10%": f3(r.ppv_top10pct)})
            cc = pd.read_csv(a / "primary_calibration.csv")
            for r in cc.itertuples():
                calr.append({"h": h, "Method": PM[r.method], "Mean predicted": f3(r.mean_predicted), "Observed": f3(r.observed),
                             "CITL (95% CI)": ci(r.citl, r.citl_ci_low, r.citl_ci_high), "Slope (95% CI)": ci(r.slope, r.slope_ci_low, r.slope_ci_high)})
            dc = pd.read_csv(a / "primary_decision_curve.csv").pivot(index="threshold", columns="method", values="net_benefit")
            for t in (0.01, 0.02, 0.05, 0.10):
                dca.append({"h": h, "Threshold (%)": f"{100 * t:.0f}", **{PM.get(k, k.replace("_", " ")): f"{dc.loc[t, k]:.4f}" for k in dc.columns}})
            tm = pd.read_csv(a / "threeclass_metrics.csv"); tm = tm[tm.subset == "all"]
            for r in tm.itertuples():
                three.append({"h": h, "Method": r.method.replace("_", " "), "Macro-F1": f3(r.macro_f1), "Balanced accuracy": f3(r.balanced_accuracy),
                              "Minority macro-recall": f3(r.minority_macro_recall), "Minimum minority recall": f3(r.minimum_minority_recall)})
            td = pd.read_csv(a / "threeclass_paired_differences.csv"); td = td[td.metric.isin(["macro_f1", "minority_macro_recall"])]
            tmi = tm.set_index("method")
            for r in td.itertuples():
                b_, a_ = r.comparison.split(" minus ")
                tdiff.append({"h": h, "Comparison": r.comparison.replace("_", " "), "Metric": r.metric.replace("_", " "),
                              "Difference (95% CI)": ci(tmi.loc[b_, r.metric] - tmi.loc[a_, r.metric], r.ci_low, r.ci_high)})
            sg = pd.read_csv(a / "primary_subgroups.csv")
            # public (deposited) files already show suppressed rows as events '<5' and AUPRC 'n.r.'; frozen files hold numbers
            sg["events"] = pd.to_numeric(sg["events"], errors="coerce")   # '<5' -> NaN = already suppressed
            for c in [c for c in sg.columns if c.startswith("auprc")]:
                sg[c] = pd.to_numeric(sg[c], errors="coerce")
            for r in sg.itertuples():
                small = pd.isna(r.events) or r.events < MIN_EVENTS_REPORTED   # small-cell rule: counts < 5 and their AUPRC are not shown
                sub.append({"h": h, "Subgroup": r.subgroup.replace("_", " "), "Level": str(r.level).title() if r.subgroup == "puskesmas" else r.level,
                            "Pairs": r.n, "Events": f"<{MIN_EVENTS_REPORTED}" if small else int(r.events),
                            **{k: "n.r." if small else f3(v) for k, v in
                               {"MCE": r.auprc_mce, "Mapped": r.auprc_v4_mapped, "LR simple": r.auprc_lr_simple, "LR 33": r.auprc_lr_33}.items()}})
        ei = e.set_index("method")
        for r in d[d.comparison.isin(["mce minus v4_mapped", "mce minus lr_simple", "mce minus lr_33"])].itertuples():
            s2.append({"Target": lab, "h": h, "Incidence (%)": f1p(ei.loc["mce", "incidence"]), "MCE AUPRC": f3(ei.loc["mce", "auprc_average_precision"]),
                       "Comparison": PM[r.comparison.split(" minus ")[1]], "Difference (95% CI)": ci(r.difference, r.ci_low, r.ci_high)})
table("tab_S5b_primary_endpoint", "New stunting among children normal at $t$: all methods (test data)", pd.DataFrame(prim),
      "Top 5\\%/10\\%: the 5\\% or 10\\% of normal pairs with the highest predicted risk. Persistence gives no ranking (AUPRC = incidence).")
if A.pr_figure is not None:
    import shutil
    shutil.copy2(A.pr_figure, A.out / "fig_pr_curves.pdf")
    TEX.append(r"""\begin{figure}[h]\centering\includegraphics[width=0.9\linewidth]{fig_pr_curves.pdf}
\caption{Precision--recall curves for new stunting among children normal at month $t$ (test data, analysis 2, exploratory). The dashed line is the incidence, which equals the precision of persistence at every recall (no ranking). The mapped analysis-1 model uses its raw mapped probabilities. The area under each curve (average precision) reproduced the frozen AUPRC in Table S14 exactly.}\label{fig_S1_pr}\end{figure}""")
table("tab_S5c_calibration", "Calibration for new stunting (child-clustered standard errors)", pd.DataFrame(calr))
table("tab_S5d_dca", "Net benefit at selected risk thresholds (decision curve analysis)", pd.DataFrame(dca), "Treat all = follow up all normal children; treat none = 0.", r"\tiny")
table("tab_S5e_threeclass", "Three-class direction of change (stable, worsening, improvement): all methods", pd.DataFrame(three))
table("tab_S5f_threeclass_diff", "Three-class paired differences (secondary)", pd.DataFrame(tdiff))
table("tab_S5g_sensitivity", "Primary endpoint under the two sensitivity definitions (and the primary definition for reference)", pd.DataFrame(s2))
table("tab_S5h_subgroups", "AUPRC by subgroup (descriptive)", pd.DataFrame(sub),
      f"Estimates in small subgroups are unstable. To limit disclosure risk and because AUPRC is uninformative with very few events, subgroups with fewer than {MIN_EVENTS_REPORTED} events are shown as <{MIN_EVENTS_REPORTED} and their AUPRC is not reported (n.r.).")

# ------------------------------------------------------------------ S6 dataset by centre (aggregate inputs)
if A.dataset_dir is not None:
    pk = pd.read_csv(A.dataset_dir / "dataset_puskesmas.csv")
    pk = pk.assign(name=pk["puskesmas"].str.title())
    t1 = pd.DataFrame({"Primary health care centre": pk["name"], "Children": pk["children"].map("{:,}".format),
                       "Eligible observations": pk["observations"].map("{:,}".format),
                       "Stunted (%)": pk["stunted_pct"].map("{:.1f}".format), "Severely stunted (%)": pk["severely_stunted_pct"].map("{:.1f}".format)})
    yr = pd.read_csv(A.dataset_dir / "master_v4_summary_by_year_puskesmas.csv")
    yr["cell"] = yr["observations"].map("{:,}".format) + " (" + yr["stunted_pct"].map("{:.1f}".format) + ")"
    t2 = yr.pivot(index="puskesmas", columns="year", values="cell").reindex(pk["puskesmas"])
    t2.index = t2.index.str.title(); t2 = t2.reset_index().rename(columns={"puskesmas": "Primary health care centre"})
    t2.columns = [str(c) for c in t2.columns]
    section("S6 Dataset by primary health care centre",
            "Eligible observations of the master dataset (January 2022 to September 2026), all children; aggregate counts only.")
    table("tab_S6a_dataset_centres", "Children, eligible observations and height-for-age status by primary health care centre", t1)
    table("tab_S6b_dataset_centre_year", "Eligible observations (percentage stunted or severely stunted) by primary health care centre and calendar year", t2,
          "2026 covers January to September. Coverage in 2022 was incomplete in several centres.")

# ------------------------------------------------------------------ S7 analysis 1 excluding the centre analysed earlier
def _mf1(cm):
    cm = np.asarray(cm, float); tp = np.diag(cm)
    p = tp / np.maximum(cm.sum(0), 1); r = tp / np.maximum(cm.sum(1), 1)
    f = np.where(p + r > 0, 2 * p * r / np.where(p + r > 0, p + r, 1), 0)
    return f.mean(), r[1:].mean(), r[1:].min()
rows11 = []
for h in (1, 3, 6):
    m = pd.read_csv(V4 / f"primary/h{h}/artifacts/test_classification_metrics.csv")
    pk = m[m.subset.str.startswith("puskesmas=")]
    for meth, lab in [("persistence", "Persistence"), ("mce", "MCE"), ("mce_ilcadt", "MCE--ILCADT")]:
        allcm = np.array(json.loads(m[(m.subset == "all") & (m.method == meth)].confusion_matrix.iloc[0]))
        x = pk[(pk.method == meth) & (pk.subset != "puskesmas=KULISUSU")]
        cm = sum(np.array(json.loads(c)) for c in x.confusion_matrix)
        k = np.array(json.loads(pk[(pk.method == meth) & (pk.subset == "puskesmas=KULISUSU")].confusion_matrix.iloc[0]))
        assert (cm + k == allcm).all(), "per-centre confusion matrices do not sum to the total"
        assert abs(_mf1(allcm)[0] - m[(m.subset == "all") & (m.method == meth)].macro_f1.iloc[0]) < 1e-9
        f, mr, mn = _mf1(cm)
        rows11.append({"h": h, "Method": lab, "Pairs": f"{int(cm.sum()):,}", "Macro-F1": f"{f:.3f}",
                       "Minority macro-recall": f"{mr:.3f}", "Minimum minority recall": f"{mn:.3f}"})
section("S7 Analysis 1 excluding the centre analysed earlier (post hoc)",
        "Test-period data of one centre had been evaluated in an earlier single-centre analysis (S1). The table reports the "
        "remaining 11 centres, derived from the frozen per-centre confusion matrices of the primary analysis (no model, threshold "
        "or prediction was changed). The per-centre matrices sum exactly to the matrix of all test pairs, and macro-F1 recomputed "
        "from that matrix equals the frozen value. Point estimates only; no confidence intervals were computed.")
table("tab_S7_eleven_centres", "Analysis 1 on the test pairs of the 11 centres not analysed earlier (post hoc point estimates)", pd.DataFrame(rows11))

# ------------------------------------------------------------------ S8 full version of Table 2
if A.dataset_dir is not None and (A.dataset_dir / "table_characteristics_h1.csv").exists():
    import re as _re
    ch = pd.read_csv(A.dataset_dir / "table_characteristics_h1.csv", dtype=str)
    assert ch.shape == (24, 4), "Table 2 source must have 24 data rows and 4 columns"
    def _lab(x):
        x = x.replace("month t+h", "month $t+h$").replace("month t:", "month $t$:").replace("month t,", "month $t$,")
        x = x.replace("between t and t+h", "between $t$ and $t+h$").replace("< 2.5", "$<$2.5")
        return _re.sub(r"(\d)-(\d)", r"\1--\2", x)
    def _val(x):
        x = _re.sub(r"(\d)-(\d)", r"\1--\2", x)
        return _re.sub(r"(^|\()-(\d)", r"\1$-$\2", x).replace(",", "\\,")
    ch.iloc[:, 0] = ch.iloc[:, 0].map(_lab)
    for c in ch.columns[1:]: ch[c] = ch[c].map(_val)
    ch = ch.rename(columns={ch.columns[0]: "Characteristic"})
    section("S8 Characteristics of children and prediction pairs (full version of Table 2)",
            "Analysis 1, $h = 1$. The main text shows a shortened version of this table (Table 2); values are identical.")
    table("tab_S8_characteristics_full", "Characteristics of children and prediction pairs by block (analysis 1, $h = 1$; full version of Table 2)", ch,
          "Values describe prediction pairs unless marked [children], which are child-level values from each child's first pair in the block; "
          "a child usually contributed several pairs and can appear in more than one block. Training: target months 2023--2024; threshold tuning: "
          "prediction and target months within 2025; test: prediction months from January 2026. HAZ, height-for-age z-score; IQR, interquartile range; "
          "MUAC, mid-upper arm circumference.")

# ------------------------------------------------------------------ S9 additional pre-specified results
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 7.5})
def fd(x):
    return f"{x:.4f}" if round(x, 3) == 0 and x != 0 else f"{x:.3f}"
MLAB = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "macro_precision": "Macro-precision", "macro_recall": "Macro-recall",
        "macro_f1": "Macro-F1", "minority_macro_recall": "Minority macro-recall", "minimum_minority_recall": "Minimum minority recall"}
r25 = []
for h in (1, 3, 6):
    a = V4 / f"primary/h{h}/artifacts"
    m = pd.read_csv(a / "test_classification_metrics.csv"); m = m[m.subset == "all"].set_index("method")
    d = pd.read_csv(a / "test_child_bootstrap_paired_differences.csv")
    d = d[(d.subset == "all") & (d.comparison == "mce_ilcadt minus persistence")].set_index("metric")
    for k, lab in MLAB.items():
        est = m.loc["mce_ilcadt", k] - m.loc["persistence", k]
        r25.append({"h": h, "Metric": lab, "Persistence": f3(m.loc["persistence", k]), "MCE--ILCADT": f3(m.loc["mce_ilcadt", k]),
                    "Difference (95% CI)": f"{fd(est)} ({d.loc[k, 'ci_low']:.3f} to {d.loc[k, 'ci_high']:.3f})"})
SLAB = {"status_t=normal": "Normal", "status_t=stunted": "Stunted", "status_t=severely_stunted": "Severely stunted"}
r26, r27, r28 = [], [], []
PN = {"mce": "MCE", "v4_mapped": "Analysis-1 model, mapped (raw)", "v4_mapped_recal": "Analysis-1 model, mapped (recalibrated)",
      "lr_simple": "LR, simple", "lr_33": "LR, 33 predictors", "best_base": "Best base learner", "RF": "Random forest", "LGBM": "LightGBM",
      "XGB": "XGBoost", "CAT": "CatBoost", "GBC": "Gradient boosting", "ADA": "AdaBoost"}
for h in (1, 3, 6):
    a = V5 / f"primary/h{h}/artifacts"
    t = pd.read_csv(a / "threeclass_metrics.csv")
    for sub, lab in SLAB.items():
        for meth, ml in [("persistence", "Persistence"), ("mce_argmax", "MCE"), ("mce_ilcadt", "MCE--ILCADT")]:
            r = t[(t.subset == sub) & (t.method == meth)].iloc[0]
            r26.append({"h": h, "Status at t": lab, "Method": ml, "Pairs": f"{int(r.n):,}", "Classes present": int(r.classes_present),
                        "Accuracy": f3(r.accuracy), "Macro-F1": f3(r.macro_f1), "Minority macro-recall": f3(r.minority_macro_recall)})
    pm = pd.read_csv(a / "threeclass_probability_metrics.csv")
    for r in pm.itertuples():
        r27.append({"h": h, "Model": PN.get(r.method, r.method), "ROC-AUC (macro, one-vs-rest)": f3(r.roc_auc_ovr_macro), "NLL": f3(r.nll),
                    "Brier score": f3(r.brier_multiclass), "ECE (top label)": f3(r.ece_top_label), "ECE (classwise)": f3(r.ece_classwise_macro)})
for run, lab in [("primary", "Primary (confirmed)"), ("sens_no_baseline_confirmation", "Without baseline confirmation"), ("sens_unconfirmed", "Without any confirmation")]:
    for h in (1, 3, 6):
        a = V5 / f"{run}/h{h}/artifacts"
        t = pd.read_csv(a / "threeclass_metrics.csv"); t = t[t.subset == "all"].set_index("method")
        d = pd.read_csv(a / "threeclass_paired_differences.csv")
        d = d[(d.comparison == "mce_ilcadt minus mce_argmax") & (d.metric == "macro_f1")].iloc[0]
        est = t.loc["mce_ilcadt", "macro_f1"] - t.loc["mce_argmax", "macro_f1"]
        r28.append({"Target definition": lab, "h": h, "MCE (arg-max)": f3(t.loc["mce_argmax", "macro_f1"]), "MCE--ILCADT": f3(t.loc["mce_ilcadt", "macro_f1"]),
                    "Difference (95% CI)": f"{fd(est)} ({d.ci_low:.3f} to {d.ci_high:.3f})"})
section("S9 Additional pre-specified results",
        "Tables S25--S28 and Figs.~S2--S3 report pre-specified results that are not shown in the main text, taken from the frozen outputs (no new analysis).")
table("tab_S9a_a1_paired_supporting", "Analysis 1: MCE--ILCADT minus persistence for all classification metrics (all test pairs)", pd.DataFrame(r25),
      "Paired differences on the same test pairs; 95\\% CIs from the child-level cluster bootstrap. Minority macro-recall is the mean recall of the stunted and severely stunted classes; "
      "minimum minority recall is the lower of the two recalls for each method, which can belong to different classes for the two methods (at $h = 6$: stunted for persistence, severely stunted for MCE--ILCADT). "
      "No interval was estimated for the difference in recall of a single class.")
table("tab_S9b_a2_status_t", "Analysis 2: three-class direction of change by height-for-age status at month $t$ (test data, point estimates)", pd.DataFrame(r26),
      "Children normal at $t$ cannot improve and severely stunted children cannot worsen, so only two outcome classes occur in these strata; macro-averaged metrics are computed over the classes present and are not comparable with the three-class values for all pairs (Table S17).")
table("tab_S9c_a2_prob", "Analysis 2: probability metrics for the three-class direction of change (test data)", pd.DataFrame(r27),
      "Computed from predicted probabilities after masking impossible classes, not from threshold-adjusted decision scores. NLL, negative log-likelihood; ECE, expected calibration error.")
table("tab_S9d_a2_thr_sens", "Analysis 2: effect of class-specific thresholds on three-class macro-F1 of MCE under the primary and the two sensitivity target definitions", pd.DataFrame(r28),
      "Paired differences on the same test pairs; 95\\% CIs from the child-level cluster bootstrap.")
# Fig S2: analysis-1 per-class calibration curves
cc = pd.read_csv(V4 / "primary/calibration/calibration_per_class_curve.csv"); cc = cc[cc.model == "meta_dnn"]
fig, ax = plt.subplots(3, 3, figsize=(7.0, 6.3))
for i, h in enumerate((1, 3, 6)):
    for j, k in enumerate(("normal", "stunted", "severely stunted")):
        x = cc[(cc.h == h) & (cc["class"] == k)].sort_values("predicted"); g = ax[i, j]
        g.fill_between(x.predicted, x.ci_low, x.ci_high, color="#9bb7d4", lw=0); g.plot(x.predicted, x.observed_spline, color="#1f4e79", lw=1)
        lim = max(x.predicted.max(), x.ci_high.max()); g.plot([0, lim], [0, lim], ls="--", color="grey", lw=0.6)
        g.set_title(f"h = {h}, {k}", fontsize=7.5); g.set_xlabel("Predicted probability"); g.set_ylabel("Observed proportion")
fig.tight_layout(); fig.savefig(A.out / "fig_S2_calibration_a1.pdf"); plt.close(fig)
TEX.append(r"""\begin{figure}[h]\centering\includegraphics[height=0.72\textheight]{fig_S2_calibration_a1.pdf}
\caption{Analysis 1: flexible calibration curves of the Meta-DNN by class (one-vs-rest; test data). Lines show the smoothed observed proportion with 95\% confidence bands; the dashed line is perfect calibration. Every class met the pre-specified condition of at least 200 events and 200 non-events.}\label{fig_S2_cal_a1}\end{figure}""")
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6))
CN = {"mce": "MCE", "v4_mapped_recal": "Analysis-1 model, mapped (recalibrated)", "lr_simple": "LR, simple", "lr_33": "LR, 33 predictors", "best_base": "Best base learner"}
for i, h in enumerate((1, 3, 6)):
    c2 = pd.read_csv(V5 / f"primary/h{h}/artifacts/primary_calibration_curve.csv"); g = ax[i]; lim = 0
    for meth, lab in CN.items():
        x = c2[c2.method == meth].sort_values("predicted"); g.plot(x.predicted, x.observed_spline, lw=1, label=lab); lim = max(lim, x.predicted.max(), x.observed_spline.max())
    g.plot([0, lim], [0, lim], ls="--", color="grey", lw=0.6); g.set_title(f"h = {h}", fontsize=7.5); g.set_xlabel("Predicted risk"); g.set_ylabel("Observed proportion")
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=6.5, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(A.out / "fig_S3_calibration_a2.pdf"); plt.close(fig)
TEX.append(r"""\begin{figure}[h]\centering\includegraphics[width=0.95\linewidth]{fig_S3_calibration_a2.pdf}
\caption{Analysis 2 (exploratory): flexible calibration curves for new stunting among children normal at month $t$ (test data). The dashed line is perfect calibration. Calibration-in-the-large and slopes are given in Table S15.}\label{fig_S3_cal_a2}\end{figure}""")

doc = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=18mm,landscape]{geometry}\usepackage[T1]{fontenc}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}
\usepackage{booktabs,longtable,caption,graphicx}\renewcommand{\thefigure}{S\arabic{figure}}\captionsetup{font=small,labelfont=bf,justification=raggedright,singlelinecheck=false}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}
{\Large\bfseries Additional file 3}\par\smallskip
{\large Supplementary methods and tables}\par\smallskip
\noindent Can a meta-confidence ensemble with class-specific decision thresholds outperform persistence in predicting future nutritional status and new stunting risk in children under five? A development and temporal validation study.\par\smallskip
\noindent All values are taken from the frozen analysis outputs (analysis 1: plan v4; analysis 2: plan v5, exploratory). Confidence intervals are 95\% child-level cluster-bootstrap intervals (2\,000 replicates) unless stated otherwise. MCE, Meta-DNN with arg-max decision; MCE--ILCADT, Meta-DNN with class-specific thresholds.
""" + "\n".join(TEX) + "\n\\end{document}\n"
(A.out / "Additional_file_3.tex").write_text(doc)
print("✓ Additional file 3 written to", A.out)
