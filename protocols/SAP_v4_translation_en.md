# Statistical Analysis Plan (SAP) v4 — English translation

> **Unofficial translation.** The binding document is the frozen Indonesian original `SAP_v4_original_id.md` (SHA-256 recorded in `freeze_logs/FREEZE_LOG_v4.txt`). This translation was prepared on 20 October 2026, after the analysis, for readers of the deposit. Where the two differ, the original applies. Numbers are written with a decimal point here (the original uses a decimal comma).

## Prediction of height-for-age status of children under five at month t+h from routine e-PPGBM data, North Buton District

Version: 1.0 (FROZEN) · Date: 16 October 2026
Status: **approved and frozen 16 October 2026**. After approval, this document is frozen (SHA-256 hash recorded) before the code is run and before the test data are opened. Changes after freezing are allowed only through a dated amendment reported in the manuscript.

Terms follow the glossary in the v4 decision log: prediction month (t), target month (t+h), prediction pair, training data, threshold-tuning data, test data.

---

## 1. Objective

To assess whether the MCE–ILCADT model, using only information available up to prediction month t, predicts height-for-age status (normal, stunted, severely stunted) at target month t+h better than persistence (the status at month t carried forward), in children under five monitored routinely at 12 primary health care centres (puskesmas) in North Buton District.

## 2. Data (Stage 1, approved)

- Source: 57 monthly e-PPGBM exports (SIGIZI KESGA), January 2022–September 2026, 12 centres, 93 villages, 100 posyandu.
- Master dataset v4: 234,073 rows, 10,321 children, 230,551 eligible observations (normal 190,959; stunted 31,106; severely stunted 8,486).
- Child identity: full ID from record linkage, replaced by a salted pseudonym (salt v4, secret). 16 children suspected of duplicate registration were excluded; 1 child with two dates of birth was assigned the modal date.
- Observations eligible as prediction month or target month: e-PPGBM height-for-age category normal, stunted, or severely stunted; HAZ −6 to +6; age 0–60 months; valid height.
- The tall class (HAZ > +3; 247 observations, 164 children) is not modelled and is reported descriptively.
- Label: e-PPGBM height-for-age category (Ministry of Health Regulation 2/2020; WHO 2006 standard). The class distribution is a *highly imbalanced multiclass classification* (severely stunted 3.7%).

## 3. Prediction task and prediction pairs (Stage 2, approved)

- Prediction pair: the same child with an eligible observation at month t and exactly at month t+h. No imputation of targets. Target months in 2022 are not used; prediction months in 2022 are allowed.
- Primary horizons: h = 1, 3, 6. Additional horizons: h = 2, 4, 5 (horizon curve). h = 12 is not analysed.
- Split (the same for all h = 1–6):

| Part | Rule |
|---|---|
| Training data | Target month January 2023–December 2024 |
| Threshold-tuning data | Prediction month ≥ January 2025 and target month ≤ December 2025 |
| Test data | Prediction month ≥ January 2026 (target month ≤ September 2026) |
| Discarded | Pairs crossing a boundary between parts |

- Number of prediction pairs (training / tuning / test): h1 90,956 / 48,866 / 34,969; h2 86,462 / 43,600 / 29,937; h3 82,489 / 38,405 / 25,071; h4 78,798 / 33,505 / 20,380; h5 74,934 / 28,701 / 15,980; h6 70,932 / 24,157 / 11,650.
- Evaluation aim: follow-up monitoring of children recorded in a later calendar period.

## 4. Predictors

All predictors use only data up to prediction month t.

**4.1 v3 predictors (28)**
- Static: sex, birth weight, birth length.
- Month t: age (months), weight, height, measurement position (lying), MUAC and MUAC-measured indicator, HAZ, WAZ, WHZ, height-for-age status at month t, weight-gain code (N, T, O, B).
- 3-month growth history: indicators of HAZ observed 1 and 3 months earlier; 1-month change in HAZ, WAZ, height, and weight; 3-month change in HAZ; number of observations up to t; months since the previous observation; number of stunted months before t; lowest HAZ up to t.

**4.2 12-month growth history (new, 5 columns)**
- Indicators of HAZ observed 6 and 12 months earlier.
- 6-month and 12-month change in HAZ.
- Number of stunted months in the 12 months before t.

**4.3 Not used**
- Programme variables (vitamin A, KPSP, MCH book, mothers' class, MBG).
- Village, posyandu, and centre are not predictors.

## 5. Model

**5.1 Base learners (6)**, hyperparameters as in v3, not tuned:
RF (400 trees, max depth 15), LightGBM (600; lr 0.05), XGBoost (600; lr 0.05; depth 6), CatBoost (800; lr 0.05; depth 6), Gradient Boosting (100; lr 0.1; depth 3), AdaBoost (300; lr 0.5). Seed 42. Median imputation and standardisation within each fold. No resampling and no class weighting.

**5.2 Out-of-fold probabilities**
5 folds grouped by child, stratified by the child's most severe target class, with at least 2 children per class per fold. Fixed class order (normal, stunted, severely stunted). 6 × 3 = 18 meta-features. For the threshold-tuning and test data, meta-features are the average of the 5 fold models.

**5.3 Meta-DNN (MCE)**
v3 architecture: input 18; Dense 64 (ReLU), Dropout 0.30; Dense 32 (ReLU), Dropout 0.20; softmax 3. Sparse categorical cross-entropy; Adam (lr 10⁻³, halved after 5 epochs without improvement, minimum 10⁻⁶); batch 64; maximum 200 epochs; early stopping 12 epochs. The number of epochs is chosen from the validation loss on 10% of the training-data children (meta-validation); the model is then retrained on all training-data children for that number of epochs. Seed 42, TensorFlow determinism enabled.

**5.4 ILCADT**
Decision score s_k = σ(logit p_k − logit τ_k), normalised; class = argmax s_k (always exactly one class; ties go to the lowest class index). τ_k is searched per class on a grid 0.01–0.99 (step 0.01) by maximising one-vs-rest F1 on the threshold-tuning data. Tie-break: higher recall, then higher precision, then closest to 0.5, then lower threshold. v3 thresholds are not used. The scores s_k are not probabilities and are not used for calibration.

## 6. Comparators and component ablation

Evaluated on the same test pairs:

| Configuration | Content |
|---|---|
| Persistence | Status at month t = prediction for the target month |
| Best base learner | Base learner with the highest macro-F1 (argmax) on the threshold-tuning data. Ties broken in the order RF, LGBM, XGB, CAT, GBC, ADA. Chosen before the test data are opened. |
| Best base + ILCADT | Selected base learner with per-class thresholds from the threshold-tuning data |
| MCE | Meta-DNN, argmax |
| MCE + ILCADT | Full model |

Additional file: all six base learners, each with argmax and + ILCADT.

## 7. Evaluation (test data, per horizon)

**7.1 Primary metric:** macro-F1.

**7.2 Supporting metrics**
- Accuracy, balanced accuracy, macro-precision, macro-recall.
- Per-class precision, recall, F1, and support; confusion matrix.
- Minority macro-recall (mean recall of stunted and severely stunted) and minimum minority recall. Minority classes are defined from the training data.
- For probabilistic models: macro one-vs-rest ROC-AUC, negative log-likelihood, multiclass Brier score, ECE (top-label and per class).

**7.3 Per-class calibration (MCE)**
Calibration-in-the-large and calibration slope with CIs accounting for clustering by child. Flexible calibration curves only when a class has ≥200 events and ≥200 non-events (Van Calster et al., 2019).

**7.4 Evaluation subsets**
- All test data.
- Pairs whose status changed between t and t+h.
- New children (not present in the training data).
- Per centre (12), from the primary test predictions without retraining.

**7.5 Uncertainty**
Child-level cluster bootstrap, 2,000 replicates, 95% percentile CIs. Paired differences are computed for: MCE + ILCADT − persistence; MCE − persistence; MCE + ILCADT − MCE; best base + ILCADT − best base; MCE − best base. No formal hypothesis tests; conclusions are based on the size of the differences and their CIs.

**7.6 Undefined metrics**
If a class does not occur in a subset, metrics for that class are reported as "not computable" (not 0), and macro averages are computed over the classes present, with a note.

## 8. Sensitivity analyses (h = 1, 3, 6)

1. **WHO label:** labels from HAZ recomputed with the WHO 2006 tables, replacing the e-PPGBM categories.
2. **Without month-t status:** height-for-age status and HAZ at month t removed from the predictors (persistence still uses the month-t status).
3. **Without flagged observations:** observations with a height decrease > 1.5 cm from the previous month are not used as prediction month or target month.

Each sensitivity analysis reruns the whole pipeline with only one change.

## 9. Additional analyses

- Horizon curve h = 1–6: macro-F1 of persistence, MCE, and MCE + ILCADT with 95% CI per horizon (Additional file).
- Descriptive report of the tall class.
- 12-fold leave-one-centre-out validation: deferred, decided after the primary run.

## 10. Planned data reporting and limitations

- Flow diagram and table of pair counts by data part, horizon, and class.
- Attrition audit: comparison of the status of children with and without a target month.
- Availability of the 12-month growth history in training vs test data (48.9% vs 73.0% for h = 1).
- Children in the test data who also appear in the training data.
- Quality of the 2022 data (used only as prediction month or history).

## 11. Freezing and reproducibility

1. Code is audited stage by stage (Stage 4) before the run.
2. Models, thresholds, the selected best base learner, test meta-features, and code are hashed (SHA-256) and recorded in FREEZE_LOG_v4 **before** test metrics are computed.
3. The test data are opened once. All results are reported, including those that do not favour the model.
4. Python and library versions are recorded. An independent rerun in a separate environment is repeated as in v3.

## 12. Deviations from v3 (reported in the manuscript)

- Dataset: 12 centres (previously 1), January 2022–September 2026 (previously January 2024–August 2026).
- Identity from record linkage (the NIK column is masked in the new exports).
- All horizons 1–6 use temporal threshold-tuning data (in v3, h = 6 used 20% of the training-data children).
- Predictors extended with the 12-month growth history.
- Component ablation and per-centre performance added.
- Rolling origin and leave-village-out validation not performed.
