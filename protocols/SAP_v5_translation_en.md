# Statistical Analysis Plan (SAP) v5 — English translation

> **Unofficial translation.** The binding document is the frozen Indonesian original `SAP_v5_original_id.md` (SHA-256 recorded in `freeze_logs/FREEZE_LOG_v5.txt`). This translation was prepared on 20 October 2026, after the analysis, for readers of the deposit. Where the two differ, the original applies. Numbers are written with a decimal point here (the original uses a decimal comma). File paths mentioned in the text refer to the authors' working project and are not part of this deposit.

## Detection of confirmed changes in height-for-age status of children under five (month t → t+h), North Buton District

Version: 1.0 (FROZEN) · 17 October 2026 · approved by the user 17 October 2026 (content identical to draft v0.3, SHA-256 d2c8d853494b0d9c45987d6a4fd2fd208538ada34e7d60a679682fb988d50df4)
Revised from v0.1 and v0.2 on the basis of two independent reviews (claude/v5_independent_review.md) and the user's decisions of 17 October 2026: (1) status at month t is confirmed by month t−1; (2) the primary endpoint is confirmed new stunting among normal children, primary metric AUPRC; (3) a contribution is claimed only if consistent at h = 1, 3, and 6.
This document is frozen (SHA-256 recorded in FREEZE_LOG_v5) BEFORE the v5 code is written and run. Changes only through a dated amendment reported in the manuscript. Terms follow the v4 decision log.

---

## 1. Objective
To assess whether the MCE–ILCADT architecture (paper 1), using only information up to prediction month t, can detect children who will undergo a confirmed change in height-for-age status at month t+h — in particular new stunting among normal children — better than simpler comparators.

Rationale: in v4, the future-status model did not exceed persistence; 89–95% of children did not change status and persistence never detects a change (claude/v4_results_summary.md).

## 2. Evidential status
- The 2026 test data were opened in v4. All v5 results on the 2026 test data are reported as **exploratory**; no contribution is claimed from these data.
- **Confirmatory:** the frozen v5 model is applied without retraining to data with prediction month ≥ October 2026. Sample size is calculated with the method of Riley et al. 2024 for the primary (binary) endpoint and with a bootstrap simulation of the power of the claim rule (section 9), and the protocol is written as a dated amendment before the new data are opened. For h = 6, confirmation is possible only after data up to about May 2027 are available.

## 3. Data, population, and target
- Master dataset v4 (234,073 rows; 10,321 children), v4 eligibility rules, e-PPGBM height-for-age category labels (Ministry of Health Regulation 2/2020).
- Horizons: h = 1, 3, 6.
- **Pairs:** the same child with eligible observations at months t−1, t, t+h, and t+h+1 (exact), and **status at month t−1 equal to status at month t** (confirmed baseline status). No imputation.
- **Target population:** routinely monitored children under five with stable status over the last two months who are still present at months t+h and t+h+1. Pairs without an observation at t+h+1 are reported in an attrition table (month-t status and unconfirmed direction of change), as in v4.

### 3.1 Target
Severity order: normal (0) < stunted (1) < severely stunted (2). d1 = sign(s(t+h) − s(t)); d2 = sign(s(t+h+1) − s(t)).
- **Worsened** if d1 = d2 = +1; **improved** if d1 = d2 = −1; otherwise **unchanged** (including transient changes).
- Class order in the code: unchanged (0), worsened (1), improved (2).
- **Primary endpoint:** confirmed new stunting = worsened among children normal at month t (binary, evaluated among children normal at month t).
- Observations at t+h and t+h+1 are used only for the label. The t−1 condition uses past information (available at month t), so it does not leak.

Rationale: a single category change can arise from measurement variation and regression to the mean, at both the initial and the final measurement (Barnett et al. 2005; Roth et al. 2025; Leroy et al. 2025; Cameron et al. 2005; Cliffer et al. 2022); modelling of differences between repeated measurements assumes length noise of about 0.5 cm (Wright et al. 2022); reliability of length/height measurement in the WHO MGRS (WHO MGRS Group 2006); the concept of persistent stunting across measurements (Alam et al. 2020); exclusion of regression to the mean in the definition of catch-up (Desmond & Casale 2017). Data: in training h = 1 (without the t−1 condition), 43.4% of confirmed "improved" and 29.0% of confirmed "worsened" came from a month-t status that had just changed from t−1, compared with 4.1% in the unchanged class.

### 3.2 Split (time blocks; month t−1 may precede the block, the confirmation month is in the same block)
| Part | Rule |
|---|---|
| Training data | Target month ≥ January 2023 and confirmation month (t+h+1) ≤ December 2024 |
| Threshold-tuning data | Prediction month ≥ January 2025 and confirmation month ≤ December 2025 |
| Test data (exploratory) | Prediction month ≥ January 2026 (confirmation month ≤ September 2026) |

Number of pairs (training / tuning / test):
| h | Pairs | Unchanged | Worsened | Improved |
|---|---|---|---|---|
| 1 | 69,113 / 38,791 / 26,115 | 67,538 / 37,997 / 25,517 | 775 / 446 / 361 | 800 / 348 / 237 |
| 3 | 61,585 / 29,499 / 17,566 | 58,634 / 28,195 / 16,728 | 1,461 / 761 / 511 | 1,490 / 543 / 327 |
| 6 | 51,172 / 17,336 / 6,494 | 47,534 / 16,039 / 6,035 | 1,736 / 780 / 307 | 1,902 / 517 / 152 |

Primary endpoint — children normal at month t (pairs; new stunting; incidence):
| h | Training | Tuning | Test |
|---|---|---|---|
| 1 | 59,427; 610 (561 children); 1.03% | 33,076; 353; 1.07% | 22,280; 294 (293 children); 1.32% |
| 3 | 53,075; 1,156 (618 children); 2.18% | 25,210; 624; 2.48% | 15,006; 427 (264 children); 2.85% |
| 6 | 44,271; 1,439 (553 children); 3.25% | 14,853; 666; 4.48% | 5,555; 270 (186 children); 4.86% |

Note: the h = 6 test data cover only prediction months January–February 2026. The three-class distribution is a *highly imbalanced multiclass classification* (minority classes 0.9–4.7% across all data parts; 1.1–3.7% in the training data).

## 4. Predictors
The 33 v4 predictors (SAP v4 section 4), all from data up to month t, including status and HAZ at month t. Programme variables, village, posyandu, and centre are not used as predictors.

## 5. Model (paper 1 architecture; retained from v4)
- 6 base learners, v4 hyperparameters, no tuning, no resampling and no class weighting.
- OOF 5 folds by child. Child stratum = the rarest direction-of-change class experienced by the child (rarity order from the training data). Children per stratum (unchanged / worsened / improved-only): h1 5,185 / 682 / 363; h3 4,836 / 742 / 415; h6 4,150 / 650 / 508. Each fold must contain ≥ 2 children for the worsened and improved classes.
- 6 × 3 = 18 meta-features; v4 Meta-DNN and epoch rule; v4 ILCADT (per-class thresholds from the tuning data, grid 0.01–0.99, one-vs-rest F1, v4 tie-break).
- **Mask on impossible classes** (all models and comparators): for children normal at month t, P(improved) = 0; for severely stunted children, P(worsened) = 0; probabilities are renormalised before metrics and before ILCADT thresholds are learned, and the mask is applied again to the final ILCADT decision scores so that an impossible class is never selected.

Rationale for no resampling: class-imbalance corrections worsen calibration without increasing AUC, and threshold shifting gives similar sensitivity/specificity (van den Goorbergh et al. 2022); calibration damage also occurs in machine learning models (Carriero et al. 2025).

## 6. Comparators (on the same test pairs)
| Comparator | Definition | Rationale |
|---|---|---|
| Persistence | Always "unchanged" (for the primary endpoint: provides no risk ranking; AUPRC = incidence) | Naive comparator |
| Mapped v4 model | Frozen v4 Meta-DNN probabilities: P(worsened) = Σ_{k>s} p_k, P(unchanged) = p_s, P(improved) = Σ_{k<s} p_k; matched per (child, month t) with a fingerprint check; argmax and ILCADT. AUPRC (primary endpoint) uses the raw mapping. A recalibrated variant (multinomial logistic recalibration, tuning data only) is used only for probability metrics (NLL, Brier, ECE, calibration, DCA) | The v4 status model mathematically contains direction-of-change probabilities (without confirmation); tests whether dedicated training adds information |
| Simple logistic regression | Month-t status as two 0/1 indicator variables (reference normal), HAZ at t and age at t with restricted cubic splines (4 knots), status × HAZ t interaction, 1-month HAZ change + observed indicator; L2 penalty (C = 1.0) on standardised predictors, lbfgs solver; median imputation from the training data; argmax and ILCADT | Christodoulou et al. 2019; Clift et al. 2023; Leroy et al. 2025 (latest LAZ the strongest predictor); coding of categorical predictors and splines: Steyerberg 2019 |
| 33-predictor logistic regression | The same 33 predictors as MCE; L2 penalty (C = 1.0), standardisation and imputation as above; argmax and ILCADT | Like-for-like comparison (same predictors, different algorithm; Christodoulou et al. 2019) |
| Ablation | Best base learner (± ILCADT), MCE (± ILCADT) | As in v4 |

The L2 penalty is used because impossible classes cause perfect separation in unpenalised logistic regression.

## 7. Evaluation (test data, per horizon)
**7.1 Primary metric:** AUPRC for confirmed new stunting among children normal at month t, from the probability P(worsened) (Saito & Rehmsmeier 2015), computed as step-wise *average precision* (scikit-learn `average_precision_score`), not trapezoidal interpolation, which overstates the area under the precision–recall curve (Davis & Goadrich 2006). No-skill reference value = incidence.

**7.2 Secondary metrics**
- Primary endpoint: AUC, sensitivity and PPV when the 5% and 10% of normal children with the highest risk are monitored, net benefit (decision curve analysis, risk threshold range 0.5–10%, the same for all three horizons; Vickers & Elkin 2006; Van Calster et al. 2018). The action assessed is closer monitoring (additional visit or counselling at the posyandu), which is cheap and low-risk; low thresholds are therefore relevant. A threshold of 10% means that staff are willing to monitor up to 9 children without the event to find 1 child who will develop new stunting.
- Calibration of the primary endpoint (CITL, slope; curve if ≥ 200 events and ≥ 200 non-events; Van Calster et al. 2019).
- Three classes (continuity with paper 1): macro-F1, per-class recall/precision/F1, balanced accuracy, minority macro-recall, minimum minority recall, confusion matrix, one-vs-rest AUC, NLL, Brier, ECE.
- Strata of month-t status: stunted children (worsened and improved), severely stunted children (improved).
- Descriptive subgroups (TRIPOD+AI): sex, age group (< 24 and ≥ 24 months), centre.
- DCA and probability metrics use probabilities, never ILCADT scores.

**7.3 Uncertainty:** child-level cluster bootstrap, 2,000 replicates, 95% percentile CIs. Paired AUPRC differences (primary endpoint): MCE − mapped v4 model; MCE − simple logistic regression; MCE − 33-predictor logistic regression. Secondary differences (three-class macro-F1, same test pairs): MCE+ILCADT − mapped v4 model+ILCADT; MCE+ILCADT − simple logistic regression+ILCADT; MCE+ILCADT − 33-predictor logistic regression+ILCADT; MCE+ILCADT − MCE; best base+ILCADT − best base.

**7.4 Claim rule (set before running)**
- A contribution of MCE (Meta-DNN probabilities; ILCADT thresholds do not affect AUPRC) to the detection of new stunting is claimed only if, on the **confirmatory data**, the 95% CI of the AUPRC difference of MCE against **all three** comparators (mapped v4 model, simple logistic regression, 33-predictor logistic regression) lies entirely above zero **at h = 1, 3, and 6** (intersection-union rule; no multiplicity correction needed).
- On the 2026 test data (exploratory), results are reported as "consistent" or "not consistent" with this rule, without a contribution claim.
- If the rule is not met, it is reported that the model was not shown to be better than the simpler comparators.

**7.5 Undefined metrics:** as in SAP v4 7.6. Precision of a class that is never predicted = 0 if the class occurs in the data (applies to persistence for the worsened/improved classes), and this is noted.

## 8. Sensitivity analyses (h = 1, 3, 6)
1. Without month-t confirmation: the t−1 observation requirement and status t−1 = status t are removed; t+h+1 confirmation retained.
2. Without any confirmation: direction of change from t to t+h only; the t−1 and t+h+1 requirements removed (v4 pairs).

## 9. Sample size
- Development: adequacy of the sample size for the primary endpoint is assessed with the criteria of Riley et al. 2020 (binary outcome) and reported.
- Confirmatory, in two parts, written as an amendment before the new data are opened: (a) precision of primary-endpoint performance (calibration, C-statistic, net benefit) with Riley et al. 2024; (b) power of the 7.4 claim rule through bootstrap simulation: the number of months of confirmatory data needed for all nine CIs of AUPRC differences (3 comparators × 3 horizons) to be able to exclude zero, with assumed differences set from the tuning data (not the test data). Horizon h = 6 is decisive because it has the fewest events.

## 10. Freezing and reproducibility
As in SAP v4 section 11: reporting follows TRIPOD+AI (Collins et al. 2024) and risk of bias is self-assessed with PROBAST+AI (Moons et al. 2025); stage-wise code audit by an independent agent; freeze manifest and FREEZE_LOG_v5 before test metrics; same environment as paper 1 (Colab runtime 2026.04, Python 3.12.13, TensorFlow 2.20.0, scikit-learn 1.6.1, XGBoost 3.1.2, LightGBM 4.6.0, CatBoost 1.2.8); independent rerun.

## 11. Planned limitations
- 2026 results exploratory; confirmation awaits new data (h = 6 around May 2027).
- The t−1 and t+h+1 observation requirements narrow the population to routinely monitored children; the h = 6 test data are small.
- The improved class may still be affected by regression to the mean even when confirmed.
- No predictors of the causes of change (illness, intake, socio-economic factors) in e-PPGBM.

## New references (not yet in the manuscript bibliography)
- Barnett AG, van der Pols JC, Dobson AJ. Regression to the mean: what it is and how to deal with it. Int J Epidemiol. 2005;34(1):215–220.
- van den Goorbergh R, van Smeden M, Timmerman D, Van Calster B. The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression. J Am Med Inform Assoc. 2022;29(9):1525–1534. doi:10.1093/jamia/ocac093
- Carriero A, Luijken K, de Hond A, Moons KGM, Van Calster B, van Smeden M. The harms of class imbalance corrections for machine learning based prediction models: a simulation study. Stat Med. 2025;44(3–4):e10320. doi:10.1002/sim.10320
- Christodoulou E, Ma J, Collins GS, Steyerberg EW, Verbakel JY, Van Calster B. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. J Clin Epidemiol. 2019;110:12–22. doi:10.1016/j.jclinepi.2019.02.004
- Davis J, Goadrich M. The relationship between precision-recall and ROC curves. In: Proceedings of the 23rd International Conference on Machine Learning (ICML). 2006. p. 233–240.
- Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS One. 2015;10(3):e0118432.
- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565–574. doi:10.1177/0272989X06295361
- Van Calster B, Wynants L, Verbeek JFM, Verbakel JY, Christodoulou E, Vickers AJ, Roobol MJ, Steyerberg EW. Reporting and interpreting decision curve analysis: a guide for investigators. Eur Urol. 2018;74(6):796–804. doi:10.1016/j.eururo.2018.08.038
- Riley RD, Ensor J, Snell KIE, Harrell FE, Martin GP, Reitsma JB, Moons KGM, Collins G, van Smeden M. Calculating the sample size required for developing a clinical prediction model. BMJ. 2020;368:m441. doi:10.1136/bmj.m441
- Riley RD, Snell KIE, Archer L, Ensor J, Debray TPA, Van Calster B, van Smeden M, Collins GS. Evaluation of clinical prediction models (part 3): calculating the sample size required for an external validation study. BMJ. 2024;384:e074821. doi:10.1136/bmj-2023-074821
- Wright CM, Haig C, Harjunmaa U, Sivakanthan H, Cole TJ. Assessing the optimal time interval between growth measurements using a combined data set of weights and heights from 5948 infants. Arch Dis Child. 2022;107(4):341–345. doi:10.1136/archdischild-2021-322479
- WHO Multicentre Growth Reference Study Group. Reliability of anthropometric measurements in the WHO Multicentre Growth Reference Study. Acta Paediatr Suppl. 2006;450:38–46. doi:10.1111/j.1651-2227.2006.tb02374.x

Other references (Roth 2025, Leroy 2025, Cameron 2005, Cliffer 2022, Alam 2020, Desmond 2017, Clift 2023, Van Calster 2019, Steyerberg 2019, TRIPOD+AI 2024, PROBAST+AI 2025) are already in the manuscript bibliography.
