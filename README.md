# MCE–ILCADT future nutritional status: code and analysis plans

Companion deposit for the manuscript

> Alam WOSN, Sari RF, Harwahyu R. *Can a meta-confidence ensemble with class-specific decision thresholds outperform persistence in predicting future nutritional status and new stunting risk in children under five? A development and temporal validation study.* (manuscript in preparation).

Version 1.0 (code and plans) · November 2026 · Corresponding author: Riri Fitri Sari (riri@ui.ac.id)

## What this deposit contains

It contains the frozen analysis plans, the analysis code, the freeze logs and the WHO reference tables. It holds no child-level data and no result files (see *Data access* and *Aggregate results*).

```
protocols/        frozen statistical analysis plans (Indonesian originals) and English translations
freeze_logs/      SHA-256 freeze records written before test metrics were computed
code/
  data_preparation/   parsing of the monthly e-PPGBM exports and construction of the pseudonymised master dataset
  v4_analysis1/       Analysis 1: status at month t+h vs persistence (SAP v4)
  v5_analysis2/       Analysis 2: confirmed change / new stunting (SAP v5)
  v3_single_centre/   earlier single-centre analysis (one centre; superseded, see Provenance)
  figures_tables/     scripts for the manuscript figures, Table 2 and Additional file 3
SHA256SUMS.txt    checksums of every file in this deposit (as deposited)
CITATION.cff      citation metadata
LICENSE-CODE, LICENSE-DOCS
```

## Analyses and their evidential status

| Analysis | Plan (frozen) | Code | Status in the manuscript |
|---|---|---|---|
| 1. Future height-for-age status (normal / stunted / severely stunted) at h = 1, 3 and 6 months (primary; h = 2, 4 and 5 for the horizon curve) vs persistence; primary metric macro-F1 | `SAP_v4_original_id.md`, 16 Oct 2026 | `code/v4_analysis1/mce_ilcadt_fp_v4.py` | Confirmatory for 11 of 12 centres; for one centre the test-period data had been evaluated earlier (see Provenance) |
| 2. Confirmed direction of change; primary endpoint AUPRC for confirmed new stunting among children normal at month t (h = 1, 3, 6) | `SAP_v5_original_id.md`, 17 Oct 2026 | `code/v5_analysis2/mce_ilcadt_fp_v5.py` | Exploratory on the 2026 test data; the pre-specified claim rule was not met. Confirmation is planned on prediction months from October 2026, with the sample size set in a dated amendment before those data are opened |

The English files `SAP_v4_translation_en.md` and `SAP_v5_translation_en.md` are unofficial translations prepared after the analysis; the Indonesian originals are binding.


## Provenance (analysis history)

1. **v2** (single centre, 31 monthly exports, frozen 11 Oct 2026): test results opened, then superseded when a 32nd monthly export became available (`freeze_logs/FREEZE_LOG_v2_superseded.txt`).
2. **v3** (same centre, 32 monthly exports, January 2024–August 2026, frozen 11 Oct 2026): rerun with unchanged model code, hyperparameters and split boundaries (`FREEZE_LOG_v3.txt`, `code/v3_single_centre/`). Its test period overlaps the Analysis 1 test period for this centre, which is why Analysis 1 is described as confirmatory for 11 of the 12 centres only.
3. **v4** — Analysis 1, district-wide (12 centres, January 2022–September 2026).
4. **v5** — Analysis 2, designed after the v4 test data had been seen.

Differences between v3 and v4 are listed in SAP v4 section 12 and in Additional file 3 (S1) of the manuscript.

## Running the code

The analyses were run on Google Colab (runtime 2026.04; Python 3.12.13, TensorFlow 2.20.0, scikit-learn 1.6.1, XGBoost 3.1.2, LightGBM 4.6.0, CatBoost 1.2.8, statsmodels 0.14.6; CPU). The main libraries are pinned in each `requirements.txt` (the v3 file lists the versions of its own, earlier run); in the v4 and v5 files `pyarrow`, `statsmodels` and `patsy` are not pinned and were taken from the Colab runtime listed above, and other dependencies (for example NumPy and pandas) came from that runtime. An exact reproduction of the environment therefore needs that Colab runtime version. All scripts need the restricted master dataset; they cannot be run on the material in this deposit alone.

**Data preparation** (`code/data_preparation/`; working scripts, run in order). Run them from a private working folder outside the repository and give the repository paths as absolute paths; `reparse_ids.py` and `stage2_pairs.py` read and write files in the current working directory:

```bash
REPO="/path/to/MCE_FP_paper2_code"      # this repository
RAW="/path/to/restricted_exports"       # monthly .xls exports: <year>/<n>. <month>.xls
OUT="/path/to/private_work"             # private working folder (never commit it)

mkdir -p "$OUT/private"                 # build_master.py writes the salt here and does not create the folder
cd "$OUT"

python "$REPO/code/data_preparation/parse_all.py" "$RAW" "$OUT"        # -> kab_raw_all.pkl, file_meta.json
python "$REPO/code/data_preparation/reparse_ids.py" "$RAW"             # adds the record-link identifier to kab_raw_all.pkl
python "$REPO/code/data_preparation/build_master.py" \
  "$OUT" "$OUT" \
  "$REPO/code/data_preparation" \
  "$REPO/code/v3_single_centre/who_tables"                             # -> master_v4_pseudonymised.pkl, master_v4_audit.json
python "$REPO/code/data_preparation/stage2_pairs.py"                   # prints prediction-pair counts per block (SAP v4 section 3)
python -c "import pandas as pd; pd.read_pickle('master_v4_pseudonymised.pkl').to_parquet('master_v4_pseudonymised.parquet')"
```

`build_master.py` applies the exclusion and eligibility rules. It imports `who_haz_copy.py` from its third argument; `code/data_preparation/who_haz_copy.py` is the copy used in the study and is byte-identical to `code/v3_single_centre/who_haz.py` (SHA-256 `3b6696131901a6daf68d38039b78844a9d6959fb4054ea587a4537fa5786a14f`, as recorded in `freeze_logs/FREEZE_LOG_v3.txt`). The analyses read the parquet file.

`build_master.py` generates a new random salt for the child pseudonyms and writes it to `<out_dir>/private/`. The salt used in the study is secret and not deposited, so a rebuild yields different pseudonyms but the same counts.

**Analysis 1** (`code/v4_analysis1/`), configured by environment variables:

```bash
export MCE_FP_MASTER_PATH=/path/master_v4_pseudonymised.parquet
export MCE_FP_BASE_DIR=/path/runs/primary
export MCE_FP_SAP_PATH=/path/SAP_v4_original_id.md   # required; its hash is written to the freeze log
export MCE_FP_HORIZONS=1,2,3,4,5,6        # primary horizons 1,3,6; 2,4,5 for the horizon curve
python mce_ilcadt_fp_v4.py
# sensitivity analyses (one change each, separate MCE_FP_BASE_DIR):
#   MCE_FP_LABEL_SOURCE=who | MCE_FP_ABLATION=no_current_status | MCE_FP_EXCLUDE_FLAGGED=1
python calibration_per_class.py --run-root /path/runs/primary --horizons 1,3,6 --out /path/calibration
```

**Analysis 2** (`code/v5_analysis2/`) additionally reads the frozen Analysis 1 run for the mapped comparator:

```bash
export MCE_FP_MASTER_PATH=... MCE_FP_BASE_DIR=... MCE_FP_V4_RUN_ROOT=/path/v4/runs/primary
export MCE_FP_SAP_PATH=/path/SAP_v5_original_id.md   # required
export MCE_FP_V5_TARGET=confirmed   # sensitivity: no_baseline_confirmation | unconfirmed
python mce_ilcadt_fp_v5.py
```

Other variables: `MCE_FP_STAGES` (default `a,b,c,d,e,f`), `MCE_FP_RANDOM_STATE` (default 42), `MCE_FP_BOOTSTRAP` (default 2000), and `MCE_FP_SMOKE=1` (reduced run for testing only).

**Notebooks.** `MCE_FP_v4_colab_original_id.ipynb` and `MCE_FP_v5_colab_original_id.ipynb` are the notebooks used for the runs, kept unchanged (instructions in Indonesian). Their cells check SHA-256 hashes against the frozen files and stop on a mismatch; anyone adapting the code should remove or update those checks.

**Figures and tables** (`code/figures_tables/`): `paper2_dataset.py` (dataset tables and monthly counts), `paper2_characteristics.py` (Table 2 and its full version, Table S24 of Additional file 3), `paper2_figures.py` (Figures 2–4 and the precision–recall figure), `make_additional_file3.py` (Additional file 3), `Figure1_framework.tex` (Figure 1, standalone TikZ; compile with pdfLaTeX). Usage is given in each script's header. `paper2_dataset.py` checks that the master matches the approved counts and stops otherwise.

**v3** (`code/v3_single_centre/`): see the header of `mce_ilcadt_future_prediction.py`; `verify_against_reference.py` compares a rerun with the frozen single-centre outputs (not included; see *Aggregate results*).

## Frozen hashes (documentation)

SHA-256 values recorded at freezing; the full lists, including model and threshold files, are in `freeze_logs/` and in each run's `shared/FREEZE_LOG_*.txt` and `h*/artifacts/freeze_manifest.json`.

| File | SHA-256 |
|---|---|
| SAP v4 (`SAP_v4_original_id.md`) | `3ec9bbe2be7108641f07be531cee25e6eedd8b55d5cde9ddf9f056db592776a6` |
| SAP v5 (`SAP_v5_original_id.md`) | `dab1e43e42c6e696a963fbba845ab6bbee9ae0096a7ce618993fcb04c48a1aa0` |
| Analysis 1 code (`mce_ilcadt_fp_v4.py`) | `7e051fb18c02c23f99b2e0cfefd97885571ff750f79f83ccb8a68f47ce52d162` |
| Analysis 2 code (`mce_ilcadt_fp_v5.py`) | `1010f55b4c3b1ebbf181bade1120988b22b118f684c2d9ff1551ed9da1731bed` |
| v3 code (`mce_ilcadt_future_prediction.py`) | `ce5038931aede2104a1a7a4df55f090f62bd5c803d1647e889e7e86b9934874a` |

Trained models are not deposited because they were fitted on restricted data.

## Aggregate results

The frozen aggregate outputs of the analyses (metrics, bootstrap intervals, thresholds, subgroup and centre-level tables) are not included in this release. Publishing centre-level aggregates of these restricted records requires permission from the data custodians, which has not yet been obtained. Until then they are available from the corresponding author on reasonable request. If permission is granted, they will be added as a new version of this record.

## Data access

The data were exported from the Indonesian Ministry of Health's restricted-access nutrition surveillance system (e-PPGBM, SIGIZI KESGA) for 12 primary health care centres in North Buton Regency and were used with a research recommendation from the North Buton Regency government. Because the records contain personal identifiers of children and caregivers, they are not public. De-identified data may be made available from the corresponding author on reasonable request and with permission of the North Buton Regency government.

The WHO length/height-for-age tables in `code/v3_single_centre/who_tables/` are from the WHO Child Growth Standards (WHO Multicentre Growth Reference Study Group, 2006). The six JSON files are unmodified copies of those distributed with pygrowup 0.8.2 (https://github.com/ewheeler/pygrowup; BSD licence, copyright UNICEF and individual contributors); its licence is reproduced in `code/v3_single_centre/who_tables/LICENSE-pygrowup.txt`.

## Licence

Licences follow BMC and Springer Nature practice:

- code: MIT, an open-source licence approved by the Open Source Initiative (`LICENSE-CODE`);
- protocols, freeze logs and documentation: CC BY 4.0 (`LICENSE-DOCS`).

The WHO tables in `code/v3_single_centre/who_tables/` remain subject to WHO's terms and to the pygrowup BSD licence (`LICENSE-pygrowup.txt`).
