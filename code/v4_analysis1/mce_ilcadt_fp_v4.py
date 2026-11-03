# -*- coding: utf-8 -*-
"""
MCE-ILCADT-FP v4: Meta-Confidence Ensemble with Independently Learned Class-Aware
Decision Thresholds for predicting toddler height-for-age status at calendar
month t+h from information available up to month t.

Dataset v4: pseudonymised master of 57 monthly e-PPGBM exports (SIGIZI KESGA),
12 puskesmas of Buton Utara District, January 2022 - September 2026.
Analysis plan: SAP_v4.md (frozen 16 October 2026, SHA-256 3ec9bbe2...).

Derived from the v3 pipeline (mce_ilcadt_future_prediction.py, sha ce5038...).
Unchanged from v3: six base learners and their hyperparameters, child-grouped
strict out-of-fold stacking (B x K = 18 meta-features), Meta-DNN architecture
and training rule, ILCADT decision rule and threshold search, metrics.
Changed for v4 (SAP section 12):
  * input is the pseudonymised master v4 (no raw exports, no PII, no salt);
  * one temporal design for every horizon h = 1..6 (training / threshold-tuning / test);
  * five 12-month growth-history predictors added (33 predictors in total);
  * component ablation: persistence, best base learner (+ ILCADT), MCE (+ ILCADT),
    plus all six base learners with and without ILCADT;
  * evaluation subsets: all, status change, new children, each puskesmas;
  * sensitivity switches: WHO labels, no status at t, no flagged observations;
  * FREEZE_LOG_v4 written before any test metric; stage (f) refuses to run if
    any frozen artefact changed.

Stages (per horizon h)
    a  environment, seeds, contract
    b  load master v4, re-check eligibility, build t-only predictors, pairs, temporal blocks
    c  six base learners, child-grouped strict OOF -> 18 meta-features
    d  Meta-DNN epoch selection and refit
    e  thresholds (MCE and every base learner) and best-base selection on the
       threshold-tuning data; FREEZE_LOG_v4
    f  frozen test evaluation

Run (Colab):
    pip install lightgbm==4.6.0 xgboost==3.1.2 catboost==1.2.8 pyarrow
    MCE_FP_MASTER_PATH=/content/MCE_FP_v4/data/master_v4_pseudonymised.parquet \
    MCE_FP_BASE_DIR=/content/drive/MyDrive/MCE_FP_v4/runs/primary \
    MCE_FP_HORIZONS=1,3,6 python mce_ilcadt_fp_v4.py

Sensitivity analyses (SAP section 8), each in its own MCE_FP_BASE_DIR:
    MCE_FP_LABEL_SOURCE=who | MCE_FP_ABLATION=no_current_status | MCE_FP_EXCLUDE_FLAGGED=1

Smoke test (MCE_FP_SMOKE=1): small models and shifted blocks that lie entirely
inside the training period (targets 2023-01 .. 2024-12). The real threshold-
tuning data (2025) and the real test data (2026) are never touched.
"""
from __future__ import annotations

import os

RANDOM_STATE = int(os.environ.get("MCE_FP_RANDOM_STATE", "42"))
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import hashlib
import json
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import lightgbm
import xgboost
import catboost
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input

# =============================================================================
# CONFIGURATION
# =============================================================================

EXPERIMENT_NAME = "MCE_ILCADT_FP_V4_BUTON_UTARA"
TIMEZONE = "Asia/Jakarta"
SAP_SHA256 = "3ec9bbe2be7108641f07be531cee25e6eedd8b55d5cde9ddf9f056db592776a6"

MASTER_PATH = Path(os.environ.get("MCE_FP_MASTER_PATH", "/content/MCE_FP_v4/data/master_v4_pseudonymised.parquet")).resolve()
BASE_DIR = Path(os.environ.get("MCE_FP_BASE_DIR", "/content/MCE_FP_v4/runs/primary")).resolve()
SAP_PATH = os.environ.get("MCE_FP_SAP_PATH", "")

HORIZONS = tuple(int(x) for x in os.environ.get("MCE_FP_HORIZONS", "1,3,6").split(",") if x.strip())
RUN_STAGES = tuple(x.strip().lower() for x in os.environ.get("MCE_FP_STAGES", "a,b,c,d,e,f").split(",") if x.strip())
SMOKE_TEST = os.environ.get("MCE_FP_SMOKE", "0") == "1"

# --- Temporal blocks (SAP section 3); boundaries are inclusive calendar months ---
# training          : target month in [TRAIN_TARGET_MIN, TRAIN_TARGET_MAX]
# threshold tuning  : prediction month >= TUNE_ANCHOR_MIN and target month <= TUNE_TARGET_MAX
# test              : prediction month >= TEST_ANCHOR_MIN (and target <= TEST_TARGET_MAX if set)
# Pairs crossing a boundary, and pairs whose target lies in 2022, are not used.
if not SMOKE_TEST:
    BLOCKS = {"TRAIN_TARGET_MIN": "2023-01", "TRAIN_TARGET_MAX": "2024-12", "TUNE_ANCHOR_MIN": "2025-01",
              "TUNE_TARGET_MAX": "2025-12", "TEST_ANCHOR_MIN": "2026-01", "TEST_TARGET_MAX": ""}
else:  # inside the real training period only
    BLOCKS = {"TRAIN_TARGET_MIN": "2023-01", "TRAIN_TARGET_MAX": "2023-12", "TUNE_ANCHOR_MIN": "2024-01",
              "TUNE_TARGET_MAX": "2024-06", "TEST_ANCHOR_MIN": "2024-07", "TEST_TARGET_MAX": "2024-12"}
MIN_CHILDREN_RAREST_CLASS_IN_TUNING = 10

# Pair counts approved in Stage 2 (SAP section 3): training / threshold tuning / test.
EXPECTED_PAIR_COUNTS = {
    1: (90956, 48866, 34969), 2: (86462, 43600, 29937), 3: (82489, 38405, 25071),
    4: (78798, 33505, 20380), 5: (74934, 28701, 15980), 6: (70932, 24157, 11650),
}
# Master v4 content (SAP section 2)
EXPECTED_MASTER = {"rows": 234073, "children": 10321, "eligible": 230551,
                   "labels": {"normal": 190959, "stunted": 31106, "severely stunted": 8486}}

# --- Analysis switches (SAP section 8); each sensitivity analysis changes one thing ---
LABEL_SOURCE = os.environ.get("MCE_FP_LABEL_SOURCE", "eppgbm").lower()      # eppgbm | who
ABLATION = os.environ.get("MCE_FP_ABLATION", "none").lower()                # none | no_current_status
EXCLUDE_FLAGGED = os.environ.get("MCE_FP_EXCLUDE_FLAGGED", "0") == "1"      # height drop > 1.5 cm
ABLATION_DROP = {"none": [], "no_current_status": ["status_t", "haz_t"]}
if LABEL_SOURCE not in ("eppgbm", "who"):
    raise ValueError("MCE_FP_LABEL_SOURCE must be 'eppgbm' or 'who'.")
if ABLATION not in ABLATION_DROP:
    raise ValueError(f"MCE_FP_ABLATION must be one of {sorted(ABLATION_DROP)}")
N_SENSITIVITY_SWITCHES = int(LABEL_SOURCE != "eppgbm") + int(ABLATION != "none") + int(EXCLUDE_FLAGGED)
if N_SENSITIVITY_SWITCHES > 1:
    raise ValueError("A sensitivity analysis changes exactly one element (SAP section 8).")
ANALYSIS = ("primary" if N_SENSITIVITY_SWITCHES == 0 else
            "sens_who_label" if LABEL_SOURCE == "who" else
            "sens_no_current_status" if ABLATION != "none" else "sens_exclude_flagged")

N_SPLITS = 5
META_VALIDATION_CHILD_FRACTION = 0.10
MIN_CHILDREN_PER_CLASS_PER_FOLD = 2

# --- Labels: WHO 2006 HAZ categories as defined in Permenkes RI No. 2/2020 ---
LABELS = ["normal", "stunted", "severely stunted"]
N_CLASSES = len(LABELS)
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
ELIGIBLE_CATEGORIES = ("normal", "stunted", "severely stunted", "tall")  # e-PPGBM category present (not outlier/missing)
HAZ_VALID = (-6.0, 6.0)
MAX_AGE_DAYS = 1856  # 60 completed months plus the days of month 60, as in master v4

WEIGHT_GAIN_CODES = ["N", "T", "O", "B"]
BASE_FEATURES = [
    # static
    "sex_male", "birth_weight_kg", "birth_length_cm",
    # month t
    "age_months_t", "weight_kg_t", "height_cm_t", "recumbent_t", "muac_cm_t", "muac_observed_t",
    "haz_t", "waz_t", "whz_t", "status_t", "wg_N_t", "wg_T_t", "wg_O_t", "wg_B_t",
    # 3-month growth history (v3)
    "haz_lag1_observed", "d_haz_1", "d_waz_1", "d_height_1", "d_weight_1",
    "haz_lag3_observed", "d_haz_3",
    "n_obs_to_t", "months_since_prev_obs", "n_prior_stunted_months", "min_haz_to_t",
    # 12-month growth history (new in v4, SAP 4.2)
    "haz_lag6_observed", "haz_lag12_observed", "d_haz_6", "d_haz_12", "n_stunted_months_last12",
]
FEAT_COLS = [f for f in BASE_FEATURES if f not in ABLATION_DROP[ABLATION]]
STATIC_FEATURES = ("sex_male", "birth_weight_kg", "birth_length_cm")
HISTORY_PREFIXES = ("d_", "n_", "months_", "min_", "haz_lag")

MODEL_ORDER = ["RF", "LGBM", "XGB", "CAT", "GBC", "ADA"]  # also the tie order for best-base selection
META_DIMENSION = len(MODEL_ORDER) * N_CLASSES

# Meta-DNN (v3, SAP 5.3)
MAX_EPOCHS = 200 if not SMOKE_TEST else 8
BATCH_SIZE = 64
INITIAL_LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 12
LR_REDUCER_PATIENCE = 5
LR_REDUCER_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01, dtype=np.float64), 2)
CONFIDENCE_EPSILON = 1e-7
RELIABILITY_BINS = 10
BOOTSTRAP_REPLICATES = int(os.environ.get("MCE_FP_BOOTSTRAP", "2000" if not SMOKE_TEST else "50"))
NUMERIC_TOLERANCE = 1e-6

MAIN_METHODS = ["persistence", "best_base", "best_base_ilcadt", "mce", "mce_ilcadt"]
COMPARISONS = [("persistence", "mce_ilcadt"), ("persistence", "mce"), ("mce", "mce_ilcadt"),
               ("best_base", "best_base_ilcadt"), ("best_base", "mce")]  # (a, b): b minus a

SHARED_DIR = BASE_DIR / "shared"
FREEZE_LOG = SHARED_DIR / "FREEZE_LOG_v4.txt"


def horizon_dir(h: int) -> Path:
    return BASE_DIR / f"h{h}" / "artifacts"


# =============================================================================
# UTILITIES
# =============================================================================


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Not JSON serialisable: {type(value)}")


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
    with path.open("r", encoding="utf-8") as f:
        json.load(f)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def month_index(text: str) -> int:
    period = pd.Period(text, freq="M")
    return int(period.year * 12 + period.month - 1)


def month_text(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def run_config() -> dict[str, Any]:
    return {"analysis": ANALYSIS, "label_source": LABEL_SOURCE, "ablation": ABLATION,
            "exclude_flagged": EXCLUDE_FLAGGED, "smoke_test": SMOKE_TEST, "blocks": BLOCKS,
            "feature_order": FEAT_COLS, "class_order": LABELS, "model_order": MODEL_ORDER}


def pair_fingerprint(child_codes, anchor_month, labels, h: int) -> str:
    """Content fingerprint of a pair table (replaces run IDs)."""
    frame = np.stack([np.asarray(child_codes, np.int64), np.asarray(anchor_month, np.int64),
                      np.asarray(labels, np.int64)], axis=1)
    digest = hashlib.sha256()
    digest.update(f"h={h};config={json.dumps(run_config(), sort_keys=True)}".encode())
    digest.update(np.ascontiguousarray(frame).tobytes())
    return digest.hexdigest()


def validate_probability_matrix(probabilities, expected_rows, expected_classes, name) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.shape != (expected_rows, expected_classes):
        raise ValueError(f"{name}: expected {(expected_rows, expected_classes)}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(matrix < -1e-10) or np.any(matrix > 1.0 + 1e-10):
        raise ValueError(f"{name} contains values outside [0, 1].")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=NUMERIC_TOLERANCE, rtol=NUMERIC_TOLERANCE):
        raise ValueError(f"Rows of {name} do not sum to one.")
    return matrix


def fresh_model(template: Any) -> Any:
    try:
        return clone(template)
    except Exception:
        return deepcopy(template)


def prepare_model_input(array: np.ndarray) -> pd.DataFrame:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(FEAT_COLS):
        raise ValueError("Processed base-model input has an invalid shape.")
    if not np.isfinite(array).all():
        raise ValueError("Processed base-model input is not finite.")
    return pd.DataFrame(array, columns=FEAT_COLS)


def predict_proba_aligned(model: Any, model_input: pd.DataFrame) -> np.ndarray:
    """Probability columns aligned to the fixed class indices 0..K-1."""
    raw = np.asarray(model.predict_proba(model_input), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64).reshape(-1)
    if raw.ndim != 2 or raw.shape[1] != len(classes):
        raise ValueError("predict_proba output is inconsistent with classes_.")
    aligned = np.zeros((len(model_input), N_CLASSES), dtype=np.float64)
    for column, k in enumerate(classes):
        if k not in range(N_CLASSES):
            raise ValueError(f"Unexpected class index {k}")
        aligned[:, k] = raw[:, column]
    sums = aligned.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Non-positive probability row sum.")
    aligned /= sums
    return validate_probability_matrix(aligned, len(model_input), N_CLASSES, "aligned probabilities").astype(np.float32)


def set_global_seed() -> None:
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)


# =============================================================================
# STAGE (a): ENVIRONMENT AND CONTRACT
# =============================================================================


def validate_contract() -> None:
    if len(FEAT_COLS) != len(set(FEAT_COLS)):
        raise ValueError("Feature names must be unique.")
    if LABELS != ["normal", "stunted", "severely stunted"]:
        raise ValueError("Class order changed.")
    b = {k: month_index(v) for k, v in BLOCKS.items() if v}
    if not (b["TRAIN_TARGET_MIN"] <= b["TRAIN_TARGET_MAX"] < b["TUNE_ANCHOR_MIN"]
            and b["TUNE_TARGET_MAX"] < b["TEST_ANCHOR_MIN"]):
        raise ValueError("Temporal blocks must be ordered: training targets < tuning prediction months; "
                         "tuning targets < test prediction months.")
    if any(h < 1 for h in HORIZONS):
        raise ValueError("Horizons must be positive integers (months).")
    if set(HORIZONS) - set(EXPECTED_PAIR_COUNTS):
        raise ValueError("SAP v4 covers h = 1..6 only.")


def run_stage_a() -> None:
    print("\n" + "=" * 96 + f"\nSTAGE (a): ENVIRONMENT, SEEDS, CONTRACT  [{ANALYSIS}{' / SMOKE' if SMOKE_TEST else ''}]\n" + "=" * 96)
    validate_contract()
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError:
        pass
    set_global_seed()
    try:
        tf.config.experimental.enable_op_determinism()
        determinism = "enabled"
    except Exception as exc:  # pragma: no cover
        determinism = f"not fully enabled: {exc}"
    save_json({
        "experiment_name": EXPERIMENT_NAME, "created_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
        "task": "X_{i,t} -> y_{i,t+h}, h in calendar months", "random_state": RANDOM_STATE,
        "horizons_months": list(HORIZONS), **run_config(), "n_features": len(FEAT_COLS),
        "meta_dimension": META_DIMENSION, "sap_sha256_expected": SAP_SHA256,
        "code_sha256": sha256_file(Path(__file__)),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                     "xgboost": xgboost.__version__, "catboost": catboost.__version__, "tensorflow": tf.__version__},
        "tensorflow_determinism": determinism,
    }, SHARED_DIR / f"environment_metadata_h{'-'.join(map(str, HORIZONS))}.json")
    print(f"  Horizons {HORIZONS} | classes {LABELS} | predictors {len(FEAT_COLS)} | meta dim {META_DIMENSION}")
    print("✓ Stage (a) completed.")


# =============================================================================
# STAGE (b1): MASTER V4, ELIGIBILITY, LABELS
# =============================================================================


def who_category(z: pd.Series) -> pd.Series:
    zr = z.round(2)
    return pd.Series(np.select([zr < -3.0, zr < -2.0, zr <= 3.0, zr > 3.0],
                               ["severely stunted", "stunted", "normal", "tall"], default="missing"), index=z.index)


def load_master() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master v4 not found: {MASTER_PATH}")
    master = pd.read_parquet(MASTER_PATH) if MASTER_PATH.suffix == ".parquet" else pd.read_pickle(MASTER_PATH)
    audit: dict[str, Any] = {"master_file": MASTER_PATH.name, "master_sha256": sha256_file(MASTER_PATH)}

    # --- contract checks against the approved master (Stage 1) ------------------
    checks = {
        "rows": len(master), "children": int(master["child_id"].nunique()),
        "eligible": int(master["eligible"].sum()),
        "labels": {k: int(v) for k, v in master.loc[master["eligible"], "label"].value_counts().items()},
    }
    if checks != EXPECTED_MASTER:
        raise ValueError(f"Master v4 content differs from the approved master: {checks}")
    if master.duplicated(["child_id", "month_index"]).any():
        raise ValueError("Duplicate child-month rows in master.")
    for column in ("puskesmas", "village", "sex_male", "birth_weight_kg", "birth_length_cm"):
        if (master.groupby("child_id")[column].nunique(dropna=True) > 1).any():
            raise ValueError(f"Child-level attribute '{column}' is not constant within a child.")
    month = master["month_index"].astype(np.int64) + month_index("2022-01")
    if not (master["calendar_month"].astype(str) == month.map(month_text)).all():
        raise ValueError("calendar_month and month_index disagree.")
    audit["master_checks"] = checks

    # --- eligibility (re-derived from the master columns, SAP section 2) --------
    category = master["tbu_category"].astype(str)
    haz = master["haz"].astype(float)
    base_ok = (category.isin(ELIGIBLE_CATEGORIES) & (master["age_days"] >= 0)
               & (master["age_days"] <= MAX_AGE_DAYS) & master["height_cm"].notna())
    eligible_eppgbm = base_ok & category.isin(LABELS) & haz.between(*HAZ_VALID)
    if not (eligible_eppgbm == master["eligible"].astype(bool)).all():
        raise ValueError("Re-derived e-PPGBM eligibility differs from master v4.")
    if LABEL_SOURCE == "eppgbm":
        label = category.where(eligible_eppgbm)
    else:
        z_who = master["haz_who"].astype(float)
        cat_who = who_category(z_who)
        eligible_who = base_ok & z_who.between(*HAZ_VALID) & cat_who.isin(LABELS)
        label = cat_who.where(eligible_who)
        both = eligible_eppgbm & eligible_who
        audit["who_vs_eppgbm"] = {
            "eligible_eppgbm": int(eligible_eppgbm.sum()), "eligible_who": int(eligible_who.sum()),
            "eligible_both": int(both.sum()),
            "category_agreement_pct_both": float((cat_who[both] == category[both]).mean() * 100)}
    eligible = label.isin(LABELS)
    if EXCLUDE_FLAGGED:
        flagged = master["flag_height_drop_gt1_5cm"].astype(int) == 1
        audit["flagged_observations_removed_from_eligibility"] = int((eligible & flagged).sum())
        eligible &= ~flagged

    # --- observation table with predictors at month t -----------------------------
    obs = pd.DataFrame({"child_id": master["child_id"].astype(str), "month": month,
                        "puskesmas": master["puskesmas"].astype(str)})
    obs["sex_male"] = master["sex_male"].astype(float)
    obs["birth_weight_kg"] = master["birth_weight_kg"].astype(float)
    obs["birth_length_cm"] = master["birth_length_cm"].astype(float)
    # completed months of age, as in v3 (e-PPGBM "Usia Saat Ukur"); master v4 stores age in days
    obs["age_months_t"] = np.floor(master["age_days"].astype(float) / 30.4375)
    obs["weight_kg_t"] = master["weight_kg"].astype(float)
    obs["height_cm_t"] = master["height_cm"].astype(float)
    obs["recumbent_t"] = master["measure_position"].map({"Terlentang": 1.0, "Berdiri": 0.0}).astype(float)
    obs["muac_cm_t"] = master["muac_cm"].astype(float)
    obs["muac_observed_t"] = master["muac_observed"].astype(float)
    obs["haz_t"] = haz.where(haz.between(*HAZ_VALID))  # e-PPGBM HAZ in every analysis (predictor)
    obs["waz_t"] = master["waz"].astype(float)
    obs["whz_t"] = master["whz"].astype(float)
    for code in WEIGHT_GAIN_CODES:
        obs[f"wg_{code}_t"] = (master["weight_gain_code_t"].astype(str) == code).astype(float)
    # status of every observation with a valid label (also used in growth history). With
    # EXCLUDE_FLAGGED, flagged observations keep their status in the history predictors and are
    # only removed as prediction or target months (SAP 8.3: exactly one change).
    obs["label_text"] = label.where(label.isin(LABELS))
    obs["eligible"] = eligible.to_numpy()
    obs["status_t"] = obs["label_text"].map(LABEL_TO_IDX)  # NaN when no valid label
    # salt-independent of the analysis: rank of the frozen pseudonym in master v4
    obs["child_code"] = obs["child_id"].rank(method="dense").astype(np.int64) - 1
    audit["label_source"] = LABEL_SOURCE
    audit["eligible_observations"] = int(eligible.sum())
    audit["eligible_label_counts"] = obs.loc[obs["eligible"], "label_text"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict()
    audit["months"] = [month_text(int(month.min())), month_text(int(month.max()))]
    obs = obs.sort_values(["child_id", "month"]).reset_index(drop=True)
    return obs, audit


# =============================================================================
# STAGE (b2): PREDICTORS AVAILABLE AT t, PAIRS, TEMPORAL BLOCKS
# =============================================================================


def add_history_features(obs: pd.DataFrame) -> pd.DataFrame:
    """Growth history from observations with month <= t only (calendar-month aligned)."""
    work = obs.copy()
    keyed = work.set_index(["child_id", "month"])

    def lagged(column: str, k: int) -> pd.Series:
        index = pd.MultiIndex.from_arrays([work["child_id"], work["month"] - k])
        return pd.Series(keyed[column].reindex(index).to_numpy(), index=work.index)

    haz_lag = {k: lagged("haz_t", k) for k in (1, 3, 6, 12)}
    for k in (1, 3, 6, 12):
        work[f"haz_lag{k}_observed"] = haz_lag[k].notna().astype(float)
        work[f"d_haz_{k}"] = work["haz_t"] - haz_lag[k]
    work["d_waz_1"] = work["waz_t"] - lagged("waz_t", 1)
    work["d_height_1"] = work["height_cm_t"] - lagged("height_cm_t", 1)
    work["d_weight_1"] = work["weight_kg_t"] - lagged("weight_kg_t", 1)

    grouped = work.groupby("child_id", sort=False)
    work["n_obs_to_t"] = grouped.cumcount().astype(float) + 1.0
    work["months_since_prev_obs"] = grouped["month"].diff()
    stunted = work["status_t"].isin([1, 2]).astype(float)
    work["n_prior_stunted_months"] = stunted.groupby(work["child_id"]).cumsum() - stunted
    work["min_haz_to_t"] = grouped["haz_t"].cummin()
    keyed_stunted = pd.Series(stunted.to_numpy(), index=pd.MultiIndex.from_arrays([work["child_id"], work["month"]]))
    last12 = np.zeros(len(work))
    for k in range(1, 13):
        index = pd.MultiIndex.from_arrays([work["child_id"], work["month"] - k])
        last12 += np.nan_to_num(keyed_stunted.reindex(index).to_numpy(), nan=0.0)
    work["n_stunted_months_last12"] = last12
    return work


def build_pairs(obs: pd.DataFrame, h: int) -> pd.DataFrame:
    anchors = obs.loc[obs["eligible"]].copy()
    targets = obs.loc[obs["eligible"], ["child_id", "month", "status_t", "haz_t"]].rename(
        columns={"month": "target_month", "status_t": "y", "haz_t": "target_haz_audit_only"})
    anchors["target_month"] = anchors["month"] + h
    pairs = anchors.merge(targets, on=["child_id", "target_month"], how="inner", validate="one_to_one")
    pairs = pairs.rename(columns={"month": "anchor_month"})
    pairs["y"] = pairs["y"].astype(np.int64)
    pairs["status_t"] = pairs["status_t"].astype(np.int64)
    pairs["h"] = h
    if (pairs["target_month"] - pairs["anchor_month"] != h).any():
        raise RuntimeError("Pair horizon mismatch.")
    return pairs.sort_values(["child_id", "anchor_month"]).reset_index(drop=True)


def child_strata(pairs: pd.DataFrame) -> pd.Series:
    """Child-level stratum = most severe target class of that child."""
    return pairs.groupby("child_id")["y"].max()


def run_stage_b() -> None:
    print("\n" + "=" * 96 + "\nSTAGE (b): MASTER V4, PREDICTORS, PAIRS, TEMPORAL BLOCKS\n" + "=" * 96)
    validate_contract()
    obs, audit = load_master()
    obs = add_history_features(obs)
    save_json(audit, SHARED_DIR / "dataset_audit.json")
    pd.DataFrame({
        "feature": FEAT_COLS,
        "information_time": ["birth (static)" if f in STATIC_FEATURES else
                             "history <= t" if f.startswith(HISTORY_PREFIXES) else "month t" for f in FEAT_COLS],
        "uses_information_after_t": False,
        "missing_fraction_all_eligible_obs": [float(obs.loc[obs["eligible"], f].isna().mean()) for f in FEAT_COLS],
    }).to_csv(SHARED_DIR / "feature_availability_audit.csv", index=False)

    b = {k: month_index(v) if v else None for k, v in BLOCKS.items()}
    codes = obs.drop_duplicates("child_id").set_index("child_id")["child_code"]
    attrition_rows, availability_rows = [], []
    for h in HORIZONS:
        out = horizon_dir(h)
        out.mkdir(parents=True, exist_ok=True)
        pairs = build_pairs(obs, h)

        anchors = obs.loc[obs["eligible"], ["child_id", "month", "status_t", "haz_t"]]
        has_target = pd.MultiIndex.from_arrays([anchors["child_id"], anchors["month"]]).isin(
            pd.MultiIndex.from_arrays([pairs["child_id"], pairs["anchor_month"]]))
        for flag, frame in ((True, anchors[has_target]), (False, anchors[~has_target])):
            share = frame["status_t"].value_counts(normalize=True).reindex(range(N_CLASSES)).fillna(0)
            attrition_rows.append({"h": h, "target_observed": flag, "prediction_months": int(len(frame)),
                                   "mean_haz_t": float(frame["haz_t"].mean()),
                                   **{f"pct_status_t_{LABELS[k]}": float(share[k] * 100) for k in range(N_CLASSES)}})

        dev = pairs.loc[(pairs["target_month"] >= b["TRAIN_TARGET_MIN"]) & (pairs["target_month"] <= b["TRAIN_TARGET_MAX"])].reset_index(drop=True)
        val = pairs.loc[(pairs["anchor_month"] >= b["TUNE_ANCHOR_MIN"]) & (pairs["target_month"] <= b["TUNE_TARGET_MAX"])].reset_index(drop=True)
        test_mask = pairs["anchor_month"] >= b["TEST_ANCHOR_MIN"]
        if b["TEST_TARGET_MAX"] is not None:
            test_mask &= pairs["target_month"] <= b["TEST_TARGET_MAX"]
        test = pairs.loc[test_mask].reset_index(drop=True)
        target_2022 = int((pairs["target_month"] < month_index("2023-01")).sum())
        crossing = len(pairs) - len(dev) - len(val) - len(test) - target_2022

        # --- leakage and contract checks ------------------------------------------
        if len(test) == 0 or len(val) == 0:
            raise RuntimeError(f"h={h}: empty threshold-tuning or test block.")
        if not SMOKE_TEST and (len(dev), len(val), len(test)) != EXPECTED_PAIR_COUNTS[h] and not EXCLUDE_FLAGGED and LABEL_SOURCE == "eppgbm":
            raise ValueError(f"h={h}: pair counts {(len(dev), len(val), len(test))} differ from Stage 2 {EXPECTED_PAIR_COUNTS[h]}.")
        if val["anchor_month"].min() <= dev["target_month"].max():
            raise RuntimeError("Temporal leakage: a tuning prediction month precedes a training target month.")
        if test["anchor_month"].min() <= max(dev["target_month"].max(), val["target_month"].max()):
            raise RuntimeError("Temporal leakage: a test prediction month precedes a training/tuning target month.")
        keys = lambda f: set(zip(f["child_id"], f["anchor_month"]))
        if keys(dev) & keys(val) or keys(dev) & keys(test) or keys(val) & keys(test):
            raise RuntimeError("A pair appears in more than one block.")
        val_counts = np.bincount(val["y"], minlength=N_CLASSES)
        rarest_children = int(val.loc[val["y"] == N_CLASSES - 1, "child_id"].nunique())
        if np.any(val_counts == 0) or rarest_children < MIN_CHILDREN_RAREST_CLASS_IN_TUNING:
            raise ValueError(f"h={h}: threshold-tuning block too small {val_counts.tolist()}, {rarest_children} children.")
        dev_counts = np.bincount(dev["y"], minlength=N_CLASSES)
        if np.any(dev_counts == 0):
            raise ValueError(f"h={h}: training block lacks a class {dev_counts.tolist()}")
        dev_children = set(dev["child_id"])
        test["new_child"] = ~test["child_id"].isin(dev_children)

        # --- child-grouped OOF folds, stratified by child stratum --------------------
        strata = child_strata(dev)
        strata = strata.loc[sorted(strata.index, key=codes.__getitem__)]  # order independent of hash values
        per_stratum = strata.value_counts().reindex(range(N_CLASSES)).fillna(0).astype(int)
        if per_stratum.min() < N_SPLITS * MIN_CHILDREN_PER_CLASS_PER_FOLD:
            raise ValueError(f"h={h}: too few children in the rarest stratum {per_stratum.tolist()}")
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        fold_of_child = pd.Series(0, index=strata.index, dtype=np.int64)
        for fold, (_, idx) in enumerate(skf.split(strata.index, strata.to_numpy()), start=1):
            fold_of_child.iloc[idx] = fold
        dev["oof_fold"] = dev["child_id"].map(fold_of_child).astype(np.int64)
        for fold in range(1, N_SPLITS + 1):
            tr_counts = np.bincount(dev.loc[dev["oof_fold"] != fold, "y"], minlength=N_CLASSES)
            va_children = dev.loc[(dev["oof_fold"] == fold) & (dev["y"] == N_CLASSES - 1), "child_id"].nunique()
            if np.any(tr_counts == 0) or va_children < MIN_CHILDREN_PER_CLASS_PER_FOLD:
                raise ValueError(f"h={h} fold {fold}: class coverage too small.")

        # --- child-grouped meta subsets inside training (epoch selection) ------------
        meta_tr, meta_va = train_test_split(strata.index.to_numpy(), test_size=META_VALIDATION_CHILD_FRACTION,
                                            stratify=strata.to_numpy(), random_state=RANDOM_STATE + 1)
        subset = pd.Series("meta_train", index=strata.index)
        subset.loc[meta_va] = "meta_validation"
        dev["meta_subset"] = dev["child_id"].map(subset)
        for name in ("meta_train", "meta_validation"):
            counts = np.bincount(dev.loc[dev["meta_subset"] == name, "y"], minlength=N_CLASSES)
            if np.any(counts == 0):
                raise ValueError(f"h={h}: {name} lacks a class {counts.tolist()}")

        for frame in (dev, val, test):
            frame["child_code"] = frame["child_id"].map(codes).astype(np.int64)
        fp = {name: pair_fingerprint(f["child_code"], f["anchor_month"], f["y"], h)
              for name, f in (("dev", dev), ("val", val), ("test", test))}
        keep = (["child_id", "child_code", "puskesmas", "anchor_month", "target_month", "h", "y", "target_haz_audit_only"]
                + FEAT_COLS + (["status_t"] if "status_t" not in FEAT_COLS else []))
        dev[keep + ["oof_fold", "meta_subset"]].to_pickle(out / "dev_pairs.pkl")
        val[keep].to_pickle(out / "tune_pairs.pkl")
        test[keep + ["new_child"]].to_pickle(out / "test_pairs.pkl")

        # 12-month history availability. "pairs": prediction pairs with a valid HAZ at t-12 (what the
        # model sees). "observations": all eligible observations in the block's prediction-month window
        # with an eligible observation at t-12 (definition behind the SAP section 10 figure).
        eligible_keys = pd.MultiIndex.from_arrays([obs.loc[obs["eligible"], "child_id"], obs.loc[obs["eligible"], "month"]])
        last_month = int(obs["month"].max())
        windows = {"training": (b["TRAIN_TARGET_MIN"] - h, b["TRAIN_TARGET_MAX"] - h),
                   "threshold_tuning": (b["TUNE_ANCHOR_MIN"], b["TUNE_TARGET_MAX"] - h),
                   "test": (b["TEST_ANCHOR_MIN"], (b["TEST_TARGET_MAX"] if b["TEST_TARGET_MAX"] is not None else last_month) - h)}
        for name, frame in (("training", dev), ("threshold_tuning", val), ("test", test)):
            for lag in (6, 12):
                availability_rows.append({"h": h, "block": name, "lag_months": lag, "unit": "pairs",
                                          "denominator": int(len(frame)), "share": float(frame[f"haz_lag{lag}_observed"].mean())})
                lo, hi = windows[name]
                window = obs.loc[obs["eligible"] & obs["month"].between(lo, hi), ["child_id", "month"]]
                found = pd.MultiIndex.from_arrays([window["child_id"], window["month"] - lag]).isin(eligible_keys)
                availability_rows.append({"h": h, "block": name, "lag_months": lag, "unit": "observations",
                                          "denominator": int(len(window)), "share": float(found.mean())})

        majority = int(np.argmax(dev_counts))
        minority = [k for k in range(N_CLASSES) if k != majority]
        rows = []
        for name, frame in (("training", dev), ("threshold_tuning", val), ("test", test),
                            ("meta_train", dev[dev["meta_subset"] == "meta_train"]),
                            ("meta_validation", dev[dev["meta_subset"] == "meta_validation"])):
            counts = np.bincount(frame["y"], minlength=N_CLASSES)
            for k, label in enumerate(LABELS):
                rows.append({"block": name, "class": label, "pairs": int(counts[k]),
                             "percent": float(counts[k] / len(frame) * 100), "children": int(frame["child_id"].nunique()),
                             "children_with_class": int(frame.loc[frame["y"] == k, "child_id"].nunique())})
        pd.DataFrame(rows).to_csv(out / "class_distribution_by_block.csv", index=False)

        span = lambda f, c: [month_text(int(f[c].min())), month_text(int(f[c].max()))]
        save_json({
            "h": h, "status": "ok", **run_config(), "fingerprint_dev": fp["dev"], "fingerprint_val": fp["val"],
            "fingerprint_test": fp["test"], "master_sha256": audit["master_sha256"],
            "n_pairs_all": int(len(pairs)), "n_pairs_target_in_2022": target_2022,
            "n_pairs_crossing_block_boundaries": int(crossing),
            "n_dev_pairs": int(len(dev)), "n_dev_children": int(dev["child_id"].nunique()),
            "n_val_pairs": int(len(val)), "n_val_children": int(val["child_id"].nunique()),
            "n_test_pairs": int(len(test)), "n_test_children": int(test["child_id"].nunique()),
            "dev_prediction_months": span(dev, "anchor_month"), "dev_target_months": span(dev, "target_month"),
            "val_prediction_months": span(val, "anchor_month"), "val_target_months": span(val, "target_month"),
            "test_prediction_months": span(test, "anchor_month"), "test_target_months": span(test, "target_month"),
            "test_children_also_in_training": int(test.loc[~test["new_child"], "child_id"].nunique()),
            "test_pairs_new_children": int(test["new_child"].sum()),
            "children_per_stratum_dev": per_stratum.tolist(),
            "majority_class_from_dev": LABELS[majority], "minority_classes_from_dev": [LABELS[k] for k in minority],
            "imbalance_ratio_majority_to_rarest_dev": float(dev_counts.max() / dev_counts.min()),
            "grouping": "child (pseudonymised); all pairs of a child share one OOF fold and one meta subset",
        }, out / "stage_b_manifest.json")
        print(f"  h={h}: training={len(dev)} / {dev['child_id'].nunique()} ch | tuning={len(val)} | "
              f"test={len(test)} / {test['child_id'].nunique()} ch | crossing={crossing} | target 2022={target_2022}")
    tag = "h" + "-".join(map(str, HORIZONS))
    pd.DataFrame(attrition_rows).to_csv(SHARED_DIR / f"attrition_audit_{tag}.csv", index=False)
    pd.DataFrame(availability_rows).to_csv(SHARED_DIR / f"history_availability_{tag}.csv", index=False)
    print("✓ Stage (b) completed.")


def load_stage_b(h: int):
    out = horizon_dir(h)
    manifest = load_json(out / "stage_b_manifest.json")
    if manifest.get("status") != "ok":
        raise RuntimeError(f"h={h}: stage (b) status is {manifest.get('status')}")
    for key, value in run_config().items():
        if manifest[key] != value:
            raise ValueError(f"Configuration '{key}' changed since stage (b): {manifest[key]} -> {value}")
    dev = pd.read_pickle(out / "dev_pairs.pkl")
    val = pd.read_pickle(out / "tune_pairs.pkl")
    test = pd.read_pickle(out / "test_pairs.pkl")
    for name, frame in (("dev", dev), ("val", val), ("test", test)):
        if pair_fingerprint(frame["child_code"], frame["anchor_month"], frame["y"], h) != manifest[f"fingerprint_{name}"]:
            raise ValueError(f"{name} pair table does not match its stage-(b) fingerprint.")
    return dev, val, test, manifest


# =============================================================================
# STAGE (c): BASE LEARNERS AND CHILD-GROUPED STRICT OOF
# =============================================================================


def fit_transform_foldwise(train_frame: pd.DataFrame, apply_frames: list[pd.DataFrame]):
    """Median imputation and standardisation fitted on the training fold only."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train_frame[FEAT_COLS]))
    applied = [scaler.transform(imputer.transform(f[FEAT_COLS])) for f in apply_frames]
    for array in [x_train, *applied]:
        if not np.isfinite(array).all():
            raise ValueError("Preprocessing produced non-finite values.")
    return x_train, applied, {"imputer": imputer, "scaler": scaler, "feature_columns": FEAT_COLS.copy(),
                              "fit_scope": "training fold only"}


def define_base_models() -> dict[str, Any]:
    scale = 0.1 if SMOKE_TEST else 1.0
    n = lambda value: max(10, int(value * scale))
    models = {
        "RF": RandomForestClassifier(n_estimators=n(400), max_depth=15, min_samples_split=10, min_samples_leaf=4,
                                     random_state=RANDOM_STATE, n_jobs=-1),
        "LGBM": LGBMClassifier(n_estimators=n(600), learning_rate=0.05, num_leaves=31, min_child_samples=30,
                               subsample=0.9, subsample_freq=1, colsample_bytree=0.9, random_state=RANDOM_STATE,
                               n_jobs=-1, verbosity=-1, deterministic=True, force_col_wise=True),
        "XGB": XGBClassifier(n_estimators=n(600), learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9,
                             objective="multi:softprob", num_class=N_CLASSES, eval_metric="mlogloss",
                             random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"),
        "CAT": CatBoostClassifier(iterations=n(800), learning_rate=0.05, depth=6, loss_function="MultiClass",
                                  random_seed=RANDOM_STATE, verbose=False, thread_count=-1, allow_writing_files=False),
        "GBC": GradientBoostingClassifier(n_estimators=n(100), learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
        "ADA": AdaBoostClassifier(n_estimators=n(300), learning_rate=0.5, random_state=RANDOM_STATE),
    }
    if list(models) != MODEL_ORDER:
        raise RuntimeError("Base-model order differs from the contract.")
    return models


def macro_f1_present(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.nanmean(classification_metrics(y_true, y_pred)["f1_per_class"]))


def run_stage_c(h: int) -> None:
    print("\n" + "=" * 96 + f"\nSTAGE (c) h={h}: BASE LEARNERS AND CHILD-GROUPED STRICT OOF\n" + "=" * 96)
    dev, val, test, manifest = load_stage_b(h)
    out = horizon_dir(h)
    y_dev = dev["y"].to_numpy(np.int64)
    folds = dev["oof_fold"].to_numpy(np.int64)
    children = dev["child_code"].to_numpy(np.int64)

    fold_cache = []
    for fold in range(1, N_SPLITS + 1):
        tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if np.intersect1d(children[tr], children[va]).size:
            raise RuntimeError(f"Fold {fold}: a child appears in training and validation.")
        x_tr, (x_va, x_tv, x_te), prep = fit_transform_foldwise(dev.iloc[tr], [dev.iloc[va], val, test])
        fold_cache.append({"fold": fold, "tr": tr, "va": va, "x_tr": x_tr, "x_va": x_va, "x_tv": x_tv, "x_te": x_te,
                           "prep": prep, "val_counts": np.bincount(y_dev[va], minlength=N_CLASSES).tolist()})

    oof = np.full((len(dev), META_DIMENSION), np.nan, dtype=np.float32)
    meta_val = np.zeros((len(val), META_DIMENSION), dtype=np.float64)
    meta_test = np.zeros((len(test), META_DIMENSION), dtype=np.float64)
    models = define_base_models()
    cv_rows = []
    bundle_dir = out / "base_fold_bundles"
    bundle_dir.mkdir(exist_ok=True)
    for b, name in enumerate(MODEL_ORDER):
        block = slice(b * N_CLASSES, (b + 1) * N_CLASSES)
        print(f"  {name}", flush=True)
        for fc in fold_cache:
            model = fresh_model(models[name])
            model.fit(prepare_model_input(fc["x_tr"]), y_dev[fc["tr"]])
            p_va = predict_proba_aligned(model, prepare_model_input(fc["x_va"]))
            oof[fc["va"], block] = p_va
            meta_val[:, block] += predict_proba_aligned(model, prepare_model_input(fc["x_tv"])) / N_SPLITS
            meta_test[:, block] += predict_proba_aligned(model, prepare_model_input(fc["x_te"])) / N_SPLITS
            cv_rows.append({"model": name, "fold": fc["fold"], "validation_class_counts": fc["val_counts"],
                            "oof_macro_f1_classes_present": macro_f1_present(y_dev[fc["va"]], p_va.argmax(1))})
            joblib.dump({"model": model, "preprocessing": fc["prep"], "fold": fc["fold"], "model_name": name},
                        bundle_dir / f"{name}_fold_{fc['fold']}.joblib", compress=3)
        meta_val[:, block] /= meta_val[:, block].sum(axis=1, keepdims=True)
        meta_test[:, block] /= meta_test[:, block].sum(axis=1, keepdims=True)
        validate_probability_matrix(oof[:, block], len(dev), N_CLASSES, f"{name} OOF")
        validate_probability_matrix(meta_val[:, block], len(val), N_CLASSES, f"{name} tuning")
        validate_probability_matrix(meta_test[:, block], len(test), N_CLASSES, f"{name} test")

    np.save(out / "oof_meta_dev.npy", oof)
    np.save(out / "meta_tune.npy", meta_val.astype(np.float32))
    np.save(out / "meta_test.npy", meta_test.astype(np.float32))
    pd.DataFrame(cv_rows).to_csv(out / "base_learners_oof_fold_metrics.csv", index=False)
    pd.DataFrame([{"meta_column_index": b * N_CLASSES + k, "base_model": m, "class_label": c}
                  for b, m in enumerate(MODEL_ORDER) for k, c in enumerate(LABELS)]).to_csv(out / "meta_feature_schema.csv", index=False)
    save_json({"h": h, "fingerprint_dev": manifest["fingerprint_dev"], "fingerprint_val": manifest["fingerprint_val"],
               "fingerprint_test": manifest["fingerprint_test"], "oof_shape": list(oof.shape),
               "meta_tune_shape": list(meta_val.shape), "meta_test_shape": list(meta_test.shape),
               "meta_dimension": META_DIMENSION, "oof_grouping": "child",
               "tuning_and_test_meta_features": "mean of the 5 fold models (no refit on all training data)",
               "tuning_labels_used": False, "test_labels_used": False}, out / "stage_c_manifest.json")
    print(f"✓ Stage (c) h={h}: OOF {oof.shape}, tuning {meta_val.shape}, test {meta_test.shape}")


def load_meta(h: int):
    dev, val, test, manifest = load_stage_b(h)
    out = horizon_dir(h)
    c_manifest = load_json(out / "stage_c_manifest.json")
    if any(c_manifest[f"fingerprint_{k}"] != manifest[f"fingerprint_{k}"] for k in ("dev", "val", "test")):
        raise ValueError("Stage-(c) artefacts belong to a different pair table.")
    oof = np.load(out / "oof_meta_dev.npy").astype(np.float32)
    meta_val = np.load(out / "meta_tune.npy").astype(np.float32)
    meta_test = np.load(out / "meta_test.npy").astype(np.float32)
    if (oof.shape != (len(dev), META_DIMENSION) or meta_val.shape != (len(val), META_DIMENSION)
            or meta_test.shape != (len(test), META_DIMENSION)):
        raise ValueError("Meta-feature shapes are inconsistent with the pair tables.")
    return dev, val, test, manifest, oof, meta_val, meta_test


def base_block(meta: np.ndarray, b: int) -> np.ndarray:
    block = np.asarray(meta[:, b * N_CLASSES:(b + 1) * N_CLASSES], dtype=np.float64)
    return block / block.sum(axis=1, keepdims=True)


# =============================================================================
# STAGE (d): META-DNN
# =============================================================================


def build_meta_dnn(input_dimension: int, number_of_classes: int) -> Sequential:
    model = Sequential([
        Input(shape=(input_dimension,), name="meta_input"),
        Dense(64, activation="relu", name="dense_64"),
        Dropout(0.30, name="dropout_1"),
        Dense(32, activation="relu", name="dense_32"),
        Dropout(0.20, name="dropout_2"),
        Dense(number_of_classes, activation="softmax", name="class_output"),
    ], name="Meta_Confidence_DNN")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


class ValidationTracker(Callback):
    """Selects the epoch with the lowest meta-validation loss (ties: higher macro-F1)."""

    def __init__(self, validation_data):
        super().__init__()
        self.x, self.y = validation_data
        self.values: list[float] = []
        self.best_epoch, self.best_macro_f1, self.best_val_loss = 0, -np.inf, np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        macro_f1 = macro_f1_present(self.y, np.argmax(self.model.predict(self.x, verbose=0), axis=1))
        val_loss = float(logs.get("val_loss", np.inf))
        self.values.append(macro_f1)
        better = val_loss < self.best_val_loss - 1e-12 or (
            abs(val_loss - self.best_val_loss) <= 1e-12 and macro_f1 > self.best_macro_f1 + 1e-12)
        if better:
            self.best_epoch, self.best_macro_f1, self.best_val_loss = epoch + 1, macro_f1, val_loss


class LearningRateRecorder(Callback):
    def __init__(self):
        super().__init__()
        self.learning_rates: list[float] = []

    def on_epoch_begin(self, epoch, logs=None):
        self.learning_rates.append(float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)))


class LearningRateReplay(Callback):
    def __init__(self, schedule):
        super().__init__()
        self.schedule = np.asarray(schedule, dtype=np.float64)

    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.learning_rate.assign(float(self.schedule[epoch]))


def run_stage_d(h: int) -> None:
    print("\n" + "=" * 96 + f"\nSTAGE (d) h={h}: META-DNN EPOCH SELECTION AND REFIT\n" + "=" * 96)
    dev, val, test, manifest, oof, meta_val, meta_test = load_meta(h)
    out = horizon_dir(h)
    y = dev["y"].to_numpy(np.int64)
    subset = dev["meta_subset"].to_numpy()
    tr, va = np.flatnonzero(subset == "meta_train"), np.flatnonzero(subset == "meta_validation")
    if len(tr) + len(va) != len(dev):
        raise RuntimeError("Every training pair must belong to meta_train or meta_validation.")
    codes = dev["child_code"].to_numpy()
    if np.intersect1d(codes[tr], codes[va]).size:
        raise RuntimeError("Children overlap between meta_train and meta_validation.")

    tf.keras.backend.clear_session(); set_global_seed()
    model = build_meta_dnn(META_DIMENSION, N_CLASSES)
    tracker, recorder = ValidationTracker((oof[va], y[va])), LearningRateRecorder()
    history = model.fit(
        oof[tr], y[tr], validation_data=(oof[va], y[va]), epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
        callbacks=[tracker, recorder,
                   ReduceLROnPlateau(monitor="val_loss", mode="min", factor=LR_REDUCER_FACTOR,
                                     patience=LR_REDUCER_PATIENCE, min_lr=MIN_LEARNING_RATE),
                   EarlyStopping(monitor="val_loss", mode="min", patience=EARLY_STOPPING_PATIENCE,
                                 min_delta=1e-6, restore_best_weights=False)],
        verbose=0 if SMOKE_TEST else 2)
    epoch = int(tracker.best_epoch)
    if epoch < 1:
        raise RuntimeError("Epoch selection failed.")
    schedule = np.asarray(recorder.learning_rates[:epoch])
    frame = pd.DataFrame(history.history); frame["epoch"] = np.arange(1, len(frame) + 1)
    frame["val_macro_f1"] = tracker.values; frame["learning_rate"] = recorder.learning_rates
    frame.to_csv(out / "meta_dnn_epoch_selection_history.csv", index=False)

    tf.keras.backend.clear_session(); set_global_seed()
    final = build_meta_dnn(META_DIMENSION, N_CLASSES)
    final.fit(oof, y, epochs=epoch, batch_size=BATCH_SIZE, shuffle=True, callbacks=[LearningRateReplay(schedule)], verbose=0)
    final.save(out / "meta_dnn.keras")
    p_val = validate_probability_matrix(final.predict(meta_val, verbose=0), len(val), N_CLASSES, "Meta-DNN tuning")
    p_test = validate_probability_matrix(final.predict(meta_test, verbose=0), len(test), N_CLASSES, "Meta-DNN test")
    reloaded = tf.keras.models.load_model(out / "meta_dnn.keras", compile=False)
    if not np.allclose(reloaded.predict(meta_test, verbose=0), p_test, atol=NUMERIC_TOLERANCE):
        raise RuntimeError("Meta-DNN reload does not reproduce test probabilities.")
    np.save(out / "meta_dnn_tune_probabilities.npy", p_val.astype(np.float32))
    np.save(out / "meta_dnn_test_probabilities.npy", p_test.astype(np.float32))
    save_json({"h": h, "fingerprint_dev": manifest["fingerprint_dev"], "fingerprint_test": manifest["fingerprint_test"],
               "selected_epoch": epoch, "selection_metric": "meta-validation loss (sparse categorical cross-entropy)",
               "tie_breaker": "higher meta-validation macro-F1", "refit_data": "all training pairs (meta_train + meta_validation)",
               "learning_rate_schedule_replayed": True, "tuning_labels_used": False, "test_labels_used": False,
               "sizes": {"meta_train": int(len(tr)), "meta_validation": int(len(va))}}, out / "stage_d_manifest.json")
    print(f"✓ Stage (d) h={h}: selected epoch {epoch}")


# =============================================================================
# STAGE (e): THRESHOLDS, BEST BASE LEARNER, FREEZE
# =============================================================================


def threshold_adjusted_scores(confidences: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """s_k = sigma(logit p_k - logit tau_k) / sum_j sigma(logit p_j - logit tau_j). Decision scores, not probabilities."""
    p = np.clip(np.asarray(confidences, np.float64), CONFIDENCE_EPSILON, 1 - CONFIDENCE_EPSILON)
    tau = np.clip(np.asarray(thresholds, np.float64), CONFIDENCE_EPSILON, 1 - CONFIDENCE_EPSILON)
    shifted = np.clip((np.log(p) - np.log1p(-p)) - (np.log(tau) - np.log1p(-tau)).reshape(1, -1), -60, 60)
    raw = 1.0 / (1.0 + np.exp(-shifted))
    return raw / raw.sum(axis=1, keepdims=True)


def apply_threshold_decision(confidences, thresholds):
    scores = threshold_adjusted_scores(confidences, thresholds)
    return scores, np.argmax(scores, axis=1).astype(np.int64)  # ties -> lowest class index


def binary_threshold_metrics(target, scores, threshold):
    pred = (scores >= threshold).astype(np.int64)
    tp = int(np.sum((target == 1) & (pred == 1))); fp = int(np.sum((target == 0) & (pred == 1)))
    fn = int(np.sum((target == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def learn_independent_thresholds(labels: np.ndarray, confidences: np.ndarray):
    thresholds, rows = np.zeros(N_CLASSES), []
    for k, label in enumerate(LABELS):
        target = (labels == k).astype(np.int64)
        if target.sum() == 0:
            raise ValueError(f"Threshold-tuning data has no {label} cases.")
        candidates = [binary_threshold_metrics(target, confidences[:, k], t) for t in THRESHOLD_GRID]
        best = max(candidates, key=lambda r: (r["f1"], r["recall"], r["precision"], -abs(r["threshold"] - 0.5), -r["threshold"]))
        thresholds[k] = best["threshold"]
        rows.append({"class": label, "positive_support": int(target.sum()), **best})
    return thresholds, pd.DataFrame(rows)


def frozen_files(h: int) -> list[Path]:
    out = horizon_dir(h)
    files = [out / n for n in ("stage_b_manifest.json", "dev_pairs.pkl", "tune_pairs.pkl", "test_pairs.pkl",
                               "oof_meta_dev.npy", "meta_tune.npy", "meta_test.npy", "meta_dnn.keras",
                               "meta_dnn_tune_probabilities.npy", "meta_dnn_test_probabilities.npy",
                               "class_specific_thresholds.npy", "base_learner_thresholds.csv", "best_base_learner.json")]
    return files + sorted((out / "base_fold_bundles").glob("*.joblib"))


def run_stage_e(h: int) -> None:
    if not SMOKE_TEST and not SAP_PATH:
        raise RuntimeError("Set MCE_FP_SAP_PATH to SAP_v4.md: the freeze must record the SAP hash.")
    if (horizon_dir(h) / "stage_f_manifest.json").exists():
        raise RuntimeError(f"h={h}: the test data were already evaluated; re-freezing is not allowed (SAP 11.3).")
    print("\n" + "=" * 96 + f"\nSTAGE (e) h={h}: THRESHOLDS AND BEST BASE LEARNER (TUNING DATA) + FREEZE\n" + "=" * 96)
    dev, val, test, manifest, oof, meta_val, meta_test = load_meta(h)
    out = horizon_dir(h)
    d_manifest = load_json(out / "stage_d_manifest.json")
    if d_manifest["fingerprint_dev"] != manifest["fingerprint_dev"]:
        raise ValueError("Stage-(d) artefacts belong to a different pair table.")
    y_val = val["y"].to_numpy(np.int64)
    model = tf.keras.models.load_model(out / "meta_dnn.keras", compile=False)
    p_val = validate_probability_matrix(model.predict(meta_val, verbose=0), len(val), N_CLASSES, "Meta-DNN tuning")
    if not np.allclose(p_val, np.load(out / "meta_dnn_tune_probabilities.npy"), atol=NUMERIC_TOLERANCE):
        raise RuntimeError("Meta-DNN tuning probabilities are not reproducible.")
    thresholds, summary = learn_independent_thresholds(y_val, p_val)
    np.save(out / "class_specific_thresholds.npy", thresholds)
    summary.to_csv(out / "class_specific_thresholds.csv", index=False)

    # every base learner: argmax macro-F1 on tuning data and its own thresholds
    base_rows, threshold_rows = [], []
    for b, name in enumerate(MODEL_ORDER):
        p = base_block(meta_val, b)
        tau, table = learn_independent_thresholds(y_val, p)
        base_rows.append({"model": name, "order": b, "tuning_macro_f1_argmax": macro_f1_present(y_val, p.argmax(1)),
                          "tuning_macro_f1_ilcadt": macro_f1_present(y_val, apply_threshold_decision(p, tau)[1])})
        threshold_rows += [{"model": name, **r} for r in table.to_dict("records")]
    base_table = pd.DataFrame(base_rows)
    best = base_table.sort_values(["tuning_macro_f1_argmax", "order"], ascending=[False, True]).iloc[0]
    base_table.to_csv(out / "base_learner_tuning_selection.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(out / "base_learner_thresholds.csv", index=False)
    save_json({"h": h, "best_base_learner": best["model"], "criterion": "macro-F1 (argmax) on threshold-tuning data",
               "tie_order": MODEL_ORDER, "tuning_macro_f1_argmax": float(best["tuning_macro_f1_argmax"]),
               "selected_before_test_evaluation": True}, out / "best_base_learner.json")
    save_json({"h": h, "fingerprint_dev": manifest["fingerprint_dev"], "fingerprint_val": manifest["fingerprint_val"],
               "mce_thresholds": dict(zip(LABELS, thresholds.tolist())), "best_base_learner": best["model"],
               "objective": "one-vs-rest F1 per class", "grid": [0.01, 0.99, 0.01],
               "tie_break": ["f1", "recall", "precision", "closest to 0.5", "lower threshold"],
               "decision": "argmax_k sigma(logit p_k - logit tau_k); always exactly one class",
               "threshold_source": "threshold-tuning block (fold-mean meta-features, same form as test)",
               "threshold_tuning_pairs": int(len(val)), "threshold_tuning_children": int(val["child_id"].nunique()),
               "test_labels_used": False}, out / "stage_e_manifest.json")

    # --- FREEZE (SAP 11.2): hash everything the test evaluation will use ----------
    entry = {"h": h, "frozen_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds"),
             **{k: v for k, v in run_config().items() if k != "feature_order"},
             "code_sha256": sha256_file(Path(__file__)), "master_sha256": manifest["master_sha256"],
             "sap_sha256_expected": SAP_SHA256,
             "sap_sha256_found": sha256_file(Path(SAP_PATH)) if SAP_PATH else "not supplied",
             "mce_thresholds": dict(zip(LABELS, thresholds.tolist())), "best_base_learner": best["model"],
             "selected_epoch": d_manifest["selected_epoch"],
             "files_sha256": {str(p.relative_to(out)): sha256_file(p) for p in frozen_files(h)}}
    if SAP_PATH and entry["sap_sha256_found"] != SAP_SHA256:
        raise ValueError("SAP_v4.md differs from the frozen SAP.")
    save_json(entry, out / "freeze_manifest.json")
    FREEZE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FREEZE_LOG.open("a", encoding="utf-8") as log:
        log.write(f"{entry['frozen_at']} | {ANALYSIS} | h={h} | code {entry['code_sha256'][:16]} | master "
                  f"{entry['master_sha256'][:16]} | tau {np.round(thresholds, 2).tolist()} | best base {best['model']} | "
                  f"epoch {entry['selected_epoch']} | freeze_manifest {sha256_file(out / 'freeze_manifest.json')[:16]}\n")
    print(f"✓ Stage (e) h={h}: MCE thresholds {np.round(thresholds, 2).tolist()}, best base {best['model']}; frozen")


# =============================================================================
# STAGE (f): FROZEN TEST EVALUATION
# =============================================================================


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, minority: Iterable[int] | None = None) -> dict[str, Any]:
    """Undefined per-class values are NaN; macro averages use classes present in y_true (SAP 7.6)."""
    y_true = np.asarray(y_true, np.int64); y_pred = np.asarray(y_pred, np.int64)
    cm = np.bincount(y_true * N_CLASSES + y_pred, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    support, predicted, tp = cm.sum(1).astype(float), cm.sum(0).astype(float), np.diag(cm).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(support > 0, np.where(predicted > 0, tp / predicted, 0.0), np.nan)
        f1 = np.where(np.isnan(recall), np.nan,
                      np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0))
    present = support > 0
    out = {"n": int(len(y_true)), "accuracy": float(tp.sum() / max(1, len(y_true))),
           "balanced_accuracy": float(np.nanmean(recall[present])) if present.any() else np.nan,
           "macro_precision": float(np.nanmean(precision[present])) if present.any() else np.nan,
           "macro_recall": float(np.nanmean(recall[present])) if present.any() else np.nan,
           "macro_f1": float(np.nanmean(f1[present])) if present.any() else np.nan,
           "classes_present": int(present.sum()),
           "precision_per_class": precision, "recall_per_class": recall, "f1_per_class": f1,
           "support_per_class": support.astype(int), "confusion_matrix": cm}
    if minority is not None:
        minority = [k for k in minority if present[k]]
        out["minority_macro_recall"] = float(np.mean(recall[minority])) if minority else np.nan
        out["minimum_minority_recall"] = float(np.min(recall[minority])) if minority else np.nan
    return out


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    p = validate_probability_matrix(probabilities, len(y_true), N_CLASSES, "probabilities")
    one_hot = np.eye(N_CLASSES)[y_true]
    present = np.bincount(y_true, minlength=N_CLASSES) > 0
    auc = (float(roc_auc_score(y_true, p, multi_class="ovr", average="macro", labels=list(range(N_CLASSES))))
           if present.all() else np.nan)
    top = p.max(1); correct = (p.argmax(1) == y_true).astype(float)
    bins = np.minimum((top * RELIABILITY_BINS).astype(int), RELIABILITY_BINS - 1)
    ece = sum(abs(top[bins == b].mean() - correct[bins == b].mean()) * (bins == b).mean()
              for b in range(RELIABILITY_BINS) if (bins == b).any())
    classwise = []
    for k in range(N_CLASSES):
        pk = p[:, k]; bk = np.minimum((pk * RELIABILITY_BINS).astype(int), RELIABILITY_BINS - 1)
        classwise.append(sum(abs(pk[bk == b].mean() - one_hot[bk == b, k].mean()) * (bk == b).mean()
                             for b in range(RELIABILITY_BINS) if (bk == b).any()))
    return {"roc_auc_ovr_macro": auc,
            "nll": float(-np.mean(np.log(np.clip(p[np.arange(len(y_true)), y_true], CONFIDENCE_EPSILON, 1.0)))),
            "brier_multiclass": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
            "ece_top_label": float(ece), "ece_classwise_macro": float(np.mean(classwise)),
            **{f"ece_{LABELS[k].replace(' ', '_')}": float(classwise[k]) for k in range(N_CLASSES)}}


SCALAR_KEYS = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
               "minority_macro_recall", "minimum_minority_recall"]


def cluster_bootstrap(children: np.ndarray, y: np.ndarray, decisions: dict[str, np.ndarray], minority: list[int],
                      comparisons: list[tuple[str, str]], replicates: int, seed: int):
    """Resample children with replacement; every pair of a sampled child is included."""
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(children, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(unique)))
    ends = np.append(starts[1:], len(order))
    members = [order[s:e] for s, e in zip(starts, ends)]
    values = {m: {k: [] for k in SCALAR_KEYS} for m in decisions}
    deltas = {c: {k: [] for k in SCALAR_KEYS} for c in comparisons}
    for _ in range(replicates):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(unique), size=len(unique))])
        current = {m: classification_metrics(y[idx], pred[idx], minority) for m, pred in decisions.items()}
        for m in decisions:
            for k in SCALAR_KEYS:
                values[m][k].append(current[m][k])
        for a, b in comparisons:
            for k in SCALAR_KEYS:
                deltas[(a, b)][k].append(current[b][k] - current[a][k])

    def summarise(samples):
        arr = np.asarray(samples, float); ok = arr[np.isfinite(arr)]
        return (np.quantile(ok, 0.025), np.quantile(ok, 0.975), len(ok)) if len(ok) else (np.nan, np.nan, 0)

    ci = [{"method": m, "metric": k, **dict(zip(["ci_low", "ci_high", "valid_replicates"], summarise(values[m][k])))}
          for m in decisions for k in SCALAR_KEYS]
    dl = [{"comparison": f"{b} minus {a}", "metric": k, **dict(zip(["ci_low", "ci_high", "valid_replicates"], summarise(deltas[(a, b)][k])))}
          for a, b in comparisons for k in SCALAR_KEYS]
    return pd.DataFrame(ci), pd.DataFrame(dl)


def verify_freeze(h: int) -> dict[str, Any]:
    out = horizon_dir(h)
    path = out / "freeze_manifest.json"
    if not path.exists():
        raise RuntimeError(f"h={h}: no freeze manifest; run stage (e) first.")
    entry = load_json(path)
    current = {str(p.relative_to(out)): sha256_file(p) for p in frozen_files(h)}
    if current != entry["files_sha256"]:
        changed = sorted(set(current.items()) ^ set(entry["files_sha256"].items()))
        raise RuntimeError(f"h={h}: frozen artefacts changed after freezing: {changed[:4]}")
    if entry["code_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError(f"h={h}: code changed after freezing.")
    if f"freeze_manifest {sha256_file(path)[:16]}" not in FREEZE_LOG.read_text(encoding="utf-8"):
        raise RuntimeError(f"h={h}: freeze manifest is not recorded in FREEZE_LOG_v4.")
    return entry


def run_stage_f(h: int) -> None:
    print("\n" + "=" * 96 + f"\nSTAGE (f) h={h}: FROZEN TEST EVALUATION\n" + "=" * 96)
    freeze = verify_freeze(h)
    dev, val, test, manifest, oof, meta_val, meta_test = load_meta(h)
    out = horizon_dir(h)
    thresholds = np.load(out / "class_specific_thresholds.npy")
    best_name = load_json(out / "best_base_learner.json")["best_base_learner"]
    base_tau = pd.read_csv(out / "base_learner_thresholds.csv")
    p_meta = np.load(out / "meta_dnn_test_probabilities.npy").astype(np.float64)
    y = test["y"].to_numpy(np.int64)
    minority = [LABEL_TO_IDX[c] for c in manifest["minority_classes_from_dev"]]

    scores, mce_ilcadt = apply_threshold_decision(p_meta, thresholds)
    np.save(out / "mce_ilcadt_test_decision_scores.npy", scores)
    decisions = {"persistence": test["status_t"].to_numpy(np.int64), "mce": p_meta.argmax(1), "mce_ilcadt": mce_ilcadt}
    probabilities = {"meta_dnn": p_meta}
    for b, name in enumerate(MODEL_ORDER):
        p = base_block(meta_test, b)
        tau = base_tau.loc[base_tau["model"] == name].set_index("class").loc[LABELS, "threshold"].to_numpy(float)
        probabilities[name] = p
        decisions[f"{name}_argmax"] = p.argmax(1)
        decisions[f"{name}_ilcadt"] = apply_threshold_decision(p, tau)[1]
    decisions["best_base"] = decisions[f"{best_name}_argmax"]
    decisions["best_base_ilcadt"] = decisions[f"{best_name}_ilcadt"]
    order = MAIN_METHODS + [f"{n}_{d}" for n in MODEL_ORDER for d in ("argmax", "ilcadt")]
    np.save(out / "test_decisions.npy", np.stack([decisions[m] for m in order], axis=1))
    save_json({"columns": order}, out / "test_decisions_columns.json")

    subsets = {"all": np.ones(len(test), bool), "status_change": decisions["persistence"] != y,
               "new_children": test["new_child"].to_numpy(bool)}
    for pk in sorted(test["puskesmas"].unique()):
        subsets[f"puskesmas={pk}"] = (test["puskesmas"] == pk).to_numpy()
    rows, per_class_rows, ci_frames, delta_frames = [], [], [], []
    children = test["child_code"].to_numpy()
    for subset_name, mask in subsets.items():
        if not mask.any():
            continue
        for method in order:
            m = classification_metrics(y[mask], decisions[method][mask], minority)
            rows.append({"subset": subset_name, "method": method, "main_table": method in MAIN_METHODS,
                         "n_children": int(np.unique(children[mask]).size),
                         **{k: m[k] for k in ["n", "classes_present"] + SCALAR_KEYS},
                         "confusion_matrix": m["confusion_matrix"].tolist()})
            for k, label in enumerate(LABELS):
                per_class_rows.append({"subset": subset_name, "method": method, "class": label,
                                       "support": int(m["support_per_class"][k]), "precision": m["precision_per_class"][k],
                                       "recall": m["recall_per_class"][k], "f1": m["f1_per_class"][k]})
        ci, dl = cluster_bootstrap(children[mask], y[mask], {k: decisions[k][mask] for k in MAIN_METHODS},
                                   minority, COMPARISONS, BOOTSTRAP_REPLICATES, RANDOM_STATE)
        ci_frames.append(ci.assign(subset=subset_name)); delta_frames.append(dl.assign(subset=subset_name))
        print(f"  subset {subset_name}: done", flush=True)
    prob_rows = [{"method": name, **probability_metrics(y, p)} for name, p in probabilities.items()]

    pd.DataFrame(rows).to_csv(out / "test_classification_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(out / "test_per_class_metrics.csv", index=False)
    pd.DataFrame(prob_rows).to_csv(out / "test_probability_metrics.csv", index=False)
    pd.concat(ci_frames).to_csv(out / "test_child_bootstrap_ci.csv", index=False)
    pd.concat(delta_frames).to_csv(out / "test_child_bootstrap_paired_differences.csv", index=False)
    save_json({"h": h, **{k: v for k, v in run_config().items() if k != "feature_order"},
               "fingerprint_test": manifest["fingerprint_test"], "freeze_verified": True,
               "frozen_at": freeze["frozen_at"], "mce_thresholds": freeze["mce_thresholds"],
               "best_base_learner": best_name, "minority_classes_from_dev": manifest["minority_classes_from_dev"],
               "bootstrap_unit": "child", "bootstrap_replicates": BOOTSTRAP_REPLICATES, "comparisons": COMPARISONS,
               "calibration_metrics_source": "probabilities only (never threshold-adjusted scores)",
               "persistence": "predicted class at t+h = status at t",
               "puskesmas_subsets": "from the main test predictions; no retraining"}, out / "stage_f_manifest.json")
    print(f"✓ Stage (f) h={h} completed.")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    valid = {"a", "b", "c", "d", "e", "f"}
    if set(RUN_STAGES) - valid:
        raise ValueError(f"Invalid stages: {sorted(set(RUN_STAGES) - valid)}")
    if "a" in RUN_STAGES:
        run_stage_a()
    if "b" in RUN_STAGES:
        run_stage_b()
    for h in HORIZONS:
        for stage, function in (("c", run_stage_c), ("d", run_stage_d), ("e", run_stage_e), ("f", run_stage_f)):
            if stage in RUN_STAGES:
                function(h)
    print("\nMCE-ILCADT-FP v4 finished. Artefacts:", BASE_DIR)


if __name__ == "__main__":
    main()
