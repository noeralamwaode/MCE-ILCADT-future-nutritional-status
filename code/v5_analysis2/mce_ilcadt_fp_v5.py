# -*- coding: utf-8 -*-
"""
MCE-ILCADT-FP v5: detection of CONFIRMED changes in toddler height-for-age status
between calendar month t and t+h (worsening / stable / improvement), Buton Utara.

Analysis plan: SAP_v5.md (frozen 17 October 2026, SHA-256 dab1e43e...).
Built on the audited v4 pipeline (mce_ilcadt_fp_v4.py, SHA-256 7e051fb1...).

Unchanged from v4: master v4 input and eligibility, 33 predictors (information up to t),
six base learners and hyperparameters, child-grouped OOF (B x K = 18 meta-features),
Meta-DNN and epoch rule, ILCADT decision rule and threshold search, freeze logic.
New in v5 (SAP v5):
  * target = direction of change from status(t) to status(t+h), confirmed at t+h+1,
    with status(t-1) == status(t) (baseline confirmed); classes stable / worsening / improvement;
  * blocks defined on the confirmation month; impossible-class mask (normal cannot improve,
    severely stunted cannot worsen) before metrics, before ILCADT tuning and on final decisions;
  * comparators: persistence (always stable), v4 Meta-DNN probabilities mapped to direction
    (raw for AUPRC; recalibrated on tuning data for probability metrics), simple penalised
    multinomial logistic regression, penalised logistic regression with the 33 predictors;
  * primary endpoint: confirmed new stunting among children normal at t; metric AUPRC
    (average precision); child-level cluster bootstrap; claim rule (SAP v5 7.4);
  * secondary: three-class metrics, strata by status at t, DCA, calibration, subgroups.
Sensitivity targets (SAP v5 section 8): MCE_FP_V5_TARGET = no_baseline_confirmation | unconfirmed.

Stages (per horizon h): a environment; b pairs/targets/blocks; c base learners OOF;
d Meta-DNN; e thresholds, comparators, best base, FREEZE; f frozen test evaluation.
Smoke test (MCE_FP_SMOKE=1): small models, blocks shifted inside 2023-2024; needs a v4 smoke
run with the same smoke blocks for the v4-mapped comparator.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input

# =============================================================================
# CONFIGURATION
# =============================================================================

EXPERIMENT_NAME = "MCE_ILCADT_FP_V5_CHANGE_DETECTION_BUTON_UTARA"
TIMEZONE = "Asia/Jakarta"
SAP_SHA256 = "dab1e43e42c6e696a963fbba845ab6bbee9ae0096a7ce618993fcb04c48a1aa0"

MASTER_PATH = Path(os.environ.get("MCE_FP_MASTER_PATH", "/content/MCE_FP_v5/data/master_v4_pseudonymised.parquet")).resolve()
BASE_DIR = Path(os.environ.get("MCE_FP_BASE_DIR", "/content/MCE_FP_v5/runs/primary")).resolve()
V4_RUN_ROOT = Path(os.environ.get("MCE_FP_V4_RUN_ROOT", "/content/MCE_FP_v4/runs/primary")).resolve()
SAP_PATH = os.environ.get("MCE_FP_SAP_PATH", "")

HORIZONS = tuple(int(x) for x in os.environ.get("MCE_FP_HORIZONS", "1,3,6").split(",") if x.strip())
RUN_STAGES = tuple(x.strip().lower() for x in os.environ.get("MCE_FP_STAGES", "a,b,c,d,e,f").split(",") if x.strip())
SMOKE_TEST = os.environ.get("MCE_FP_SMOKE", "0") == "1"

# --- target definition (SAP v5 3.1 and 8) --------------------------------------
TARGET = os.environ.get("MCE_FP_V5_TARGET", "confirmed").lower()
TARGET_RULES = {
    "confirmed": {"baseline_confirmation": True, "target_confirmation": True},
    "no_baseline_confirmation": {"baseline_confirmation": False, "target_confirmation": True},
    "unconfirmed": {"baseline_confirmation": False, "target_confirmation": False},
}
if TARGET not in TARGET_RULES:
    raise ValueError(f"MCE_FP_V5_TARGET must be one of {sorted(TARGET_RULES)}")
BASELINE_CONFIRMATION = TARGET_RULES[TARGET]["baseline_confirmation"]
TARGET_CONFIRMATION = TARGET_RULES[TARGET]["target_confirmation"]
ANALYSIS = "primary" if TARGET == "confirmed" else f"sens_{TARGET}"

# --- temporal blocks on the LAST month used by the label (confirmation month) ---
if not SMOKE_TEST:
    BLOCKS = {"TRAIN_TARGET_MIN": "2023-01", "TRAIN_LAST_MAX": "2024-12", "TUNE_ANCHOR_MIN": "2025-01",
              "TUNE_LAST_MAX": "2025-12", "TEST_ANCHOR_MIN": "2026-01", "TEST_LAST_MAX": ""}
else:  # inside the real training period only
    BLOCKS = {"TRAIN_TARGET_MIN": "2023-01", "TRAIN_LAST_MAX": "2023-12", "TUNE_ANCHOR_MIN": "2024-01",
              "TUNE_LAST_MAX": "2024-06", "TEST_ANCHOR_MIN": "2024-07", "TEST_LAST_MAX": "2024-12"}

# Pair counts in SAP v5 3.2 (confirmed) and independent recounts for the sensitivity targets.
EXPECTED_PAIR_COUNTS = {
    "confirmed": {1: (69113, 38791, 26115), 3: (61585, 29499, 17566), 6: (51172, 17336, 6494)},
    "no_baseline_confirmation": {1: (80504, 42266, 28861), 3: (73152, 32464, 19702), 6: (62517, 19204, 7373)},
    "unconfirmed": {1: (90956, 48866, 34969), 3: (82489, 38405, 25071), 6: (70932, 24157, 11650)},
}
EXPECTED_MASTER = {"rows": 234073, "children": 10321, "eligible": 230551,
                   "labels": {"normal": 190959, "stunted": 31106, "severely stunted": 8486}}

# --- status labels (input) and direction labels (target) -------------------------
STATUS_LABELS = ["normal", "stunted", "severely stunted"]
STATUS_TO_IDX = {label: i for i, label in enumerate(STATUS_LABELS)}
LABELS = ["stable", "worsening", "improvement"]
N_CLASSES = len(LABELS)
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
STABLE, WORSENING, IMPROVEMENT = 0, 1, 2
ELIGIBLE_CATEGORIES = ("normal", "stunted", "severely stunted", "tall")
HAZ_VALID = (-6.0, 6.0)
MAX_AGE_DAYS = 1856

N_SPLITS = 5
META_VALIDATION_CHILD_FRACTION = 0.10
MIN_CHILDREN_PER_CLASS_PER_FOLD = 2
MIN_CHILDREN_RARE_CLASS_IN_TUNING = 10

WEIGHT_GAIN_CODES = ["N", "T", "O", "B"]
FEAT_COLS = [
    "sex_male", "birth_weight_kg", "birth_length_cm",
    "age_months_t", "weight_kg_t", "height_cm_t", "recumbent_t", "muac_cm_t", "muac_observed_t",
    "haz_t", "waz_t", "whz_t", "status_t", "wg_N_t", "wg_T_t", "wg_O_t", "wg_B_t",
    "haz_lag1_observed", "d_haz_1", "d_waz_1", "d_height_1", "d_weight_1",
    "haz_lag3_observed", "d_haz_3",
    "n_obs_to_t", "months_since_prev_obs", "n_prior_stunted_months", "min_haz_to_t",
    "haz_lag6_observed", "haz_lag12_observed", "d_haz_6", "d_haz_12", "n_stunted_months_last12",
]
STATIC_FEATURES = ("sex_male", "birth_weight_kg", "birth_length_cm")
HISTORY_PREFIXES = ("d_", "n_", "months_", "min_", "haz_lag")

MODEL_ORDER = ["RF", "LGBM", "XGB", "CAT", "GBC", "ADA"]
META_DIMENSION = len(MODEL_ORDER) * N_CLASSES

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

# comparators (SAP v5 6)
LR_C = 1.0
LR_MAX_ITER = 5000
RECALIBRATION_C = 1e4  # effectively unpenalised recalibration on tuning data
SPLINE_QUANTILES = (0.05, 0.35, 0.65, 0.95)  # restricted cubic spline, 4 knots
DCA_THRESHOLDS = np.round(np.arange(0.005, 0.1001, 0.005), 3)  # 0.5%-10%
WORKLOADS = (0.05, 0.10)
AGE_BAND_MONTHS = 24

PRIMARY_COMPARATORS = ["v4_mapped", "lr_simple", "lr_33"]  # claim rule: MCE minus each
PROB_METHODS = ["mce", "v4_mapped", "v4_mapped_recal", "lr_simple", "lr_33", "best_base"]

SHARED_DIR = BASE_DIR / "shared"
FREEZE_LOG = SHARED_DIR / "FREEZE_LOG_v5.txt"


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
    return {"analysis": ANALYSIS, "target": TARGET, "baseline_confirmation": BASELINE_CONFIRMATION,
            "target_confirmation": TARGET_CONFIRMATION, "smoke_test": SMOKE_TEST, "blocks": BLOCKS,
            "feature_order": FEAT_COLS, "class_order": LABELS, "model_order": MODEL_ORDER}


def pair_fingerprint(child_codes, anchor_month, labels, h: int) -> str:
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


def apply_mask(probabilities: np.ndarray, status_t: np.ndarray) -> np.ndarray:
    """Impossible classes (SAP v5 5): normal at t cannot improve; severely stunted cannot worsen."""
    p = np.asarray(probabilities, dtype=np.float64).copy()
    status_t = np.asarray(status_t, dtype=np.int64)
    p[status_t == 0, IMPROVEMENT] = 0.0
    p[status_t == 2, WORSENING] = 0.0
    sums = p.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Mask removed all probability mass.")
    return p / sums


# =============================================================================
# STAGE (a)
# =============================================================================


def validate_contract() -> None:
    if len(FEAT_COLS) != len(set(FEAT_COLS)) or len(FEAT_COLS) != 33:
        raise ValueError("Predictor contract (33 unique predictors) violated.")
    if LABELS != ["stable", "worsening", "improvement"]:
        raise ValueError("Class order changed.")
    b = {k: month_index(v) for k, v in BLOCKS.items() if v}
    if not (b["TRAIN_TARGET_MIN"] <= b["TRAIN_LAST_MAX"] < b["TUNE_ANCHOR_MIN"] and b["TUNE_LAST_MAX"] < b["TEST_ANCHOR_MIN"]):
        raise ValueError("Temporal blocks must be ordered.")
    if set(HORIZONS) - {1, 3, 6}:
        raise ValueError("SAP v5 covers h = 1, 3, 6 only.")


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
    import statsmodels
    save_json({
        "experiment_name": EXPERIMENT_NAME, "created_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
        "task": "direction of confirmed change, status(t) -> status(t+h)", "random_state": RANDOM_STATE,
        "horizons_months": list(HORIZONS), **run_config(), "meta_dimension": META_DIMENSION,
        "sap_sha256_expected": SAP_SHA256, "code_sha256": sha256_file(Path(__file__)),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                     "xgboost": xgboost.__version__, "catboost": catboost.__version__, "tensorflow": tf.__version__,
                     "statsmodels": statsmodels.__version__},
        "tensorflow_determinism": determinism,
    }, SHARED_DIR / f"environment_metadata_h{'-'.join(map(str, HORIZONS))}.json")
    print(f"  Horizons {HORIZONS} | target {TARGET} | classes {LABELS} | predictors {len(FEAT_COLS)}")
    print("✓ Stage (a) completed.")


# =============================================================================
# STAGE (b1): MASTER V4 AND PREDICTORS (identical rules to v4, e-PPGBM labels)
# =============================================================================


def load_master() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master v4 not found: {MASTER_PATH}")
    master = pd.read_parquet(MASTER_PATH) if MASTER_PATH.suffix == ".parquet" else pd.read_pickle(MASTER_PATH)
    audit: dict[str, Any] = {"master_file": MASTER_PATH.name, "master_sha256": sha256_file(MASTER_PATH)}
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

    category = master["tbu_category"].astype(str)
    haz = master["haz"].astype(float)
    base_ok = (category.isin(ELIGIBLE_CATEGORIES) & (master["age_days"] >= 0)
               & (master["age_days"] <= MAX_AGE_DAYS) & master["height_cm"].notna())
    eligible = base_ok & category.isin(STATUS_LABELS) & haz.between(*HAZ_VALID)
    if not (eligible == master["eligible"].astype(bool)).all():
        raise ValueError("Re-derived eligibility differs from master v4.")
    label = category.where(eligible)

    obs = pd.DataFrame({"child_id": master["child_id"].astype(str), "month": month,
                        "puskesmas": master["puskesmas"].astype(str)})
    obs["sex_male"] = master["sex_male"].astype(float)
    obs["birth_weight_kg"] = master["birth_weight_kg"].astype(float)
    obs["birth_length_cm"] = master["birth_length_cm"].astype(float)
    obs["age_months_t"] = np.floor(master["age_days"].astype(float) / 30.4375)
    obs["weight_kg_t"] = master["weight_kg"].astype(float)
    obs["height_cm_t"] = master["height_cm"].astype(float)
    obs["recumbent_t"] = master["measure_position"].map({"Terlentang": 1.0, "Berdiri": 0.0}).astype(float)
    obs["muac_cm_t"] = master["muac_cm"].astype(float)
    obs["muac_observed_t"] = master["muac_observed"].astype(float)
    obs["haz_t"] = haz.where(haz.between(*HAZ_VALID))
    obs["waz_t"] = master["waz"].astype(float)
    obs["whz_t"] = master["whz"].astype(float)
    for code in WEIGHT_GAIN_CODES:
        obs[f"wg_{code}_t"] = (master["weight_gain_code_t"].astype(str) == code).astype(float)
    obs["label_text"] = label
    obs["eligible"] = eligible.to_numpy()
    obs["status_t"] = obs["label_text"].map(STATUS_TO_IDX)
    obs["child_code"] = obs["child_id"].rank(method="dense").astype(np.int64) - 1
    audit["eligible_observations"] = int(eligible.sum())
    audit["months"] = [month_text(int(month.min())), month_text(int(month.max()))]
    obs = obs.sort_values(["child_id", "month"]).reset_index(drop=True)
    return obs, audit


def add_history_features(obs: pd.DataFrame) -> pd.DataFrame:
    """Growth history from observations with month <= t only (identical to v4)."""
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


# =============================================================================
# STAGE (b2): PAIRS, DIRECTION TARGET, BLOCKS
# =============================================================================


def status_lookup(obs: pd.DataFrame) -> pd.Series:
    e = obs.loc[obs["eligible"]]
    return pd.Series(e["status_t"].to_numpy(np.float64), index=pd.MultiIndex.from_arrays([e["child_id"], e["month"]]))


def build_pairs(obs: pd.DataFrame, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (pairs, attrition candidates). Labels use status at t+h (and t+h+1); predictors use <= t."""
    key = status_lookup(obs)
    anchors = obs.loc[obs["eligible"]].copy()
    look = lambda k: key.reindex(pd.MultiIndex.from_arrays([anchors["child_id"], anchors["month"] + k])).to_numpy()
    anchors["status_prev"] = look(-1)
    anchors["status_target"] = look(h)
    anchors["status_confirm"] = look(h + 1)
    anchors = anchors.loc[anchors["status_target"].notna()].copy()
    if BASELINE_CONFIRMATION:
        anchors = anchors.loc[anchors["status_prev"].notna() & (anchors["status_prev"] == anchors["status_t"])].copy()
    s = anchors["status_t"].to_numpy(np.int64)
    d1 = np.sign(anchors["status_target"].to_numpy() - s)
    anchors["direction_unconfirmed"] = d1
    attrition = anchors.copy()
    if TARGET_CONFIRMATION:
        anchors = anchors.loc[anchors["status_confirm"].notna()].copy()
        s = anchors["status_t"].to_numpy(np.int64)
        d1 = np.sign(anchors["status_target"].to_numpy() - s)
        d2 = np.sign(anchors["status_confirm"].to_numpy() - s)
        direction = np.where((d1 != 0) & (d1 == d2), d1, 0)
        anchors["last_month"] = anchors["month"] + h + 1
    else:
        direction = d1
        anchors["last_month"] = anchors["month"] + h
    anchors["y"] = np.select([direction == 1, direction == -1], [WORSENING, IMPROVEMENT], STABLE).astype(np.int64)
    anchors["target_month"] = anchors["month"] + h
    anchors["status_t"] = anchors["status_t"].astype(np.int64)
    anchors["h"] = h
    pairs = anchors.rename(columns={"month": "anchor_month"})
    if ((pairs["status_t"] == 0) & (pairs["y"] == IMPROVEMENT)).any() or ((pairs["status_t"] == 2) & (pairs["y"] == WORSENING)).any():
        raise RuntimeError("Impossible direction label created.")
    return pairs.sort_values(["child_id", "anchor_month"]).reset_index(drop=True), attrition


def run_stage_b() -> None:
    print("\n" + "=" * 96 + "\nSTAGE (b): PAIRS, CONFIRMED-CHANGE TARGET, TEMPORAL BLOCKS\n" + "=" * 96)
    validate_contract()
    obs, audit = load_master()
    obs = add_history_features(obs)
    save_json(audit, SHARED_DIR / "dataset_audit.json")
    pd.DataFrame({
        "feature": FEAT_COLS,
        "information_time": ["birth (static)" if f in STATIC_FEATURES else
                             "history <= t" if f.startswith(HISTORY_PREFIXES) else "month t" for f in FEAT_COLS],
        "uses_information_after_t": False,
    }).to_csv(SHARED_DIR / "feature_availability_audit.csv", index=False)

    b = {k: month_index(v) if v else None for k, v in BLOCKS.items()}
    codes = obs.drop_duplicates("child_id").set_index("child_id")["child_code"]
    attrition_rows = []
    for h in HORIZONS:
        out = horizon_dir(h)
        out.mkdir(parents=True, exist_ok=True)
        pairs, cand = build_pairs(obs, h)

        dev = pairs.loc[(pairs["target_month"] >= b["TRAIN_TARGET_MIN"]) & (pairs["last_month"] <= b["TRAIN_LAST_MAX"])].reset_index(drop=True)
        val = pairs.loc[(pairs["anchor_month"] >= b["TUNE_ANCHOR_MIN"]) & (pairs["last_month"] <= b["TUNE_LAST_MAX"])].reset_index(drop=True)
        test_mask = pairs["anchor_month"] >= b["TEST_ANCHOR_MIN"]
        if b["TEST_LAST_MAX"] is not None:
            test_mask &= pairs["last_month"] <= b["TEST_LAST_MAX"]
        test = pairs.loc[test_mask].reset_index(drop=True)

        # attrition (SAP v5 3): pairs with t+h observed, with vs without the confirmation month, per block window
        if TARGET_CONFIRMATION:
            cand = cand.assign(has_confirmation=cand["status_confirm"].notna())
            windows = {"training": (cand["month"] + h >= b["TRAIN_TARGET_MIN"]) & (cand["month"] + h + 1 <= b["TRAIN_LAST_MAX"]),
                       "threshold_tuning": (cand["month"] >= b["TUNE_ANCHOR_MIN"]) & (cand["month"] + h + 1 <= b["TUNE_LAST_MAX"]),
                       "test": (cand["month"] >= b["TEST_ANCHOR_MIN"]) & ((cand["month"] + h + 1 <= b["TEST_LAST_MAX"]) if b["TEST_LAST_MAX"] is not None else True)}
            for name, w in windows.items():
                for flag, frame in cand.loc[w].groupby("has_confirmation"):
                    attrition_rows.append({"h": h, "block": name, "confirmation_observed": bool(flag), "pairs": int(len(frame)),
                                           **{f"pct_status_t_{STATUS_LABELS[k]}": float((frame["status_t"] == k).mean() * 100) for k in range(3)},
                                           "pct_unconfirmed_worsening": float((frame["direction_unconfirmed"] == 1).mean() * 100),
                                           "pct_unconfirmed_improvement": float((frame["direction_unconfirmed"] == -1).mean() * 100)})

        if len(test) == 0 or len(val) == 0:
            raise RuntimeError(f"h={h}: empty tuning or test block.")
        if not SMOKE_TEST and (len(dev), len(val), len(test)) != EXPECTED_PAIR_COUNTS[TARGET][h]:
            raise ValueError(f"h={h}: pair counts {(len(dev), len(val), len(test))} differ from {EXPECTED_PAIR_COUNTS[TARGET][h]}.")
        if val["anchor_month"].min() <= dev["last_month"].max():
            raise RuntimeError("Temporal leakage: a tuning prediction month precedes a training label month.")
        if test["anchor_month"].min() <= max(dev["last_month"].max(), val["last_month"].max()):
            raise RuntimeError("Temporal leakage: a test prediction month precedes a training/tuning label month.")
        keys = lambda f: set(zip(f["child_id"], f["anchor_month"]))
        if keys(dev) & keys(val) or keys(dev) & keys(test) or keys(val) & keys(test):
            raise RuntimeError("A pair appears in more than one block.")
        for name, frame in (("training", dev), ("tuning", val)):
            counts = np.bincount(frame["y"], minlength=N_CLASSES)
            if np.any(counts == 0):
                raise ValueError(f"h={h}: {name} block lacks a class {counts.tolist()}")
        for k in (WORSENING, IMPROVEMENT):
            if val.loc[val["y"] == k, "child_id"].nunique() < MIN_CHILDREN_RARE_CLASS_IN_TUNING:
                raise ValueError(f"h={h}: too few children with {LABELS[k]} in tuning data.")
        dev_counts = np.bincount(dev["y"], minlength=N_CLASSES)
        test["new_child"] = ~test["child_id"].isin(set(dev["child_id"]))

        # child strata: rarest direction class experienced (rarity order from training counts)
        rarity = sorted([WORSENING, IMPROVEMENT], key=lambda k: dev_counts[k])
        experienced = dev.groupby("child_id")["y"].agg(lambda s: set(s.tolist()))
        strata = experienced.map(lambda cls: next((k for k in rarity if k in cls), STABLE)).astype(np.int64)
        strata = strata.loc[sorted(strata.index, key=codes.__getitem__)]
        per_stratum = strata.value_counts().reindex(range(N_CLASSES)).fillna(0).astype(int)
        if per_stratum.min() < N_SPLITS * MIN_CHILDREN_PER_CLASS_PER_FOLD:
            raise ValueError(f"h={h}: too few children in a stratum {per_stratum.tolist()}")
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        fold_of_child = pd.Series(0, index=strata.index, dtype=np.int64)
        for fold, (_, idx) in enumerate(skf.split(strata.index, strata.to_numpy()), start=1):
            fold_of_child.iloc[idx] = fold
        dev["oof_fold"] = dev["child_id"].map(fold_of_child).astype(np.int64)
        for fold in range(1, N_SPLITS + 1):
            tr_counts = np.bincount(dev.loc[dev["oof_fold"] != fold, "y"], minlength=N_CLASSES)
            if np.any(tr_counts == 0):
                raise ValueError(f"h={h} fold {fold}: training fold lacks a class.")
            for k in (WORSENING, IMPROVEMENT):
                if dev.loc[(dev["oof_fold"] == fold) & (dev["y"] == k), "child_id"].nunique() < MIN_CHILDREN_PER_CLASS_PER_FOLD:
                    raise ValueError(f"h={h} fold {fold}: fewer than {MIN_CHILDREN_PER_CLASS_PER_FOLD} children with {LABELS[k]}.")

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
        keep = (["child_id", "child_code", "puskesmas", "anchor_month", "target_month", "last_month", "h", "y",
                 "status_prev", "status_target", "status_confirm", "direction_unconfirmed"] + FEAT_COLS)
        dev[keep + ["oof_fold", "meta_subset"]].to_pickle(out / "dev_pairs.pkl")
        val[keep].to_pickle(out / "tune_pairs.pkl")
        test[keep + ["new_child"]].to_pickle(out / "test_pairs.pkl")

        rows = []
        for name, frame in (("training", dev), ("threshold_tuning", val), ("test", test)):
            counts = np.bincount(frame["y"], minlength=N_CLASSES)
            normal = frame.loc[frame["status_t"] == 0]
            for k, label in enumerate(LABELS):
                rows.append({"block": name, "class": label, "pairs": int(counts[k]), "percent": float(counts[k] / len(frame) * 100),
                             "children": int(frame["child_id"].nunique()),
                             "children_with_class": int(frame.loc[frame["y"] == k, "child_id"].nunique())})
            rows.append({"block": name, "class": "PRIMARY: new stunting among normal at t", "pairs": int((normal["y"] == WORSENING).sum()),
                         "percent": float((normal["y"] == WORSENING).mean() * 100), "children": int(normal["child_id"].nunique()),
                         "children_with_class": int(normal.loc[normal["y"] == WORSENING, "child_id"].nunique())})
        pd.DataFrame(rows).to_csv(out / "class_distribution_by_block.csv", index=False)

        span = lambda f, c: [month_text(int(f[c].min())), month_text(int(f[c].max()))]
        save_json({
            "h": h, "status": "ok", **run_config(), "fingerprint_dev": fp["dev"], "fingerprint_val": fp["val"],
            "fingerprint_test": fp["test"], "master_sha256": audit["master_sha256"],
            "n_dev_pairs": int(len(dev)), "n_val_pairs": int(len(val)), "n_test_pairs": int(len(test)),
            "n_dev_children": int(dev["child_id"].nunique()), "n_test_children": int(test["child_id"].nunique()),
            "dev_prediction_months": span(dev, "anchor_month"), "dev_label_months": span(dev, "last_month"),
            "val_prediction_months": span(val, "anchor_month"), "val_label_months": span(val, "last_month"),
            "test_prediction_months": span(test, "anchor_month"), "test_label_months": span(test, "last_month"),
            "children_per_stratum_dev": per_stratum.tolist(), "rarity_order_from_dev": [LABELS[k] for k in rarity],
            "minority_classes_from_dev": [LABELS[WORSENING], LABELS[IMPROVEMENT]],
            "test_pairs_new_children": int(test["new_child"].sum()),
        }, out / "stage_b_manifest.json")
        print(f"  h={h}: training={len(dev)} | tuning={len(val)} | test={len(test)} | "
              f"training classes {dev_counts.tolist()} | strata {per_stratum.tolist()}")
    if attrition_rows:
        pd.DataFrame(attrition_rows).to_csv(SHARED_DIR / f"attrition_confirmation_h{'-'.join(map(str, HORIZONS))}.csv", index=False)
    print("✓ Stage (b) completed.")


def load_stage_b(h: int):
    out = horizon_dir(h)
    manifest = load_json(out / "stage_b_manifest.json")
    if manifest.get("status") != "ok":
        raise RuntimeError(f"h={h}: stage (b) status is {manifest.get('status')}")
    for key, value in run_config().items():
        if manifest[key] != value:
            raise ValueError(f"Configuration '{key}' changed since stage (b).")
    dev = pd.read_pickle(out / "dev_pairs.pkl")
    val = pd.read_pickle(out / "tune_pairs.pkl")
    test = pd.read_pickle(out / "test_pairs.pkl")
    for name, frame in (("dev", dev), ("val", val), ("test", test)):
        if pair_fingerprint(frame["child_code"], frame["anchor_month"], frame["y"], h) != manifest[f"fingerprint_{name}"]:
            raise ValueError(f"{name} pair table does not match its stage-(b) fingerprint.")
    return dev, val, test, manifest


# =============================================================================
# STAGE (c): BASE LEARNERS AND CHILD-GROUPED STRICT OOF (identical to v4)
# =============================================================================


def fit_transform_foldwise(train_frame: pd.DataFrame, apply_frames: list[pd.DataFrame]):
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
    save_json({"h": h, "fingerprint_dev": manifest["fingerprint_dev"], "fingerprint_val": manifest["fingerprint_val"],
               "fingerprint_test": manifest["fingerprint_test"], "oof_shape": list(oof.shape),
               "meta_tune_shape": list(meta_val.shape), "meta_test_shape": list(meta_test.shape),
               "oof_grouping": "child", "tuning_and_test_meta_features": "mean of the 5 fold models",
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
# STAGE (d): META-DNN (identical to v4)
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
               "selected_epoch": epoch, "selection_metric": "meta-validation loss", "tie_breaker": "higher macro-F1",
               "refit_data": "all training pairs", "tuning_labels_used": False, "test_labels_used": False,
               "note": "probabilities stored unmasked; the impossible-class mask is applied in stages (e) and (f)"},
              out / "stage_d_manifest.json")
    print(f"✓ Stage (d) h={h}: selected epoch {epoch}")


# =============================================================================
# STAGE (e): THRESHOLDS, COMPARATORS, BEST BASE, FREEZE
# =============================================================================


def threshold_adjusted_scores(confidences: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(confidences, np.float64), CONFIDENCE_EPSILON, 1 - CONFIDENCE_EPSILON)
    tau = np.clip(np.asarray(thresholds, np.float64), CONFIDENCE_EPSILON, 1 - CONFIDENCE_EPSILON)
    shifted = np.clip((np.log(p) - np.log1p(-p)) - (np.log(tau) - np.log1p(-tau)).reshape(1, -1), -60, 60)
    raw = 1.0 / (1.0 + np.exp(-shifted))
    return raw / raw.sum(axis=1, keepdims=True)


def apply_threshold_decision(confidences, thresholds, status_t):
    """ILCADT on masked probabilities; the mask is applied again to the final scores (SAP v5 5)."""
    scores = threshold_adjusted_scores(confidences, thresholds)
    status_t = np.asarray(status_t, np.int64)
    scores[status_t == 0, IMPROVEMENT] = 0.0
    scores[status_t == 2, WORSENING] = 0.0
    scores = scores / scores.sum(axis=1, keepdims=True)
    return scores, np.argmax(scores, axis=1).astype(np.int64)


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


# --- comparator: v4 Meta-DNN probabilities mapped to direction ---------------------


def load_v4_mapped(h: int, frames: dict[str, pd.DataFrame]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Map frozen v4 status probabilities to direction for v5 tuning/test pairs (matched on child_code, prediction month)."""
    art = V4_RUN_ROOT / f"h{h}" / "artifacts"
    freeze = load_json(art / "freeze_manifest.json")
    if freeze.get("analysis") != "primary" or freeze.get("label_source") != "eppgbm" or freeze.get("ablation") != "none" \
            or freeze.get("exclude_flagged") or bool(freeze.get("smoke_test")) != SMOKE_TEST:
        raise ValueError(f"h={h}: {V4_RUN_ROOT} is not the v4 primary analysis (smoke={SMOKE_TEST}).")
    if freeze.get("master_sha256") != sha256_file(MASTER_PATH):
        raise ValueError(f"h={h}: v4 run used a different master file.")
    freeze_hash = sha256_file(art / "freeze_manifest.json")[:16]
    if f"freeze_manifest {freeze_hash}" not in (V4_RUN_ROOT / "shared" / "FREEZE_LOG_v4.txt").read_text(encoding="utf-8"):
        raise ValueError(f"h={h}: v4 freeze manifest is not recorded in FREEZE_LOG_v4.")
    files = {"tune": ("tune_pairs.pkl", "meta_dnn_tune_probabilities.npy"), "test": ("test_pairs.pkl", "meta_dnn_test_probabilities.npy")}
    hashes = {}
    for pair_file, prob_file in files.values():
        for f in (pair_file, prob_file):
            digest = sha256_file(art / f)
            if digest != freeze["files_sha256"][f]:
                raise ValueError(f"v4 artefact {f} (h={h}) differs from its frozen version.")
            hashes[f] = digest
    out = {}
    for name, (pair_file, prob_file) in files.items():
        v4_pairs = pd.read_pickle(art / pair_file)
        p4 = np.load(art / prob_file).astype(np.float64)
        p4 /= p4.sum(axis=1, keepdims=True)
        index = pd.Series(np.arange(len(v4_pairs)), index=pd.MultiIndex.from_arrays([v4_pairs["child_code"], v4_pairs["anchor_month"]]))
        frame = frames[name]
        pos = index.reindex(pd.MultiIndex.from_arrays([frame["child_code"], frame["anchor_month"]])).to_numpy()
        if np.isnan(pos.astype(float)).any():
            raise ValueError(f"h={h}: {int(np.isnan(pos.astype(float)).sum())} v5 {name} pairs are missing from the v4 run.")
        pos = pos.astype(np.int64)
        if not (v4_pairs["status_t"].to_numpy(np.int64)[pos] == frame["status_t"].to_numpy(np.int64)).all():
            raise ValueError(f"h={h}: status at t differs between v4 and v5 {name} pairs.")
        p = p4[pos]
        s = frame["status_t"].to_numpy(np.int64)
        mapped = np.zeros((len(frame), N_CLASSES))
        for j in range(3):
            rows = s == j
            mapped[rows, STABLE] = p[rows, j]
            mapped[rows, WORSENING] = p[rows, j + 1:].sum(axis=1)
            mapped[rows, IMPROVEMENT] = p[rows, :j].sum(axis=1)
        out[name] = apply_mask(mapped, s)
    return out, {"v4_run_root": str(V4_RUN_ROOT), "v4_files_sha256": hashes, "v4_freeze_manifest_sha256_16": freeze_hash}


def recalibrate_multinomial(p_tune: np.ndarray, y_tune: np.ndarray, s_tune, p_apply: np.ndarray, s_apply) -> tuple[np.ndarray, Any]:
    """Multinomial logistic recalibration fitted on tuning data only (used for probability metrics/DCA)."""
    feat = lambda p: np.log(np.clip(p, 1e-6, 1.0))
    model = LogisticRegression(C=RECALIBRATION_C, solver="lbfgs", max_iter=LR_MAX_ITER)
    model.fit(feat(p_tune), y_tune)
    raw = predict_proba_lr(model, feat(p_apply))
    return apply_mask(raw, s_apply), model


# --- comparators: penalised logistic regression -----------------------------------


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis (Harrell), linear term + (k-2) non-linear terms."""
    k = len(knots)
    t = knots
    pos = lambda v: np.maximum(v, 0.0) ** 3
    scale = (t[-1] - t[0]) ** 2
    cols = [x]
    for j in range(k - 2):
        term = (pos(x - t[j]) - pos(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                + pos(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])) / scale
        cols.append(term)
    return np.column_stack(cols)


class SimpleLRDesign:
    """Status indicators, RCS(HAZ_t), RCS(age_t), status x HAZ_t, d_haz_1 (+ observed flag). Fitted on training only."""

    def fit(self, frame: pd.DataFrame):
        self.haz_knots = np.quantile(frame["haz_t"].dropna(), SPLINE_QUANTILES)
        self.age_knots = np.quantile(frame["age_months_t"].dropna(), SPLINE_QUANTILES)
        self.medians = {c: float(frame[c].median()) for c in ("haz_t", "age_months_t", "d_haz_1")}
        x = self._raw(frame)
        self.scaler = StandardScaler().fit(x)
        return self

    def _raw(self, frame: pd.DataFrame) -> np.ndarray:
        haz = frame["haz_t"].fillna(self.medians["haz_t"]).to_numpy(float)
        age = frame["age_months_t"].fillna(self.medians["age_months_t"]).to_numpy(float)
        d1 = frame["d_haz_1"].fillna(self.medians["d_haz_1"]).to_numpy(float)
        s = frame["status_t"].to_numpy(np.int64)
        stunted, severe = (s == 1).astype(float), (s == 2).astype(float)
        return np.column_stack([stunted, severe, rcs_basis(haz, self.haz_knots), rcs_basis(age, self.age_knots),
                                stunted * haz, severe * haz, d1, frame["haz_lag1_observed"].to_numpy(float)])

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.scaler.transform(self._raw(frame))
        if not np.isfinite(x).all():
            raise ValueError("Simple LR design is not finite.")
        return x


def predict_proba_lr(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    aligned = np.zeros((len(x), N_CLASSES))
    for column, k in enumerate(model.classes_):
        aligned[:, int(k)] = raw[:, column]
    return aligned / aligned.sum(axis=1, keepdims=True)


def fit_lr_comparators(dev: pd.DataFrame, frames: dict[str, pd.DataFrame], out: Path) -> dict[str, dict[str, np.ndarray]]:
    y = dev["y"].to_numpy(np.int64)
    result = {}
    design = SimpleLRDesign().fit(dev)
    lr_simple = LogisticRegression(C=LR_C, solver="lbfgs", max_iter=LR_MAX_ITER).fit(design.transform(dev), y)
    result["lr_simple"] = {name: apply_mask(predict_proba_lr(lr_simple, design.transform(f)), f["status_t"]) for name, f in frames.items()}
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(dev[FEAT_COLS])
    scaler = StandardScaler().fit(imputer.transform(dev[FEAT_COLS]))
    tr33 = lambda f: scaler.transform(imputer.transform(f[FEAT_COLS]))
    lr_33 = LogisticRegression(C=LR_C, solver="lbfgs", max_iter=LR_MAX_ITER).fit(tr33(dev), y)
    result["lr_33"] = {name: apply_mask(predict_proba_lr(lr_33, tr33(f)), f["status_t"]) for name, f in frames.items()}
    joblib.dump({"lr_simple": lr_simple,
                 "lr_simple_design_params": {"haz_knots": design.haz_knots, "age_knots": design.age_knots,
                                             "medians": design.medians, "scaler_mean": design.scaler.mean_,
                                             "scaler_scale": design.scaler.scale_}, "lr_33": lr_33, "lr_33_imputer": imputer,
                 "lr_33_scaler": scaler, "C": LR_C, "fit_scope": "all training pairs"}, out / "lr_comparators.joblib", compress=3)
    for name, model in (("lr_simple", lr_simple), ("lr_33", lr_33)):
        if model.n_iter_.max() >= LR_MAX_ITER:
            raise RuntimeError(f"{name} did not converge.")
    return result


def frozen_files(h: int) -> list[Path]:
    out = horizon_dir(h)
    files = [out / n for n in ("stage_b_manifest.json", "dev_pairs.pkl", "tune_pairs.pkl", "test_pairs.pkl",
                               "oof_meta_dev.npy", "meta_tune.npy", "meta_test.npy", "meta_dnn.keras",
                               "meta_dnn_tune_probabilities.npy", "meta_dnn_test_probabilities.npy",
                               "lr_comparators.joblib", "v4_recalibration.joblib", "comparator_test_probabilities.npz",
                               "thresholds_all_methods.csv", "best_base_learner.json")]
    return files + sorted((out / "base_fold_bundles").glob("*.joblib"))


def run_stage_e(h: int) -> None:
    print("\n" + "=" * 96 + f"\nSTAGE (e) h={h}: THRESHOLDS, COMPARATORS, BEST BASE (TUNING DATA) + FREEZE\n" + "=" * 96)
    if not SMOKE_TEST and not SAP_PATH:
        raise RuntimeError("Set MCE_FP_SAP_PATH to SAP_v5.md: the freeze must record the SAP hash.")
    if (horizon_dir(h) / "stage_f_manifest.json").exists():
        raise RuntimeError(f"h={h}: the test data were already evaluated; re-freezing is not allowed.")
    dev, val, test, manifest, oof, meta_val, meta_test = load_meta(h)
    out = horizon_dir(h)
    d_manifest = load_json(out / "stage_d_manifest.json")
    if d_manifest["fingerprint_dev"] != manifest["fingerprint_dev"]:
        raise ValueError("Stage-(d) artefacts belong to a different pair table.")
    y_val = val["y"].to_numpy(np.int64)
    s_val, s_test = val["status_t"].to_numpy(np.int64), test["status_t"].to_numpy(np.int64)
    model = tf.keras.models.load_model(out / "meta_dnn.keras", compile=False)
    p_val_raw = validate_probability_matrix(model.predict(meta_val, verbose=0), len(val), N_CLASSES, "Meta-DNN tuning")
    if not np.allclose(p_val_raw, np.load(out / "meta_dnn_tune_probabilities.npy"), atol=NUMERIC_TOLERANCE):
        raise RuntimeError("Meta-DNN tuning probabilities are not reproducible.")

    probs_tune = {"mce": apply_mask(p_val_raw, s_val)}
    probs_test = {"mce": apply_mask(np.load(out / "meta_dnn_test_probabilities.npy"), s_test)}
    for b, name in enumerate(MODEL_ORDER):
        probs_tune[name] = apply_mask(base_block(meta_val, b), s_val)
        probs_test[name] = apply_mask(base_block(meta_test, b), s_test)
    v4, v4_info = load_v4_mapped(h, {"tune": val, "test": test})
    probs_tune["v4_mapped"], probs_test["v4_mapped"] = v4["tune"], v4["test"]
    probs_test["v4_mapped_recal"], recal_model = recalibrate_multinomial(v4["tune"], y_val, s_val, v4["test"], s_test)
    joblib.dump({"model": recal_model, "features": "log(clip(p_mapped, 1e-6, 1))", "fit_scope": "tuning data only",
                 "C": RECALIBRATION_C}, out / "v4_recalibration.joblib", compress=3)
    lr = fit_lr_comparators(dev, {"tune": val, "test": test}, out)
    for name in ("lr_simple", "lr_33"):
        probs_tune[name], probs_test[name] = lr[name]["tune"], lr[name]["test"]

    # best base learner: macro-F1 (argmax, masked) on tuning data; ties by MODEL_ORDER (as v4)
    base_rows = [{"model": name, "order": b, "tuning_macro_f1_argmax": macro_f1_present(y_val, probs_tune[name].argmax(1))}
                 for b, name in enumerate(MODEL_ORDER)]
    base_table = pd.DataFrame(base_rows)
    best = base_table.sort_values(["tuning_macro_f1_argmax", "order"], ascending=[False, True]).iloc[0]["model"]
    base_table.to_csv(out / "base_learner_tuning_selection.csv", index=False)
    probs_tune["best_base"], probs_test["best_base"] = probs_tune[best], probs_test[best]
    save_json({"h": h, "best_base_learner": best, "criterion": "three-class macro-F1 (argmax, masked) on tuning data",
               "tie_order": MODEL_ORDER, "selected_before_test_evaluation": True}, out / "best_base_learner.json")

    # ILCADT thresholds on masked tuning probabilities for every decision method
    threshold_rows = []
    for name in ["mce", "v4_mapped", "lr_simple", "lr_33"] + MODEL_ORDER:
        tau, table = learn_independent_thresholds(y_val, probs_tune[name])
        threshold_rows += [{"method": name, **r} for r in table.to_dict("records")]
    pd.DataFrame(threshold_rows).to_csv(out / "thresholds_all_methods.csv", index=False)
    np.savez_compressed(out / "comparator_test_probabilities.npz", **{k: v for k, v in probs_test.items() if k not in MODEL_ORDER + ["mce", "best_base"]})
    mce_tau = pd.DataFrame(threshold_rows).query("method == 'mce'").set_index("class").loc[LABELS, "threshold"].to_numpy(float)

    save_json({"h": h, "fingerprint_dev": manifest["fingerprint_dev"], "fingerprint_val": manifest["fingerprint_val"],
               "mce_thresholds": dict(zip(LABELS, mce_tau.tolist())), "best_base_learner": best, **v4_info,
               "impossible_class_mask": "applied to probabilities before tuning/metrics and to final ILCADT scores",
               "lr_comparators": {"penalty": "l2", "C": LR_C, "solver": "lbfgs", "spline_knot_quantiles": SPLINE_QUANTILES},
               "v4_recalibration": {"data": "tuning only", "C": RECALIBRATION_C, "use": "probability metrics and DCA"},
               "test_labels_used": False}, out / "stage_e_manifest.json")

    entry = {"h": h, "frozen_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds"),
             **{k: v for k, v in run_config().items() if k != "feature_order"},
             "code_sha256": sha256_file(Path(__file__)), "master_sha256": manifest["master_sha256"],
             "sap_sha256_expected": SAP_SHA256, "sap_sha256_found": sha256_file(Path(SAP_PATH)) if SAP_PATH else "not supplied",
             "mce_thresholds": dict(zip(LABELS, mce_tau.tolist())), "best_base_learner": best,
             "selected_epoch": d_manifest["selected_epoch"], **v4_info,
             "files_sha256": {str(p.relative_to(out)): sha256_file(p) for p in frozen_files(h)}}
    if SAP_PATH and entry["sap_sha256_found"] != SAP_SHA256:
        raise ValueError("SAP_v5.md differs from the frozen SAP.")
    save_json(entry, out / "freeze_manifest.json")
    FREEZE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FREEZE_LOG.open("a", encoding="utf-8") as log:
        log.write(f"{entry['frozen_at']} | {ANALYSIS} | h={h} | code {entry['code_sha256'][:16]} | master "
                  f"{entry['master_sha256'][:16]} | tau {np.round(mce_tau, 2).tolist()} | best base {best} | "
                  f"epoch {entry['selected_epoch']} | freeze_manifest {sha256_file(out / 'freeze_manifest.json')[:16]}\n")
    print(f"✓ Stage (e) h={h}: MCE thresholds {np.round(mce_tau, 2).tolist()}, best base {best}; frozen")


# =============================================================================
# STAGE (f): FROZEN TEST EVALUATION
# =============================================================================


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, minority: Iterable[int] | None = None) -> dict[str, Any]:
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
            **{f"ece_{LABELS[k]}": float(classwise[k]) for k in range(N_CLASSES)}}


SCALAR_KEYS = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
               "minority_macro_recall", "minimum_minority_recall"]


def child_members(children: np.ndarray) -> list[np.ndarray]:
    unique, inverse = np.unique(children, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(unique)))
    return [order[s:e] for s, e in zip(starts, np.append(starts[1:], len(order)))]


def cluster_bootstrap(children, y, decisions, minority, comparisons, replicates, seed):
    rng = np.random.default_rng(seed)
    members = child_members(children)
    values = {m: {k: [] for k in SCALAR_KEYS} for m in decisions}
    deltas = {c: {k: [] for k in SCALAR_KEYS} for c in comparisons}
    for _ in range(replicates):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), size=len(members))])
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


def ranking_at_workload(y: np.ndarray, score: np.ndarray, fraction: float) -> tuple[float, float]:
    """Sensitivity and PPV when the top `fraction` of children by score are monitored (stable sort; ties by order)."""
    n_flag = int(np.ceil(fraction * len(y)))
    order = np.argsort(-score, kind="stable")[:n_flag]
    tp = y[order].sum()
    return float(tp / max(1, y.sum())), float(tp / max(1, n_flag))


def net_benefit(y: np.ndarray, score: np.ndarray, pt: float) -> float:
    n = len(y); flag = score >= pt
    tp = np.sum(flag & (y == 1)); fp = np.sum(flag & (y == 0))
    return float(tp / n - fp / n * pt / (1 - pt))


def weak_calibration(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> dict:
    import statsmodels.api as sm
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    out = {}
    try:
        m0 = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit(cov_type="cluster", cov_kwds={"groups": groups})
        a, se = float(m0.params[0]), float(m0.bse[0])
        out.update(citl=a, citl_ci_low=a - 1.96 * se, citl_ci_high=a + 1.96 * se)
    except Exception as exc:
        out.update(citl=np.nan, citl_ci_low=np.nan, citl_ci_high=np.nan, citl_error=str(exc)[:120])
    try:
        m1 = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
        b, se = float(m1.params[1]), float(m1.bse[1])
        out.update(slope=b, slope_ci_low=b - 1.96 * se, slope_ci_high=b + 1.96 * se)
    except Exception as exc:
        out.update(slope=np.nan, slope_ci_low=np.nan, slope_ci_high=np.nan, slope_error=str(exc)[:120])
    return out


def calibration_curve(y: np.ndarray, p: np.ndarray, groups: np.ndarray, method: str) -> list[dict]:
    """Flexible calibration curve: logistic regression on a natural cubic spline (3 df) of logit(p), cluster-robust CI,
    evaluated between the 1st and 99th percentiles of p (same specification as calibration_per_class.py, v3/v4)."""
    import patsy
    import statsmodels.api as sm
    eps = 1e-4
    logit = lambda v: np.log(np.clip(v, eps, 1 - eps) / (1 - np.clip(v, eps, 1 - eps)))
    try:
        design = patsy.dmatrix("cr(x, df=3, constraints='center')", {"x": logit(p)}, return_type="dataframe")
        with np.errstate(all="ignore"):
            fit = sm.GLM(y, design, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
        lo, hi = np.quantile(p, [0.01, 0.99])
        grid = np.linspace(max(lo, eps), min(hi, 1 - eps), 60)
        gx = patsy.build_design_matrices([design.design_info], {"x": logit(grid)}, return_type="dataframe")[0]
        pred = fit.get_prediction(gx).summary_frame(alpha=0.05)
        return [{"method": method, "predicted": float(g), "observed_spline": float(a), "ci_low": float(l), "ci_high": float(u)}
                for g, a, l, u in zip(grid, pred["mean"], pred["mean_ci_lower"], pred["mean_ci_upper"])]
    except (np.linalg.LinAlgError, ValueError) as exc:
        return [{"method": method, "error": f"curve not estimable: {type(exc).__name__}"}]


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
        raise RuntimeError(f"h={h}: freeze manifest is not recorded in FREEZE_LOG_v5.")
    return entry


def run_stage_f(h: int) -> None:
    print("\n" + "=" * 96 + f"\nSTAGE (f) h={h}: FROZEN TEST EVALUATION\n" + "=" * 96)
    freeze = verify_freeze(h)
    dev, val, test, manifest, oof, meta_val, meta_test = load_meta(h)
    out = horizon_dir(h)
    y = test["y"].to_numpy(np.int64)
    s = test["status_t"].to_numpy(np.int64)
    children = test["child_code"].to_numpy()
    best = load_json(out / "best_base_learner.json")["best_base_learner"]
    tau_table = pd.read_csv(out / "thresholds_all_methods.csv")
    tau = lambda m: tau_table.query("method == @m").set_index("class").loc[LABELS, "threshold"].to_numpy(float)
    comp = dict(np.load(out / "comparator_test_probabilities.npz"))

    probs = {"mce": apply_mask(np.load(out / "meta_dnn_test_probabilities.npy"), s)}
    for b, name in enumerate(MODEL_ORDER):
        probs[name] = apply_mask(base_block(meta_test, b), s)
    probs.update(comp)
    probs["best_base"] = probs[best]
    for name, p in probs.items():
        validate_probability_matrix(p, len(test), N_CLASSES, name)

    decisions = {"persistence": np.full(len(test), STABLE, dtype=np.int64)}
    for name in ["mce", "v4_mapped", "lr_simple", "lr_33"] + MODEL_ORDER:
        decisions[f"{name}_argmax"] = probs[name].argmax(1)
        decisions[f"{name}_ilcadt"] = apply_threshold_decision(probs[name], tau(name), s)[1]
    decisions["best_base_argmax"], decisions["best_base_ilcadt"] = decisions[f"{best}_argmax"], decisions[f"{best}_ilcadt"]
    if ((s == 0) & np.any([d == IMPROVEMENT for d in decisions.values()], axis=0)).any() or \
       ((s == 2) & np.any([d == WORSENING for d in decisions.values()], axis=0)).any():
        raise RuntimeError("An impossible class was predicted.")
    order = list(decisions)
    np.save(out / "test_decisions.npy", np.stack([decisions[m] for m in order], axis=1))
    save_json({"columns": order}, out / "test_decisions_columns.json")
    rng_seed = RANDOM_STATE

    # ---- PRIMARY: confirmed new stunting among children normal at t (AUPRC) ----------
    normal = s == 0
    yb = (y[normal] == WORSENING).astype(int)
    ch_n = children[normal]
    prim_methods = {"persistence": np.zeros(normal.sum())}
    prim_methods.update({m: probs[m][normal, WORSENING] for m in PROB_METHODS})
    prim_rows = []
    for m, score in prim_methods.items():
        row = {"method": m, "n": int(normal.sum()), "events": int(yb.sum()), "incidence": float(yb.mean()),
               "auprc_average_precision": float(average_precision_score(yb, score)),
               "auc": float(roc_auc_score(yb, score)) if m != "persistence" else 0.5}
        for w in WORKLOADS:
            sens, ppv = ranking_at_workload(yb, score, w) if m != "persistence" else (0.0, np.nan)
            row[f"sensitivity_top{int(w * 100)}pct"], row[f"ppv_top{int(w * 100)}pct"] = sens, ppv
        prim_rows.append(row)
    pd.DataFrame(prim_rows).to_csv(out / "primary_endpoint_metrics.csv", index=False)

    members = child_members(ch_n)
    rng = np.random.default_rng(rng_seed)
    boot = {m: [] for m in prim_methods if m != "persistence"}
    for _ in range(BOOTSTRAP_REPLICATES):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), size=len(members))])
        if yb[idx].sum() == 0:
            for m in boot:
                boot[m].append(np.nan)
            continue
        for m in boot:
            boot[m].append(average_precision_score(yb[idx], prim_methods[m][idx]))
    boot = {m: np.asarray(v, float) for m, v in boot.items()}
    ci_rows, diff_rows = [], []
    q = lambda a: (float(np.nanquantile(a, 0.025)), float(np.nanquantile(a, 0.975)), int(np.isfinite(a).sum()))
    for m, arr in boot.items():
        lo, hi, n_ok = q(arr)
        ci_rows.append({"method": m, "metric": "auprc", "ci_low": lo, "ci_high": hi, "valid_replicates": n_ok})
    point = {r["method"]: r["auprc_average_precision"] for r in prim_rows}
    for comp_name in PRIMARY_COMPARATORS + ["v4_mapped_recal", "best_base"]:
        lo, hi, n_ok = q(boot["mce"] - boot[comp_name])
        diff_rows.append({"comparison": f"mce minus {comp_name}", "metric": "auprc", "difference": point["mce"] - point[comp_name],
                          "ci_low": lo, "ci_high": hi, "valid_replicates": n_ok, "primary_claim_comparator": comp_name in PRIMARY_COMPARATORS})
    pd.DataFrame(ci_rows).to_csv(out / "primary_bootstrap_ci.csv", index=False)
    diffs = pd.DataFrame(diff_rows)
    diffs.to_csv(out / "primary_paired_differences.csv", index=False)
    claim = diffs[diffs["primary_claim_comparator"]]
    save_json({"h": h, "rule": "CI 95% of AUPRC(MCE) minus each of v4_mapped, lr_simple, lr_33 entirely above 0, at h = 1, 3, 6, on confirmatory data",
               "data_status": "exploratory (2026 test data already opened in v4)" if not SMOKE_TEST else "smoke test",
               "all_three_ci_above_zero_this_horizon": bool((claim["ci_low"] > 0).all()),
               "ci_low_by_comparator": dict(zip(claim["comparison"], claim["ci_low"]))}, out / "claim_check.json")

    # DCA (probabilities only) and calibration for the primary endpoint
    dca_rows = []
    for pt in DCA_THRESHOLDS:
        dca_rows.append({"threshold": pt, "method": "treat_all", "net_benefit": float(yb.mean() - (1 - yb.mean()) * pt / (1 - pt))})
        dca_rows.append({"threshold": pt, "method": "treat_none", "net_benefit": 0.0})
        for m in ["mce", "v4_mapped_recal", "lr_simple", "lr_33", "best_base"]:
            dca_rows.append({"threshold": pt, "method": m, "net_benefit": net_benefit(yb, prim_methods[m], pt)})
    pd.DataFrame(dca_rows).to_csv(out / "primary_decision_curve.csv", index=False)
    groups = pd.factorize(ch_n)[0]
    cal_rows = [{"method": m, "events": int(yb.sum()), "non_events": int((1 - yb).sum()),
                 "mean_predicted": float(prim_methods[m].mean()), "observed": float(yb.mean()),
                 "curve_eligible_200_200": bool(yb.sum() >= 200 and (1 - yb).sum() >= 200),
                 **weak_calibration(yb, prim_methods[m], groups)}
                for m in ["mce", "v4_mapped_recal", "lr_simple", "lr_33", "best_base"]]
    pd.DataFrame(cal_rows).to_csv(out / "primary_calibration.csv", index=False)
    curve_rows = []
    if yb.sum() >= 200 and (1 - yb).sum() >= 200:  # Van Calster et al. 2019
        for m in ["mce", "v4_mapped_recal", "lr_simple", "lr_33", "best_base"]:
            curve_rows += calibration_curve(yb, prim_methods[m], groups, m)
    pd.DataFrame(curve_rows).to_csv(out / "primary_calibration_curve.csv", index=False)

    # subgroups (descriptive): sex, age band, puskesmas
    sub_rows = []
    tn = test.loc[normal].reset_index(drop=True)
    groups_def = {"sex": np.where(tn["sex_male"] == 1, "male", "female"),
                  "age_band": np.where(tn["age_months_t"] < AGE_BAND_MONTHS, f"<{AGE_BAND_MONTHS} mo", f">={AGE_BAND_MONTHS} mo"),
                  "puskesmas": tn["puskesmas"].to_numpy()}
    for gname, gval in groups_def.items():
        for level in sorted(set(gval)):
            mask = gval == level
            row = {"subgroup": gname, "level": level, "n": int(mask.sum()), "events": int(yb[mask].sum())}
            for m in ["mce"] + PRIMARY_COMPARATORS:
                row[f"auprc_{m}"] = float(average_precision_score(yb[mask], prim_methods[m][mask])) if yb[mask].sum() else np.nan
            sub_rows.append(row)
    pd.DataFrame(sub_rows).to_csv(out / "primary_subgroups.csv", index=False)

    # ---- SECONDARY: three-class metrics, strata, probability metrics -----------------
    minority = [WORSENING, IMPROVEMENT]
    subsets = {"all": np.ones(len(test), bool), "status_t=stunted": s == 1, "status_t=severely_stunted": s == 2,
               "status_t=normal": s == 0, "new_children": test["new_child"].to_numpy(bool)}
    rows, per_class_rows = [], []
    for sname, mask in subsets.items():
        if not mask.any():
            continue
        for method in order:
            mm = classification_metrics(y[mask], decisions[method][mask], minority)
            rows.append({"subset": sname, "method": method, "n": mm["n"], "classes_present": mm["classes_present"],
                         **{k: mm[k] for k in SCALAR_KEYS}, "confusion_matrix": mm["confusion_matrix"].tolist()})
            for k, label in enumerate(LABELS):
                per_class_rows.append({"subset": sname, "method": method, "class": label, "support": int(mm["support_per_class"][k]),
                                       "precision": mm["precision_per_class"][k], "recall": mm["recall_per_class"][k], "f1": mm["f1_per_class"][k]})
    pd.DataFrame(rows).to_csv(out / "threeclass_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(out / "threeclass_per_class.csv", index=False)
    pd.DataFrame([{"method": m, **probability_metrics(y, probs[m])} for m in PROB_METHODS + MODEL_ORDER]).to_csv(
        out / "threeclass_probability_metrics.csv", index=False)
    main = ["persistence", "mce_ilcadt", "mce_argmax", "v4_mapped_ilcadt", "lr_simple_ilcadt", "lr_33_ilcadt", "best_base_argmax", "best_base_ilcadt"]
    comparisons = [("v4_mapped_ilcadt", "mce_ilcadt"), ("lr_simple_ilcadt", "mce_ilcadt"), ("lr_33_ilcadt", "mce_ilcadt"),
                   ("mce_argmax", "mce_ilcadt"), ("best_base_argmax", "best_base_ilcadt")]
    ci, dl = cluster_bootstrap(children, y, {k: decisions[k] for k in main}, minority, comparisons, BOOTSTRAP_REPLICATES, RANDOM_STATE)
    ci.to_csv(out / "threeclass_bootstrap_ci.csv", index=False)
    dl.to_csv(out / "threeclass_paired_differences.csv", index=False)

    save_json({"h": h, **{k: v for k, v in run_config().items() if k != "feature_order"},
               "fingerprint_test": manifest["fingerprint_test"], "freeze_verified": True, "frozen_at": freeze["frozen_at"],
               "best_base_learner": best, "bootstrap_unit": "child", "bootstrap_replicates": BOOTSTRAP_REPLICATES,
               "primary_endpoint": "confirmed new stunting among children normal at t; AUPRC = average precision",
               "persistence_note": "persistence never predicts change: AUPRC = incidence; precision of never-predicted classes = 0",
               "calibration_and_dca_source": "probabilities only (never ILCADT scores)",
               "data_status": "exploratory" if not SMOKE_TEST else "smoke"}, out / "stage_f_manifest.json")
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
    summarise_claims()
    print("\nMCE-ILCADT-FP v5 finished. Artefacts:", BASE_DIR)


def summarise_claims() -> None:
    """SAP v5 7.4: combine per-horizon claim checks (h = 1, 3, 6) into one verdict."""
    rows = {}
    for h in (1, 3, 6):
        path = horizon_dir(h) / "claim_check.json"
        if path.exists():
            rows[h] = load_json(path)
    if not rows:
        return
    complete = set(rows) == {1, 3, 6}
    consistent = complete and all(r["all_three_ci_above_zero_this_horizon"] for r in rows.values())
    save_json({"horizons_available": sorted(rows), "complete": complete,
               "consistent_at_h1_h3_h6": consistent if complete else None,
               "data_status": "exploratory (no contribution claim; SAP v5 7.4)" if not SMOKE_TEST else "smoke",
               "per_horizon": {str(h): r["all_three_ci_above_zero_this_horizon"] for h, r in rows.items()}},
              SHARED_DIR / "claim_summary.json")


if __name__ == "__main__":
    main()
