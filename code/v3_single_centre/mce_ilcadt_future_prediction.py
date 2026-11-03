# -*- coding: utf-8 -*-
"""
MCE-ILCADT-FP: Meta-Confidence Ensemble with Independently Learned Class-Aware
Decision Thresholds for FUTURE (t -> t+h) toddler nutritional-status prediction.

Derived from: mce_ilcadt_220826_paper_1-4.py  (Stages a-f, IFLS5 cross-sectional)

WHAT CHANGED VERSUS THE CROSS-SECTIONAL PIPELINE
------------------------------------------------
* Unit of analysis : child-month PAIR (i, t) -> (i, t+h), not a single row.
* Target           : TB/U (HAZ) class observed at calendar month t+h.
* Predictors       : only information recorded up to and including month t.
* Classes          : derived from data (normal / stunted / severely stunted);
                     "tall" (HAZ > +3) is excluded because n = 4 observations.
* Grouping         : child identity (pseudonymised), replacing feature-identical
                     duplicate groups.  Mixed future labels for identical
                     feature vectors are legitimate and are NOT an error.
* Split            : temporal development / held-out boundary on calendar month,
                     child-grouped OOF folds and child-grouped meta subsets.
* Evaluation       : + persistence baseline, + OvR ROC-AUC, + child-level
                     cluster bootstrap, explicit handling of undefined metrics.
* Provenance       : no run-ID dependency; stages verify a content fingerprint
                     of the pair table, class order, feature order and shapes.

STAGES (per horizon h)
    a  environment, seeds, contracts
    b  load all monthly e-PPGBM exports (one file per month), validate identity/time, clean,
       derive labels, build t-only features, form pairs, temporal split
    c  six base learners, child-grouped strict OOF -> B*K meta-features
    d  Meta-DNN epoch selection and fresh refit (threshold-dev excluded)
    e  class-specific thresholds learned on threshold-development only
    f  frozen held-out evaluation

Run (Colab / local):
    pip install lightgbm==4.6.0 xgboost==3.1.2 catboost==1.2.8 tensorflow lxml
    MCE_FP_DATA_DIR=/path/to/xls_folder MCE_FP_HORIZONS=1,3,6 \
        python mce_ilcadt_future_prediction.py

Privacy: the raw exports contain NIK, child and parent names and addresses.
Stage (b) drops these columns and keeps only a salted SHA-256 pseudonym.
Set MCE_FP_ID_SALT to a secret value and never publish it.
"""
from __future__ import annotations

import os

RANDOM_STATE = int(os.environ.get("MCE_FP_RANDOM_STATE", "42"))
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import hashlib
import json
import random
import re
import secrets
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
from lxml import html as lxml_html
from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
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

EXPERIMENT_NAME = "MCE_ILCADT_FUTURE_PREDICTION_BUTUR"
TIMEZONE = "Asia/Jakarta"

DATA_DIR = Path(os.environ.get("MCE_FP_DATA_DIR", "/content/buturk_dataset")).resolve()
BASE_DIR = Path(os.environ.get("MCE_FP_BASE_DIR", "/content/MCE_ILCADT_FP")).resolve()
FILE_GLOB = "Daftar-Anak-Berdasarkan-Status-Gizi-*.xls"

HORIZONS = tuple(
    int(item) for item in os.environ.get("MCE_FP_HORIZONS", "1,3,6").split(",") if item.strip()
)
RUN_STAGES = tuple(
    item.strip().lower()
    for item in os.environ.get("MCE_FP_STAGES", "a,b,c,d,e,f").split(",")
    if item.strip()
)

# Three temporal blocks (calendar months); every boundary is strict:
#   development            : target month <= DEV_TARGET_MONTH_MAX
#   internal temporal val. : anchor >= TEMPORAL_VAL_ANCHOR_MIN and target <= TEMPORAL_VAL_TARGET_MAX
#   real-world temporal test: anchor >= TEST_ANCHOR_MONTH_MIN
# Pairs that cross a boundary are dropped. September 2025 is absent from the
# exports and separates validation targets from test anchors.
# Primary design ("temporal_threshold", h = 1 and 3): thresholds are learned on
# the temporal validation block. When that block cannot hold h-month pairs with
# every class (h = 6), the horizon falls back to the secondary design
# ("child_grouped_threshold"): development = target <= TEMPORAL_VAL_TARGET_MAX
# and thresholds are learned on 20% of development children.
DEV_TARGET_MONTH_MAX = os.environ.get("MCE_FP_DEV_TARGET_MAX", "2025-02")
TEMPORAL_VAL_ANCHOR_MIN = os.environ.get("MCE_FP_VAL_ANCHOR_MIN", "2025-03")
TEMPORAL_VAL_TARGET_MAX = os.environ.get("MCE_FP_VAL_TARGET_MAX", "2025-08")
TEST_ANCHOR_MONTH_MIN = os.environ.get("MCE_FP_TEST_ANCHOR_MIN", "2025-10")
MIN_CHILDREN_RAREST_CLASS_IN_VAL = 10
# Optional last test anchor month (rolling-origin analysis uses fixed-length test windows).
TEST_ANCHOR_MONTH_MAX = os.environ.get("MCE_FP_TEST_ANCHOR_MAX", "")
# Leave-one-village-out: integer village code held out as the ONLY test village;
# development and validation then use the other villages. Empty = all villages.
HOLDOUT_VILLAGE = os.environ.get("MCE_FP_HOLDOUT_VILLAGE", "")
# Feature ablation: "none" (primary), "no_current_status" (drop status_t, haz_t),
# "no_status_history" (also drop every HAZ-derived history feature).
ABLATION = os.environ.get("MCE_FP_ABLATION", "none").lower()
ABLATION_DROP = {
    "none": [],
    "no_current_status": ["status_t", "haz_t"],
    "no_status_history": ["status_t", "haz_t", "d_haz_1", "d_haz_3", "min_haz_to_t", "n_prior_stunted_months"],
}

# Label source: "eppgbm" (official e-PPGBM TB/U category, primary) or "who"
# (category from HAZ recomputed with WHO 2006 LMS tables, sensitivity analysis).
LABEL_SOURCE = os.environ.get("MCE_FP_LABEL_SOURCE", "eppgbm").lower()
WHO_TABLE_DIR = os.environ.get("MCE_FP_WHO_TABLE_DIR", "")

N_SPLITS = int(os.environ.get("MCE_FP_N_SPLITS", "5"))
THRESHOLD_DEVELOPMENT_CHILD_FRACTION = 0.20
META_VALIDATION_CHILD_FRACTION_WITHIN_POOL = 0.10
MIN_CHILDREN_PER_CLASS_PER_FOLD = 2

INCLUDE_PROGRAM_VARIABLES = os.environ.get("MCE_FP_INCLUDE_PROGRAM_VARS", "0") == "1"
SMOKE_TEST = os.environ.get("MCE_FP_SMOKE", "0") == "1"

# --- Label definition: WHO 2006 HAZ, Permenkes RI No. 2/2020 categories ------
# Boundary convention verified on the exports: HAZ = -2.00 -> "Normal",
# HAZ = -3.00 -> "Pendek".
LABELS = ["normal", "stunted", "severely stunted"]
N_CLASSES = len(LABELS)
LABEL_TO_IDX = {label: index for index, label in enumerate(LABELS)}
TBU_CATEGORY_TO_LABEL = {
    "Normal": "normal",
    "Pendek": "stunted",
    "Sangat Pendek": "severely stunted",
    "Tinggi": "tall",
}
EXCLUDED_LABELS = ("tall",)

# --- WHO biologically implausible z-score flags (WHO Anthro) ------------------
HAZ_VALID = (-6.0, 6.0)
WAZ_VALID = (-6.0, 5.0)
WHZ_VALID = (-5.0, 5.0)
# --- Pragmatic plausibility bounds (to be confirmed with the data owner) -----
BIRTH_WEIGHT_KG_VALID = (0.5, 6.0)
BIRTH_LENGTH_CM_VALID = (30.0, 60.0)
WEIGHT_KG_VALID = (1.0, 40.0)
HEIGHT_CM_VALID = (40.0, 125.0)
MUAC_CM_VALID = (7.0, 30.0)  # 0 is recorded when MUAC was not measured
MAX_AGE_MONTHS = 60

RAW_COLUMNS = [
    "No", "NIK", "Nama", "JK", "Tgl Lahir", "BB Lahir", "TB Lahir", "Nama Ortu",
    "Prov", "Kab/Kota", "Kec", "Pukesmas", "Desa/Kel", "Posyandu", "RT", "RW",
    "Alamat", "Usia Saat Ukur", "Tanggal Pengukuran", "Berat", "Tinggi",
    "Cara Ukur", "LiLA", "BB/U", "ZS BB/U", "TB/U", "ZS TB/U", "BB/TB",
    "ZS BB/TB", "Naik Berat Badan", "Jml Vit A", "KPSP", "KIA",
    "Kelas Ibu Balita", "MBG", "Detail",
]
PII_COLUMNS = ["NIK", "Nama", "Nama Ortu", "Alamat", "RT", "RW", "Detail"]
STATIC_COLUMNS = ["JK", "Tgl Lahir", "BB Lahir", "TB Lahir"]

WEIGHT_GAIN_CODES = ["N", "T", "O", "B"]  # e-PPGBM "Naik Berat Badan" codes

BASE_FEATURES = [
    # static, known from birth
    "sex_male", "birth_weight_kg", "birth_length_cm",
    # measured at month t
    "age_months_t", "weight_kg_t", "height_cm_t", "recumbent_t",
    "muac_cm_t", "muac_observed_t",
    "haz_t", "waz_t", "whz_t", "status_t",
    "wg_N_t", "wg_T_t", "wg_O_t", "wg_B_t",
    # history up to month t (calendar-month aligned, never after t)
    "haz_lag1_observed", "d_haz_1", "d_waz_1", "d_height_1", "d_weight_1",
    "haz_lag3_observed", "d_haz_3",
    "n_obs_to_t", "months_since_prev_obs", "n_prior_stunted_months", "min_haz_to_t",
]
PROGRAM_FEATURES = [
    "vita_t", "kpsp_sesuai_t", "kpsp_meragukan_t", "kpsp_penyimpangan_t",
    "kia_sesuai_t", "kelas_ibu_ya_t", "kelas_ibu_tidak_t", "mbg_ya_t", "mbg_tidak_t",
]
if ABLATION not in ABLATION_DROP:
    raise ValueError(f"MCE_FP_ABLATION must be one of {sorted(ABLATION_DROP)}")
FEAT_COLS = [
    f for f in BASE_FEATURES + (PROGRAM_FEATURES if INCLUDE_PROGRAM_VARIABLES else [])
    if f not in ABLATION_DROP[ABLATION]
]
# status_t is always stored with each pair (persistence baseline, transition subset),
# even when it is not a predictor.
AUDIT_COLS = ["status_t"] if "status_t" not in FEAT_COLS else []

MODEL_ORDER = ["RF", "LGBM", "XGB", "CAT", "GBC", "ADA"]
META_DIMENSION = len(MODEL_ORDER) * N_CLASSES

# Meta-DNN (retained from the original code; not claimed optimal)
MAX_EPOCHS = 200 if not SMOKE_TEST else 8
BATCH_SIZE = 64
INITIAL_LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 12
LR_REDUCER_PATIENCE = 5
LR_REDUCER_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6
# Epoch selection for the Meta-DNN. Changed from the original "macro_f1" to
# "val_loss" BEFORE the held-out test was opened (11 Oct 2026): with only 6-8
# severely stunted children in meta-validation, macro-F1 peaked at epoch 1
# (h = 3, 6) while validation loss kept improving. Thresholds handle decisions.
EPOCH_SELECTION_METRIC = os.environ.get("MCE_FP_EPOCH_SELECTION", "val_loss")
if EPOCH_SELECTION_METRIC not in ("val_loss", "macro_f1"):
    raise ValueError("MCE_FP_EPOCH_SELECTION must be 'val_loss' or 'macro_f1'.")

THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01, dtype=np.float64), 2)
CONFIDENCE_EPSILON = 1e-7
RELIABILITY_BINS = 10
BOOTSTRAP_REPLICATES = int(os.environ.get("MCE_FP_BOOTSTRAP", "2000" if not SMOKE_TEST else "50"))
NUMERIC_TOLERANCE = 1e-6


def horizon_dir(h: int) -> Path:
    return BASE_DIR / f"h{h}" / "artifacts"


SHARED_DIR = BASE_DIR / "shared"
PRIVATE_DIR = BASE_DIR / "private_do_not_share"


# =============================================================================
# SHARED UTILITIES (retained / adapted from the original pipeline)
# =============================================================================


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False, default=_json_default)
    with path.open("r", encoding="utf-8") as file:
        json.load(file)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Period, pd.Timestamp)):
        return str(value)
    raise TypeError(f"Not JSON serialisable: {type(value)}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def month_index(period_text: str) -> int:
    period = pd.Period(period_text, freq="M")
    return int(period.year * 12 + period.month - 1)


def month_text(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def pair_fingerprint(child_codes: np.ndarray, anchor_month: np.ndarray, labels: np.ndarray, h: int) -> str:
    """Content fingerprint used instead of a run ID."""
    frame = np.stack(
        [np.asarray(child_codes, np.int64), np.asarray(anchor_month, np.int64), np.asarray(labels, np.int64)],
        axis=1,
    )
    digest = hashlib.sha256()
    digest.update(f"h={h};labels={LABELS};features={FEAT_COLS};models={MODEL_ORDER}".encode())
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


def fresh_model(model_template: Any) -> Any:
    try:
        return clone(model_template)
    except Exception:
        return deepcopy(model_template)


def prepare_model_input(processed_array: np.ndarray, feature_columns: list[str]) -> pd.DataFrame:
    array = np.asarray(processed_array, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(feature_columns):
        raise ValueError("Processed base-model input has an invalid shape.")
    if not np.isfinite(array).all():
        raise ValueError("Processed base-model input is not finite.")
    return pd.DataFrame(array, columns=feature_columns)


def predict_proba_aligned(fitted_model: Any, model_input: pd.DataFrame, expected_n_classes: int) -> np.ndarray:
    """Align probability columns to the fixed class indices 0..K-1."""
    raw = np.asarray(fitted_model.predict_proba(model_input), dtype=np.float64)
    classes = np.asarray(fitted_model.classes_, dtype=np.int64).reshape(-1)
    if raw.ndim != 2 or raw.shape[1] != len(classes):
        raise ValueError("predict_proba output is inconsistent with classes_.")
    aligned = np.zeros((len(model_input), expected_n_classes), dtype=np.float64)
    for source_column, class_index in enumerate(classes):
        if class_index not in range(expected_n_classes):
            raise ValueError(f"Unexpected fitted class index: {class_index}")
        aligned[:, class_index] = raw[:, source_column]
    row_sums = aligned.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Non-positive probability row sum.")
    aligned /= row_sums
    return validate_probability_matrix(aligned, len(model_input), expected_n_classes, "aligned probabilities").astype(np.float32)


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
        raise ValueError("Class order changed; downstream artifacts would be inconsistent.")
    if META_DIMENSION != len(MODEL_ORDER) * N_CLASSES:
        raise ValueError("META_DIMENSION must equal B x K.")
    if not (month_index(DEV_TARGET_MONTH_MAX) < month_index(TEMPORAL_VAL_ANCHOR_MIN)
            and month_index(TEMPORAL_VAL_TARGET_MAX) < month_index(TEST_ANCHOR_MONTH_MIN)
            and (not TEST_ANCHOR_MONTH_MAX or month_index(TEST_ANCHOR_MONTH_MAX) >= month_index(TEST_ANCHOR_MONTH_MIN))):
        raise ValueError("Temporal blocks must be ordered: dev targets < validation anchors; validation targets < test anchors.")
    if any(h < 1 for h in HORIZONS):
        raise ValueError("Horizons must be positive integers (months).")


def run_stage_a() -> None:
    print("\n" + "=" * 96 + "\nSTAGE (a): ENVIRONMENT, SEEDS, CONTRACT\n" + "=" * 96)
    validate_contract()
    for directory in (BASE_DIR, SHARED_DIR, PRIVATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
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
    save_json(
        {
            "experiment_name": EXPERIMENT_NAME,
            "task": "future nutritional-status prediction, X_{i,t} -> y_{i,t+h}",
            "created_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
            "random_state": RANDOM_STATE,
            "horizons_months": list(HORIZONS),
            "dev_target_month_max": DEV_TARGET_MONTH_MAX,
            "temporal_val_anchor_min": TEMPORAL_VAL_ANCHOR_MIN,
            "temporal_val_target_max": TEMPORAL_VAL_TARGET_MAX,
            "test_anchor_month_min": TEST_ANCHOR_MONTH_MIN,
            "label_source": LABEL_SOURCE,
            "test_anchor_month_max": TEST_ANCHOR_MONTH_MAX or None,
            "holdout_village": HOLDOUT_VILLAGE or None,
            "ablation": ABLATION,
            "n_splits": N_SPLITS,
            "class_order": LABELS,
            "excluded_labels": list(EXCLUDED_LABELS),
            "feature_order": FEAT_COLS,
            "program_variables_included": INCLUDE_PROGRAM_VARIABLES,
            "base_model_order": MODEL_ORDER,
            "meta_dimension": META_DIMENSION,
            "smoke_test": SMOKE_TEST,
            "versions": {
                "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                "xgboost": xgboost.__version__, "catboost": catboost.__version__,
                "tensorflow": tf.__version__,
            },
            "tensorflow_determinism": determinism,
        },
        SHARED_DIR / "environment_metadata.json",
    )
    print(f"  Horizons: {HORIZONS} | classes: {LABELS} | features: {len(FEAT_COLS)} | meta dim: {META_DIMENSION}")
    print("✓ Stage (a) completed.")


# =============================================================================
# STAGE (b1): LOADING AND LONGITUDINAL VALIDATION
# =============================================================================


def parse_export_file(path: Path) -> pd.DataFrame:
    """Parse one e-PPGBM export (HTML table saved with .xls extension)."""
    text = path.read_bytes().decode("utf-8", errors="replace")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)  # removes the commented-out header
    document = lxml_html.fromstring(text)
    tables = document.xpath("//table")
    if len(tables) != 1:
        raise ValueError(f"{path.name}: expected exactly one table, found {len(tables)}")
    headers = [th.text_content().strip() for th in tables[0].xpath(".//thead//th")]
    if headers != RAW_COLUMNS:
        raise ValueError(f"{path.name}: column contract differs: {headers}")
    rows, malformed = [], 0
    for tr in tables[0].xpath(".//tbody/tr"):
        values = [td.text_content().replace("\xa0", " ").strip() for td in tr.xpath("./td")]
        if len(values) != len(headers):
            malformed += 1
            continue
        record = dict(zip(headers, values))
        links = " ".join(tr.xpath(".//a/@href") + tr.xpath(".//a/@onclick"))
        linked_ids = set(re.findall(r"/Balita/(?:perkembangan|ukur)/([^./]+)\.html", links))
        linked_ids |= set(re.findall(r"view_penerima\('([^']+)'\)", links))
        record["_id_link_consistent"] = linked_ids == {record["NIK"].strip()}
        rows.append(record)
    if malformed:
        raise ValueError(f"{path.name}: {malformed} malformed rows")
    caption = tables[0].xpath(".//caption")
    frame = pd.DataFrame(rows)
    frame["source_file"] = path.name
    frame["export_caption"] = caption[0].text_content().strip() if caption else ""
    return frame


def get_id_salt() -> str:
    salt = os.environ.get("MCE_FP_ID_SALT")
    if salt:
        return salt
    salt_path = PRIVATE_DIR / "id_salt.txt"
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    if not salt_path.exists():
        salt_path.write_text(secrets.token_hex(32), encoding="utf-8")
    return salt_path.read_text(encoding="utf-8").strip()


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")


def bounded(series: pd.Series, bounds: tuple[float, float]) -> tuple[pd.Series, int]:
    invalid = series.notna() & ~series.between(*bounds)
    return series.mask(invalid), int(invalid.sum())


VILLAGE_NAMES: dict[int, str] = {}


def resolve_child_month_duplicates(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Identical records of one child-month -> keep one; any conflicting content -> drop that child-month."""
    keys = ["NIK", "month"]
    content = [c for c in raw.columns if c not in ("No", "source_file", "export_caption")]
    duplicated_rows = int(raw.duplicated(keys, keep=False).sum())
    deduplicated = raw.drop_duplicates(subset=content, keep="first")
    conflict = deduplicated.duplicated(keys, keep=False)
    conflicting_keys = deduplicated.loc[conflict, keys].drop_duplicates()
    cleaned = deduplicated.loc[~conflict]
    return cleaned, {
        "duplicate_rows_found": duplicated_rows,
        "identical_rows_removed": int(len(raw) - len(deduplicated)),
        "conflicting_child_months_dropped": int(len(conflicting_keys)),
        "rows_dropped_for_conflict": int(conflict.sum()),
    }


def load_and_validate_observations() -> tuple[pd.DataFrame, dict[str, Any]]:
    files = sorted(DATA_DIR.glob(FILE_GLOB))
    files = [path for path in files if not path.name.startswith("._")]
    if not files:
        raise FileNotFoundError(f"No export files under {DATA_DIR}")
    frames = [parse_export_file(path) for path in files]
    raw = pd.concat(frames, ignore_index=True)
    audit: dict[str, Any] = {"n_files": len(files), "n_raw_rows": int(len(raw))}

    # --- identity -------------------------------------------------------------
    raw["NIK"] = raw["NIK"].str.strip()
    if (raw["NIK"] == "").any() or (~raw["NIK"].str.fullmatch(r"\d+")).any():
        raise ValueError("Empty or non-numeric child identifiers detected.")
    if not raw["_id_link_consistent"].all():
        raise ValueError("Identifier cell disagrees with the identifier in the record links.")
    audit["identifier_length_counts"] = raw["NIK"].str.len().value_counts().sort_index().to_dict()

    # --- time -----------------------------------------------------------------
    measured = pd.to_datetime(raw["Tanggal Pengukuran"], errors="raise", format="%Y-%m-%d")
    raw["month"] = (measured.dt.year * 12 + measured.dt.month - 1).astype(np.int64)
    months_per_file = raw.groupby("source_file")["month"].nunique()
    if (months_per_file != 1).any():
        raise ValueError("At least one export file spans more than one calendar month.")
    file_month = raw.groupby("source_file")["month"].first()
    if file_month.duplicated().any():
        raise ValueError("Two export files cover the same calendar month.")
    all_months = np.arange(raw["month"].min(), raw["month"].max() + 1)
    missing_months = sorted(set(all_months) - set(raw["month"].unique()))
    audit["observed_months"] = [month_text(m) for m in sorted(raw["month"].unique())]
    audit["missing_calendar_months"] = [month_text(m) for m in missing_months]
    audit["rows_per_month"] = {month_text(k): int(v) for k, v in raw["month"].value_counts().sort_index().items()}

    # --- child-month duplicates ------------------------------------------------
    raw, audit["child_month_duplicates"] = resolve_child_month_duplicates(raw)
    # --- static attribute consistency -----------------------------------------
    inconsistent = {c: int((raw.groupby("NIK")[c].nunique(dropna=False) > 1).sum()) for c in STATIC_COLUMNS}
    audit["children_with_inconsistent_static_attributes"] = inconsistent
    bad_children = set()
    for column in STATIC_COLUMNS:
        counts = raw.groupby("NIK")[column].nunique(dropna=False)
        bad_children |= set(counts[counts > 1].index)
    raw = raw.loc[~raw["NIK"].isin(bad_children)].copy()
    audit["children_excluded_for_identity_inconsistency"] = len(bad_children)
    audit["constant_location_per_child"] = bool(
        (raw.groupby("NIK")[["Desa/Kel", "Posyandu"]].nunique() <= 1).all().all()
    )

    # --- pseudonymisation; PII never leaves this function ----------------------
    salt = get_id_salt()
    raw["child_pid"] = raw["NIK"].map(lambda v: hashlib.sha256((salt + v).encode()).hexdigest()[:20])
    if raw.groupby("child_pid")["NIK"].nunique().max() != 1:
        raise RuntimeError("Pseudonym collision.")
    audit["n_children"] = int(raw["child_pid"].nunique())
    # Salt-independent integer code (rank of the source identifier). It fixes the
    # child order used by every split, so folds and meta subsets are reproducible
    # for any salt. The identifier itself is dropped below.
    raw["child_code"] = raw["NIK"].rank(method="dense").astype(np.int64) - 1
    raw = raw.drop(columns=PII_COLUMNS + ["_id_link_consistent"])

    # --- numeric conversion and validity bounds --------------------------------
    obs = pd.DataFrame({"child_pid": raw["child_pid"], "child_code": raw["child_code"], "month": raw["month"]})
    # village code: never a predictor; kept for leave-one-village-out analysis
    village = raw["Desa/Kel"].astype("category")
    obs["village_code"] = village.cat.codes.astype(np.int64)
    VILLAGE_NAMES.update(dict(enumerate(village.cat.categories)))
    # age in days: only to check date validity and to recompute WHO HAZ
    age_days = (measured.loc[raw.index] - pd.to_datetime(raw["Tgl Lahir"], errors="coerce")).dt.days
    audit["measurement_before_birth_rows"] = int((age_days < 0).sum())
    audit["birth_date_unparsable_rows"] = int(age_days.isna().sum())  # excluded via eligibility
    obs["sex_male"] = raw["JK"].map({"L": 1.0, "P": 0.0})
    age = raw["Usia Saat Ukur"].str.extract(r"(\d+)\s*Tahun\s*-\s*(\d+)\s*Bulan")
    obs["age_months_t"] = age[0].astype(float) * 12 + age[1].astype(float)
    invalid_counts: dict[str, int] = {}
    for target, source, bounds in (
        ("birth_weight_kg", "BB Lahir", BIRTH_WEIGHT_KG_VALID),
        ("birth_length_cm", "TB Lahir", BIRTH_LENGTH_CM_VALID),
        ("weight_kg_t", "Berat", WEIGHT_KG_VALID),
        ("height_cm_t", "Tinggi", HEIGHT_CM_VALID),
        ("waz_t", "ZS BB/U", WAZ_VALID),
        ("whz_t", "ZS BB/TB", WHZ_VALID),
    ):
        obs[target], invalid_counts[target] = bounded(to_number(raw[source]), bounds)
    muac = to_number(raw["LiLA"])
    muac = muac.mask(muac == 0)
    obs["muac_cm_t"], invalid_counts["muac_cm_t"] = bounded(muac, MUAC_CM_VALID)
    obs["muac_observed_t"] = obs["muac_cm_t"].notna().astype(float)
    obs["recumbent_t"] = raw["Cara Ukur"].map({"Terlentang": 1.0, "Berdiri": 0.0})
    for code in WEIGHT_GAIN_CODES:
        obs[f"wg_{code}_t"] = (raw["Naik Berat Badan"] == code).astype(float)
    obs["vita_t"] = (raw["Jml Vit A"].str.strip() != "").astype(float)
    obs["kpsp_sesuai_t"] = (raw["KPSP"] == "Sesuai").astype(float)
    obs["kpsp_meragukan_t"] = (raw["KPSP"] == "Meragukan").astype(float)
    obs["kpsp_penyimpangan_t"] = (raw["KPSP"] == "Penyimpangan").astype(float)
    obs["kia_sesuai_t"] = (raw["KIA"] == "Sesuai").astype(float)
    obs["kelas_ibu_ya_t"] = (raw["Kelas Ibu Balita"] == "Ya").astype(float)
    obs["kelas_ibu_tidak_t"] = (raw["Kelas Ibu Balita"] == "Tidak").astype(float)
    obs["mbg_ya_t"] = (raw["MBG"] == "Ya").astype(float)
    obs["mbg_tidak_t"] = (raw["MBG"] == "Tidak").astype(float)
    audit["values_set_missing_by_validity_rule"] = invalid_counts

    # --- label derivation (also the eligibility flag for anchor/target) --------
    haz = to_number(raw["ZS TB/U"])
    category = raw["TB/U"].map(TBU_CATEGORY_TO_LABEL)
    derived = pd.Series(np.select(
        [haz < -3.0, haz < -2.0, haz <= 3.0, haz > 3.0],
        ["severely stunted", "stunted", "normal", "tall"], default="missing"), index=raw.index)
    haz_plausible = haz.between(*HAZ_VALID)
    agree = category.notna() & (category == derived)
    audit["tbu_category_counts"] = raw["TB/U"].value_counts().to_dict()
    audit["category_vs_haz_disagreements"] = int((category.notna() & haz.notna() & ~agree).sum())
    obs["haz_t"] = haz.where(haz_plausible)
    label_eppgbm = derived.where(agree & haz_plausible)

    if WHO_TABLE_DIR:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import who_haz
        z_who = pd.Series(who_haz.haz_who(age_days, raw["JK"], to_number(raw["Tinggi"]), raw["Cara Ukur"],
                                          who_haz.load_tables(WHO_TABLE_DIR)), index=raw.index)
        who_label = pd.Series(np.select(
            [z_who.round(2) < -3.0, z_who.round(2) < -2.0, z_who.round(2) <= 3.0, z_who.round(2) > 3.0],
            ["severely stunted", "stunted", "normal", "tall"], default="missing"), index=raw.index)
        both = label_eppgbm.notna() & z_who.notna()
        diff = (z_who - haz).abs()
        audit["who_haz_verification"] = {
            "rows_compared": int(both.sum()),
            "category_agreement_pct": float((label_eppgbm[both] == who_label[both]).mean() * 100),
            "abs_diff_le_0_05_pct": float((diff[both] <= 0.05).mean() * 100),
            "abs_diff_gt_0_5_rows": int((diff[both] > 0.5).sum()),
            "median_abs_diff": float(diff[both].median()),
        }
        label_who = who_label.where(z_who.between(*HAZ_VALID) & category.notna())
    elif LABEL_SOURCE == "who":
        raise ValueError("LABEL_SOURCE='who' requires MCE_FP_WHO_TABLE_DIR.")
    if LABEL_SOURCE not in ("eppgbm", "who"):
        raise ValueError("MCE_FP_LABEL_SOURCE must be 'eppgbm' or 'who'.")
    obs["label_text"] = label_who if LABEL_SOURCE == "who" else label_eppgbm
    audit["label_source"] = LABEL_SOURCE
    obs["eligible"] = (
        obs["label_text"].isin(LABELS)
        & (obs["age_months_t"] <= MAX_AGE_MONTHS)
        & (age_days >= 0).to_numpy()
    )
    obs["status_t"] = obs["label_text"].map(LABEL_TO_IDX)
    audit["eligible_observations"] = int(obs["eligible"].sum())
    audit["eligible_label_counts"] = obs.loc[obs["eligible"], "label_text"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict()
    audit["excluded_observation_reasons"] = {
        "tall_haz_gt_3": int((obs["label_text"] == "tall").sum()),
        "system_outlier_or_missing_category": int(category.isna().sum()),
        "haz_missing_or_implausible": int((~haz_plausible).sum()),
        "age_gt_60_months": int((obs["age_months_t"] > MAX_AGE_MONTHS).sum()),
        "measurement_date_before_birth_date": int((age_days < 0).sum()),
    }
    obs = obs.sort_values(["child_pid", "month"]).reset_index(drop=True)
    return obs, audit


# =============================================================================
# STAGE (b2): FEATURES AVAILABLE AT t, PAIRS, TEMPORAL SPLIT
# =============================================================================


def add_history_features(obs: pd.DataFrame) -> pd.DataFrame:
    """History features use observations with month <= t only."""
    work = obs.copy()
    keyed = work.set_index(["child_pid", "month"])

    def lagged(column: str, k: int) -> pd.Series:
        index = pd.MultiIndex.from_arrays([work["child_pid"], work["month"] - k])
        return pd.Series(keyed[column].reindex(index).to_numpy(), index=work.index)

    haz_1, haz_3 = lagged("haz_t", 1), lagged("haz_t", 3)
    work["haz_lag1_observed"] = haz_1.notna().astype(float)
    work["haz_lag3_observed"] = haz_3.notna().astype(float)
    work["d_haz_1"] = work["haz_t"] - haz_1
    work["d_haz_3"] = work["haz_t"] - haz_3
    work["d_waz_1"] = work["waz_t"] - lagged("waz_t", 1)
    work["d_height_1"] = work["height_cm_t"] - lagged("height_cm_t", 1)
    work["d_weight_1"] = work["weight_kg_t"] - lagged("weight_kg_t", 1)

    grouped = work.groupby("child_pid", sort=False)
    work["n_obs_to_t"] = grouped.cumcount().astype(float) + 1.0
    work["months_since_prev_obs"] = grouped["month"].diff()
    stunted_flag = work["status_t"].isin([1, 2]).astype(float)
    work["n_prior_stunted_months"] = stunted_flag.groupby(work["child_pid"]).cumsum() - stunted_flag
    work["min_haz_to_t"] = grouped["haz_t"].cummin()
    return work


def build_pairs(obs: pd.DataFrame, h: int) -> pd.DataFrame:
    anchors = obs.loc[obs["eligible"]].copy()
    targets = obs.loc[obs["eligible"], ["child_pid", "month", "status_t", "haz_t"]].rename(
        columns={"month": "target_month", "status_t": "y", "haz_t": "target_haz_audit_only"}
    )
    anchors["target_month"] = anchors["month"] + h
    pairs = anchors.merge(targets, on=["child_pid", "target_month"], how="inner", validate="one_to_one")
    pairs = pairs.rename(columns={"month": "anchor_month"})
    pairs["y"] = pairs["y"].astype(np.int64)
    pairs["h"] = h
    if (pairs["target_month"] - pairs["anchor_month"] != h).any():
        raise RuntimeError("Pair horizon mismatch.")
    return pairs.sort_values(["child_pid", "anchor_month"]).reset_index(drop=True)


def child_strata(pairs: pd.DataFrame) -> pd.Series:
    """Child-level stratum = most severe target class observed for that child."""
    return pairs.groupby("child_pid")["y"].max()


def child_stratified_split(children: pd.Series, fraction: float, seed: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    keep, hold = train_test_split(
        children.index.to_numpy(), test_size=fraction, stratify=children.to_numpy(), random_state=seed
    )
    if np.intersect1d(keep, hold).size:
        raise RuntimeError(f"{name}: children overlap.")
    return np.asarray(keep), np.asarray(hold)


def run_stage_b() -> None:
    print("\n" + "=" * 96 + "\nSTAGE (b): LOADING, LONGITUDINAL VALIDATION, PAIRS, TEMPORAL SPLIT\n" + "=" * 96)
    validate_contract()
    obs, audit = load_and_validate_observations()
    obs = add_history_features(obs)
    save_json(audit, SHARED_DIR / "dataset_longitudinal_audit.json")

    availability = pd.DataFrame(
        {
            "feature": FEAT_COLS,
            "information_time": [
                "birth (static)" if f in ("sex_male", "birth_weight_kg", "birth_length_cm")
                else "history <= t" if f.startswith(("d_", "n_", "months_", "min_", "haz_lag"))
                else "month t"
                for f in FEAT_COLS
            ],
            "uses_information_after_t": False,
            "missing_fraction_all_eligible_obs": [float(obs.loc[obs["eligible"], f].isna().mean()) for f in FEAT_COLS],
        }
    )
    availability.to_csv(SHARED_DIR / "feature_availability_audit.csv", index=False)

    dev_max, test_min = month_index(DEV_TARGET_MONTH_MAX), month_index(TEST_ANCHOR_MONTH_MIN)
    val_min, val_max = month_index(TEMPORAL_VAL_ANCHOR_MIN), month_index(TEMPORAL_VAL_TARGET_MAX)
    child_codes = obs.drop_duplicates("child_pid").set_index("child_pid")["child_code"].to_dict()
    first_seen = obs.groupby("child_pid")["month"].min()

    # attrition audit: eligible anchors with versus without an observed target at t+h
    attrition_rows = []

    for h in HORIZONS:
        out = horizon_dir(h)
        out.mkdir(parents=True, exist_ok=True)
        pairs = build_pairs(obs, h)
        pairs["child_code"] = pairs["child_pid"].map(child_codes).astype(np.int64)

        anchors = obs.loc[obs["eligible"], ["child_pid", "month", "status_t", "haz_t"]].copy()
        has_target = pd.MultiIndex.from_arrays([anchors["child_pid"], anchors["month"]]).isin(
            pd.MultiIndex.from_arrays([pairs["child_pid"], pairs["anchor_month"]]))
        for flag, frame in ((True, anchors[has_target]), (False, anchors[~has_target])):
            counts = frame["status_t"].value_counts(normalize=True).reindex(range(N_CLASSES)).fillna(0)
            attrition_rows.append({"h": h, "target_observed": flag, "anchors": int(len(frame)),
                                   "mean_haz_t": float(frame["haz_t"].mean()),
                                   **{f"pct_status_t_{LABELS[k]}": float(counts[k] * 100) for k in range(N_CLASSES)}})

        if HOLDOUT_VILLAGE:
            hv = int(HOLDOUT_VILLAGE)
            if hv not in set(obs["village_code"]):
                raise ValueError(f"Unknown village code {hv}; see village_codes.csv")
            train_pool = pairs.loc[pairs["village_code"] != hv]
            test_pool = pairs.loc[pairs["village_code"] == hv]
        else:
            train_pool, test_pool = pairs, pairs
        test_mask = test_pool["anchor_month"] >= test_min
        if TEST_ANCHOR_MONTH_MAX:
            test_mask &= test_pool["anchor_month"] <= month_index(TEST_ANCHOR_MONTH_MAX)
        test = test_pool.loc[test_mask].reset_index(drop=True)
        val = train_pool.loc[(train_pool["anchor_month"] >= val_min) & (train_pool["target_month"] <= val_max)].reset_index(drop=True)
        val_children_rarest = int(val.loc[val["y"] == N_CLASSES - 1, "child_pid"].nunique()) if len(val) else 0
        val_ok = len(val) > 0 and np.all(np.bincount(val["y"], minlength=N_CLASSES) > 0) \
            and val_children_rarest >= MIN_CHILDREN_RAREST_CLASS_IN_VAL
        if val_ok:
            design = "temporal_threshold"
            dev = train_pool.loc[train_pool["target_month"] <= dev_max].reset_index(drop=True)
        else:
            design = "child_grouped_threshold"
            dev = train_pool.loc[train_pool["target_month"] <= val_max].reset_index(drop=True)
            val = train_pool.iloc[0:0].copy()
        gap = len(pairs) - len(dev) - len(val) - len(test)
        if len(test) == 0:
            print(f"  h={h}: no held-out pairs under the temporal boundary; horizon skipped.")
            save_json({"h": h, "status": "skipped_no_test_pairs"}, out / "stage_b_manifest.json")
            continue
        if HOLDOUT_VILLAGE and (set(test["child_pid"]) & (set(dev["child_pid"]) | set(val["child_pid"]))):
            raise RuntimeError("Leave-one-village-out: a test child also appears in development/validation.")
        if test["anchor_month"].min() <= max(dev["target_month"].max(), val["target_month"].max() if len(val) else -1):
            raise RuntimeError("Temporal leakage: a test anchor precedes a development/validation target.")
        if len(val) and val["anchor_month"].min() <= dev["target_month"].max():
            raise RuntimeError("Temporal leakage: a validation anchor precedes a development target.")
        test["new_child"] = test["child_pid"].map(first_seen).ge(test_min).to_numpy()

        dev_counts = np.bincount(dev["y"], minlength=N_CLASSES)
        if np.any(dev_counts == 0):
            raise ValueError(f"h={h}: development set lacks a class: {dev_counts.tolist()}")
        strata = child_strata(dev)
        strata = strata.loc[sorted(strata.index, key=child_codes.__getitem__)]  # salt-independent order
        children_per_stratum = strata.value_counts().reindex(range(N_CLASSES)).fillna(0).astype(int)
        if children_per_stratum.min() < N_SPLITS * MIN_CHILDREN_PER_CLASS_PER_FOLD:
            raise ValueError(
                f"h={h}: too few children in the rarest stratum ({children_per_stratum.tolist()}) "
                f"for {N_SPLITS} child-grouped folds."
            )

        # child-grouped OOF folds (stratified on child stratum)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        fold_of_child = pd.Series(0, index=strata.index, dtype=np.int64)
        for fold, (_, val_idx) in enumerate(skf.split(strata.index, strata.to_numpy()), start=1):
            fold_of_child.iloc[val_idx] = fold
        dev["oof_fold"] = dev["child_pid"].map(fold_of_child).astype(np.int64)

        # child-grouped meta subsets inside development
        subset = pd.Series("", index=strata.index)
        if design == "child_grouped_threshold":
            pool, thr_children = child_stratified_split(strata, THRESHOLD_DEVELOPMENT_CHILD_FRACTION, RANDOM_STATE, "threshold-dev")
            subset.loc[thr_children] = "threshold_development"
        else:
            pool = strata.index.to_numpy()
        meta_tr_children, meta_val_children = child_stratified_split(
            strata.loc[pool], META_VALIDATION_CHILD_FRACTION_WITHIN_POOL, RANDOM_STATE + 1, "meta-val"
        )
        subset.loc[meta_val_children] = "meta_validation"
        subset.loc[meta_tr_children] = "meta_train"
        dev["meta_subset"] = dev["child_pid"].map(subset)
        dev_subsets = ["meta_train", "meta_validation"] + (["threshold_development"] if design == "child_grouped_threshold" else [])
        for name in dev_subsets:
            counts = np.bincount(dev.loc[dev["meta_subset"] == name, "y"], minlength=N_CLASSES)
            if np.any(counts == 0):
                raise ValueError(f"h={h}: {name} lacks a class: {counts.tolist()}")

        fingerprint_dev = pair_fingerprint(dev["child_code"], dev["anchor_month"], dev["y"], h)
        fingerprint_val = pair_fingerprint(val["child_code"], val["anchor_month"], val["y"], h)
        fingerprint_test = pair_fingerprint(test["child_code"], test["anchor_month"], test["y"], h)
        keep_cols = ["child_pid", "child_code", "village_code", "anchor_month", "target_month", "h", "y",
                     "target_haz_audit_only"] + FEAT_COLS + AUDIT_COLS
        dev[keep_cols + ["oof_fold", "meta_subset"]].to_pickle(out / "dev_pairs.pkl")
        val[keep_cols].to_pickle(out / "val_pairs.pkl")
        test[keep_cols + ["new_child"]].to_pickle(out / "test_pairs.pkl")

        majority = int(np.argmax(dev_counts))
        minority = [k for k in range(N_CLASSES) if k != majority]
        rows = []
        frames = [("development", dev), ("internal_temporal_validation", val), ("held_out_test", test)]
        frames += [(name, dev.loc[dev["meta_subset"] == name]) for name in dev_subsets]
        for name, frame in frames:
            if len(frame) == 0:
                continue
            counts = np.bincount(frame["y"], minlength=N_CLASSES)
            for k, label in enumerate(LABELS):
                rows.append({"subset": name, "class": label, "pairs": int(counts[k]),
                             "percent": float(counts[k] / len(frame) * 100),
                             "children": int(frame["child_pid"].nunique()),
                             "children_with_class": int(frame.loc[frame["y"] == k, "child_pid"].nunique())})
        pd.DataFrame(rows).to_csv(out / "class_distribution_by_subset.csv", index=False)

        span = lambda frame, col: [month_text(frame[col].min()), month_text(frame[col].max())] if len(frame) else None
        save_json(
            {
                "h": h,
                "status": "ok",
                "design": design,
                "analysis_role": "primary" if design == "temporal_threshold" else "secondary (exploratory)",
                "label_source": LABEL_SOURCE,
                "ablation": ABLATION,
                "holdout_village": int(HOLDOUT_VILLAGE) if HOLDOUT_VILLAGE else None,
                "test_anchor_month_max": TEST_ANCHOR_MONTH_MAX or None,
                "fingerprint_dev": fingerprint_dev,
                "fingerprint_val": fingerprint_val,
                "fingerprint_test": fingerprint_test,
                "n_dev_pairs": int(len(dev)), "n_dev_children": int(dev["child_pid"].nunique()),
                "n_val_pairs": int(len(val)), "n_val_children": int(val["child_pid"].nunique()) if len(val) else 0,
                "n_test_pairs": int(len(test)), "n_test_children": int(test["child_pid"].nunique()),
                "n_pairs_excluded_at_boundaries": int(gap),
                "dev_anchor_months": span(dev, "anchor_month"), "dev_target_months": span(dev, "target_month"),
                "val_anchor_months": span(val, "anchor_month"), "val_target_months": span(val, "target_month"),
                "test_anchor_months": span(test, "anchor_month"), "test_target_months": span(test, "target_month"),
                "test_children_also_in_dev": int(len(set(test["child_pid"]) & set(dev["child_pid"]))),
                "test_pairs_new_children": int(test["new_child"].sum()),
                "children_per_stratum_dev": children_per_stratum.tolist(),
                "majority_class_from_dev": LABELS[majority],
                "minority_classes_from_dev": [LABELS[k] for k in minority],
                "imbalance_ratio_majority_to_rarest_dev": float(dev_counts.max() / dev_counts.min()),
                "feature_order": FEAT_COLS,
                "class_order": LABELS,
                "grouping": "child identity (pseudonymised); all pairs of a child share one fold/subset",
            },
            out / "stage_b_manifest.json",
        )
        print(f"  h={h} [{design}]: dev={len(dev)} / {dev['child_pid'].nunique()} ch, "
              f"val={len(val)}, test={len(test)} / {test['child_pid'].nunique()} ch, boundary-excluded={gap}")
    pd.DataFrame(attrition_rows).to_csv(SHARED_DIR / "attrition_audit_by_horizon.csv", index=False)
    village_map = obs.drop_duplicates("village_code")[["village_code"]].sort_values("village_code")
    village_map["village_name"] = village_map["village_code"].map(VILLAGE_NAMES)
    village_map["children"] = village_map["village_code"].map(obs.groupby("village_code")["child_pid"].nunique())
    village_map.to_csv(SHARED_DIR / "village_codes.csv", index=False)
    print("✓ Stage (b) completed.")


def load_stage_b(h: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    out = horizon_dir(h)
    manifest = load_json(out / "stage_b_manifest.json")
    if manifest.get("status") != "ok":
        raise RuntimeError(f"h={h}: stage (b) status is {manifest.get('status')}")
    dev = pd.read_pickle(out / "dev_pairs.pkl")
    val = pd.read_pickle(out / "val_pairs.pkl")
    test = pd.read_pickle(out / "test_pairs.pkl")
    if manifest["feature_order"] != FEAT_COLS or manifest["class_order"] != LABELS:
        raise ValueError("Feature or class contract changed since stage (b).")
    for name, frame in (("dev", dev), ("val", val), ("test", test)):
        if pair_fingerprint(frame["child_code"], frame["anchor_month"], frame["y"], h) != manifest[f"fingerprint_{name}"]:
            raise ValueError(f"{name} pair table does not match its stage-(b) fingerprint.")
    return dev, val, test, manifest


# =============================================================================
# STAGE (c): BASE LEARNERS AND CHILD-GROUPED STRICT OOF
# =============================================================================


def fit_transform_foldwise(train_frame: pd.DataFrame, apply_frames: list[pd.DataFrame]) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Median imputation and standardisation fitted on the training fold only."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_processed = scaler.fit_transform(imputer.fit_transform(train_frame[FEAT_COLS]))
    applied = [scaler.transform(imputer.transform(frame[FEAT_COLS])) if len(frame) else np.zeros((0, len(FEAT_COLS)))
               for frame in apply_frames]
    for array in [train_processed, *applied]:
        if not np.isfinite(array).all():
            raise ValueError("Preprocessing produced non-finite values.")
    return train_processed, applied, {"imputer": imputer, "scaler": scaler, "feature_columns": FEAT_COLS.copy(), "fit_scope": "training fold only"}


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
        tr_counts = np.bincount(y_dev[tr], minlength=N_CLASSES)
        if np.any(tr_counts == 0):
            raise ValueError(f"Fold {fold}: training fold lacks a class {tr_counts.tolist()}")
        x_tr, (x_va, x_tv, x_te), prep = fit_transform_foldwise(dev.iloc[tr], [dev.iloc[va], val, test])
        fold_cache.append({"fold": fold, "tr": tr, "va": va, "x_tr": x_tr, "x_va": x_va, "x_tv": x_tv, "x_te": x_te, "prep": prep,
                           "val_counts": np.bincount(y_dev[va], minlength=N_CLASSES).tolist()})

    oof = np.full((len(dev), META_DIMENSION), np.nan, dtype=np.float32)
    meta_test = np.zeros((len(test), META_DIMENSION), dtype=np.float64)
    meta_val = np.zeros((len(val), META_DIMENSION), dtype=np.float64)
    models = define_base_models()
    cv_rows = []
    bundle_dir = out / "base_fold_bundles"
    bundle_dir.mkdir(exist_ok=True)
    for b, name in enumerate(MODEL_ORDER):
        block = slice(b * N_CLASSES, (b + 1) * N_CLASSES)
        print(f"  {name}")
        for fc in fold_cache:
            model = fresh_model(models[name])
            model.fit(prepare_model_input(fc["x_tr"], FEAT_COLS), y_dev[fc["tr"]])
            p_va = predict_proba_aligned(model, prepare_model_input(fc["x_va"], FEAT_COLS), N_CLASSES)
            p_te = predict_proba_aligned(model, prepare_model_input(fc["x_te"], FEAT_COLS), N_CLASSES)
            oof[fc["va"], block] = p_va
            meta_test[:, block] += p_te / N_SPLITS
            if len(val):
                meta_val[:, block] += predict_proba_aligned(model, prepare_model_input(fc["x_tv"], FEAT_COLS), N_CLASSES) / N_SPLITS
            cv_rows.append({"model": name, "fold": fc["fold"], "validation_class_counts": fc["val_counts"],
                            "macro_f1_classes_present": macro_f1_present(y_dev[fc["va"]], p_va.argmax(1))})
            joblib.dump({"model": model, "preprocessing": fc["prep"], "fold": fc["fold"], "model_name": name},
                        bundle_dir / f"{name}_fold_{fc['fold']}.joblib", compress=3)
        meta_test[:, block] /= meta_test[:, block].sum(axis=1, keepdims=True)
        if len(val):
            meta_val[:, block] /= meta_val[:, block].sum(axis=1, keepdims=True)
            validate_probability_matrix(meta_val[:, block], len(val), N_CLASSES, f"{name} temporal validation")
        validate_probability_matrix(oof[:, block], len(dev), N_CLASSES, f"{name} OOF")
        validate_probability_matrix(meta_test[:, block], len(test), N_CLASSES, f"{name} held-out")

    np.save(out / "oof_meta_dev.npy", oof)
    np.save(out / "meta_test.npy", meta_test.astype(np.float32))
    np.save(out / "meta_val.npy", meta_val.astype(np.float32))
    pd.DataFrame(cv_rows).to_csv(out / "base_learners_oof_fold_metrics.csv", index=False)
    pd.DataFrame([{"meta_column_index": b * N_CLASSES + k, "base_model": m, "class_label": c}
                  for b, m in enumerate(MODEL_ORDER) for k, c in enumerate(LABELS)]).to_csv(out / "meta_feature_schema.csv", index=False)
    save_json({"h": h, "fingerprint_dev": manifest["fingerprint_dev"], "fingerprint_val": manifest["fingerprint_val"],
               "fingerprint_test": manifest["fingerprint_test"], "meta_val_shape": list(meta_val.shape),
               "oof_shape": list(oof.shape), "meta_test_shape": list(meta_test.shape), "meta_dimension": META_DIMENSION,
               "oof_grouping": "child", "test_meta_features": "mean of the K fold models (no refit on full dev)",
               "temporal_validation_meta_features": "mean of the same K fold models as the test block",
               "held_out_labels_used": False}, out / "stage_c_manifest.json")
    print(f"✓ Stage (c) h={h}: OOF {oof.shape}, held-out meta {meta_test.shape}")


def load_meta(h: int):
    dev, val, test, manifest = load_stage_b(h)
    out = horizon_dir(h)
    c_manifest = load_json(out / "stage_c_manifest.json")
    if any(c_manifest[f"fingerprint_{k}"] != manifest[f"fingerprint_{k}"] for k in ("dev", "val", "test")):
        raise ValueError("Stage-(c) artifacts belong to a different pair table.")
    oof = np.load(out / "oof_meta_dev.npy").astype(np.float32)
    meta_val = np.load(out / "meta_val.npy").astype(np.float32)
    meta_test = np.load(out / "meta_test.npy").astype(np.float32)
    if (oof.shape != (len(dev), META_DIMENSION) or meta_test.shape != (len(test), META_DIMENSION)
            or meta_val.shape != (len(val), META_DIMENSION)):
        raise ValueError("Meta-feature shapes are inconsistent with the pair tables.")
    return dev, val, test, manifest, oof, meta_val, meta_test


# =============================================================================
# STAGE (d): META-DNN (architecture retained from the original code)
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


class ValidationMacroF1Tracker(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.x, self.y = validation_data
        self.values: list[float] = []
        self.best_epoch, self.best_macro_f1, self.best_val_loss, self.best_weights = 0, -np.inf, np.inf, None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        macro_f1 = macro_f1_present(self.y, np.argmax(self.model.predict(self.x, verbose=0), axis=1))
        val_loss = float(logs.get("val_loss", np.inf))
        logs["val_macro_f1"] = macro_f1
        self.values.append(macro_f1)
        if EPOCH_SELECTION_METRIC == "val_loss":
            better = val_loss < self.best_val_loss - 1e-12 or (
                abs(val_loss - self.best_val_loss) <= 1e-12 and macro_f1 > self.best_macro_f1 + 1e-12)
        else:
            better = macro_f1 > self.best_macro_f1 + 1e-12 or (
                abs(macro_f1 - self.best_macro_f1) <= 1e-12 and val_loss < self.best_val_loss - 1e-12)
        if better:
            self.best_epoch, self.best_macro_f1, self.best_val_loss = epoch + 1, macro_f1, val_loss
            self.best_weights = self.model.get_weights()


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
    print("\n" + "=" * 96 + f"\nSTAGE (d) h={h}: META-DNN EPOCH SELECTION AND FRESH REFIT\n" + "=" * 96)
    dev, val, test, manifest, oof, meta_val, meta_test = load_meta(h)
    out = horizon_dir(h)
    y = dev["y"].to_numpy(np.int64)
    subset = dev["meta_subset"].to_numpy()
    tr, va = np.flatnonzero(subset == "meta_train"), np.flatnonzero(subset == "meta_validation")
    th = np.flatnonzero(subset == "threshold_development")
    codes = dev["child_code"].to_numpy()
    for a, b, label in ((tr, va, "train/val"), (tr, th, "train/threshold"), (va, th, "val/threshold")):
        if np.intersect1d(codes[a], codes[b]).size:
            raise RuntimeError(f"Children overlap between meta subsets: {label}")

    tf.keras.backend.clear_session(); set_global_seed()
    model = build_meta_dnn(META_DIMENSION, N_CLASSES)
    tracker, recorder = ValidationMacroF1Tracker((oof[va], y[va])), LearningRateRecorder()
    history = model.fit(
        oof[tr], y[tr], validation_data=(oof[va], y[va]), epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
        callbacks=[tracker, recorder,
                   ReduceLROnPlateau(monitor="val_loss", mode="min", factor=LR_REDUCER_FACTOR, patience=LR_REDUCER_PATIENCE, min_lr=MIN_LEARNING_RATE),
                   EarlyStopping(monitor="val_loss" if EPOCH_SELECTION_METRIC == "val_loss" else "val_macro_f1",
                                 mode="min" if EPOCH_SELECTION_METRIC == "val_loss" else "max",
                                 patience=EARLY_STOPPING_PATIENCE, min_delta=1e-6, restore_best_weights=False)],
        verbose=0 if SMOKE_TEST else 2)
    epoch = int(tracker.best_epoch)
    if epoch < 1:
        raise RuntimeError("Epoch selection failed.")
    schedule = np.asarray(recorder.learning_rates[:epoch])
    frame = pd.DataFrame(history.history); frame["epoch"] = np.arange(1, len(frame) + 1)
    frame["val_macro_f1"] = tracker.values; frame["learning_rate"] = recorder.learning_rates
    frame.to_csv(out / "meta_dnn_epoch_selection_history.csv", index=False)

    refit = np.concatenate([tr, va])
    tf.keras.backend.clear_session(); set_global_seed()
    final = build_meta_dnn(META_DIMENSION, N_CLASSES)
    final.fit(oof[refit], y[refit], epochs=epoch, batch_size=BATCH_SIZE, shuffle=True, callbacks=[LearningRateReplay(schedule)], verbose=0)
    final.save(out / "meta_dnn.keras")
    p_test = validate_probability_matrix(final.predict(meta_test, verbose=0), len(test), N_CLASSES, "Meta-DNN held-out")
    reloaded = tf.keras.models.load_model(out / "meta_dnn.keras", compile=False)
    if not np.allclose(reloaded.predict(meta_test, verbose=0), p_test, atol=NUMERIC_TOLERANCE):
        raise RuntimeError("Meta-DNN reload does not reproduce held-out probabilities.")
    np.save(out / "meta_dnn_test_probabilities.npy", p_test.astype(np.float32))
    save_json({"h": h, "fingerprint_dev": manifest["fingerprint_dev"], "fingerprint_test": manifest["fingerprint_test"],
               "selected_epoch": epoch, "selection_metric": ("meta-validation loss (sparse categorical cross-entropy)"
                                    if EPOCH_SELECTION_METRIC == "val_loss" else "meta-validation macro-F1 (classes present)"),
               "tie_breaker": "higher macro-F1" if EPOCH_SELECTION_METRIC == "val_loss" else "lower validation loss", "refit_subsets": ["meta_train", "meta_validation"],
               "threshold_development_excluded_from_weights": True, "held_out_labels_used": False,
               "sizes": {"meta_train": int(len(tr)), "meta_validation": int(len(va)), "threshold_development": int(len(th))}},
              out / "stage_d_manifest.json")
    print(f"✓ Stage (d) h={h}: selected epoch {epoch}")


# =============================================================================
# STAGE (e): CLASS-SPECIFIC THRESHOLDS (decision rule retained)
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
    tp = int(np.sum((target == 1) & (pred == 1))); fp = int(np.sum((target == 0) & (pred == 1))); fn = int(np.sum((target == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def learn_independent_thresholds(labels: np.ndarray, confidences: np.ndarray):
    thresholds, rows = np.zeros(N_CLASSES), []
    for k, label in enumerate(LABELS):
        target = (labels == k).astype(np.int64)
        if target.sum() == 0:
            raise ValueError(f"Threshold-development set has no {label} cases.")
        candidates = [binary_threshold_metrics(target, confidences[:, k], t) for t in THRESHOLD_GRID]
        best = max(candidates, key=lambda r: (r["f1"], r["recall"], r["precision"], -abs(r["threshold"] - 0.5), -r["threshold"]))
        thresholds[k] = best["threshold"]
        rows.append({"class": label, "positive_support": int(target.sum()), **best})
    return thresholds, pd.DataFrame(rows)


def run_stage_e(h: int) -> None:
    print("\n" + "=" * 96 + f"\nSTAGE (e) h={h}: CLASS-SPECIFIC THRESHOLDS (TEMPORAL VALIDATION OR THRESHOLD-DEVELOPMENT)\n" + "=" * 96)
    dev, val, test, manifest, oof, meta_val, meta_test = load_meta(h)
    out = horizon_dir(h)
    d_manifest = load_json(out / "stage_d_manifest.json")
    if d_manifest["fingerprint_dev"] != manifest["fingerprint_dev"]:
        raise ValueError("Stage-(d) artifacts belong to a different pair table.")
    model = tf.keras.models.load_model(out / "meta_dnn.keras", compile=False)
    if manifest["design"] == "temporal_threshold":
        source = "internal temporal validation block (fold-ensemble meta-features, same form as test)"
        x_th, y_th, ch_th = meta_val, val["y"].to_numpy(), val["child_pid"]
    else:
        source = "20% of development children (OOF meta-features); secondary design"
        th = np.flatnonzero(dev["meta_subset"].to_numpy() == "threshold_development")
        x_th, y_th, ch_th = oof[th], dev["y"].to_numpy()[th], dev.iloc[th]["child_pid"]
    p_th = validate_probability_matrix(model.predict(x_th, verbose=0), len(y_th), N_CLASSES, "threshold-learning probabilities")
    thresholds, summary = learn_independent_thresholds(y_th, p_th)
    np.save(out / "class_specific_thresholds.npy", thresholds)
    summary.to_csv(out / "class_specific_thresholds.csv", index=False)
    save_json({"h": h, "fingerprint_dev": manifest["fingerprint_dev"], "thresholds": dict(zip(LABELS, thresholds.tolist())),
               "objective": "one-vs-rest F1 per class", "grid": [0.01, 0.99, 0.01],
               "tie_break": ["f1", "recall", "precision", "closest to 0.5", "lower threshold"],
               "decision": "argmax_k sigma(logit p_k - logit tau_k); always exactly one class",
               "frozen_before_held_out_evaluation": True, "held_out_labels_used": False,
               "threshold_learning_source": source,
               "threshold_learning_pairs": int(len(y_th)),
               "threshold_learning_children": int(ch_th.nunique())},
              out / "stage_e_manifest.json")
    print(f"✓ Stage (e) h={h}: thresholds frozen")


# =============================================================================
# STAGE (f): FROZEN HELD-OUT EVALUATION
# =============================================================================


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, minority: Iterable[int] | None = None) -> dict[str, Any]:
    """Undefined per-class values are NaN; macro averages use classes present in y_true."""
    y_true = np.asarray(y_true, np.int64); y_pred = np.asarray(y_pred, np.int64)
    cm = np.bincount(y_true * N_CLASSES + y_pred, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    support, predicted, tp = cm.sum(1).astype(float), cm.sum(0).astype(float), np.diag(cm).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(predicted > 0, tp / predicted, np.where(support > 0, 0.0, np.nan))
        f1 = np.where(np.isnan(recall), np.nan, np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0))
    present = support > 0
    out = {"n": int(len(y_true)), "accuracy": float(tp.sum() / max(1, len(y_true))),
           "balanced_accuracy": float(np.nanmean(recall[present])),
           "macro_precision": float(np.nanmean(precision[present])), "macro_recall": float(np.nanmean(recall[present])),
           "macro_f1": float(np.nanmean(f1[present])), "classes_present": int(present.sum()),
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
    ece = sum(abs(top[bins == b].mean() - correct[bins == b].mean()) * (bins == b).mean() for b in range(RELIABILITY_BINS) if (bins == b).any())
    classwise = []
    for k in range(N_CLASSES):
        pk = p[:, k]; bk = np.minimum((pk * RELIABILITY_BINS).astype(int), RELIABILITY_BINS - 1)
        classwise.append(sum(abs(pk[bk == b].mean() - one_hot[bk == b, k].mean()) * (bk == b).mean() for b in range(RELIABILITY_BINS) if (bk == b).any()))
    return {"roc_auc_ovr_macro": auc,
            "nll": float(-np.mean(np.log(np.clip(p[np.arange(len(y_true)), y_true], CONFIDENCE_EPSILON, 1.0)))),
            "brier_multiclass": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
            "ece_top_label": float(ece), "ece_classwise_macro": float(np.mean(classwise))}


SCALAR_KEYS = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "minority_macro_recall", "minimum_minority_recall"]


def cluster_bootstrap(children: np.ndarray, y: np.ndarray, decisions: dict[str, np.ndarray], minority: list[int],
                      comparisons: list[tuple[str, str]], replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample CHILDREN with replacement; all pairs of a sampled child are included."""
    rng = np.random.default_rng(seed)
    unique = np.unique(children)
    members = {c: np.flatnonzero(children == c) for c in unique}
    values = {m: {k: [] for k in SCALAR_KEYS} for m in decisions}
    deltas = {cmp: {k: [] for k in SCALAR_KEYS} for cmp in comparisons}
    for _ in range(replicates):
        idx = np.concatenate([members[c] for c in rng.choice(unique, size=len(unique), replace=True)])
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
    ci_rows = [{"method": m, "metric": k, **dict(zip(["ci_low", "ci_high", "valid_replicates"], summarise(values[m][k])))} for m in decisions for k in SCALAR_KEYS]
    delta_rows = [{"comparison": f"{b} minus {a}", "metric": k, **dict(zip(["ci_low", "ci_high", "valid_replicates"], summarise(deltas[(a, b)][k])))} for a, b in comparisons for k in SCALAR_KEYS]
    return pd.DataFrame(ci_rows), pd.DataFrame(delta_rows)


def run_stage_f(h: int) -> None:
    print("\n" + "=" * 96 + f"\nSTAGE (f) h={h}: FROZEN HELD-OUT EVALUATION\n" + "=" * 96)
    dev, val, test, manifest, oof, meta_val, meta_test = load_meta(h)
    out = horizon_dir(h)
    e_manifest = load_json(out / "stage_e_manifest.json")
    if e_manifest["fingerprint_dev"] != manifest["fingerprint_dev"]:
        raise ValueError("Stage-(e) thresholds belong to a different pair table.")
    thresholds = np.load(out / "class_specific_thresholds.npy")
    p_meta = np.load(out / "meta_dnn_test_probabilities.npy").astype(np.float64)
    y = test["y"].to_numpy(np.int64)
    minority = [LABEL_TO_IDX[c] for c in manifest["minority_classes_from_dev"]]

    scores, ilcadt = apply_threshold_decision(p_meta, thresholds)
    np.save(out / "mce_ilcadt_test_decision_scores.npy", scores)
    decisions = {"persistence": test["status_t"].to_numpy(np.int64), "meta_dnn_argmax": p_meta.argmax(1), "mce_ilcadt": ilcadt}
    probabilities = {"meta_dnn": p_meta}
    for b, name in enumerate(MODEL_ORDER):
        block = meta_test[:, b * N_CLASSES:(b + 1) * N_CLASSES].astype(np.float64)
        block /= block.sum(1, keepdims=True)
        probabilities[name] = block
        decisions[f"{name}_argmax"] = block.argmax(1)

    subsets = {"all": np.ones(len(test), bool), "status_change": decisions["persistence"] != y, "new_children": test["new_child"].to_numpy(bool)}
    rows, per_class_rows = [], []
    for subset_name, mask in subsets.items():
        if not mask.any():
            continue
        for method, pred in decisions.items():
            m = classification_metrics(y[mask], pred[mask], minority)
            rows.append({"subset": subset_name, "method": method, **{k: m[k] for k in ["n", "classes_present"] + SCALAR_KEYS},
                         "confusion_matrix": m["confusion_matrix"].tolist()})
            for k, label in enumerate(LABELS):
                per_class_rows.append({"subset": subset_name, "method": method, "class": label, "support": int(m["support_per_class"][k]),
                                       "precision": m["precision_per_class"][k], "recall": m["recall_per_class"][k], "f1": m["f1_per_class"][k]})
    prob_rows = [{"method": name, **probability_metrics(y, p)} for name, p in probabilities.items()]
    ci, deltas = cluster_bootstrap(test["child_code"].to_numpy(), y, {k: decisions[k] for k in ("persistence", "meta_dnn_argmax", "mce_ilcadt")},
                                   minority, [("meta_dnn_argmax", "mce_ilcadt"), ("persistence", "mce_ilcadt"), ("persistence", "meta_dnn_argmax")],
                                   BOOTSTRAP_REPLICATES, RANDOM_STATE)
    pd.DataFrame(rows).to_csv(out / "heldout_classification_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(out / "heldout_per_class_metrics.csv", index=False)
    pd.DataFrame(prob_rows).to_csv(out / "heldout_probability_metrics.csv", index=False)
    ci.to_csv(out / "heldout_child_bootstrap_ci.csv", index=False)
    deltas.to_csv(out / "heldout_child_bootstrap_paired_deltas.csv", index=False)
    save_json({"h": h, "fingerprint_test": manifest["fingerprint_test"], "thresholds_frozen": e_manifest["thresholds"],
               "minority_classes_from_dev": manifest["minority_classes_from_dev"], "bootstrap_unit": "child",
               "bootstrap_replicates": BOOTSTRAP_REPLICATES,
               "calibration_metrics_source": "raw probabilities only (never threshold-adjusted scores)",
               "persistence_baseline": "y_hat(t+h) = status(t)",
               "design": manifest["design"], "analysis_role": manifest["analysis_role"],
               "label_source": manifest["label_source"],
               "ablation": manifest.get("ablation", "none"),
               "holdout_village": manifest.get("holdout_village"),
               "primary_evidence": "status_change subset (status_t != y) versus persistence"}, out / "stage_f_manifest.json")
    if SMOKE_TEST:
        print("  SMOKE TEST: metrics written to disk but not printed (not a valid experiment).")
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
        if load_json(horizon_dir(h) / "stage_b_manifest.json").get("status") != "ok":
            continue
        for stage, function in (("c", run_stage_c), ("d", run_stage_d), ("e", run_stage_e), ("f", run_stage_f)):
            if stage in RUN_STAGES:
                function(h)
    print("\nMCE-ILCADT future-prediction pipeline finished. Artifacts:", BASE_DIR)


if __name__ == "__main__":
    main()
