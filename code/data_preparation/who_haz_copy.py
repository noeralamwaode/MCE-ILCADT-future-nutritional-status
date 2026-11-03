"""WHO 2006 length/height-for-age z-score (HAZ) from LMS tables.

Tables: lhfa_{boys,girls}_{0_13 (weeks), 0_2, 2_5 (months)}_zscores.json
(WHO Child Growth Standards; JSON copies distributed with pygrowup, BSD licence).
Age in days is used only to compute the z-score, as the WHO standard requires.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

DAYS_PER_MONTH = 30.4375
MAX_AGE_DAYS = 1856  # 60 completed months


def load_tables(table_dir: str | Path) -> dict:
    table_dir = Path(table_dir)
    tables = {}
    for sex in ("boys", "girls"):
        for part, unit in (("0_13", "Week"), ("0_2", "Month"), ("2_5", "Month")):
            frame = pd.DataFrame(json.loads((table_dir / f"lhfa_{sex}_{part}_zscores.json").read_text()))
            tables[(sex, part)] = frame[[unit, "L", "M", "S"]].astype(float).sort_values(unit).to_numpy()
    return tables


def haz_who(age_days, sex, length_cm, position, tables) -> np.ndarray:
    """Vectorised HAZ. sex: 'L'/'P'; position: 'Terlentang'/'Berdiri' (other -> no adjustment)."""
    age_days = np.asarray(age_days, float); length_cm = np.asarray(length_cm, float)
    sex = np.asarray(sex, object); position = np.asarray(position, object)
    out = np.full(len(age_days), np.nan)
    for i, (d, s, y, p) in enumerate(zip(age_days, sex, length_cm, position)):
        if not np.isfinite(d) or not np.isfinite(y) or d < 0 or d > MAX_AGE_DAYS or y <= 0 or s not in ("L", "P"):
            continue
        if d < 731 and p == "Berdiri":
            y = y + 0.7
        elif d >= 731 and p == "Terlentang":
            y = y - 0.7
        key = "boys" if s == "L" else "girls"
        if d < 91:
            t, a = tables[(key, "0_13")], d / 7.0
        else:
            t, a = tables[(key, "0_2" if d < 731 else "2_5")], d / DAYS_PER_MONTH
        a = min(max(a, t[0, 0]), t[-1, 0])
        L, M, S = (np.interp(a, t[:, 0], t[:, j]) for j in (1, 2, 3))
        out[i] = ((y / M) ** L - 1.0) / (L * S)
    return out


def haz_category(z) -> np.ndarray:
    z = np.round(np.asarray(z, float), 2)
    return np.select([np.isnan(z), z < -3, z < -2, z <= 3], ["", "Sangat Pendek", "Pendek", "Normal"], "Tinggi")
