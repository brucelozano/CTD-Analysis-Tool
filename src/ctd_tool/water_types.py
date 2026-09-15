from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


LCOW_SSHA_OFFSET_M = 0.067
LCOW_T300_THRESHOLD_C = 15.92
LCOW_INDEX_T300_TERM_C = 15.922
CW_T300_THRESHOLD_C = 13.46


def classify_water_types(
    matches: pd.DataFrame,
    *,
    require_ok_matches: bool = True,
) -> pd.DataFrame:
    frame = matches.copy()

    # Ensure numeric columns.
    for col in ("t300_c", "ssha_i_m", "ssha_gom_m"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = np.nan

    if require_ok_matches and "match_status" in frame.columns:
        mask = frame["match_status"] == "ok"
    else:
        mask = pd.Series(True, index=frame.index)

    frame["classification_input_ok"] = mask & frame["t300_c"].notna() & frame["ssha_i_m"].notna() & frame["ssha_gom_m"].notna()
    frame["ssha_threshold_lcow_m"] = frame["ssha_gom_m"] + LCOW_SSHA_OFFSET_M
    frame["ssha_delta_i_minus_gom_m"] = frame["ssha_i_m"] - frame["ssha_gom_m"]

    # Johnston/Boswell Eq. 1 style LCOW index.
    frame["lcow_index_raw"] = (
        frame["ssha_i_m"] - (frame["ssha_gom_m"] + LCOW_SSHA_OFFSET_M) + frame["t300_c"] - LCOW_INDEX_T300_TERM_C
    )

    frame["water_type"] = frame.apply(_classify_row, axis=1)

    # Additional indices used in Boswell et al. 2020 style continuum.
    frame["cw_index_raw"] = CW_T300_THRESHOLD_C - frame["t300_c"]
    frame["cw_index_strength"] = frame["cw_index_raw"].clip(lower=0.0)
    frame["frontal_raw"] = frame["ssha_delta_i_minus_gom_m"]

    frame["lcow_index_scaled_1_2"] = _minmax_scale(frame["lcow_index_raw"], lower=1.0, upper=2.0)
    frame["frontal_index_scaled_0_1"] = _minmax_scale(frame["frontal_raw"], lower=0.0, upper=1.0)
    frame["cw_index_scaled_0_to_neg1"] = _cw_scaled(frame["cw_index_strength"])

    cols = [
        "water_type",
        "classification_input_ok",
        "ssha_i_m",
        "ssha_gom_m",
        "ssha_threshold_lcow_m",
        "ssha_delta_i_minus_gom_m",
        "t300_c",
        "lcow_index_raw",
        "lcow_index_scaled_1_2",
        "cw_index_raw",
        "cw_index_scaled_0_to_neg1",
        "frontal_index_scaled_0_1",
    ]
    for col in cols:
        if col not in frame.columns:
            frame[col] = np.nan

    return frame


def classification_summary(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame["water_type"].value_counts(dropna=False).to_dict() if "water_type" in frame.columns else {}
    return {
        "rows": int(len(frame)),
        "input_ok_rows": int(frame["classification_input_ok"].sum()) if "classification_input_ok" in frame.columns else 0,
        "water_type_counts": {str(k): int(v) for k, v in counts.items()},
        "thresholds": {
            "ssha_offset_lcow_m": LCOW_SSHA_OFFSET_M,
            "t300_lcow_c": LCOW_T300_THRESHOLD_C,
            "t300_cw_c": CW_T300_THRESHOLD_C,
            "lcow_index_t300_term_c": LCOW_INDEX_T300_TERM_C,
        },
    }


def _classify_row(row: pd.Series) -> str:
    if not bool(row.get("classification_input_ok", False)):
        return "UNKNOWN"

    ssha_i = float(row["ssha_i_m"])
    ssha_gom = float(row["ssha_gom_m"])
    t300 = float(row["t300_c"])

    if (ssha_i > ssha_gom + LCOW_SSHA_OFFSET_M) and (t300 > LCOW_T300_THRESHOLD_C):
        return "LCOW"
    if (ssha_i <= ssha_gom) and (t300 < CW_T300_THRESHOLD_C):
        return "CW"
    if (ssha_i > ssha_gom) and (ssha_i < ssha_gom + LCOW_SSHA_OFFSET_M) and (CW_T300_THRESHOLD_C <= t300 <= LCOW_T300_THRESHOLD_C):
        return "MIX"
    return "UNKNOWN"


def _minmax_scale(series: pd.Series, *, lower: float, upper: float) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    vmin = vals.min(skipna=True)
    vmax = vals.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(np.nan, index=series.index)
    scaled = (vals - vmin) / (vmax - vmin)
    return lower + scaled * (upper - lower)


def _cw_scaled(cw_strength: pd.Series) -> pd.Series:
    vals = pd.to_numeric(cw_strength, errors="coerce").fillna(0.0)
    vmax = vals.max()
    if vmax <= 0:
        return pd.Series(0.0, index=cw_strength.index)
    # 0 means weakest CW signal; -1 strongest CW signal.
    return -(vals / vmax)
