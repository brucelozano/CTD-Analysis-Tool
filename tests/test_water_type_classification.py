from __future__ import annotations

import pandas as pd

from ctd_tool.water_types import classification_summary, classify_water_types


def test_classification_rules_lcow_cw_mix_unknown() -> None:
    matches = pd.DataFrame(
        [
            {"cast_id": "A", "match_status": "ok", "ssha_i_m": 0.18, "ssha_gom_m": 0.10, "t300_c": 16.2},
            {"cast_id": "B", "match_status": "ok", "ssha_i_m": 0.08, "ssha_gom_m": 0.10, "t300_c": 12.2},
            {"cast_id": "C", "match_status": "ok", "ssha_i_m": 0.13, "ssha_gom_m": 0.10, "t300_c": 14.2},
            {"cast_id": "D", "match_status": "ok", "ssha_i_m": 0.19, "ssha_gom_m": 0.10, "t300_c": 14.2},
        ]
    )
    out = classify_water_types(matches, require_ok_matches=True)
    classes = dict(zip(out["cast_id"], out["water_type"]))
    assert classes["A"] == "LCOW"
    assert classes["B"] == "CW"
    assert classes["C"] == "MIX"
    assert classes["D"] == "UNKNOWN"


def test_non_ok_rows_excluded_by_default() -> None:
    matches = pd.DataFrame(
        [
            {"cast_id": "A", "match_status": "error", "ssha_i_m": 0.18, "ssha_gom_m": 0.10, "t300_c": 16.2},
        ]
    )
    out = classify_water_types(matches)
    assert out.iloc[0]["classification_input_ok"] == False  # noqa: E712
    assert out.iloc[0]["water_type"] == "UNKNOWN"


def test_classification_summary_counts() -> None:
    matches = pd.DataFrame(
        [
            {"cast_id": "A", "match_status": "ok", "ssha_i_m": 0.18, "ssha_gom_m": 0.10, "t300_c": 16.2},
            {"cast_id": "B", "match_status": "ok", "ssha_i_m": 0.08, "ssha_gom_m": 0.10, "t300_c": 12.2},
        ]
    )
    out = classify_water_types(matches)
    summary = classification_summary(out)
    assert summary["rows"] == 2
    assert summary["water_type_counts"]["LCOW"] == 1
    assert summary["water_type_counts"]["CW"] == 1
