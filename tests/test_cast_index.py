from __future__ import annotations

from pathlib import Path

from ctd_tool.hycom.cast_index import build_cast_index


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "DP09_CTD_CNV" / "PS23_22_Sutton_CTD_Processed"


def test_cast_index_exports_selected_rows() -> None:
    frame = build_cast_index(DATA_ROOT, prefer_binned=True)
    assert not frame.empty
    assert "is_selected" in frame.columns
    assert frame["is_selected"].sum() > 0


def test_cast_index_prefers_binned_when_available() -> None:
    frame = build_cast_index(DATA_ROOT, prefer_binned=True)
    family = frame[frame["cast_family_id"] == "B252N_CTD_248_converted"]
    assert len(family) == 2
    selected = family[family["is_selected"] == True]  # noqa: E712
    assert len(selected) == 1
    assert bool(selected.iloc[0]["is_binned"]) is True
