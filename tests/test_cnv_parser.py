from __future__ import annotations

from pathlib import Path

import pytest

from ctd_tool.cnv_parser import CnvParseError, parse_cnv


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "DP09_CTD_CNV" / "PS23_22_Sutton_CTD_Processed"


def test_parse_full_profile_cnv() -> None:
    path = DATA_ROOT / "B082D_CTD_255_converted.cnv"
    cast = parse_cnv(path)
    assert cast.is_sv_only is False
    assert len(cast.data) > 1000
    assert "depth_m" in cast.data.columns
    assert "temperature_c" in cast.data.columns
    assert "salinity_psu" in cast.data.columns
    assert cast.canonical_map["depth_m"] == "depSM"


def test_parse_sv_only_cnv_classification() -> None:
    path = DATA_ROOT / "Sound Velocity" / "B082D_CTD_255_converted_SV.cnv"
    cast = parse_cnv(path)
    assert cast.is_sv_only is True
    assert "depth_m" in cast.data.columns
    assert "sound_velocity_m_s" in cast.data.columns
    assert "temperature_c" not in cast.data.columns


def test_parse_missing_end_marker_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "broken.cnv"
    bad_file.write_text(
        "\n".join(
            [
                "* Sea-Bird SBE 9 Data File:",
                "# nquan = 2",
                "# name 0 = depSM: Depth [salt water, m]",
                "# name 1 = t090C: Temperature [ITS-90, deg C]",
                "1.0 20.0",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(CnvParseError):
        parse_cnv(bad_file)

