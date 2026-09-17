from __future__ import annotations

from pathlib import Path

from ctd_tool.discovery import describe_cnv_path


def test_describe_cnv_path_b001_night_pattern() -> None:
    info = describe_cnv_path(Path("B001_Night_244_converted.cnv"))
    assert info.site == "B001"
    assert info.day_night == "N"
    assert info.is_binned is False


def test_describe_cnv_path_b001_night_bin_pattern() -> None:
    info = describe_cnv_path(Path("B001_Night_244_converted_bin.cnv"))
    assert info.site == "B001"
    assert info.day_night == "N"
    assert info.is_binned is True


def test_describe_cnv_path_b082d_ctd_pattern() -> None:
    info = describe_cnv_path(Path("B082D_CTD_255_converted.cnv"))
    assert info.site == "B082"
    assert info.day_night == "D"
    assert info.ctd_number == 255


def test_describe_cnv_path_utah_n_ctd_pattern() -> None:
    info = describe_cnv_path(Path("UTAH_N_CTD_262_converted.cnv"))
    assert info.site == "UTAH"
    assert info.day_night == "N"
    assert info.ctd_number == 262
