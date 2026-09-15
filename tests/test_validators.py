from __future__ import annotations

from pathlib import Path

from ctd_tool.validators import parse_and_validate_cnv


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "DP09_CTD_CNV" / "PS23_22_Sutton_CTD_Processed"


def test_validate_full_profile_success() -> None:
    path = DATA_ROOT / "B082D_CTD_255_converted.cnv"
    result = parse_and_validate_cnv(path)
    assert result.cast is not None
    assert result.skipped is False
    assert result.report.status in {"ok", "warning"}
    assert len(result.report.errors) == 0


def test_validate_excludes_sv_only_by_default() -> None:
    path = DATA_ROOT / "Sound Velocity" / "B082D_CTD_255_converted_SV.cnv"
    result = parse_and_validate_cnv(path)
    assert result.cast is not None
    assert result.skipped is True
    assert result.skip_reason == "sv_only_excluded"


def test_validate_detects_row_issues(tmp_path: Path) -> None:
    path = tmp_path / "row_issue.cnv"
    path.write_text(
        "\n".join(
            [
                "* Sea-Bird SBE 9 Data File:",
                "# nquan = 4",
                "# nvalues = 3",
                "# interval = seconds: 1",
                "# bad_flag = -9.99",
                "# name 0 = depSM: Depth [salt water, m]",
                "# name 1 = t090C: Temperature [ITS-90, deg C]",
                "# name 2 = sal00: Salinity, Practical [PSU]",
                "# name 3 = flag: flag",
                "*END*",
                "1.0 20.0 35.0 0.0",
                "2.0 BAD 35.1 0.0",
                "3.0 19.5 35.2",
            ]
        ),
        encoding="utf-8",
    )
    result = parse_and_validate_cnv(path)
    assert result.cast is not None
    assert result.report.status == "warning"
    warning_codes = {w.code for w in result.report.warnings}
    assert "non_numeric_data" in warning_codes
    assert "row_length_mismatch" in warning_codes

