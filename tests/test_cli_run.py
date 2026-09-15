from __future__ import annotations

import json
from pathlib import Path

from ctd_tool.cli import _build_parser


def test_run_parser_accepts_quick_preset() -> None:
    parser = _build_parser()
    args = parser.parse_args(["run", "data_dir", "--preset", "quick", "--skip-plots"])
    assert args.command == "run"
    assert args.preset == "quick"
    assert args.skip_plots is True


def test_run_quick_writes_summary(tmp_path: Path) -> None:
    data_dir = tmp_path / "cnv"
    data_dir.mkdir(parents=True, exist_ok=True)
    cnv = data_dir / "TEST_D_CTD_001.cnv"
    cnv.write_text(
        "\n".join(
            [
                "* Sea-Bird SBE 9 Data File:",
                "# nquan = 5",
                "# nvalues = 4",
                "# start_time = May 08 2023 07:57:51 [NMEA time, header]",
                "# name 0 = depSM: Depth [salt water, m]",
                "# name 1 = latitude: Latitude [deg]",
                "# name 2 = longitude: Longitude [deg]",
                "# name 3 = sal00: Salinity, Practical [PSU]",
                "# name 4 = t090C: Temperature [ITS-90, deg C]",
                "*END*",
                "0 28.0 -88.0 35.0 25.0",
                "100 28.0 -88.0 35.1 18.0",
                "200 28.0 -88.0 35.2 14.0",
                "300 28.0 -88.0 35.3 11.0",
            ]
        ),
        encoding="utf-8",
    )

    out_root = tmp_path / "run_out"
    parser = _build_parser()
    args = parser.parse_args(
        [
            "run",
            str(data_dir),
            "--out-root",
            str(out_root),
            "--preset",
            "quick",
            "--skip-plots",
        ]
    )
    args.func(args)

    summary_path = out_root / "run_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["preset"] == "quick"
    assert summary["run_ssha"] is False
    assert summary["counts"]["scan_total_cnv_files"] == 1
    assert summary["counts"]["analysis_cast_rows"] == 1
