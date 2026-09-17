from __future__ import annotations

from pathlib import Path

import pandas as pd

from ctd_tool.cast import CastHeader, CastRecord
from ctd_tool.plots import _infer_day_night_from_cast


def _cast_for_name(name: str) -> CastRecord:
    header = CastHeader(
        source_path=Path(f"{name}.cnv"),
        start_time=None,
        nquan=None,
        nvalues=None,
        interval_label=None,
        interval_value=None,
        bad_flag=None,
        variables={},
    )
    return CastRecord(
        header=header,
        data=pd.DataFrame(
            [
                {"depth_m": 0.0, "temperature_c": 25.0, "salinity_psu": 35.0},
                {"depth_m": 10.0, "temperature_c": 24.5, "salinity_psu": 35.1},
            ]
        ),
        canonical_map={
            "depth_m": "depth_m",
            "temperature_c": "temperature_c",
            "salinity_psu": "salinity_psu",
        },
        is_sv_only=False,
    )


def test_infer_day_night_handles_b175_converted_and_bin_names() -> None:
    cases = {
        "B175D_CTD_245_converted": "D",
        "B175N_CTD_246_converted": "N",
        "B175D_CTD_245_converted_bin": "D",
        "B175N_CTD_246_converted_bin": "N",
    }
    for cast_name, expected in cases.items():
        cast = _cast_for_name(cast_name)
        assert _infer_day_night_from_cast(cast) == expected
