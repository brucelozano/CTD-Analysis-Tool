from __future__ import annotations

from pathlib import Path

import pandas as pd

from ctd_tool.cast import CastHeader, CastRecord
from ctd_tool.profiles import prepare_profile


def _cast_from_rows(rows: list[dict[str, float]]) -> CastRecord:
    header = CastHeader(
        source_path=Path("test_cast.cnv"),
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
        data=pd.DataFrame(rows),
        canonical_map={
            "depth_m": "depth_m",
            "temperature_c": "temperature_c",
            "salinity_psu": "salinity_psu",
        },
        is_sv_only=False,
    )


def test_prepare_profile_uses_downcast_segment_before_max_depth() -> None:
    cast = _cast_from_rows(
        [
            {"depth_m": 0.0, "temperature_c": 25.0, "salinity_psu": 35.0},
            {"depth_m": 10.0, "temperature_c": 24.0, "salinity_psu": 35.1},
            {"depth_m": 20.0, "temperature_c": 23.0, "salinity_psu": 35.2},
            {"depth_m": 30.0, "temperature_c": 22.0, "salinity_psu": 35.3},
            {"depth_m": 20.0, "temperature_c": 21.0, "salinity_psu": 35.4},
            {"depth_m": 10.0, "temperature_c": 20.0, "salinity_psu": 35.5},
        ]
    )
    profile = prepare_profile(cast)
    assert profile["depth_m"].tolist() == [0.0, 10.0, 20.0, 30.0]
    # Confirms we did not average in the upcast value at 20 m.
    row_20 = profile.loc[profile["depth_m"] == 20.0].iloc[0]
    assert float(row_20["temperature_c"]) == 23.0


def test_prepare_profile_handles_reverse_order_profile() -> None:
    cast = _cast_from_rows(
        [
            {"depth_m": 30.0, "temperature_c": 22.0, "salinity_psu": 35.3},
            {"depth_m": 20.0, "temperature_c": 23.0, "salinity_psu": 35.2},
            {"depth_m": 10.0, "temperature_c": 24.0, "salinity_psu": 35.1},
            {"depth_m": 0.0, "temperature_c": 25.0, "salinity_psu": 35.0},
        ]
    )
    profile = prepare_profile(cast)
    assert profile["depth_m"].tolist() == [0.0, 10.0, 20.0, 30.0]
