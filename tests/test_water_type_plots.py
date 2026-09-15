from __future__ import annotations

from pathlib import Path

import pandas as pd

from ctd_tool.plots import generate_water_type_plots


def test_generate_water_type_plots_creates_expected_files(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "cast_id": "A",
                "cast_time_utc": "2023-05-01T00:00:00Z",
                "lat_deg": 28.0,
                "lon_deg": -88.0,
                "t300_c": 12.5,
                "ssha_i_m": -0.2,
                "ssha_gom_m": -0.1,
                "ssha_delta_i_minus_gom_m": -0.1,
                "lcow_index_raw": -3.0,
                "water_type": "CW",
            },
            {
                "cast_id": "B",
                "cast_time_utc": "2023-05-02T00:00:00Z",
                "lat_deg": 28.2,
                "lon_deg": -87.8,
                "t300_c": 14.2,
                "ssha_i_m": -0.05,
                "ssha_gom_m": -0.1,
                "ssha_delta_i_minus_gom_m": 0.05,
                "lcow_index_raw": -1.5,
                "water_type": "MIX",
            },
        ]
    )

    created = generate_water_type_plots(frame, tmp_path)
    names = sorted(path.name for path in created)
    expected = sorted(
        [
            "water_type_map.png",
            "t300_vs_ssha_delta_thresholds.png",
            "lcow_index_hist.png",
            "water_type_time_series.png",
            "threshold_distance.png",
            "water_type_counts.png",
        ]
    )
    assert names == expected
    for name in expected:
        assert (tmp_path / name).exists()
