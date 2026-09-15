from __future__ import annotations

import pandas as pd
import xarray as xr

from ctd_tool.hycom.matcher import match_cast_index_to_ssha, verify_ssha_matches
from ctd_tool.hycom.opendap_client import OpendapSshaClient
from ctd_tool.hycom.sources import SSHADataSource


def _source() -> SSHADataSource:
    return SSHADataSource(
        key="test_source",
        description="test",
        url_template="test://{year}",
        variable_name="ssh",
        time_name="time",
        lat_name="lat",
        lon_name="lon",
        engine=None,
        gulf_lat_min=27.8,
        gulf_lat_max=28.1,
        gulf_lon_min=-88.2,
        gulf_lon_max=-87.9,
    )


def _dataset_loader(_: int) -> xr.Dataset:
    times = pd.to_datetime(["2023-05-08T07:00:00Z", "2023-05-08T08:00:00Z"])
    ds = xr.Dataset(
        data_vars={
            "ssh": (
                ("time", "lat", "lon"),
                [
                    [[0.10, 0.20], [0.30, 0.40]],
                    [[0.11, 0.21], [0.31, 0.41]],
                ],
            )
        },
        coords={
            "time": times,
            "lat": [27.9, 28.0],
            "lon": [-88.1, -88.0],
        },
    )
    return ds


def test_match_cast_index_success() -> None:
    cast_index = pd.DataFrame(
        [
            {
                "cast_id": "CAST_A",
                "cast_family_id": "CAST_A",
                "source_path": "x.cnv",
                "is_binned": True,
                "is_selected": True,
                "cast_time_utc": "2023-05-08T07:10:00Z",
                "cast_date_utc": "2023-05-08",
                "lat_deg": 27.95,
                "lon_deg": -88.05,
                "t300_c": 12.3,
                "depth_max_m": 1500.0,
                "qc_status": "ok",
                "qc_warning_codes": "",
                "qc_error_codes": "",
            }
        ]
    )
    client = OpendapSshaClient(_source(), dataset_loader=_dataset_loader)
    matches = match_cast_index_to_ssha(cast_index, client, max_time_delta_minutes=90.0)
    assert len(matches) == 1
    row = matches.iloc[0]
    assert row["match_status"] == "ok"
    assert row["ssha_i_m"] is not None
    assert row["ssha_gom_m"] is not None


def test_match_cast_index_time_threshold_error() -> None:
    cast_index = pd.DataFrame(
        [
            {
                "cast_id": "CAST_B",
                "cast_family_id": "CAST_B",
                "source_path": "x.cnv",
                "is_binned": True,
                "is_selected": True,
                "cast_time_utc": "2023-05-08T11:30:00Z",
                "cast_date_utc": "2023-05-08",
                "lat_deg": 27.95,
                "lon_deg": -88.05,
                "t300_c": 12.3,
                "depth_max_m": 1500.0,
                "qc_status": "ok",
                "qc_warning_codes": "",
                "qc_error_codes": "",
            }
        ]
    )
    client = OpendapSshaClient(_source(), dataset_loader=_dataset_loader)
    matches = match_cast_index_to_ssha(cast_index, client, max_time_delta_minutes=30.0)
    assert matches.iloc[0]["match_status"] == "error"


def test_verify_matches_strict() -> None:
    cast_index = pd.DataFrame(
        [
            {
                "cast_id": "CAST_C",
                "cast_family_id": "CAST_C",
                "source_path": "x.cnv",
                "is_binned": True,
                "is_selected": True,
                "cast_time_utc": "2023-05-08T07:00:00Z",
                "cast_date_utc": "2023-05-08",
                "lat_deg": 27.9,
                "lon_deg": -88.1,
                "t300_c": 12.3,
                "depth_max_m": 1500.0,
                "qc_status": "ok",
                "qc_warning_codes": "",
                "qc_error_codes": "",
            }
        ]
    )
    client = OpendapSshaClient(_source(), dataset_loader=_dataset_loader)
    matches = match_cast_index_to_ssha(cast_index, client, max_time_delta_minutes=90.0)
    report = verify_ssha_matches(matches, client, max_time_delta_minutes=90.0, strict=True, spot_check_count=1)
    assert report["status"] == "ok"
    assert report["spot_checks_failed"] == 0


def test_match_cast_index_handles_client_exception_per_row() -> None:
    class ExplodingClient:
        def match_point(self, **_: object) -> object:
            raise RuntimeError("boom")

    cast_index = pd.DataFrame(
        [
            {
                "cast_id": "CAST_ERR",
                "cast_family_id": "CAST_ERR",
                "source_path": "x.cnv",
                "is_binned": True,
                "is_selected": True,
                "cast_time_utc": "2023-05-08T07:00:00Z",
                "cast_date_utc": "2023-05-08",
                "lat_deg": 27.9,
                "lon_deg": -88.1,
                "t300_c": 12.3,
                "depth_max_m": 1500.0,
                "qc_status": "ok",
                "qc_warning_codes": "",
                "qc_error_codes": "",
            }
        ]
    )
    matches = match_cast_index_to_ssha(cast_index, ExplodingClient(), max_time_delta_minutes=90.0)
    assert len(matches) == 1
    assert matches.iloc[0]["match_status"] == "error"
    assert "match_exception" in str(matches.iloc[0]["qc_message"])
