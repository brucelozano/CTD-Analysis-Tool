from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

from ctd_tool.hycom.sources import SSHADataSource


@dataclass
class SshaPointMatch:
    status: str
    source_key: str
    source_url: str
    variable_name: str
    cast_time_utc: str
    cast_lat_deg: float
    cast_lon_deg: float
    query_lon_deg: float
    matched_time_utc: str | None = None
    time_delta_min: float | None = None
    ssha_i_m: float | None = None
    ssha_gom_m: float | None = None
    ssha_delta_m: float | None = None
    match_method: str = "nearest"
    qc_message: str | None = None


class OpendapSshaClient:
    def __init__(
        self,
        source: SSHADataSource,
        *,
        dataset_loader: Callable[[int], xr.Dataset] | None = None,
    ) -> None:
        self.source = source
        self._dataset_loader = dataset_loader
        self._dataset_cache: dict[int, xr.Dataset] = {}
        self._daily_mean_cache: dict[tuple[int, str], float] = {}

    def match_point(
        self,
        *,
        cast_time: datetime,
        cast_lat: float,
        cast_lon: float,
        max_time_delta_minutes: float,
    ) -> SshaPointMatch:
        cast_time = _ensure_utc(cast_time)
        year = cast_time.year
        source_url = self.source.url_template.format(year=year)

        try:
            ds = self._get_dataset(year)
        except Exception as exc:
            return SshaPointMatch(
                status="error",
                source_key=self.source.key,
                source_url=source_url,
                variable_name=self.source.variable_name,
                cast_time_utc=_iso(cast_time),
                cast_lat_deg=cast_lat,
                cast_lon_deg=cast_lon,
                query_lon_deg=cast_lon,
                qc_message=f"dataset_open_failed: {exc}",
            )

        try:
            var = ds[self.source.variable_name]
            lat_coord = ds[self.source.lat_name]
            lon_coord = ds[self.source.lon_name]
            time_coord = ds[self.source.time_name]
        except KeyError as exc:
            return SshaPointMatch(
                status="error",
                source_key=self.source.key,
                source_url=source_url,
                variable_name=self.source.variable_name,
                cast_time_utc=_iso(cast_time),
                cast_lat_deg=cast_lat,
                cast_lon_deg=cast_lon,
                query_lon_deg=cast_lon,
                qc_message=f"missing_coordinate_or_variable: {exc}",
            )

        query_lon = _convert_lon_for_dataset(cast_lon, lon_coord)
        lat_ok = _in_bounds(cast_lat, lat_coord)
        lon_ok = _in_bounds(query_lon, lon_coord)
        if not lat_ok or not lon_ok:
            return SshaPointMatch(
                status="error",
                source_key=self.source.key,
                source_url=source_url,
                variable_name=self.source.variable_name,
                cast_time_utc=_iso(cast_time),
                cast_lat_deg=cast_lat,
                cast_lon_deg=cast_lon,
                query_lon_deg=query_lon,
                qc_message="coordinate_out_of_bounds",
            )

        query_time = _query_timestamp_for_coord(cast_time, time_coord)
        point = var.sel(
            {
                self.source.time_name: query_time,
                self.source.lat_name: cast_lat,
                self.source.lon_name: query_lon,
            },
            method="nearest",
        )

        matched_time = pd.Timestamp(point[self.source.time_name].values).to_pydatetime()
        matched_time = _ensure_utc(matched_time)
        delta_min = abs((matched_time - cast_time).total_seconds()) / 60.0
        if delta_min > max_time_delta_minutes:
            return SshaPointMatch(
                status="error",
                source_key=self.source.key,
                source_url=source_url,
                variable_name=self.source.variable_name,
                cast_time_utc=_iso(cast_time),
                cast_lat_deg=cast_lat,
                cast_lon_deg=cast_lon,
                query_lon_deg=query_lon,
                matched_time_utc=_iso(matched_time),
                time_delta_min=delta_min,
                qc_message=f"time_delta_exceeds_threshold({max_time_delta_minutes})",
            )

        ssha_i = _clean_scalar(float(point.values), var.attrs)
        if ssha_i is None:
            return SshaPointMatch(
                status="error",
                source_key=self.source.key,
                source_url=source_url,
                variable_name=self.source.variable_name,
                cast_time_utc=_iso(cast_time),
                cast_lat_deg=cast_lat,
                cast_lon_deg=cast_lon,
                query_lon_deg=query_lon,
                matched_time_utc=_iso(matched_time),
                time_delta_min=delta_min,
                qc_message="ssha_value_nan",
            )

        try:
            ssha_gom = self._daily_gom_mean(ds, cast_time.date())
        except Exception as exc:
            return SshaPointMatch(
                status="error",
                source_key=self.source.key,
                source_url=source_url,
                variable_name=self.source.variable_name,
                cast_time_utc=_iso(cast_time),
                cast_lat_deg=cast_lat,
                cast_lon_deg=cast_lon,
                query_lon_deg=query_lon,
                matched_time_utc=_iso(matched_time),
                time_delta_min=delta_min,
                ssha_i_m=ssha_i,
                qc_message=f"gom_mean_failed: {exc}",
            )

        return SshaPointMatch(
            status="ok",
            source_key=self.source.key,
            source_url=source_url,
            variable_name=self.source.variable_name,
            cast_time_utc=_iso(cast_time),
            cast_lat_deg=cast_lat,
            cast_lon_deg=cast_lon,
            query_lon_deg=query_lon,
            matched_time_utc=_iso(matched_time),
            time_delta_min=delta_min,
            ssha_i_m=ssha_i,
            ssha_gom_m=ssha_gom,
            ssha_delta_m=ssha_i - ssha_gom,
        )

    def _get_dataset(self, year: int) -> xr.Dataset:
        if year in self._dataset_cache:
            return self._dataset_cache[year]

        if self._dataset_loader is not None:
            ds = self._dataset_loader(year)
        else:
            url = self.source.url_template.format(year=year)
            ds = self._open_remote_dataset(url)

        self._dataset_cache[year] = ds
        return ds

    def _open_remote_dataset(self, url: str) -> xr.Dataset:
        # pydap engine is generally the safest xarray option for HTTPS OPeNDAP.
        if self.source.engine:
            return xr.open_dataset(url, engine=self.source.engine, decode_times=True)
        return xr.open_dataset(url, decode_times=True)

    def _daily_gom_mean(self, ds: xr.Dataset, cast_date: date) -> float:
        key = (cast_date.year, cast_date.isoformat())
        if key in self._daily_mean_cache:
            return self._daily_mean_cache[key]

        var = ds[self.source.variable_name]
        time_coord = ds[self.source.time_name]
        day_start = datetime(cast_date.year, cast_date.month, cast_date.day, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1) - timedelta(seconds=1)
        query_start, query_end = _query_time_slice_for_coord(day_start, day_end, time_coord)

        day_slice = var.sel({self.source.time_name: slice(query_start, query_end)})
        if day_slice.sizes.get(self.source.time_name, 0) == 0:
            raise ValueError("no_time_steps_for_day")

        lat_coord = ds[self.source.lat_name]
        lon_coord = ds[self.source.lon_name]
        lon_min = _convert_lon_for_dataset(self.source.gulf_lon_min, lon_coord)
        lon_max = _convert_lon_for_dataset(self.source.gulf_lon_max, lon_coord)

        lat_slice = _build_slice_for_coord(lat_coord, self.source.gulf_lat_min, self.source.gulf_lat_max)
        lon_slice = _build_slice_for_coord(lon_coord, lon_min, lon_max)
        gom = day_slice.sel(
            {
                self.source.lat_name: lat_slice,
                self.source.lon_name: lon_slice,
            }
        )
        if self.source.gulf_stride > 1:
            gom = gom.isel(
                {
                    self.source.lat_name: slice(None, None, self.source.gulf_stride),
                    self.source.lon_name: slice(None, None, self.source.gulf_stride),
                }
            )
        gom = _mask_dataarray(gom, var.attrs)
        value = float(gom.mean(skipna=True).values)
        if np.isnan(value):
            raise ValueError("gom_mean_nan")
        self._daily_mean_cache[key] = value
        return value


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _ensure_utc(value).isoformat().replace("+00:00", "Z")


def _query_timestamp_for_coord(cast_time: datetime, time_coord: xr.DataArray) -> pd.Timestamp:
    time_index = time_coord.to_index()
    if len(time_index) == 0:
        raise ValueError("empty_time_coordinate")
    cast_ts = pd.Timestamp(cast_time)
    sample_tz = getattr(time_index, "tz", None)
    if sample_tz is None:
        return cast_ts.tz_localize(None)
    if cast_ts.tz is None:
        cast_ts = cast_ts.tz_localize("UTC")
    return cast_ts.tz_convert(sample_tz)


def _query_time_slice_for_coord(
    start_time: datetime,
    end_time: datetime,
    time_coord: xr.DataArray,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    time_index = time_coord.to_index()
    sample_tz = getattr(time_index, "tz", None)
    start_ts = pd.Timestamp(start_time)
    end_ts = pd.Timestamp(end_time)
    if sample_tz is None:
        return start_ts.tz_localize(None), end_ts.tz_localize(None)
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tz is None:
        end_ts = end_ts.tz_localize("UTC")
    return start_ts.tz_convert(sample_tz), end_ts.tz_convert(sample_tz)


def _convert_lon_for_dataset(lon: float, lon_coord: xr.DataArray) -> float:
    lon_vals = np.asarray(lon_coord.values, dtype=float)
    lon_min = float(np.nanmin(lon_vals))
    lon_max = float(np.nanmax(lon_vals))
    if lon_min >= 0.0 and lon < 0.0:
        return lon + 360.0
    if lon_max <= 180.0 and lon > 180.0:
        return lon - 360.0
    return lon


def _in_bounds(value: float, coord: xr.DataArray) -> bool:
    vals = np.asarray(coord.values, dtype=float)
    return float(np.nanmin(vals)) <= value <= float(np.nanmax(vals))


def _build_slice_for_coord(coord: xr.DataArray, min_value: float, max_value: float) -> slice:
    vals = np.asarray(coord.values, dtype=float)
    ascending = vals[0] <= vals[-1]
    if ascending:
        return slice(min_value, max_value)
    return slice(max_value, min_value)


def _mask_dataarray(arr: xr.DataArray, attrs: dict) -> xr.DataArray:
    out = arr
    for key in ("_FillValue", "missing_value"):
        if key in attrs:
            try:
                fill_val = float(attrs[key])
            except (TypeError, ValueError):
                continue
            out = out.where(out != fill_val)

    if "valid_range" in attrs:
        vr = attrs["valid_range"]
        try:
            lo = float(vr[0])
            hi = float(vr[1])
            out = out.where((out >= lo) & (out <= hi))
        except (TypeError, ValueError, IndexError):
            pass

    out = out.where(np.abs(out) < 1e20)
    return out


def _clean_scalar(value: float, attrs: dict) -> float | None:
    if np.isnan(value):
        return None
    for key in ("_FillValue", "missing_value"):
        if key in attrs:
            try:
                fill_val = float(attrs[key])
            except (TypeError, ValueError):
                continue
            if value == fill_val:
                return None
    if abs(value) >= 1e20:
        return None
    if "valid_range" in attrs:
        vr = attrs["valid_range"]
        try:
            lo = float(vr[0])
            hi = float(vr[1])
            if value < lo or value > hi:
                return None
        except (TypeError, ValueError, IndexError):
            pass
    return value
