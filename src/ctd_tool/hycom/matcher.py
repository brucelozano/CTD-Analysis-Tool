from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ctd_tool.hycom.opendap_client import OpendapSshaClient


def match_cast_index_to_ssha(
    cast_index: pd.DataFrame,
    client: OpendapSshaClient,
    *,
    max_time_delta_minutes: float = 90.0,
    selected_only: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    work = cast_index.copy()
    if selected_only and "is_selected" in work.columns:
        work = work[work["is_selected"] == True]  # noqa: E712

    if limit is not None and limit > 0:
        work = work.head(limit)

    rows: list[dict[str, Any]] = []
    for row in work.to_dict(orient="records"):
        out = _match_one_row(row, client, max_time_delta_minutes=max_time_delta_minutes)
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_match_results(matches: pd.DataFrame) -> dict[str, Any]:
    if matches.empty:
        return {
            "total_casts": 0,
            "ok_matches": 0,
            "error_matches": 0,
            "max_time_delta_min": None,
        }
    status_counts = matches["match_status"].value_counts(dropna=False).to_dict()
    max_delta = pd.to_numeric(matches["time_delta_min"], errors="coerce").max()
    return {
        "total_casts": int(len(matches)),
        "ok_matches": int(status_counts.get("ok", 0)),
        "error_matches": int(status_counts.get("error", 0)),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "max_time_delta_min": float(max_delta) if pd.notna(max_delta) else None,
    }


def verify_ssha_matches(
    matches: pd.DataFrame,
    client: OpendapSshaClient,
    *,
    max_time_delta_minutes: float,
    strict: bool = True,
    spot_check_count: int = 5,
    value_tolerance_m: float = 1e-6,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "total_rows": int(len(matches)),
        "ok_rows": 0,
        "error_rows": 0,
        "duplicate_cast_ids": 0,
        "time_delta_violations": 0,
        "spot_checks_run": 0,
        "spot_checks_failed": 0,
        "status": "ok",
        "messages": [],
    }
    if matches.empty:
        report["status"] = "error"
        report["messages"].append("No rows to verify.")
        return report

    report["ok_rows"] = int((matches["match_status"] == "ok").sum())
    report["error_rows"] = int((matches["match_status"] == "error").sum())
    report["duplicate_cast_ids"] = int(matches["cast_id"].duplicated().sum())

    deltas = pd.to_numeric(matches.get("time_delta_min"), errors="coerce")
    violations = int((deltas > max_time_delta_minutes).fillna(False).sum())
    report["time_delta_violations"] = violations

    if report["duplicate_cast_ids"] > 0:
        report["messages"].append(f"Duplicate cast_id rows: {report['duplicate_cast_ids']}.")
    if violations > 0:
        report["messages"].append(f"Rows exceeding time delta threshold: {violations}.")
    if strict and report["error_rows"] > 0:
        report["messages"].append(f"Strict mode: {report['error_rows']} rows already have match_status=error.")

    ok_rows = matches[matches["match_status"] == "ok"].head(max(spot_check_count, 0))
    for _, row in ok_rows.iterrows():
        cast_time = _parse_utc(row.get("cast_time_utc"))
        lat = _try_float(row.get("lat_deg"))
        lon = _try_float(row.get("lon_deg"))
        if lat is None:
            lat = _try_float(row.get("cast_lat_deg"))
        if lon is None:
            lon = _try_float(row.get("cast_lon_deg"))
        expected_ssha = _try_float(row.get("ssha_i_m"))
        report["spot_checks_run"] += 1
        if cast_time is None or lat is None or lon is None or expected_ssha is None:
            report["spot_checks_failed"] += 1
            continue
        try:
            result = client.match_point(
                cast_time=cast_time,
                cast_lat=lat,
                cast_lon=lon,
                max_time_delta_minutes=max_time_delta_minutes,
            )
        except Exception:
            report["spot_checks_failed"] += 1
            continue
        if result.status != "ok" or result.ssha_i_m is None:
            report["spot_checks_failed"] += 1
            continue
        if abs(result.ssha_i_m - expected_ssha) > value_tolerance_m:
            report["spot_checks_failed"] += 1

    if report["spot_checks_failed"] > 0:
        report["messages"].append(
            f"Spot-check mismatches: {report['spot_checks_failed']} of {report['spot_checks_run']}."
        )

    if strict and (
        report["error_rows"] > 0
        or report["duplicate_cast_ids"] > 0
        or report["time_delta_violations"] > 0
        or report["spot_checks_failed"] > 0
    ):
        report["status"] = "error"
    elif report["messages"]:
        report["status"] = "warning"
    return report


def _match_one_row(
    row: dict[str, Any],
    client: OpendapSshaClient,
    *,
    max_time_delta_minutes: float,
) -> dict[str, Any]:
    cast_id = row.get("cast_id")
    cast_time = _parse_utc(row.get("cast_time_utc"))
    lat = _try_float(row.get("lat_deg"))
    lon = _try_float(row.get("lon_deg"))

    if cast_time is None or lat is None or lon is None:
        return {
            "cast_id": cast_id,
            "cast_family_id": row.get("cast_family_id"),
            "source_path": row.get("source_path"),
            "cast_time_utc": row.get("cast_time_utc"),
            "lat_deg": row.get("lat_deg"),
            "lon_deg": row.get("lon_deg"),
            "t300_c": row.get("t300_c"),
            "match_status": "error",
            "qc_message": "missing_cast_time_or_coordinates",
        }

    try:
        point = client.match_point(
            cast_time=cast_time,
            cast_lat=lat,
            cast_lon=lon,
            max_time_delta_minutes=max_time_delta_minutes,
        )
    except Exception as exc:
        return {
            "cast_id": cast_id,
            "cast_family_id": row.get("cast_family_id"),
            "source_path": row.get("source_path"),
            "cast_time_utc": row.get("cast_time_utc"),
            "lat_deg": row.get("lat_deg"),
            "lon_deg": row.get("lon_deg"),
            "t300_c": row.get("t300_c"),
            "match_status": "error",
            "qc_message": f"match_exception: {exc}",
        }
    try:
        out = asdict(point)
    except Exception as exc:
        return {
            "cast_id": cast_id,
            "cast_family_id": row.get("cast_family_id"),
            "source_path": row.get("source_path"),
            "cast_time_utc": row.get("cast_time_utc"),
            "lat_deg": row.get("lat_deg"),
            "lon_deg": row.get("lon_deg"),
            "t300_c": row.get("t300_c"),
            "match_status": "error",
            "qc_message": f"match_result_serialize_failed: {exc}",
        }
    out.update(
        {
            "cast_id": cast_id,
            "cast_family_id": row.get("cast_family_id"),
            "source_path": row.get("source_path"),
            "is_binned": row.get("is_binned"),
            "is_selected": row.get("is_selected"),
            "cast_date_utc": row.get("cast_date_utc"),
            "t300_c": row.get("t300_c"),
            "depth_max_m": row.get("depth_max_m"),
            "qc_status_local": row.get("qc_status"),
            "qc_warnings_local": row.get("qc_warning_codes"),
            "qc_errors_local": row.get("qc_error_codes"),
            "match_status": out.pop("status"),
        }
    )
    return out


def _parse_utc(value: Any) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _try_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out
