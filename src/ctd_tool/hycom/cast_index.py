from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

import pandas as pd

from ctd_tool.discovery import describe_cnv_path, discover_cnv_files
from ctd_tool.profiles import compute_profile_metrics
from ctd_tool.validators import parse_and_validate_cnv


@dataclass
class CastIndexRow:
    cast_id: str
    cast_family_id: str
    source_path: str
    is_binned: bool
    is_selected: bool
    cast_time_utc: str | None
    cast_date_utc: str | None
    lat_deg: float | None
    lon_deg: float | None
    depth_max_m: float | None
    t300_c: float | None
    qc_status: str
    qc_warning_codes: str
    qc_error_codes: str


def build_cast_index(root_dir: str | Path, *, prefer_binned: bool = True) -> pd.DataFrame:
    infos = discover_cnv_files(root_dir)
    rows: list[CastIndexRow] = []

    for info in infos:
        result = parse_and_validate_cnv(info.path, reject_sv_only=True)
        if result.cast is None or result.skipped:
            continue
        cast = result.cast
        metrics = compute_profile_metrics(cast)
        cast_time = _parse_header_time(cast.header.start_time)
        lat_val = _safe_float(cast.data["latitude_deg"].median()) if "latitude_deg" in cast.data.columns else None
        lon_val = _safe_float(cast.data["longitude_deg"].median()) if "longitude_deg" in cast.data.columns else None
        warnings = ",".join(w.code for w in result.report.warnings)
        errors = ",".join(e.code for e in result.report.errors)
        cast_family = _cast_family_id(cast.cast_id)
        rows.append(
            CastIndexRow(
                cast_id=cast.cast_id,
                cast_family_id=cast_family,
                source_path=str(cast.header.source_path),
                is_binned=info.is_binned,
                is_selected=True,
                cast_time_utc=cast_time.isoformat().replace("+00:00", "Z") if cast_time else None,
                cast_date_utc=cast_time.date().isoformat() if cast_time else None,
                lat_deg=lat_val,
                lon_deg=lon_val,
                depth_max_m=_safe_float(metrics["depth_max_m"]),
                t300_c=_safe_float(metrics["t300_c"]),
                qc_status=result.report.status,
                qc_warning_codes=warnings,
                qc_error_codes=errors,
            )
        )

    frame = pd.DataFrame(asdict(r) for r in rows)
    if frame.empty:
        return frame

    if prefer_binned:
        frame["is_selected"] = False
        for _, fam_df in frame.groupby("cast_family_id", sort=False):
            selected_idx = _select_preferred_row_index(fam_df)
            frame.loc[selected_idx, "is_selected"] = True
    else:
        frame["is_selected"] = True

    order_cols = [
        "cast_id",
        "cast_family_id",
        "source_path",
        "is_binned",
        "is_selected",
        "cast_time_utc",
        "cast_date_utc",
        "lat_deg",
        "lon_deg",
        "depth_max_m",
        "t300_c",
        "qc_status",
        "qc_warning_codes",
        "qc_error_codes",
    ]
    return frame.loc[:, order_cols].sort_values(["cast_family_id", "is_binned"]).reset_index(drop=True)


def _cast_family_id(cast_id: str) -> str:
    return re.sub(r"_bin$", "", cast_id, flags=re.IGNORECASE)


def _select_preferred_row_index(fam_df: pd.DataFrame) -> int:
    # Prefer binned product when available, then deepest profile.
    ranked = fam_df.copy()
    ranked["_score_binned"] = ranked["is_binned"].astype(int)
    ranked["_score_depth"] = ranked["depth_max_m"].fillna(-1.0)
    ranked = ranked.sort_values(["_score_binned", "_score_depth"], ascending=[False, False])
    return int(ranked.index[0])


def _parse_header_time(raw: str | None) -> datetime | None:
    if not raw:
        return None
    candidate = raw.split("[", 1)[0].strip()
    candidate = " ".join(candidate.split())
    for fmt in ("%b %d %Y %H:%M:%S",):
        try:
            dt = datetime.strptime(candidate, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(value_f):
        return None
    return value_f
