from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ctd_tool.cast import CastRecord, ValidationReport
from ctd_tool.cnv_parser import CnvParseError, parse_cnv_detailed

REQUIRED_ANALYSIS_COLUMNS = ("depth_m", "temperature_c", "salinity_psu")


@dataclass
class ValidationResult:
    path: Path
    cast: CastRecord | None
    report: ValidationReport
    skipped: bool = False
    skip_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "cast_id": self.cast.cast_id if self.cast else self.path.stem,
            "status": self.report.status,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "report": self.report.to_dict(),
        }


def parse_and_validate_cnv(
    path: str | Path,
    *,
    reject_sv_only: bool = True,
    min_depth_range_m: float = 50.0,
) -> ValidationResult:
    source_path = Path(path)
    report = ValidationReport()

    try:
        parsed = parse_cnv_detailed(source_path)
    except CnvParseError as exc:
        report.add_error("parse_error", str(exc))
        return ValidationResult(path=source_path, cast=None, report=report)

    cast = CastRecord(
        header=parsed.header,
        data=parsed.data,
        canonical_map=parsed.canonical_map,
        is_sv_only=(
            "depth_m" in parsed.canonical_map
            and "sound_velocity_m_s" in parsed.canonical_map
            and "temperature_c" not in parsed.canonical_map
            and "salinity_psu" not in parsed.canonical_map
        ),
    )

    if parsed.row_length_errors > 0:
        report.add_warning(
            "row_length_mismatch",
            f"{parsed.row_length_errors} row(s) had column count mismatches.",
        )
    if parsed.row_parse_errors > 0:
        report.add_warning(
            "non_numeric_data",
            f"{parsed.row_parse_errors} value(s) could not be parsed as floats.",
        )

    if cast.header.nquan is not None and cast.header.nquan != len(cast.header.variables):
        report.add_error(
            "nquan_mismatch",
            f"Header nquan={cast.header.nquan} but parsed {len(cast.header.variables)} variables.",
        )
    if cast.header.nvalues is not None and cast.header.nvalues != len(cast.data):
        report.add_warning(
            "nvalues_mismatch",
            f"Header nvalues={cast.header.nvalues} but parsed {len(cast.data)} rows.",
        )

    if cast.is_sv_only and reject_sv_only:
        report.add_warning("sv_only_excluded", "SV-only CNV file excluded from default analysis.")
        cast.report = report
        return ValidationResult(
            path=source_path,
            cast=cast,
            report=report,
            skipped=True,
            skip_reason="sv_only_excluded",
        )

    _validate_required_columns(cast, report)
    if report.status != "error":
        _validate_depth_profile(cast, report, min_depth_range_m=min_depth_range_m)
        _validate_physical_ranges(cast, report)

    cast.report = report
    return ValidationResult(path=source_path, cast=cast, report=report)


def _validate_required_columns(cast: CastRecord, report: ValidationReport) -> None:
    missing = [name for name in REQUIRED_ANALYSIS_COLUMNS if name not in cast.data.columns]
    if missing:
        report.add_error(
            "missing_required_columns",
            f"Missing required columns: {', '.join(missing)}.",
        )


def _validate_depth_profile(
    cast: CastRecord,
    report: ValidationReport,
    *,
    min_depth_range_m: float,
) -> None:
    depth = cast.data["depth_m"].astype(float)
    valid_depth = depth.dropna()
    if valid_depth.empty:
        report.add_error("depth_missing", "Depth column contains no valid values.")
        return

    depth_range = float(valid_depth.max() - valid_depth.min())
    if depth_range < min_depth_range_m:
        report.add_warning(
            "insufficient_depth_range",
            f"Depth range is {depth_range:.2f} m, below {min_depth_range_m:.2f} m threshold.",
        )

    diffs = np.diff(valid_depth.to_numpy())
    if np.any(diffs < 0):
        report.add_warning(
            "non_monotonic_depth",
            "Depth values were non-monotonic; likely includes turn-around/upcast rows. Downcast segment is used in analysis.",
        )

    if valid_depth.duplicated().any():
        report.add_warning("duplicate_depths", "Profile contains duplicate depth values.")


def _validate_physical_ranges(cast: CastRecord, report: ValidationReport) -> None:
    temp = cast.data["temperature_c"].astype(float)
    sal = cast.data["salinity_psu"].astype(float)

    temp_bad = int(((temp < -2) | (temp > 40)).fillna(False).sum())
    sal_bad = int(((sal < 0) | (sal > 42)).fillna(False).sum())

    if temp_bad > 0:
        report.add_warning("temperature_range", f"{temp_bad} temperature values are outside -2 to 40 C.")
    if sal_bad > 0:
        report.add_warning("salinity_range", f"{sal_bad} salinity values are outside 0 to 42 PSU.")
