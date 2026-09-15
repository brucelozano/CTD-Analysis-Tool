from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from ctd_tool.cast import CastHeader, CastRecord


class CnvParseError(ValueError):
    """Raised when a CNV file cannot be parsed."""


NAME_RE = re.compile(r"^# name\s+(\d+)\s*=\s*([^:]+):\s*(.*)$")
INT_RE = re.compile(r"^#\s*([a-zA-Z_]+)\s*=\s*(.+)$")
INTERVAL_RE = re.compile(r"^# interval =\s*([^:]+):\s*([-\d.]+)")

CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "depth_m": ("depSM", "prDM"),
    "temperature_c": ("t090C", "t190C"),
    "salinity_psu": ("sal00", "sal11"),
    "sound_velocity_m_s": ("svCM", "avgsvCM"),
    "latitude_deg": ("latitude",),
    "longitude_deg": ("longitude",),
    "time_s": ("timeS",),
    "conductivity_s_m": ("c0S/m", "c1S/m"),
    "oxygen_mg_l": ("sbeox0Mg/L", "sbeox1Mg/L"),
    "fluorescence_mg_m3": ("flECO-AFL",),
    "flag": ("flag",),
}


@dataclass
class ParsedCnv:
    header: CastHeader
    data: pd.DataFrame
    canonical_map: dict[str, str]
    row_parse_errors: int
    row_length_errors: int
    has_end_marker: bool


def parse_cnv(path: str | Path) -> CastRecord:
    parsed = _parse_cnv(path)
    return CastRecord(
        header=parsed.header,
        data=parsed.data,
        canonical_map=parsed.canonical_map,
        is_sv_only=is_sv_only(parsed.canonical_map, parsed.header.variables),
    )


def parse_cnv_detailed(path: str | Path) -> ParsedCnv:
    return _parse_cnv(path)


def _parse_cnv(path: str | Path) -> ParsedCnv:
    source_path = Path(path)
    if not source_path.exists():
        raise CnvParseError(f"File not found: {source_path}")

    lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_lines: list[str] = []
    data_lines: list[str] = []
    in_data = False
    has_end_marker = False

    for line in lines:
        if not in_data:
            header_lines.append(line)
            if line.strip() == "*END*":
                in_data = True
                has_end_marker = True
            continue
        if line.strip():
            data_lines.append(line)

    if not has_end_marker:
        raise CnvParseError(f"Missing *END* marker in {source_path}")

    header = _parse_header(source_path, header_lines)
    if not header.variables:
        raise CnvParseError(f"Missing '# name N =' column definitions in {source_path}")

    frame, row_parse_errors, row_length_errors = _parse_data_block(data_lines, header.variables)
    frame = _apply_bad_flag(frame, header.bad_flag)
    canonical_map = _canonicalize_columns(frame)

    return ParsedCnv(
        header=header,
        data=frame,
        canonical_map=canonical_map,
        row_parse_errors=row_parse_errors,
        row_length_errors=row_length_errors,
        has_end_marker=has_end_marker,
    )


def _parse_header(source_path: Path, lines: Iterable[str]) -> CastHeader:
    nquan: int | None = None
    nvalues: int | None = None
    start_time: str | None = None
    interval_label: str | None = None
    interval_value: float | None = None
    bad_flag: float | None = None
    variables: dict[int, str] = {}
    variable_long_names: dict[str, str] = {}
    variable_units: dict[str, str | None] = {}
    extra: dict[str, str] = {}

    for line in lines:
        stripped = line.strip()

        name_match = NAME_RE.match(stripped)
        if name_match:
            idx = int(name_match.group(1))
            raw_name = name_match.group(2).strip()
            long_part = name_match.group(3).strip()
            long_name, units = _split_long_name_units(long_part)
            variables[idx] = raw_name
            variable_long_names[raw_name] = long_name
            variable_units[raw_name] = units
            continue

        if stripped.startswith("# start_time ="):
            start_time = stripped.split("=", 1)[1].strip()
            continue

        interval_match = INTERVAL_RE.match(stripped)
        if interval_match:
            interval_label = interval_match.group(1).strip()
            interval_value = float(interval_match.group(2))
            continue

        if stripped.startswith("# bad_flag ="):
            try:
                bad_flag = float(stripped.split("=", 1)[1].strip())
            except ValueError:
                bad_flag = None
            continue

        if stripped.startswith("# nquan ="):
            nquan = _try_int(stripped.split("=", 1)[1].strip())
            continue

        if stripped.startswith("# nvalues ="):
            nvalues = _try_int(stripped.split("=", 1)[1].strip())
            continue

        int_match = INT_RE.match(stripped)
        if int_match:
            extra[int_match.group(1)] = int_match.group(2).strip()

    return CastHeader(
        source_path=source_path,
        start_time=start_time,
        nquan=nquan,
        nvalues=nvalues,
        interval_label=interval_label,
        interval_value=interval_value,
        bad_flag=bad_flag,
        variables=dict(sorted(variables.items(), key=lambda kv: kv[0])),
        variable_long_names=variable_long_names,
        variable_units=variable_units,
        extra_metadata=extra,
    )


def _parse_data_block(data_lines: list[str], variables: dict[int, str]) -> tuple[pd.DataFrame, int, int]:
    expected_cols = len(variables)
    col_names = [variables[i] for i in sorted(variables)]
    rows: list[list[float]] = []
    row_parse_errors = 0
    row_length_errors = 0

    for line in data_lines:
        parts = line.split()
        if len(parts) != expected_cols:
            row_length_errors += 1
            if len(parts) < expected_cols:
                parts = parts + ["nan"] * (expected_cols - len(parts))
            else:
                parts = parts[:expected_cols]

        row_vals: list[float] = []
        for token in parts:
            try:
                row_vals.append(float(token))
            except ValueError:
                row_vals.append(np.nan)
                row_parse_errors += 1
        rows.append(row_vals)

    return pd.DataFrame(rows, columns=col_names), row_parse_errors, row_length_errors


def _apply_bad_flag(frame: pd.DataFrame, bad_flag: float | None) -> pd.DataFrame:
    if bad_flag is None:
        return frame
    return frame.replace(bad_flag, np.nan)


def _canonicalize_columns(frame: pd.DataFrame) -> dict[str, str]:
    canonical_map: dict[str, str] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        raw = next((name for name in aliases if name in frame.columns), None)
        if raw is None:
            continue
        canonical_map[canonical] = raw
        if canonical not in frame.columns:
            frame[canonical] = frame[raw]
    return canonical_map


def is_sv_only(canonical_map: dict[str, str], variables: dict[int, str]) -> bool:
    if "temperature_c" in canonical_map or "salinity_psu" in canonical_map:
        return False
    raw_names = {v for _, v in sorted(variables.items())}
    if raw_names.issubset({"depSM", "svCM", "flag"}):
        return True
    return (
        "depth_m" in canonical_map
        and "sound_velocity_m_s" in canonical_map
        and len(raw_names) <= 4
    )


def _split_long_name_units(raw: str) -> tuple[str, str | None]:
    if "[" not in raw or "]" not in raw:
        return raw.strip(), None
    left = raw[: raw.rfind("[")].strip()
    right = raw[raw.rfind("[") + 1 : raw.rfind("]")].strip()
    return left, right or None


def _try_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None
