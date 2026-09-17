from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class CnvFileInfo:
    path: Path
    is_sv_folder: bool
    site: str | None
    day_night: str | None
    ctd_number: int | None
    is_binned: bool


CTD_NUM_RE = re.compile(r"_CTD_(\d+)", re.IGNORECASE)


def discover_cnv_files(root_dir: str | Path) -> list[CnvFileInfo]:
    root = Path(root_dir)
    if not root.exists():
        return []

    files = sorted(root.rglob("*.cnv"))
    return [describe_cnv_path(path) for path in files]


def describe_cnv_path(path: Path) -> CnvFileInfo:
    stem = path.stem
    is_sv_folder = any(part.lower() == "sound velocity" for part in path.parts)
    site, day_night = _parse_site_and_daynight(stem)
    ctd_number = _parse_ctd_number(stem)
    is_binned = "_bin" in stem.lower()
    return CnvFileInfo(
        path=path,
        is_sv_folder=is_sv_folder,
        site=site,
        day_night=day_night,
        ctd_number=ctd_number,
        is_binned=is_binned,
    )


def _parse_ctd_number(stem: str) -> int | None:
    match = CTD_NUM_RE.search(stem)
    if not match:
        return None
    return int(match.group(1))


def _parse_site_and_daynight(stem: str) -> tuple[str | None, str | None]:
    normalized = _strip_suffix_tokens(stem)
    tokens = [t for t in normalized.split("_") if t]
    if not tokens:
        return None, None

    # Explicit token labels are highest-confidence.
    explicit = _parse_explicit_day_night(tokens)
    if explicit is not None:
        idx, day_night = explicit
        site = "_".join(tokens[:idx]) or None
        if site:
            return site, day_night

    parts = re.split(r"_CTD_", normalized, flags=re.IGNORECASE)
    prefix = parts[0] if parts else normalized
    prefix_tokens = [t for t in prefix.split("_") if t]
    if not prefix_tokens:
        return None, None

    # Pattern like UTAH_N_CTD_262...
    last_token = prefix_tokens[-1].upper()
    if last_token in {"D", "N"}:
        return "_".join(prefix_tokens[:-1]) or None, last_token

    # Pattern like B082D_CTD_255...
    last = prefix_tokens[-1]
    station_match = re.match(r"^([A-Za-z]+[0-9]+)([DN])$", last, flags=re.IGNORECASE)
    if station_match:
        return station_match.group(1), station_match.group(2).upper()

    return prefix or None, None


def _strip_suffix_tokens(stem: str) -> str:
    out = stem
    changed = True
    while changed:
        changed = False
        for suffix in ("_bin", "_sv", "_converted"):
            if out.lower().endswith(suffix):
                out = out[: -len(suffix)]
                changed = True
    return out


def _parse_explicit_day_night(tokens: list[str]) -> tuple[int, str] | None:
    for i, token in enumerate(tokens):
        upper = token.upper()
        if upper in {"D", "DAY"}:
            return i, "D"
        if upper in {"N", "NIGHT"}:
            return i, "N"
    return None
