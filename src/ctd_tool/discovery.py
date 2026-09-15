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
    parts = re.split(r"_CTD_", stem, flags=re.IGNORECASE)
    if not parts:
        return None, None
    prefix = parts[0]
    tokens = prefix.split("_")
    if not tokens:
        return None, None

    # Pattern like UTAH_N_CTD_262...
    if tokens[-1].upper() in {"D", "N"}:
        return "_".join(tokens[:-1]) or None, tokens[-1].upper()

    # Pattern like B082D_CTD_255...
    last = tokens[-1]
    if len(last) > 1 and last[-1].upper() in {"D", "N"}:
        return "_".join(tokens[:-1] + [last[:-1]]) or None, last[-1].upper()

    return prefix or None, None
