from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SSHADataSource:
    key: str
    description: str
    url_template: str
    variable_name: str
    time_name: str
    lat_name: str
    lon_name: str
    engine: str | None = "pydap"
    gulf_lat_min: float = 18.0
    gulf_lat_max: float = 32.0
    gulf_lon_min: float = -98.0
    gulf_lon_max: float = -77.0
    gulf_stride: int = 4


HYCOM_GOM_REANALYSIS = SSHADataSource(
    key="hycom_gom_reanalysis_2d",
    description="HYCOM-TSIS 1/25 degree Gulf of Mexico reanalysis (2d hourly).",
    url_template="https://tds.hycom.org/thredds/dodsC/GOMb0.04/reanalysis/{year}/2d",
    variable_name="ssh",
    time_name="MT",
    lat_name="Latitude",
    lon_name="Longitude",
    engine="pydap",
)


SOURCES: dict[str, SSHADataSource] = {
    HYCOM_GOM_REANALYSIS.key: HYCOM_GOM_REANALYSIS,
}


def get_source_by_key(key: str) -> SSHADataSource:
    if key not in SOURCES:
        known = ", ".join(sorted(SOURCES))
        raise KeyError(f"Unknown source '{key}'. Available: {known}")
    return SOURCES[key]
