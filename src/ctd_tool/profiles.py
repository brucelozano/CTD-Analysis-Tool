from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ctd_tool.cast import CastRecord


def compute_profile_metrics(
    cast: CastRecord,
    *,
    surface_window_m: float = 10.0,
    deep_window_m: float = 10.0,
    mixed_layer_delta_t_c: float = 0.5,
) -> dict[str, Any]:
    profile = prepare_profile(cast)
    metrics: dict[str, Any] = {
        "cast_id": cast.cast_id,
        "source_path": str(cast.header.source_path),
        "n_rows_total": int(len(cast.data)),
        "n_rows_valid_core": int(len(profile)),
        "depth_min_m": np.nan,
        "depth_max_m": np.nan,
        "surface_temp_c": np.nan,
        "surface_sal_psu": np.nan,
        "deep_temp_c": np.nan,
        "deep_sal_psu": np.nan,
        "t300_c": np.nan,
        "thermocline_depth_m": np.nan,
        "thermocline_gradient_c_per_m": np.nan,
        "mixed_layer_depth_m": np.nan,
        "sound_velocity_min_m_s": np.nan,
        "sound_velocity_max_m_s": np.nan,
    }

    if profile.empty:
        return metrics

    depth = profile["depth_m"].to_numpy()
    temp = profile["temperature_c"].to_numpy()
    sal = profile["salinity_psu"].to_numpy()

    metrics["depth_min_m"] = float(np.nanmin(depth))
    metrics["depth_max_m"] = float(np.nanmax(depth))

    surface_limit = metrics["depth_min_m"] + surface_window_m
    deep_limit = metrics["depth_max_m"] - deep_window_m

    surface_mask = depth <= surface_limit
    deep_mask = depth >= deep_limit

    if np.any(surface_mask):
        metrics["surface_temp_c"] = float(np.nanmedian(temp[surface_mask]))
        metrics["surface_sal_psu"] = float(np.nanmedian(sal[surface_mask]))
    if np.any(deep_mask):
        metrics["deep_temp_c"] = float(np.nanmedian(temp[deep_mask]))
        metrics["deep_sal_psu"] = float(np.nanmedian(sal[deep_mask]))

    metrics["t300_c"] = _interpolate_temperature_at_depth(depth, temp, target_depth_m=300.0)
    thermo_depth, thermo_grad = _thermocline(depth, temp)
    metrics["thermocline_depth_m"] = thermo_depth
    metrics["thermocline_gradient_c_per_m"] = thermo_grad
    metrics["mixed_layer_depth_m"] = _mixed_layer_depth(depth, temp, metrics["surface_temp_c"], mixed_layer_delta_t_c)

    if "sound_velocity_m_s" in cast.data.columns:
        sv = cast.data["sound_velocity_m_s"].astype(float)
        if sv.notna().any():
            metrics["sound_velocity_min_m_s"] = float(np.nanmin(sv))
            metrics["sound_velocity_max_m_s"] = float(np.nanmax(sv))

    return metrics


def prepare_profile(cast: CastRecord) -> pd.DataFrame:
    cols = ["depth_m", "temperature_c", "salinity_psu"]
    profile = cast.data.loc[:, cols].copy()
    profile = profile.apply(pd.to_numeric, errors="coerce")
    profile = profile.dropna(subset=cols)
    if profile.empty:
        return profile
    profile = _extract_downcast_segment(profile)
    if profile.empty:
        return profile
    profile = profile.sort_values("depth_m").reset_index(drop=True)
    profile = profile.groupby("depth_m", as_index=False).mean(numeric_only=True)
    return profile


def _extract_downcast_segment(profile: pd.DataFrame) -> pd.DataFrame:
    depth = profile["depth_m"].to_numpy()
    if len(depth) <= 1:
        return profile.reset_index(drop=True)

    max_idx = int(np.nanargmax(depth))
    if max_idx == 0:
        # Some processed files can be written in reverse row order (deep -> shallow).
        # Reverse first so the profile is shallow -> deep before downstream processing.
        return profile.iloc[::-1].reset_index(drop=True)

    # Standard case: keep only the downcast portion up to deepest sample.
    return profile.iloc[: max_idx + 1].reset_index(drop=True)


def _interpolate_temperature_at_depth(depth: np.ndarray, temp: np.ndarray, *, target_depth_m: float) -> float:
    if np.nanmin(depth) > target_depth_m or np.nanmax(depth) < target_depth_m:
        return np.nan
    return float(np.interp(target_depth_m, depth, temp))


def _thermocline(depth: np.ndarray, temp: np.ndarray) -> tuple[float, float]:
    if len(depth) < 3:
        return np.nan, np.nan
    grad = np.gradient(temp, depth)
    idx = int(np.nanargmin(grad))
    return float(depth[idx]), float(grad[idx])


def _mixed_layer_depth(
    depth: np.ndarray,
    temp: np.ndarray,
    surface_temp_c: float,
    delta_t_c: float,
) -> float:
    if np.isnan(surface_temp_c):
        return np.nan
    threshold = surface_temp_c - delta_t_c
    crossed = np.where(temp <= threshold)[0]
    if crossed.size == 0:
        return float(depth[-1])
    return float(depth[int(crossed[0])])
