from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ctd_tool.cast import CastRecord
from ctd_tool.discovery import describe_cnv_path
from ctd_tool.profiles import prepare_profile

if "MPLCONFIGDIR" not in os.environ:
    local_mpl_dir = Path.cwd() / ".mplconfig"
    local_mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(local_mpl_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_cast_profiles(cast: CastRecord, output_dir: str | Path) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = prepare_profile(cast)
    if profile.empty:
        return []

    created: list[Path] = []
    depth = profile["depth_m"].to_numpy()
    temp = profile["temperature_c"].to_numpy()
    sal = profile["salinity_psu"].to_numpy()

    # Temperature-Depth
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(temp, depth, color="tab:red", lw=1.5)
    ax.invert_yaxis()
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"{cast.cast_id} - T-D Profile")
    ax.grid(alpha=0.25)
    td_path = out_dir / f"{cast.cast_id}_TD.png"
    fig.tight_layout()
    fig.savefig(td_path, dpi=180)
    plt.close(fig)
    created.append(td_path)

    # Salinity-Depth
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(sal, depth, color="tab:blue", lw=1.5)
    ax.invert_yaxis()
    ax.set_xlabel("Salinity (PSU)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"{cast.cast_id} - S-D Profile")
    ax.grid(alpha=0.25)
    sd_path = out_dir / f"{cast.cast_id}_SD.png"
    fig.tight_layout()
    fig.savefig(sd_path, dpi=180)
    plt.close(fig)
    created.append(sd_path)

    # Temperature-Salinity (color by depth)
    fig, ax = plt.subplots(figsize=(6, 5))
    points = ax.scatter(sal, temp, c=depth, s=8, cmap="viridis")
    ax.set_xlabel("Salinity (PSU)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title(f"{cast.cast_id} - T-S Diagram")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(points, ax=ax)
    cbar.set_label("Depth (m)")
    ts_path = out_dir / f"{cast.cast_id}_TS.png"
    fig.tight_layout()
    fig.savefig(ts_path, dpi=180)
    plt.close(fig)
    created.append(ts_path)

    return created


def plot_day_night_overlay(casts: list[CastRecord], output_dir: str | Path, site_name: str) -> Path | None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    day_profiles = []
    night_profiles = []
    for cast in casts:
        profile = prepare_profile(cast)
        if profile.empty:
            continue
        day_night = _infer_day_night_from_cast(cast)
        if day_night == "D":
            day_profiles.append((cast.cast_id, profile))
        elif day_night == "N":
            night_profiles.append((cast.cast_id, profile))

    if not day_profiles and not night_profiles:
        return None

    fig, ax = plt.subplots(figsize=(6, 8))
    for cast_id, profile in day_profiles:
        ax.plot(profile["temperature_c"], profile["depth_m"], color="tab:red", alpha=0.5, label=f"{cast_id} day")
    for cast_id, profile in night_profiles:
        ax.plot(profile["temperature_c"], profile["depth_m"], color="tab:blue", alpha=0.5, label=f"{cast_id} night")

    ax.invert_yaxis()
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"{site_name} Day vs Night T-D Overlay")
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        # Keep only first 8 labels to avoid unreadable legends with many casts.
        ax.legend(handles[:8], labels[:8], fontsize=8)

    out_path = out_dir / f"{site_name}_day_night_overlay_TD.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _infer_day_night_from_cast(cast: CastRecord) -> str | None:
    info = describe_cnv_path(cast.header.source_path)
    if info.day_night in {"D", "N"}:
        return info.day_night

    # Fallback for unusual naming where site parser may fail.
    cast_upper = cast.cast_id.upper()
    if "_D_" in cast_upper:
        return "D"
    if "_N_" in cast_upper:
        return "N"
    return None


WATER_TYPE_COLORS = {
    "LCOW": "tab:red",
    "MIX": "tab:orange",
    "CW": "tab:blue",
    "UNKNOWN": "tab:gray",
}


def generate_water_type_plots(classified: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _prepare_water_type_frame(classified)
    created: list[Path] = []

    created.extend(_plot_water_type_map(frame, out_dir))
    created.extend(_plot_t300_vs_ssha(frame, out_dir))
    created.extend(_plot_lcow_hist(frame, out_dir))
    created.extend(_plot_water_type_time_series(frame, out_dir))
    created.extend(_plot_threshold_distance(frame, out_dir))
    created.extend(_plot_water_type_counts(frame, out_dir))
    return created


def _prepare_water_type_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "cast_lat_deg" in out.columns and "lat_deg" not in out.columns:
        out["lat_deg"] = pd.to_numeric(out["cast_lat_deg"], errors="coerce")
    if "cast_lon_deg" in out.columns and "lon_deg" not in out.columns:
        out["lon_deg"] = pd.to_numeric(out["cast_lon_deg"], errors="coerce")

    for col in ("t300_c", "ssha_i_m", "ssha_gom_m", "lcow_index_raw"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan

    if "ssha_delta_i_minus_gom_m" not in out.columns:
        out["ssha_delta_i_minus_gom_m"] = out["ssha_i_m"] - out["ssha_gom_m"]
    else:
        out["ssha_delta_i_minus_gom_m"] = pd.to_numeric(out["ssha_delta_i_minus_gom_m"], errors="coerce")

    if "water_type" not in out.columns:
        out["water_type"] = "UNKNOWN"

    if "cast_time_utc" in out.columns:
        out["cast_time_dt"] = pd.to_datetime(out["cast_time_utc"], errors="coerce", utc=True)
    else:
        out["cast_time_dt"] = pd.NaT
    return out


def _plot_water_type_map(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    if not {"lat_deg", "lon_deg"}.issubset(frame.columns):
        return []
    plot_df = frame.dropna(subset=["lat_deg", "lon_deg"])
    if plot_df.empty:
        return []

    fig, ax = plt.subplots(figsize=(7, 6))
    for wt, sub in plot_df.groupby("water_type", dropna=False):
        color = WATER_TYPE_COLORS.get(str(wt), "tab:gray")
        ax.scatter(sub["lon_deg"], sub["lat_deg"], s=35, alpha=0.85, color=color, label=str(wt))
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Water Type by Cast Location")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    out_path = out_dir / "water_type_map.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return [out_path]


def _plot_t300_vs_ssha(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    needed = {"t300_c", "ssha_delta_i_minus_gom_m"}
    if not needed.issubset(frame.columns):
        return []
    plot_df = frame.dropna(subset=["t300_c", "ssha_delta_i_minus_gom_m"])
    if plot_df.empty:
        return []

    fig, ax = plt.subplots(figsize=(7, 6))
    for wt, sub in plot_df.groupby("water_type", dropna=False):
        color = WATER_TYPE_COLORS.get(str(wt), "tab:gray")
        ax.scatter(
            sub["ssha_delta_i_minus_gom_m"],
            sub["t300_c"],
            s=38,
            alpha=0.85,
            color=color,
            label=str(wt),
        )
    ax.axvline(0.0, color="black", ls="--", lw=1, alpha=0.8)
    ax.axvline(0.067, color="black", ls=":", lw=1.2, alpha=0.8)
    ax.axhline(13.46, color="black", ls="--", lw=1, alpha=0.8)
    ax.axhline(15.92, color="black", ls=":", lw=1.2, alpha=0.8)
    ax.set_xlabel("SSHA_i - SSHA_GOM (m)")
    ax.set_ylabel("T300 (C)")
    ax.set_title("T300 vs SSHA Delta with Classification Thresholds")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    out_path = out_dir / "t300_vs_ssha_delta_thresholds.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return [out_path]


def _plot_lcow_hist(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    if "lcow_index_raw" not in frame.columns:
        return []
    vals = frame["lcow_index_raw"].dropna()
    if vals.empty:
        return []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(vals, bins=min(12, max(4, len(vals))), color="tab:purple", alpha=0.8, edgecolor="black")
    ax.set_xlabel("LCOW Index (raw)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of LCOW Index")
    ax.grid(alpha=0.2)
    out_path = out_dir / "lcow_index_hist.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return [out_path]


def _plot_water_type_time_series(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    if "cast_time_dt" not in frame.columns:
        return []
    plot_df = frame.dropna(subset=["cast_time_dt"]).sort_values("cast_time_dt")
    if plot_df.empty:
        return []

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(plot_df["cast_time_dt"], plot_df["t300_c"], marker="o", lw=1.3, color="tab:red")
    axes[0].axhline(13.46, color="black", ls="--", lw=0.9)
    axes[0].axhline(15.92, color="black", ls=":", lw=0.9)
    axes[0].set_ylabel("T300 (C)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(plot_df["cast_time_dt"], plot_df["ssha_i_m"], marker="o", lw=1.3, color="tab:blue", label="SSHA_i")
    axes[1].plot(plot_df["cast_time_dt"], plot_df["ssha_gom_m"], marker="o", lw=1.3, color="tab:gray", label="SSHA_GOM")
    axes[1].set_ylabel("SSHA (m)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    axes[2].plot(plot_df["cast_time_dt"], plot_df["lcow_index_raw"], marker="o", lw=1.3, color="tab:green")
    axes[2].set_ylabel("LCOW Index")
    axes[2].set_xlabel("Cast Time (UTC)")
    axes[2].grid(alpha=0.25)

    fig.suptitle("Water-Type Inputs and LCOW Index Over Time")
    out_path = out_dir / "water_type_time_series.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return [out_path]


def _plot_threshold_distance(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    plot_df = frame.copy()
    plot_df["dist_t300_to_lcow_c"] = plot_df["t300_c"] - 15.92
    plot_df["dist_ssha_to_lcow_m"] = plot_df["ssha_i_m"] - (plot_df["ssha_gom_m"] + 0.067)
    plot_df = plot_df.dropna(subset=["dist_t300_to_lcow_c", "dist_ssha_to_lcow_m"])
    if plot_df.empty:
        return []

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(plot_df))
    ax.plot(x, plot_df["dist_t300_to_lcow_c"], marker="o", lw=1.3, label="T300 - 15.92", color="tab:red")
    ax.plot(x, plot_df["dist_ssha_to_lcow_m"], marker="o", lw=1.3, label="SSHA_i - (SSHA_GOM + 0.067)", color="tab:blue")
    ax.axhline(0.0, color="black", ls="--", lw=1)
    ax.set_xlabel("Cast index (sorted input order)")
    ax.set_ylabel("Distance to LCOW threshold")
    ax.set_title("Distance to LCOW Thresholds")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    out_path = out_dir / "threshold_distance.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return [out_path]


def _plot_water_type_counts(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    counts = frame["water_type"].fillna("UNKNOWN").value_counts()
    if counts.empty:
        return []

    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = counts.index.astype(str).tolist()
    colors = [WATER_TYPE_COLORS.get(label, "tab:gray") for label in labels]
    ax.bar(labels, counts.values, color=colors, alpha=0.9)
    ax.set_xlabel("Water Type")
    ax.set_ylabel("Count")
    ax.set_title("Water Type Counts")
    ax.grid(axis="y", alpha=0.25)
    out_path = out_dir / "water_type_counts.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return [out_path]
