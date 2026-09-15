from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ctd_tool.cnv_parser import CnvParseError, is_sv_only, parse_cnv_detailed
from ctd_tool.discovery import describe_cnv_path, discover_cnv_files
from ctd_tool.profiles import compute_profile_metrics
from ctd_tool.validators import ValidationResult, parse_and_validate_cnv
from ctd_tool.water_types import classification_summary, classify_water_types
from ctd_tool.hycom.cast_index import build_cast_index


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ctd-tool",
        description="CNV-first CTD analysis tool.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the end-to-end pipeline with simple presets.")
    run_p.add_argument("root_dir", type=Path, help="Directory containing processed CNV files.")
    run_p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root directory (default: out/run_YYYYmmdd_HHMMSSZ).",
    )
    run_p.add_argument(
        "--preset",
        choices=("quick", "full"),
        default="full",
        help="quick=local QC/analysis/plots only; full=includes SSHA, verify, classification, water-type plots.",
    )
    run_p.add_argument("--skip-ssha", action="store_true", help="Skip SSHA/verification/classification stages.")
    run_p.add_argument("--skip-plots", action="store_true", help="Skip profile and water-type plotting stages.")
    run_p.add_argument("--overlay-day-night", action="store_true", help="Generate day/night overlay T-D plots by site.")
    run_p.add_argument("--include-sv-only", action="store_true", help="Include SV-only CNVs in validate/analyze/plot.")
    run_p.add_argument(
        "--keep-all-variants",
        action="store_true",
        help="Keep both binned and non-binned cast variants in cast-index and SSHA matching.",
    )
    run_p.add_argument("--source", type=str, default="hycom_gom_reanalysis_2d", help="SSHA source key.")
    run_p.add_argument(
        "--max-time-delta-min",
        type=float,
        default=90.0,
        help="Maximum allowed |cast_time - model_time| in minutes for SSHA matching and verification.",
    )
    run_p.add_argument("--limit", type=int, default=None, help="Optional cast row limit for SSHA trial runs.")
    run_p.add_argument("--include-unselected", action="store_true", help="Include unselected cast variants in SSHA matching.")
    run_p.add_argument("--spot-check-count", type=int, default=5, help="Number of ok rows to re-query in verify stage.")
    run_p.add_argument("--no-strict-verify", action="store_true", help="Disable strict verification status behavior.")
    run_p.add_argument(
        "--include-non-ok-matches",
        action="store_true",
        help="Include non-ok match rows during classification (default excludes).",
    )
    run_p.set_defaults(func=cmd_run)

    scan_p = sub.add_parser("scan", help="Discover CNV files and classify full vs SV-only.")
    scan_p.add_argument("root_dir", type=Path)
    scan_p.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    scan_p.set_defaults(func=cmd_scan)

    val_p = sub.add_parser("validate", help="Run parsing and QC checks.")
    val_p.add_argument("root_dir", type=Path)
    val_p.add_argument("--out", type=Path, required=True, help="Validation JSON report output path.")
    val_p.add_argument("--include-sv-only", action="store_true", help="Do not exclude SV-only CNVs.")
    val_p.set_defaults(func=cmd_validate)

    ana_p = sub.add_parser("analyze", help="Compute easy/moderate profile metrics.")
    ana_p.add_argument("root_dir", type=Path)
    ana_p.add_argument("--out", type=Path, required=True, help="Output directory.")
    ana_p.add_argument("--include-sv-only", action="store_true", help="Include SV-only CNVs.")
    ana_p.set_defaults(func=cmd_analyze)

    plot_p = sub.add_parser("plot", help="Generate T-D, S-D, and T-S plots.")
    plot_p.add_argument("root_dir", type=Path)
    plot_p.add_argument("--out", type=Path, required=True, help="Output directory for figures.")
    plot_p.add_argument("--overlay-day-night", action="store_true", help="Generate day/night overlay T-D plots by site.")
    plot_p.add_argument("--include-sv-only", action="store_true", help="Include SV-only CNVs.")
    plot_p.set_defaults(func=cmd_plot)

    idx_p = sub.add_parser("export-cast-index", help="Export cast index with time/lat/lon/T300.")
    idx_p.add_argument("root_dir", type=Path)
    idx_p.add_argument("--out", type=Path, required=True, help="Cast index CSV path.")
    idx_p.add_argument(
        "--keep-all-variants",
        action="store_true",
        help="Keep both binned and non-binned variants (default selects preferred cast per family).",
    )
    idx_p.set_defaults(func=cmd_export_cast_index)

    ssha_p = sub.add_parser("fetch-ssha", help="Fetch SSHA from HYCOM OPeNDAP for indexed casts.")
    ssha_p.add_argument("cast_index_csv", type=Path)
    ssha_p.add_argument("--out-dir", type=Path, required=True, help="Output directory for SSHA artifacts.")
    ssha_p.add_argument(
        "--source",
        type=str,
        default="hycom_gom_reanalysis_2d",
        help="SSHA source key.",
    )
    ssha_p.add_argument(
        "--max-time-delta-min",
        type=float,
        default=90.0,
        help="Maximum allowed |cast_time - model_time| in minutes for strict matching.",
    )
    ssha_p.add_argument("--limit", type=int, default=None, help="Optional cast row limit for trial runs.")
    ssha_p.add_argument(
        "--include-unselected",
        action="store_true",
        help="Match rows even if is_selected is false.",
    )
    ssha_p.set_defaults(func=cmd_fetch_ssha)

    verify_p = sub.add_parser("verify-ssha", help="Verify SSHA match quality and spot-check re-queries.")
    verify_p.add_argument("matches_csv", type=Path)
    verify_p.add_argument("--out", type=Path, required=True, help="Verification report JSON path.")
    verify_p.add_argument(
        "--source",
        type=str,
        default="hycom_gom_reanalysis_2d",
        help="SSHA source key used for spot-check re-query.",
    )
    verify_p.add_argument("--strict", action="store_true", help="Treat any mismatch as error.")
    verify_p.add_argument("--spot-check-count", type=int, default=5, help="Number of ok rows to re-query.")
    verify_p.add_argument(
        "--max-time-delta-min",
        type=float,
        default=90.0,
        help="Maximum allowed |cast_time - model_time| in minutes.",
    )
    verify_p.set_defaults(func=cmd_verify_ssha)

    class_p = sub.add_parser(
        "classify-water-type",
        help="Classify water types using Johnston/Boswell SSHA + T300 rules.",
    )
    class_p.add_argument("matches_csv", type=Path, help="Input SSHA matches CSV.")
    class_p.add_argument("--out-dir", type=Path, required=True, help="Output directory for classification files.")
    class_p.add_argument(
        "--include-non-ok-matches",
        action="store_true",
        help="Include rows even when match_status is not ok (default excludes them).",
    )
    class_p.set_defaults(func=cmd_classify_water_type)

    wplot_p = sub.add_parser(
        "plot-water-type",
        help="Generate water-type diagnostic plots from classification CSV.",
    )
    wplot_p.add_argument("classification_csv", type=Path, help="Input water_type_classification.csv file.")
    wplot_p.add_argument("--out-dir", type=Path, required=True, help="Output directory for water-type figures.")
    wplot_p.set_defaults(func=cmd_plot_water_type)

    return parser


def cmd_run(args: argparse.Namespace) -> None:
    out_root = args.out_root or _default_run_out_root()
    out_root.mkdir(parents=True, exist_ok=True)
    run_ssha = (args.preset == "full") and (not args.skip_ssha)

    scan_out = out_root / "scan.json"
    validate_out = out_root / "validation.json"
    analysis_out = out_root / "analysis"
    profile_plot_out = out_root / "plots"
    ssha_out = out_root / "ssha"
    cast_index_out = ssha_out / "cast_index.csv"
    matches_out = ssha_out / "ssha_matches.csv"
    verify_out = ssha_out / "verification_report.json"
    classification_out = out_root / "classification"
    class_csv_out = classification_out / "water_type_classification.csv"
    water_plot_out = classification_out / "plots"

    cmd_scan(argparse.Namespace(root_dir=args.root_dir, out=scan_out))
    cmd_validate(
        argparse.Namespace(
            root_dir=args.root_dir,
            out=validate_out,
            include_sv_only=args.include_sv_only,
        )
    )
    cmd_analyze(
        argparse.Namespace(
            root_dir=args.root_dir,
            out=analysis_out,
            include_sv_only=args.include_sv_only,
        )
    )
    if not args.skip_plots:
        cmd_plot(
            argparse.Namespace(
                root_dir=args.root_dir,
                out=profile_plot_out,
                overlay_day_night=args.overlay_day_night,
                include_sv_only=args.include_sv_only,
            )
        )

    cmd_export_cast_index(
        argparse.Namespace(
            root_dir=args.root_dir,
            out=cast_index_out,
            keep_all_variants=args.keep_all_variants,
        )
    )

    if run_ssha:
        cmd_fetch_ssha(
            argparse.Namespace(
                cast_index_csv=cast_index_out,
                out_dir=ssha_out,
                source=args.source,
                max_time_delta_min=args.max_time_delta_min,
                limit=args.limit,
                include_unselected=args.include_unselected,
            )
        )
        cmd_verify_ssha(
            argparse.Namespace(
                matches_csv=matches_out,
                out=verify_out,
                source=args.source,
                strict=not args.no_strict_verify,
                spot_check_count=args.spot_check_count,
                max_time_delta_min=args.max_time_delta_min,
            )
        )
        cmd_classify_water_type(
            argparse.Namespace(
                matches_csv=matches_out,
                out_dir=classification_out,
                include_non_ok_matches=args.include_non_ok_matches,
            )
        )
        if not args.skip_plots:
            cmd_plot_water_type(
                argparse.Namespace(
                    classification_csv=class_csv_out,
                    out_dir=water_plot_out,
                )
            )

    run_summary = _build_run_summary(
        root_dir=args.root_dir,
        out_root=out_root,
        preset=args.preset,
        run_ssha=run_ssha,
        plots_enabled=not args.skip_plots,
        source=args.source,
    )
    _write_json(out_root / "run_summary.json", run_summary)
    print(f"Run complete: out_root={out_root} summary={out_root / 'run_summary.json'}")


def cmd_scan(args: argparse.Namespace) -> None:
    summary = _scan_summary(args.root_dir)
    _print_summary(summary)
    if args.out:
        _write_json(args.out, summary)


def cmd_validate(args: argparse.Namespace) -> None:
    results = _collect_validation_results(
        args.root_dir,
        reject_sv_only=not args.include_sv_only,
    )
    report = _validation_report_payload(args.root_dir, results)
    _print_validation_summary(report)
    _write_json(args.out, report)


def cmd_analyze(args: argparse.Namespace) -> None:
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _collect_validation_results(
        args.root_dir,
        reject_sv_only=not args.include_sv_only,
    )
    report = _validation_report_payload(args.root_dir, results)
    _write_json(out_dir / "validation_report.json", report)

    valid_casts = [
        r.cast for r in results if r.cast is not None and not r.skipped and r.report.status != "error"
    ]

    metrics = [compute_profile_metrics(cast) for cast in valid_casts]
    pd.DataFrame(metrics).to_csv(out_dir / "metrics_per_cast.csv", index=False)

    casts_dir = out_dir / "casts"
    casts_dir.mkdir(parents=True, exist_ok=True)
    for cast in valid_casts:
        cast.data.to_csv(casts_dir / f"{cast.cast_id}.csv", index=False)

    print(f"Saved metrics for {len(valid_casts)} cast(s) to {out_dir}")


def cmd_plot(args: argparse.Namespace) -> None:
    from ctd_tool.plots import plot_cast_profiles, plot_day_night_overlay

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _collect_validation_results(
        args.root_dir,
        reject_sv_only=not args.include_sv_only,
    )
    valid_casts = [
        r.cast for r in results if r.cast is not None and not r.skipped and r.report.status != "error"
    ]

    cast_plot_dir = out_dir / "casts"
    cast_plot_dir.mkdir(parents=True, exist_ok=True)
    total_plots = 0
    for cast in valid_casts:
        created = plot_cast_profiles(cast, cast_plot_dir)
        total_plots += len(created)

    if args.overlay_day_night:
        overlays_dir = out_dir / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)
        by_site: dict[str, list] = {}
        for cast in valid_casts:
            info = describe_cnv_path(cast.header.source_path)
            site = info.site or "unknown_site"
            by_site.setdefault(site, []).append(cast)
        for site, casts in by_site.items():
            plot_day_night_overlay(casts, overlays_dir, site)

    print(f"Generated {total_plots} plot file(s) for {len(valid_casts)} cast(s).")


def cmd_export_cast_index(args: argparse.Namespace) -> None:
    frame = build_cast_index(args.root_dir, prefer_binned=not args.keep_all_variants)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    selected_count = int(frame["is_selected"].sum()) if "is_selected" in frame.columns else len(frame)
    print(f"Exported {len(frame)} cast row(s), selected={selected_count}, file={args.out}")


def cmd_fetch_ssha(args: argparse.Namespace) -> None:
    from ctd_tool.hycom.matcher import match_cast_index_to_ssha, summarize_match_results
    from ctd_tool.hycom.opendap_client import OpendapSshaClient
    from ctd_tool.hycom.sources import get_source_by_key

    source = get_source_by_key(args.source)
    cast_index = pd.read_csv(args.cast_index_csv)
    client = OpendapSshaClient(source)
    matches = match_cast_index_to_ssha(
        cast_index,
        client,
        max_time_delta_minutes=args.max_time_delta_min,
        selected_only=not args.include_unselected,
        limit=args.limit,
    )
    summary = summarize_match_results(matches)
    summary.update(
        {
            "source_key": source.key,
            "source_description": source.description,
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "max_time_delta_min": args.max_time_delta_min,
        }
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out_dir / "ssha_matches.csv", index=False)
    _write_json(out_dir / "ssha_qc_report.json", summary)
    print(
        "SSHA fetch complete: "
        f"total={summary['total_casts']} ok={summary['ok_matches']} error={summary['error_matches']}"
    )


def cmd_verify_ssha(args: argparse.Namespace) -> None:
    from ctd_tool.hycom.opendap_client import OpendapSshaClient
    from ctd_tool.hycom.matcher import verify_ssha_matches
    from ctd_tool.hycom.sources import get_source_by_key

    matches = pd.read_csv(args.matches_csv)
    source = get_source_by_key(args.source)
    client = OpendapSshaClient(source)
    report = verify_ssha_matches(
        matches,
        client,
        max_time_delta_minutes=args.max_time_delta_min,
        strict=args.strict,
        spot_check_count=args.spot_check_count,
    )
    report.update(
        {
            "source_key": source.key,
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "matches_csv": str(args.matches_csv),
        }
    )
    _write_json(args.out, report)
    print(
        "SSHA verify complete: "
        f"status={report['status']} spot_checks={report['spot_checks_run']} failed={report['spot_checks_failed']}"
    )


def cmd_classify_water_type(args: argparse.Namespace) -> None:
    matches = pd.read_csv(args.matches_csv)
    classified = classify_water_types(
        matches,
        require_ok_matches=not args.include_non_ok_matches,
    )
    summary = classification_summary(classified)
    summary.update({"input_matches_csv": str(args.matches_csv)})

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    classified.to_csv(out_dir / "water_type_classification.csv", index=False)
    compact = _build_compact_classification_report(classified)
    compact.to_csv(out_dir / "water_type_compact_report.csv", index=False)
    _write_json(out_dir / "water_type_summary.json", summary)

    counts = summary.get("water_type_counts", {})
    counts_text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(
        "Water-type classification complete: "
        f"rows={summary['rows']} ({counts_text}); compact_report={out_dir / 'water_type_compact_report.csv'}"
    )


def _build_compact_classification_report(classified: pd.DataFrame) -> pd.DataFrame:
    working = classified.copy()
    if "cast_lat_deg" in working.columns and "lat_deg" not in working.columns:
        working["lat_deg"] = working["cast_lat_deg"]
    if "cast_lon_deg" in working.columns and "lon_deg" not in working.columns:
        working["lon_deg"] = working["cast_lon_deg"]

    compact_cols = [
        "cast_id",
        "cast_time_utc",
        "lat_deg",
        "lon_deg",
        "t300_c",
        "ssha_i_m",
        "ssha_gom_m",
        "water_type",
        "lcow_index_raw",
    ]

    for col in compact_cols:
        if col not in working.columns:
            working[col] = pd.NA
    return working.loc[:, compact_cols]


def cmd_plot_water_type(args: argparse.Namespace) -> None:
    from ctd_tool.plots import generate_water_type_plots

    classified = pd.read_csv(args.classification_csv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    created = generate_water_type_plots(classified, out_dir)
    print(f"Generated {len(created)} water-type plot file(s) in {out_dir}")


def _collect_validation_results(root_dir: Path, *, reject_sv_only: bool) -> list[ValidationResult]:
    infos = discover_cnv_files(root_dir)
    return [
        parse_and_validate_cnv(info.path, reject_sv_only=reject_sv_only)
        for info in infos
    ]


def _scan_summary(root_dir: Path) -> dict[str, Any]:
    infos = discover_cnv_files(root_dir)
    summary = {
        "root_dir": str(root_dir),
        "total_cnv_files": len(infos),
        "full_profile_cnv_files": 0,
        "sv_only_cnv_files": 0,
        "parse_errors": 0,
        "files": [],
    }
    for info in infos:
        file_entry: dict[str, Any] = {
            "path": str(info.path),
            "site": info.site,
            "day_night": info.day_night,
            "ctd_number": info.ctd_number,
            "is_binned": info.is_binned,
        }
        try:
            parsed = parse_cnv_detailed(info.path)
            sv_only = is_sv_only(parsed.canonical_map, parsed.header.variables)
            file_entry["is_sv_only"] = sv_only
            if sv_only:
                summary["sv_only_cnv_files"] += 1
            else:
                summary["full_profile_cnv_files"] += 1
        except CnvParseError as exc:
            summary["parse_errors"] += 1
            file_entry["is_sv_only"] = None
            file_entry["parse_error"] = str(exc)
        summary["files"].append(file_entry)
    return summary


def _validation_report_payload(root_dir: Path, results: list[ValidationResult]) -> dict[str, Any]:
    return {
        "root_dir": str(root_dir),
        "total_files": len(results),
        "ok_files": sum(1 for r in results if r.report.status == "ok" and not r.skipped),
        "warning_files": sum(1 for r in results if r.report.status == "warning" and not r.skipped),
        "error_files": sum(1 for r in results if r.report.status == "error"),
        "skipped_files": sum(1 for r in results if r.skipped),
        "files": [r.to_dict() for r in results],
    }


def _default_run_out_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return Path("out") / f"run_{stamp}"


def _build_run_summary(
    *,
    root_dir: Path,
    out_root: Path,
    preset: str,
    run_ssha: bool,
    plots_enabled: bool,
    source: str,
) -> dict[str, Any]:
    scan_report = _read_json_or_none(out_root / "scan.json")
    validate_report = _read_json_or_none(out_root / "validation.json")
    analysis_report = _read_json_or_none(out_root / "analysis" / "validation_report.json")
    metrics = _read_csv_or_none(out_root / "analysis" / "metrics_per_cast.csv")
    cast_index = _read_csv_or_none(out_root / "ssha" / "cast_index.csv")
    ssha_report = _read_json_or_none(out_root / "ssha" / "ssha_qc_report.json")
    verify_report = _read_json_or_none(out_root / "ssha" / "verification_report.json")
    water_report = _read_json_or_none(out_root / "classification" / "water_type_summary.json")

    counts: dict[str, Any] = {
        "scan_total_cnv_files": int(scan_report.get("total_cnv_files", 0)) if scan_report else 0,
        "scan_sv_only_cnv_files": int(scan_report.get("sv_only_cnv_files", 0)) if scan_report else 0,
        "validation_ok_files": int(validate_report.get("ok_files", 0)) if validate_report else 0,
        "validation_warning_files": int(validate_report.get("warning_files", 0)) if validate_report else 0,
        "validation_error_files": int(validate_report.get("error_files", 0)) if validate_report else 0,
        "analysis_cast_rows": int(len(metrics)) if metrics is not None else 0,
        "profile_plot_pngs": len(list((out_root / "plots").rglob("*.png"))) if plots_enabled else 0,
        "cast_index_rows": int(len(cast_index)) if cast_index is not None else 0,
        "cast_index_selected_rows": int(cast_index["is_selected"].sum()) if cast_index is not None and "is_selected" in cast_index.columns else 0,
    }

    if run_ssha:
        counts.update(
            {
                "ssha_total_casts": int(ssha_report.get("total_casts", 0)) if ssha_report else 0,
                "ssha_ok_matches": int(ssha_report.get("ok_matches", 0)) if ssha_report else 0,
                "ssha_error_matches": int(ssha_report.get("error_matches", 0)) if ssha_report else 0,
                "verify_spot_checks_run": int(verify_report.get("spot_checks_run", 0)) if verify_report else 0,
                "verify_spot_checks_failed": int(verify_report.get("spot_checks_failed", 0)) if verify_report else 0,
                "classification_rows": int(water_report.get("rows", 0)) if water_report else 0,
                "water_type_plot_pngs": len(list((out_root / "classification" / "plots").glob("*.png"))) if plots_enabled else 0,
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "root_dir": str(root_dir),
        "out_root": str(out_root),
        "preset": preset,
        "run_ssha": run_ssha,
        "plots_enabled": plots_enabled,
        "source": source,
        "counts": counts,
        "statuses": {
            "validation_status": analysis_report.get("error_files", 0) == 0 if analysis_report else None,
            "verify_status": verify_report.get("status") if verify_report else None,
        },
        "water_type_counts": water_report.get("water_type_counts", {}) if water_report else {},
        "artifacts": {
            "scan_report_json": str(out_root / "scan.json"),
            "validation_report_json": str(out_root / "validation.json"),
            "analysis_metrics_csv": str(out_root / "analysis" / "metrics_per_cast.csv"),
            "cast_index_csv": str(out_root / "ssha" / "cast_index.csv"),
            "ssha_matches_csv": str(out_root / "ssha" / "ssha_matches.csv") if run_ssha else None,
            "ssha_verification_json": str(out_root / "ssha" / "verification_report.json") if run_ssha else None,
            "water_type_classification_csv": str(out_root / "classification" / "water_type_classification.csv") if run_ssha else None,
            "water_type_summary_json": str(out_root / "classification" / "water_type_summary.json") if run_ssha else None,
        },
    }


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_csv_or_none(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"Scanned: {summary['total_cnv_files']} CNV file(s)")
    print(f"Full-profile: {summary['full_profile_cnv_files']}")
    print(f"SV-only: {summary['sv_only_cnv_files']}")
    print(f"Parse errors: {summary['parse_errors']}")


def _print_validation_summary(report: dict[str, Any]) -> None:
    print(f"Validated: {report['total_files']} CNV file(s)")
    print(f"OK: {report['ok_files']}")
    print(f"Warnings: {report['warning_files']}")
    print(f"Errors: {report['error_files']}")
    print(f"Skipped: {report['skipped_files']}")

