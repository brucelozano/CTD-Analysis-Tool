# CNV-First CTD Analysis Tool

CLI-first Python tool for working with processed Sea-Bird `.cnv` CTD files.

## Objectives

- Discovers and classifies `.cnv` files as full-profile vs SV-only.
- Parses Sea-Bird CNV headers and data tables.
- Normalizes canonical fields (`depth_m`, `temperature_c`, `salinity_psu`, etc.).
- Runs QC/error checking with machine-readable reports.
- Computes easy-to-moderate profile metrics (`T300`, thermocline proxy, mixed-layer proxy).
- Uses downcast-first profile extraction for metric/plot calculations (upcast turn-around rows are excluded).
- Supports one-command pipeline runs with `quick` and `full` presets.
- Generates per-cast plots:
  - Temperature-Depth (T-D)
  - Salinity-Depth (S-D)
  - Temperature-Salinity (T-S)



## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```



## Usage



### Recommended: one-command pipeline

Quick local workflow (no SSHA fetch/classification):

```bash
ctd-tool run "DP09_CTD_CNV/PS23_22_Sutton_CTD_Processed" --preset quick --out-root out/run_quick
```

Full workflow (includes SSHA, verify, classification, and water-type plots):

```bash
ctd-tool run "DP09_CTD_CNV/PS23_22_Sutton_CTD_Processed" --preset full --out-root out/run_full --overlay-day-night
```

Each run writes a machine-readable manifest:

- `run_summary.json`: stage statuses, row counts, water-type counts, and artifact paths.



### Advanced: stage-by-stage commands



### Scan files

```bash
ctd-tool scan "DP09_CTD_CNV/PS23_22_Sutton_CTD_Processed" --out out/scan.json
```



### Validate files

```bash
ctd-tool validate "DP09_CTD_CNV/PS23_22_Sutton_CTD_Processed" --out out/validation.json
```



### Analyze casts

```bash
ctd-tool analyze "DP09_CTD_CNV/PS23_22_Sutton_CTD_Processed" --out out/analysis
```



### Plot profiles

```bash
ctd-tool plot "DP09_CTD_CNV/PS23_22_Sutton_CTD_Processed" --out out/plots --overlay-day-night
```



### Export cast index for SSHA matching

```bash
ctd-tool export-cast-index "DP09_CTD_CNV/PS23_22_Sutton_CTD_Processed" --out out/ssha/cast_index.csv
```



### Fetch SSHA from HYCOM OPeNDAP

```bash
ctd-tool fetch-ssha out/ssha/cast_index.csv --out-dir out/ssha --source hycom_gom_reanalysis_2d
```



### Verify SSHA matches with strict spot checks

```bash
ctd-tool verify-ssha out/ssha/ssha_matches.csv --out out/ssha/verification_report.json --strict
```



### Classify water type (Johnston/Boswell)

```bash
ctd-tool classify-water-type out/ssha/ssha_matches.csv --out-dir out/classification
```



### Plot water-type diagnostics

```bash
ctd-tool plot-water-type out/classification/water_type_classification.csv --out-dir out/classification/plots
```



## Output Artifacts

- `scan.json`: discovered files and basic classification.
- `validation.json` or `analysis/validation_report.json`: QC status and messages.
- `analysis/metrics_per_cast.csv`: core cast-level metrics.
- `analysis/casts/*.csv`: per-cast parsed table export.
- `plots/casts/*.png`: per-cast T-D, S-D, T-S figures.
- `plots/overlays/*.png`: optional day/night site overlays.
- `ssha/cast_index.csv`: selected casts with UTC time, lat/lon, and T300.
- `ssha/ssha_matches.csv`: cast-to-model SSHA matches and diagnostics.
- `ssha/ssha_qc_report.json`: aggregate OPeNDAP match summary.
- `ssha/verification_report.json`: strict verification + spot-check outcomes.
- `classification/water_type_classification.csv`: per-cast water type and index columns.
- `classification/water_type_compact_report.csv`: lab-shareable compact classification table.
- `classification/water_type_summary.json`: class counts and threshold metadata.
- `classification/plots/*.png`: map, threshold scatter, index histogram, time-series, threshold-distance, and class-count plots.
- `run_summary.json`: single-run summary of stage outputs, counts, statuses, and key artifact paths.



## Default Input Policy

- Full-profile CNV files are analyzed by default.
- SV-only CNV files (depth + sound speed only) are excluded by default.
- Use `--include-sv-only` to include SV-only files in `validate`, `analyze`, or `plot`.



## OPeNDAP Matching Notes

- Default SSHA source is HYCOM Gulf of Mexico reanalysis (`hycom_gom_reanalysis_2d`).
- Strict mode expects close time alignment between cast and model output (`--max-time-delta-min`, default `90`).
- `fetch-ssha` writes both point SSHA (`ssha_i_m`) and daily Gulf mean SSHA (`ssha_gom_m`) for classification inputs.
- `fetch-ssha` records per-cast errors as row-level `match_status=error` without aborting the full run.



## Water-Type Equations Implemented

- LCOW classification: `ssha_i > ssha_gom + 0.067` and `t300 > 15.92`
- CW classification: `ssha_i <= ssha_gom` and `t300 < 13.46`
- MIX classification: `ssha_i` between CW and LCOW SSHA bounds, and `t300` between 13.46 and 15.92
- LCOW raw index (Boswell et al. style):  
`lcow_index_raw = ssha_i - (ssha_gom + 0.067) + t300 - 15.922`

