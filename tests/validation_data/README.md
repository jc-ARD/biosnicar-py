# Sea Ice Validation Suite

Empirical datasets and validation scripts for the BioSNICAR sea ice extension (`biosnicar.sea_ice`).

> [!IMPORTANT] Source-of-truth policy (read first)
> **Tracked = scripts + curated docs. Generated = regenerable, not committed.**
> Result files (`*_results.json`, `parameter_retrieval_results.*`) and figure
> PNGs are **git-ignored** — they are snapshots that go stale silently after
> emulator/LUT rebuilds. Regenerate them by running the script; treat the
> headline numbers in the curated docs (`docs/sea_ice_validation.md`,
> `docs/SEA_ICE_RETRIEVAL.md`) as the verified record. After any emulator or
> physics rebuild, re-run the relevant script before quoting numbers.

## Validation inventory and status

| Script | Tests | Modality / data | Status |
|---|---|---|---|
| `sea_ice_emulator_sheba_validation.py` | classification + satellite bands + pond depth (blocks A–E) | SHEBA + Morassutti, VIS–NIR | **current** — primary inversion validation |
| `sheba_classification_validation.py` | classification across spectral windows (VIS vs VIS+SWIR) + label provenance | SHEBA + Morassutti | **current (historical motivation)** — the VIS+SWIR degeneracy it documents is now handled automatically by per-emulator band masks (C2); kept for provenance/regression |
| `parameter_retrieval_validation.py` | synthetic per-parameter recovery (bias/RMSE/R²/DFS-style) | forward-model, seed 2026 | **current** — synthetic (inverse-crime); see caveat in script |
| `../../scripts/smith_retrieval_validation.py` | **independent-campaign** retrieval hold-out | Smith/MOSAiC 2020, 350–2500 nm | **current** — the strongest independent test (different year/site/instrument) |
| `../../scripts/plot_sheba_fits.py` | observed-vs-retrieved spectra figures | SHEBA | **current** — figures |
| `../../scripts/plot_inversion_performance.py` | confusion matrix + parameter scatter + young-ice curve | synthetic | **current** — synthetic demo (disclaimed in-figure) |
| `../../scripts/experiments/fyi_bare_audit.py` | FYI_bare emulator accuracy + degeneracy | held-out forward-model, seed 777 | **current** — emulator audit backbone |
| `run_global_validation.py` | **forward-model preset** accuracy (FYI_WINTER_SNOW/BARE), *not* the inversion | Grenfell + Smith | **current but distinct track** — validates the forward presets, not `retrieve_sea_ice()`; Smith melt-season "failures" are an expected season mismatch |

Curated reports: `docs/sea_ice_validation.md` (inversion, current), plus the
forward-model-track reports `docs/sea_ice_validation_global.md` and
`docs/sea_ice_validation_meltpond.md` (older, preset-validation track —
read alongside `sea_ice_validation.md` §6, which supersedes them for the
inversion).

---

## Quick start

```bash
# Run the full global validation (all datasets, all figures, report)
uv run python tests/validation_data/run_global_validation.py \
  --plots  tests/validation_data/figures/global/ \
  --json   tests/validation_data/global_results.json \
  --report docs/sea_ice_validation_global.md

# Run dataset-specific validation with parameter sensitivity sweep
uv run python tests/validation_data/Grenfell_light_2007/validate_grenfell_light_2007.py \
  --plots  tests/validation_data/Grenfell_light_2007/figures/ \
  --json   tests/validation_data/Grenfell_light_2007/results.json \
  --report docs/sea_ice_validation.md \
  --sweep   # also run grain radius and SZA sensitivity tables

# Morassutti melt pond dataset (out of scope for MVP but infrastructure present)
uv run python tests/validation_data/morassutti1995/validate_morassutti1995.py
```

All scripts are self-contained: they locate their data relative to their own file path and import the model from the repository root. They can be run from any working directory.

---

## Folder structure

```
tests/validation_data/
│
├── README.md                          ← this file
├── run_global_validation.py           ← main entry point: all datasets combined
├── global_results.json                ← machine-readable results (last run)
│
├── figures/
│   └── global/
│       ├── fig_global_spectral_overview.png
│       ├── fig_global_residuals.png
│       ├── fig_global_bba_scatter.png
│       └── fig_global_swir_detail.png
│
├── Grenfell_light_2007/               ← SHEBA 1998 dataset
│   ├── .gitignore                     ← raw CSVs excluded from git
│   ├── validate_grenfell_light_2007.py
│   ├── results.json
│   └── figures/
│       ├── fig1_spring_spectral_comparison.png
│       ├── fig2_residuals.png
│       ├── fig3_grain_radius_sensitivity.png
│       └── fig4_summer_mismatch.png
│   [raw data: ICEDATA_OPTICS_SPECALB_ALB*.CSV — not in git, see download below]
│
├── smith_2021/                        ← MOSAiC 2020 dataset
│   └── resource_map_doi_10_18739_A2FT8DK8Z/
│       └── data/SpectralAlbedoData/   ← 124 processed CSV files
│
└── morassutti1995/                    ← Canadian Arctic melt ponds 1994
    ├── validate_morassutti1995.py
    ├── manifest.json
    └── pond.dat
```

---

## Datasets

### 1. Grenfell & Light (2007) — SHEBA spectral albedo

| Field | Value |
|---|---|
| Citation | Grenfell, T.C. & Light, B. (2007). UCAR/NCAR EOL dataset 13.825 |
| DOI | [10.5065/D6765CQ1](https://doi.org/10.5065/D6765CQ1) |
| Campaign | SHEBA (Surface Heat Budget of the Arctic Ocean) |
| Location | Arctic Ocean, ~76°N, 130–170°W (drifting) |
| Period | April 8 – September 3, 1998 |
| Instrument | Portable spectrometer |
| Spectral range | 400–1000 nm, ~2.6 nm resolution (ALBV files) |
| | 1100–2005 nm, ~21 nm resolution (ALBI files — not yet integrated) |
| Format | One CSV per date per transect. Rows = wavelengths. Columns = individual positions (m) along a 200-m survey line. Header rows contain date, sky conditions, lat/lon. |

**File types:**
- `ALBV*` — visible range. Columns are individual survey positions; surface type is inferred from the season (Apr–May = all snow).
- `ALBI*` — infrared range. Columns are averaged WI (white ice), MP (melt pond), Total. Available from June 11 onward.

**What we use:** `ALBV` files only. Spring dates (Apr 8 – Jun 2) are extracted for snow comparison; summer dates (Aug–Sep) for bare-ice reference.

**Download:** Not in git (30 MB). Download from the DOI above and place the `ICEDATA_OPTICS_SPECALB_*.CSV` files directly in `Grenfell_light_2007/`.

**Known issue:** `ALBV0419` internal header reads "17-Apr-98" (filename says Apr 19). Flagged in output with ⚠.

---

### 2. Smith et al. (2021) — MOSAiC Leg 4 spectral albedo

| Field | Value |
|---|---|
| Citation | Smith, M.M. et al. (2021). Arctic Data Center |
| DOI | [10.18739/A2FT8DK8Z](https://doi.org/10.18739/A2FT8DK8Z) |
| Campaign | MOSAiC (Multidisciplinary drifting Observatory for the Study of Arctic Climate), Leg 4 |
| Location | Arctic Ocean, ~82°N (drifting with ice floe) |
| Period | June 13 – September 19, 2020 |
| Instrument | ASD FieldSpec spectrometer |
| Spectral range | 350–2500 nm, 1 nm resolution |
| Format | One CSV per transect per date. Complex header (10 rows) then spectral data. Each column = one spatial position with per-position surface type and quality flag. |

**File naming:** `spectralalbedo_mosaic_YYYYMMDD[_HHMM]_LINECODE_processedv0.csv`

Line codes and their physical meaning:

| Code | Name | Notes |
|---|---|---|
| IS | Ice station | Early transit; mixed snow/pond |
| LDL | Lemon Drop Line | Primary FYI transect |
| RBB | Root Beer Barrel Line | Secondary FYI transect |
| SYI | Second Year Ice | |
| ROV4 | ROV grid | Underwater ROV deployment site |
| RS | Reunion Stakes | Stake array |
| DB | Drone Bones | Drone operations area |
| STERN | Stern area | Near ship |
| KINDER / BOUNTY / TOBLERONE | Named lines | Late-season refreezing transects (Aug–Sep) |
| SNOWTARGET | Snow target | Calibration-style fresh snow measurements |

**Per-position metadata (from file header):**
- `Surface type`: `S` = snow, `P` = melt pond, `S/P` = boundary, `NaN` = unknown
- `Change in incident (%)`: quality flag — values >10% indicate variable cloud/sun conditions. We keep only ≤10%.

**Solar zenith angle:** computed from the UTC start time and ship lat/lon in each file header (analytical formula, accurate to ~1–2°).

**Surface evolution during campaign:**
- Jun 13–22: snow-covered FYI with proto-ponds forming
- Jun 24 – Jul: mixed snow and fully developed melt ponds
- Aug 21 – Sep 19: refreezing conditions; KINDER/SNOWTARGET lines have fresh/refrozen snow

**Quality-filtered snow records used in validation:** 44 files with at least one snow position where `change_in_incident ≤ 10%`.

**Important caveat:** Most of this dataset is **summer melt-season** data. Comparing winter model presets to July snow observations will show large BBA errors (+0.2–0.4) that are physically expected, not model faults. See *Output interpretation* below.

---

### 3. Morassutti (1995) — Canadian Arctic melt ponds

| Field | Value |
|---|---|
| Citation | Morassutti, M. (1995). NSIDC G01169 |
| DOI | [10.7265/N55Q4T1C](https://doi.org/10.7265/N55Q4T1C) |
| Location | Barrow Strait, Nunavut, Canada (~74°N) |
| Period | May 27 – June 26, 1994 |
| Spectral range | 400–1000 nm, ~100 nm resolution (6 bands) |

**Status:** Out of scope for MVP. All 504 observations are melt pond measurements. Retained for future melt pond (v0.2) development. Validation script documents the dataset and its limitations.

---

## Running the global validation

### `run_global_validation.py` — combined entry point

Loads both Grenfell and Smith datasets, runs model comparisons against all usable records, and produces four figures.

```bash
uv run python tests/validation_data/run_global_validation.py [options]

Options:
  --plots DIR    Save figures to DIR (creates if absent)
  --json  FILE   Save per-record numeric results to FILE
  --report FILE  Save Markdown validation report to FILE
  --show         Display interactive matplotlib figures
```

**What it does internally:**

1. Loads all spring Grenfell snow records (Apr 8 – Jun 2) and summer bare-ice records (Aug – Sep).
2. Loads all Smith 2021 files; for each file, extracts positions where `surface_type='S'` and `change_in_incident ≤ 10%`, computes the mean spectrum across qualifying positions.
3. Runs `FYI_WINTER_SNOW.compute_albedo(sza)` for each snow record and `FYI_WINTER_BARE.compute_albedo(sza)` for each bare-ice record.
4. SZA: Grenfell uses noon SZA approximation at 76°N; Smith uses actual UTC time + ship lat/lon.
5. Computes spectral RMSE and mean bias in three windows: VIS (400–700 nm), NIR (700–1000 nm), SWIR (1000–2400 nm, Smith only).
6. Computes flux-weighted BBA in the 400–1000 nm common window.

**Runtime:** ~3–5 minutes on a modern laptop (44 Smith files × ~2000 wavelengths × model forward pass each).

---

### `Grenfell_light_2007/validate_grenfell_light_2007.py` — dataset-specific validation

Adds detailed per-date spectral comparison with two model configurations:

- **DEFAULT** — `FYI_WINTER_SNOW` preset as shipped (grain radius 200 µm, snow density 300 kg/m³)
- **ALIGNED** — parameter-tuned to SHEBA spring conditions (grain radius 400 µm, snow density 250 kg/m³)

```bash
uv run python tests/validation_data/Grenfell_light_2007/validate_grenfell_light_2007.py [options]

Options:
  --plots DIR    Save 4 figures to DIR
  --json  FILE   Save results
  --report FILE  Save Markdown report
  --sweep        Run grain radius (100–800 µm) and SZA (±15°) sensitivity tables
  --show         Display interactive plots
```

The `--sweep` flag runs additional parameter sensitivity analysis that takes ~2–3 minutes extra and prints tables showing how RMSE varies with grain radius and SZA offset.

---

## Output interpretation

### Pass/fail criteria

Both scripts apply the criteria from the original build spec:

| Criterion | Threshold |
|---|---|
| BBA absolute error (`\|model − obs\|`, 400–1000 nm) | ≤ 0.05 |
| Spectral RMSE (400–1000 nm) | ≤ 0.10 |

### Reading the results table

```
Source      Date         SZA    Obs BBA  Mod BBA  BBA Δ   RMSE  Vis bias  NIR bias  SWIR RMSE  Pass?
grenfell    1998-04-08   69°     0.935    0.941   +0.006   0.025  +0.017    -0.020     n/a         ✓
smith       2020-06-13   62°     0.802    0.932   +0.130   0.131  +0.129    +0.131     0.054       ✗
```

- **Obs BBA / Mod BBA** — flux-weighted broadband albedo in 400–1000 nm window.
- **BBA Δ** — model minus observed. Positive = model too bright.
- **RMSE** — spectral root-mean-square error across all wavelengths in 400–1000 nm.
- **Vis / NIR bias** — mean signed error in visible (400–700 nm) and NIR (700–1000 nm) sub-windows separately. Useful for diagnosing where the error is concentrated.
- **SWIR RMSE** — spectral RMSE in 1000–2400 nm (Smith data only; `n/a` for Grenfell).
- **Pass?** — ✓ if both `|BBA Δ| ≤ 0.05` and `RMSE ≤ 0.10`; ✗ otherwise.

### Expected failure modes and their causes

| Observation | Likely cause | Action needed |
|---|---|---|
| Large positive BBA Δ for Smith Jun–Jul | Season mismatch: summer melt-season snow vs winter model | Not a bug; melt-season parameterisation is v0.2 scope |
| Positive Vis bias for Grenfell spring (~+0.03) | Snow grain radius 200 µm too small for metamorphosed spring snow | Consider grain=300–400 µm for late spring; see `--sweep` output |
| Negative NIR bias for Grenfell summer bare ice | Summer ice has larger air bubbles; winter model insufficient | Not a bug; requires summer ice parameterisation |
| SWIR RMSE ~0.04 for Smith Jun 21–22 | 1000–1300 nm window overestimated; 1300–1500 nm ok | Grain radius and bubble microstructure effects in near-SWIR |
| SZA sensitivity visible in NIR | Fresnel angle changes with SZA; NIR more sensitive than VIS | Expected physical behaviour |

### Figures guide

**Global validation (`figures/global/`):**

| Figure | What to look for |
|---|---|
| `fig_global_spectral_overview.png` | Left: Grenfell spring (red) and Smith early June (blue) should be close to the black model band. Grey traces are summer melt-season snow — expect model to be above them. Right: full 350–2400 nm Smith spectra showing absorption band shapes at 1.5 µm and 2.0 µm. |
| `fig_global_residuals.png` | Flat residuals near zero = good. A uniform positive offset in VIS = grain radius too small. NIR negative = not enough bubble scattering. The green ±0.05 band is the target. |
| `fig_global_bba_scatter.png` | Points should cluster near the 1:1 line. Spring snow should be close; summer snow will be far above (model too bright). |
| `fig_global_swir_detail.png` | Per-file SWIR comparison. Blue shaded bands are water vapour absorption windows where both model and obs should be near zero. Check the 1000–1300 nm shoulder carefully. |

**Grenfell-specific (`Grenfell_light_2007/figures/`):**

| Figure | What to look for |
|---|---|
| `fig1_spring_spectral_comparison.png` | 7-row panel — one per date. Orange = DEFAULT config, blue = ALIGNED (400 µm grain). Bar charts show per-band model − obs differences. |
| `fig2_residuals.png` | Residual spectra overlaid for all spring dates. Default shows a clean positive VIS shelf (+0.02–0.05). Aligned reduces this but introduces negative NIR. |
| `fig3_grain_radius_sensitivity.png` | RMSE vs grain radius for April dates. VIS minimum at ~400 µm, NIR degrades monotonically. Full RMSE minimum at ~200 µm (balanced tradeoff). Use this to decide the right grain size for your application. |
| `fig4_summer_mismatch.png` | August bare ice comparisons. Model (orange) underestimates NIR substantially — the NIR gap is the physics driver for future summer parameterisation. |

---

## Adding a new dataset

1. Create a directory under `tests/validation_data/<dataset_name>/`.
2. Write a loader function that returns a list of `ObsRecord` objects (see `run_global_validation.py` for the dataclass definition). Required fields: `source`, `date`, `sza`, `wl_nm`, `alb`, `alb_std`, `n_spectra`, `surface`, `sky`, `lat`, `lon`.
3. Import your loader in `run_global_validation.py` and add the records to `all_records`.
4. Document the dataset in this README under a new subsection.

The `ObsRecord` dataclass is intentionally minimal — it stores the mean spectrum and spatial standard deviation at each wavelength. Per-position spectra are not retained in memory after parsing.

---

## Known limitations of the current validation

- **No cold bare-winter ice data.** The best available bare-ice spectra are summer SHEBA (Aug–Sep), which differ from winter by ~0.3 in NIR. Validation of `FYI_WINTER_BARE` against true winter conditions is not possible with current datasets.
- **Grenfell ALBI (1100–2000 nm) not yet integrated.** The IR extension of the SHEBA dataset covers the same spectral range as the Smith SWIR window but has not been parsed into the validation framework.
- **SZA approximation for Grenfell.** Measurement time is not recorded in Grenfell files; noon SZA at 76°N is used. Actual measurements could be ±1–2 hours off noon.
- **Smith surface type 'S/P'.** Boundary positions are excluded; this may under-represent transitional snow surfaces.
- **Melt ponds excluded.** Surface type 'P' is not compared to any model (melt pond support is v0.2).
- **September Smith data.** The KINDER, SNOWTARGET, and TOBLERONE lines from Aug 21 – Sep 19 represent refreezing conditions with BBA 0.56–0.77. These records are currently classified as summer_snow and flagged as season mismatch, but the Sep refreeze data may actually be reasonable to compare against winter presets. This is unexplored.

---

## References

- Grenfell, T. C., and B. Light (2007). SHEBA Spectral Albedo. UCAR/NCAR EOL 13.825. doi:10.5065/D6765CQ1
- Smith, M. M. et al. (2021). MOSAiC Leg 4 Surface Spectral Albedo. Arctic Data Center. doi:10.18739/A2FT8DK8Z
- Morassutti, M. (1995). Sea Ice Melt Pond Data from the Canadian Arctic. NSIDC G01169. doi:10.7265/N55Q4T1C
- Cox, G. F. N. & Weeks, W. F. (1983). Equations for brine volumes in sea ice. *J. Glaciology*, 29(102).
- Sihvola, A. (1999). *Electromagnetic Mixing Formulas*. IEE.
- Timco, G. W. & Frederking, R. M. W. (1996). A review of sea ice density. *Cold Reg. Sci. Tech.*, 24(1).
