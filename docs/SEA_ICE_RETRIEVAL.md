# Sea Ice Retrieval — End-to-End Guide

This guide covers the complete sea ice classification and retrieval system:
what it does scientifically, how it is built technically, and how to use it
with field spectrometers, drone hyperspectral imagers, and satellite scenes.

Companion documents:

| Document | Content |
|---|---|
| [sea_ice.md](sea_ice.md) | Forward-model physics primer (brine optics, melt ponds, young ice, open water) |
| [SEA_ICE_EMULATOR.md](SEA_ICE_EMULATOR.md) | Emulator architecture, training, accuracy audits, custom emulators |
| [INVERSION.md](INVERSION.md) | Core `retrieve()` optimiser, SSA framework, bounds, MCMC details |
| [sea_ice_validation.md](sea_ice_validation.md) | SHEBA / Morassutti / Grenfell & Maykut validation results |
| [BANDS.md](BANDS.md) | Platform band definitions and SRF convolution |

---

## 1. What the system does

Given an observed albedo — a full spectrum or a handful of satellite bands —
`retrieve_sea_ice()` answers two questions simultaneously:

1. **What kind of surface is this?** One of seven optical surface types
   (classification by goodness-of-fit).
2. **What are its physical properties?** Snow grain radius, pond depth, ice
   thickness, bubble radius, etc., with 1-sigma uncertainties.

Each surface type has a dedicated forward model of the BioSNICAR radiative
transfer physics; the observation is fitted by every candidate, and the
surface type is assigned to the model with the lowest classification
chi-squared. This "retrieve-then-classify" design avoids the inactive-parameter
problem of a single universal model (e.g. snow grain radius is meaningless when
there is no snow) and keeps every retrieval well-conditioned.

### The seven surface types

| Type | Optical state | Primary retrieved parameters | Forward model |
|---|---|---|---|
| `FYI_bare` | Cold bare first-year ice | brine volume fraction, bubble radius, BC, ρ_DL | MLP emulator (60k samples) |
| `FYI_snow` | Snow-covered FYI | tau_snow (→ snow depth), grain radius, T, BC | MLP emulator |
| `FYI_summer` | Ablating FYI with surface scattering layer | SSL grain radius, T, bubble radius, BC | MLP emulator |
| `MYI_bare` | Desalinated multiyear ice | brine volume fraction, bubble radius, BC | MLP emulator |
| `FYI_pond` | Melt pond (any ice age) | pond depth, T, BC | MLP emulator |
| `young_ice` | Semi-transparent new ice 0.5–30 cm | ice thickness, T, salinity, ocean albedo | MLP emulator of thin-slab two-stream model (layer_type=6) |
| `open_water` | Ice-free ocean | wind speed | Analytical (Cox & Munk Fresnel + pure-seawater subsurface) — no training data |

Internal labels are primary because they encode optical distinctions that
ice-chart codes collapse (`FYI_bare`, `FYI_snow`, `FYI_summer` all map to
SIGRID-3 `SM/SN`). WMO / SIGRID-3 codes are available as secondary output via
`result.to_wmo()` and `result.to_sigrid3()`; young ice resolves to SA/SB/SI/SJ
by retrieved thickness.

### Reparameterisations (why the parameters are what they are)

Albedo constrains *combinations* of physical parameters better than the raw
parameters themselves. Three reparameterisations remove the resulting
degeneracies:

- **`brine_volume_fraction`** instead of (temperature, salinity) for bare ice:
  many (T, S) pairs give identical brine volume and identical spectra.
  Temperature is recovered post-hoc via `invert_brine_volume(Vb, S_ref)` with
  reference salinities 6 psu (FYI) / 2 psu (MYI).
- **`tau_snow` = snow depth / grain radius** instead of (depth, radius) for
  snow: the radiative response depends on their ratio (optical depth), not on
  depth independently. Direct depth retrieval piled 6/7 SHEBA spring cases
  onto training bounds; tau_snow reduced this to 0. Physical `snow_depth` (m)
  is derived into `result.parameters`.
- **`ice_thickness_cm`** (not metres) for young ice: the optimiser's
  log10(x+1) conditioning is a no-op on the metre scale. Metres are derived
  into `result.parameters["ice_thickness"]`.

---

## 2. User guide by data source

### 2.1 Field spectroradiometer (e.g. ASD, 350–2500 nm @ 1 nm)

```python
import numpy as np
from biosnicar.sea_ice.spectral_utils import (
    resample_to_model_grid, estimate_instrument_uncertainty,
)
from biosnicar.sea_ice.retrieve import retrieve_sea_ice

# 1. Resample to the 480-band model grid (Gaussian SRF convolution)
spectrum = resample_to_model_grid(asd_values, asd_wavelengths_nm,
                                  method="gaussian", fwhm=3.0)
# Bands outside instrument coverage are NaN — build the mask:
mask = np.isfinite(spectrum)

# 2. Per-band uncertainty from a simple SNR model
unc = estimate_instrument_uncertainty(np.nan_to_num(spectrum), {"snr": 200})

# 3. Retrieve (always pass the acquisition month and geometry)
result = retrieve_sea_ice(
    observed=np.nan_to_num(spectrum),
    wavelength_mask=mask,
    obs_uncertainty=unc,
    solzen=61, direct=1, known_month=7,
)
print(result.summary())
print(result.quality_flag_description())
print(result.classification_summary())   # WMO + SIGRID-3
```

### 2.2 Satellite bands (Sentinel-2, Landsat 8/9, PlanetScope, MODIS)

Provide surface (atmospherically corrected) albedo per band. Per-band
1-sigma uncertainties default from the platform SNR table
(`biosnicar.sea_ice.sensor_config.PLATFORM_SNR`) when not supplied.

```python
result = retrieve_sea_ice(
    observed=[0.71, 0.68, 0.65, 0.52, 0.08],
    platform="sentinel2",
    observed_band_names=["B2", "B3", "B4", "B8", "B11"],
    solzen=62, direct=1, known_month=6,
)
```

Include at least VIS + NIR bands; a SWIR band (S2 B11, L8 B6) helps
discriminate snow from bare ice — the per-type classification band masks
ensure SWIR is only *used* for the types it helps (see §4.2).

### 2.3 Whole scenes (batch + spatial export)

Requires the geo extras: `pip install biosnicar[geo]`.

```python
from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch

scene = retrieve_sea_ice_batch(
    image,                                # (H, W, bands) or (N, bands)
    platform="sentinel2",
    observed_band_names=["B2", "B3", "B4", "B8", "B11"],
    solzen=62, direct=1, known_month=6,
    spatial_coords=latlon,                # (H, W, 2) → enables H3 export
    crs="EPSG:32633", transform=affine,   # → enables GeoTIFF export
    n_jobs=-1, chunksize=500,
)
print(scene.summary())              # per-type counts + flagged pixels
ds = scene.to_xarray()              # surface_type, confidence, cost,
                                    # quality_flags, param_*, unc_*, lat/lon
scene.to_geotiff("out.tif")         # multi-band float32 raster
scene.to_h3_geojson("out.geojson", resolution=9)   # ~0.1 km² hexagons
scene.to_netcdf("out.nc")
df = scene.to_dataframe()
```

Masked/cloudy pixels (any non-finite value) are skipped and carry
`surface_type_code = 255`. The integer code table is
`biosnicar.sea_ice.SURFACE_TYPE_CODES` (0=FYI_bare … 6=young_ice).
Per-pixel L-BFGS-B is practical to ~100k pixels; the `SeaIceSceneResult`
interface is engine-agnostic, so a vectorised inverse network can replace the
optimiser later without changing any export.

### 2.4 Publication-grade uncertainty (MCMC)

```python
result = retrieve_sea_ice(
    observed=spectrum, solzen=60, direct=1, known_month=7,
    method="mcmc", mcmc_walkers=32, mcmc_steps=2000, mcmc_burn=500,
)
chains = result.all_fits[result.surface_type].chains   # (steps, walkers, n_params)
```

`uncertainty` then holds posterior standard deviations instead of the Hessian
approximation. Runtime is minutes per observation (vs seconds for L-BFGS-B) —
use on selected pixels, not scenes. See [INVERSION.md](INVERSION.md) for
sampler details and convergence diagnostics.

### 2.5 Optimal estimation (`method="oe"`) — calibrated, with information content

Optimal estimation (Rodgers 2000) is the principled Bayesian path: it combines
the measurement (weighted by its error covariance) with the prior and returns a
posterior mean **and covariance**, plus two diagnostics nothing else provides —
the **averaging kernel** (how much of each answer came from the data vs the
prior) and **degrees of freedom for signal (DFS)** (how many parameters the
observation genuinely constrained). Classification becomes proper Bayesian model
selection: per-surface-type **probabilities** from each type's model evidence.

```python
res = retrieve_sea_ice(observed=spectrum, solzen=60, direct=1, known_month=7,
                       method="oe", obs_uncertainty=sigma_per_band)
res.surface_type          # winner
res.confidence            # winning type's posterior probability (calibrated)
res.class_probabilities   # {type: probability}, sums to 1
res.dfs                   # information content of the winning fit
res.averaging_kernel_diag # {param: 0..1}  ~1 = measured, ~0 = prior-driven
res.uncertainty           # 1-sigma per parameter (posterior covariance)
```

Why reach for it: it is the option that gives **calibrated uncertainty** (for an
ensemble to fuse), **honest provenance** (the averaging kernel flags
prior-driven answers — e.g. summer-snow brightness set by the seasonal prior,
not the spectrum), and a per-band-set **information budget** (DFS is high for
hyperspectral, low for 4-band satellite — so the same code retrieves a rich
state from a spectrum but reports only what a few bands can support). Speed is
comparable to or faster than L-BFGS-B per pixel. Worked demonstration:
`examples/17_optimal_estimation.py`.

> **Honest status:** the measurement covariance currently contains instrument
> noise only. The forward-model-error term — which makes the *uncertainties*
> (not the point estimates) trustworthy — is pending empirical calibration
> against field residuals (see [SEA_ICE_DEVELOPMENT_PLAN.md](SEA_ICE_DEVELOPMENT_PLAN.md)).
> Until then, treat OE error bars as optimistic.

---

## 3. Reading the result

```text
SeaIceRetrievalResult
  surface_type / surface_description   what won, in plain language
  confidence                           (2nd-best − best) / 2nd-best cost; 0 = tie
  parameters                           retrieved values + derived quantities
                                       (snow_depth, ice_thickness in SI units)
  uncertainty                          1-sigma per parameter (Hessian or posterior)
  cost / cost_per_type                 classification chi-squared, winner and all
  quality_flags                        uint8 bitmask — see below
  all_fits                             full per-type RetrievalResult objects
  to_outputs()                         Outputs object → .BBA, .to_platform(...)
  to_wmo() / to_sigrid3()              ice-chart translations with caveats
```

### Quality flags

Every retrieval carries a `uint8` bitmask
(`biosnicar.sea_ice.quality_flags.QualityFlag`); unpack with
`result.quality_flag_description()`:

| Flag | Bit | Meaning | Typical action |
|---|---|---|---|
| `poor_fit` | 0x01 | unweighted RMS albedo residual > 0.03 — no candidate explains the spectrum (scale-free across spectral/band modes and any uncertainty weighting) | inspect: mixed pixel, cloud, unmodelled surface |
| `low_confidence` | 0x02 | confidence < 0.20 | accept type cautiously, check `cost_per_type` |
| `at_bounds` | 0x04 | a parameter within 1% of its training bound | parameter value is a limit, not an estimate |
| `spectrally_ambiguous` | 0x08 | best and 2nd-best cost within 10% | both candidate types are plausible |
| `no_convergence` | 0x10 | optimiser did not converge | re-run with different x0 or MCMC |
| `open_water_likely` | 0x20 | classified as open water (informational) | — |
| `young_ice_likely` | 0x40 | classified as young ice (informational) | — |

Thresholds are empirical from the SHEBA validation suite. In batch output the
bitmask is the `quality_flags` array (GeoTIFF band 4 by default).

### Which parameters can you trust?

From the synthetic parameter-retrieval validation (50 forward-model spectra
per type, seed 2026 — full table in
`tests/validation_data/parameter_retrieval_results.md`):

- **Well identified (R² 0.97–1.00)**: snow grain radius, SSL grain radius,
  pond depth, young-ice thickness, black carbon; bubble radius and wind speed
  follow at R² 0.66–0.86.
- **Not identifiable from albedo alone (negative R²)**: sea-ice temperature
  under snow/pond/SSL, brine volume when bubble scattering dominates, ρ_DL,
  ocean albedo under thicker young ice, tau_snow beyond optical saturation
  (~10 grain-radius units of snow). These rely on the seasonal priors, return
  with large uncertainties, and may sit at bounds — that is the physics, not
  a bug. Constrain them with `fixed_params` or `regularization` when auxiliary
  data exist.

### `known_month` — always pass it when known

Seasonal Gaussian priors block physically impossible solutions (the classic
failure: fitting August spectra with −25 °C snow). Summer months additionally
exclude `young_ice` from the candidate fleet; freeze-up months add a thin-ice
thickness prior. Details: [INVERSION.md § known_month](INVERSION.md#known_month--seasonal-physical-priors).

---

## 4. Technical architecture

### 4.1 Module map

```text
biosnicar/sea_ice/
  emulator_configs.py   SEA_ICE_EMULATOR_CONFIGS registry: params/bounds,
                        transform_fn (params → run_model kwargs), band_mask,
                        emulator_file or model_factory; loader
  retrieve.py           retrieve_sea_ice(), retrieve_sea_ice_batch(),
                        SeaIceRetrievalResult, classification cost (band masks),
                        seasonal priors, derived parameters
  open_water.py         OpenWaterModel (analytical, Emulator-compatible duck type)
  quality_flags.py      QualityFlag bitmask + thresholds
  scene_result.py       SeaIceSceneResult (xarray core, exports), SURFACE_TYPE_CODES
  sensor_config.py      PLATFORM_SNR → default obs_uncertainty
  spectral_utils.py     resample_to_model_grid, vis masks, SNR model
  ice_chart_mapping.py  WMO / SIGRID-3 translation
  sea_ice_optics.py     layer_type=4 brine + bubble optics (forward model)
biosnicar/optical_properties/column_OPs.py
                        _compute_young_ice_ops() — layer_type=6 thin slab
biosnicar/inverse/optimize.py
                        core retrieve(): bounds, log-space conditioning,
                        L-BFGS-B (+DE pre-search), Nelder-Mead, DE, emcee MCMC
scripts/build_sea_ice_emulators.py     train/retrain emulators
scripts/experiments/fyi_bare_audit.py  held-out accuracy audit harness
tests/validation_data/                 SHEBA suite + parameter validation
```

### 4.2 Classification vs fitting (band masks)

Parameter fitting always uses the **full observation**. Classification
re-scores each fitted candidate on a per-type wavelength mask
(`band_mask` in the config registry):

- `vis_only` (400–1000 nm): `FYI_bare`, `FYI_summer`, `MYI_bare` — white-ice
  SWIR overlaps coarse snow and *reduces* summer bare-ice accuracy from 100%
  to 42% if included.
- `vis_swir` (400–2500 nm): `FYI_snow`, `FYI_pond`, `young_ice`,
  `open_water` — SWIR is discriminative (grain size, water absorption,
  transmittance).

Masked chi-squared is rescaled to the full observation band count so costs
remain comparable across masks, and the Gaussian-prior penalty is included so
`known_month` keeps steering classification. In band mode the mask resolves
through each band's SRF-weighted centre wavelength. Effect on SHEBA:
full-spectrum accuracy unchanged; satellite band-mode summer accuracy
81%/75% → 88%/88% (S2/L8).

### 4.3 The duck-typed model interface

Anything exposing `param_names`, `bounds`, `predict(**params) → (480,)` and
`flx_slr` can join the fleet — neural emulators and analytical models are
interchangeable (this is how `open_water` works with zero training data). To
add a surface type: write a `transform_fn` (or analytical class), register it
in `SEA_ICE_EMULATOR_CONFIGS` with bounds and a `band_mask`, train if needed
via `scripts/build_sea_ice_emulators.py`, and add WMO/SIGRID-3 entries in
`ice_chart_mapping.py`. See
[SEA_ICE_EMULATOR.md § Building Custom Emulators](SEA_ICE_EMULATOR.md#building-custom-emulators).

### 4.4 Optimisation details

Per candidate type: L-BFGS-B polished from a lightweight differential-
evolution pre-search (escapes local minima), with order-of-magnitude
parameters (BC, bubble radius, tau_snow, ice_thickness_cm, …) conditioned in
log10(x+1) space for both Latin-hypercube training sampling and optimisation.
Uncertainties from the finite-difference Hessian (or MCMC posterior).
Emulator evaluations are ~microseconds, so a full seven-type classification
takes seconds.

---

## 5. Scientific basis and validation summary

| Claim | Evidence |
|---|---|
| Spring (dry) snow classification | SHEBA Grenfell & Light (2007): 6/7 full-spectrum, 7/7 S2/L8 band mode |
| Summer bare ice classification | SHEBA: 16/16 full-spectrum with `known_month`; 14/16 band mode |
| Summer **snow** classification | Smith 2021 MOSAiC (independent hold-out): only 2/44 as FYI_snow — melting summer snow is a *true optical degeneracy* with SSL/bare ice (VIS–NIR ~50/50 separable; SWIR encodes shared grain/wetness), not a missing class (§6.4–6.5). Spring numbers do **not** generalise to melt-season snow. |
| Melt pond depth | Morassutti (1995): 77% classification over 504 records; depth accuracy rises from ~7% (<5 cm) to ~94% (>30 cm) — the only metric with independent in-situ ground truth |
| Young ice albedo vs thickness | Grenfell & Maykut (1977) Table 3: all checkpoints within ±0.03 (two-stream slab, frazil scattering 3.0 m⁻¹) |
| Open water albedo | Fresnel/Cox & Munk physics: BBA 0.02 (SZA 20°) → 0.07 (60°) → ~0.3 (80°, calm); wind darkens high-SZA water |
| Emulator fidelity | FYI_bare held-out spectral R² 0.9985, BBA MAE 0.0023 (2000 forward-model spectra) |
| Parameter identifiability | Synthetic validation, seed 2026 — see §3 |

Full tables, provenance caveats (which labels are independent vs inferred),
and known limitations: [sea_ice_validation.md](sea_ice_validation.md).

### Honest limitations

- **`young_ice` and `open_water` have no independent spectral field
  validation.** Their retrieval performance is demonstrated only on synthetic
  observations from the same physics implementations used for inversion
  (open water is fully self-inverting — the analytical model has no
  emulator). Young ice is anchored to Grenfell & Maykut (1977) at the
  broadband level only; open water to literature albedo parameterisations.
  Treat these two classes as physics-based with synthetic-only verification
  until field spectra are run through the system. The snow/bare/pond classes
  are validated against real SHEBA and Morassutti spectra.
- Shallow ponds (<10 cm) genuinely overlap bare ice spectrally —
  classification ~30% there; rely on the `spectrally_ambiguous` flag.
- Classification is per-pixel optical state: ice concentration, floe size,
  ridging and thickness of mature ice require independent measurement
  (encoded as `??` in SIGRID-3 output).
- All inputs must be *surface* albedo (atmospherically corrected); no
  atmospheric correction is performed.
- Mixed pixels are fitted as the single best pure type; the `poor_fit` flag
  is the main mixed-pixel symptom.
- The FYI_snow temperature bound is intentionally capped at −5 °C: relaxing
  it lets warm-snow solutions mimic summer bare ice (validated regression).
