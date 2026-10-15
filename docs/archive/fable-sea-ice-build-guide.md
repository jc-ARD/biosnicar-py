# BioSNICAR Sea Ice Extension — Fable Build Guide

**Created**: 2026-06-12  
**Status**: Planning document — for use as Fable session context  
**Branch**: feature/sea-ice-mvp  
**Goal**: Bring the sea ice extension to research-grade quality for satellite inversions,
drone hyperspectral data, and field spectroradiometers. The build is split into five
blocks in rough priority order.

---

## 0. Project context

BioSNICAR-py is a Python radiative transfer model for snow and ice surfaces. It solves
the adding-doubling radiative transfer equation for a multilayer column and returns
spectral albedo, BBA, and flux components.

The sea ice extension (`biosnicar/sea_ice/`) adds:
- `layer_type=4` (sea ice) and `layer_type=5` (melt pond) forward modelling
- Maxwell-Garnett effective medium for ice+brine; air bubble scattering via LUT
- Five neural-network emulators for fast inversion: `FYI_bare`, `FYI_snow`,
  `FYI_summer`, `MYI_bare`, `FYI_pond`
- `retrieve_sea_ice()` — fits all five emulators against an observed spectrum,
  classifies by lowest chi-squared, returns `SeaIceRetrievalResult`
- `known_month` seasonal priors that block physically impossible temperatures
- WMO / SIGRID-3 secondary classification output

All 613 tests pass. The codebase is in `biosnicar/sea_ice/`. Key files:

| File | Role |
|---|---|
| `biosnicar/sea_ice/sea_ice_optics.py` | Forward model: τ/ω/g per layer |
| `biosnicar/sea_ice/effective_medium.py` | Maxwell-Garnett brine+ice |
| `biosnicar/sea_ice/brine_volume.py` | Cox & Weeks brine volume; `invert_brine_volume()` |
| `biosnicar/sea_ice/emulator_configs.py` | Training configs + transform functions |
| `biosnicar/sea_ice/retrieve.py` | `retrieve_sea_ice()`, `SeaIceRetrievalResult` |
| `biosnicar/sea_ice/ice_chart_mapping.py` | WMO/SIGRID-3 mapping |
| `biosnicar/inverse/optimize.py` | Core `retrieve()` function |
| `biosnicar/inverse/emulator.py` | `Emulator` class |
| `docs/young-ice-build-spec.md` | Full spec for the young ice extension |

---

## 1. Block A — Physics correctness

These are correctness fixes that must be resolved before any publication use.
They do not add user-visible features but change model outputs.

### A1: Audit the FYI_bare emulator R² ≈ 0.70

The `FYI_bare` emulator achieves R² ≈ 0.70 on held-out spectra despite the
`brine_volume_fraction` reparameterisation. This is notably worse than the other
emulators (R² > 0.97). The root cause is unclear.

**Investigate in this order:**

1. **Residual degeneracy**: Check whether `rho_DL` (drained layer density) and
   `brine_volume_fraction` are degenerate for the FYI_bare transform. In
   `_transform_fyi_bare`, a high-rho DL at low Vb may produce the same spectrum
   as a low-rho DL at higher Vb, since both affect scattering. If degenerate,
   replace the pair with a combined parameter (e.g., effective scattering optical
   depth of the drained layer, `tau_DL`).

2. **Training sample density**: FYI_bare currently uses 30,000 samples for 6
   parameters. For a 6D space, the effective density may be too low in some
   subregions. Try doubling to 60,000 and check whether R² improves.

3. **PCA dimensionality**: The FYI_bare spectrum requires ~27 PCA components
   (reported in docs/sea_ice_validation.md). If the emulator's output layer is
   not wide enough to represent this, increase the output PCA dimension beyond
   the current default.

4. **Architecture**: The current FYI_bare network is (256, 256, 128, 64). Try
   adding a layer: (256, 256, 256, 128, 64). The increase in parameters is modest
   (< 50k), training time impact is small.

**Acceptance criterion**: FYI_bare R² > 0.90 on a 2000-spectrum held-out test
set, without degrading retrieval accuracy on the SHEBA validation suite.

### A2: Snow optical depth reparameterisation (FYI_snow)

In SHEBA validation, `snow_depth` and `snow_grain_radius` for `FYI_snow` both
hit their training bounds. This indicates the parameter combination is degenerate
— the radiative response depends on their product (optical depth), not independently.

Replace the pair with two better-conditioned parameters:
- `tau_snow` = `snow_depth / snow_grain_radius` (effective optical depth proxy)
- `snow_grain_radius` (retained — it controls the spectral shape independently)

Or more physically:
- `snow_depth` retained (it is observable independently from albedo via SWE)
- `log(snow_grain_radius)` — log-space since the NIR sensitivity is roughly
  logarithmic in grain size

The simpler fix: add both `snow_depth` and `snow_grain_radius` to `_LOG_SPACE_PARAMS`
in `biosnicar/inverse/optimize.py` and `biosnicar/inverse/emulator.py`. This won't
eliminate the degeneracy but will move the bound-hitting from linear space to
log space, where the optimizer has more freedom.

**Check**: Does the existing code already list these in `_LOG_SPACE_PARAMS`? If
not, add them. Then re-run the SHEBA spring classification test and confirm that
bound-hitting warnings drop substantially.

**Acceptance criterion**: FYI_snow SHEBA spring retrievals produce < 10% of
samples with `at_bounds` quality flags (see Block C1).

### A3: Verify MYI air bubble LUT coverage

The `sea_ice_bubble_radius` for MYI ranges 200–2000 µm. The `bubbly_air.npz` LUT
has 10 µm grid spacing throughout this range (verified — 549 total entries from
10 to 25000 µm). The `_nearest_lut_radius` snap function is correct.

**However**: check that the snap function handles the case where the requested
radius falls outside the LUT bounds (< 10 µm or > 25000 µm) gracefully. Add a
clamp and a warning if the requested radius is outside the LUT range, rather than
silently using the nearest endpoint.

**File**: `biosnicar/sea_ice/sea_ice_optics.py`, function `_nearest_lut_radius`.

---

## 2. Block B — Missing surface classes

These are new classification targets. Without them, `retrieve_sea_ice()` forces
every spectrum into one of five existing types, which causes the observed failure
modes.

### B1: Open water class

**This is the highest-priority single addition.** Open water is currently
misclassified as `FYI_pond` (dark albedo) or `FYI_bare` (low NIR). Adding it
as a sixth class fixes this directly.

**Physics**: Open water albedo has two components:
1. Specular Fresnel reflection at the air–water interface (depends on solar
   zenith angle and wind speed / roughness)
2. Diffuse upwelling from sub-surface volume scattering and absorption
   (controlled by water optical properties — essentially zero in NIR, non-zero
   in blue VIS due to Rayleigh and particle scattering)

A suitable first-order analytical model:
```
α_water(λ) = R_fresnel(solzen, wind_speed) + R_subsurface(λ)
```
where `R_subsurface(λ)` can be taken from a tabulated pure-seawater spectrum
(Morel & Maritorena 2001; or simply a low-albedo template from the literature).

**Implementation approach** (no training data needed):

Create `biosnicar/sea_ice/open_water.py` with:
```python
class OpenWaterModel:
    """Analytical open water albedo model.
    
    Parameters: solzen (20-80°), wind_speed_ms (0-15 m/s).
    No training required — forward model is analytical.
    """
    param_names = ["solzen", "wind_speed_ms"]
    
    def predict(self, solzen, wind_speed_ms=3.0, **kwargs):
        """Return 480-band open water albedo spectrum."""
        ...
    
    def predict_platform(self, platform, band_names, **kwargs):
        """Return band-averaged albedo for the given platform."""
        ...
```

The `OpenWaterModel.predict()` should implement:
- Cox (1974) or Wu (1990) Fresnel model for rough water (solzen + wind_speed)
- Pure-seawater subsurface contribution from tabulated data or Morel & Maritorena
- Typical Arctic open water: BBA ≈ 0.03–0.07, monotonically decreasing in NIR

Wrap `OpenWaterModel` with an interface matching `Emulator` so it can be passed
to `retrieve_sea_ice()` alongside the five neural network emulators. The
`retrieve_sea_ice()` function already accepts an `emulators` dict — just add
`"open_water": OpenWaterModel()` to the default fleet.

The key spectral signatures that distinguish open water:
- Very low broadband albedo (< 0.1)
- Near-monotonic decrease from blue VIS to NIR
- No reflectance peak at 550 nm (unlike shallow ponds)
- Strong decline beyond 700 nm (no ice scattering)

**Acceptance criterion**: `retrieve_sea_ice()` with a synthetic open water
spectrum (BBA ≈ 0.04–0.06, solzen=60) classifies as `open_water` with
confidence > 0.90.

### B2: Young ice (Beer-Lambert thin-slab model)

The full build specification is in `docs/young-ice-build-spec.md` (7 tasks,
Y-1 through Y-7). Key points:

- **Physics**: thin-slab Beer-Lambert model, NOT Maxwell-Garnett. The ice is
  semi-transparent; the sub-ice ocean contributes to observed albedo. The formula
  is Grenfell & Maykut (1977):
  ```
  α(λ) = ρ_surface(λ) + [1−ρ_surface(λ)]² × T²(λ) × R_ocean(λ)
                         ──────────────────────────────────────────
                         1 − ρ_surface(λ) × R_ocean(λ) × T²(λ)
  ```
  where `T(λ) = exp(−κ_eff(λ) × d)`.

- **New layer_type=6** in the adding-doubling solver (bypasses the τ/ω/g
  pipeline and returns a boundary-condition albedo directly)

- **Parameters**: `ice_thickness` (0.005–0.30 m), `sea_ice_temperature` (−20 to
  −2°C), `sea_ice_salinity` (10–35 psu — new ice is NOT desalinated), `ocean_albedo`
  (0.03–0.08), `solzen`, `direct`

- **`ice_thickness` should be in log-space** for both sampling and inversion
  (transmittance is exponential in thickness)

- **WMO mapping**: already stubbed in `ice_chart_mapping.py` — SA (< 1cm), SB
  (1–10cm), SI (10–15cm), SJ (15–30cm)

- **`known_month` prior for young ice**: freeze-up months (10, 11, 12, 1, 2)
  get `ice_thickness ~ (0.05, 0.08)`. Summer months (5–9) should explicitly
  exclude young ice — add a summer exclusion flag.

- **Validation target**: Grenfell & Maykut (1977) Table 3: grease ice BBA ≈
  0.05–0.08, dark nilas ≈ 0.08–0.12, grey ice ≈ 0.15–0.22.

Follow the task list in `docs/young-ice-build-spec.md` exactly. The spec is
implementation-ready; do not re-derive the physics.

---

## 3. Block C — Retrieval robustness

These changes make `retrieve_sea_ice()` more reliable in real-world use where
spectra are noisy, band combinations vary, and not every pixel belongs to a
clean surface type.

### C1: Quality flag system

Add a `quality_flags` field to `SeaIceRetrievalResult` and a corresponding
array to the batch output.

Define flags as a `uint8` bitmask in `biosnicar/sea_ice/quality_flags.py`:

```python
class QualityFlag:
    POOR_FIT           = 0x01  # best cost > POOR_FIT_THRESHOLD
    LOW_CONFIDENCE     = 0x02  # confidence < LOW_CONFIDENCE_THRESHOLD
    AT_BOUNDS          = 0x04  # any retrieved parameter within 1% of training bound
    SPECTRALLY_AMBIGUOUS = 0x08  # best and second-best cost within 10%
    NO_CONVERGENCE     = 0x10  # optimizer reported convergence=False
    OPEN_WATER_LIKELY  = 0x20  # classified as open_water (informational)
    YOUNG_ICE_LIKELY   = 0x40  # classified as young_ice during freeze-up (informational)

POOR_FIT_THRESHOLD      = 0.05   # chi-squared — empirical from SHEBA validation
LOW_CONFIDENCE_THRESHOLD = 0.20  # confidence metric (see retrieve.py)
```

Populate `quality_flags` in `retrieve_sea_ice()` after the winner is determined.
The `AT_BOUNDS` flag should check the winning emulator's retrieved parameters
against its config bounds in `SEA_ICE_EMULATOR_CONFIGS`.

Add `quality_flag_description()` → `dict[str, bool]` method to `SeaIceRetrievalResult`
that unpacks the bitmask into human-readable flag names.

### C2: Per-emulator band selection (fixes snow/ice misclassification)

**Background**: Adding SWIR bands (1100–2000 nm) to the classification hurts
summer bare ice accuracy (100% → 42%) because bare ice SWIR overlaps with coarse
snow. The fix is per-emulator band masks, not a global wavelength selection.

Add an optional `band_mask` field to each entry in `SEA_ICE_EMULATOR_CONFIGS`:

```python
"FYI_bare": {
    ...
    "band_mask": "vis_only",     # 400-1000 nm only for classification chi-squared
},
"FYI_snow": {
    ...
    "band_mask": "vis_swir",     # all bands — SWIR helps discriminate from bare
},
```

The `band_mask` values `"vis_only"` and `"vis_swir"` resolve to wavelength index
slices. Respect this mask when computing the chi-squared for classification in
`retrieve_sea_ice()`, but NOT when fitting emulator parameters (parameter fitting
always uses the mask the emulator was trained with).

Suggested band_mask assignments based on validation results:
- `FYI_bare`: `"vis_only"` (400–1000 nm)
- `FYI_snow`: `"vis_swir"` (400–2500 nm — SWIR distinguishes from bare)
- `FYI_summer`: `"vis_only"` (SSL spectrum has distinct VIS shape; SWIR ambiguous)
- `MYI_bare`: `"vis_only"` (large bubbles → high NIR; SWIR not needed)
- `FYI_pond`: `"vis_swir"` (pond depth affects SWIR via water absorption)
- `open_water`: `"vis_swir"` (water strongly absorbs NIR/SWIR — definitive)
- `young_ice`: `"vis_swir"` (thin ice transmittance is wavelength-dependent through SWIR)

This is a design decision — document the rationale in `SEA_ICE_EMULATOR.md`.

### C3: Sensor-specific obs_uncertainty defaults

When `obs_uncertainty=None` and a `platform` is specified, auto-populate
`obs_uncertainty` from a platform-specific SNR table rather than leaving it as
equal-weight chi-squared.

Add `PLATFORM_SNR` dict to `biosnicar/sea_ice/retrieve.py` (or a new
`biosnicar/sea_ice/sensor_config.py`):

```python
PLATFORM_SNR = {
    # {platform: {band_name: typical_1sigma_albedo_uncertainty}}
    # Values are approximate L2A product accuracy estimates
    "sentinel2": {
        "B2": 0.02, "B3": 0.02, "B4": 0.02,
        "B8": 0.025, "B8A": 0.025, "B11": 0.03, "B12": 0.04,
    },
    "landsat8": {
        "B2": 0.02, "B3": 0.02, "B4": 0.02, "B5": 0.025,
        "B6": 0.03, "B7": 0.04,
    },
    "planetscope": {
        "B1": 0.025, "B2": 0.025, "B3": 0.025, "B4": 0.03,
    },
    "modis": {
        "B1": 0.02, "B2": 0.025, "B3": 0.02, "B4": 0.02,
        "B5": 0.03, "B6": 0.03, "B7": 0.04,
    },
}
```

In `retrieve_sea_ice()`, when `platform` is not None and `obs_uncertainty` is
None, look up `PLATFORM_SNR[platform]` and populate `obs_uncertainty` from the
matched band names. Add a `WORLDVIEW` entry for Maxar WorldView-3 when that data
becomes available.

---

## 4. Block D — Input/output pipeline

These additions make the system usable at scale with real satellite data.

### D1: Batch retrieval interface

Add `retrieve_sea_ice_batch()` to `biosnicar/sea_ice/retrieve.py`:

```python
def retrieve_sea_ice_batch(
    observed,                    # (N, bands) or (H, W, bands) array
    n_jobs=-1,                   # joblib parallelism; -1 = all CPUs
    chunksize=500,               # pixels per job
    spatial_coords=None,         # (N, 2) or (H, W, 2) lat/lon array
    crs=None,                    # EPSG string, e.g. "EPSG:4326"
    transform=None,              # affine transform (rasterio.Affine) for image input
    **kwargs,                    # passed through to retrieve_sea_ice()
) -> SeaIceSceneResult:
    ...
```

The function:
1. Flattens spatial arrays to (N, bands) if H×W×bands input given
2. Dispatches pixel chunks to `retrieve_sea_ice()` via joblib
3. Assembles results into `SeaIceSceneResult`
4. Re-shapes to (H, W) if spatial input was 2D

**Design note**: This approach (parallelised L-BFGS-B per pixel) is practical
for scenes up to ~100k pixels. For regional mosaics (millions of pixels), the
architecture will need to migrate to a vectorised inverse network trained on
the emulator outputs. Design the `SeaIceSceneResult` API now so the underlying
retrieval engine can be swapped later without changing the output interface.

### D2: `SeaIceSceneResult` — spatial output object

Create `biosnicar/sea_ice/scene_result.py`:

```python
@dataclass
class SeaIceSceneResult:
    """Batch retrieval result for a spatial scene.
    
    Core object is an xarray Dataset. All export methods operate on it.
    """
    _ds: xr.Dataset              # internal — access via to_xarray()
    crs: Optional[str]           # EPSG string
    transform: Optional[Any]     # rasterio.Affine or None
    
    # Array fields (all shape N or H×W):
    # surface_type       : str (categorical)
    # surface_type_code  : uint8 (integer code for raster export)
    # confidence         : float32
    # cost               : float32
    # quality_flags      : uint8 (bitmask)
    # parameters.*       : float32 per retrieved parameter
    # uncertainty.*      : float32 per retrieved parameter
    # lat, lon           : float32 if coordinates provided
    
    def to_xarray(self) -> xr.Dataset:
        """Return the underlying xarray Dataset."""
        return self._ds
    
    def to_dataframe(self) -> pd.DataFrame:
        """Flatten to a pandas DataFrame (one row per pixel)."""
        return self._ds.to_dataframe().reset_index()
    
    def to_geotiff(self, path: str, variables=None, compress="lzw"):
        """Write selected variables to a multi-band GeoTIFF.
        
        Requires rioxarray and rasterio. The CRS and affine transform
        must have been supplied to retrieve_sea_ice_batch().
        
        variables: list of variable names to write; defaults to
            ['surface_type_code', 'confidence', 'cost', 'quality_flags'].
        """
        ...
    
    def to_netcdf(self, path: str):
        """Write the full Dataset to NetCDF4 (xarray native)."""
        self._ds.to_netcdf(path)
    
    def to_h3_geojson(self, path: str, resolution: int = 8,
                      variables=None, aggregation="mode"):
        """Aggregate pixels to H3 hexagonal cells and write GeoJSON.
        
        resolution: H3 resolution (0–15). At resolution 8, cells are ~0.7 km²;
            at resolution 9, ~0.1 km². For Sentinel-2 at 10m, resolution 9–10
            is appropriate.
        variables: list of variables to include in each feature's properties.
        aggregation: 'mode' for categorical (surface_type), 'mean' for continuous.
            Applied per-variable: categorical vars use mode, continuous use mean.
        
        Requires h3 (pip install h3). GeoJSON output is a FeatureCollection
        with one feature per H3 cell.
        """
        ...
```

**Internal representation**: Each variable is a named array in the xarray Dataset.
Categorical variables (surface_type) stored as string; also add a `surface_type_code`
integer array for raster export (0=FYI_bare, 1=FYI_snow, 2=FYI_summer, 3=MYI_bare,
4=FYI_pond, 5=open_water, 6=young_ice, 255=no_data).

Add a `SURFACE_TYPE_CODES` dict to `retrieve.py` for the encoding.

**Dependencies to add to `requirements.txt` or `pyproject.toml`**:
- `xarray>=2023.0`
- `rioxarray>=0.15`
- `rasterio>=1.3`
- `h3>=3.7`
- `geopandas>=0.14` (for to_h3_geojson geometry construction)
- `joblib>=1.3` (for batch parallelism — likely already present)

Make these optional dependencies with a clear install instruction. The core
`retrieve_sea_ice()` should continue to work without them.

### D3: Hyperspectral resampling utility

For drone and field spectroradiometer data arriving on an instrument-native
wavelength grid (e.g., 350–2500 nm at 1 nm resolution), provide a resampling
function that converts to BioSNICAR's 480-band grid (defined in the model
wavelength config).

Add `biosnicar/sea_ice/spectral_utils.py`:

```python
def resample_to_model_grid(
    spectrum: np.ndarray,        # (N_instrument_bands,)
    instrument_wavelengths: np.ndarray,  # nm, same length
    method: str = "gaussian",    # "gaussian", "linear", "box"
    fwhm: float = None,          # instrument FWHM in nm (for Gaussian)
) -> np.ndarray:
    """Resample an observed spectrum onto BioSNICAR's 480-band wavelength grid.
    
    Uses spectral response function convolution. For field spectrometers,
    'gaussian' with the instrument FWHM is most physically correct. For
    drone hyperspectral imagers with narrow, uniform bands, 'linear' interpolation
    is adequate.
    
    Returns a 480-element array on the model grid.
    """
    ...

def trim_to_vis(spectrum_480: np.ndarray) -> np.ndarray:
    """Return the VIS-only (400-1000 nm) subset of a 480-band spectrum.
    
    Use this when preparing bare ice observations for classification — adding
    SWIR bands degrades bare ice classification accuracy (see docs/SEA_ICE_EMULATOR.md).
    """
    ...
```

Also add: `estimate_instrument_uncertainty(spectrum, snr_model)` that returns
a per-band `obs_uncertainty` array from a simple SNR model.

---

## 5. Block E — Validation and documentation

### E1: Synthetic parameter retrieval validation

The existing SHEBA validation tests classification accuracy (which surface type
wins) but not whether the retrieved *parameter values* are physically correct.

Add a dedicated validation script `tests/validation_data/parameter_retrieval_validation.py`:

For each surface type, generate 50 test spectra using the forward model
(NOT the emulator — use `transform_fn(params) → run_model(**kw)` directly),
with parameters drawn from a held-out random seed distinct from the training set.
Then run `retrieve_sea_ice(surface_types=[this_type], known_month=...)` and compare
retrieved vs true parameters.

Report for each parameter:
- Bias (mean(retrieved - true))
- RMSE
- R² (retrieved vs true)
- Fraction of retrievals where the parameter hit a training bound

This is the minimum required for a methods paper. The script should output a
formatted table that can be included in `docs/sea_ice_validation.md`.

**Important**: use `np.random.default_rng(2026)` for reproducibility. Document
the seed in the script.

### E2: MCMC integration

Wire up MCMC as a first-class `method` option in `retrieve()`:
`retrieve_sea_ice(observed=..., method="mcmc", n_samples=2000, burnin=500)`

The MCMC path should use the existing `emcee` or `scipy.stats` infrastructure
if already present, or add `emcee` as an optional dependency. The `RetrievalResult`
already has `uncertainty` fields — for MCMC, populate these with the posterior
standard deviation rather than the Hessian approximation.

The user will use MCMC on a subset of pixels for publication-quality uncertainty
quantification, not for scene-level processing. Make it opt-in and document the
expected runtime (minutes per pixel vs seconds for L-BFGS-B).

### E3: Documentation pass

After all blocks are complete, update:

- `docs/SEA_ICE_EMULATOR.md`:
  - Add `open_water` and `young_ice` to the surface types table
  - Document per-emulator band selection (C2)
  - Add parameter retrieval validation results table (E1)
  - Update the SHEBA classification accuracy table

- `docs/INVERSION.md`:
  - Add `ice_thickness` to sea ice parameters
  - Add batch retrieval API section
  - Add MCMC vs L-BFGS-B section

- `docs/sea_ice_validation.md`:
  - Add parameter retrieval validation table
  - Add young ice validation (Grenfell & Maykut 1977 comparison)

- `README.md` or top-level docs:
  - Add a quickstart example for batch satellite retrieval with spatial output

---

## 6. Architecture decisions (do not re-derive)

**Why internal labels are kept as primary (not WMO codes)**: `FYI_bare`,
`FYI_snow`, and `FYI_summer` all map to the same SIGRID-3 code `SM/SN`. The
internal labels encode optical-state distinctions that WMO codes collapse. WMO
codes are available as secondary output via `result.to_wmo()` and `result.to_sigrid3()`.

**Why VIS-only for bare ice classification**: Adding SWIR bands drops summer
bare ice accuracy from 100% to 42% because the white ice SWIR signature overlaps
with coarse snow. This is documented in `docs/SEA_ICE_EMULATOR.md` and handled
by per-emulator band masks (C2 above).

**Why brine_volume_fraction not (T, S)**: The (T, S) pair is degenerate for
bare ice emulators — many (T, S) combinations produce identical brine volumes and
thus identical spectra. `brine_volume_fraction` eliminates this degeneracy. Post-hoc
temperature recovery uses `invert_brine_volume(Vb, S_ref)` with fixed reference
salinities `FYI_BARE_S_REF=6 psu`, `MYI_BARE_S_REF=2 psu`.

**Why xarray Dataset as the core scene output**: It preserves spatial coordinates,
serialises natively to NetCDF, works with rioxarray for GeoTIFF, and bridges
naturally to H3 aggregation via geopandas. The three output formats (GeoTIFF,
H3 GeoJSON, Python pipeline) are export methods on one object, not separate code paths.

**MYI physics**: The model is physically sound. It separates brine absorption
(Maxwell-Garnett with brine inclusions — correct since MYI has low but non-zero
brine) from air bubble scattering (via bubbly_air LUT with 10 µm grid spacing,
confirmed adequate). The air fraction is derived from the density balance. No
physics change is required for MYI.

**Batch scaling path**: `retrieve_sea_ice_batch()` uses joblib L-BFGS-B (adequate
for ~100k pixels). The `SeaIceSceneResult` API is designed so the retrieval engine
can be replaced by a vectorised inverse network for regional-mosaic scale without
changing the output interface. Do not over-engineer for the inverse network now;
design the interface to be swappable.

---

## 7. Priority order

If time is limited, implement in this order:

1. **B1** (open water class) — fixes the most common real-world failure mode  
2. **C1** (quality flags) — makes classification outputs trustworthy  
3. **D1 + D2** (batch interface + SeaIceSceneResult) — needed for satellite work  
4. **D3** (hyperspectral resampling) — needed for field/drone data  
5. **A2** (snow optical depth reparameterisation) — fixes bound-hitting  
6. **A1** (FYI_bare R² investigation) — physics correctness  
7. **B2** (young ice) — full spec in `docs/young-ice-build-spec.md`  
8. **C2** (per-emulator band masks) — improves snow/ice discrimination  
9. **C3** (sensor SNR defaults) — convenience for satellite users  
10. **E1** (parameter retrieval validation) — required before publication  
11. **E2** (MCMC) — required for publication uncertainty quantification  
12. **E3** (documentation pass) — last, after all code is final  

---

## 8. Test coverage requirements

All new functionality must have tests. Follow the existing test structure in `tests/`.

- `OpenWaterModel.predict()`: test that BBA is in [0.03, 0.10] for solzen 20–80°
- Quality flags: test each flag condition individually
- `retrieve_sea_ice_batch()`: test on a 10-pixel synthetic batch
- `SeaIceSceneResult.to_xarray()`, `.to_netcdf()`, `.to_geotiff()`,
  `.to_h3_geojson()`: smoke tests (write to temp file, read back, check shape)
- `resample_to_model_grid()`: test with a known flat spectrum and confirm
  the output is also flat
- Young ice: per Y-6 spec — BBA vs thickness curve against Grenfell & Maykut values

Do not write tests that use emulator `.predict()` as both the input spectrum
generator and the retrieval target — this produces cost≈0 and zero information.
Always generate test spectra via the forward model (`transform_fn → run_model`).

---

## 9. Known constraints

- The project uses Python 3.10+. No walrus operator or structural pattern matching
  above 3.10 syntax.
- All emulator training data files are `.npz` in `data/emulators/`. File naming
  convention: `sea_ice_{surface_type}_{n_params}param.npz`.
- The 480-band wavelength grid is defined in the model config — do not hardcode
  wavelength arrays. Use `biosnicar.utils.WAVELENGTHS` or equivalent.
- Do not add comments explaining what the code does — only add comments where
  the WHY is non-obvious. No docstrings longer than 3–4 lines unless the function
  has complex physics.
- Run `pytest` before reporting any block complete. All 613 existing tests must
  continue to pass.
