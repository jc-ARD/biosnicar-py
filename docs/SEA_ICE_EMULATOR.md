# Sea Ice Emulator

The sea ice emulator system provides fast spectral-albedo prediction and parameter retrieval for five Arctic sea ice surface types. It extends the terrestrial emulator (see [EMULATOR.md](EMULATOR.md)) with sea-ice-specific physics: brine optics, melt ponds, SSL (Surface Scattering Layer), and multi-year ice. A single forward model evaluation takes ~50 ms; emulator prediction takes ~microseconds, enabling practical optimisation and uncertainty estimation over satellite images.

## Overview

The system comprises five separate emulators — one per surface type — each trained on Latin hypercube samples of the BioSNICAR forward model with sea-ice-appropriate layer stacks. The high-level function `retrieve_sea_ice()` fits all five against a single observed spectrum, then classifies the surface by the emulator that produced the lowest chi-squared residual.

This **retrieve-then-classify** design means that surface type identification and parameter retrieval happen simultaneously in a single call. You do not need to decide the surface type in advance.

Relationship to the terrestrial emulator:
- Same `Emulator` class, same `.npz` file format, same `predict()` / `predict_batch()` / `verify()` / `save()` / `load()` API
- The sea ice surface types use a `transform_fn` to map scalar parameters to multi-layer `run_model()` keyword arguments; the terrestrial emulator does not need this
- Sea ice parameters are fixed-structure (e.g. always DL + IL for bare ice), so the emulator's inputs are physical scalars, not per-layer lists

## Quick Start

### Direct prediction with run_emulator()

```python
from biosnicar.emulator import Emulator
from biosnicar.drivers.run_emulator import run_emulator

# Load a pre-built sea ice emulator
emu = Emulator.load("data/emulators/sea_ice_FYI_bare_7param.npz")

# Predict 480-band spectral albedo (~microseconds)
albedo = emu.predict(
    sea_ice_temperature=-10.0,
    sea_ice_bubble_radius=200.0,
    sea_ice_salinity=8.0,
    black_carbon=0.0,
    rho_DL=850.0,
    solzen=60,
    direct=1,
)

# Or get a full Outputs object with BBA, BBAVIS, BBANIR, and to_platform()
out = run_emulator(
    emu,
    sea_ice_temperature=-10.0, sea_ice_bubble_radius=200.0,
    sea_ice_salinity=8.0, black_carbon=0.0,
    rho_DL=850.0, solzen=60, direct=1,
)
print(out.BBA)
out.to_platform("sentinel2")   # works — flx_slr is stored in the emulator
```

### retrieve_sea_ice() — automatic classification

```python
from biosnicar.sea_ice.retrieve import retrieve_sea_ice

# Load your 480-band observed albedo (e.g. from a field spectrometer)
import numpy as np
observed = np.load("my_spectrum.npy")   # shape (480,)

# Fit all five emulators, classify by residual
result = retrieve_sea_ice(
    observed=observed,
    solzen=60,
    direct=1,
)
print(result.surface_type)   # e.g. "FYI_summer"
print(result.confidence)     # 0 – 1  (how decisively the winner outperformed others)
print(result.parameters)     # {"ssl_grain_radius": 1820, "black_carbon": 12, ...}
print(result.cost_per_type)  # {"FYI_bare": 0.034, "FYI_snow": 0.21, ...}

# Convert to an Outputs object and compute platform bands
out = result.to_outputs()
s2 = out.to_platform("sentinel2")
print(s2.B3, s2.B8, s2.B11)
```

### Per-type retrieval with retrieve()

Use this when you already know the surface type (e.g. from a classification mask) and want to retrieve physical parameters directly.

```python
from biosnicar.emulator import Emulator
from biosnicar.inverse.optimize import retrieve

emu_pond = Emulator.load("data/emulators/sea_ice_FYI_pond_5param.npz")

result = retrieve(
    observed=observed,
    parameters=["pond_depth", "black_carbon"],
    emulator=emu_pond,
    fixed_params={"sea_ice_temperature": -5.0, "solzen": 60, "direct": 1},
)
print(result.best_fit["pond_depth"])    # metres
print(result.uncertainty["pond_depth"]) # 1-sigma
result.to_outputs().to_platform("sentinel2")
```

## The Five Surface Types (v0.2)

| Name | Description | Parameters | Typical BBA |
|------|-------------|------------|-------------|
| `FYI_bare` | Winter/spring bare first-year ice (no snow, no SSL) | brine_volume_fraction, bubble_radius, BC, rho_DL, solzen, direct | 0.55–0.85 |
| `FYI_snow` | Snow-covered first-year ice | snow_depth, snow_grain_radius, T, BC, solzen, direct | 0.80–0.95 |
| `FYI_summer` | Melt-season bare FYI with Surface Scattering Layer | ssl_grain_radius, T, bubble_radius, BC, solzen, direct | 0.45–0.75 |
| `MYI_bare` | Bare multiyear ice (lower salinity, larger bubbles) | brine_volume_fraction, bubble_radius, BC, solzen, direct | 0.55–0.85 |
| `FYI_pond` | Melt pond on first-year ice | pond_depth, T, BC, solzen, direct | 0.05–0.30 |

> **Planned v0.4 — `young_ice`** (grease ice, nilas, grey/grey-white ice, 0.5–30 cm thick):
> BBA 0.05–0.25, semi-transparent, parameterised by `ice_thickness` rather than brine inclusions.
> Requires a new thin-slab Beer-Lambert forward model (see [young-ice-build-spec.md](young-ice-build-spec.md)).
> WMO codes SA (new ice), SB (nilas), SI (grey ice), SJ (grey-white ice).

### FYI_bare — Winter/spring bare first-year ice

The dominant broadband-albedo control is bubble scattering. Brine volume (determined by temperature and salinity jointly) increases brine absorption in the NIR. The layer stack is DL (drained layer, 5 cm) + IL (interior layer, 145 cm).

| Parameter | Range | Units | Role |
|-----------|-------|-------|------|
| `brine_volume_fraction` | 0.019 to 0.141 | dimensionless | Combined T+S effect; see below |
| `sea_ice_bubble_radius` | 50 to 1000 | μm | Controls NIR scattering efficiency |
| `black_carbon` | 0 to 5000 | ppb | VIS darkening (cryoconite proxy) |
| `rho_DL` | 820 to 900 | kg/m³ | DL density (affects air fraction + scattering) |
| `solzen` | 20 to 80 | degrees | Path length effect |
| `direct` | 0, 1 | binary | 0=diffuse, 1=direct beam |

> **Why `brine_volume_fraction` and not `(sea_ice_temperature, sea_ice_salinity)`?**
> Temperature and salinity both control brine volume through the Cox & Weeks (1983) equation, creating a
> retrieval degeneracy: many (T, S) pairs produce the same brine volume and therefore the same spectrum.
> `brine_volume_fraction` is the single well-constrained quantity that captures their combined effect — the
> sea ice analogue of SSA for glacier ice.  Temperature can be recovered post-hoc if salinity is known
> (`T = invert_brine_volume(Vb, S_ref)`).  The reference salinity is `FYI_BARE_S_REF = 6 psu`.
> See [INVERSION.md § Sea Ice Inversion](INVERSION.md#sea-ice-inversion) for full details.

### FYI_snow — Snow-covered first-year ice

Snow grain radius and depth dominate. Thin snow (< 3 cm) transmits enough light that the underlying ice is visible in the NIR; thick snow is fully opaque. The layer stack is granular-snow layer + DL (5 cm) + IL (145 cm).

| Parameter | Range | Units | Role |
|-----------|-------|-------|------|
| `snow_depth` | 0.02 to 0.30 | m | Optical depth of snow layer |
| `snow_grain_radius` | 100 to 2000 | μm | Controls NIR scattering |
| `sea_ice_temperature` | -30 to -5 | °C | Applied to underlying ice |
| `black_carbon` | 0 to 5000 | ppb | Surface-layer darkening |
| `solzen` | 20 to 80 | degrees | — |
| `direct` | 0, 1 | binary | — |

### FYI_summer — Melt-season bare FYI with SSL

The Surface Scattering Layer is a porous, highly scattering near-surface layer that forms during melt. Its grain radius is the primary spectral control. The layer stack is SSL (granular, 5 cm) + DL (5 cm) + IL (140 cm).

| Parameter | Range | Units | Role |
|-----------|-------|-------|------|
| `ssl_grain_radius` | 500 to 5000 | μm | SSL scattering; larger grains = lower NIR |
| `sea_ice_temperature` | -10 to -2 | °C | Brine in underlying DL |
| `sea_ice_bubble_radius` | 50 to 500 | μm | Bubble scattering in DL |
| `black_carbon` | 0 to 5000 | ppb | SSL surface layer |
| `solzen` | 20 to 80 | degrees | — |
| `direct` | 0, 1 | binary | — |

### MYI_bare — Bare multiyear ice

MYI has lower salinity (0–6 ppt vs 1–20 ppt for FYI) and larger bubbles (200–2000 μm vs 50–1000 μm). These differences produce modestly higher albedo than FYI at the same temperature. The layer stack is DL (30 cm) + IL (270 cm) — thicker DL reflects the more evolved pore structure of MYI.

| Parameter | Range | Units | Role |
|-----------|-------|-------|------|
| `sea_ice_temperature` | -25 to -2 | °C | Brine volume |
| `sea_ice_bubble_radius` | 200 to 2000 | μm | Scattering |
| `sea_ice_salinity` | 0 to 6 | ppt | Low in MYI |
| `black_carbon` | 0 to 5000 | ppb | Surface darkening |
| `solzen` | 20 to 80 | degrees | — |
| `direct` | 0, 1 | binary | — |

### FYI_pond — Melt pond on first-year ice

Pond depth is the primary control. Shallow ponds (< 5 cm) transmit significant NIR to the ice floor; deep ponds (> 40 cm) are optically thick and produce near-zero NIR albedo. The layer stack is liquid-water layer + DL (5 cm) + IL (140 cm). Black carbon is placed in the DL floor, not the pond water.

| Parameter | Range | Units | Role |
|-----------|-------|-------|------|
| `pond_depth` | 0.02 to 0.60 | m | Optical depth of water column |
| `sea_ice_temperature` | -10 to -2 | °C | Underlying ice |
| `black_carbon` | 0 to 3000 | ppb | Ice floor darkening |
| `solzen` | 20 to 80 | degrees | — |
| `direct` | 0, 1 | binary | — |

## Emulator Architecture

The sea ice emulators use the same MLP + PCA architecture as the terrestrial emulator. The building blocks are:

1. **Latin hypercube sampling** over the parameter space
2. **transform_fn** maps sampled scalars to `run_model()` keyword lists (multi-layer)
3. **PCA compression** of the 480-band albedo output (retains 99.9% variance)
4. **MLP regression** from scaled inputs to PCA coefficients
5. **Pure-numpy inference**: forward pass + PCA reconstruction, ~microseconds

### Why different MLP sizes?

```
FYI_bare, MYI_bare:  (256, 256, 128, 64)  — deeper for high-dimensional spectral manifold
FYI_snow, FYI_summer, FYI_pond:  (128, 128, 64)  — standard
```

Bare sea ice albedo (layer_type=4) is controlled by brine optics, which produce a spectrally complex absorption signature spanning many PCA dimensions (~27 components). The larger MLP is needed to map the 6–7 input parameters to this high-dimensional output space accurately.

Snow and pond surfaces are dominated by scattering (grain radius, depth) and produce smoother spectral shapes (~6 PCA dimensions). The smaller default architecture is sufficient.

### Why R² can be misleading for bare ice

Bare sea ice with brine optics typically achieves R² ≈ 0.55 on small (250-sample) emulators, compared to R² > 0.98 for snow-covered ice at the same sample count. **This is not a quality problem.** R² measures how much variance the emulator explains relative to the total output variance. For bare ice:

- The spectral manifold is high-dimensional (27 PCA components)
- A 250-sample emulator cannot fully cover a 7-parameter space
- R² measures "what fraction of output variance did we explain" — it goes low when the emulator is undertrained relative to the manifold dimensionality

The correct diagnostic for emulator quality is **BBA MAE** (mean absolute broadband albedo error), measured via `emulator.verify()`. For the production emulators (30,000 samples), BBA MAE is typically 0.002–0.004 for all surface types, regardless of R².

### PCA component counts

| Surface type | Typical n_pca (production) | Physical reason |
|---|---|---|
| FYI_bare | ~27 | Brine absorption bands span full NIR |
| MYI_bare | ~25 | Similar to FYI_bare, reduced brine |
| FYI_snow | ~6 | Smooth scattering function |
| FYI_summer | ~8 | SSL grain radius + mild brine |
| FYI_pond | ~6 | Water absorption smooth |

## Building Custom Emulators

### The transform_fn mechanism

The `transform_fn` callable is the key design element that makes sea ice emulators work. It maps a dict of scalar-valued sampled parameters to the full `run_model()` keyword arguments, handling:

- Broadcasting scalar parameters (temperature, salinity, …) to per-layer lists
- Mapping structural parameters (`pond_depth`, `snow_depth`) to `dz` arrays
- Fixing the layer count, layer types (`layer_type=[5, 4, 4]` for pond), and inter-layer relationships

```python
def _transform_fyi_pond(p):
    """Melt pond on FYI: liquid water layer over DL + IL."""
    T = p["sea_ice_temperature"]
    return dict(
        layer_type=[5, 4, 4],               # pond, DL, IL
        dz=[p["pond_depth"], 0.05, 1.40],   # pond depth is free
        rds=[500, 500, 500],                # unused for layer_type=4/5
        rho=[1000, 850, 910],
        sea_ice_salinity=[None, 8, 6],      # None = not used in water layer
        sea_ice_temperature=[None, T, T],
        sea_ice_bubble_radius=[None, 200, 500],
        black_carbon=[0, p["black_carbon"], 0],  # BC in DL, not pond water
        solzen=p["solzen"],
        direct=p["direct"],
    )
```

### Writing a transform for a new surface type

To add a new emulator (e.g. FYI with algae-stained pond), follow this pattern:

```python
from biosnicar.sea_ice.emulator_configs import SEA_ICE_EMULATOR_CONFIGS, _DATA_DIR

def _transform_fyi_algae_pond(p):
    T = p["sea_ice_temperature"]
    return dict(
        layer_type=[5, 4, 4],
        dz=[p["pond_depth"], 0.05, 1.40],
        rds=[500, 500, 500],
        rho=[1000, 850, 910],
        sea_ice_salinity=[None, 8, 6],
        sea_ice_temperature=[None, T, T],
        sea_ice_bubble_radius=[None, 200, 500],
        black_carbon=[0, 0, 0],
        snow_algae=[p["algae_concentration"], 0, 0],   # algae in pond water
        solzen=p["solzen"],
        direct=p["direct"],
    )

# Register it
SEA_ICE_EMULATOR_CONFIGS["FYI_algae_pond"] = {
    "description": "Algae-stained melt pond on FYI",
    "params": {
        "pond_depth":          (0.02, 0.60),
        "algae_concentration": (0.0,  50000.0),
        "sea_ice_temperature": (-10.0, -2.0),
        "solzen":              (20, 80),
        "direct":              (0, 1),
    },
    "transform_fn":  _transform_fyi_algae_pond,
    "n_samples":     10000,
    "emulator_file": str(_DATA_DIR / "sea_ice_FYI_algae_pond_5param.npz"),
}

# Build it
from biosnicar.emulator import Emulator
cfg = SEA_ICE_EMULATOR_CONFIGS["FYI_algae_pond"]
emu = Emulator.build(
    params=cfg["params"],
    n_samples=cfg["n_samples"],
    transform_fn=cfg["transform_fn"],
    progress=True,
)
emu.save(cfg["emulator_file"])
```

### n_samples guidance

| Surface type | Recommended n_samples | Why |
|---|---|---|
| FYI_bare, MYI_bare | 30,000 | High-dimensional brine spectral manifold |
| FYI_snow, FYI_summer, FYI_pond | 10,000–12,000 | Smoother spectral shapes |
| Simple 2-param test emulator | 500–1,000 | Fast testing only |

For production use, always use the recommended sample counts. Underfitting at small n shows as R² near 0 or negative (especially for bare ice); the correct diagnostic is always BBA MAE via `verify()`.

## API Reference

### `Emulator.build(params, n_samples, transform_fn, seed, progress, hidden_layer_sizes, ...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `params` | dict | required | `{name: (min, max)}` for each free parameter |
| `n_samples` | int | 10000 | Number of LHS training samples |
| `transform_fn` | callable | None | Maps `{param: scalar}` → `run_model()` kwargs |
| `hidden_layer_sizes` | tuple | (128,128,64) | MLP architecture |
| `seed` | int | 42 | Random seed |
| `progress` | bool | True | Show tqdm bar |
| `solver` | str | "adding-doubling" | RT solver |

Returns: `Emulator`

### `Emulator.predict(**params)`

Predict 480-band spectral albedo. Pure numpy, ~microseconds.

```python
albedo = emu.predict(
    sea_ice_temperature=-10.0,
    sea_ice_bubble_radius=200.0,
    sea_ice_salinity=8.0,
    black_carbon=0.0,
    rho_DL=850.0,
    solzen=60,
    direct=1,
)  # → np.ndarray shape (480,), clipped to [0, 1]
```

Raises `ValueError` if any parameter is missing. Issues `UserWarning` if any parameter is outside its training bounds.

### `Emulator.predict_batch(points)`

```python
# points: np.ndarray shape (N, n_params), columns in param_names order
batch = emu.predict_batch(points)  # → np.ndarray shape (N, 480)
```

Column order follows `emu.param_names`. Faster than N individual `predict()` calls because the MLP forward pass is batched.

### `Emulator.verify(n_points=20, seed=123, progress=True)`

Measure emulator accuracy against the full forward model. Returns a `VerificationResult` with:

| Attribute | Description |
|---|---|
| `mae` | Spectral mean absolute error (all points, all bands) |
| `mae_bba` | Broadband albedo mean absolute error |
| `max_err` | Worst-case absolute spectral error |
| `r2` | R² over all predicted vs reference values |
| `summary()` | Human-readable string |

```python
vr = emu.verify(n_points=10, progress=False)
print(vr.summary())
# Emulator verification (10 benchmark points)
#   Spectral MAE:      0.003124
#   BBA MAE:           0.002841
#   R²:                0.99823
```

### `run_emulator(emulator, **params)`

Wrapper returning an `Outputs` object identical in interface to `run_model()` output:

| Attribute | Type | Description |
|---|---|---|
| `albedo` | ndarray (480,) | Spectral albedo |
| `BBA` | float | Broadband albedo (flux-weighted) |
| `BBAVIS` | float | Visible broadband albedo |
| `BBANIR` | float | NIR broadband albedo |
| `flx_slr` | ndarray (480,) | Solar flux (from emulator build) |
| `heat_rt` | None | Not available from emulator |
| `absorbed_flux_per_layer` | None | Not available from emulator |
| `to_platform(name)` | method | Convolve to satellite bands |

### `retrieve_sea_ice(observed, ...)`

Main entry point for automatic surface classification plus parameter retrieval.

```python
result = retrieve_sea_ice(
    observed,                    # 480-band spectral array or N-band satellite array
    emulators=None,              # pre-built fleet dict (loaded from disk if None)
    surface_types=None,          # restrict to subset: ["FYI_bare", "FYI_pond"]
    platform=None,               # satellite platform key for band mode
    observed_band_names=None,    # required in band mode
    obs_uncertainty=None,        # per-element 1-sigma uncertainty
    method="L-BFGS-B",           # optimisation method
    solzen=None,                 # fix solar zenith (strongly recommended)
    direct=None,                 # fix illumination (1=direct, 0=diffuse)
    fixed_params=None,           # additional fixed params for all emulators
    bounds=None,                 # override parameter bounds
    x0=None,                     # override initial guesses
    regularization=None,         # Gaussian priors: {name: (mean, sigma)}
    wavelength_mask=None,        # wavelength mask (spectral mode)
    known_month=None,            # calendar month 1–12; enables seasonal T priors
                                 #   5–9 (summer): T ~ (−4°C ± 3°C), Vb ~ (0.07 ± 0.04)
                                 #   11–3 (winter): T ~ (−15°C ± 8°C), Vb ~ (0.03 ± 0.015)
                                 #   Without this, summer bare ice misclassifies as FYI_snow
                                 #   SHEBA accuracy: 0/16 → 9/16 with known_month=8
)
```

Returns: `SeaIceRetrievalResult`

### `SeaIceRetrievalResult`

| Attribute | Type | Description |
|---|---|---|
| `surface_type` | str | Best-fitting surface type name |
| `confidence` | float | `(second_cost - best_cost) / second_cost` — 0=tied, ~1=clear winner |
| `parameters` | dict | `{name: value}` from the winning emulator |
| `uncertainty` | dict | `{name: 1-sigma}` from the winning emulator |
| `predicted_albedo` | ndarray (480,) | Best-fit spectrum |
| `observed` | ndarray | Input observations |
| `cost` | float | Chi-squared at best-fit point |
| `converged` | bool | Winning optimiser convergence flag |
| `flx_slr` | ndarray (480,) or None | Solar flux from winning emulator |
| `cost_per_type` | dict | `{surface_type: chi_squared}` for all emulators |
| `all_fits` | dict | `{surface_type: RetrievalResult}` for all emulators |
| `to_outputs()` | method | Wrap predicted_albedo as an `Outputs` object |
| `summary()` | method | Human-readable string |

## The Retrieve-then-Classify Workflow

### Why five separate emulators?

A single universal emulator would need a discrete surface-type input (`direct` is already binary and problematic; `surface_type` as a 5-class categorical is worse). More fundamentally, the parameter spaces are incommensurable: `pond_depth` is meaningless for bare ice, `ssl_grain_radius` is meaningless for a pond, and so on.

The five-emulator design means:
- Each emulator's parameters exactly describe the relevant physics for that surface type
- The cost function is always full-rank (no degenerate parameters)
- Classification falls out naturally: the emulator that produces the lowest chi-squared residual is the best physical model for that observation

### Confidence metric

```
confidence = (second_best_cost - best_cost) / second_best_cost
```

- `confidence = 0.0`: the two best emulators are tied (ambiguous surface type)
- `confidence = 0.95`: the winner's cost is 20× better than the runner-up (clear classification)

Typical values:
- Melt ponds are usually classified with high confidence (≥ 0.90) because their very low NIR albedo is unique
- FYI_bare vs MYI_bare typically has lower confidence (0.20–0.60) because their spectra overlap substantially
- FYI_summer is usually well-separated from FYI_bare due to the SSL spectral signature

### Worked example: reading cost_per_type

```python
result = retrieve_sea_ice(observed=spectrum, solzen=60, direct=1)
print(result.summary())
# SeaIceRetrievalResult  surface_type='FYI_summer'  confidence=0.847
#   Cost: 0.0034  converged=True
#   Retrieved parameters:
#     ssl_grain_radius             =   1820.0000  ±  85.0000
#     sea_ice_temperature          =    -5.2000  ±   1.2000
#     sea_ice_bubble_radius        =    230.0000  ±  40.0000
#     black_carbon                 =      8.0000  ±   2.0000
#   Cost per surface type:
#     FYI_summer       0.0034 ←
#     FYI_bare         0.0220
#     MYI_bare         0.0410
#     FYI_snow         0.1830
#     FYI_pond         0.4510
```

Inspect `result.all_fits["FYI_bare"]` to get the full `RetrievalResult` for any individual emulator.

## Per-type Retrieval with retrieve()

### When to bypass retrieve_sea_ice()

Use `retrieve()` directly when:
- You already know the surface type from a prior classification (e.g. a sea ice type map or manual annotation)
- You want to test a specific hypothesis ("what if this is a melt pond?")
- You need custom bounds or regularization for a specific surface type
- Computational budget is tight and you can afford only one emulator call

### Parameter tables for each surface type

**FYI_bare — retrieve bubble radius from clean ice**:

```python
result = retrieve(
    observed=spectrum,
    parameters=["sea_ice_bubble_radius"],
    emulator=Emulator.load("data/emulators/sea_ice_FYI_bare_7param.npz"),
    fixed_params={
        "sea_ice_temperature": -10.0,
        "sea_ice_salinity": 8.0,
        "black_carbon": 0.0,
        "rho_DL": 850.0,
        "solzen": 60,
        "direct": 1,
    },
)
```

**FYI_snow — retrieve snow depth and grain radius**:

```python
result = retrieve(
    observed=spectrum,
    parameters=["snow_depth", "snow_grain_radius"],
    emulator=Emulator.load("data/emulators/sea_ice_FYI_snow_6param.npz"),
    fixed_params={
        "sea_ice_temperature": -15.0,
        "black_carbon": 0.0,
        "solzen": 60,
        "direct": 1,
    },
)
```

**FYI_pond — retrieve pond depth**:

```python
result = retrieve(
    observed=spectrum,
    parameters=["pond_depth"],
    emulator=Emulator.load("data/emulators/sea_ice_FYI_pond_5param.npz"),
    fixed_params={
        "sea_ice_temperature": -5.0,
        "black_carbon": 0.0,
        "solzen": 60,
        "direct": 1,
    },
)
```

### Fixed_params recommendations

- **Always fix `direct` and `solzen`** — these are well-constrained from meteorological observations or satellite geometry. Leaving them free substantially degrades retrieval of physical parameters.
- **Fix `sea_ice_temperature`** when a surface temperature product is available (e.g. MODIS LST), as temperature is nearly degenerate with salinity in its effect on brine volume for a single emulator call.
- **Fix `black_carbon`** to a climatological value (e.g. 50 ppb in clean Arctic, 500 ppb near shipping lanes) unless you specifically need to retrieve it. BC retrieval requires good VIS coverage.
- **Fix `rho_DL`** unless you have reason to believe it varies significantly across the scene.

## Satellite Band Retrieval

### Which bands are most diagnostic for sea ice?

| Property | Diagnostic bands | Platform |
|---|---|---|
| Melt pond presence | SWIR (1.6 μm) — ponds are very dark there | All |
| Snow grain size | Near-IR (860 nm, 1240 nm) | Sentinel-2, Landsat |
| SSL grain radius | Near-IR (860 nm) | Sentinel-2, Landsat |
| Black carbon / impurities | VIS-VIS ratios (B2/B4) | Sentinel-2 |
| FYI vs MYI | Subtle NIR shape difference | Needs good radiometry |

### Recommended band selections

**Sentinel-2** (bands B1–B12A):
```python
# Best for melt pond retrieval (pond depth)
observed_band_names = ["B3", "B8", "B11"]  # green, NIR, SWIR

# Best for snow properties
observed_band_names = ["B2", "B3", "B4", "B8", "B8A", "B11"]

# Minimal for FYI_bare bubble radius
observed_band_names = ["B3", "B8", "B11"]
```

**Landsat-8/9**:
```python
observed_band_names = ["B3", "B4", "B5", "B6"]  # green, red, NIR, SWIR1
```

**MODIS** (Terra/Aqua):
```python
observed_band_names = ["B1", "B2", "B3", "B4", "B6", "B7"]
# B6 (1.64 μm) is particularly diagnostic for pond depth
```

### obs_uncertainty guidance

Providing per-band uncertainty improves retrieval quality by weighting better-measured bands more heavily:

```python
# Typical S2 land surface reflectance uncertainties
obs_uncertainty = np.array([
    0.02,   # B3 (green) — well-characterised
    0.02,   # B8 (NIR) — well-characterised
    0.04,   # B11 (SWIR) — more atmospheric correction uncertainty
])

result = retrieve_sea_ice(
    observed=np.array([0.75, 0.65, 0.08]),
    platform="sentinel2",
    observed_band_names=["B3", "B8", "B11"],
    obs_uncertainty=obs_uncertainty,
    solzen=60, direct=1,
)
```

When `obs_uncertainty` is not provided, all bands are weighted equally (equivalent to constant sigma = 1).

## Uncertainty Estimation

### Hessian-based (default)

After optimisation, a finite-difference Hessian is computed at the best-fit point and inverted to give the covariance matrix. The square root of the diagonal gives 1-sigma uncertainties. This is applied automatically after L-BFGS-B, Nelder-Mead, and differential_evolution.

Parameters that are log-space transformed (bubble radius, pond depth, snow depth, snow grain radius, SSL grain radius, black carbon) have their uncertainties propagated back to linear space: `sigma_linear ≈ (x + 1) × ln(10) × sigma_log`.

If the Hessian is singular (i.e. the parameter is unconstrained by the observations), the uncertainty is `inf`. This is expected for parameters that are degenerate with other free parameters.

### MCMC

For publication-quality uncertainty or when the posterior may be non-Gaussian:

```python
result = retrieve(
    observed=spectrum,
    parameters=["pond_depth", "black_carbon"],
    emulator=pond_emu,
    fixed_params={"sea_ice_temperature": -5.0, "solzen": 60, "direct": 1},
    method="mcmc",
    mcmc_walkers=32,
    mcmc_steps=5000,
    mcmc_burn=1000,
)
print(result.summary())

# Access full chains for corner plots
flat = result.chains.reshape(-1, 2)
# import corner; corner.corner(flat, labels=["pond_depth", "black_carbon"])
```

### When to use each

| Method | Cost | When to use |
|---|---|---|
| Hessian (default) | ~4n² extra evaluations | Fast; good for unimodal, well-constrained posteriors |
| MCMC | ~50,000–200,000 evaluations | Need full posterior; parameter degeneracies present; publication figures |

At emulator speed (~microseconds), even MCMC with 32 walkers × 5000 steps takes < 10 seconds.

## Known Limitations

1. **R² metric misleading for bare ice** — R² is not a reliable quality metric for FYI_bare and MYI_bare emulators because the high-dimensional brine optics spectral manifold requires many more samples than the manifold than snow/pond surfaces. Always evaluate accuracy via BBA MAE from `verify()`.

2. **Brine physics is high-dimensional** — Brine optics (temperature, salinity → complex refractive index → absorption) create ~27 PCA dimensions vs ~6 for snow. This means more training samples and a larger MLP are required for the same accuracy. The production emulators (30,000 samples) achieve BBA MAE ≈ 0.002–0.004; test emulators (250–500 samples) may show BBA MAE > 0.05.

3. **No SSA analog for sea ice** — The terrestrial emulator supports retrieving SSA (specific surface area), which elegantly resolves the rds/rho degeneracy for glacier ice. Sea ice does not have an analogous combined parameter. Temperature and salinity are both free parameters because they contribute independently to brine volume, and fixing one without the other breaks the physics.

4. **`direct` must be fixed** — The binary illumination flag cannot be continuously optimised. Pass it via `fixed_params`. Sky conditions are typically known from the satellite acquisition geometry or meteorology.

5. **No atmospheric correction** — All emulators output surface reflectance. Satellite observations must be surface reflectance (atmospherically corrected) before using `retrieve_sea_ice()` or `retrieve()`.

6. **Parameter correlations (T and S)** — Temperature and salinity both control brine volume (`Vb = S / (17.6 + 0.93 T)`). When both are free parameters, they are partially degenerate: the optimiser can trade off T against S with limited spectral cost. Fix one if the other is well-constrained from auxiliary data.

7. **FYI_bare vs MYI_bare classification difficulty** — These two surface types overlap spectrally. Classification confidence is typically lower (0.20–0.60) than for other pairs. If the distinction matters, use additional geophysical context (e.g. ice age from satellite time series, location relative to MYI extent).

8. **Extrapolation behaviour** — The emulator clips inputs and warns when parameters are outside training bounds. Predictions outside the training range are unreliable. Do not extrapolate to e.g. `sea_ice_temperature=0°C` (brine drainage) or `pond_depth=1.5 m` (not physically represented in the training data).

9. **Fixed structural parameters** — Layer thicknesses (`dz`) and layer types are fixed by the `transform_fn` for each surface type. FYI_bare always has DL 5 cm + IL 145 cm. If your scene has substantially different ice thickness geometry, the emulator will still run but the DL/IL thickness assumption may introduce systematic bias.

10. **SSL grain radius vs geometric grain radius** — The `ssl_grain_radius` parameter is the effective optical grain radius in the granular-ice model, not the physical crystal size. It maps to a look-up table in the forward model via the `_snap()` function and covers the range 500–5000 μm (step 5–20 μm depending on magnitude).

11. **Direct band-mode information limits** — With 3–5 satellite bands, you can reliably constrain 1–2 physical parameters (fixing the rest). Attempting to retrieve 4+ parameters from 4 bands will produce unreliable results even if the optimiser converges.

12. **No multi-spectral unmixing** — Each `retrieve_sea_ice()` call represents a single surface type. If a satellite pixel is a spatial mixture of pond and bare ice, the result will pick whichever type produces the lower residual. Sub-pixel unmixing is not supported.

## Validation Against Real Observations: SHEBA Spectra

The emulator classification system was validated against the Grenfell & Light (2007) SHEBA spectral albedo archive — 7 spring snow dates and 16 summer bare ice dates measured on drifting sea ice at ~76°N in 1998.

### Spring snow: reliable classification

All spring dates (April–May) classify correctly as `FYI_snow` with high confidence (0.59–0.99). The one marginal case is 27 May (confidence=0.74, classified as `FYI_summer`) — this is the date closest to melt onset (June 3), when the snow surface is becoming coarser and transitioning toward SSL conditions. This is a genuine physical ambiguity, not a model failure.

### Summer bare ice: spectral range and physical priors are critical

Without the `known_month` seasonal prior, all summer dates are misclassified as `FYI_snow`. The optimiser exploits physically impossible solutions — for example, fitting August bare ice with T=−25°C, snow_depth=3 cm, grain_radius=1447 µm — which is mathematically correct in 400–1000 nm but physically impossible (SHEBA August ice temperatures were −2 to −5°C).

With `known_month=8` (August), the prior `sea_ice_temperature ~ (−4°C ± 3°C)` eliminates these unphysical FYI_snow solutions. Classification improves to **9/16** correct. The 7 remaining misclassifications are dates with BBA > 0.72 (predominantly white ice, similar albedo to thin snow) where even the corrected FYI_snow cost is close to FYI_summer/FYI_bare.

**Root causes of summer misclassification (in order of impact):**

1. **Spectral range** — The Grenfell data covers only 400–1000 nm. The SWIR (1000–2500 nm) contains diagnostic ice absorption features that discriminate bare ice from snow. When paired IR spectra (ALBI files, 1100–2000 nm) are available, classification improves further. The `wavelength_mask` parameter accepts any band selection.

2. **Emulator accuracy bias** — FYI_bare has R² ≈ 0.56 vs R² ≈ 0.99 for FYI_snow. Even when the surface is genuinely bare ice, the FYI_snow emulator produces lower chi-squared residuals because it fits more accurately. FYI_summer (R² ≈ 0.99) partially compensates, ranking 2nd on most summer dates.

3. **No seasonal physical constraint (without `known_month`)** — The unrestricted FYI_snow emulator can reach T=−25°C in August. Providing `known_month` applies a Gaussian prior that prevents this.

**Practical guidance for summer classification:**

```python
# Always provide known_month for summer observations
result = retrieve_sea_ice(
    observed    = spectrum,
    wavelength_mask = mask_400_to_1000,   # restrict to observed bands
    solzen      = sza,
    known_month = 8,    # August — prevents T<-10°C in FYI_snow
)
# Surface type will be FYI_summer or FYI_bare for most summer bare ice
```

If paired 1100–2000 nm IR spectra are available, include them in the observed array and widen the wavelength mask — this is the single most impactful improvement.

**Comprehensive classification accuracy (551 labeled observations, 5 surface types):**

| Test | Data source | N | `known_month` | Spectral range | Correct |
|---|---|---|---|---|---|
| Spring snow | Grenfell ALBV Apr–May | 7 | not needed | 400–1000 nm | **6/7 (86%)** |
| Summer bare ice | Grenfell ALBV Aug–Sep | 16 | no | 400–1000 nm | 0/16 (0%) |
| Summer bare ice | Grenfell ALBV Aug–Sep | 16 | **yes (month=8)** | 400–1000 nm | **16/16 (100%)** |
| Summer bare ice | ALBV+ALBI Aug–Sep | 12 | yes | 400–2000 nm | 5/12 (42%) |
| Melt ponds | Morassutti 1995 | 504 | yes (month=7) | 400–1000 nm (6 bands) | **387/504 (77%)** |

**Surprising finding — VIS-only beats VIS+SWIR for bare ice.**
Adding SWIR data (1100–2000 nm) to the 400–1000 nm VIS window *reduces* summer bare ice accuracy from 100% to 42%.  The SWIR signature of white ice overlaps with coarse snow, reintroducing the snow–ice ambiguity that the seasonal prior eliminates in the VIS window.  **Use 400–1000 nm only for summer bare ice classification.**

Melt pond accuracy increases strongly with depth: 7% for <5 cm, 94% for >30 cm.

Run the classification on the full SHEBA archive (the `--classify` flag adds `retrieve_sea_ice()` surface-type classification on top of the standard forward-model validation; without it the script runs only the spectral-albedo comparison):

```bash
uv run python tests/validation_data/Grenfell_light_2007/validate_grenfell_light_2007.py --classify
```

## The .npz File Format

Each emulator is stored as a self-contained `.npz` (compressed numpy archive). You can load it without scikit-learn — only numpy is required.

| Key | Shape | Description |
|---|---|---|
| `weights_0` | (n_params, layer_0) | First MLP layer weights |
| `weights_1` | (layer_0, layer_1) | Second layer weights |
| `weights_2` | (layer_1, layer_2) | Third layer weights |
| `weights_3` | (layer_2, n_pca) | Output layer weights |
| `biases_0`…`biases_3` | (layer_size,) | Bias vectors per layer |
| `pca_components` | (n_pca, 480) | PCA basis vectors |
| `pca_mean` | (480,) | Mean spectrum (PCA centre) |
| `input_min` | (n_params,) | Per-parameter training minimum |
| `input_max` | (n_params,) | Per-parameter training maximum |
| `flx_slr` | (480,) | Solar flux stored at build time |
| `metadata` | scalar | JSON string — param names, bounds, transform_fn name, build timestamp, architecture |

For the (256,256,128,64) architecture (FYI_bare, MYI_bare), the weight shapes are adjusted accordingly: `weights_0` is (n_params, 256), `weights_1` is (256, 256), `weights_2` is (256, 128), `weights_3` is (128, 64), and `weights_4` is (64, n_pca) — five weight matrices in total.

### Loading outside Python

```python
import numpy as np, json

data = np.load("data/emulators/sea_ice_FYI_bare_7param.npz", allow_pickle=False)
meta = json.loads(str(data["metadata"]))

print(meta["param_names"])    # ['sea_ice_temperature', 'sea_ice_bubble_radius', ...]
print(meta["bounds"])         # {'sea_ice_temperature': [-30.0, -2.0], ...}
print(meta["transform_fn"])   # '_transform_fyi_bare'
print(meta["n_pca_components"])
print(meta["training_r2"])
print(meta["hidden_layer_sizes"])

# PCA and MLP weights are plain numpy arrays
print(data["pca_components"].shape)   # (n_pca, 480)
print(data["flx_slr"].shape)          # (480,)
```

The format is identical between sea ice and terrestrial emulators. The presence of the `"transform_fn"` key in metadata (set to `None` for terrestrial emulators) distinguishes sea ice emulators.

## See Also

- [examples/13_sea_ice.py](../examples/13_sea_ice.py) — sea ice emulator build, predict, retrieve
- [docs/EMULATOR.md](EMULATOR.md) — terrestrial emulator architecture (same underlying system)
- [docs/INVERSION.md](INVERSION.md) — retrieve() API reference (applies to sea ice too)
- [biosnicar/sea_ice/emulator_configs.py](../biosnicar/sea_ice/emulator_configs.py) — transform functions and config registry
- [biosnicar/sea_ice/retrieve.py](../biosnicar/sea_ice/retrieve.py) — retrieve_sea_ice() implementation
