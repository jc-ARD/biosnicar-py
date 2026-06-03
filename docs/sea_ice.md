# Sea Ice Extension — Primer

**Version**: v0.1 (MVP)
**Module**: `biosnicar.sea_ice`

---

## How sea ice relates to the other layer types

BioSNICAR uses a `layer_type` integer per layer to select the optical-property model. The full set is:

| `layer_type` | Physical model | Fresnel surface |
|---|---|---|
| 0 | Granular snow or ice — discrete grains in air | No |
| 1 | Solid bubbly glacier ice + Fresnel air-ice surface | **Yes** |
| 2 | Solid bubbly glacier ice, no Fresnel correction | No |
| 3 | Granular water/ice sphere mixture (slush) | No |
| **4** | **Sea ice — brine inclusions via Maxwell-Garnett** | **Yes** |

Types 1 and 2 are the same optical model (bulk ice matrix, absorption from `k_ice`, scattering from air/water bubble LUTs) — the only difference is that type 1 triggers the Fresnel surface reflection in the solver and type 2 does not. Type 3 is a different physical picture altogether: discrete spheres of ice and water in air, combined by linear volume-fraction mixing, with no effective medium theory.

Sea ice (`layer_type=4`) is closest to type 1 in structure — solid bulk ice with inclusions and a Fresnel surface — but replaces the pure-ice absorption with an effective medium calculation that accounts for brine pockets. It adds three required per-layer inputs (`sea_ice_temperature`, `sea_ice_salinity`, `sea_ice_bubble_radius`) and draws from a separate pre-computed LUT. See [docs/METHODS.md](METHODS.md) for the full layer-type reference.

---

## What this extension does and doesn't do

### Does
- Computes spectral albedo (480 bands, 0.205–4.995 µm) for sea-ice columns with optional snow cover.
- Models brine inclusions via the liquidus brine salinity and Maxwell-Garnett effective medium.
- Models melt ponds as liquid water layers (layer_type=5) with realistic water absorption.
- Accepts per-layer physical inputs: thickness, temperature, salinity, density, bubble radius, pond depth.
- Provides five pre-configured presets: three winter sea-ice types plus two melt pond depths.
- Works with the existing BioSNICAR `run_model()` API — no new classes required.
- Fully backward-compatible: all existing terrestrial-ice functionality unchanged.

### Doesn't (deferred to later versions)
- **Sea-ice algae** (v0.3): algal blooms in sea ice are not in the impurity database.
- **Salty snow** (v0.3): snow on sea ice is treated as fresh.
- **Pond bottom darkening** (v0.3): melt pond model assumes a white-ice bottom; real ponds are darkened by algae, sediment, and dissolved organic matter.
- **Vertical T/S profiles** (v0.3): each layer uses a single T and S value.
- **Antarctic-specific tuning** (v0.3): validation uses Arctic data only.
- **Inverse retrieval** (v1.0): parameter retrieval from observations is not yet implemented.

---

## Physics in brief

### Why sea ice is different from glacier ice

Sea ice forms from seawater and retains dissolved salt as **brine inclusions**: pockets of liquid concentrated seawater that remain liquid even at temperatures well below 0 °C. These inclusions affect the ice's complex refractive index and therefore its spectral albedo. First-year ice (FYI) typically contains 4–12 psu of bulk salinity, while multiyear ice (MYI) has been desalinated by summer brine drainage and contains only 1–3 psu.

### Cox & Weeks (1983) brine volume

The fraction of the ice that is liquid brine (**brine volume fraction ν_b**) depends on bulk salinity S (psu) and temperature T (°C):

```
For −22.9 ≤ T ≤ −2 °C:
  F1(T) = −4.732 − 22.45T − 0.6397T² − 0.01074T³
  F2(T) = 0.08903 − 0.01763T − 5.330×10⁻⁴T² − 8.801×10⁻⁶T³
  ρ_i   = 0.917 − 1.403×10⁻⁴T    (pure ice density, g/cm³)
  ν_b   = (S × ρ_i) / (F1 − ρ_i × S × F2)
```

At T = −10 °C and S = 8 psu, ν_b ≈ 0.045 (4.5% of the ice is liquid brine). At T = −2 °C, ν_b can reach 25% or more.

### Brine refractive index — liquidus constraint

A critical physical insight: brine inside sea ice is not at the bulk ice salinity. It is at the **liquidus (phase-equilibrium) salinity** determined by temperature alone:

```
S_brine ≈ −18.7 × T_C    (psu)    [linear liquidus approximation]
```

At T = −10 °C this gives S_brine ≈ 187 psu — roughly 23× the typical bulk ice salinity of 8 psu. The v0.1 implementation mistakenly used the bulk salinity (8 psu) in the RI formula, making the brine-ice optical contrast 23–47× too small. This has been corrected in v0.2.

**Real part** — Quan & Fry (1995) full wavelength-dependent formula applied at the liquidus salinity S_brine, capped at 250 psu (near the NaCl eutectic at ~−21 °C):

```
n(S_b, T, λ) = n₀ + (n₁ + n₂T + n₃T²)·S_b + n₄T²
             + (n₅ + n₆·S_b + n₇·T)/λ + n₈/λ² + n₉/λ³
```

where λ is in nm. Formula covers 400–700 nm; salt contribution fixed at the 700-nm value in NIR (weakly wavelength-dependent beyond the visible). At T = −10 °C this gives Δn_re ≈ +0.037 vs the v0.1 value of +0.002 — a 23× increase.

**Imaginary part** — NaCl has **no absorption above 400 nm**. The v0.1 multiplicative correction was physically wrong and has been removed. The corrected implementation adds only:
1. A small UV ionic contribution (Cl⁻ electronic band, decaying to essentially zero by 0.4 µm)
2. A temperature correction in the visible from Pegau et al. (1997)

In NIR (> 700 nm), k_brine ≈ k_water — the dominant O-H overtone absorption bands of water are nearly independent of dissolved salt concentration.

**Net effect**: the corrected brine RI has substantially larger real-part contrast with pure ice (driving slightly more absorption in the effective medium), but no spurious visible absorption. FYI BBA changes from ~0.56 (v0.1) to ~0.51 (v0.2), which is more physically consistent with SHEBA observations.

### Maxwell-Garnett effective medium

The combined ice+brine medium is modelled as pure ice (host) containing spherical brine inclusions at volume fraction ν_b. The effective complex permittivity is:

```
ε_eff = ε_ice × [ε_brine(1 + 2ν_b) + 2ε_ice(1 − ν_b)] /
                [ε_brine(1 − ν_b) + ε_ice(2 + ν_b)]
```

The effective refractive index n_eff + ik_eff is obtained via √ε_eff. The increased imaginary part (absorption) from the brine inclusions directly reduces the single-scattering albedo relative to pure ice.

*Known limitation*: Maxwell-Garnett is most accurate for dilute inclusions (ν_b < 0.3). For sea ice near the melting point (ν_b > 0.3), the Bruggeman symmetric mixing formula would be more appropriate — this is deferred to a future version.

### Air bubble scattering

Sea ice also contains air bubbles whose scattering dominates the NIR optical depth. The air volume fraction is inferred from the density balance:

```
ν_air = (917 − ρ_si + (ρ_brine − 917) × ν_b) / 917
```

Scattering coefficients are looked up from the existing `bubbly_air.npz` LUT (pre-computed Mie theory for air bubbles in pure ice). *MVP approximation*: the effective-medium correction to bubble scattering (host RI changes slightly with brine content) is < 5% and is neglected.

---

## Known approximations and their consequences

### Sea ice (layer_type=4)

| Approximation | Effect | When it matters |
|---|---|---|
| Liquidus linear approximation S_b = −18.7T | ±5% error in S_brine at T < −15 °C (near eutectic) | Very cold ice |
| Q&F (1995) real-part formula extrapolated above 40 psu | ~10% error in Δn_re at S_b > 100 psu | T < −5 °C |
| Liquidus salinity capped at 250 psu | Underestimates real-part correction below −13 °C | Very cold ice |
| Air bubble scattering uses pure-ice LUT | < 5% error in scattering | All conditions |
| Maxwell-Garnett for spherical inclusions only | < 5% for ν_b < 0.15 | Near-melting ice (T > −3 °C) |
| Snow layer treated as fresh water | Overestimates snow albedo for salty snow | Snow on FYI |
| Single T and S per layer | Ignores vertical gradients | Thick layers |

### Melt ponds (layer_type=5)

| Approximation | Effect | When it matters |
|---|---|---|
| Pure liquid water at 0 °C | Neglects temperature effect on water k (small) | All conditions |
| Clear, particle-free water | Overestimates visible BBA by ~0.2–0.4 | All ponds |
| White sea-ice pond bottom | Overestimates visible BBA | Most real ponds have dark bottoms from algae, sediment, or DOM |
| No Fresnel correction at air-water surface | ~2% underestimate of surface reflection | Small effect |

The melt pond model correctly predicts NIR albedo (dominated by water absorption, independent of bottom properties). Visible BBA is overestimated because real pond bottoms are darkened by biological and sedimentary material not yet in the model. The NIR is the most diagnostic band for pond comparison.

---

## YAML configuration

Add `LAYER_TYPE: 4` to enable sea ice. The required additional fields are:
`SEA_ICE_TEMPERATURE`, `SEA_ICE_SALINITY`, and `SEA_ICE_BUBBLE_RADIUS` (all per-layer lists).

```yaml
ICE:
  DZ: [0.05, 1.45]          # layer thicknesses (m)
  LAYER_TYPE: [4, 4]        # 4 = sea ice
  RHO: [920, 915]           # bulk density (kg/m³)
  SEA_ICE_TEMPERATURE: [-25, -10]   # temperature (°C)
  SEA_ICE_SALINITY:    [12, 8]      # bulk salinity (psu)
  SEA_ICE_BUBBLE_RADIUS: [100, 200] # air bubble radius (µm)
  RF: 2                     # Picard 2016 refractive index (recommended)
  # Other fields (RDS, SHP, etc.) are required by the parser but unused for layer_type=4
  RDS: [500, 500]
  SHP: [0, 0]
  CDOM: [0, 0]
  WATER_COATING: [0, 0]
  LWC: [0, 0]
  LWC_PCT_BBL: 0
  HEX_SIDE: [10000, 10000]
  HEX_LENGTH: [10000, 10000]
  SHP_FCTR: [0, 0]
  AR: [0, 0]
```

*Note*: Use the adding-doubling solver (the default) — sea ice requires Fresnel interfaces at the air-ice boundary, which the Toon solver does not handle correctly.

---

## Python API

Sea ice uses the same `run_model()` entry point as terrestrial ice.
Both return an `Outputs` object with identical attribute names.

### Quick start with a preset

```python
from biosnicar import run_model, FYI_WINTER_BARE

# By preset dict
outputs = run_model(preset=FYI_WINTER_BARE, solzen=60)

# Or by preset name
outputs = run_model(preset="FYI_WINTER_BARE", solzen=60)

print(f"Broadband albedo:  {outputs.BBA:.3f}")       # or outputs.broadband
print(f"Visible albedo:    {outputs.BBAVIS:.3f}")     # or outputs.visible
print(f"NIR albedo:        {outputs.BBANIR:.3f}")     # or outputs.nir

import matplotlib.pyplot as plt
plt.plot(outputs.wavelengths, outputs.albedo)         # or outputs.spectrum
plt.xlabel("Wavelength (µm)")
plt.ylabel("Spectral albedo")
plt.title("FYI Winter Bare — SZA 60°")
plt.xlim(0.3, 2.5)
plt.ylim(0, 1)
plt.show()
```

### Custom column — flat kwargs

Sea ice uses the same flat keyword-argument style as terrestrial ice.
Simply add `layer_type=4` and three sea-ice-specific lists:

```python
from biosnicar import run_model

# Snow on sea ice — identical call style to snow on glacier ice
outputs = run_model(
    solzen=60, direct=1, incoming=2,
    layer_type=[0, 4, 4],              # snow + two sea ice layers
    dz=[0.10, 0.05, 1.50],
    rds=[200, 500, 500],               # snow grain; 500 is unused dummy for sea ice
    rho=[300, 895, 895],
    sea_ice_salinity=[None, 10, 6],    # None for snow layer
    sea_ice_temperature=[None, -20, -8],
    sea_ice_bubble_radius=[None, 100, 200],
    black_carbon=500,                  # impurities on any/all layers
)
print(outputs.BBA)
```

The illumination parameters match `run_model()` exactly:
- `solzen` — solar zenith angle in degrees (1–89)
- `direct` — 1 = direct beam (clear), 0 = diffuse (cloudy/overcast)
- `incoming` — irradiance spectrum index (0–6): 0=mid-lat winter, 1=mid-lat summer, 2=sub-Arctic winter, 3=sub-Arctic summer, 4=summit, 5=high mountain, 6=tropical

### `Outputs` attributes (same as terrestrial ice)

| Attribute | Alias | Description |
|---|---|---|
| `.BBA` | `.broadband` | Flux-weighted broadband albedo |
| `.BBAVIS` | `.visible` | Flux-weighted visible (0.205–0.75 µm) albedo |
| `.BBANIR` | `.nir` | Flux-weighted NIR (0.75–4.995 µm) albedo |
| `.albedo` | `.spectrum` | Spectral albedo array, shape (480,) |
| `.wavelengths` | | Wavelength grid in µm, shape (480,) |
| `.flx_slr` | | Solar flux spectrum, shape (480,) |
| `.to_platform(name)` | | Satellite/GCM band convolution |
| `.plot()` | | Spectral albedo plot |

---

## The three presets

Presets are plain dicts of `run_model()` kwargs. Use them as a starting
point and layer overrides on top:

```python
from biosnicar import run_model, FYI_WINTER_BARE, FYI_WINTER_SNOW, MYI_WINTER_BARE

# Merge preset with additional overrides
outputs = run_model(preset=FYI_WINTER_SNOW, solzen=75, black_carbon=2000)
```

### `FYI_WINTER_BARE`
Bare first-year ice in winter. Two sea-ice layers (T=−25/−20°C, S=12/8 psu, ρ=895 kg/m³). Typical BBA: 0.50–0.60.

### `FYI_WINTER_SNOW`
First-year ice with 15 cm of snow (ρ=250 kg/m³, grain=200 µm). Calibrated against SHEBA April 1998 data (RMSE ≈ 0.025–0.038, 7/7 pass). Typical BBA: 0.75–0.85.

### `MYI_WINTER_BARE`
Bare multiyear ice in winter. Two layers (T=−20/−8°C, S=1/3 psu, ρ=860/870 kg/m³, large bubbles 500/700 µm). Typical BBA: 0.49–0.52.

### `FYI_POND_SHALLOW`
Shallow melt pond (10 cm) on summer FYI. Typical BBA: 0.40–0.45, NIR < 0.15.

### `FYI_POND_DEEP`
Deep melt pond (40 cm) on summer FYI. Typical BBA: 0.28–0.32, NIR < 0.02.

---

## Melt ponds (layer_type=5)

Melt ponds form when surface snow and ice melt in summer and the meltwater collects in topographic depressions. They can reduce surface albedo dramatically — a 20 cm deep pond has BBA ≈ 0.35, vs ~0.78 for snow-covered FYI.

### Physics

A melt pond layer is modelled as **liquid water with near-zero scattering**:

```
absorption coefficient:  α(λ) = 4π k_water(λ) / λ    [m⁻¹]
scattering coefficient:  β(λ) ≈ 0.003 × (0.55/λ_µm)⁴ [m⁻¹]  (Rayleigh, for stability)
single-scattering albedo: ω ≈ β / (α + β) → 0 in NIR, ~0.01 in visible
```

Water optical constants k(λ) come from Rowe et al. (2020) at 0 °C — appropriate for melt pond water (near 0 °C in summer). Key properties:
- **Visible (400–700 nm)**: water is nearly transparent. Most light passes through and reflects from the ice below. Pond VIS ≈ ice_VIS × exp(−2α × depth).
- **NIR (700–1000 nm)**: strong O-H absorption. A 5 cm pond reduces NIR by ~50%; a 20 cm pond reduces it by ~90%.
- **SWIR (> 1000 nm)**: essentially opaque even at < 1 cm depth.

### Usage

```python
from biosnicar import run_model

# Melt pond via presets
outputs = run_model(preset="FYI_POND_SHALLOW", solzen=60)
outputs = run_model(preset="FYI_POND_DEEP",    solzen=60)

# Custom pond depth — same flat-kwargs style as all other layer types
outputs = run_model(
    solzen=60,
    layer_type=[5, 4, 4],         # pond on top of FYI
    dz=[0.15, 0.05, 1.45],        # 15 cm pond
    rds=[500, 500, 500],
    rho=[1000, 895, 895],          # 1000 kg/m³ for liquid water
    sea_ice_salinity=[None, 12, 8],
    sea_ice_temperature=[None, -5, -5],   # summer FYI conditions
    sea_ice_bubble_radius=[None, 100, 200],
)
print(f"BBA={outputs.BBA:.3f}  NIR={outputs.BBANIR:.3f}")
```

### Expected albedo vs pond depth

| Depth | BBA | VIS (400–700 nm) | NIR (700–1000 nm) |
|---|---|---|---|
| 0 (bare FYI) | ~0.51 | ~0.73 | ~0.27 |
| 5 cm | ~0.45 | ~0.72 | ~0.16 |
| 10 cm | ~0.41 | ~0.70 | ~0.10 |
| 20 cm | ~0.36 | ~0.66 | ~0.05 |
| 40 cm | ~0.30 | ~0.58 | ~0.02 |

*Model conditions: SZA=60°, summer FYI below pond. Assumes clear water and white-ice bottom.*

### Validation against Morassutti (1995) — NIR

NIR comparisons against Morassutti (1995) Canadian Arctic melt pond data (doi:10.7265/N55Q4T1C):

| Depth bin | Observed NIR | Model NIR | Notes |
|---|---|---|---|
| 5–10 cm | ~0.10 | ~0.10 | Good agreement |
| 10–20 cm | ~0.06 | ~0.05 | Good agreement |
| 20–30 cm | ~0.03 | ~0.02 | Good agreement |
| 0–5 cm (VIS) | ~0.51 | ~0.72 | Model too bright: dark pond bottoms not modelled |

NIR is the most reliable diagnostic. VIS overestimation is a known and documented limitation of the clear-water, white-ice-bottom assumption. See `tests/validation_data/morassutti1995/` for the full validation script.

---

## Validation summary

### Sea ice (winter, snow-covered and bare)

Validated against Grenfell & Light (2007) SHEBA spectral albedo, doi:10.5065/D6765CQ1.
Spring snow (April–May 1998, ~76°N). See `docs/sea_ice_validation.md` for full results.

| Condition | Observed BBA (400–1000 nm) | Model BBA | Spectral RMSE | Pass? |
|---|---|---|---|---|
| FYI snow-covered (Apr) | 0.912–0.935 | 0.934–0.941 | 0.023–0.031 | ✓ 7/7 |
| FYI bare (Aug–Sep, summer ice) | 0.618–0.809 | ~0.56 | ~0.19 | n/a — season mismatch |
| MYI bare (inferred) | 0.50–0.65 | ~0.49 | — | Qualitative |

### Melt ponds

Validated against Morassutti (1995) Canadian Arctic melt pond data (NSIDC G01169, doi:10.7265/N55Q4T1C). 504 records, summer 1994, Barrow Strait, Nunavut, 6 bands (400–1000 nm). See `tests/validation_data/morassutti1995/validate_morassutti1995.py`.

| Depth bin | Observed NIR | Model NIR | Observed BBA | Model BBA | Notes |
|---|---|---|---|---|---|
| 5–10 cm | 0.095 | ~0.10 | 0.229 | ~0.41 | NIR ✓, VIS high (dark bottoms) |
| 10–20 cm | 0.058 | ~0.05 | 0.205 | ~0.36 | NIR ✓, VIS high |
| 20–30 cm | 0.032 | ~0.02 | 0.182 | ~0.33 | NIR ✓, VIS high |

NIR validates well. BBA overestimated by ~0.15–0.18 because the model assumes clear water with white sea-ice bottom; real ponds have dark bottoms from algae and sediment (see *Known approximations*).

---

## References

- Cox, G. F. N. & Weeks, W. F. (1983). Equations for determining the gas and brine volumes in sea-ice samples. *J. Glaciology*, 29(102), 306–316.
- Grenfell, T. C. & Light, B. (2007). SHEBA Spectral Albedo. UCAR/NCAR EOL. doi:10.5065/D6765CQ1
- Light, B. et al. (2004). Two-dimensional Monte Carlo model of radiative transfer in sea ice. *J. Geophys. Res.*, 109, C03028.
- Maxwell Garnett, J. C. (1904). Colours in metal glasses and in metallic films. *Phil. Trans. R. Soc. Lond. A*, 203, 385–420.
- Morassutti, M. (1995). Sea Ice Melt Pond Data from the Canadian Arctic. NSIDC G01169. doi:10.7265/N55Q4T1C
- Pegau, W. S., Gray, D. & Zaneveld, J. R. V. (1997). Absorption and attenuation of visible and near-infrared light in water. *Limnol. Oceanogr.*, 42(3), 443–452.
- Perovich, D. K. et al. (2002). Seasonal evolution of the albedo of multiyear Arctic sea ice. *J. Geophys. Res.*, 107(C10), 8044.
- Picard, G. et al. (2016). Refinement of the ice absorption spectrum. *J. Glaciology*.
- Quan, X. & Fry, E. S. (1995). Empirical equation for the refractive index of seawater. *Appl. Optics*, 34, 3477.
- Rowe, P. M. et al. (2020). Refractive index of liquid water at 0 °C. *J. Geophys. Res.*, 125, e2019JD031822.
- Sihvola, A. (1999). *Electromagnetic Mixing Formulas and Applications*. IEE.
- Timco, G. W. & Frederking, R. M. W. (1996). A review of sea ice density. *Cold Reg. Sci. Tech.*, 24(1), 1–6.

---

## Cross-references to other BioSNICAR documentation

- [METHODS.md](METHODS.md) — emulator architecture, inversion framework, band convolution.
- [SUBSURFACE.md](SUBSURFACE.md) — subsurface light field calculation.
- [BANDS.md](BANDS.md) — satellite band convolution (Sentinel-2, MODIS, etc.).
- [PLOTTING.md](PLOTTING.md) — spectral plotting utilities.
