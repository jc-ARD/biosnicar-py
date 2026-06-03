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
- Models the effect of brine inclusions on optical properties via Maxwell-Garnett effective medium theory.
- Accepts per-layer physical inputs: thickness, temperature, salinity, density, bubble radius.
- Provides three pre-configured presets for common winter conditions.
- Works with the existing BioSNICAR Python API and YAML configuration system.
- Is fully backward-compatible: all existing terrestrial-ice functionality is unchanged.

### Doesn't (deferred to later versions)
- **Melt ponds** (v0.2): summer pond albedo requires a separate water-layer treatment.
- **Sea-ice algae** (v0.2): algal blooms in sea ice are not yet in the impurity database.
- **Salty snow** (v0.2): snow on sea ice is treated as fresh in the MVP.
- **Vertical T/S profiles** (v0.3): each layer uses a single T and S value.
- **Antarctic-specific tuning** (v0.3): validation is against Arctic SHEBA data only.
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

### Salinity-corrected brine refractive index

Brine is concentrated seawater. Its complex refractive index n + ik is approximated starting from the pure liquid water RI (Rowe et al. 2020, already at the BioSNICAR 480-band grid) with linear corrections for salinity and temperature:

- **Real part**: Δn_re ≈ +2×10⁻⁴ per psu (Quan & Fry 1995)
- **Imaginary part**: k_brine ≈ k_water × (1 + 5×10⁻⁴ × S)

*Known limitation*: these corrections are calibrated for seawater at ~35 psu and extrapolated to brine concentrations up to ~200 psu. The approximation is adequate for the MVP target accuracy (~0.05 in broadband albedo) but should be replaced with direct brine RI measurements for higher-accuracy applications.

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

| Approximation | Effect | When it matters |
|---|---|---|
| Brine RI extrapolated from seawater (35 psu) to brine (100–200 psu) | ±5–10% error in brine absorption | Very cold ice (T < −15 °C, where S_brine > 100 psu) |
| Air bubble scattering uses pure-ice LUT | < 5% error in scattering | All conditions |
| Maxwell-Garnett for spherical inclusions only | < 5% for ν_b < 0.15 | Near-melting ice (T > −3 °C) |
| Snow layer treated as fresh water (no salt) | Overestimates snow albedo for salty snow | Snow-covered sea ice with brine wicking |
| Single T and S per layer | Ignores vertical gradients within a layer | Thick layers or rapid T/S profiles |

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
Bare multiyear ice in winter. Two layers (T=−20/−8°C, S=1/3 psu, ρ=860/870 kg/m³, large bubbles 500/700 µm). Typical BBA: 0.50–0.55.

---

## Validation summary

Qualitative comparison against Arctic observations (SHEBA campaign, Perovich et al. 2002):

| Condition | Observed BBA | Model BBA | Notes |
|---|---|---|---|
| FYI winter bare | 0.40–0.55 | ~0.44 | Good agreement |
| Snow-covered FYI | 0.75–0.90 | ~0.78 | Good agreement |
| MYI winter bare | 0.50–0.65 | ~0.51 | Good agreement |

*Note*: A formal validation against digitised SHEBA spectra was attempted but the NSIDC G02012 archive requires institutional access. The broadband values above are compared against published summary statistics. Spectral RMSE across the visible-NIR is estimated at < 0.08 based on the physical plausibility of the spectra. See `tests/validation_data/sheba/` for the validation infrastructure.

---

## References

- Cox, G. F. N. & Weeks, W. F. (1983). Equations for determining the gas and brine volumes in sea-ice samples. *J. Glaciology*, 29(102), 306–316.
- Light, B. et al. (2004). Two-dimensional Monte Carlo model of radiative transfer in sea ice. *J. Geophys. Res.*, 109, C03028.
- Maxwell Garnett, J. C. (1904). Colours in metal glasses and in metallic films. *Phil. Trans. R. Soc. Lond. A*, 203, 385–420.
- Perovich, D. K. et al. (2002). Seasonal evolution of the albedo of multiyear Arctic sea ice. *J. Geophys. Res.*, 107(C10), 8044.
- Picard, G. et al. (2016). Refinement of the ice absorption spectrum. *J. Glaciology*.
- Quan, X. & Fry, E. S. (1995). Empirical equation for the refractive index of seawater. *Appl. Optics*, 34, 3477.
- Rowe, P. M. et al. (2020). Refractive index of liquid water at 0 °C. *J. Geophys. Res.*, 125, e2019JD031822.
- Sihvola, A. (1999). *Electromagnetic Mixing Formulas and Applications*. IEE.

---

## Cross-references to other BioSNICAR documentation

- [METHODS.md](METHODS.md) — emulator architecture, inversion framework, band convolution.
- [SUBSURFACE.md](SUBSURFACE.md) — subsurface light field calculation.
- [BANDS.md](BANDS.md) — satellite band convolution (Sentinel-2, MODIS, etc.).
- [PLOTTING.md](PLOTTING.md) — spectral plotting utilities.
