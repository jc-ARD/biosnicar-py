# Young Ice Extension — Build Specification

**Created**: 2026-06-08  
**Status**: ✅ IMPLEMENTED — v0.4 (this spec is now a historical record; see docs/SEA_ICE_RETRIEVAL.md and docs/sea_ice_validation.md §7 for the as-built result, incl. the two-stream/internal-reflectance deviation from this spec)  
**Tags**: #biosnicar #sea-ice #young-ice #build-spec  
**Depends on**: Sea ice extension v0.2 (completed), sea-ice emulator (completed)

---

## 0. Mission

Add a `young_ice` surface type to BioSNICAR's sea-ice classification system.

"Young ice" covers the earliest stages of sea ice formation — grease ice, nilas, and grey/grey-white ice — typically 0.5–30 cm thick.  These surfaces have a fundamentally different optical character from mature first-year and multiyear ice: they are **semi-transparent**, so the underlying ocean contributes significantly to the observed albedo.  Broadband albedo ranges from ~0.05 (grease ice) to ~0.25 (light nilas, 5–10 cm).

Young ice is important for remote sensing because:
- It covers large Arctic areas during freeze-up (autumn, early winter)
- Its low albedo drives strong surface energy absorption
- Its spectral signature is distinct enough to be classified from VIS/NIR observations
- It maps to WMO/SIGRID-3 ice types SA (new ice), SB (nilas), SI (grey ice), SJ (grey-white ice) — none of which are currently represented

The existing five surface types (FYI_snow, FYI_bare, FYI_summer, MYI_bare, FYI_pond) all represent mature ice.  This extension adds the thin-ice regime that precedes them.

---

## 1. Physics

### 1.1 Why the current forward model is insufficient

The existing `layer_type=4` (sea ice) model uses Maxwell-Garnett effective medium theory to represent brine inclusions in a thick, optically opaque ice column.  It assumes:
- The ice column is deep enough that no radiation reaches the ocean below
- Scattering and absorption are parameterised for centimetre-to-metre scale features

For ice thinner than ~15–20 cm, these assumptions break down because:
1. **Sub-ice ocean contributes**: radiation transmitted through the ice reflects off the dark ocean (albedo ~0.03–0.07) and returns upward, adding to the observed albedo
2. **Ice structure is different**: grease ice consists of frazil crystals in suspension; nilas is a thin elastic sheet — neither is well-described by the Mie-sphere bubble model

### 1.2 Optical model for young ice

The appropriate model is a **thin-slab transmittance approach** combining:

1. **Fresnel reflection at the air–ice interface** (already implemented in `layer_type=1/4`)

2. **Beer-Lambert attenuation through the ice slab**:
   ```
   T(λ) = exp(−κ_eff(λ) × d)
   ```
   where `d` is ice thickness and `κ_eff(λ)` is the effective extinction coefficient of the young ice (absorption + scattering).

3. **Reflection from the sub-ice ocean**:
   ```
   R_ocean(λ) ≈ 0.04  (slightly wavelength-dependent, seawater)
   ```

4. **Combined thin-slab albedo** (two-stream approximation):
   ```
   α(λ) = ρ_surface(λ) + [1 − ρ_surface(λ)]² × T²(λ) × R_ocean(λ)
                           ──────────────────────────────────────────
                           1 − ρ_surface(λ) × R_ocean(λ) × T²(λ)
   ```

This is the standard thin-slab formula from Grenfell & Maykut (1977) and Perovich (1996).

### 1.3 Effective extinction coefficient

`κ_eff(λ)` for young ice is dominated by:
- Pure ice absorption (from the existing Rowe et al. 2020 dataset)
- Scattering from frazil/crystal boundaries (much larger feature size than brine inclusions)
- Brine absorption (salinity is very high in new ice: 15–30 psu)

A first-order parameterisation:
```
κ_eff(λ, T, S) = κ_abs_ice(λ, T) + κ_brine(λ, T, S) + κ_scatter(λ, d)
```

where `κ_scatter` captures the frazil crystal scattering and scales weakly with thickness as the ice consolidates.  For a v0.4 emulator, this can be simplified to treat `κ_eff` as a function of temperature and salinity only (fixing the scatter contribution), since the emulator will learn the residual.

### 1.4 Parameter space

| Parameter | Range | Units | Role |
|---|---|---|---|
| `ice_thickness` | 0.005–0.30 | m | Primary control — determines transmittance |
| `sea_ice_temperature` | −20 to −2 | °C | Controls brine properties and absorption |
| `sea_ice_salinity` | 10–35 | psu | New ice is not desalinated; high-S |
| `ocean_albedo` | 0.03–0.08 | — | Underlying ocean reflectance |
| `solzen` | 20–80 | ° | Illumination geometry |
| `direct` | 0, 1 | binary | Clear vs overcast |

Note: `ice_thickness` replaces `brine_volume_fraction` as the primary parameter — for young ice, thickness is the key observable and the primary controller of sub-ice transmission.

---

## 2. Implementation plan

### Task Y-1: Forward model — young ice optical properties

**File**: `biosnicar/optical_properties/column_OPs.py`  
**New function**: `_compute_young_ice_ops(thickness, T, S, ocean_albedo, model_config)`

Implement the thin-slab Beer-Lambert forward model described in §1.2:
1. Compute `κ_abs_ice(λ)` from the existing Rowe et al. pure-ice absorption data
2. Compute `κ_brine(λ, T, S)` from the existing liquidus-corrected brine RI (already in `brine_optics.py`)
3. Set `κ_scatter` from a literature-derived empirical constant for new ice  
4. Compute `T(λ) = exp(−κ_eff × d)`
5. Compute `α(λ)` using the two-stream thin-slab formula
6. Return as a 480-band albedo array (this replaces the standard τ/ω/g pipeline for this layer type)

The young ice layer does not need its own LUT — the computation is analytic given existing optical data.

**Acceptance criterion**: `_compute_young_ice_ops(d=0.01, T=-10, S=25, ocean_albedo=0.04)` returns a flat-ish spectrum with BBA ≈ 0.05–0.08; `d=0.15` returns BBA ≈ 0.15–0.20.  Validate against Figure 3 of Grenfell & Maykut (1977).

### Task Y-2: New layer_type=6 in the adding-doubling solver

**File**: `biosnicar/rt_solvers/adding_doubling_solver.py`  
Add a branch for `layer_type=6` that calls `_compute_young_ice_ops()` and bypasses the standard `get_layer_OPs()` pipeline.  The young ice layer is treated as a boundary condition (prescribed albedo spectrum) rather than a scattering layer.

**Acceptance criterion**: `run_model(layer_type=6, ice_thickness=0.05, sea_ice_temperature=-10, sea_ice_salinity=20, ocean_albedo=0.04, solzen=60, direct=1)` runs without error and returns BBA ≈ 0.10.

### Task Y-3: Emulator configuration

**File**: `biosnicar/sea_ice/emulator_configs.py`

Add a `young_ice` entry to `SEA_ICE_EMULATOR_CONFIGS`:

```python
def _transform_young_ice(p):
    """Young ice: thin slab over dark ocean."""
    return dict(
        layer_type=6,
        ice_thickness=p["ice_thickness"],
        sea_ice_temperature=p["sea_ice_temperature"],
        sea_ice_salinity=p["sea_ice_salinity"],
        ocean_albedo=p["ocean_albedo"],
        solzen=p["solzen"],
        direct=p["direct"],
    )

"young_ice": {
    "description": "Young ice — grease ice, nilas, grey ice (0.5–30 cm)",
    "params": {
        "ice_thickness":      (0.005, 0.30),
        "sea_ice_temperature": (-20.0, -2.0),
        "sea_ice_salinity":    (10.0,  35.0),
        "ocean_albedo":        (0.03,  0.08),
        "solzen":              (20,    80),
        "direct":              (0,     1),
    },
    "transform_fn":       _transform_young_ice,
    "n_samples":          8000,
    "hidden_layer_sizes": (128, 128, 64),  # thin-ice physics is smooth
    "emulator_file": str(_DATA_DIR / "sea_ice_young_ice_6param.npz"),
},
```

`ice_thickness` is in log-space (add to `_LOG_SAMPLE_PARAMS` in emulator.py and `_LOG_SPACE_PARAMS` in optimize.py) since the NIR transmittance is exponential in thickness.

**Acceptance criterion**: `build_sea_ice_emulators.py young_ice` runs to completion with R² > 0.99 (the thin-slab physics is smooth and well-conditioned, so good R² is expected).

### Task Y-4: WMO/SIGRID-3 mapping

**File**: `biosnicar/sea_ice/ice_chart_mapping.py`

Add `young_ice` entries to `_SIGRID3_TYPE`, `_SIGRID3_MELT`, and the `map_to_wmo()` / `map_to_sigrid3()` logic:

| Ice thickness | WMO term | SIGRID-3 code | SG |
|---|---|---|---|
| < 1 cm | Grease ice / frazil | SA | 1 |
| 1–5 cm | Dark nilas | SB | 1 |
| 5–10 cm | Light nilas | SB | 1 |
| 10–15 cm | Grey ice | SI | 1 |
| 15–30 cm | Grey-white ice | SJ | 1 |

The parameter `ice_thickness` directly maps to WMO stage of development — this is a cleaner mapping than for the other surface types, which require inferring structural properties from optical state.

### Task Y-5: retrieve_sea_ice() integration

**File**: `biosnicar/sea_ice/retrieve.py`

Add `young_ice` to the default emulator fleet loaded by `retrieve_sea_ice()`.  The `known_month` prior for young ice (months 10–2, freeze-up season):
```python
elif m in (10, 11, 12, 1, 2):   # freeze-up months
    season_priors["ice_thickness"] = (0.05, 0.08)  # thin but not grease ice
    season_priors["sea_ice_temperature"] = (-12.0, 6.0)
```

Adjust the summer (months 5–9) prior to explicitly exclude young ice — it is physically impossible in summer.

### Task Y-6: Validation

Validate against **Grenfell & Maykut (1977)** Table 3 / Figure 3 (albedo vs thickness for young Arctic sea ice, 400–1000 nm).  The key checkpoints:
- Grease ice (~1 cm): BBA ≈ 0.05–0.08
- Dark nilas (~3 cm): BBA ≈ 0.08–0.12
- Light nilas (~8 cm): BBA ≈ 0.10–0.18
- Grey ice (~12 cm): BBA ≈ 0.15–0.22

Add a validation block to `sea_ice_emulator_sheba_validation.py` if SHEBA or other young ice spectral data can be located (the SHEBA campaign started in October 1997 during freeze-up, so some young ice data may exist in the archive).

### Task Y-7: Documentation

Update:
- `docs/SEA_ICE_EMULATOR.md` — add `young_ice` to the five-surface-type table (making it six)
- `docs/INVERSION.md` — add `ice_thickness` to the sea ice parameters table
- `docs/sea_ice_validation.md` — add young ice validation results

---

## 3. Acceptance criteria (summary)

| Task | Criterion |
|---|---|
| Y-1 | `_compute_young_ice_ops(0.01, -10, 25, 0.04)` → BBA ≈ 0.05–0.10 |
| Y-2 | `run_model(layer_type=6, ...)` runs, BBA ≈ 0.10 for d=0.05 m |
| Y-3 | Emulator builds with R² > 0.98; `young_ice` in `SEA_ICE_EMULATOR_CONFIGS` |
| Y-4 | `map_to_sigrid3()` returns SA/SB/SI/SJ for young ice with correct SG |
| Y-5 | `retrieve_sea_ice()` classifies synthetic young ice spectra correctly |
| Y-6 | Forward model BBA agrees with Grenfell & Maykut (1977) within ±0.03 |
| Y-7 | Docs updated; `docs/young-ice-build-spec.md` linked from build plan |

---

## 4. Scope decisions

**In scope:**
- Grease ice, nilas, grey ice, grey-white ice (WMO SA, SB, SI, SJ)
- Parameterisation by ice thickness, temperature, salinity, ocean albedo
- Emulator trained on the thin-slab forward model
- WMO/SIGRID-3 mapping with thickness-resolved codes
- Integration into `retrieve_sea_ice()` classification fleet

**Out of scope (v0.4):**
- Pancake ice (SY) — requires a 2D lateral-scattering model
- Ice rind (SR) — thin transparent refrozen surface layer; low priority
- Snow on young ice — young ice is unlikely to carry snow before it thickens
- Antarctic young ice — may differ in salinity regime; defer to v0.5
- Automated thickness retrieval from satellite data — the emulator retrieves thickness from spectral observations, but converting to a reliable map requires additional processing

**Key uncertainty:**
The scatter coefficient `κ_scatter` for frazil/young ice is poorly constrained in the literature.  Grenfell & Maykut (1977) provide extinction coefficients empirically but the spectral dependence is measured only at a few wavelengths.  The v0.4 implementation should treat this as a tunable constant calibrated to the published albedo values, with a clear comment noting the uncertainty.

---

## 5. References

- Grenfell, T. C. & Maykut, G. A. (1977). The optical properties of ice and snow in the Arctic Basin. *Journal of Glaciology*, 18(80), 445–463.
- Perovich, D. K. (1996). The optical properties of sea ice. CRREL Monograph 96-1.
- Light, B. et al. (2008). Optical properties of melting first-year Arctic sea ice. *Journal of Geophysical Research*, 113, C00A02.
- WMO (2014). WMO Sea-Ice Nomenclature. WMO No. 259.
- JCOMM (2014). SIGRID-3: A vector archive format for sea ice georeferenced information. JCOMM Technical Report No. 23.
