# Sea Ice Extension — Build Specification

**Created**: 2026-05-14
**Tags**: #biosnicar #sea-ice #build-spec #engineering #implementation
**Purpose**: Step-by-step build instructions for extending BioSNICAR with sea-ice radiative-transfer support to MVP scope. Written to be executable by a competent engineer (human or another Claude instance) without further hand-holding. Each task has a clear goal, file targets, implementation guidance, and acceptance criteria.

**Audience**: Strong Python engineer; familiarity with scientific computing (numpy, scipy); does not require prior BioSNICAR or sea-ice-physics knowledge — context is provided where needed.

**Prior reading**:
- [[Sea Ice Extension - Scoping]] — the decision context and risk register (not required reading, but provides background).
- The existing BioSNICAR documents in this folder: [[Architecture and Modules]], [[Physics and Radiative Transfer]], [[Optical Properties and Data]], [[Usage and API]] — these explain how the code is structured. Read at least the first three before starting Task 1.

---

## 0. Mission

Extend BioSNICAR — a Python radiative-transfer code for snow and glacial ice — to also handle sea ice. Sea ice differs from terrestrial ice primarily because it contains brine inclusions: micro-pockets of liquid concentrated seawater that remain liquid at sub-zero temperatures. These inclusions change the ice's complex refractive index and produce different albedo spectra than freshwater ice.

The MVP delivers a forward model that, given a sea-ice layer specification (thickness, temperature, salinity, density, optional snow layer), produces a 480-band albedo spectrum that matches published in-situ measurements to within ~%.

Out of scope for MVP: melt ponds, sea-ice algae, salty snow, vertical T/S profiles, spheroid inclusion geometry, Antarctic-specific tuning, inverse retrieval. Each of these is a planned future extension; the MVP must be designed so they can be added cleanly without refactoring.

---

## 1. Prerequisites for the engineer

Before starting, the engineer should:

1. **Have the BioSNICAR repository cloned and working.** Run the existing test suite to confirm baseline:
   ```bash
   cd ~/Code/biosnicar-py
   pytest tests/
   ```
   All tests must pass before touching anything. If they do not, fix the environment first.

2. **Read ~/Documents/'Obsidian Vault'/Projects/BioSNICAR/Physics and Radiative Transfer in full.** Understand the 480-band spectral grid, the three ice-refractive-index variants (`Wrn84`, `Wrn08`, `Pic16`), the LUT structure, the difference between `layer_type=0` (granular snow) and `layer_type=1` (bubbly glacier ice), the adding-doubling solver vs the Toon method, and the `mix_in_impurities()` function in `column_OPs.py`.

3. **Run a baseline calculation** for both `layer_type=0` and `layer_type=1`. Confirm you can produce plausible albedo spectra for snow and bare glacier ice. Save these as reference outputs for regression-testing later.

4. **Read this entire build spec end-to-end** before starting Task 1. Some decisions in Task 2 affect the data structures used in Task 1.

If at any point the existing BioSNICAR code is unclear, stop and ask. Do not guess.

---

## 2. Repository orientation

Critical paths (relative to `~/Code/biosnicar-py`):

- `biosnicar/classes/ice.py` — `Ice` dataclass defining a single layer's properties.
- `biosnicar/classes/illumination.py` — solar source and atmospheric profile loading.
- `biosnicar/classes/impurities.py` — light-absorbing-particle (LAP) handling.
- `biosnicar/classes/model_config.py` — overall configuration including layer types.
- `biosnicar/rt_solvers/adding_doubling_solver.py` — preferred RT solver. Use this.
- `biosnicar/rt_solvers/toon_rt_solver.py` — legacy; do NOT extend to support sea ice. Sea ice requires Fresnel interfaces which only the adding-doubling solver handles correctly.
- `biosnicar/optical_properties/column_OPs.py` — `mix_in_impurities()` and per-layer optical-property assembly. Sea ice will be handled here via a new branch.
- `biosnicar/optical_properties/van_diedenhoven.py` — geometric-optics for hexagonal crystals; not relevant here but documents the LUT-like patterns used.
- `data/OP_data/wavelengths.csv` — the 480-band spectral grid.
- `data/OP_data/rfidx_ice.npz` — complex refractive index of pure ice (three variants).
- `data/OP_data/480band/luts/bubbly_air.npz` — LUT for bubbly ice optical properties. This file is the closest pattern for what `sea_ice.npz` will look like.
- `data/OP_data/lap.npz` — LAP optical properties (BC, dust, algae). Sea-ice algae (deferred to v0.2) will live here.
- `inputs.yaml` — runtime config; the schema needs extending for `LAYER_TYPE: 2`.

After Task 5 (LUT pre-computation), a new path will exist:
- `data/OP_data/480band/luts/sea_ice.npz`
- `data/OP_data/brine_rfidx.npz`

After all tasks, new modules will exist:
- `biosnicar/sea_ice/` — new top-level submodule containing all sea-ice-specific code.
- `biosnicar/sea_ice/__init__.py`
- `biosnicar/sea_ice/brine_volume.py` (Task 1)
- `biosnicar/sea_ice/brine_optics.py` (Task 2)
- `biosnicar/sea_ice/effective_medium.py` (Task 3)
- `biosnicar/sea_ice/sea_ice_optics.py` (Task 4)
- `biosnicar/sea_ice/presets.py` (Task 8)
- `biosnicar/sea_ice/api.py` — high-level `SeaIceColumn`, `SeaIceLayer`, etc. (Task 9)

Plus modifications to existing files:
- `biosnicar/classes/ice.py` (Task 6 — add `layer_type=2` handling)
- `biosnicar/optical_properties/column_OPs.py` (Task 6)
- `inputs.yaml` (Task 7 — schema extension)
- `tests/` — new tests throughout
- `docs/sea_ice.md` — new primer (Task 12)
- `notebooks/sea_ice_mvp.ipynb` — new demo (Task 13)

---

## 3. Scope

### In scope (MVP, v0.1)

- Cox & Weeks (1983) brine-volume-fraction calculation.
- Brine complex refractive index via salinity-corrected pure-seawater spectrum.
- Maxwell-Garnett effective medium for spherical brine inclusions.
- New `layer_type=2` (sea ice) plumbed through the existing infrastructure.
- Pre-computed LUT covering sea-ice parameter space.
- Three presets: `FYI_WINTER_BARE`, `FYI_WINTER_SNOW_COVERED`, `MYI_WINTER_BARE`.
- Python builder API: `SeaIceColumn`, `SeaIceLayer`, `SnowLayer`, `SeaIcePreset`.
- YAML configuration extension supporting `LAYER_TYPE: 2`.
- Validation against at least one SHEBA reference spectrum.
- Documentation and demo notebook.

### Out of scope (deferred to later versions)

- Melt pond layer (v0.2).
- Sea-ice algae as impurity class (v0.2 or v0.3).
- Salty snow with brine wicking (v0.2).
- Vertical T/S profile callables (constant per layer is fine for MVP) (v0.3).
- Antarctic-specific validation and tuning (v0.3).
- Spheroid brine inclusion geometry (v0.4).
- Inverse retrieval (v1.0).
- Methods paper (separate work, standard-release stage).

### Explicit anti-patterns

- Do NOT extend the Toon solver (`toon_rt_solver.py`). Sea ice requires Fresnel interfaces at the air-ice boundary; only the adding-doubling solver handles these correctly. Trying to make Toon work for sea ice produces subtle, hard-to-detect errors.
- Do NOT modify the `layer_type=0` or `layer_type=1` code paths. Sea ice is `layer_type=2`. Existing functionality must be unchanged. Regression-test continuously.
- Do NOT implement sea-ice algae in the MVP. The optical data is not in `lap.npz`, the relevant species differ from snow algae, and getting it right requires lab measurements or surrogate-data calibration. Defer.
- Do NOT couple to a thermodynamic model. The MVP takes T and S as inputs. Albedo evolution over a melt season is out of scope.

---

## 4. Implementation tasks

Each task has a goal, target files, implementation guidance, and acceptance criteria. Tasks are sequenced; do them in order.

### Task 1: Cox & Weeks brine volume calculator

**Goal**: A function that, given salinity (psu) and temperature (°C), returns the brine volume fraction `νb` in sea ice.

**Files**: Create `biosnicar/sea_ice/__init__.py` (empty for now) and `biosnicar/sea_ice/brine_volume.py`.

**Physics**: From Cox & Weeks (1983). The brine volume fraction is the fraction of the ice that is liquid brine rather than solid ice. It depends on the bulk salinity `S` (psu) and ice temperature `T` (°C). The Cox-Weeks equations use empirical polynomial fits to phase-diagram data.

For temperatures between −2°C and −22.9°C (the most operationally common range), the formulation is:

```
F1(T) = -4.732 - 22.45*T - 0.6397*T^2 - 0.01074*T^3
F2(T) = 0.08903 - 0.01763*T - 5.330e-4*T^2 - 8.801e-6*T^3
rho_i = 0.917 - 1.403e-4 * T          # g/cm^3, pure ice density
rho_b = 1.0 + (S * rho_i * F2) / F1   # g/cm^3, sea-ice bulk density
nu_b = (S * rho_i) / (F1 - rho_i * S * F2)   # brine volume fraction (dimensionless)
```

For temperatures between −22.9°C and −44°C (very cold ice), the F1 and F2 polynomial coefficients differ. Implement both ranges; raise `ValueError` if T < −44°C or T > −2°C.

**Function signature**:
```python
def compute_brine_volume(salinity_psu: float, temperature_C: float) -> float:
    """Compute brine volume fraction in sea ice via Cox & Weeks (1983).
    
    Args:
        salinity_psu: Bulk salinity in practical salinity units (typically 1-15 for sea ice).
        temperature_C: Ice temperature in degrees Celsius (must be between -44 and -2).
    
    Returns:
        Brine volume fraction (dimensionless, typically 0.01-0.30).
    """
```

Also implement a vectorised version that accepts numpy arrays for both inputs and broadcasts sensibly.

**Acceptance criteria**:
- Unit tests verify the function against tabulated values from Cox & Weeks (1983), Table 1. Specific reference values:
  - S=10 psu, T=−2°C → ν_b ≈ 0.190
  - S=8 psu, T=−10°C → ν_b ≈ 0.031
  - S=4 psu, T=−15°C → ν_b ≈ 0.011
  - Tolerance: 5% relative error.
- Raises `ValueError` for T outside [−44, −2] or S < 0.
- Vectorised version produces identical results to scalar version when called element-wise.

**Source**: [Cox, G. F. N. & Weeks, W. F. (1983). Equations for determining the gas and brine volumes in sea-ice samples. *Journal of Glaciology*, 29(102), 306–316.](https://www.cambridge.org/core/journals/journal-of-glaciology/article/equations-for-determining-the-gas-and-brine-volumes-in-seaice-samples/3DAA0EFCFA1C2C0DEAA5C5FA3C7DBC9C). If access to the paper is needed, the formulation is also reproduced in many subsequent sea-ice modelling papers including Light et al. (2004).

---

### Task 2: Brine complex refractive index

**Goal**: A function that, given salinity, temperature, and wavelengths, returns the complex refractive index spectrum of the brine.

**Files**: `biosnicar/sea_ice/brine_optics.py`. Pre-computed table at `data/OP_data/brine_rfidx.npz`.

**Physics**: Brine is concentrated seawater. Its complex refractive index `n + ik` is well-known for pure water (Pope & Fry 1997 + Hale & Querry 1973 + Segelstein 1981) but less so for hypersaline brine at sub-zero temperatures. The MVP approximation: start from pure-water RI, apply a salinity correction and a temperature correction. Document the approximation explicitly in the code and in the user docs.

**Implementation steps**:

a. Load the pure-water complex RI spectrum at the BioSNICAR 480-band grid. If a usable `pure_water_rfidx.npz` does not already exist in `data/OP_data/`, derive one by combining Pope & Fry (visible) with Segelstein (extended). Pope & Fry table is available at [http://omlc.org/spectra/water/abs/index.html](http://omlc.org/spectra/water/abs/index.html).

b. Apply a salinity correction. The simplest defensible approach: scale the absorption coefficient by `(1 + alpha * S)` where alpha is a small empirical constant (~0.001 per psu) derived from Pegau & Zaneveld (1993) seawater absorption work. Refractive index real part: shift by ~0.0002 per psu (Quan & Fry 1995 for seawater).

c. Apply a temperature correction. Pure-water absorption has measurable T-dependence in the visible-NIR (Trabjerg & Højerslev 1995, Pegau et al. 1997). For MVP: apply the published seawater temperature corrections; assume they extrapolate adequately to brine temperatures.

d. Pre-compute the result over a grid of (S, T) values and save as `data/OP_data/brine_rfidx.npz` with shape `(n_S, n_T, 480, 2)` — the last dimension storing the real and imaginary parts. Grid: S in [10, 20, 50, 100, 150, 200] psu; T in [−2, −5, −10, −15, −20, −25, −30] °C. Smaller grids if memory becomes an issue.

**Function signature**:
```python
def compute_brine_rfidx(
    salinity_psu: float,
    temperature_C: float,
    wavelengths_um: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute complex refractive index of brine at given S, T.
    
    Returns:
        Complex array of shape (n_wavelengths,) — n_real + 1j * n_imag.
    """
```

Internally, the function interpolates the pre-computed table. If `wavelengths_um` is `None`, use the BioSNICAR 480-band grid.

**Acceptance criteria**:
- At S=35 psu, T=20°C, output matches pure-seawater RI from the literature within 5% across the visible-NIR.
- At S=0 psu, output equals the pure-water RI exactly.
- Interpolation between grid points is smooth and produces no negative imaginary parts (which would be unphysical).
- Unit test that loads `brine_rfidx.npz` and checks shape and dtype.

**Source**: [Pope, R.M. & Fry, E.S. (1997). Absorption spectrum (380–700 nm) of pure water. *Applied Optics*, 36, 8710–8723.](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-36-33-8710). [Segelstein (1981 PhD thesis, U Missouri-Kansas City)](https://omlc.org/spectra/water/data/segelstein81_index.txt) for extended-spectrum. Pegau, W.S. & Zaneveld, J.R.V. (1993) for salinity dependence.

**Documentation requirement**: Add a clear note in the code's docstring explaining that this is an approximation and listing the known limitations (calibration based on seawater near 35 psu, not on brine at 100+ psu).

---

### Task 3: Maxwell-Garnett effective medium

**Goal**: A function that, given the host's complex permittivity, the inclusion's complex permittivity, and the inclusion volume fraction, returns the effective complex permittivity of the composite medium.

**Files**: `biosnicar/sea_ice/effective_medium.py`.

**Physics**: Maxwell-Garnett is the standard effective-medium approximation for sparse, non-touching spherical inclusions in a host matrix. The formula:

```
eps_eff = eps_host * (eps_inc * (1 + 2*f) + 2 * eps_host * (1 - f)) /
                    (eps_inc * (1 - f)   + eps_host * (2 + f))
```

where `f` is the inclusion volume fraction, `eps_host` and `eps_inc` are the complex permittivities. Permittivity is the square of refractive index: `eps = (n + ik)^2`.

**Function signature**:
```python
def maxwell_garnett(
    eps_host: np.ndarray,        # complex array, shape (n_wavelengths,)
    eps_inclusion: np.ndarray,   # complex array, same shape
    volume_fraction: float,      # between 0 and 1
) -> np.ndarray:
    """Maxwell-Garnett effective complex permittivity.
    
    Suitable for sparse spherical inclusions (f < ~0.3). For higher
    fractions, use Bruggeman (not implemented in MVP).
    
    Returns:
        Effective complex permittivity, same shape as inputs.
    """
```

Also provide a helper to convert between refractive index (`n + ik`) and complex permittivity (`eps_real + 1j * eps_imag`).

**Acceptance criteria**:
- At f=0, output equals `eps_host` exactly.
- At f→1, output approaches `eps_inclusion` (slowly — Maxwell-Garnett is asymmetric).
- For real-valued inputs, output is real.
- For known reference cases (e.g. air in water at f=0.5, the literature spread is wide so just check the imaginary part has the right sign), output is physically plausible.
- Unit test with at least three test cases including f=0, f=0.1, f=0.3.

**Source**: Sihvola, A. (1999). *Electromagnetic Mixing Formulas and Applications*. IEE Electromagnetic Waves Series 47. The Maxwell-Garnett formula has many equivalent algebraic forms; the one above (in terms of permittivities, for sphere inclusions) is the most common.

**Note**: Document explicitly in the code that this is the sphere-inclusion variant. The spheroid generalisation (Niccolai/Sihvola) is deferred.

---

### Task 4: Sea-ice optical properties per layer

**Goal**: A function that, given a sea-ice layer specification (thickness, T, S, density, bubble distribution), returns the optical-depth, single-scattering-albedo, and asymmetry-parameter spectra at the BioSNICAR 480-band grid.

**Files**: `biosnicar/sea_ice/sea_ice_optics.py`.

**Implementation steps**:

a. Compute the brine volume fraction `ν_b` via `compute_brine_volume(S, T)`.

b. Compute the brine complex RI spectrum via `compute_brine_rfidx(S, T)`.

c. Load the pure-ice complex RI from `data/OP_data/rfidx_ice.npz` (Picard 2016 variant `Pic16` is the default).

d. Apply Maxwell-Garnett mixing of brine into pure ice: effective ice-with-brine permittivity per wavelength.

e. Compute the gas-bubble contribution. Use the existing `bubbly_air.npz` LUT pattern, but the bubble parameters depend on age class:
   - FYI: small bubbles, ~0.1 mm radius, density ~915 kg/m³
   - MYI: larger bubbles, ~1 mm radius, density ~870 kg/m³
   - User-specified: pass explicitly

f. Mie scattering (via existing `miepython` integration) for the bubbles in the ice-with-brine effective medium. This gives the scattering optical properties.

g. Compute absorption from the imaginary part of the effective complex RI directly (no Mie needed for absorption in an absorbing host).

h. Combine into (τ, ω, g) per wavelength.

**Function signature**:
```python
def compute_sea_ice_optics(
    thickness_m: float,
    salinity_psu: float,
    temperature_C: float,
    density_kg_m3: float,
    bubble_radius_um: float,
    ri_variant: str = "Pic16",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute optical properties of a sea-ice layer.
    
    Returns:
        (tau, ssa, asm) tuple — each is array of length 480 (one value per band).
    """
```

**Acceptance criteria**:
- For S=0 (no salt), result reduces approximately to the existing `bubbly_air.npz` output for the same density and bubble parameters. This is the critical regression check — sea ice with no salt should behave like glacier ice.
- Optical depth scales linearly with thickness (verify this).
- All output values are physically valid: τ ≥ 0, 0 ≤ ω ≤ 1, |g| ≤ 1.
- Unit tests for at least three parameter combinations.

---

### Task 5: LUT pre-computation

**Goal**: Pre-compute sea-ice optical properties over a sensible parameter grid and save as a LUT for fast runtime interpolation.

**Files**: New script `scripts/build_sea_ice_lut.py`. New LUT file `data/OP_data/480band/luts/sea_ice.npz`.

**Implementation**:

a. Define the parameter grid:
   - Temperature: [−2, −5, −10, −15, −20, −25, −30] °C (7 points)
   - Salinity: [1, 4, 6, 8, 10, 12] psu (6 points)
   - Density: [870, 890, 910, 920] kg/m³ (4 points)
   - Bubble radius: [100, 300, 500, 1000] μm (4 points)
   - Total: 7 × 6 × 4 × 4 = 672 parameter combinations

b. For each combination, call `compute_sea_ice_optics()` to compute (τ_per_meter, ω, g) — store per-metre optical depth rather than per-layer, so runtime layer-thickness scaling is just multiplication.

c. Save as a structured `.npz` file with arrays:
   ```
   sea_ice.npz:
     T_grid: shape (7,)
     S_grid: shape (6,)
     rho_grid: shape (4,)
     bubble_radius_grid: shape (4,)
     tau_per_m: shape (7, 6, 4, 4, 480)
     ssa: shape (7, 6, 4, 4, 480)
     asm: shape (7, 6, 4, 4, 480)
   ```

d. The pre-computation is offline. Expect it to take ~1-2 hours on a workstation. Add a progress bar. Run once; commit the resulting `.npz` to the repo (it's a build artefact, but small enough — probably 10-50 MB — to commit).

**Acceptance criteria**:
- Script runs end-to-end without errors.
- Output file loads cleanly into numpy.
- Spot-check: at S=1 psu, T=−20°C, density=910, bubble_radius=300 — output is similar to existing `bubbly_air.npz` at matching density and bubble radius (small differences expected from brine; large differences indicate a bug).
- All grid-point values are physical (τ ≥ 0, 0 ≤ ω ≤ 1, |g| ≤ 1).
- Runtime interpolation off the grid is implemented and tested.

---

### Task 6: Layer-type plumbing

**Goal**: Wire `LAYER_TYPE: 2` (sea ice) through the BioSNICAR config and optical-property dispatch infrastructure.

**Files to modify**:
- `biosnicar/classes/ice.py` — extend the `Ice` dataclass to support sea-ice parameters.
- `biosnicar/optical_properties/column_OPs.py` — dispatch on layer type.
- `biosnicar/classes/model_config.py` — update validation logic if it explicitly enumerates valid layer types.

**Implementation**:

a. In `Ice` dataclass, add fields with defaults that are `None` (so existing usage doesn't break):
   ```python
   salinity_psu: Optional[float] = None
   temperature_C: Optional[float] = None
   bubble_radius_um: Optional[float] = None
   ```

b. In `column_OPs.py`, find the function or block that selects optical properties based on `layer_type`. Add a third branch:
   ```python
   if layer.layer_type == 2:
       tau, ssa, asm = _interpolate_sea_ice_lut(
           layer.salinity_psu, layer.temperature_C,
           layer.density, layer.bubble_radius_um,
       )
       tau = tau * layer.thickness_m  # scale per-metre LUT
   ```

c. Implement `_interpolate_sea_ice_lut()` as a 4-D interpolation over the LUT grid. Use `scipy.interpolate.RegularGridInterpolator`. Cache the loaded LUT at module level.

d. Ensure the existing `layer_type=0` and `layer_type=1` paths are untouched.

**Acceptance criteria**:
- All pre-existing tests still pass (regression check is critical here).
- A new integration test demonstrates that a `layer_type=2` layer produces an albedo spectrum.
- The new code path does not load the LUT until first use (lazy loading).

---

### Task 7: YAML config schema

**Goal**: Extend `inputs.yaml` to support `LAYER_TYPE: 2`.

**Files**: `inputs.yaml`, and wherever the YAML parser is implemented (likely `biosnicar/classes/model_config.py`).

**Implementation**:

Add a sea-ice section to the layer schema:

```yaml
layers:
  - LAYER_TYPE: 2
    THICKNESS_M: 2.0
    TEMPERATURE_C: -10.0
    SALINITY_PSU: 6.0
    BUBBLE_RADIUS_UM: 300
    DENSITY_KG_M3: 910
```

Update the YAML loader to populate the new `Ice` dataclass fields when `LAYER_TYPE == 2`.

Add a validation step: when `LAYER_TYPE: 2`, the fields `TEMPERATURE_C`, `SALINITY_PSU`, `BUBBLE_RADIUS_UM`, `DENSITY_KG_M3` must all be present. Otherwise raise `ValueError` with a helpful message.

**Acceptance criteria**:
- A YAML config with a `LAYER_TYPE: 2` layer parses successfully.
- Missing required field raises a clear error.
- Existing YAML configs (with only `LAYER_TYPE: 0` or `1`) still parse correctly.

---

### Task 8: Presets

**Goal**: Three preset sea-ice configurations for common operational scenarios.

**Files**: `biosnicar/sea_ice/presets.py`.

**Implementation**:

```python
from dataclasses import dataclass
from enum import Enum

@dataclass
class SeaIcePreset:
    name: str
    snow_thickness_m: float
    snow_density_kg_m3: float
    snow_grain_radius_um: float
    ice_layers: list  # list of dicts with thickness, T, S, density, bubble_radius

# Concrete presets as named instances or class methods:

FYI_WINTER_BARE = SeaIcePreset(
    name="FYI_WINTER_BARE",
    snow_thickness_m=0.0,
    snow_density_kg_m3=0.0,
    snow_grain_radius_um=0.0,
    ice_layers=[
        dict(thickness_m=0.05, temperature_C=-25, salinity_psu=12, density_kg_m3=920, bubble_radius_um=100),  # surface
        dict(thickness_m=1.45, temperature_C=-10, salinity_psu=8,  density_kg_m3=915, bubble_radius_um=200),  # bulk
    ],
)

FYI_WINTER_SNOW_COVERED = SeaIcePreset(
    name="FYI_WINTER_SNOW_COVERED",
    snow_thickness_m=0.15,
    snow_density_kg_m3=300,
    snow_grain_radius_um=200,
    ice_layers=[...],  # similar to above
)

MYI_WINTER_BARE = SeaIcePreset(
    name="MYI_WINTER_BARE",
    snow_thickness_m=0.0,
    snow_density_kg_m3=0.0,
    snow_grain_radius_um=0.0,
    ice_layers=[
        dict(thickness_m=0.3, temperature_C=-20, salinity_psu=1, density_kg_m3=850, bubble_radius_um=500),  # desalinated upper layer
        dict(thickness_m=2.7, temperature_C=-8,  salinity_psu=3, density_kg_m3=880, bubble_radius_um=700),  # bulk MYI
    ],
)
```

**Acceptance criteria**:
- Each preset can be instantiated and converted into a `SeaIceColumn` (Task 9).
- Each preset, when run through the full RT pipeline, produces a physically plausible albedo spectrum (broadband albedo in the range [0.4, 0.9] for these winter conditions).
- Unit test that all three presets produce non-error output.

---

### Task 9: Python builder API

**Goal**: User-friendly Python API for constructing and running sea-ice columns.

**Files**: `biosnicar/sea_ice/api.py`.

**Implementation**:

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class SnowLayer:
    thickness_m: float
    density_kg_m3: float
    grain_radius_um: float
    # for v0.2 add salinity_psu

@dataclass  
class SeaIceLayer:
    thickness_m: float
    temperature_C: float
    salinity_psu: float
    density_kg_m3: float = 915
    bubble_radius_um: float = 200
    layer_class: str = "FYI"  # FYI | MYI; informational, used to validate density ranges

@dataclass
class SeaIceColumn:
    layers: List  # SnowLayer or SeaIceLayer, top-down
    
    def compute_albedo(
        self,
        sza_deg: float = 60,
        atmosphere: str = "sub_arctic_winter",
        sky: str = "clear",
    ) -> "Albedo":
        """Run the full RT pipeline and return albedo + diagnostics."""
        # Convert to BioSNICAR Ice / Illumination / ModelConfig objects
        # Call adding_doubling_solver
        # Return wrapper with .spectrum, .broadband, .visible, .nir attributes
```

Also add `from_preset()` constructor:
```python
@classmethod
def from_preset(cls, preset: SeaIcePreset) -> "SeaIceColumn":
    """Build a column from a preset."""
```

**Acceptance criteria**:
- `SeaIceColumn.from_preset(FYI_WINTER_BARE).compute_albedo()` returns an object with `.spectrum`, `.broadband`, `.visible`, `.nir` attributes.
- The Python API output matches the YAML-config output for equivalent inputs.
- API docstrings are present and accurate.

---

### Task 10: Validation-set assembly

**Goal**: Assemble a clean set of SHEBA-era spectral albedo measurements as a test set.

**Files**: New directory `tests/validation_data/sheba/`. Each measurement as a CSV with columns `wavelength_um, albedo, uncertainty`. A `manifest.json` documenting conditions for each measurement.

**Sources**:
- Perovich et al. SHEBA-era papers, especially [Perovich et al. (2002), *J. Geophys. Res.*](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2000JC000438) for bare ice, melt-affected ice, and snow.
- [NSIDC G02012 SHEBA archive](https://nsidc.org/data/g02012)
- Grenfell et al. (1994), Allison et al. (1993) for reference cases.

**Process**:
a. Pull 5-10 spectra from published figures or data archives.
b. Resample to the BioSNICAR 480-band wavelength grid (use linear or cubic interpolation).
c. Document for each: ice age class (FYI/MYI), snow cover, approximate T and S where reported, date, sky condition.
d. Save as standardised CSV format.
e. Write a manifest cataloguing each entry.

**Acceptance criteria**:
- At least 5 spectra in the validation set.
- Each documented with reproducible source citation.
- All spectra at the 480-band grid.
- Manifest is human-readable.

**Note**: If accessing SHEBA data proves harder than expected, fall back to digitising published figures from the Perovich JGR series. Document the digitisation process.

---

### Task 11: Validation tests

**Goal**: Automated comparison of model output against the validation set.

**Files**: `tests/test_sea_ice_validation.py`.

**Implementation**:

For each entry in the validation set, run the model with conditions matching the measurement, compute the spectral RMSE and the broadband-albedo error, and compare against tolerance.

```python
def test_sheba_fyi_winter_bare():
    measurement = load_validation_spectrum("sheba_fyi_winter_2000-03-15.csv")
    column = SeaIceColumn(layers=[
        SeaIceLayer(thickness_m=1.8, temperature_C=-20, salinity_psu=8, ...),
    ])
    model = column.compute_albedo(sza_deg=75, atmosphere="sub_arctic_winter")
    assert_spectral_rmse(model.spectrum, measurement.albedo, max_rmse=0.10)
    assert_broadband_close(model.broadband, measurement.broadband, max_abs_diff=0.05)
```

**Acceptance criteria**:
- At least 3 of the 5+ validation entries pass the tolerance test.
- Failures (if any) are documented as known issues, not silent.
- A validation report (`docs/sea_ice_validation.md`) summarises results.

---

### Task 12: Primer documentation

**Goal**: A documentation page that a researcher could read to onboard themselves to the sea-ice extension.

**Files**: `docs/sea_ice.md`.

**Content**:
- What the extension does and doesn't do.
- The physics in brief: brine inclusions, Cox-Weeks, salinity-corrected RI, Maxwell-Garnett.
- The known approximations and their consequences.
- YAML usage example.
- Python API usage example.
- The three presets with brief descriptions.
- Validation summary.
- References.

Target length: ~2000 words; readable in 15 minutes.

**Acceptance criteria**:
- Reads coherently end-to-end.
- A non-BioSNICAR user can adopt the extension by reading only this doc.
- All code snippets in the doc run without modification.
- Cross-links to relevant existing BioSNICAR docs.

---

### Task 13: Demo notebook

**Goal**: An end-to-end Jupyter notebook reproducing one of the SHEBA validation spectra and showing the API surface.

**Files**: `notebooks/sea_ice_mvp.ipynb`.

**Content**:
1. Markdown intro.
2. Load preset and compute albedo.
3. Plot the resulting spectrum.
4. Overlay a SHEBA reference.
5. Show varying T, S, snow cover and how spectrum changes.
6. Show how to construct a custom column.
7. Brief note on limitations.

**Acceptance criteria**:
- Runs end-to-end without errors on a clean install.
- Produces clear, labelled plots.
- Cells are well-commented.

---

### Task 14: Test suite completion

**Goal**: All new code has unit and integration tests; existing tests still pass.

**Files**: `tests/test_brine_volume.py`, `tests/test_brine_optics.py`, `tests/test_effective_medium.py`, `tests/test_sea_ice_optics.py`, `tests/test_sea_ice_api.py`.

**Coverage targets**:
- Brine volume: test at 3 known reference points (see Task 1 acceptance).
- Brine RI: test that S=0 reduces to pure water; test continuity at grid boundaries.
- Maxwell-Garnett: test edge cases (f=0, f=0.5).
- Sea-ice optics: test that S=0 approximates `layer_type=1` output.
- API: test that all three presets run; test that YAML and Python APIs produce matching outputs.

**Acceptance criteria**:
- Test coverage on new modules ≥ 80% (use `pytest --cov`).
- All tests pass.
- Pre-existing terrestrial-ice tests still pass (regression check).

---

### Task 15: Changelog and release

**Goal**: Document the new feature in `CHANGELOG.rst` and tag a release.

**Files**: `CHANGELOG.rst`, git tag.

**Content**:
- New section: "v0.X-sea-ice-mvp (2026-XX-XX)".
- Itemise: new `layer_type=2` support, new `biosnicar/sea_ice/` module, new LUTs, new docs, new validation tests.
- Note known limitations.

**Acceptance criteria**:
- Changelog entry is clear and complete.
- Git tag applied.
- Branch ready for merge to main (or merged with feature flag).

---

## 5. Acceptance criteria for the MVP overall

Before declaring the MVP done:

1. ✅ All pre-existing BioSNICAR tests pass — no regression in terrestrial-ice functionality.
2. ✅ New test suite passes; coverage ≥ 80% on new modules.
3. ✅ Three presets produce physically plausible albedo spectra (broadband in [0.4, 0.9] for winter conditions).
4. ✅ Validation: at least 3 of 5+ SHEBA reference spectra match within max spectral RMSE 0.10 and max broadband error 0.05.
5. ✅ Documentation page exists and is coherent.
6. ✅ Demo notebook runs end-to-end.
7. ✅ YAML config schema accepts `LAYER_TYPE: 2`.
8. ✅ Python API (`SeaIceColumn`, `from_preset()`) works as documented.
9. ✅ Code review by at least one other person.
10. ✅ Branch tagged as `v0.1-sea-ice` and either merged or ready for merge.

---

## 6. Common pitfalls and gotchas

Specific failure modes to watch for:

- **Salinity unit confusion.** PSU (Practical Salinity Unit) ≈ ‰ (parts per thousand) for practical purposes, but some papers use g/kg explicitly. Document the unit used; convert at the boundary.

- **Temperature sign confusion.** Sea-ice temperature is always negative (in °C). If you see positive temperatures inside the code, something is wrong.

- **Brine volume of zero.** At very cold temperatures or very low salinities, brine volume becomes very small but never zero (mathematically). At zero brine volume, the Maxwell-Garnett formula reduces to pure ice — verify this limit explicitly.

- **Complex permittivity vs complex refractive index.** Permittivity is the square of refractive index. Code that mixes the two is a common bug source. Use explicit named conversions.

- **Sign convention for imaginary index.** BioSNICAR (and most optics codes) use `n + ik` with `k > 0` for absorbing media. Some physics codes use `n - ik`. Pick one and stay consistent.

- **480-band grid indexing.** Bands are indexed 0–479. Bands 0–50 cover the visible (200–700 nm); bands 50–479 cover the NIR. Wavelength units in the codebase are micrometres (μm); some papers report in nanometres. Watch the unit at boundaries.

- **Layer thickness scaling.** The LUT stores per-metre optical depth (`tau_per_m`). At runtime, multiply by `layer.thickness_m`. Forgetting this scaling produces wrong albedo.

- **Adding-doubling vs Toon solver.** Sea ice requires the adding-doubling solver (Fresnel interfaces). The Toon solver will produce wrong results for sea ice and is silently chosen by some legacy configs. Verify the solver before running.

- **Density and bubble correlation.** In real sea ice, lower density is partly because of more bubbles. The LUT treats them as independent parameters; for physically-consistent presets, choose density and bubble radius together (FYI: high density, small bubbles; MYI: lower density, larger bubbles).

- **Snow on sea ice in the MVP is FRESH snow.** Salty snow is deferred to v0.2. If a user supplies a `SnowLayer` for snow on sea ice in the MVP, it is treated as fresh — document this limitation.

- **The SHEBA spectra are sometimes broadband or band-averaged, not 480-band.** When digitising, you may need to interpolate or be honest about the resolution mismatch.

---

## 7. References

Primary source papers (linked):

- [Cox & Weeks (1983) — Equations for determining the gas and brine volumes in sea-ice samples](https://www.cambridge.org/core/journals/journal-of-glaciology/article/equations-for-determining-the-gas-and-brine-volumes-in-seaice-samples/3DAA0EFCFA1C2C0DEAA5C5FA3C7DBC9C) — Task 1.
- [Pope & Fry (1997) — Absorption spectrum of pure water 380-700 nm](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-36-33-8710) — Task 2.
- [Segelstein (1981) pure water complex RI extended-spectrum data](https://omlc.org/spectra/water/data/segelstein81_index.txt) — Task 2.
- Pegau, W.S. & Zaneveld, J.R.V. (1993). Temperature-dependent absorption of light by water. *Limnology and Oceanography* 38(1), 188–192 — Task 2 (salinity correction).
- Picard, G. et al. (2016). Refinement of the ice absorption spectrum in the visible. *Journal of Glaciology* — Task 4 (pure-ice RI).
- Sihvola, A. (1999). *Electromagnetic Mixing Formulas and Applications*. IEE — Task 3 (Maxwell-Garnett).
- [Light, B. et al. (2004) — Two-dimensional Monte Carlo model of radiative transfer in sea ice](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2003JC001887) — methodological reference, validation comparison.
- [Perovich et al. (2002) — Seasonal evolution of the albedo of multiyear Arctic sea ice](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2000JC000438) — validation reference for SHEBA.
- [Grenfell, T.C. & Maykut, G.A. (1977) — The optical properties of ice and snow in the Arctic Basin](https://www.cambridge.org/core/journals/journal-of-glaciology/article/optical-properties-of-ice-and-snow-in-the-arctic-basin/A0FC8B49DA1F45F1F7CDE9BBE9C8B5D9) — foundational reference.

Existing BioSNICAR documentation to consult during implementation:

- [[Physics and Radiative Transfer]] — full physics overview.
- [[Architecture and Modules]] — code structure.
- [[Optical Properties and Data]] — LUT structure.
- [[Usage and API]] — current API patterns.

Data archives:

- [NSIDC G02012 — SHEBA dataset](https://nsidc.org/data/g02012)
- [PANGAEA — MOSAiC data archive](https://www.pangaea.de/) — for future v0.2+ validation; not needed for MVP.

---

## 8. Definition of done

The MVP is complete when:

- A new user can `pip install biosnicar`, run the demo notebook, and produce a sea-ice albedo spectrum that matches a SHEBA reference within the documented tolerance.
- An existing user's terrestrial-ice workflows are unaffected.
- The methods paper writer (a future task, not part of MVP) has a working implementation to describe.

If you are uncertain about any task, stop and check with the project lead before guessing. Wrong-direction implementation work is expensive to unwind.

---
