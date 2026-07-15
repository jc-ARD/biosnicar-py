# Inversion

The inverse module retrieves ice physical properties from observed albedo using an emulator-powered optimisation framework. Given spectral or satellite-band observations, it estimates specific surface area (SSA), impurity concentrations, and optionally bubble radius and density, with uncertainty.

**The recommended approach is to retrieve SSA** — the physically meaningful optical parameter — rather than bubble radius and density individually. See [Why SSA?](#why-ssa-as-the-primary-retrieval-parameter) for the rationale.

## Quick Start

### SSA retrieval (recommended)

```python
from biosnicar.emulator import Emulator
from biosnicar.inverse import retrieve

emu = Emulator.load("data/emulators/glacier_ice_8_param_default.npz")

# observed = your 480-band spectral albedo measurement
result = retrieve(
    observed=observed,
    parameters=["ssa", "black_carbon", "glacier_algae", "dust"],
    emulator=emu,
    fixed_params={"direct": 1, "solzen": 50, "snow_algae": 0},
)
print(result.summary())
print(result.best_fit["ssa"])       # m²/kg
print(result.uncertainty["ssa"])    # 1-sigma on SSA
print(result.derived)               # {"rds_internal": ..., "rho_ref": ...}
```

### Satellite band retrieval

```python
import numpy as np

# Band mode: fix sky conditions, retrieve SSA + impurities
result = retrieve(
    observed=np.array([0.82, 0.78, 0.75, 0.45, 0.03]),
    parameters=["ssa", "black_carbon", "glacier_algae"],
    emulator=emu,
    platform="sentinel2",
    observed_band_names=["B2", "B3", "B4", "B8", "B11"],
    obs_uncertainty=np.array([0.02, 0.02, 0.02, 0.03, 0.05]),
    fixed_params={"direct": 1, "solzen": 50, "dust": 1000, "snow_algae": 0},
)
print(result.summary())
```

### Direct rds/rho retrieval (when density is known)

```python
# When rho is known from in-situ measurement, retrieve rds directly
result = retrieve(
    observed=observed,
    parameters=["rds", "black_carbon", "glacier_algae"],
    emulator=emu,
    fixed_params={"direct": 1, "solzen": 50, "rho": 600, "dust": 1000, "snow_algae": 0},
)
```

## API Reference

### `retrieve()`

Main entry point for parameter retrieval.

| Parameter             | Type          | Default      | Description                                                                            |
| --------------------- | ------------- | ------------ | -------------------------------------------------------------------------------------- |
| `observed`            | ndarray       | required     | 480-band spectral albedo or N-band satellite array                                     |
| `parameters`          | list[str]     | required     | Names of parameters to retrieve                                                        |
| `emulator`            | Emulator      | None         | Trained emulator (fast mode)                                                           |
| `forward_fn`          | callable      | None         | Direct forward function (slow mode). Either `emulator` or `forward_fn` required.       |
| `platform`            | str           | None         | Satellite platform key (e.g. `"sentinel2"`). Activates band mode.                      |
| `observed_band_names` | list[str]     | None         | Band names matching `observed` (e.g. `["B3", "B4", "B11"]`). Required with `platform`. |
| `obs_uncertainty`     | ndarray       | None         | Per-element 1-sigma measurement uncertainty                                            |
| `bounds`              | dict          | None         | `{name: (lo, hi)}` overriding `DEFAULT_BOUNDS`                                         |
| `x0`                  | dict          | None         | `{name: value}` overriding `DEFAULT_X0`                                                |
| `regularization`      | dict          | None         | `{name: (prior_mean, prior_sigma)}` Gaussian priors                                    |
| `wavelength_mask`     | ndarray[bool] | None         | Wavelength mask for spectral mode (True = include)                                     |
| `method`              | str           | `"L-BFGS-B"` | `"L-BFGS-B"`, `"Nelder-Mead"`, `"differential_evolution"`, `"mcmc"`, or `"oe"` (optimal estimation) |
| `mcmc_walkers`        | int           | 32           | MCMC walkers (only for `method="mcmc"`)                                                |
| `mcmc_steps`          | int           | 2000         | MCMC steps                                                                             |
| `mcmc_burn`           | int           | 500          | MCMC burn-in steps to discard                                                          |
| `fixed_params`        | dict          | None         | `{name: value}` for known parameters not to be optimised                               |
| `ssa_rho`             | float         | None         | Reference density (kg/m³) for SSA decomposition. See [SSA Reference Density](#ssa-reference-density). |

**Returns:** `RetrievalResult`

### `RetrievalResult`

| Attribute             | Type            | Description                                                             |
| --------------------- | --------------- | ----------------------------------------------------------------------- |
| `best_fit`            | dict            | `{param_name: optimal_value}`                                           |
| `cost`                | float           | Final chi-squared value                                                 |
| `uncertainty`         | dict            | `{param_name: 1_sigma}` from Hessian or MCMC                           |
| `predicted_albedo`    | ndarray (480,)  | Spectrum at best-fit point                                              |
| `observed`            | ndarray         | Input observations                                                      |
| `converged`           | bool            | Optimiser convergence flag                                              |
| `method`              | str             | Optimisation method used                                                |
| `n_function_evals`    | int             | Number of forward evaluations                                           |
| `derived`             | dict            | Internal decomposition quantities. In SSA mode: `{"rds_internal": float, "rho_ref": float}`. Empty otherwise. |
| `chains`              | ndarray or None | MCMC chains `(n_steps, n_walkers, n_params)`. Only for `method="mcmc"`. |
| `acceptance_fraction` | float or None   | MCMC acceptance rate                                                    |
| `autocorr_time`       | ndarray or None | MCMC autocorrelation time per parameter                                 |
| `posterior_covariance`| dict or None    | OE posterior covariance `{(p_i,p_j): value}` (retrieval space). Only for `method="oe"`. |
| `averaging_kernel_diag`| dict or None   | OE per-parameter information content (~1 measured, ~0 prior-driven). Only for `method="oe"`. |
| `dfs`                 | float or None   | OE degrees of freedom for signal = how many parameters the data genuinely constrained. Only for `method="oe"`. |
| `log_evidence`        | float or None   | OE Laplace log model-evidence (used for surface-type classification). Only for `method="oe"`. |

`result.summary()` returns a human-readable string.

### Optimal estimation (`method="oe"`)

Optimal estimation (Rodgers 2000) replaces "find the best-fit point" with a full
Bayesian retrieval: it combines the measurement (weighted by its error
covariance `S_e`) with a prior (weighted by `S_a`) and returns a **posterior
mean and covariance**, an **averaging kernel** (how much of each retrieved
parameter came from the data vs the prior), and **degrees of freedom for signal**
(DFS — how many parameters the observation genuinely constrained). It is the
recommended path when calibrated uncertainty, honest provenance, or
information-content diagnostics are needed — and it is what
`retrieve_sea_ice(..., method="oe")` uses to produce **per-surface-type
posterior probabilities** (`class_probabilities`) from each type's model
evidence, instead of an ad-hoc lowest-cost vote.

Priors are taken from `regularization` (Gaussian per-parameter); parameters with
no prior get a weak prior spanning their range. `obs_uncertainty` supplies the
diagonal of `S_e`. **Note:** `S_e` currently holds instrument noise only; the
forward-model error term (which makes the posterior uncertainties trustworthy)
is pending empirical calibration — until then OE point estimates are reliable
but its error bars are optimistic. Speed is comparable to or faster than
L-BFGS-B per pixel (a Gauss–Newton step uses a cheap emulator Jacobian);
~20–25× the work of a single forward eval, vs ~thousands for MCMC.

When `rds` and `rho` are both in `best_fit`, the `.ssa` and `.ssa_uncertainty` properties compute SSA and its propagated uncertainty from rds and rho.

### Default Bounds and Initial Guesses

| Parameter       | Bounds        | Default x0 | Units                        |
| --------------- | ------------- | ---------- | ---------------------------- |
| `ssa`           | (0.01, 300)   | 2.0        | m²/kg (specific surface area)|
| `rds`           | (100, 5000)   | 1000       | um (bubble effective radius) |
| `rho`           | (100, 917)    | 500        | kg/m³ (density)              |
| `solzen`        | (20, 80)      | 50         | degrees (solar zenith angle) |
| `direct`        | (0, 1)        | 1          | binary (0=cloudy, 1=clear)   |
| `black_carbon`  | (0, 100000)   | 100        | ppb (black carbon)           |
| `snow_algae`    | (0, 500000)   | 10000      | cells/mL (snow algae)        |
| `glacier_algae` | (0, 100000)   | 100        | cells/mL (glacier algae)     |
| `dust`          | (0, 500000)   | 100        | ppb (mineral dust)           |

When using an emulator, bounds for `ssa` are auto-computed from the emulator's rds training range and the reference density, ensuring the internal (rds, rho) decomposition stays within the emulator's valid region. Bounds for other parameters are auto-filled from the emulator's training range.

**Notes:**
- `direct` is a binary flag and **cannot** be included in `parameters` — it raises a `ValueError`. Pass it via `fixed_params` instead (see [Binary Parameters](#binary-parameters)).
- `ssa` **cannot** appear alongside `rds` or `rho` in `parameters` — it raises a `ValueError`. SSA replaces both (see [SSA Constraints](#ssa-constraints)).

### Cost Functions

- **`spectral_cost`**: Chi-squared over 480-band spectral albedo, with optional per-wavelength uncertainty, wavelength mask, and regularization
- **`band_cost`**: Chi-squared over satellite bands. Predicts 480-band albedo, convolves via `to_platform()`, compares to observed band values.

## Why SSA as the Primary Retrieval Parameter?

### The rds/rho degeneracy problem

The spectral shape of ice albedo in the NIR is controlled by the **specific surface area** (SSA), not by bubble radius (`rds`) or density (`rho`) individually. SSA for bubbly ice is:

```
SSA = 3 * (1 - rho/917) / (rds_m * rho)    [m² kg⁻¹]
```

where `rds_m` is the bubble radius in metres and 917 kg/m³ is the density of pure ice.

Many different (rds, rho) combinations produce the same SSA and therefore nearly identical spectra. This is a **many-to-one degeneracy**: the inversion has a continuous valley of equally good solutions in (rds, rho) space. Retrieving rds and rho separately gives:

| Parameter | Typical individual error | SSA error from same retrieval |
| --------- | ----------------------- | ----------------------------- |
| rds       | ~73%                    | —                             |
| rho       | ~33%                    | —                             |
| **SSA**   | —                       | **~5.5%**                     |

The individual parameter errors are high because the optimiser can trade off rds against rho along the valley floor without changing the cost. But SSA — the quantity that actually determines the spectrum — is tightly constrained.

### How SSA retrieval works

When `"ssa"` is in `parameters`, the inversion:

1. **Optimises in SSA space** — the optimiser sees SSA as a single parameter, not two degenerate ones
2. **Decomposes SSA into (rds, rho)** for each emulator call using a reference density:
   ```
   rds_um = 3 * (1 - rho_ref/917) / (SSA * rho_ref) / 1e-6
   ```
3. **Reports SSA directly** in `result.best_fit["ssa"]` and `result.uncertainty["ssa"]`
4. **Records the internal decomposition** in `result.derived` for transparency

The reference density is arbitrary because SSA dominates the spectral response. Second-order effects (asymmetry parameter g, wavelength-dependent Q_sca) depend weakly on the individual rds value, but these are small compared to emulator error.

### When to use SSA vs direct rds/rho

| Scenario | Approach |
| -------- | -------- |
| General case (no independent density measurement) | Use `"ssa"` — it is better constrained than either rds or rho alone |
| Independent rho measurement available (e.g. from ice cores or pits) | Fix rho via `fixed_params`, retrieve `"rds"` directly — the degeneracy is broken |
| Publication needing rds and rho separately | Use SSA retrieval + compute rds from SSA and independently measured rho |

### SSA Reference Density

The reference density used to decompose SSA into (rds, rho) for emulator calls is resolved in this order:

1. `ssa_rho` keyword argument (explicit override)
2. `fixed_params["rho"]` (if rho is fixed)
3. Midpoint of the emulator's rho training range
4. Fallback: 500.0 kg/m³

The choice of reference density has minimal impact on the retrieved SSA because the spectral response is dominated by SSA, not by the individual (rds, rho) decomposition. Different reference densities yield the same SSA but different internal rds values in `result.derived`.

### SSA Constraints

- `"ssa"` **cannot** be retrieved alongside `"rds"` or `"rho"` in the same `parameters` list — this raises a `ValueError`:
  ```
  ValueError: Cannot retrieve 'ssa' alongside 'rds' or 'rho'.
  SSA replaces both — the emulator decomposes SSA into (rds, rho)
  internally using a reference density.
  ```
- Power users who need separate (rds, rho) retrieval can still do so — just pass `parameters=["rds", "rho", ...]` without `"ssa"`.

## Observation Modes

### Spectral mode (default)

Pass a 480-element array of spectral albedo (e.g. from a field spectrometer). All wavelengths are compared against the emulator prediction. Use `wavelength_mask` to exclude noisy bands (e.g. water vapour absorption at 1.38 and 1.87 um).

This mode provides the maximum information content and can typically constrain 4-5 parameters simultaneously.

### Satellite band mode

Set `platform` and `observed_band_names` to work natively in band space. The emulator predicts the full 480-band spectrum internally, convolves to the platform's bands, and compares to your observations. No need to reconstruct a continuous spectrum from satellite data.

Supported platforms: `sentinel2`, `sentinel3`, `landsat8`, `modis`.

**Important:** Band mode provides substantially less spectral information than spectral mode (typically 4-5 broadband values vs 480 spectral values). It cannot constrain as many free parameters. For robust band-mode retrieval:
- Fix well-constrained parameters (`direct`, `solzen`, `dust`) via `fixed_params`
- Limit free parameters to 2-3 (typically SSA plus 1-2 dominant impurities)
- Use `obs_uncertainty` to weight bands appropriately

### Direct model mode

Pass a `forward_fn` callable instead of an emulator. Slower (~seconds per retrieval) but uses the exact forward model. Useful for validating emulator-based results.

```python
from biosnicar import run_model
import numpy as np

result = retrieve(
    observed=observed,
    parameters=["rds"],
    forward_fn=lambda rds: np.array(run_model(rds=int(rds), layer_type=1).albedo),
    bounds={"rds": (100, 5000)},
)
```

## Retrievable Parameters

| Parameter       | Physical meaning            | Spectral effect                                                   | Sensitivity      |
| --------------- | --------------------------- | ----------------------------------------------------------------- | ---------------- |
| `ssa`           | Specific surface area       | Dominates NIR shape: higher SSA = more scattering = higher NIR    | **Very high** (NIR) |
| `rds`           | Bubble effective radius     | Controls NIR albedo: larger bubbles = less scattering = lower NIR | High (NIR)       |
| `rho`           | Ice density                 | Controls optical depth per unit thickness                         | Moderate         |
| `solzen`        | Solar zenith angle          | Path length effect; higher SZA = lower albedo                     | Moderate         |
| `black_carbon`  | Black carbon (upper layer)  | Broadband darkening, strongest in visible                         | High (VIS)       |
| `snow_algae`    | Snow algae (upper layer)    | Visible darkening with carotenoid and chl-a features              | High (VIS)       |
| `glacier_algae` | Glacier algae (upper layer) | Visible darkening with carotenoid and chl-a features              | High (VIS)       |
| `dust`          | Mineral dust (upper layer)  | Weak broadband visible darkening                                  | Low (see caveat) |

The impurity names correspond to the YAML keys in `inputs.yaml`. The defaults are `black_carbon`, `snow_algae`, `glacier_algae`, and `dust`.

### Dust sensitivity caveat

Mineral dust has a very flat spectral signature at typical environmental concentrations (< ~5000 ppb). The cost surface is essentially insensitive to dust concentration in this regime — the spectral effect of 100 vs 5000 ppb dust is smaller than typical measurement noise. Dust retrievals are unreliable unless:
- Concentrations are very high (> ~10,000 ppb)
- Strong prior constraints are applied via `regularization`
- Dust is fixed via `fixed_params` when it is not the primary parameter of interest

## Binary Parameters

The `direct` parameter (sky condition: 0 = diffuse, 1 = direct beam) is a binary flag. It cannot be continuously optimised — the cost surface is flat everywhere except at the two valid values, and the forward model treats it as a discrete switch.

Including `direct` in `parameters` raises a `ValueError`:

```
ValueError: Binary parameters {'direct'} cannot be continuously optimised.
Pass them via `fixed_params` instead.
```

Sky conditions are typically known from meteorological observations or assumed for a given scene, so this is not a practical limitation.

## Fixed Parameters

### Fixed per emulator (at build time)

| Parameter    | Typical value | Why fixed                                                |
| ------------ | ------------- | -------------------------------------------------------- |
| `layer_type` | 1 (solid ice) | Defines the physical model                               |
| `dz`         | [0.02, 0.146] | Weakly constrained; trades off with density              |
| `incoming`   | 0-6           | Atmospheric irradiance; discrete index, not interpolable |

To change any of these, build a separate emulator. Emulators are small (~100-200 KB), so maintaining a library for different configurations is practical.

### Fixed per retrieval (`fixed_params`)

Use `fixed_params` to constrain parameters when ancillary information is available. The parameter must be part of the emulator's input space.

```python
# Sky conditions known (direct MUST be fixed — binary parameter)
result = retrieve(..., fixed_params={"direct": 1})

# SZA and dust known — retrieve SSA + impurities
result = retrieve(
    ...,
    parameters=["ssa", "black_carbon", "glacier_algae"],
    fixed_params={"direct": 1, "solzen": 53.2, "dust": 500, "snow_algae": 0},
)
```

This reduces the optimisation dimensionality and tightens uncertainties on the remaining parameters.

## Log-Space Parameterisation

Impurity concentrations (`black_carbon`, `snow_algae`, `glacier_algae`, `dust`) and SSA are optimised in log10(x + 1) space internally. This is critical for robust retrieval because these parameters span orders of magnitude (0 to 500,000 for impurities; 0.01 to 300 for SSA).

### Why log space?

In linear parameter space:
- The cost surface is extremely flat at high concentrations (the spectral signal saturates)
- Gradient-based optimisers get trapped at high concentrations where the gradient is near zero
- Initial guesses at the midpoint of bounds (e.g. 250,000 for a [0, 500,000] range) are far from typical environmental values

The log10(x + 1) transformation:
- Compresses the range: [0, 500,000] maps to [0, 5.7]
- Makes the cost surface approximately equally sensitive across orders of magnitude
- Places the initial guess near typical environmental concentrations (~700 ppb)
- The +1 offset ensures log10(0 + 1) = 0, so zero concentration maps cleanly

For SSA, log-space is appropriate because SSA spans from ~0.01 m²/kg (dense ice with large bubbles) to ~300 m²/kg (fresh snow), covering four orders of magnitude.

### User-facing behaviour

This transformation is **fully transparent** — you always provide bounds, x0, and read results in linear space. The log transformation is applied and reversed internally. Uncertainties are propagated back via the chain rule.

## Optimisation Methods

### L-BFGS-B (default) — Hybrid DE + Gradient Polish

The default method uses a two-phase strategy to combine global exploration with fast local convergence:

1. **Phase 1: Quick differential evolution (DE) pre-search.** For problems with 2+ free parameters, a quick DE run (maxiter=100, popsize=10) globally explores the parameter space to identify the basin of attraction containing the global minimum. This typically requires ~1000-3000 function evaluations.

2. **Phase 2: L-BFGS-B polishing.** The DE result (if better than the default x0) seeds a quasi-Newton L-BFGS-B optimisation that converges to machine-precision accuracy in ~100-200 additional evaluations.

**Why not pure L-BFGS-B?** The ice albedo cost surface has multiple local minima. In testing, 50 random-restart L-BFGS-B runs all converged to the same suboptimal minimum (dust overestimated by 500x). The quick DE pre-search reliably escapes local minima at modest computational cost.

**Why not pure DE?** DE alone is accurate to ~1-3 significant figures. L-BFGS-B polishing refines this to machine precision.

### Nelder-Mead

Derivative-free simplex method. More robust than L-BFGS-B when the cost surface is noisy (e.g. when using the direct forward model instead of the smooth emulator).

Bounds are enforced via a penalty function (returns 1e20 for out-of-bounds evaluations) since scipy's Nelder-Mead does not natively support box constraints in older scipy versions.

### Differential Evolution (standalone)

Rigorous global search. Use when the cost surface is known to be multimodal or when no good initial guess is available. Slower but guarantees convergence to the global minimum given sufficient iterations.

### MCMC

Full Bayesian posterior sampling via `emcee`. Reports posterior median and standard deviation. Full chains available for corner plots and correlation analysis. Requires `emcee>=3`.

### When to use each optimiser

| Method                     | Evaluations (typical)   | Best for                                                                                             |
| -------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------- |
| **L-BFGS-B** (hybrid)     | ~1,500-3,000            | Default. Fast, accurate. Hybrid DE pre-search avoids local minima.                                   |
| **Nelder-Mead**            | ~300-500                | Derivative-free fallback. Robust to noisy cost surfaces (e.g. direct forward model).                 |
| **differential_evolution** | ~500-5,000              | Global search. Use when initial guess is poor or cost surface is multimodal.                         |
| **mcmc**                   | ~50,000-200,000         | Full Bayesian posterior. Publication-quality uncertainty, parameter correlations, bimodal detection.  |

## Design Rationale

### Why SSA instead of rds and rho?

The spectral albedo of bubbly ice is controlled by SSA, not by rds or rho individually. This is because the scattering optical depth per unit mass scales with SSA in the geometric-optics limit: more surface area per unit mass means more scattering events per unit path length, producing higher albedo.

The practical consequence is that retrieving rds and rho separately from spectral observations produces **large individual errors** (73% and 33% in synthetic tests) even when the **SSA error is only ~5.5%**. The optimiser wanders along the valley of constant SSA in (rds, rho) space — each individual parameter is poorly constrained, but their combination (SSA) is tightly constrained.

Optimising directly in SSA space eliminates this degeneracy. The optimiser has one parameter instead of two, the cost surface has a clean minimum instead of a valley, and the reported uncertainty reflects the true information content of the observations.

### Why upper-layer impurities only?

Impurity concentrations are retrieved for the **surface layer only** (top ~2 cm). This reflects the physics:
- Biological impurities live at the ice surface (light for photosynthesis)
- Mineral impurities accumulate via deposition and melt concentration
- The spectral signal is dominated by the surface layer
- Subsurface impurities are undetectable: photons interacting with deeply buried particles are absorbed before reaching the surface

### Why not retrieve layer thickness?

Layer thickness (`dz`) and density (`rho`) trade off nearly perfectly in their effect on optical depth: doubling thickness has the same radiative effect as doubling density. This degeneracy makes them jointly unidentifiable from spectral observations alone. Fixing `dz` breaks the degeneracy.

### Why chi-squared cost function?

Chi-squared is the standard cost for physical retrieval because:
- **Probabilistic interpretation**: negative log-likelihood under Gaussian errors
- **Uncertainty weighting**: well-measured bands contribute more
- **Free uncertainty estimates**: the Hessian at the minimum gives the covariance matrix
- **MCMC compatible**: `log_prob = -0.5 * cost`

### Why Hessian uncertainty as default?

The finite-difference Hessian at the optimum costs ~4n² extra emulator evaluations (negligible at microsecond speed). Inverting it gives the covariance matrix under a Gaussian approximation, which is accurate for well-constrained unimodal problems. Use MCMC when you need the full posterior or suspect parameter degeneracies.

## Uncertainty Estimation

### Hessian-based (default)

Applied automatically after L-BFGS-B, Nelder-Mead, and differential evolution:
1. Compute finite-difference Hessian at the optimum (4-point central differences, step size = 1e-4 × parameter range)
2. Invert to get the covariance matrix
3. Report `sqrt(diag(cov))` as 1-sigma uncertainties
4. Singular Hessian → infinity uncertainty (unconstrained parameter)

For parameters optimised in log space (impurities and SSA), the Hessian is computed in log space and uncertainties are propagated back to linear space: `sigma_linear ≈ (x + 1) × ln(10) × sigma_log`. This means uncertainty scales with the retrieved value — reflecting the physical reality that spectral sensitivity to these parameters is approximately logarithmic.

### MCMC posteriors

With `method="mcmc"`, the full posterior distribution is sampled:
- Uniform prior within parameter bounds
- Log-likelihood = `-0.5 * chi_squared`
- Reports posterior median and standard deviation
- Full chains available for corner plots and correlation analysis
- Requires `emcee` (`pip install emcee>=3`)

### Regularization (Gaussian priors)

Add prior information via `regularization={name: (mean, sigma)}`. This adds a penalty term to the cost function: `((value - mean) / sigma)²`. Useful for constraining degenerate parameters when prior knowledge exists (e.g. density from in-situ measurements, dust from nearby samples). Can also be applied to SSA if an independent estimate is available.

## Worked Examples

### Field spectrometer measurement

```python
emu = Emulator.load("data/emulators/glacier_ice_8_param_default.npz")

# Measured 480-band albedo from ASD FieldSpec
# SSA retrieval — no need to know rho
result = retrieve(
    observed=measured_albedo,
    parameters=["ssa", "black_carbon", "glacier_algae", "dust"],
    emulator=emu,
    fixed_params={"direct": 1, "solzen": 50, "snow_algae": 0},
)
print(result.summary())
print(f"SSA = {result.best_fit['ssa']:.2f} ± {result.uncertainty['ssa']:.2f} m²/kg")
```

### Sentinel-2 pixel

```python
import numpy as np

result = retrieve(
    observed=np.array([0.82, 0.78, 0.75, 0.45, 0.03]),
    parameters=["ssa", "black_carbon", "glacier_algae"],
    emulator=emu,
    platform="sentinel2",
    observed_band_names=["B2", "B3", "B4", "B8", "B11"],
    obs_uncertainty=np.array([0.02, 0.02, 0.02, 0.03, 0.05]),
    fixed_params={"direct": 1, "solzen": 50, "dust": 1000, "snow_algae": 0},
)
```

### Partial retrieval with known density

```python
# When rho is known, retrieve rds directly (no SSA needed)
result = retrieve(
    observed=measured_albedo,
    parameters=["rds", "black_carbon", "glacier_algae"],
    emulator=emu,
    fixed_params={"direct": 1, "solzen": 50, "rho": 750, "dust": 500, "snow_algae": 0},
)
```

### SSA retrieval with custom reference density

```python
# Override the reference density used for SSA decomposition
result = retrieve(
    observed=measured_albedo,
    parameters=["ssa", "black_carbon", "glacier_algae"],
    emulator=emu,
    fixed_params={"direct": 1, "solzen": 50, "dust": 500, "snow_algae": 0},
    ssa_rho=700.0,  # use 700 kg/m³ as reference density
)
# Different ssa_rho gives same SSA, different internal rds
print(result.derived)  # {"rds_internal": ..., "rho_ref": 700.0}
```

### Computing true SSA for validation

```python
from biosnicar.inverse.result import _compute_ssa

# For synthetic tests, compute true SSA from known physical parameters
true_ssa = _compute_ssa(rds_um=1000, rho=600)  # m²/kg
ssa_error = abs(result.best_fit["ssa"] - true_ssa) / true_ssa * 100
print(f"SSA error: {ssa_error:.1f}%")
```

### MCMC for publication-quality uncertainty

```python
result = retrieve(
    observed=measured_albedo,
    parameters=["ssa", "black_carbon", "glacier_algae"],
    emulator=emu,
    method="mcmc",
    mcmc_walkers=32,
    mcmc_steps=5000,
    mcmc_burn=1000,
    fixed_params={"direct": 1, "solzen": 50, "snow_algae": 0},
)

# Posterior statistics
print(result.summary())

# Full chains for corner plots
flat_chain = result.chains.reshape(-1, 3)
# import corner; corner.corner(flat_chain, labels=["SSA", "BC", "algae"])
```

## Known Limitations

1. **Binary parameters** — `direct` cannot be retrieved; it must be passed via `fixed_params`. This is enforced with a `ValueError`.

2. **SSA/rds/rho mutual exclusion** — `"ssa"` cannot appear alongside `"rds"` or `"rho"` in `parameters`. SSA replaces both. To retrieve rds and rho individually, omit `"ssa"`.

3. **Dust sensitivity** — mineral dust has a very flat spectral signature at typical environmental concentrations (< ~5000 ppb). Dust retrievals are unreliable unless concentrations are very high or strong priors are used. Consider fixing dust via `fixed_params` when it is not the primary parameter of interest.

4. **Band-mode information limits** — satellite band observations (4-5 broadband values) cannot constrain more than 2-3 free parameters. Fix sky conditions and dust for robust band-mode retrievals.

5. **Accuracy near bounds** — MLP predictions may be less accurate at extreme edges of training range. Set bounds slightly wider than expected retrieval range.

6. **No extrapolation** — predictions outside training bounds are unreliable. The emulator clips and warns but does not refuse.

7. **Single-layer impurities** — only surface-layer impurity concentrations are retrieved. Suitable for glacier ice (surface-dominated signal) but not for distributed impurity profiles.

8. **Solid ice only** — the default emulator assumes `layer_type=1`. For snow retrievals, build a custom emulator with `layer_type=0`.

9. **No atmospheric correction** — satellite reflectances are assumed to be surface reflectance. Atmospheric correction is out of scope.

10. **Parameter degeneracies** — some combinations produce similar spectra (e.g. low-concentration BC and algae trade off; dust and algae are spectrally degenerate at broadband resolution). Hessian uncertainties will be large; MCMC reveals the correlation structure. Regularization can help. The rds/rho degeneracy is addressed by using SSA.

11. **Irradiance profile fixed per emulator** — the `incoming` parameter (atmosphere) is a discrete index (0-6), not continuously interpolable. Build separate emulators for different atmospheric conditions.

12. **SSA reference density is cosmetic** — the internal (rds, rho) decomposition in `result.derived` depends on the reference density. Different `ssa_rho` values give different internal rds but the same SSA. Do not interpret `rds_internal` as a physical measurement.

## Sea Ice Inversion

Sea ice retrieval uses the same `retrieve()` function and the same four optimisation methods as glacier ice.  This section describes the differences.

### The `brine_volume_fraction` parameter

The primary difference between glacier ice and sea ice inversion is that **sea ice has no SSA analogue** — there is no single quantity like SSA that collapses the spectral degeneracy of all ice-structure parameters simultaneously.  However, the dominant degeneracy in bare sea ice inversion is between temperature and salinity: both control brine volume fraction `Vb` via the Cox & Weeks (1983) equation, and many (T, S) pairs produce the same `Vb` and therefore the same spectrum.

`brine_volume_fraction` resolves this degeneracy in the same way SSA resolves rds/rho degeneracy:

| Glacier ice | Sea ice |
|---|---|
| rds and rho are degenerate through SSA | T and S are degenerate through Vb |
| Retrieve SSA, not (rds, rho) | Retrieve `brine_volume_fraction`, not (T, S) |
| Post-hoc: rds = f(SSA, rho_ref) | Post-hoc: T = f(Vb, S_ref) |
| rho uncertainty ~33% individually | S uncertainty large individually |
| SSA uncertainty ~5.5% | Vb removes the (T, S) degeneracy but is itself **weakly identified** when bubble scattering dominates — synthetic recovery R² is negative for FYI/MYI bare ice with 16–24% at-bounds (E1, `parameter_retrieval_results.md`); the well-constrained bare-ice parameter is `sea_ice_bubble_radius` (R² 0.66–0.82) |

The `FYI_bare` and `MYI_bare` emulators are trained with `brine_volume_fraction` as a parameter.  Their default bounds in `DEFAULT_BOUNDS` are:

```python
"brine_volume_fraction": (0.005, 0.15),   # FYI; narrow to (0.005, 0.05) for MYI
```

Post-hoc temperature recovery when salinity is known:

```python
from biosnicar.sea_ice.brine_volume import invert_brine_volume
from biosnicar.sea_ice.emulator_configs import FYI_BARE_S_REF   # = 6 psu

result = retrieve(observed=spectrum, parameters=["brine_volume_fraction", ...], ...)
T = invert_brine_volume(result.best_fit["brine_volume_fraction"], FYI_BARE_S_REF)
```

### retrieve_sea_ice() — classify and retrieve simultaneously

When the surface type is not known in advance, `retrieve_sea_ice()` is the recommended entry point.  It fits all seven surface-type models (six emulators + the analytical open-water model) against the observation and classifies by the lowest band-masked chi-squared residual:

```python
from biosnicar.sea_ice.retrieve import retrieve_sea_ice

result = retrieve_sea_ice(
    observed = spectrum,          # 480-band or satellite band array
    solzen   = 60,                # fix known conditions
    direct   = 1,
)

result.surface_type          # "FYI_bare", "FYI_snow", "FYI_summer", "MYI_bare", "FYI_pond"
result.confidence            # 0-1; how decisively it outperformed the next best
result.parameters            # {param: value} from the winning emulator
result.cost_per_type         # {type: chi_squared} for every candidate type
result.all_fits              # {type: RetrievalResult} for full access to each fit
result.to_outputs()          # Outputs object with .BBA, .to_platform(), etc.
```

`retrieve_sea_ice()` also accepts `platform` and `observed_band_names` for satellite band classification:

```python
result = retrieve_sea_ice(
    observed            = np.array([s2.B3, s2.B8, s2.B11]),
    platform            = "sentinel2",
    observed_band_names = ["B3", "B8", "B11"],
    obs_uncertainty     = np.array([0.02, 0.02, 0.03]),
    solzen              = 60,
    direct              = 1,
)
```

### Sea ice default bounds and initial guesses

| Parameter | Bounds | x₀ | Notes |
|---|---|---|---|
| `brine_volume_fraction` | (0.005, 0.15) | 0.04 | ≈T=−10°C at S=6 psu for FYI |
| `sea_ice_bubble_radius` | (50, 2000) µm | 200 | log-space in optimiser |
| `sea_ice_temperature` | (−30, −2) °C | −10 | Legacy — prefer `brine_volume_fraction` |
| `sea_ice_salinity` | (0, 20) psu | 6 | Legacy — prefer `brine_volume_fraction` |
| `rho_DL` | (820, 900) kg/m³ | 850 | DL density; fix unless specifically retrieving |
| `tau_snow` | (50, 3000) | 400 | Snow depth / grain radius (dimensionless); log-space. Replaces direct `snow_depth` retrieval — physical depth is derived as `tau_snow * snow_grain_radius * 1e-6` m and returned in `result.parameters["snow_depth"]` |
| `snow_depth` | (0.02, 0.30) m | 0.10 | Derived output for FYI_snow; direct retrieval only for custom emulators |
| `snow_grain_radius` | (50, 2000) µm | 500 | log-space in optimiser |
| `ssl_grain_radius` | (500, 5000) µm | 2000 | log-space in optimiser |
| `pond_depth` | (0.02, 0.60) m | 0.15 | log-space in optimiser |
| `ice_thickness_cm` | (0.5, 30) cm | — | Young ice (layer_type=6); log-space. Metres derived in `result.parameters["ice_thickness"]` |
| `ocean_albedo` | (0.03, 0.08) | — | Sub-ice ocean reflectance for young ice |

### Sea ice known limitations

1. **`brine_volume_fraction` and `sea_ice_bubble_radius` are partially correlated** — both affect NIR albedo.  When retrieving both simultaneously, the Hessian uncertainties may be large.  Use MCMC to visualise the posterior and correlation structure, or fix one and retrieve the other.

2. **Surface type classification requires good spectral coverage** — classification from 3 satellite bands is possible but less reliable than from the full 480-band spectrum.  Ensure the bands you provide span at least VIS and NIR (e.g. B3 + B8 for Sentinel-2), or include a SWIR band (B11) for better bare-ice vs snow discrimination.

3. **Bare ice "training R²" is a misleading metric** — FYI_bare and MYI_bare report training R² ≈ 0.70 vs >0.99 for the other types.  This is an artifact of scoring in PCA-coefficient space (uniform average over ~26 components, where near-zero-variance tail components score poorly).  The true held-out *spectral* R² of FYI_bare is 0.997 with BBA MAE ≈ 0.003 (2000 forward-model spectra).  See [SEA_ICE_EMULATOR.md](SEA_ICE_EMULATOR.md) for the full audit.

4. **Reference salinity is a fixed assumption** — the `FYI_bare` emulator uses `FYI_BARE_S_REF = 6 psu` and `MYI_bare` uses `MYI_BARE_S_REF = 2 psu` internally.  If the true bulk salinity is very different (e.g. highly saline new ice), build a custom emulator with a different `S_ref` in the transform function.

5. **Pond floor properties are fixed** — the `FYI_pond` emulator calibrates floor black carbon at 1200 ppb (from Morassutti 1995 validation).  If the floor is known to be unusually clean or dirty, use `retrieve()` directly with `black_carbon` as a free parameter.

6. **No SSL for bare winter ice** — the `FYI_bare` emulator has no SSL.  For summer observations where an SSL is present, use `FYI_summer`.  The classification (`retrieve_sea_ice()`) handles this automatically.

7. **Summer classification requires `known_month`** — without the seasonal prior, `retrieve_sea_ice()` will misclassify summer bare ice as `FYI_snow` by exploiting physically impossible temperatures (T=−25°C in August).  Always pass `known_month` for summer observations.  See the SHEBA validation results in [SEA_ICE_EMULATOR.md](SEA_ICE_EMULATOR.md#validation-against-real-observations-sheba-spectra).

### Batch retrieval over scenes

`retrieve_sea_ice_batch()` parallelises `retrieve_sea_ice()` over `(N, bands)` pixel lists or `(H, W, bands)` images via joblib (install extras: `pip install biosnicar[geo]`):

```python
from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch

scene = retrieve_sea_ice_batch(
    image,                        # (H, W, 480) or (H, W, n_bands)
    platform="sentinel2",         # band mode — obs_uncertainty defaults
    observed_band_names=["B2", "B3", "B4", "B8", "B11"],
    solzen=62, direct=1, known_month=6,
    spatial_coords=latlon,        # (H, W, 2) for H3 export
    crs="EPSG:32633", transform=affine,   # for GeoTIFF export
    n_jobs=-1, chunksize=500,
)
scene.to_xarray()                 # xarray Dataset
scene.to_geotiff("scene.tif")     # surface_type_code, confidence, cost, flags
scene.to_h3_geojson("scene.geojson", resolution=9)
scene.to_netcdf("scene.nc")
print(scene.summary())
```

Non-finite pixels are skipped (`surface_type_code=255`).  Per-pixel L-BFGS-B is practical to ~100k pixels; the `SeaIceSceneResult` interface is engine-agnostic so a vectorised inverse network can replace the optimiser later without changing exports.

### MCMC through retrieve_sea_ice()

`retrieve_sea_ice(..., method="mcmc", mcmc_walkers=32, mcmc_steps=2000, mcmc_burn=500)` runs the emcee sampler per surface type and fills `uncertainty` with posterior standard deviations.  Expect minutes per observation (vs seconds for L-BFGS-B) — use on selected pixels for publication-grade uncertainty, not scene processing.  Chains: `result.all_fits[result.surface_type].chains`.

### `known_month` — seasonal physical priors

The `known_month` parameter (integer 1–12) adds Gaussian priors on temperature-related parameters to prevent emulators from reaching physically impossible temperature regimes for the given season.

```python
# Summer observation (May–September) — prevents T < -10°C solutions
result = retrieve_sea_ice(
    observed    = spectrum,
    wavelength_mask = mask_400_1000,
    solzen      = 60,
    direct      = 1,
    known_month = 8,      # August — applies T ~ (−4°C ± 3°C) prior
)
```

| Season | Months | `sea_ice_temperature` prior | `brine_volume_fraction` prior | Effect |
|---|---|---|---|---|
| Melt season | 5–9 | (−4°C, σ=3°C) | derived per type | Prevents T < −10°C; rules out unphysical FYI_snow solutions. **Excludes `young_ice` from the candidate fleet** (physically impossible in summer) |
| Deep winter | 11–3 | (−15°C, σ=8°C) | derived per type | Prevents near-melting temperatures in winter |
| Freeze-up | 10–2 | October gets (−12°C, σ=6°C); Nov–Feb keep the winter prior | — | Adds `ice_thickness_cm ~ (5 cm, σ=8 cm)` prior for young ice |
| Transitional | 4 | None | None | No prior applied |

Without `known_month`, the SHEBA summer classification accuracy is 0/16. The seasonal prior alone recovered 9/16 (2026-05 era); combined with the per-type classification band masks (VIS-only for bare-ice candidates) the **current suite achieves 16/16** — see `docs/sea_ice_validation.md` §6.3 (re-verified 2026-07-15).

The `brine_volume_fraction` prior is **derived per emulator** from the seasonal
temperature prior via Cox & Weeks at that emulator's reference salinity
(`vb_s_ref`: 6 psu for FYI_bare, 2 psu for MYI_bare) — brine volume scales with
salinity, so a single shared prior would systematically penalise MYI.

Caller-supplied `regularization` overrides these priors on a key-by-key basis.

> **MCMC note:** parameters optimised in log10(x+1) space are *sampled* in that
> space with uniform bounds, which implies a non-uniform (1/(x+1)) prior in
> linear space.  For most parameters the likelihood dominates; be aware of it
> when posteriors are weakly constrained.

> **Use 400–1000 nm only for summer bare ice classification.**  Adding SWIR data (1100–2000 nm) reduces accuracy from 100% to 42% because the white-ice SWIR signature overlaps with coarse snow, reintroducing the ambiguity that `known_month` eliminates in the VIS/NIR window.  Always restrict the `wavelength_mask` to 400–1000 nm for summer observations unless you have a specific reason to include SWIR.

### Sea ice worked examples

```python
# Spectral retrieval: melt pond depth
from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
emus = load_sea_ice_emulators()
result = retrieve(
    observed     = measured_pond_spectrum,
    parameters   = ["pond_depth"],
    emulator     = emus["FYI_pond"],
    fixed_params = {"sea_ice_temperature": -5.0, "black_carbon": 1200.0,
                    "solzen": 60, "direct": 1},
)
print(f"Pond depth = {result.best_fit['pond_depth']:.3f} ± "
      f"{result.uncertainty['pond_depth']:.3f} m")

# Full classification from field spectrometer
from biosnicar.sea_ice.retrieve import retrieve_sea_ice
result = retrieve_sea_ice(observed=measured_spectrum, solzen=55, direct=1)
print(f"Surface type: {result.surface_type} (confidence={result.confidence:.2f})")
print(f"Parameters:  {result.parameters}")

# Post-hoc T from brine_volume_fraction
from biosnicar.sea_ice.brine_volume import invert_brine_volume
from biosnicar.sea_ice.emulator_configs import FYI_BARE_S_REF
if result.surface_type == "FYI_bare":
    T = invert_brine_volume(result.parameters["brine_volume_fraction"], FYI_BARE_S_REF)
    print(f"Temperature (at S={FYI_BARE_S_REF} psu): {T:.1f}°C")

# Summer bare ice classification — always provide known_month
from biosnicar.sea_ice.retrieve import retrieve_sea_ice
import numpy as np
wavelengths_nm = np.arange(205, 4999, 10)
mask_400_1000  = (wavelengths_nm >= 400) & (wavelengths_nm <= 1000)

result_summer = retrieve_sea_ice(
    observed        = observed_spectrum,      # 480-band from field spectrometer
    wavelength_mask = mask_400_1000,          # only compare the observed window
    solzen          = 60,
    direct          = 1,
    known_month     = 8,                      # August: T prior (−4°C ± 3°C)
)
print(f"Surface type: {result_summer.surface_type}  confidence={result_summer.confidence:.2f}")
# Without known_month, summer bare ice often misclassifies as FYI_snow
# because the optimiser can fit August spectra with T=−25°C (physically impossible)

# MCMC for posterior on pond depth
result_mcmc = retrieve(
    observed      = measured_pond_spectrum,
    parameters    = ["pond_depth", "black_carbon"],
    emulator      = emus["FYI_pond"],
    method        = "mcmc",
    mcmc_walkers  = 32,
    mcmc_steps    = 2000,
    mcmc_burn     = 500,
    fixed_params  = {"sea_ice_temperature": -5.0, "solzen": 60, "direct": 1},
)
```

## See Also

- [docs/EMULATOR.md](EMULATOR.md) — emulator architecture and design
- [docs/SEA_ICE_EMULATOR.md](SEA_ICE_EMULATOR.md) — complete sea ice emulator reference
- [docs/METHODS.md](METHODS.md) — detailed technical methods (paper-quality)
- [examples/07_inversion_spectral.py](../examples/07_inversion_spectral.py) — SSA spectral retrieval (includes sea ice)
- [examples/08_inversion_satellite.py](../examples/08_inversion_satellite.py) — satellite band retrieval (includes sea ice)
- [examples/09_inversion_methods.py](../examples/09_inversion_methods.py) — optimiser comparison (includes sea ice)
- [examples/14_sea_ice_emulator.py](../examples/14_sea_ice_emulator.py) — sea ice emulator demonstrations
- [examples/15_sea_ice_inversion.py](../examples/15_sea_ice_inversion.py) — sea ice inversion demonstrations
