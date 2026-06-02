# Technical Methods: Emulator and Inversion Framework

This document describes the emulator architecture, inversion methodology, and band convolution system in BioSNICAR, with sufficient detail for a methods section of a scientific publication.

## 1. Forward Model Overview

BioSNICAR solves the two-stream radiative transfer (RT) equation for a multi-layer snow/ice column containing light-absorbing impurities (black carbon, mineral dust, snow algae, glacier algae). The model operates on a 480-band spectral grid from 0.205 to 4.995 um at 10 nm resolution. A single forward evaluation takes approximately 50 ms. Two RT solvers are available:

- **Adding-doubling** (default): numerically exact for the two-stream equations; iteratively doubles layers until the full column is resolved. Stable across the full parameter space but slower.
- **Toon**: tri-diagonal matrix formulation (Toon et al., 1989). Faster for multi-layer problems but can produce unphysical albedo values (outside [0, 1]) at extreme parameter combinations due to numerical instability at wavelengths with very high or very low single-scattering albedo.

The forward model accepts physical parameters including bubble effective radius (`rds`, um), ice density (`rho`, kg m^-3), solar zenith angle (`solzen`, degrees), sky condition (`direct`, binary: 0 = diffuse/cloudy, 1 = direct/clear), and impurity concentrations in the surface layer (black carbon in ppb; algae in cells mL^-1; dust in ppb).

### 1.1 Layer types

Each layer in the column is assigned a `layer_type` integer that selects the optical-property calculation path:

| `layer_type` | Description |
|---|---|
| 0 | Granular snow or ice — spherical, spheroidal, hexagonal plate, or Koch snowflake grains |
| 1 | Solid bubbly glacier ice — bulk ice medium with air inclusions; Fresnel reflection at the air-ice boundary |
| 2 | Solid bubbly glacier ice — same as 1 but without Fresnel correction |
| 3 | Mixed water/ice spheres — interspersed liquid water and ice grains |
| 4 | **Sea ice** — brine inclusions via Maxwell-Garnett effective medium (v0.1, see Section 1.2) |

### 1.2 Sea ice (layer_type = 4)

Sea ice contains brine inclusions: pockets of concentrated liquid seawater that remain liquid below 0 °C. Their presence modifies the complex refractive index of the ice and therefore its optical properties. The sea ice model (v0.1 MVP) uses a three-step pipeline:

**Step 1 — Brine volume fraction** (Cox & Weeks 1983).  
Given bulk salinity S (psu) and temperature T (°C), the brine volume fraction ν_b is:

```
F1(T) = −4.732 − 22.45T − 0.6397T² − 0.01074T³     (−22.9 ≤ T ≤ −2 °C)
ν_b   = (S × ρ_i) / (F1 − ρ_i × S × F2)
```

**Step 2 — Effective medium** (Maxwell-Garnett).  
The complex permittivity of pure ice ε_ice and of brine ε_brine are combined at volume fraction ν_b:

```
ε_eff = ε_ice × [ε_brine(1 + 2ν_b) + 2ε_ice(1 − ν_b)] /
                [ε_brine(1 − ν_b)  + ε_ice(2 + ν_b)]
```

The effective refractive index n_eff + ik_eff = √ε_eff determines the absorption coefficient of the ice+brine medium.

**Step 3 — Combined optical properties**.  
Absorption comes from the imaginary part of n_eff. Scattering comes from air bubbles, using the existing `bubbly_air.npz` LUT; the air volume fraction is inferred from the density balance. The result is (τ, ω, g) at all 480 bands, stored in a pre-computed LUT at `data/OP_data/480band/luts/sea_ice.npz` covering temperature (7 points), salinity (6 points), density (4 points), and bubble radius (4 points) — 672 combinations total.

Sea ice requires the **adding-doubling solver** (the default). The Toon solver does not handle the Fresnel air-ice interface required for sea ice.

Brine RI is computed from Rowe et al. (2020) liquid water data at 0 °C with linear salinity and temperature corrections (Quan & Fry 1995; Pegau et al. 1997).

Known MVP limitations: air bubble scattering uses the pure-ice LUT (~5% approximation); Maxwell-Garnett valid for ν_b < 0.3; brine RI calibrated at ~35 psu. See [docs/sea_ice.md](sea_ice.md) for full detail.

## 2. Neural Network Emulator

### 2.1 Motivation

Many applications require thousands to millions of forward model evaluations: gradient-based optimisation (~100-500 evaluations), global optimisation (~500-5000), and Bayesian posterior sampling via MCMC (~50,000-200,000). At 50 ms per evaluation, MCMC with 32 walkers and 2000 steps (plus 500 burn-in) would require ~53 minutes. An emulator that predicts in microseconds makes these workflows practical.

### 2.2 Architecture: PCA + MLP

The emulator uses a two-stage architecture:

1. **PCA compression**: The 480-band spectral output is projected onto a truncated principal component basis retaining 99.9% of the variance in the training spectra. This typically yields 8-15 PCA components depending on the spectral diversity of the training set. The MLP therefore predicts ~10 scalar coefficients rather than 480 spectral values.

2. **Multi-layer perceptron**: A fully-connected feedforward network maps scaled input parameters to PCA coefficients. The architecture is:

```
Input (n_params) -> [128] -> ReLU -> [128] -> ReLU -> [64] -> ReLU -> [n_pca] (linear)
```

The predicted PCA coefficients are projected back to the 480-band spectral domain via the PCA basis vectors and mean spectrum:

```
albedo = coefficients @ pca_components + pca_mean
```

The output is clipped to [0, 1] to ensure physical plausibility.

#### 2.2.1 Why PCA + MLP?

**PCA compression** serves three purposes:
- **Regularisation**: The MLP learns smooth spectral structure (the dominant modes of variation) rather than fitting noise or minor numerical artefacts in individual bands.
- **Dimensionality reduction**: Predicting ~10 PCA coefficients instead of 480 band values reduces the output layer size by ~50x, yielding a more compact network and faster training.
- **Spectral coherence**: Spectral albedo is highly correlated across adjacent bands. PCA captures this correlation structure explicitly, preventing the MLP from learning 480 independent values that might produce physically inconsistent spectral shapes.

**MLP** was chosen over alternatives based on the following considerations:

| Approach | Limitation for this application |
|---|---|
| Gaussian process (GP) | O(n^3) training and O(n) prediction. With 5000+ training points, training becomes impractical and prediction takes milliseconds — too slow for MCMC inner loops. Sparse GP approximations lose accuracy. |
| RBF interpolation | Dense O(n^2) memory and O(n^3) solve. Stores all training points, yielding large files. Prediction cost scales with training set size. |
| Cartesian grid interpolation | Scales as O(N^d) where d is the number of parameters. A 7-parameter emulator at 20 grid points per dimension would require 20^7 ≈ 1.3 billion forward runs — infeasible. |
| Polynomial regression | Cannot capture the nonlinear spectral features at absorption bands or the interaction effects between impurities and grain size. |
| Random forest / gradient boosting | Produces step-function approximations. The discontinuous output breaks gradient-based optimisers (L-BFGS-B), which require smooth cost surfaces. |

The MLP provides O(N) training scaling with training set size, microsecond inference via matrix multiplications, smooth differentiable output compatible with gradient-based optimisation, and compact storage (~100-200 KB as numpy arrays).

#### 2.2.2 Network size

The 128-128-64 architecture was chosen empirically:
- Shallower networks (e.g. 64-32) underfit spectral features near absorption bands, producing systematic errors of ~0.02 in albedo.
- Deeper or wider networks (e.g. 256-256-128) show marginal accuracy improvement (~0.1% relative) while tripling storage and inference time.
- ReLU activation is sufficient for the smooth physical relationships between parameters and albedo and avoids the computational overhead of exp/tanh evaluations.

### 2.3 Training Data Generation

#### 2.3.1 Latin hypercube sampling (LHS)

Training samples are generated via Latin hypercube sampling rather than random or grid sampling. LHS stratifies each parameter's marginal distribution into N equal intervals with exactly one sample per stratum, then independently shuffles the columns. This provides:
- Uniform marginal coverage of each parameter's range (unlike random sampling, which can leave gaps).
- Far fewer samples than a Cartesian grid: a 7-parameter emulator requires ~10,000-150,000 LHS samples vs ~10^9+ grid points.
- Space-filling properties that improve MLP generalisation.

The implementation avoids requiring `scipy.stats.qmc` (added in scipy 1.7) for broader compatibility.

#### 2.3.2 Parameter snapping

Several parameters require discretisation to match the forward model's lookup-table structure:
- **Bubble radius** (`rds`): Snapped to the nearest entry in the ice optical property LUT. The grid has step 5 for rds in [10, 100), step 10 for [100, 5000], and step 500 for (5000, 25000].
- **Solar zenith angle** (`solzen`): Rounded to the nearest integer and clamped to [1, 89] degrees, matching the available irradiance data keys.
- **Sky condition** (`direct`): Binary parameter snapped to {0, 1} during LHS generation via rounding. This ensures the MLP sees approximately equal numbers of clear-sky and cloudy samples.

#### 2.3.3 Unphysical spectrum filtering

The RT solver occasionally produces negative albedo values at pathological parameter combinations (e.g. very large grain radius combined with high impurity loading at wavelengths where the single-scattering albedo approaches 0 or 1). Including these unphysical spectra would distort the PCA basis and degrade MLP training. Spectra with any value outside [0, 1.01] are automatically excluded from training, with a warning reporting the number dropped. The 1% tolerance above unity accounts for minor numerical overshoot that is physically benign.

### 2.4 Input Scaling and MLP Training

Inputs are min-max scaled to [0, 1] using the parameter bounds as the scaling range. This ensures all parameters contribute equally to the MLP optimisation regardless of their physical units.

Training uses scikit-learn's `MLPRegressor` with:
- Adam optimiser (default in scikit-learn)
- Early stopping with 10% validation holdout, preventing overfitting
- Maximum 2000 iterations (typically converges in ~200-500)
- Fixed random seed for reproducibility

Scikit-learn was chosen for training because writing a correct MLP training loop (backpropagation, Adam momentum, early stopping, validation splitting) from scratch would be hundreds of error-prone lines. However, inference extracts the trained weights as plain numpy arrays. The forward pass is four matrix multiplications with ReLU activations:

```python
for W, b in zip(weights, biases):
    x = x @ W + b
    x = max(x, 0)  # ReLU, except on last layer
albedo = x @ pca_components + pca_mean
```

This means users who load a pre-built `.npz` file never need scikit-learn installed. Only numpy is required at inference time. PyTorch/TensorFlow were considered but rejected as ~500 MB+ dependencies that are excessive for a 3-layer MLP.

### 2.5 Serialisation Format

Emulators are saved as compressed numpy `.npz` archives containing:
- Weight matrices and bias vectors for each layer
- PCA components (n_pca x 480) and mean spectrum (480,)
- Input scaling arrays (min, max)
- Solar flux spectrum (480,) for downstream band convolution
- JSON metadata string with parameter names, bounds, training statistics, and build configuration

This format is version-independent (no pickle), safe to share (no arbitrary code execution), human-inspectable via `np.load()`, and compact (~100-200 KB for a 7-parameter emulator).

### 2.6 Verification

The `Emulator.verify()` method benchmarks emulator accuracy against the full forward model on held-out parameter sets generated via a separate LHS draw (different seed). It reports:
- Mean and maximum absolute spectral error across all bands and test points
- Broadband albedo (BBA) error statistics
- R^2 (coefficient of determination) over all predicted vs reference albedo values
- Per-point error distributions for identifying problematic regions of parameter space

Benchmark points that produce unphysical forward-model output are flagged and excluded from aggregate statistics (but retained in the per-point arrays for inspection).

### 2.7 Expected Accuracy

With 5000+ training samples, typical glacier ice emulators achieve:
- R^2 > 0.999
- Mean absolute spectral error < 0.005 albedo units
- Maximum BBA error < 0.01

Accuracy degrades at the edges of training bounds and at extreme parameter combinations (very high simultaneous impurity loading). Increasing n_samples mitigates both effects.

## 3. Band Convolution System

### 3.1 Purpose

The forward model and emulator produce 480-band spectral albedo. Satellite instruments measure broadband averages weighted by their spectral response functions (SRFs). The band convolution module converts between these representations, enabling both:
- Forward comparison: predicting what a satellite would observe given model parameters
- Inverse comparison: computing the cost function in the satellite's native band space

### 3.2 SRF-Weighted Convolution

For a sensor band with spectral response function S(lambda), the band-averaged albedo is:

```
alpha_band = Sum_i[ alpha(lambda_i) * S(lambda_i) * F(lambda_i) ] / Sum_i[ S(lambda_i) * F(lambda_i) ]
```

where F(lambda) is the spectral solar flux (W m^-2 per band). The flux weighting accounts for the fact that photons are not uniformly distributed across wavelength — bands where solar irradiance is high contribute more to the effective broadband albedo.

SRFs are stored as CSV files on the model's 480-band wavelength grid. If the SRF wavelength grid does not match the model grid exactly, values are linearly interpolated.

### 3.3 Interval Averaging

For climate model coupling (CESM, MAR, HadCM3), broadband albedo in fixed wavelength intervals (e.g. visible 0.3-0.7 um, near-infrared 0.7-5.0 um) is computed via flux-weighted averaging:

```
alpha_interval = Sum[ alpha(lambda_i) * F(lambda_i) ] / Sum[ F(lambda_i) ]    for lambda_i in [lo, hi)
```

### 3.4 Supported Platforms

| Platform | Bands | Type | Typical use |
|---|---|---|---|
| Sentinel-2 | B1-B12 (13 bands) | SRF convolution | High-resolution ice mapping |
| Sentinel-3 | Oa01-Oa21 (21 bands) | SRF convolution | Ocean and land colour |
| Landsat 8 | B1-B7 (7 bands) | SRF convolution | Long-term ice monitoring |
| MODIS | B1-B7 (7 bands) | SRF convolution | Global daily coverage |
| CESM (2-band) | vis, nir | Interval average | Earth system modelling |
| CESM-RRTMG | 14 shortwave bands | Interval average | Detailed radiation schemes |
| MAR | vis, nir | Interval average | Regional climate modelling |

### 3.5 Derived Indices

Band results include standard spectral indices computed from band ratios:
- **NDSI** (Normalised Difference Snow Index): (Green - SWIR) / (Green + SWIR)
- **NDWI** (Normalised Difference Water Index): (Green - NIR) / (Green + NIR)

These are available as attributes of the returned `BandResult` object where the platform defines the relevant bands.

## 4. Inversion Framework

### 4.1 Problem Formulation

Given an observed albedo spectrum (or satellite band values), the inversion estimates the physical parameters theta = (SSA, black_carbon, glacier_algae, ...) that best explain the observations. This is formulated as a least-squares minimisation:

```
theta* = argmin_theta  J(theta)
```

where the cost function J is a chi-squared statistic:

```
J(theta) = Sum_i [ (alpha_hat_i(theta) - alpha_obs_i)^2 / sigma_i^2 ] + Sum_k [ (theta_k - mu_k)^2 / sigma_prior,k^2 ]
```

The first term is the data misfit: the sum of squared normalised residuals between the predicted albedo alpha_hat (from the emulator or forward model) and the observed albedo alpha_obs. When per-element measurement uncertainties sigma_i are provided, they weight the residuals (well-measured bands contribute more). Without uncertainties, all elements are weighted equally (sigma = 1). For truly Gaussian errors (reasonable for calibrated spectrometer or satellite reflectance data), chi-squared is the statistically optimal cost function. Alternatives like absolute-deviation or Huber loss would only make sense if you expected heavy-tailed outliers in the observations, which isn't typical for these data.

The second term is an optional Tikhonov regularisation that encodes prior knowledge as Gaussian penalties on parameter values. This is useful for breaking parameter degeneracies when independent information is available (e.g. density from in-situ measurements).

### 4.2 Observation Modes

#### 4.2.1 Spectral mode

In spectral mode, the observed array contains 480 spectral albedo values (e.g. from a field spectrometer such as an ASD FieldSpec). The cost function compares all 480 bands simultaneously. An optional boolean wavelength mask can exclude noisy regions (e.g. water vapour absorption at 1.38 and 1.87 um).

This mode provides the maximum information content and can typically constrain 4-7 parameters simultaneously.

#### 4.2.2 Satellite band mode

In band mode, the observed array contains N broadband satellite values (typically 4-5 bands). The emulator predicts the full 480-band spectrum, which is convolved to the platform's bands via `to_platform()` before computing the cost. This avoids the ill-posed problem of reconstructing a continuous spectrum from broadband observations.

Band mode provides substantially less spectral information than spectral mode (5 broadband values vs 480 spectral values). Consequently, it cannot constrain as many free parameters. For robust band-mode retrieval:
- Fix well-constrained parameters (density, sky condition, solar zenith) via `fixed_params`
- Limit free parameters to 2-4 (typically grain radius plus 1-2 dominant impurities)
- Use `obs_uncertainty` to weight bands appropriately

### 4.3 Log-Space Parameterisation for Impurity Concentrations

Impurity concentrations (black carbon, snow algae, glacier algae, mineral dust) span orders of magnitude: from 0 to ~500,000 ppb or cells mL^-1. In linear parameter space, this creates a severely ill-conditioned optimisation landscape:

- The cost surface is extremely flat at high concentrations (the spectral signal saturates — doubling from 250,000 to 500,000 ppb produces negligible albedo change).
- The gradient provides useful descent direction only near the true value when it is low.
- Linear-space initial guesses placed at the midpoint of bounds (e.g. 250,000 for a [0, 500,000] range) are far from typical environmental values and lie in the flat region of the cost surface.

To address this, impurity parameters are optimised in log10(x + 1) space:

```
x_log = log10(x_linear + 1)
```

This transformation:
- Compresses the dynamic range: [0, 500,000] maps to [0, 5.7]
- Makes the cost surface approximately equally sensitive across orders of magnitude
- Places the log-space midpoint at ~2.85, corresponding to ~707 ppb in linear space — much closer to typical environmental concentrations than 250,000

The +1 offset ensures log10(0 + 1) = 0, so zero concentration maps to zero in log space (no singularity at the lower bound).

Uncertainties are propagated back to linear space via the chain rule:

```
sigma_linear ≈ (x + 1) * ln(10) * sigma_log10
```

User-facing inputs (bounds, x0, results) are always in linear space; the log transformation is applied and reversed internally.

### 4.4 SSA Parameterisation

#### 4.4.1 The rds/rho Degeneracy

The spectral albedo of bubbly ice in the NIR is controlled primarily by the specific surface area (SSA), defined for spherical air bubbles as:

```
SSA = 3 * phi / (r * rho)    [m² kg⁻¹]
```

where phi = 1 - rho/917 is the air volume fraction, r is the bubble radius in metres, and rho is the bulk ice density in kg m^-3.  SSA represents the total air-ice interface area per unit mass — the quantity that determines the scattering optical depth in the geometric-optics limit.

Many different (rds, rho) combinations produce the same SSA and therefore nearly identical spectra, creating a **many-to-one degeneracy**.  In (rds, rho) space, the cost surface has a continuous valley along curves of constant SSA.  Optimisers wander freely along this valley, producing large individual errors in rds and rho despite the underlying SSA being well constrained.

Quantitatively, in synthetic retrieval experiments (emulator-generated spectra, L-BFGS-B optimisation, 5 parameters):

| Metric          | Typical error  |
| --------------- | -------------- |
| rds (individual)| ~73%           |
| rho (individual)| ~33%           |
| **SSA (derived from rds, rho)** | **~5.5%** |

The large individual errors cancel in the SSA combination because the optimiser moves along the valley floor — rds increases while rho decreases (or vice versa) in a way that preserves SSA.

#### 4.4.2 SSA as a Free Parameter

To exploit this structure, the inversion supports retrieving SSA directly as a free parameter instead of rds and rho.  When `"ssa"` is in the `parameters` list:

1. **Optimisation in SSA space.**  The optimiser sees SSA as a single scalar parameter.  The 2D valley in (rds, rho) space collapses to a clean 1D minimum in SSA space.

2. **Internal decomposition.**  Each emulator evaluation requires physical (rds, rho) values.  The SSA forward wrapper decomposes SSA into (rds, rho) using a reference density rho_ref:

```
rds_um = 3 * (1 - rho_ref / 917) / (SSA * rho_ref) / 1e-6
```

The emulator is called with `rds = rds_um` and `rho = rho_ref`.

3. **Arbitrary reference density.**  The choice of rho_ref has minimal impact on the predicted spectrum because SSA dominates the spectral response.  Second-order effects (the asymmetry parameter g and wavelength-dependent extinction efficiency Q_sca depend weakly on the individual bubble radius) are small compared to the emulator's inherent approximation error (~0.005 albedo units).  The reference density defaults to the midpoint of the emulator's rho training range.

4. **Bounds from emulator.**  SSA bounds are auto-computed from the emulator's rds training range and rho_ref, ensuring the decomposed rds stays within the emulator's valid region:

```
SSA_max = _compute_ssa(rds_min, rho_ref)    # small bubbles → high SSA
SSA_min = _compute_ssa(rds_max, rho_ref)    # large bubbles → low SSA
```

5. **Log-space optimisation.**  SSA spans ~0.01 to ~300 m² kg^-1 (four orders of magnitude from dense ice with large bubbles to fresh snow), so it is included in the set of log-space-transformed parameters alongside impurity concentrations.

#### 4.4.3 Constraint: Mutual Exclusion with rds and rho

`"ssa"` cannot appear alongside `"rds"` or `"rho"` in the free parameter list.  This constraint is enforced with a `ValueError` because:

- SSA is a function of (rds, rho) — retrieving all three would introduce a degenerate third dimension.
- If rho is independently known (e.g. from ice core measurements), the user should fix rho via `fixed_params` and retrieve `rds` directly.  The degeneracy is broken by the external constraint, and rds is well-determined.
- If neither rds nor rho is independently known, SSA is the physically meaningful quantity to retrieve.

### 4.5 Optimisation Methods

#### 4.5.1 L-BFGS-B with Hybrid DE Pre-Search (Default)

The default optimisation uses a two-phase strategy:

**Phase 1: Differential evolution (DE) pre-search.** For problems with 2 or more free parameters, a quick DE run (maxiter=100, population size=10 per parameter) globally explores the parameter space. DE is a stochastic population-based method that maintains a population of candidate solutions and evolves them via mutation, crossover, and selection. It is effective at identifying the basin of attraction containing the global minimum without requiring gradient information. The pre-search typically requires ~1000-3000 function evaluations.

**Phase 2: L-BFGS-B polishing.** The DE result (if better than the default initial guess) seeds a quasi-Newton L-BFGS-B optimisation. L-BFGS-B approximates the inverse Hessian from recent gradient evaluations and performs line searches within box constraints. It converges quadratically near the minimum, typically requiring ~100-200 additional evaluations to reach machine-precision accuracy.

**Rationale for the hybrid approach.** Pure L-BFGS-B is fast but vulnerable to local minima. The ice albedo cost surface has a problematic structure: spectrally similar combinations of parameters (e.g. moderate grain size with high dust vs large grain size with no dust) create multiple local minima. In testing, multi-start L-BFGS-B with 50 random restarts all converged to the same suboptimal minimum (cost=0.00024, dust overestimated by 500x). The quick DE pre-search reliably identifies the global basin at modest computational cost (~1-2 seconds with the emulator).

Pure DE could be used alone, but the DE solution is typically accurate only to ~1-3 significant figures. L-BFGS-B polishing refines this to machine precision in ~100 evaluations.

Convergence criteria:
- L-BFGS-B: ftol = 1e-12 (relative function tolerance)
- Maximum 2000 iterations

#### 4.5.2 Nelder-Mead

The Nelder-Mead simplex method is a derivative-free optimiser that maintains a simplex of n+1 points in n-dimensional parameter space and evolves it via reflection, expansion, contraction, and shrinkage operations. It is more robust than L-BFGS-B when the cost surface is noisy (e.g. when using the direct forward model instead of the smooth emulator) or when the gradient is unreliable.

Since `scipy.optimize.minimize` does not natively support bounds for Nelder-Mead (bounds support was added in scipy 1.7, but the minimum supported version is 1.4), bounds are enforced via a penalty function that returns 1e20 for any evaluation outside the feasible region.

Convergence criteria: fatol = 1e-12, xatol = 1e-10.

#### 4.5.3 Differential Evolution (Standalone)

For problems where the cost surface is known to be multimodal or when no good initial guess is available, standalone DE provides a rigorous global search. The implementation uses scipy's `differential_evolution` with:
- maxiter = 1000
- tol = 1e-10
- Default population size and mutation/crossover strategies (scipy defaults)
- Fixed seed for reproducibility

DE is slower than L-BFGS-B (typically 500-5000 evaluations) but guarantees convergence to the global minimum given sufficient iterations.

#### 4.5.4 MCMC (Markov Chain Monte Carlo)

For full Bayesian uncertainty quantification, the `emcee` ensemble sampler (Foreman-Mackey et al., 2013) explores the posterior distribution. The implementation uses:

- **Uniform priors** within parameter bounds (improper prior outside bounds returns -infinity log-probability)
- **Log-likelihood**: -0.5 * J(theta), where J is the chi-squared cost
- **Ensemble sampler** with configurable number of walkers (default 32), steps (default 2000), and burn-in (default 500)
- **Initialisation**: Gaussian ball around the x0 initial guess with spread = 1% of parameter range, clipped to bounds

**Why emcee?** The affine-invariant ensemble sampler handles correlated parameters without requiring tuning of proposal distributions. This is important because ice parameters are often correlated (e.g. grain radius and density have opposing effects on NIR albedo). The ensemble approach also parallelises naturally across walkers.

Post-processing:
- Best-fit parameters: posterior median (more robust than MAP for skewed distributions)
- Uncertainty: posterior standard deviation
- Convergence diagnostic: mean acceptance fraction > 0.1 (crude but practical)
- Autocorrelation time: estimated when chains are long enough, used to assess effective sample size

The full chain array is returned for downstream analysis (corner plots, marginal distributions, parameter correlations).

### 4.6 Binary Parameter Handling

The `direct` parameter (sky condition: 0 = diffuse, 1 = direct beam) is a binary flag that cannot be meaningfully optimised by continuous methods. Attempting to minimise over a continuous [0, 1] range for a parameter that the forward model treats as a discrete switch produces an undefined gradient and pathological optimiser behaviour (the cost surface is flat everywhere except at the two valid values).

The inversion module explicitly forbids including binary parameters in the `parameters` list, raising a `ValueError` with guidance to pass them via `fixed_params` instead. This design choice reflects the physical reality: sky conditions are typically known from meteorological observations or assumed for a given scene.

### 4.7 Uncertainty Estimation

#### 4.7.1 Hessian-Based Uncertainty (Default)

After optimisation (L-BFGS-B, Nelder-Mead, or DE), parameter uncertainties are estimated from the Hessian of the cost function at the optimum. The procedure is:

1. **Finite-difference Hessian**: A four-point central difference scheme computes the second-derivative matrix:

```
H_ij = [ f(x+h_i+h_j) - f(x+h_i-h_j) - f(x-h_i+h_j) + f(x-h_i-h_j) ] / (4 * h_i * h_j)
```

where step sizes h_i = 1e-4 * (upper_bound_i - lower_bound_i). This requires 4n^2 cost function evaluations (negligible at microsecond emulator speed).

2. **Covariance matrix**: The inverse Hessian approximates the parameter covariance matrix under the Gaussian (quadratic) approximation to the cost surface near the minimum:

```
Cov(theta) ≈ H^-1
```

3. **1-sigma uncertainties**: sqrt(diag(Cov)) gives the marginal standard deviation for each parameter. Off-diagonal elements encode parameter correlations.

4. **Singular Hessian handling**: If the Hessian is singular (indicating an unconstrained parameter direction — the cost surface is flat along that axis), the uncertainty is reported as infinity. This typically occurs for parameters with minimal spectral sensitivity (e.g. dust at low concentrations, where the spectral signature is below detection threshold).

**Step size rationale**: h = 1e-4 * parameter range balances truncation error (h too large) against floating-point cancellation error (h too small). For typical parameter ranges of O(100-500,000), this gives steps of O(0.01-50), well above machine epsilon but small enough for the quadratic approximation to hold.

#### 4.7.2 Log-Space Uncertainty Propagation

For parameters optimised in log10(x + 1) space, the Hessian is computed in log space (where the cost surface is better conditioned) and then transformed to linear-space uncertainties via:

```
sigma_linear = (x_linear + 1) * ln(10) * sigma_log10
```

This first-order propagation follows from the derivative of the inverse transform: d/d(x_log) [10^x_log - 1] = ln(10) * 10^x_log = ln(10) * (x_linear + 1).

The consequence is that uncertainty scales with the retrieved value — a concentration of 100 ppb might have sigma = 50 ppb, while 10,000 ppb might have sigma = 5,000 ppb. This reflects the physical reality that spectral sensitivity to impurities is approximately proportional to concentration on a logarithmic scale.

### 4.8 Cost Function Details

#### 4.8.1 Spectral Cost

```
J_spectral = Sum_{i=1}^{480} [ (alpha_hat_i - alpha_obs_i)^2 / sigma_i^2 ]
```

When `obs_uncertainty` is not provided, sigma_i = 1 for all bands (unweighted least squares). When a `wavelength_mask` is provided, only bands where the mask is True contribute to the sum.

#### 4.8.2 Band Cost

```
J_band = Sum_{j=1}^{N_bands} [ (alpha_hat_j - alpha_obs_j)^2 / sigma_j^2 ]
```

where alpha_hat_j is obtained by SRF convolution (Section 3.2) of the predicted 480-band spectrum to the j-th satellite band. This formulation avoids the ill-posed problem of interpolating a continuous spectrum from broadband observations — the forward model always operates in full spectral resolution.

#### 4.8.3 Regularisation Term

When Gaussian priors are specified:

```
J_reg = Sum_k [ (theta_k - mu_k)^2 / sigma_prior,k^2 ]
```

This is added to either the spectral or band cost. The prior mean mu_k and standard deviation sigma_prior,k encode independent knowledge about parameter values. For example, if field density measurements give rho = 700 +/- 50 kg m^-3, setting `regularization={"rho": (700, 50)}` constrains the retrieval to be consistent with this observation.

### 4.9 Practical Considerations for Retrieval

#### 4.9.1 Parameter Sensitivity and Degeneracies

Not all parameters are equally well constrained by spectral or band observations:

- **SSA (specific surface area)**: The dominant control on NIR albedo shape. Very well constrained from both spectral and band observations. Retrieving SSA directly avoids the rds/rho degeneracy and yields ~5.5% error in synthetic tests vs ~73% for rds and ~33% for rho individually. **Recommended as the primary ice optical parameter** unless independent density measurements are available.
- **Grain radius (`rds`)**: Strong spectral signature in the NIR (0.8-2.5 um). Well constrained *only when rho is fixed*. When both rds and rho are free, they are degenerate (see Section 4.4.1).
- **Ice density (`rho`)**: Affects overall optical depth. Degenerate with rds (Section 4.4.1) and with layer thickness (`dz`). Best constrained when independent density measurements are available.
- **Black carbon**: Broadband visible darkening. Well constrained from spectral data. Partially degenerate with glacier algae at low concentrations (both darken the visible with somewhat similar spectral shapes).
- **Glacier algae**: Visible darkening with characteristic carotenoid and chlorophyll-a absorption features. Spectrally distinguishable from black carbon at 480-band resolution but not from 5-band satellite data.
- **Mineral dust**: Very flat spectral signature at typical environmental concentrations (< ~5000 ppb). The cost surface is essentially insensitive to dust concentration in this regime — the spectral effect of 100 vs 5000 ppb dust is smaller than model noise. Dust retrievals are unreliable unless concentrations are very high (> 10,000 ppb) or strong prior constraints are applied.

#### 4.9.2 Recommended Retrieval Configurations

For **spectral data** (field spectrometer, 480 bands): retrieve SSA plus up to 3 impurity parameters (black_carbon, glacier_algae, dust), though dust should be treated with caution. Fix `direct` and `solzen` from known observing conditions. If independent density measurements are available, fix rho and retrieve rds directly instead of SSA.

For **satellite bands** (Sentinel-2, Landsat 8, MODIS, 4-7 broadband values): fix dust and sky conditions. Retrieve 2-3 parameters maximum (typically SSA plus 1-2 dominant impurities). The limited spectral information from broadband observations cannot constrain more parameters than the number of observed bands, and in practice the information content is lower due to band correlations. SSA is particularly advantageous in band mode because it collapses the rds/rho degeneracy into a single well-constrained parameter.

#### 4.9.3 Computational Performance

| Configuration | Evaluations | Wall time (emulator) |
|---|---|---|
| L-BFGS-B, 3 params | ~1,500 (DE + polish) | ~0.5 s |
| L-BFGS-B, 5 params | ~3,000 (DE + polish) | ~2 s |
| Differential evolution, 5 params | ~5,000 | ~3 s |
| MCMC, 32 walkers x 2000 steps | ~64,000 | ~30 s |

All timings assume emulator-based forward evaluation (~microseconds per call). Direct forward model evaluation (50 ms per call) would increase these by ~3-4 orders of magnitude.

## 5. Implementation Details

### 5.1 Software Dependencies

| Component | Dependency | Version | Notes |
|---|---|---|---|
| Forward model | numpy, scipy | any modern | Core computation |
| Emulator inference | numpy only | any modern | No sklearn needed |
| Emulator training | scikit-learn | >= 1.0 | Build-time only |
| Inversion (default) | numpy, scipy | any modern | L-BFGS-B, NM, DE |
| Inversion (MCMC) | emcee | >= 3 | Optional |
| Band convolution | numpy | any modern | CSV-based SRFs |

### 5.2 Code Organisation

```
biosnicar/
  emulator.py          # Emulator class (build, predict, verify, save/load)
  inverse/
    optimize.py        # retrieve() dispatcher, log-space transform, hybrid DE+L-BFGS-B
    cost.py            # spectral_cost(), band_cost()
    result.py          # RetrievalResult dataclass
  bands/
    _core.py           # SRF loading, srf_convolve(), interval_average()
    __init__.py        # to_platform(), BandResult
  drivers/
    run_model.py       # Forward model entry point
    run_emulator.py    # Emulator wrapper returning Outputs object
```

### 5.3 Reproducibility

All stochastic operations use configurable random seeds:
- LHS sampling: `seed` parameter in `Emulator.build()` (default 42)
- MLP training: same seed via scikit-learn's `random_state`
- DE optimisation: fixed seed (42) in both pre-search and standalone modes
- MCMC initialisation: numpy default RNG (set `np.random.seed()` externally for reproducibility)
- Verification benchmarks: separate seed (default 123) to ensure independence from training

## 6. References

- Toon, O. B., McKay, C. P., Ackerman, T. P., & Santhanam, K. (1989). Rapid calculation of radiative heating rates and photodissociation rates in inhomogeneous multiple scattering atmospheres. *Journal of Geophysical Research*, 94(D13), 16287-16301.
- Foreman-Mackey, D., Hogg, D. W., Lang, D., & Goodman, J. (2013). emcee: The MCMC Hammer. *Publications of the Astronomical Society of the Pacific*, 125(925), 306-312.
- Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. *SIAM Journal on Scientific Computing*, 16(5), 1190-1208.
- Storn, R., & Price, K. (1997). Differential evolution — a simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11(4), 341-359.
