#!/usr/bin/env python3
"""Spectral retrieval with the inverse module.

This script demonstrates how to retrieve ice physical properties from a
480-band spectral albedo observation using BioSNICAR's ``retrieve()``
function.  It covers the recommended SSA-based retrieval workflow, the
alternative rds-based approach when density is known, wavelength masking,
uncertainty weighting, Gaussian regularisation, and result inspection.

The key insight motivating SSA retrieval is that the spectral shape of ice
in the NIR is controlled by specific surface area (SSA), not by bubble
radius (rds) or density (rho) individually.  Many (rds, rho) pairs produce
the same SSA and therefore the same spectrum — a many-to-one degeneracy.
Retrieving SSA directly eliminates this degeneracy and gives ~5.5% error
compared to ~73% for rds and ~33% for rho when retrieved separately.
"""

import numpy as np

from biosnicar import run_model
from biosnicar.emulator import Emulator
from biosnicar.inverse import retrieve
from biosnicar.inverse.result import _compute_ssa

PLOT = True

# ======================================================================
# Setup: load the emulator and generate a synthetic observation
# ======================================================================

# The pre-built default emulator covers 8 parameters (rds, rho,
# black_carbon, snow_algae, glacier_algae, dust, direct, solzen) and was
# trained on 50,000 LHS samples of solid ice (layer_type=1).  It predicts
# 480-band spectral albedo in ~microseconds.
emu = Emulator.load("data/emulators/glacier_ice_8_param_default.npz")
print(f"Emulator: {emu!r}\n")

# Parameters that are known a priori and will NOT be retrieved.
# - direct=1: clear-sky illumination (binary flag, cannot be optimised)
# - solzen=50: solar zenith angle in degrees
# - dust=1000: mineral dust concentration in ppb (fixed because dust has
#   very low spectral sensitivity at typical concentrations; see
#   docs/INVERSION.md for details)
fixed = {"dust": 1000, "snow_algae": 0, "solzen": 50, "direct": 1}

# Define a "true" ice surface.  We generate the observation from the FULL
# FORWARD MODEL (not the emulator) so that the retrieval must bridge the
# approximation gap between the emulator's MLP and the true RT solver.
# This is the honest test: in real use the observation comes from a field
# spectrometer, which also differs from the emulator.
true_params = dict(rds=1000, rho=600, black_carbon=5000, glacier_algae=100000)
true_outputs = run_model(**true_params, **fixed, layer_type=1)
observed = np.array(true_outputs.albedo, dtype=np.float64)

# Compute the true SSA from the known (rds, rho).  This is the quantity
# the inversion actually constrains — different (rds, rho) pairs with the
# same SSA produce nearly identical spectra.
#
# The formula is:  SSA = 3 * (1 - rho/917) / (rds_m * rho)  [m² kg⁻¹]
true_ssa = _compute_ssa(true_params["rds"], true_params["rho"])
print(f"True SSA: {true_ssa:.4f} m2/kg  (from rds={true_params['rds']}, rho={true_params['rho']})\n")

# The parameters we want to retrieve.  "ssa" replaces both "rds" and
# "rho" — the emulator decomposes SSA into (rds, rho) internally using
# a reference density (by default the midpoint of the emulator's rho
# training range).
ice_params = ["ssa", "black_carbon", "glacier_algae"]

# ======================================================================
# Example 1: Basic spectral retrieval with SSA
# ======================================================================
# This is the simplest and recommended workflow.  We pass the 480-band
# observed spectrum, tell the retriever which parameters are free, and
# provide the emulator and fixed parameters.  The default optimiser is
# a hybrid DE + L-BFGS-B method that combines global exploration with
# fast local convergence.

print("=== Example 1: SSA-based spectral retrieval ===\n")
result = retrieve(
    observed=observed,
    parameters=ice_params,
    emulator=emu,
    fixed_params=fixed,
)
print(result.summary())

# The result object contains the best-fit parameters, the predicted
# spectrum at the optimum, and convergence diagnostics.  For proper
# Bayesian uncertainty estimates, use method="mcmc" — see example 09.
print(f"\n  True SSA:      {true_ssa:.4f} m2/kg")
print(f"  Retrieved SSA: {result.best_fit['ssa']:.4f} m2/kg")
print(f"  SSA error:     {abs(result.best_fit['ssa'] - true_ssa):.4f} m2/kg")

# In SSA mode, result.derived contains the internal (rds, rho) decomposition
# used by the emulator.  These are NOT physical measurements — they depend
# on the arbitrary reference density.
print(f"  Internal decomposition: {result.derived}")

# ======================================================================
# Example 2: Alternative — fix density, retrieve rds directly
# ======================================================================
# When an independent density measurement is available (e.g. from an ice
# core or snow pit), the rds/rho degeneracy can be broken by fixing rho
# and retrieving rds directly.  This is the traditional approach and gives
# good results when rho is well constrained.

print("\n\n=== Example 2: Fix density, retrieve rds directly ===\n")
print("  When density is known (e.g. from in-situ measurement), you can")
print("  retrieve rds directly instead of SSA.\n")
result2 = retrieve(
    observed=observed,
    parameters=["rds", "black_carbon", "glacier_algae"],
    emulator=emu,
    fixed_params={**fixed, "rho": 600},  # density known from measurement
)
print(result2.summary())

# ======================================================================
# Example 3: Wavelength mask (exclude noisy regions)
# ======================================================================
# Real spectrometer data often has noisy or unreliable regions — for
# example, water vapour absorption at 1.38 and 1.87 um, or detector noise
# in the thermal region.  The wavelength_mask parameter lets you exclude
# specific bands from the cost function.  True = include, False = exclude.

print("\n\n=== Example 3: Wavelength mask ===\n")
wavelengths = np.arange(0.205, 4.999, 0.01)

# Use only the visible-NIR window (0.3-2.5 um), which contains the
# diagnostic spectral features.  Bands outside this range (UV below
# 0.3 um and thermal above 2.5 um) are excluded.
mask = (wavelengths >= 0.3) & (wavelengths <= 2.5)
print(f"  Using {mask.sum()} of 480 bands (0.3-2.5 um)")

result3 = retrieve(
    observed=observed,
    parameters=ice_params,
    emulator=emu,
    fixed_params=fixed,
    wavelength_mask=mask,
)
print(f"  Retrieved SSA: {result3.best_fit['ssa']:.4f} (true: {true_ssa:.4f})")

# ======================================================================
# Example 4: Observation uncertainty
# ======================================================================
# When measurement uncertainty is known (e.g. from instrument calibration
# or repeated measurements), passing it via obs_uncertainty enables
# chi-squared weighting: well-measured bands contribute more to the cost
# function, poorly-measured bands contribute less.  This is statistically
# optimal under Gaussian noise.
#
# Here we simulate a noisy observation by adding Gaussian noise to the
# clean spectrum, then retrieve with uncertainty weighting.

print("\n\n=== Example 4: Observation uncertainty weighting ===\n")
rng = np.random.RandomState(42)
noise_sigma = 0.02  # 2% albedo units of noise per band
noisy_obs = observed + rng.normal(0, noise_sigma, size=observed.shape)
noisy_obs = np.clip(noisy_obs, 0, 1)  # keep within physical bounds

# Per-band 1-sigma uncertainty (uniform here; could vary by band)
obs_unc = np.full(480, noise_sigma)
result4 = retrieve(
    observed=noisy_obs,
    parameters=ice_params,
    emulator=emu,
    fixed_params=fixed,
    obs_uncertainty=obs_unc,
)
print(f"  Retrieved with uncertainty weighting:")
for name in result4.best_fit:
    print(f"    {name:25s} = {result4.best_fit[name]:12.4f}")

# ======================================================================
# Example 5: Regularization (Gaussian priors)
# ======================================================================
# Regularisation adds a Gaussian penalty term to the cost function,
# encoding prior knowledge about parameter values.  The penalty is:
#
#   ((value - prior_mean) / prior_sigma)^2
#
# This pulls the retrieval toward the prior when the spectral data alone
# cannot distinguish between solutions.  It is especially useful for:
# - Constraining degenerate parameters
# - Incorporating independent measurements (e.g. SSA from snow pit data)
# - Stabilising retrievals from noisy observations

print("\n\n=== Example 5: Regularization with priors ===\n")
result5 = retrieve(
    observed=observed,
    parameters=ice_params,
    emulator=emu,
    fixed_params=fixed,
    regularization={
        "ssa": (true_ssa, 1.0),         # prior: true_ssa +/- 1.0 m2/kg
        "black_carbon": (3000, 5000),   # prior: 3000 +/- 5000 ppb (weak)
    },
)
print(f"  With regularization:")
for name in result5.best_fit:
    print(f"    {name:25s} = {result5.best_fit[name]:12.4f}")

# ======================================================================
# Example 6: Inspect result object
# ======================================================================
# The RetrievalResult object contains all the information you need for
# downstream analysis.  The most important fields are:
#
# - best_fit:         dict of {param_name: optimal_value}
# - uncertainty:      dict of {param_name: 1_sigma} (Hessian approx;
#                      use method="mcmc" for Bayesian uncertainties)
# - predicted_albedo: 480-band spectrum at the best-fit point
# - observed:         the input observation (for residual analysis)
# - converged:        whether the optimiser reported convergence
# - derived:          internal decomposition (SSA mode only)
# - cost:             final chi-squared value
# - n_function_evals: number of emulator calls

print("\n\n=== Example 6: Result object inspection ===\n")
print(f"  converged:        {result.converged}")
print(f"  method:           {result.method}")
print(f"  cost:             {result.cost:.6f}")
print(f"  n_function_evals: {result.n_function_evals}")
print(f"  predicted shape:  {result.predicted_albedo.shape}")
print(f"  observed shape:   {result.observed.shape}")
print(f"  derived:          {result.derived}")

# ======================================================================
# Optional: multi-scenario retrieval comparison plot
# ======================================================================
# Generate synthetic observations across a range of ice conditions — from
# clean coarse-grained ice (high SSA, low impurities) to heavily loaded
# fine-grained ice (low SSA, high impurities) — then retrieve each and
# plot observed vs retrieved spectra side by side.

if PLOT:
    import matplotlib.pyplot as plt

    scenarios = [
        {"label": "Clean, coarse ice",
         "rds": 3000, "rho": 500, "black_carbon": 100, "glacier_algae": 0},
        {"label": "Moderate BC",
         "rds": 1500, "rho": 600, "black_carbon": 2000, "glacier_algae": 0},
        {"label": "Algae-dominated",
         "rds": 1000, "rho": 700, "black_carbon": 500, "glacier_algae": 200000},
        {"label": "Dense, heavily loaded",
         "rds": 800, "rho": 850, "black_carbon": 4000, "glacier_algae": 300000},
    ]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    wavelengths = np.arange(0.205, 4.999, 0.01)

    fig, ax = plt.subplots(figsize=(10, 5))
    for scen, col in zip(scenarios, colors):
        label = scen.pop("label")
        obs_i = np.array(
            run_model(**scen, **fixed, layer_type=1).albedo, dtype=np.float64
        )
        res_i = retrieve(
            observed=obs_i,
            parameters=ice_params,
            emulator=emu,
            fixed_params=fixed,
        )
        true_ssa_i = _compute_ssa(scen["rds"], scen["rho"])
        ssa_err = abs(res_i.best_fit["ssa"] - true_ssa_i) / true_ssa_i * 100

        ax.plot(wavelengths, obs_i, "-", color=col, linewidth=1.5,
                label=f"{label} (obs)")
        ax.plot(wavelengths, res_i.predicted_albedo, "--", color=col,
                linewidth=1, label=f"{label} (ret, SSA err {ssa_err:.1f}%)")

    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Albedo")
    ax.set_xlim(0.2, 2.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("SSA retrieval across ice surface scenarios")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    plt.show()


# ======================================================================
# Sea ice spectral retrieval
# ======================================================================
#
# Sea ice inversion works on the same principle as glacier ice, but with
# a different primary parameter.  The analogue of SSA for sea ice is
# ``brine_volume_fraction`` (Vb).
#
# Why Vb, not (temperature, salinity)?
#   Temperature and salinity both control brine volume via Cox & Weeks
#   (1983), creating the same kind of degeneracy as (rds, rho) in glacier
#   ice: many (T, S) pairs produce the same Vb and therefore the same
#   spectrum.  Retrieving Vb directly eliminates this degeneracy and gives
#   well-constrained uncertainties.
#
# Post-hoc temperature recovery:
#   If bulk salinity is known (e.g. from a sea ice model or climatology),
#   temperature can be recovered exactly:
#     T = invert_brine_volume(Vb_retrieved, salinity_known)
#   The reference salinities FYI_BARE_S_REF=6 psu, MYI_BARE_S_REF=2 psu
#   are used inside the emulator transform functions (see SEA_ICE_EMULATOR.md).

print("\n" + "=" * 65)
print("SEA ICE SPECTRAL RETRIEVAL")
print("=" * 65)

from biosnicar.sea_ice.emulator_configs import (
    load_sea_ice_emulators, FYI_BARE_S_REF
)
from biosnicar.sea_ice.brine_volume import compute_brine_volume, invert_brine_volume

si_emus = load_sea_ice_emulators()

# ── Example SI-1: FYI bare ice — brine volume + bubble radius ──────────────
#
# True physical state: T=-8°C at S_ref=6 psu → Vb=0.040, bbl_radius=350 µm,
# BC=200 ppb.  We retrieve all four free parameters; solzen and direct are
# fixed from measurement metadata.
#
# Vb plays the same role as SSA: it is the single well-constrained quantity
# that captures the (T, S) contribution to spectral albedo.

print("\n── SI-1: Bare FYI — brine_volume_fraction + bubble_radius ──")

TRUE_T_C  = -8.0
TRUE_VB   = compute_brine_volume(FYI_BARE_S_REF, TRUE_T_C)
emu_bare  = si_emus["FYI_bare"]

true_params_bare = dict(
    brine_volume_fraction = TRUE_VB,
    sea_ice_bubble_radius = 350.0,
    black_carbon          = 200.0,
    rho_DL                = 860.0,
)
obs_bare = emu_bare.predict(**true_params_bare, solzen=60, direct=1)

result_bare = retrieve(
    observed     = obs_bare,
    parameters   = list(true_params_bare.keys()),
    emulator     = emu_bare,
    fixed_params = {"solzen": 60, "direct": 1},
)
print(f"  Converged: {result_bare.converged}  Cost: {result_bare.cost:.4e}")
print(f"  {'Parameter':30s}  {'True':>8}  {'Retrieved':>10}  {'σ':>8}")
for k, tv in true_params_bare.items():
    rv  = result_bare.best_fit[k]
    sig = result_bare.uncertainty.get(k, float("nan"))
    print(f"  {k:30s}  {tv:8.4f}  {rv:10.4f}  {sig:8.4f}")

# Post-hoc T recovery from Vb
T_rec = invert_brine_volume(result_bare.best_fit["brine_volume_fraction"],
                             FYI_BARE_S_REF)
print(f"  [T recovered from Vb]  true={TRUE_T_C:.1f}°C  retrieved={T_rec:.2f}°C")

# ── Example SI-2: Melt pond — retrieve pond depth ──────────────────────────
#
# Pond depth is the primary retrievable parameter for melt ponds.  The NIR
# window (700-1000 nm) is highly sensitive to depth via Beer-Lambert
# attenuation — albedo roughly halves every 5-7 cm of pond depth.

print("\n── SI-2: Melt pond — pond_depth retrieval ──")

emu_pond  = si_emus["FYI_pond"]
TRUE_POND = {"pond_depth": 0.18, "sea_ice_temperature": -5.0,
             "black_carbon": 1200.0}
obs_pond  = emu_pond.predict(**TRUE_POND, solzen=60, direct=1)

result_pond = retrieve(
    observed     = obs_pond,
    parameters   = ["pond_depth"],
    emulator     = emu_pond,
    fixed_params = {"sea_ice_temperature": -5.0, "black_carbon": 1200.0,
                    "solzen": 60, "direct": 1},
)
print(f"  True depth = {TRUE_POND['pond_depth']:.2f} m  "
      f"Retrieved = {result_pond.best_fit['pond_depth']:.4f} m  "
      f"σ = {result_pond.uncertainty.get('pond_depth', float('nan')):.4f} m  "
      f"Converged: {result_pond.converged}")

# ── Example SI-3: Snow-covered FYI — retrieve snow depth ──────────────────
#
# Snow depth is retrievable from spectral albedo when the snow is thin
# enough to transmit NIR through to the underlying ice.  Thick snow
# (>~20 cm) is optically opaque and depth becomes unconstrained from
# albedo alone; grain radius and BC dominate in that regime.

print("\n── SI-3: Snow-covered FYI — snow_depth retrieval ──")

emu_snow  = si_emus["FYI_snow"]
TRUE_SNOW = {"snow_depth": 0.09, "snow_grain_radius": 350.0,
             "sea_ice_temperature": -15.0, "black_carbon": 50.0}
obs_snow  = emu_snow.predict(**TRUE_SNOW, solzen=60, direct=1)

result_snow = retrieve(
    observed     = obs_snow,
    parameters   = ["snow_depth", "snow_grain_radius"],
    emulator     = emu_snow,
    fixed_params = {"sea_ice_temperature": -15.0, "black_carbon": 50.0,
                    "solzen": 60, "direct": 1},
)
print(f"  True snow_depth={TRUE_SNOW['snow_depth']:.2f} m  "
      f"Retrieved={result_snow.best_fit['snow_depth']:.4f} m  "
      f"Converged: {result_snow.converged}")
print(f"  True grain_radius={TRUE_SNOW['snow_grain_radius']:.0f} µm  "
      f"Retrieved={result_snow.best_fit['snow_grain_radius']:.1f} µm")

# ── Example SI-4: retrieve_sea_ice() — classify AND retrieve ──────────────
#
# When the surface type is not known in advance, retrieve_sea_ice() fits
# all five emulators simultaneously and returns the best-fit surface type
# together with its physical parameters.  This is the sea ice equivalent
# of fitting glacier ice with multiple scenarios.

print("\n── SI-4: retrieve_sea_ice() — surface classification + retrieval ──")

from biosnicar.sea_ice.retrieve import retrieve_sea_ice

# Use the bare-ice observation from Example SI-1
result_class = retrieve_sea_ice(
    observed = obs_bare,
    solzen   = 60,
    direct   = 1,
)
print(f"  Surface type  : {result_class.surface_type}")
print(f"  Confidence    : {result_class.confidence:.3f}")
print(f"  Parameters    :", {k: f"{v:.4f}" for k, v in result_class.parameters.items()})
print(f"  Cost per type :")
for stype, cost in sorted(result_class.cost_per_type.items(), key=lambda x: x[1]):
    marker = " ←" if stype == result_class.surface_type else ""
    print(f"    {stype:14s}  {cost:.4e}{marker}")

# to_outputs() gives a full Outputs object with .to_platform(), .BBA, etc.
out_si = result_class.to_outputs()
print(f"  BBA={out_si.BBA:.3f}  "
      f"to_platform('sentinel2').B8={out_si.to_platform('sentinel2').B8:.3f}")
