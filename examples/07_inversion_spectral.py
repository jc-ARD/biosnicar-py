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
from biosnicar import DATA_DIR

emu = Emulator.load(DATA_DIR / "emulators" / "glacier_ice_8_param_default.npz")
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
    load_sea_ice_emulators, SEA_ICE_EMULATOR_CONFIGS, FYI_BARE_S_REF
)
from biosnicar.sea_ice.brine_volume import compute_brine_volume, invert_brine_volume

si_emus = load_sea_ice_emulators()

# Seeded random test parameters — non-round values that demonstrably
# cannot coincide with "convenient" training samples.
_rng07 = np.random.default_rng(2025)
def _rand07(name):
    b = si_emus[name].bounds
    p = {k: float(_rng07.uniform((lo+hi)/2 - 0.4*(hi-lo), (lo+hi)/2 + 0.4*(hi-lo)))
         for k, (lo, hi) in b.items() if k not in ("solzen", "direct")}
    p["solzen"] = 60; p["direct"] = 1
    return p

def _si_fwd(type_name, params, solzen=60, direct=1):
    """Forward-model spectrum for a sea ice surface type.

    Using emu.predict() as the synthetic observation would give trivially
    perfect retrieval (the emulator fitting its own output, cost≈0).
    This function uses the full RT solver so residuals reflect real
    emulator approximation error — the honest test.
    """
    run_kw = SEA_ICE_EMULATOR_CONFIGS[type_name]["transform_fn"](
        {**params, "solzen": solzen, "direct": direct}
    )
    return np.array(run_model(**run_kw).albedo)

# ── Example SI-1: FYI bare ice — brine volume + bubble radius ──────────────
#
# True physical state: T=-8°C at S_ref=6 psu → Vb=0.040, bbl_radius=350 µm,
# BC=200 ppb.  We retrieve all four free parameters; solzen and direct are
# fixed from measurement metadata.
#
# Vb plays the same role as SSA: it is the single well-constrained quantity
# that captures the (T, S) contribution to spectral albedo.

print("\n── SI-1: Bare FYI — brine_volume_fraction + bubble_radius ──")

emu_bare         = si_emus["FYI_bare"]
true_params_bare = _rand07("FYI_bare")
TRUE_VB  = true_params_bare["brine_volume_fraction"]
TRUE_T_C = invert_brine_volume(TRUE_VB, FYI_BARE_S_REF)
obs_bare = _si_fwd("FYI_bare", true_params_bare)

_FIXED_KEYS = {"solzen", "direct"}
result_bare = retrieve(
    observed     = obs_bare,
    parameters   = [k for k in true_params_bare if k not in _FIXED_KEYS],
    emulator     = emu_bare,
    fixed_params = {k: true_params_bare[k] for k in _FIXED_KEYS},
)
print(f"  Converged: {result_bare.converged}  Cost: {result_bare.cost:.4e}")
print(f"  {'Parameter':30s}  {'True':>8}  {'Retrieved':>10}  {'σ':>8}")
for k, tv in true_params_bare.items():
    if k in _FIXED_KEYS:
        continue
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
TRUE_POND = _rand07("FYI_pond")
obs_pond  = _si_fwd("FYI_pond", TRUE_POND)

result_pond = retrieve(
    observed     = obs_pond,
    parameters   = ["pond_depth"],
    emulator     = emu_pond,
    fixed_params = {k: v for k, v in TRUE_POND.items() if k != "pond_depth"
                    and k in emu_pond.bounds},
)
print(f"  True depth = {TRUE_POND['pond_depth']:.4f} m  "
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
TRUE_SNOW = _rand07("FYI_snow")
obs_snow  = _si_fwd("FYI_snow", TRUE_SNOW)

result_snow = retrieve(
    observed     = obs_snow,
    parameters   = ["snow_depth", "snow_grain_radius"],
    emulator     = emu_snow,
    fixed_params = {k: v for k, v in TRUE_SNOW.items()
                    if k not in ("snow_depth", "snow_grain_radius")
                    and k in emu_snow.bounds},
)
print(f"  True snow_depth={TRUE_SNOW['snow_depth']:.4f} m  "
      f"Retrieved={result_snow.best_fit['snow_depth']:.4f} m  "
      f"Converged: {result_snow.converged}")
print(f"  True grain_radius={TRUE_SNOW['snow_grain_radius']:.1f} µm  "
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

if PLOT:
    import matplotlib.pyplot as plt

    # ── Sea ice: spectral retrieval figure ────────────────────────────────
    # 4 panels — one per sea ice retrieval example.  Each shows the observed
    # spectrum (solid) and the best-fit retrieved spectrum (dashed), with
    # residual shading.  The fourth panel shows all five emulators fitted
    # against the bare-ice observation, coloured by residual magnitude.

    fig_si, axes_si = plt.subplots(2, 2, figsize=(13, 8))
    fig_si.suptitle("Sea ice: spectral retrieval quality — obs (solid) vs retrieved (dashed)",
                    fontsize=10)

    # Panel 1: bare FYI (SI-1)
    ax = axes_si[0, 0]
    ax.plot(wavelengths, obs_bare, "k-", lw=1.5, label="Observed")
    ax.plot(wavelengths, result_bare.predicted_albedo, "--", color="#1b7837",
            lw=1.5, label=f"Retrieved (cost={result_bare.cost:.1e})")
    ax.fill_between(wavelengths,
                    result_bare.predicted_albedo, obs_bare, alpha=0.12, color="#1b7837")
    ax.axvline(0.7, color="gray", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    vb_ret = result_bare.best_fit["brine_volume_fraction"]
    bbl_ret = result_bare.best_fit["sea_ice_bubble_radius"]
    ax.set_title(f"SI-1: Bare FYI — Vb={vb_ret:.4f}, bbl={bbl_ret:.0f} µm\n"
                 f"(true Vb={TRUE_VB:.4f}, bbl=350 µm)")
    ax.set_ylabel("Spectral albedo"); ax.legend(fontsize=8)

    # Panel 2: melt pond (SI-2)
    ax = axes_si[0, 1]
    ax.plot(wavelengths, obs_pond, "k-", lw=1.5, label="Observed")
    ax.plot(wavelengths, result_pond.predicted_albedo, "--", color="#4575b4",
            lw=1.5, label=f"Retrieved (cost={result_pond.cost:.1e})")
    ax.fill_between(wavelengths,
                    result_pond.predicted_albedo, obs_pond, alpha=0.12, color="#4575b4")
    ax.axvline(0.7, color="gray", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    d_ret = result_pond.best_fit["pond_depth"]
    ax.set_title(f"SI-2: Melt pond — depth={d_ret:.4f} m\n"
                 f"(true depth={TRUE_POND['pond_depth']:.2f} m)")
    ax.legend(fontsize=8)

    # Panel 3: snow-covered FYI (SI-3)
    ax = axes_si[1, 0]
    ax.plot(wavelengths, obs_snow, "k-", lw=1.5, label="Observed")
    ax.plot(wavelengths, result_snow.predicted_albedo, "--", color="#6a3d9a",
            lw=1.5, label=f"Retrieved (cost={result_snow.cost:.1e})")
    ax.fill_between(wavelengths,
                    result_snow.predicted_albedo, obs_snow, alpha=0.12, color="#6a3d9a")
    ax.axvline(0.7, color="gray", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    sd_ret = result_snow.best_fit["snow_depth"]
    rds_ret = result_snow.best_fit["snow_grain_radius"]
    ax.set_title(f"SI-3: Snow-covered FYI — depth={sd_ret:.4f} m, rds={rds_ret:.0f} µm\n"
                 f"(true: {TRUE_SNOW['snow_depth']:.2f} m, {TRUE_SNOW['snow_grain_radius']:.0f} µm)")
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Spectral albedo"); ax.legend(fontsize=8)

    # Panel 4: retrieve_sea_ice() — residuals for all five emulators
    # The winning emulator (FYI_bare) has flat near-zero residuals;
    # the others show systematic misfit that reveals the wrong surface type.
    ax = axes_si[1, 1]
    type_col_si = {"FYI_bare": "#1b7837", "FYI_snow": "#4575b4",
                   "FYI_summer": "#d73027", "MYI_bare": "#762a83",
                   "FYI_pond": "#e08214"}
    for stype, fit in result_class.all_fits.items():
        resid = fit.predicted_albedo - obs_bare
        rmse  = float(np.sqrt(np.mean(resid ** 2)))
        winner = "★ " if stype == result_class.surface_type else ""
        lw     = 2.0 if stype == result_class.surface_type else 1.0
        ax.plot(wavelengths, resid, color=type_col_si[stype], lw=lw,
                label=f"{winner}{stype} RMSE={rmse:.4f}")
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0.7, color="gray", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlim(0.3, 2.5); ax.set_ylim(-0.6, 0.6)
    ax.set_title("SI-4: retrieve_sea_ice() residuals per emulator\n"
                 "(flat residual = correct surface type)")
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Predicted − Observed")
    ax.legend(fontsize=7)

    fig_si.tight_layout()
    plt.show()
