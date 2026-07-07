#!/usr/bin/env python3
"""End-to-end workflow: emulator -> synthetic observation -> retrieval -> validation.

This script demonstrates the complete BioSNICAR inversion pipeline from
start to finish:

  1. Load a pre-built emulator
  2. Generate a synthetic Sentinel-2 observation from the full forward model
  3. Retrieve SSA and impurity concentrations from the satellite bands
  4. Validate the retrieval against the known truth
  5. Use the retrieved parameters for downstream analysis

This is the recommended starting point for users who want to understand
the full workflow before adapting it to their own data.  Each step includes
detailed commentary explaining the rationale and practical considerations.
"""

import numpy as np

from biosnicar import DATA_DIR, run_emulator, run_model, to_platform
from biosnicar.emulator import Emulator
from biosnicar.inverse import retrieve
from biosnicar.inverse.result import _compute_ssa

PLOT = False

# ======================================================================
# Step 1: Load pre-built emulator
# ======================================================================
# The emulator is a neural-network surrogate that predicts 480-band
# spectral albedo in ~microseconds (vs ~50 ms for the full RT model).
# The pre-built default emulator at this path covers 8 parameters and
# was trained on 50,000 LHS samples of solid glacier ice (layer_type=1).
#
# For custom parameter ranges or different ice configurations, build your
# own emulator using Emulator.build() — see example 04.

print("Step 1: Load emulator\n")
emu = Emulator.load(DATA_DIR / "emulators" / "glacier_ice_8_param_default.npz")
print(f"  {emu!r}")

# ======================================================================
# Step 2: Define a "true" ice surface and generate synthetic observation
# ======================================================================
# In a real workflow, the observation comes from a satellite image or
# field spectrometer.  Here we define a "true" ice surface and generate
# a synthetic observation from the FULL FORWARD MODEL (not the emulator).
# This is the honest test — the retrieval must bridge the approximation
# gap between the emulator's MLP predictions and the true RT solver.

print("\nStep 2: Generate synthetic Sentinel-2 observation\n")

# Known conditions (not retrieved): observing geometry and dust.
# Fixing poorly-constrained parameters improves retrieval of the rest.
# Dust is fixed at 1000 ppb — it has very low spectral sensitivity at
# typical environmental concentrations (< ~5000 ppb), so including it
# as a free parameter would add noise without improving the fit.
fixed = {"solzen": 50, "direct": 1, "dust": 1000, "snow_algae": 0}

# The "true" physical parameters of the ice surface.  In a real
# application, these are unknown — they are what we want to retrieve.
true_params = {
    "rds": 1200,        # bubble effective radius (um)
    "rho": 700,         # bulk ice density (kg/m3)
    "black_carbon": 3000,     # black carbon concentration (ppb)
    "glacier_algae": 80000,   # glacier algae concentration (cells/mL)
}

# Compute the true SSA — this is the quantity the inversion actually
# constrains.  SSA = 3 * (1 - rho/917) / (rds_m * rho) [m² kg⁻¹].
# Different (rds, rho) pairs with the same SSA produce nearly identical
# spectra, so SSA is the physically meaningful retrieval target.
true_ssa = _compute_ssa(true_params["rds"], true_params["rho"])

# Run the full forward model to get a 480-band spectral albedo.
true_outputs = run_model(**true_params, **fixed, layer_type=1)
true_albedo = np.array(true_outputs.albedo, dtype=np.float64)

# Convolve the full spectrum to Sentinel-2 bands.  The flx_slr (solar
# flux spectrum) from the emulator is used for flux-weighted band
# averaging, which accounts for the fact that photons are not uniformly
# distributed across wavelength.
s2_true = to_platform(true_albedo, "sentinel2", flx_slr=emu.flx_slr)

# Select which Sentinel-2 bands to use for retrieval.  These span
# visible (B2-B4, sensitive to impurities), NIR (B8, sensitive to SSA),
# and SWIR (B11, sensitive to SSA).  This combination provides
# complementary information for resolving both impurities and ice
# optical properties.
band_names = ["B2", "B3", "B4", "B8", "B11"]
obs_values = np.array([getattr(s2_true, b) for b in band_names])

# Add realistic measurement noise to simulate real satellite data.
# The noise level (0.5% albedo units) is typical for Sentinel-2 L2A
# surface reflectance over bright targets like ice.
rng = np.random.RandomState(42)
noise_sigma = np.full(len(band_names), 0.005)
obs_noisy = obs_values + rng.normal(0, noise_sigma)
obs_noisy = np.clip(obs_noisy, 0, 1)

print("  True parameters:")
print(f"    SSA:            {true_ssa:.4f} m2/kg  (from rds={true_params['rds']}, rho={true_params['rho']})")
print(f"    black_carbon:   {true_params['black_carbon']}")
print(f"    glacier_algae:  {true_params['glacier_algae']}")
print(f"\n  S2 bands (noisy): {dict(zip(band_names, obs_noisy.round(4)))}")

# ======================================================================
# Step 3: Retrieve ice properties (SSA + impurities)
# ======================================================================
# This is the core retrieval step.  We pass:
#   - observed: the noisy satellite band values
#   - parameters: which quantities to retrieve (SSA + two impurities)
#   - emulator: the trained emulator for fast forward evaluation
#   - platform/observed_band_names: tell the cost function to work in
#     band space, not spectral space
#   - obs_uncertainty: per-band measurement noise for chi-squared weighting
#   - fixed_params: known parameters that are NOT retrieved
#
# The default optimiser (L-BFGS-B with DE pre-search) typically converges
# in ~1-2 seconds with the emulator.  For Bayesian uncertainty estimates,
# use method="mcmc" — see example 09.

print("\nStep 3: Retrieve ice properties\n")
result = retrieve(
    observed=obs_noisy,
    parameters=["ssa", "black_carbon", "glacier_algae"],
    emulator=emu,
    platform="sentinel2",
    observed_band_names=band_names,
    obs_uncertainty=noise_sigma,
    fixed_params=fixed,
)
print(result.summary())

# ======================================================================
# Step 4: Validate against truth
# ======================================================================
# In a real application you wouldn't have the true values, but for
# synthetic tests this comparison shows how well the retrieval works.
# The SSA error should be much smaller than separate rds/rho errors
# would be (~5.5% vs ~73% and ~33% respectively).

print("\n\nStep 4: Validation\n")

# Build truth dict in retrieval parameter space (SSA, not rds/rho).
true_retrieval = {
    "ssa": true_ssa,
    "black_carbon": true_params["black_carbon"],
    "glacier_algae": true_params["glacier_algae"],
}

print(
    f"  {'Parameter':25s} {'True':>10s} {'Retrieved':>10s} {'Error':>10s}"
)
print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10}")
for name in result.best_fit:
    true_val = true_retrieval[name]
    ret_val = result.best_fit[name]
    err = abs(ret_val - true_val)
    print(f"  {name:25s} {true_val:10.4f} {ret_val:10.4f} {err:10.4f}")

# The derived dict contains the internal (rds, rho) decomposition used
# by the emulator.  rho_ref is the reference density (arbitrary) and
# rds_internal is the bubble radius computed from SSA and rho_ref.
# These are NOT physical measurements — they exist only to satisfy the
# emulator's input requirements.
print(f"\n  Internal decomposition: {result.derived}")

# ======================================================================
# Step 5: Use retrieved parameters downstream
# ======================================================================
# The retrieved SSA and impurity concentrations can be used for further
# analysis: computing broadband albedo (BBA), generating spectra for
# climate models, or mapping spatial patterns.
#
# To call the emulator downstream, we need physical (rds, rho) values.
# The internal decomposition from result.derived provides these.

print("\n\nStep 5: Use retrieved spectrum downstream\n")

# Generate a full Outputs object from the retrieved parameters.
# This gives access to BBA, BBAVIS, BBANIR, and .to_platform() chaining.
emu_params = {
    "rds": result.derived["rds_internal"],
    "rho": result.derived["rho_ref"],
    "black_carbon": result.best_fit["black_carbon"],
    "glacier_algae": result.best_fit["glacier_algae"],
}
outputs = run_emulator(emu, **emu_params, **fixed)
print(f"  BBA (retrieved):     {outputs.BBA:.4f}")

# Compute true BBA for comparison (flux-weighted broadband albedo).
true_bba = float(np.sum(emu.flx_slr * true_albedo) / np.sum(emu.flx_slr))
print(f"  BBA (true):          {true_bba:.4f}")

# Convolve the retrieved spectrum to other platforms for multi-platform
# analysis or climate model coupling.
s2_ret = outputs.to_platform("sentinel2")
cesm_ret = outputs.to_platform("cesm2band")
print(f"  S2 NDSI (retrieved): {s2_ret.NDSI:.4f}")
print(f"  CESM vis (retr.):    {cesm_ret.vis:.4f}")
print(f"  CESM nir (retr.):    {cesm_ret.nir:.4f}")

# ======================================================================
# Optional: plot true vs retrieved spectrum and parameter recovery
# ======================================================================
if PLOT:
    import matplotlib.pyplot as plt

    wavelengths = np.arange(0.205, 4.999, 0.01)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: spectral comparison showing the 480-band true spectrum vs
    # the retrieved spectrum.  A good retrieval produces near-perfect
    # overlap in the visible-NIR range.
    ax1.plot(wavelengths, true_albedo, "k-", label="True", linewidth=1.5)
    ax1.plot(wavelengths, result.predicted_albedo, "r--", label="Retrieved (SSA)")
    ax1.set_xlabel("Wavelength (um)")
    ax1.set_ylabel("Albedo")
    ax1.set_xlim(0.2, 2.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("True vs retrieved spectrum (SSA mode)")
    ax1.legend()

    # Right: parameter recovery bar chart comparing true and retrieved
    # values.  Note the very different scales (SSA in m²/kg vs impurities
    # in ppb or cells/mL), so this is mainly useful for visual inspection.
    names = list(result.best_fit.keys())
    true_plot_vals = [true_retrieval[n] for n in names]
    ret_vals = [result.best_fit[n] for n in names]
    x = np.arange(len(names))
    width = 0.35
    ax2.bar(x - width / 2, true_plot_vals, width, label="True")
    ax2.bar(x + width / 2, ret_vals, width, label="Retrieved")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Value")
    ax2.set_title("Parameter recovery")
    ax2.legend()

    fig.tight_layout()
    plt.show()
