#!/usr/bin/env python3
"""Satellite band-mode retrieval.

This script demonstrates how to retrieve ice physical properties from
satellite band observations (e.g. Sentinel-2, Landsat 8, MODIS) using
BioSNICAR's ``retrieve()`` function with the ``platform`` keyword.

In band mode, the emulator predicts the full 480-band spectrum internally,
convolves it to the satellite's spectral response functions, and compares
to your broadband observations.  You never need to reconstruct a
continuous spectrum from satellite data.

IMPORTANT: Band-mode retrieval provides substantially less spectral
information than full spectral mode (typically 4-5 broadband values vs 480
spectral bands).  This limits the number of parameters that can be
reliably constrained.  Best practice is to:
  - Fix poorly-constrained parameters (dust, sky conditions) via fixed_params
  - Limit free parameters to 2-3 (typically SSA plus 1-2 impurities)
  - Use obs_uncertainty to weight bands according to measurement quality

SSA is the recommended retrieval metric because it collapses the rds/rho
degeneracy into a single well-constrained parameter — particularly
important when observations are limited to a few broadband values.
"""

import numpy as np

from biosnicar import run_model, to_platform
from biosnicar.emulator import Emulator
from biosnicar.inverse import retrieve
from biosnicar.inverse.result import _compute_ssa

PLOT = True

# ======================================================================
# Setup: load emulator and generate a synthetic satellite observation
# ======================================================================

# Load the pre-built 8-parameter glacier ice emulator.
emu = Emulator.load("data/emulators/glacier_ice_8_param_default.npz")

# Parameters that are known a priori and will NOT be retrieved.
# Dust is fixed at 1000 ppb because it has very low spectral sensitivity
# at typical environmental concentrations — the effect of 100 vs 5000 ppb
# is smaller than typical measurement noise (see docs/INVERSION.md).
fixed = {"solzen": 50, "direct": 1, "dust": 1000, "snow_algae": 0}

# Define the "true" ice surface.  We generate the observation from the
# FULL FORWARD MODEL (not the emulator) so that the retrieval is honest.
# In real use, the observation comes from a satellite image.
true_params = dict(rds=1000, rho=600, black_carbon=5000, glacier_algae=50000)
true_outputs = run_model(**true_params, **fixed, layer_type=1)
true_albedo = np.array(true_outputs.albedo, dtype=np.float64)

# Convolve the full spectrum to Sentinel-2 bands using the emulator's
# solar flux spectrum for flux weighting.  This is what a Sentinel-2
# pixel would measure over this ice surface.
s2_obs = to_platform(true_albedo, "sentinel2", flx_slr=emu.flx_slr)

# Compute true SSA for validation.
true_ssa = _compute_ssa(true_params["rds"], true_params["rho"])
print(f"True SSA: {true_ssa:.4f} m2/kg  (from rds={true_params['rds']}, rho={true_params['rho']})\n")

# SSA + impurities to retrieve.  SSA replaces separate rds/rho retrieval,
# which is especially important in band mode where the limited spectral
# information cannot resolve the rds/rho degeneracy.
ice_params = ["ssa", "black_carbon", "glacier_algae"]

# Select which Sentinel-2 bands to use.  These span visible through SWIR
# and capture both impurity absorption (VIS) and ice grain scattering (NIR/SWIR).
band_names = ["B2", "B3", "B4", "B8", "B11"]
obs_values = np.array([getattr(s2_obs, b) for b in band_names])
print("Synthetic Sentinel-2 observation:")
for name, val in zip(band_names, obs_values):
    print(f"  {name}: {val:.4f}")

# ======================================================================
# Example 1: Basic Sentinel-2 retrieval
# ======================================================================
# The simplest band-mode workflow.  Pass the band values, the platform
# name, and the band names.  The emulator handles the internal
# full-spectrum prediction and band convolution automatically.

print("\n=== Example 1: Sentinel-2 SSA retrieval ===\n")
result = retrieve(
    observed=obs_values,
    parameters=ice_params,
    emulator=emu,
    platform="sentinel2",
    observed_band_names=band_names,
    fixed_params=fixed,
)
print(result.summary())
print(f"\n  True SSA: {true_ssa:.4f},  Retrieved SSA: {result.best_fit['ssa']:.4f}")
print(f"  Internal decomposition: {result.derived}")

# ======================================================================
# Example 2: With measurement uncertainty
# ======================================================================
# Real satellite observations have per-band uncertainty from calibration,
# atmospheric correction, and sensor noise.  Passing obs_uncertainty
# enables chi-squared weighting: well-measured bands (low sigma) contribute
# more to the cost function.
#
# Typical per-band uncertainties for Sentinel-2 L2A surface reflectance:
# - VIS bands (B2-B4): ~0.02 (good calibration)
# - NIR (B8): ~0.03 (moderate)
# - SWIR (B11): ~0.05 (higher uncertainty from atmospheric water vapour)

print("\n\n=== Example 2: With measurement uncertainty ===\n")
obs_unc = np.array([0.02, 0.02, 0.02, 0.03, 0.05])  # per-band 1-sigma

result2 = retrieve(
    observed=obs_values,
    parameters=ice_params,
    emulator=emu,
    platform="sentinel2",
    observed_band_names=band_names,
    fixed_params=fixed,
    obs_uncertainty=obs_unc,
)
for name in result2.best_fit:
    print(f"  {name:25s} = {result2.best_fit[name]:10.4f}")

# ======================================================================
# Example 3: Fewer free parameters
# ======================================================================
# With only 5 broadband observations, information content is limited.
# Fixing additional parameters (here the impurity concentrations are
# already minimal) can improve the retrieval of the remaining ones.
# This example shows that even with 3 free parameters, band mode can
# produce reasonable results for SSA and dominant impurities.

print("\n\n=== Example 3: Retrieve SSA + impurities (dust fixed) ===\n")
result3 = retrieve(
    observed=obs_values,
    parameters=["ssa", "black_carbon", "glacier_algae"],
    emulator=emu,
    platform="sentinel2",
    observed_band_names=band_names,
    fixed_params=fixed,
)
print(f"  SSA:            {result3.best_fit['ssa']:10.4f}  (true: {true_ssa:.4f})")
print(f"  black_carbon:   {result3.best_fit['black_carbon']:10.1f}  (true: {true_params['black_carbon']:.0f})")
print(f"  glacier_algae:  {result3.best_fit['glacier_algae']:10.1f}  (true: {true_params['glacier_algae']:.0f})")

# ======================================================================
# Example 4: Landsat 8 retrieval
# ======================================================================
# The same workflow applies to any supported platform.  Landsat 8 has
# fewer spectral bands than Sentinel-2 but covers a similar wavelength
# range.  Note that different platforms have different band names (B2-B6
# for Landsat 8 OLI vs B2-B11 for Sentinel-2).

print("\n\n=== Example 4: Landsat 8 SSA retrieval ===\n")
l8_obs = to_platform(true_albedo, "landsat8", flx_slr=emu.flx_slr)
l8_band_names = ["B2", "B3", "B4", "B5", "B6"]
l8_values = np.array([getattr(l8_obs, b) for b in l8_band_names])

result4 = retrieve(
    observed=l8_values,
    parameters=ice_params,
    emulator=emu,
    platform="landsat8",
    observed_band_names=l8_band_names,
    fixed_params=fixed,
)
print(f"  Landsat 8 retrieved SSA: {result4.best_fit['ssa']:.4f} (true: {true_ssa:.4f})")
print(f"  Converged: {result4.converged}")

# ======================================================================
# Example 5: MODIS retrieval
# ======================================================================
# MODIS provides daily global coverage but has broader spectral bands
# and lower spatial resolution than Sentinel-2 or Landsat 8.  The same
# retrieval framework applies — the emulator handles the SRF convolution
# differences automatically.

print("\n\n=== Example 5: MODIS SSA retrieval ===\n")
modis_obs = to_platform(true_albedo, "modis", flx_slr=emu.flx_slr)
modis_band_names = ["B1", "B2", "B3", "B4"]
modis_values = np.array([getattr(modis_obs, b) for b in modis_band_names])

result5 = retrieve(
    observed=modis_values,
    parameters=ice_params,
    emulator=emu,
    platform="modis",
    observed_band_names=modis_band_names,
    fixed_params=fixed,
)
print(f"  MODIS retrieved SSA: {result5.best_fit['ssa']:.4f} (true: {true_ssa:.4f})")

# ======================================================================
# Optional: plot true spectrum vs retrieved spectra from different sensors
# ======================================================================
if PLOT:
    import matplotlib.pyplot as plt

    wavelengths = np.arange(0.205, 4.999, 0.01)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wavelengths, true_albedo, "k-", label="True spectrum", alpha=0.5)
    ax.plot(wavelengths, result.predicted_albedo, "r--", label="S2 retrieval")
    ax.plot(wavelengths, result4.predicted_albedo, "b:", label="L8 retrieval")
    for name, val in zip(band_names, obs_values):
        ax.axhline(val, color="green", alpha=0.2, linewidth=0.5)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Albedo")
    ax.set_xlim(0.2, 2.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("Band-mode SSA retrieval from S2 and L8 observations")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ======================================================================
# Sea ice satellite band-mode retrieval
# ======================================================================
#
# Sea ice parameters can also be retrieved from satellite band observations.
# The same constraints as for glacier ice apply: limited to 2-3 free
# parameters with 3-5 observed bands.  The recommended parameter to
# retrieve is ``brine_volume_fraction`` (the sea ice analogue of SSA),
# plus one or two surface properties (bubble_radius, black_carbon, etc.).
#
# retrieve_sea_ice() also accepts platform + observed_band_names arguments,
# enabling full surface-type classification directly from satellite bands.

print("\n" + "=" * 65)
print("SEA ICE SATELLITE BAND-MODE RETRIEVAL")
print("=" * 65)

from biosnicar.sea_ice.emulator_configs import (
    load_sea_ice_emulators, SEA_ICE_EMULATOR_CONFIGS, FYI_BARE_S_REF
)
from biosnicar.sea_ice.brine_volume import compute_brine_volume
from biosnicar.sea_ice.retrieve import retrieve_sea_ice

si_emus = load_sea_ice_emulators()


def _si_fwd08(type_name, params, solzen=60, direct=1):
    """Forward-model Outputs for a sea ice surface type (honest test obs)."""
    run_kw = SEA_ICE_EMULATOR_CONFIGS[type_name]["transform_fn"](
        {**params, "solzen": solzen, "direct": direct}
    )
    return run_model(**run_kw)

# ── Example SB-1: FYI bare ice from Sentinel-2 ────────────────────────────
#
# Retrieve bubble_radius (NIR scattering) and brine_volume_fraction
# (broadband level) from Sentinel-2 B3 (green), B8 (NIR), B11 (SWIR).
# The three bands provide good spectral leverage for these two parameters:
#   B3 (green, 559 nm)  — dominated by scattering; weak brine absorption
#   B8 (NIR, 835 nm)    — sensitive to bubble scattering and brine volume
#   B11 (SWIR, 1610 nm) — strong ice absorption; bubble-radius diagnostic
#
# rho_DL is fixed at its prior (850 kg/m³); it cannot be reliably
# recovered from 3 bands alone.  Use regularization if rho_DL matters.

print("\n── SB-1: FYI bare ice from Sentinel-2 (B3, B8, B11) ──")

emu_bare = si_emus["FYI_bare"]
# Seeded random test parameters — non-round values prevent any coincidental
# proximity to training samples and make the example independently reproducible.
_rng08 = np.random.default_rng(2025)
_b8    = emu_bare.bounds
true_si = {k: float(_rng08.uniform((lo+hi)/2 - 0.4*(hi-lo), (lo+hi)/2 + 0.4*(hi-lo)))
           for k, (lo, hi) in _b8.items() if k not in ("solzen", "direct")}
true_si["solzen"] = 60; true_si["direct"] = 1
TRUE_VB_S2 = true_si["brine_volume_fraction"]

out_true_si   = _si_fwd08("FYI_bare", true_si)   # forward model, not emulator
si_s2         = out_true_si.to_platform("sentinel2")
obs_si_s2     = np.array([si_s2.B3, si_s2.B8, si_s2.B11])
obs_unc_si_s2 = np.array([0.02, 0.02, 0.03])   # VIS / NIR / SWIR 1-sigma

result_si_s2 = retrieve(
    observed            = obs_si_s2,
    parameters          = ["brine_volume_fraction", "sea_ice_bubble_radius"],
    emulator            = emu_bare,
    platform            = "sentinel2",
    observed_band_names = ["B3", "B8", "B11"],
    obs_uncertainty     = obs_unc_si_s2,
    fixed_params        = {"black_carbon": true_si["black_carbon"],
                           "rho_DL": true_si["rho_DL"],
                           "solzen": 60, "direct": 1},
)
print(f"  True  Vb={TRUE_VB_S2:.4f}  bbl={true_si['sea_ice_bubble_radius']:.0f} µm")
print(f"  Retr. Vb={result_si_s2.best_fit['brine_volume_fraction']:.4f}  "
      f"bbl={result_si_s2.best_fit['sea_ice_bubble_radius']:.1f} µm  "
      f"Converged: {result_si_s2.converged}")

# ── Example SB-2: Melt pond depth from Sentinel-2 ─────────────────────────
#
# Pond depth is particularly well-constrained in band mode because the NIR
# (B8) and SWIR (B11) bands are sensitive to water column absorption, which
# follows Beer-Lambert with a characteristic depth scale of ~5-7 cm.
# Even two bands (B3+B8) are sufficient to constrain pond depth reliably.

print("\n── SB-2: Melt pond depth from Sentinel-2 (B3, B8) ──")

emu_pond    = si_emus["FYI_pond"]
_b8p        = emu_pond.bounds
true_pond   = {k: float(_rng08.uniform((lo+hi)/2 - 0.4*(hi-lo), (lo+hi)/2 + 0.4*(hi-lo)))
               for k, (lo, hi) in _b8p.items() if k not in ("solzen", "direct")}
true_pond["solzen"] = 60; true_pond["direct"] = 1
out_pond    = _si_fwd08("FYI_pond", true_pond)   # forward model, not emulator
pond_s2     = out_pond.to_platform("sentinel2")
obs_pond_s2 = np.array([pond_s2.B3, pond_s2.B8])

result_pond_s2 = retrieve(
    observed            = obs_pond_s2,
    parameters          = ["pond_depth"],
    emulator            = emu_pond,
    platform            = "sentinel2",
    observed_band_names = ["B3", "B8"],
    obs_uncertainty     = np.array([0.02, 0.02]),
    fixed_params        = {k: v for k, v in true_pond.items()
                           if k != "pond_depth"},
)
print(f"  True depth = {true_pond['pond_depth']:.2f} m  "
      f"Retrieved = {result_pond_s2.best_fit['pond_depth']:.4f} m  "
      f"Converged: {result_pond_s2.converged}")

# ── Example SB-3: retrieve_sea_ice() with satellite bands ─────────────────
#
# retrieve_sea_ice() also accepts platform + observed_band_names.  It fits
# all five surface-type emulators against the same band observations and
# classifies the surface by lowest chi-squared residual.
# This is the most powerful single-call workflow: surface classification
# and physical parameter retrieval in one step, directly from satellite
# observations without a prior assumption of surface type.

print("\n── SB-3: retrieve_sea_ice() surface classification from Sentinel-2 ──")

# Use the bare-ice S2 observation from SB-1
result_classify_s2 = retrieve_sea_ice(
    observed            = obs_si_s2,
    platform            = "sentinel2",
    observed_band_names = ["B3", "B8", "B11"],
    obs_uncertainty     = obs_unc_si_s2,
    solzen              = 60,
    direct              = 1,
)
print(f"  Surface type  : {result_classify_s2.surface_type}  "
      f"(confidence={result_classify_s2.confidence:.3f})")
print(f"  Ranked costs  :")
for stype, cost in sorted(result_classify_s2.cost_per_type.items(), key=lambda x: x[1]):
    marker = " ←" if stype == result_classify_s2.surface_type else ""
    print(f"    {stype:14s}  {cost:.4f}{marker}")

# Full Outputs compatibility — to_platform() works on the classification result
print(f"  to_platform().B8 = "
      f"{result_classify_s2.to_outputs().to_platform('sentinel2').B8:.3f}")

if PLOT:
    import matplotlib.pyplot as plt
    from biosnicar.bands import to_platform as _to_platform_fn

    wavelengths = np.arange(0.205, 4.999, 0.01)

    # ── Sea ice satellite band-mode figure ────────────────────────────────
    # 3 panels: SB-1 (bare FYI from S2), SB-2 (pond from S2),
    # SB-3 (classification cost per type for each input).

    fig_sib, axes_sib = plt.subplots(1, 3, figsize=(16, 4.5))
    fig_sib.suptitle("Sea ice: satellite band-mode retrieval (Sentinel-2)",
                     fontsize=10)
    s2_band_wl = {"B3": 0.559, "B8": 0.835, "B11": 1.610}

    # Panel 1: SB-1 bare FYI — observed bands + retrieved full spectrum
    ax = axes_sib[0]
    out_retr_s2 = result_si_s2.to_outputs()
    ax.plot(wavelengths, out_true_si.albedo, "k-", lw=1.5,
            label="True spectrum")
    ax.plot(wavelengths, out_retr_s2.albedo, "--", color="#1b7837",
            lw=1.5, label="Retrieved spectrum")
    for bname, bwl in s2_band_wl.items():
        obs_val = obs_si_s2[list(s2_band_wl.keys()).index(bname)]
        ax.plot(bwl, obs_val, "o", color="#d95f02", ms=8, zorder=5)
    ax.plot([], [], "o", color="#d95f02", ms=8, label="S2 observed bands")
    ax.axvline(0.7, color="gray", lw=0.5, ls=":", alpha=0.4)
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    vb_s2 = result_si_s2.best_fit["brine_volume_fraction"]
    bbl_s2 = result_si_s2.best_fit["sea_ice_bubble_radius"]
    ax.set_title(f"SB-1: FYI bare ice — S2 (B3, B8, B11)\n"
                 f"Retrieved Vb={vb_s2:.4f}, bbl={bbl_s2:.0f} µm")
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Spectral albedo")
    ax.legend(fontsize=8)

    # Panel 2: SB-2 melt pond — observed bands + retrieved spectrum
    ax = axes_sib[1]
    out_retr_pond = result_pond_s2.to_outputs()
    ax.plot(wavelengths, out_pond.albedo, "k-", lw=1.5,
            label="True spectrum")
    ax.plot(wavelengths, out_retr_pond.albedo, "--", color="#4575b4",
            lw=1.5, label="Retrieved spectrum")
    for bname, bwl in {"B3": 0.559, "B8": 0.835}.items():
        obs_val = obs_pond_s2[list({"B3": 0.559, "B8": 0.835}.keys()).index(bname)]
        ax.plot(bwl, obs_val, "o", color="#d95f02", ms=8, zorder=5)
    ax.plot([], [], "o", color="#d95f02", ms=8, label="S2 observed bands")
    ax.axvline(0.7, color="gray", lw=0.5, ls=":", alpha=0.4)
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    d_pond_ret = result_pond_s2.best_fit["pond_depth"]
    ax.set_title(f"SB-2: Melt pond — S2 (B3, B8)\n"
                 f"Retrieved depth={d_pond_ret:.4f} m  (true={true_pond['pond_depth']:.2f} m)")
    ax.set_xlabel("Wavelength (µm)"); ax.legend(fontsize=8)

    # Panel 3: SB-3 classification — cost per type for each of the two inputs
    ax = axes_sib[2]
    types_sorted = sorted(result_classify_s2.cost_per_type,
                          key=lambda k: result_classify_s2.cost_per_type[k])
    x = np.arange(len(types_sorted))
    w = 0.35
    costs_bare  = [result_classify_s2.cost_per_type[t] for t in types_sorted]
    # Run classification on the pond obs too for comparison
    result_pond_cls = retrieve_sea_ice(
        observed=obs_si_s2[:2],   # B3, B8 only
        platform="sentinel2",
        observed_band_names=["B3", "B8"],
        obs_uncertainty=np.array([0.02, 0.02]),
        solzen=60, direct=1,
    )
    costs_pond = [result_pond_cls.cost_per_type.get(t, float("nan"))
                  for t in types_sorted]
    bars1 = ax.bar(x - w/2, costs_bare,  w, color="#1b7837", alpha=0.8,
                   label="Bare ice input")
    bars2 = ax.bar(x + w/2, costs_pond, w, color="#4575b4", alpha=0.8,
                   label="Pond input")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(types_sorted, rotation=20, fontsize=8)
    ax.set_ylabel("Chi-squared residual (log scale)")
    ax.set_title("SB-3: Classification costs (S2 band mode)\n"
                 "Lower = better fit for that surface type")
    ax.legend(fontsize=8)

    fig_sib.tight_layout()
    plt.show()
