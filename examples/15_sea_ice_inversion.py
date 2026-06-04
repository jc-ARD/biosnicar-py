#!/usr/bin/env python3
"""Sea ice inversion — retrieve physical parameters from spectral albedo.

This example demonstrates:
  1. Synthetic retrieval: generate a spectrum, invert it, check accuracy
  2. retrieve_sea_ice(): fit all surface types and classify automatically
  3. Satellite band mode: retrieve from Sentinel-2 band albedos
  4. Uncertainty estimation with the Hessian and MCMC methods
  5. The to_outputs() / to_platform() chain on a retrieval result

Prerequisites
-------------
Build the emulators first:

    python scripts/build_sea_ice_emulators.py

Or use fast builds for testing:

    python scripts/build_sea_ice_emulators.py --fast
"""

import matplotlib.pyplot as plt
import numpy as np

from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
from biosnicar.sea_ice.retrieve import retrieve_sea_ice
from biosnicar.inverse import retrieve

PLOT = True
WAVELENGTHS = np.arange(0.205, 4.999, 0.01)


# ── Load emulators ───────────────────────────────────────────────────────────

print("Loading sea ice emulators…")
emulators = load_sea_ice_emulators()


# ── 1. Synthetic spectral retrieval — bare FYI ───────────────────────────────

print("\n── 1. Bare FYI synthetic retrieval ──")

emu_bare = emulators["FYI_bare"]

# True parameters — brine_volume_fraction replaces the degenerate (T, S) pair.
# Vb=0.041 corresponds to ~T=-8°C at S_ref=6 psu (FYI_BARE_S_REF).
# Post-hoc T recovery: invert_brine_volume(Vb, S_ref=6) → T
from biosnicar.sea_ice.brine_volume import compute_brine_volume, invert_brine_volume
from biosnicar.sea_ice.emulator_configs import FYI_BARE_S_REF
TRUE_T_C = -8.0
TRUE_VB  = compute_brine_volume(FYI_BARE_S_REF, TRUE_T_C)

TRUE = dict(
    brine_volume_fraction = TRUE_VB,
    sea_ice_bubble_radius = 350.0,
    black_carbon          = 200.0,
    rho_DL                = 860.0,
)

# Generate synthetic observed spectrum
obs = emu_bare.predict(**TRUE, solzen=60, direct=1)

# Invert: retrieve all four physical parameters (fix solzen and direct)
result = retrieve(
    observed      = obs,
    parameters    = list(TRUE.keys()),
    emulator      = emu_bare,
    fixed_params  = {"solzen": 60, "direct": 1},
    method        = "L-BFGS-B",
)

# Post-hoc T recovery from retrieved Vb
Vb_ret = result.best_fit["brine_volume_fraction"]
T_ret  = invert_brine_volume(Vb_ret, FYI_BARE_S_REF)

print(f"  Converged: {result.converged}  Cost: {result.cost:.4f}")
print(f"  {'Parameter':28s}  {'True':>9}  {'Retrieved':>10}  {'Error':>8}  {'σ':>8}")
print("  " + "-"*64)
for param, true_val in TRUE.items():
    ret_val = result.best_fit[param]
    unc     = result.uncertainty.get(param, float("nan"))
    err     = ret_val - true_val
    print(f"  {param:28s}  {true_val:>9.4f}  {ret_val:>10.4f}  {err:>+8.4f}  {unc:>8.4f}")
print(f"  {'[T recovered from Vb]':28s}  {TRUE_T_C:>9.2f}  {T_ret:>10.2f}  {T_ret-TRUE_T_C:>+8.2f}°C")


# ── 2. retrieve_sea_ice() — classify surface type ────────────────────────────

print("\n── 2. Surface-type classification ──")

# Test with three different surface types
test_cases = {
    "bare_ice":  emulators["FYI_bare"].predict(
        brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0,
        black_carbon=0.0, rho_DL=850.0, solzen=60, direct=1,
    ),
    "snow":      emulators["FYI_snow"].predict(
        snow_depth=0.15, snow_grain_radius=300.0,
        sea_ice_temperature=-15.0, black_carbon=0.0,
        solzen=60, direct=1,
    ),
    "melt_pond": emulators["FYI_pond"].predict(
        pond_depth=0.20, sea_ice_temperature=-5.0,
        black_carbon=1200.0, solzen=60, direct=1,
    ),
}

for true_label, spectrum in test_cases.items():
    r = retrieve_sea_ice(
        observed=spectrum,
        emulators=emulators,
        solzen=60,
        direct=1,
    )
    match = (true_label == "bare_ice" and r.surface_type == "FYI_bare") or \
            (true_label == "snow" and "snow" in r.surface_type.lower()) or \
            (true_label == "melt_pond" and "pond" in r.surface_type.lower())
    correct = "✓" if match else "?"
    print(f"  True: {true_label:12s}  →  Retrieved: {r.surface_type:14s}  "
          f"confidence={r.confidence:.3f}  {correct}")


# ── 3. Full result from retrieve_sea_ice() ───────────────────────────────────

print("\n── 3. Full retrieve_sea_ice() result ──")

r = retrieve_sea_ice(
    observed     = test_cases["bare_ice"],
    emulators    = emulators,
    solzen       = 60,
    direct       = 1,
)

print(r.summary())

# to_outputs() → to_platform() chain
out   = r.to_outputs()
bands = out.to_platform("sentinel2")
print(f"\n  Sentinel-2 bands: B3={bands.B3:.3f}  B8={bands.B8:.3f}  B11={bands.B11:.3f}")


# ── 4. Satellite band mode retrieval ─────────────────────────────────────────

print("\n── 4. Satellite band retrieval (Sentinel-2) ──")

# Generate synthetic S2 observations from a known sea ice state
# Vb=0.060 ≈ T=-5°C at S_ref=6 psu
TRUE_VB_S2 = compute_brine_volume(FYI_BARE_S_REF, -5.0)
true_state = dict(
    brine_volume_fraction=TRUE_VB_S2, sea_ice_bubble_radius=250.0,
    black_carbon=500.0, rho_DL=855.0,
)
from biosnicar.drivers.run_emulator import run_emulator
out_true  = run_emulator(emu_bare, **true_state, solzen=60, direct=1)
obs_bands = out_true.to_platform("sentinel2")
# Use B3 (green), B8 (NIR), B11 (SWIR) — spectrally diverse
obs_s2    = np.array([obs_bands.B3, obs_bands.B8, obs_bands.B11])

result_s2 = retrieve(
    observed             = obs_s2,
    parameters           = ["sea_ice_bubble_radius", "black_carbon"],
    emulator             = emu_bare,
    platform             = "sentinel2",
    observed_band_names  = ["B3", "B8", "B11"],
    obs_uncertainty      = np.array([0.02, 0.02, 0.03]),
    fixed_params         = {
        "brine_volume_fraction": TRUE_VB_S2,
        "rho_DL": 855.0, "solzen": 60, "direct": 1,
    },
)

print(f"  True  Vb={true_state['brine_volume_fraction']:.4f}  bubble_radius={true_state['sea_ice_bubble_radius']:.0f} µm   "
      f"black_carbon={true_state['black_carbon']:.0f} ppb")
print(f"  Retr. bubble_radius={result_s2.best_fit['sea_ice_bubble_radius']:.0f} µm   "
      f"black_carbon={result_s2.best_fit['black_carbon']:.0f} ppb")
print(f"  Converged: {result_s2.converged}  Cost: {result_s2.cost:.4f}")

# to_platform() from satellite retrieval result
pred_out   = result_s2.to_outputs()
pred_bands = pred_out.to_platform("sentinel2")
print(f"  Predicted S2 bands: B3={pred_bands.B3:.3f}  B8={pred_bands.B8:.3f}  "
      f"B11={pred_bands.B11:.3f}")


# ── 5. Plots ──────────────────────────────────────────────────────────────────

if PLOT:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Panel 1: synthetic retrieval fit quality
    ax = axes[0]
    pred = result.predicted_albedo
    ax.plot(WAVELENGTHS, obs, "k-", lw=1.5, label="Observed (synthetic truth)")
    ax.plot(WAVELENGTHS, pred, "--", color="#e07b39", lw=1.5,
            label=f"Retrieved  (cost={result.cost:.3f})")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    ax.axvline(0.7, color="k", lw=0.5, ls=":", alpha=0.4)
    ax.set_title("FYI bare ice: synthetic retrieval\n"
                 "(5 parameters retrieved, solzen fixed)")
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Spectral albedo")
    ax.legend(fontsize=8)

    # Panel 2: cost per surface type from retrieve_sea_ice()
    ax = axes[1]
    r_full = retrieve_sea_ice(
        observed=test_cases["bare_ice"], emulators=emulators, solzen=60, direct=1,
    )
    names  = list(r_full.cost_per_type.keys())
    costs  = [r_full.cost_per_type[n] for n in names]
    colors = ["#2ca02c" if n == r_full.surface_type else "#9ecae1" for n in names]
    ax.bar(names, costs, color=colors)
    ax.set_ylabel("Chi-squared residual")
    ax.set_title(f"Surface type classification\n"
                 f"Winner: {r_full.surface_type}  (confidence={r_full.confidence:.3f})")
    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.show()
