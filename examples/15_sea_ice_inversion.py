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

from biosnicar import run_model
from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators, SEA_ICE_EMULATOR_CONFIGS
from biosnicar.sea_ice.retrieve import retrieve_sea_ice
from biosnicar.inverse import retrieve


def _fwd(type_name, params, solzen=60, direct=1):
    """Generate a forward-model spectrum for a sea ice emulator configuration.

    Uses the emulator's transform function to build the correct run_model()
    kwargs, then runs the full radiative transfer solver.  This produces an
    honest test observation — the emulator has never seen this exact spectrum,
    so retrieval residuals reflect real emulator approximation error rather
    than trivially zero (which happens when predict() generates its own obs).
    """
    run_kw = SEA_ICE_EMULATOR_CONFIGS[type_name]["transform_fn"](
        {**params, "solzen": solzen, "direct": direct}
    )
    return run_model(**run_kw)

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
# Generate test parameters by random draw — seeded for reproducibility.
# Values are non-round to rule out any coincidental proximity to training samples.
_rng15 = np.random.default_rng(2025)
_b = emulators["FYI_bare"].bounds
TRUE = {k: float(_rng15.uniform((lo+hi)/2 - 0.4*(hi-lo), (lo+hi)/2 + 0.4*(hi-lo)))
        for k, (lo, hi) in _b.items() if k not in ("solzen", "direct")}
TRUE["solzen"] = 60; TRUE["direct"] = 1
TRUE_VB  = TRUE["brine_volume_fraction"]
TRUE_T_C = invert_brine_volume(TRUE_VB, FYI_BARE_S_REF)
print(f"  True parameters (seeded random): {TRUE}")

# Generate synthetic observed spectrum from the FORWARD MODEL — not the emulator.
# Using the emulator's own predict() as the "observation" would make retrieval
# trivially perfect (cost ≈ 0) and meaningless.  The forward model is independent
# of the emulator and exercises the full radiative transfer pipeline.
obs_fwd = _fwd("FYI_bare", TRUE)
obs     = np.array(obs_fwd.albedo)

_FIXED = {"solzen", "direct"}
result = retrieve(
    observed      = obs,
    parameters    = [k for k in TRUE if k not in _FIXED],
    emulator      = emu_bare,
    fixed_params  = {k: TRUE[k] for k in _FIXED},
    method        = "L-BFGS-B",
)

# Post-hoc T recovery from retrieved Vb
Vb_ret = result.best_fit["brine_volume_fraction"]
T_ret  = invert_brine_volume(Vb_ret, FYI_BARE_S_REF)

print(f"  Converged: {result.converged}  Cost: {result.cost:.4f}")
print(f"  {'Parameter':28s}  {'True':>9}  {'Retrieved':>10}  {'Error':>8}  {'σ':>8}")
print("  " + "-"*64)
for param, true_val in TRUE.items():
    if param in _FIXED:
        continue
    ret_val = result.best_fit[param]
    unc     = result.uncertainty.get(param, float("nan"))
    err     = ret_val - true_val
    print(f"  {param:28s}  {true_val:>9.4f}  {ret_val:>10.4f}  {err:>+8.4f}  {unc:>8.4f}")
print(f"  {'[T recovered from Vb]':28s}  {TRUE_T_C:>9.2f}  {T_ret:>10.2f}  {T_ret-TRUE_T_C:>+8.2f}°C")


# ── 2. retrieve_sea_ice() — classify surface type ────────────────────────────

print("\n── 2. Surface-type classification ──")

# Test with three different surface types
def _rand_params(emu_name, rng):
    b = emulators[emu_name].bounds
    return {k: float(rng.uniform((lo+hi)/2 - 0.4*(hi-lo), (lo+hi)/2 + 0.4*(hi-lo)))
            for k, (lo, hi) in b.items() if k not in ("solzen", "direct")}

_rng_cls = np.random.default_rng(2026)
test_cases = {
    "bare_ice":  np.array(_fwd("FYI_bare",
        {**_rand_params("FYI_bare", _rng_cls), "solzen": 60, "direct": 1}).albedo),
    "snow":      np.array(_fwd("FYI_snow",
        {**_rand_params("FYI_snow", _rng_cls), "solzen": 60, "direct": 1}).albedo),
    "melt_pond": np.array(_fwd("FYI_pond",
        {**_rand_params("FYI_pond", _rng_cls), "solzen": 60, "direct": 1}).albedo),
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
out_true  = _fwd("FYI_bare", true_state)        # forward model — not emulator
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
    # ── Figure 1: retrieval fit quality + cost-per-type classification ─────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Panel 1: obs vs retrieved spectrum for bare FYI retrieval
    ax = axes[0]
    pred = result.predicted_albedo
    ax.plot(WAVELENGTHS, obs, "k-", lw=1.5, label="Observed (synthetic truth)")
    ax.plot(WAVELENGTHS, pred, "--", color="#e07b39", lw=1.5,
            label=f"Retrieved  (cost={result.cost:.2e})")
    ax.fill_between(WAVELENGTHS, obs, pred,
                    where=(obs > pred), alpha=0.15, color="#e07b39", label="Residual")
    ax.fill_between(WAVELENGTHS, obs, pred,
                    where=(obs <= pred), alpha=0.15, color="#3182bd")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    ax.axvline(0.7, color="k", lw=0.5, ls=":", alpha=0.4)
    ax.text(0.72, 0.04, "NIR →", fontsize=8, color="gray")
    ax.set_title("FYI bare ice: spectral fit\n"
                 "(Vb + bubble_radius + BC + rho_DL retrieved)")
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Spectral albedo")
    ax.legend(fontsize=8)

    # Panel 2: log-scale cost per surface type — shows decision margin
    ax = axes[1]
    r_full = retrieve_sea_ice(
        observed=test_cases["bare_ice"], emulators=emulators, solzen=60, direct=1,
    )
    names_s = sorted(r_full.cost_per_type, key=lambda k: r_full.cost_per_type[k])
    costs_s = [r_full.cost_per_type[n] for n in names_s]
    bar_colors = ["#2ca02c" if n == r_full.surface_type else "#aec7e8"
                  for n in names_s]
    bars = ax.bar(names_s, costs_s, color=bar_colors, log=True)
    ax.set_ylabel("Chi-squared residual (log scale)")
    ax.set_title(f"Surface type classification — bare FYI input\n"
                 f"Winner: {r_full.surface_type}  (confidence={r_full.confidence:.3f})")
    ax.tick_params(axis="x", rotation=20)
    for bar, cost in zip(bars, costs_s):
        ax.text(bar.get_x() + bar.get_width() / 2, cost * 1.5,
                f"{cost:.1e}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    plt.show()

    # ── Figure 2: multi-scenario retrieval — obs and fitted spectra ────────
    # Four different sea ice states retrieved simultaneously with
    # retrieve_sea_ice(). Demonstrates classification and fit quality
    # across the full range of surface types.
    # All scenario observations from the forward model, not the emulator.
    scenarios_si = {
        "Bare FYI (cold, clean)": np.array(_fwd("FYI_bare",
            dict(brine_volume_fraction=0.025, sea_ice_bubble_radius=200.0,
                 black_carbon=0.0, rho_DL=860.0)).albedo),
        "Bare FYI (warm, dirty)": np.array(_fwd("FYI_bare",
            dict(brine_volume_fraction=0.095, sea_ice_bubble_radius=400.0,
                 black_carbon=2000.0, rho_DL=840.0)).albedo),
        "Melt pond (shallow)":    np.array(_fwd("FYI_pond",
            dict(pond_depth=0.08, sea_ice_temperature=-4.0,
                 black_carbon=1200.0)).albedo),
        "Snow-covered FYI":       np.array(_fwd("FYI_snow",
            dict(snow_depth=0.12, snow_grain_radius=400.0,
                 sea_ice_temperature=-12.0, black_carbon=100.0)).albedo),
    }
    scen_colors = ["#1b7837", "#d73027", "#4575b4", "#6a3d9a"]

    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    fig2.suptitle("Multi-scenario sea ice retrieval — retrieve_sea_ice() fits all five emulators",
                  fontsize=10)
    for ax, (label, obs_scen), col in zip(axes2, scenarios_si.items(), scen_colors):
        r_scen = retrieve_sea_ice(
            observed=obs_scen, emulators=emulators, solzen=60, direct=1,
        )
        pred_scen = r_scen.predicted_albedo
        ax.plot(WAVELENGTHS, obs_scen, "-", color=col, lw=1.5, label="Observed")
        ax.plot(WAVELENGTHS, pred_scen, "--", color="k", lw=1.0,
                alpha=0.7, label=f"Retrieved")
        ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
        ax.axvline(0.7, color="gray", lw=0.5, ls=":", alpha=0.4)
        ax.set_title(f"{label}\n→ {r_scen.surface_type} "
                     f"(conf={r_scen.confidence:.2f})", fontsize=8)
        ax.set_xlabel("Wavelength (µm)")
        if ax is axes2[0]:
            ax.set_ylabel("Spectral albedo")
        ax.legend(fontsize=7)

    fig2.tight_layout()
    plt.show()

    # ── Figure 3: residual spectra — winner vs. losers ─────────────────────
    # Shows the spectral structure of residuals for each emulator when fitted
    # against the bare-ice observation.  The winning emulator (FYI_bare) has
    # flat, near-zero residuals; the losers show characteristic misfit patterns
    # that reveal why they cannot fit the surface type.
    fig3, axes3 = plt.subplots(1, 5, figsize=(18, 3.5), sharey=True)
    fig3.suptitle(
        "Residual spectra (predicted − observed) for each emulator\n"
        "flat near-zero = correct surface type; systematic shape = wrong type",
        fontsize=9)
    type_col = {"FYI_bare": "#1b7837", "FYI_snow": "#4575b4",
                "FYI_summer": "#d73027", "MYI_bare": "#762a83",
                "FYI_pond": "#e08214"}
    obs_bare_plot = test_cases["bare_ice"]
    for ax, (stype, fit) in zip(axes3, r_full.all_fits.items()):
        resid = fit.predicted_albedo - obs_bare_plot
        ax.plot(WAVELENGTHS, resid, color=type_col[stype], lw=1.2)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
        ax.axvline(0.7, color="gray", lw=0.5, ls=":", alpha=0.4)
        ax.fill_between(WAVELENGTHS, resid, 0,
                        where=(resid > 0), alpha=0.2, color=type_col[stype])
        ax.fill_between(WAVELENGTHS, resid, 0,
                        where=(resid < 0), alpha=0.2, color=type_col[stype])
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        winner_marker = " ★" if stype == r_full.surface_type else ""
        ax.set_title(f"{stype}{winner_marker}\nRMSE={rmse:.4f}", fontsize=8)
        ax.set_xlim(0.3, 2.5); ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Wavelength (µm)")
        if ax is axes3[0]:
            ax.set_ylabel("Predicted − Observed")

    fig3.tight_layout()
    plt.show()
