#!/usr/bin/env python3
"""Comparison of optimisation methods for inversion.

This script runs the same SSA retrieval problem with four different
optimisation methods and compares their speed, accuracy, and convergence
characteristics.  This helps you choose the right method for your use case.

The methods compared are:
  - L-BFGS-B (default): hybrid DE pre-search + quasi-Newton polish.
    Fast and accurate for most problems.
  - Nelder-Mead: derivative-free simplex. More robust to noisy cost
    surfaces (e.g. when using the direct forward model).
  - differential_evolution: global stochastic search. Slower but explores
    the full parameter space — use when the initial guess is poor or the
    cost surface is multimodal.
  - mcmc (optional): full Bayesian posterior sampling via emcee. Gives
    publication-quality uncertainty estimates and reveals parameter
    correlations. Much slower.

Set MCMC = True to include MCMC (adds ~30-60 seconds).
"""

import time

import numpy as np

from biosnicar import run_model
from biosnicar.emulator import Emulator
from biosnicar.inverse import retrieve
from biosnicar.inverse.result import _compute_ssa

PLOT = False
MCMC = False

# ======================================================================
# Setup: load emulator and generate a synthetic observation
# ======================================================================

# Load the pre-built 8-parameter glacier ice emulator.
emu = Emulator.load("data/emulators/glacier_ice_8_param_default.npz")

# Parameters that are known a priori and will NOT be retrieved.
# Dust is fixed at 1000 ppb (low spectral sensitivity — see docs).
fixed = {"solzen": 50, "direct": 1, "dust": 1000, "snow_algae": 0}

# Generate a synthetic observation from the FULL FORWARD MODEL (not the
# emulator) for an honest comparison — the retrieval must bridge the
# emulator's approximation gap.
true_params = dict(rds=1000, rho=600, black_carbon=5000, glacier_algae=50000)
true_outputs = run_model(**true_params, **fixed, layer_type=1)
observed = np.array(true_outputs.albedo, dtype=np.float64)

# SSA replaces (rds, rho) as the physically-meaningful ice optical
# parameter.  It eliminates the rds/rho degeneracy and gives ~5.5%
# error compared to ~73% for rds and ~33% for rho individually.
true_ssa = _compute_ssa(true_params["rds"], true_params["rho"])
parameters = ["ssa", "black_carbon", "glacier_algae"]

# Build a dict of true values in retrieval parameter space for error
# computation.
true_vals = {
    "ssa": true_ssa,
    "black_carbon": true_params["black_carbon"],
    "glacier_algae": true_params["glacier_algae"],
}

print(f"True parameters: SSA={true_ssa:.4f} m2/kg, BC={true_params['black_carbon']}, "
      f"GA={true_params['glacier_algae']}\n")

# ======================================================================
# Compare gradient-based and derivative-free methods
# ======================================================================
# We run each method on the same problem and record the result plus
# elapsed wall-clock time.  All methods use the same emulator, bounds,
# and fixed parameters — only the optimisation strategy differs.

methods = ["L-BFGS-B", "Nelder-Mead", "differential_evolution"]
results = {}

for method in methods:
    t0 = time.time()
    result = retrieve(
        observed=observed,
        parameters=parameters,
        emulator=emu,
        method=method,
        fixed_params=fixed,
    )
    elapsed = time.time() - t0
    results[method] = (result, elapsed)

    print(f"=== {method} ===")
    print(f"  Time:       {elapsed:.3f} s")
    print(f"  Converged:  {result.converged}")
    print(f"  Cost:       {result.cost:.6f}")
    print(f"  Func evals: {result.n_function_evals}")
    for name in parameters:
        err = abs(result.best_fit[name] - true_vals[name])
        print(
            f"  {name:25s} = {result.best_fit[name]:10.4f}  "
            f"(err={err:.4f}, unc={result.uncertainty[name]:.4f})"
        )
    print(f"  Internal decomposition: {result.derived}")
    print()

# ======================================================================
# Summary table
# ======================================================================
# A compact comparison of the three methods.  The SSA error is the
# absolute difference between the retrieved and true SSA (in m²/kg).
# Lower cost = better spectral fit; fewer evals = faster.

print("=== Summary ===\n")
print(
    f"  {'Method':30s} {'Time (s)':>10s} {'Cost':>12s} {'Evals':>8s} {'SSA err':>10s}"
)
print(f"  {'-' * 30} {'-' * 10} {'-' * 12} {'-' * 8} {'-' * 10}")
for method in methods:
    r, t = results[method]
    ssa_err = abs(r.best_fit["ssa"] - true_ssa)
    print(
        f"  {method:30s} {t:10.3f} {r.cost:12.6f} {r.n_function_evals:8d} {ssa_err:10.4f}"
    )

# ======================================================================
# When to use each method
# ======================================================================
print("\n=== Guidance ===\n")
print("  L-BFGS-B:                Fast default. Uses gradients and box constraints.")
print("                           Best for well-constrained, smooth problems.")
print("  Nelder-Mead:             Derivative-free. More robust to noisy cost surfaces.")
print("                           Use when L-BFGS-B fails to converge.")
print("  differential_evolution:  Global search. Slower but explores full bounds.")
print(
    "                           Use when initial guess is poor or cost is multimodal."
)
print("  mcmc:                    Full Bayesian posterior. Use for publication-quality")
print("                           uncertainty or when parameters are degenerate.")

# ======================================================================
# MCMC (optional, slower)
# ======================================================================
# MCMC (Markov Chain Monte Carlo) samples the full posterior distribution
# rather than finding a single point estimate.  This gives:
# - Posterior median and standard deviation (more robust uncertainty)
# - Full chains for corner plots and correlation analysis
# - Detection of bimodal or skewed posteriors
#
# The tradeoff is speed: MCMC requires ~50,000-200,000 emulator calls vs
# ~1,000-3,000 for L-BFGS-B.  With the microsecond emulator this is
# still only ~30-60 seconds.

if MCMC:
    print("\n=== MCMC (32 walkers, 1000 steps, 200 burn-in) ===\n")
    t0 = time.time()
    result_mcmc = retrieve(
        observed=observed,
        parameters=parameters,
        emulator=emu,
        method="mcmc",
        mcmc_walkers=32,
        mcmc_steps=1000,
        mcmc_burn=200,
        fixed_params=fixed,
    )
    elapsed = time.time() - t0
    print(f"  Time:       {elapsed:.1f} s")
    print(f"  Acceptance: {result_mcmc.acceptance_fraction:.3f}")
    print(f"  Chain shape: {result_mcmc.chains.shape}")
    for name in parameters:
        print(
            f"  {name:25s} = {result_mcmc.best_fit[name]:10.4f} "
            f"+/- {result_mcmc.uncertainty[name]:.4f}"
        )

    # Trace plots show the MCMC walkers exploring parameter space over
    # time.  Well-mixed chains look like "hairy caterpillars" — the
    # walkers should be exploring the same region and not stuck in
    # separate modes.
    if PLOT:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 2.5 * len(parameters)))
        for i, (ax, name) in enumerate(zip(axes, parameters)):
            ax.plot(result_mcmc.chains[:, :, i], alpha=0.3, linewidth=0.5)
            ax.axhline(true_vals[name], color="r", linestyle="--", label="True")
            ax.set_ylabel(name)
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("Step")
        fig.suptitle("MCMC trace plots")
        fig.tight_layout()
        plt.show()

elif PLOT:
    import matplotlib.pyplot as plt

    # Compare the retrieved spectra from the three optimisation methods.
    # They should be nearly identical if all converged to the same minimum.
    wavelengths = np.arange(0.205, 4.999, 0.01)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wavelengths, observed, "k-", label="Observed", linewidth=2)
    colors = ["r", "b", "g"]
    for (method, (result, _)), color in zip(results.items(), colors):
        ax.plot(
            wavelengths,
            result.predicted_albedo,
            "--",
            color=color,
            label=method,
            linewidth=1,
        )
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Albedo")
    ax.set_xlim(0.2, 2.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieved spectra from different optimisers (SSA mode)")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ======================================================================
# Sea ice: optimiser comparison
# ======================================================================
#
# The same four optimisers are available for sea ice retrieval.  The
# choice follows the same logic as for glacier ice:
#
#   L-BFGS-B (default): best for most applications; hybrid DE pre-search
#     escapes local minima, then gradient polish converges precisely.
#   Nelder-Mead: use when running the direct forward model (no emulator)
#     or when the cost surface is very noisy (real observations with high
#     measurement noise).
#   differential_evolution: use when the initial guess is very poor (e.g.
#     completely unknown surface conditions).
#   MCMC: use when you need publication-quality posterior distributions
#     and parameter correlations (e.g. brine_volume_fraction vs
#     sea_ice_bubble_radius correlation in NIR).
#
# We use FYI_pond pond_depth retrieval as the demonstration case because:
#   - It has a single clear free parameter (pond_depth)
#   - The cost surface is approximately quadratic and well-behaved
#   - All four methods converge quickly
#   - The MCMC posterior is clean and easy to interpret

print("\n" + "=" * 65)
print("SEA ICE: OPTIMISER COMPARISON")
print("=" * 65)

from biosnicar.sea_ice.emulator_configs import (
    load_sea_ice_emulators, SEA_ICE_EMULATOR_CONFIGS
)

si_emus  = load_sea_ice_emulators(["FYI_pond"])
emu_pond = si_emus["FYI_pond"]

# Seeded random test parameters — non-round values, reproducible.
_rng09       = np.random.default_rng(2025)
_b9          = si_emus["FYI_pond"].bounds
TRUE_POND_PARAMS = {k: float(_rng09.uniform((lo+hi)/2 - 0.4*(hi-lo),
                                             (lo+hi)/2 + 0.4*(hi-lo)))
                   for k, (lo, hi) in _b9.items() if k not in ("solzen", "direct")}
TRUE_POND_PARAMS["solzen"] = 60; TRUE_POND_PARAMS["direct"] = 1
TRUE_POND_DEPTH = TRUE_POND_PARAMS["pond_depth"]

# Generate observation from the FORWARD MODEL — not the emulator.
_run_kw = SEA_ICE_EMULATOR_CONFIGS["FYI_pond"]["transform_fn"](TRUE_POND_PARAMS)
obs_pond = np.array(run_model(**_run_kw).albedo)
fixed_pond = {k: v for k, v in TRUE_POND_PARAMS.items() if k != "pond_depth"}

print(f"\n  True pond_depth = {TRUE_POND_DEPTH} m")
print(f"  {'Method':25s}  {'Time (s)':>9}  {'Cost':>10}  {'Depth (m)':>10}  {'Error':>8}  {'Converged':>10}")
print("  " + "-" * 78)

si_methods = ["L-BFGS-B", "Nelder-Mead", "differential_evolution"]
si_results = {}
for method in si_methods:
    t0 = time.time()
    r = retrieve(
        observed     = obs_pond,
        parameters   = ["pond_depth"],
        emulator     = emu_pond,
        fixed_params = fixed_pond,
        method       = method,
    )
    elapsed = time.time() - t0
    si_results[method] = r
    depth = r.best_fit["pond_depth"]
    err   = depth - TRUE_POND_DEPTH
    print(f"  {method:25s}  {elapsed:9.3f}  {r.cost:10.2e}  {depth:10.4f}  "
          f"{err:+8.4f}  {r.converged!s:>10}")

# ── Optional MCMC: posterior for pond_depth ────────────────────────────────
#
# With a single free parameter (pond_depth), the MCMC posterior is clean
# and easy to interpret.  The posterior width reflects both the NIR
# spectral sensitivity and measurement noise (none in this synthetic case).
# In practice with noisy observations, the posterior reflects how well
# depth is constrained by the spectral information content.

if MCMC:
    print("\n  MCMC: pond_depth posterior (32 walkers, 500 steps)")
    t0 = time.time()
    r_mcmc = retrieve(
        observed      = obs_pond,
        parameters    = ["pond_depth"],
        emulator      = emu_pond,
        fixed_params  = fixed_pond,
        method        = "mcmc",
        mcmc_walkers  = 32,
        mcmc_steps    = 500,
        mcmc_burn     = 100,
    )
    elapsed = time.time() - t0
    depth_mcmc = r_mcmc.best_fit["pond_depth"]
    unc_mcmc   = r_mcmc.uncertainty["pond_depth"]
    print(f"  pond_depth = {depth_mcmc:.4f} ± {unc_mcmc:.4f} m  "
          f"(acceptance={r_mcmc.acceptance_fraction:.3f}, time={elapsed:.1f}s)")
    print(f"  True depth = {TRUE_POND_DEPTH:.4f} m  "
          f"Error = {depth_mcmc - TRUE_POND_DEPTH:+.4f} m")
    print(f"  Chain shape: {r_mcmc.chains.shape}  "
          "(use corner.corner() for full posterior plot)")

# Guidance summary: method selection for sea ice
print("\n  Method guidance for sea ice retrieval:")
print("   L-BFGS-B  — default; fastest for all emulator-based retrievals")
print("   Nelder-Mead — use with direct forward_fn (no emulator)")
print("   DE         — use when Vb and bbl_radius are both unknown (multimodal)")
print("   MCMC       — use when parameter correlations matter (e.g. Vb vs bbl)")
print()
print("  Key constraint: brine_volume_fraction and sea_ice_bubble_radius")
print("  both affect NIR albedo — retrieve one at a time unless using MCMC")
print("  or strong regularization.")

if PLOT:
    import matplotlib.pyplot as plt

    wavelengths = np.arange(0.205, 4.999, 0.01)

    # ── Sea ice: method comparison figure ─────────────────────────────────
    # Two panels: (1) convergence and accuracy summary table as a visual,
    # (2) obs vs retrieved spectrum for each method.

    fig_si09, axes_si09 = plt.subplots(1, 2, figsize=(13, 4))

    method_labels = list(si_results.keys())
    method_colors = {"L-BFGS-B": "#2ca02c",
                     "Nelder-Mead": "#ff7f0e",
                     "differential_evolution": "#1f77b4"}

    # Panel 1: retrieved pond depth + cost for each method
    ax = axes_si09[0]
    x = np.arange(len(method_labels))
    depths = [si_results[m].best_fit["pond_depth"] for m in method_labels]
    costs  = [si_results[m].cost for m in method_labels]
    bar_c  = [method_colors[m] for m in method_labels]
    bars = ax.bar(x, depths, color=bar_c, alpha=0.85)
    ax.axhline(TRUE_POND_DEPTH, color="k", lw=1.5, ls="--",
               label=f"True depth = {TRUE_POND_DEPTH} m")
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"cost={cost:.1e}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylabel("Retrieved pond depth (m)")
    ax.set_ylim(0, TRUE_POND_DEPTH * 1.3)
    ax.set_title("Sea ice: pond depth retrieval by method\n"
                 "(FYI_pond emulator, fixed T and BC)")
    ax.legend(fontsize=8)

    # Panel 2: obs vs retrieved spectra for each method
    ax = axes_si09[1]
    ax.plot(wavelengths, obs_pond, "k-", lw=1.8, label="Observed (truth)", zorder=5)
    for method, r in si_results.items():
        ax.plot(wavelengths, r.predicted_albedo, "--",
                color=method_colors[method], lw=1.3, alpha=0.85,
                label=f"{method} (depth={r.best_fit['pond_depth']:.4f} m)")
    ax.axvline(0.7, color="gray", lw=0.5, ls=":", alpha=0.4)
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 0.55)
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Spectral albedo")
    ax.set_title("Retrieved spectra — all methods agree\n"
                 "(differences are sub-pixel; all converge to same solution)")
    ax.legend(fontsize=8)

    fig_si09.tight_layout()
    plt.show()
