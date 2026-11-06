#!/usr/bin/env python3
"""Pre-compute sea-ice optical property LUT and save to data/OP_data/480band/luts/sea_ice.npz.

Grid: T(7) × S(6) × density(8) × bubble_radius(11) = 3696 combinations.
Each point stores (tau_per_m, ssa, g) at 480 wavelengths.

Run from the repo root:
    uv run python scripts/build_sea_ice_lut.py

Expected runtime: ~30 s on a modern laptop (the per-point cost is dominated by
the cached brine-RI LUT, which is built once up front).

Rebuilding this LUT invalidates the shipped sea-ice emulators in
``data/emulators/`` — they are trained on forward-model output that reads it.
Retrain them with ``scripts/build_sea_ice_emulators.py`` and re-audit with
``scripts/audit_sea_ice_emulators.py`` after any change to the grid or to the
physics in ``biosnicar/sea_ice/``.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biosnicar.sea_ice.brine_optics import build_brine_rfidx_lut, invalidate_lut_cache
from biosnicar.sea_ice.sea_ice_optics import compute_sea_ice_optics_per_metre
import biosnicar

_OUT_PATH = biosnicar.DATA_DIR / "OP_data" / "480band" / "luts" / "sea_ice.npz"

# Parameter grids
T_GRID = np.array([-2, -5, -10, -15, -20, -25, -30], dtype=float)
S_GRID = np.array([1, 4, 6, 8, 10, 12], dtype=float)
# Density grid must cover every density the model is actually asked for, since
# column_OPs._interpolate_sea_ice_lut CLAMPS to [min, max] rather than
# extrapolating.  The old grid started at 870, but the shipped presets use a
# drained-layer density of 850 (Jin et al. 2023) and the FYI_bare emulator
# samples rho_DL over (820, 900) — so most of that sampled range was being
# clamped flat to 870 and the parameter did nothing over it.  820 and 850 are
# added to close that gap.
RHO_GRID = np.array([700, 780, 820, 850, 870, 890, 910, 920], dtype=float)
# Bubble-radius grid must cover every radius the emulators sample, for the
# same reason as RHO_GRID: _interpolate_sea_ice_lut CLAMPS rather than
# extrapolating.  The old grid was [100, 300, 500, 1000], against sampled
# ranges of 50-1000 (FYI_bare), 50-500 (FYI_summer) and 200-2000 (MYI_bare) —
# so tau and ssa were bit-identical for every radius below 100 and above
# 1000, making the bottom of FYI_bare's range and ~30% of MYI_bare's log
# range inert.  Log-spaced, since scattering goes as 1/r and varies fastest
# at the small end.
BUBBLE_GRID = np.array(
    [50, 75, 100, 150, 200, 300, 450, 700, 1000, 1400, 2000], dtype=float
)

N_TOTAL = len(T_GRID) * len(S_GRID) * len(RHO_GRID) * len(BUBBLE_GRID)


def _progress(done, total, t0):
    pct = 100 * done / total
    elapsed = time.time() - t0
    eta = elapsed / done * (total - done) if done > 0 else 0
    bar = "=" * int(pct / 2) + " " * (50 - int(pct / 2))
    print(
        f"\r[{bar}] {pct:5.1f}%  {done}/{total}  "
        f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
        end="",
        flush=True,
    )


def build_lut():
    print("Building brine RI LUT first...")
    build_brine_rfidx_lut()
    invalidate_lut_cache()
    print("  done.")

    nT, nS, nR, nB = len(T_GRID), len(S_GRID), len(RHO_GRID), len(BUBBLE_GRID)
    tau_lut = np.zeros((nT, nS, nR, nB, 480))
    ssa_lut = np.zeros((nT, nS, nR, nB, 480))
    asm_lut = np.zeros((nT, nS, nR, nB, 480))

    print(f"Computing {N_TOTAL} grid points...")
    t0 = time.time()
    done = 0

    for iT, T in enumerate(T_GRID):
        for iS, S in enumerate(S_GRID):
            for iR, rho in enumerate(RHO_GRID):
                for iB, bbl in enumerate(BUBBLE_GRID):
                    try:
                        tau, ssa, g = compute_sea_ice_optics_per_metre(
                            salinity_psu=S,
                            temperature_C=T,
                            density_kg_m3=rho,
                            bubble_radius_um=bbl,
                        )
                        tau_lut[iT, iS, iR, iB] = tau
                        ssa_lut[iT, iS, iR, iB] = ssa
                        asm_lut[iT, iS, iR, iB] = g
                    except Exception as exc:
                        print(
                            f"\n  WARNING: ({T}°C, {S}psu, {rho}kg/m³, "
                            f"{bbl}µm) failed: {exc}"
                        )
                    done += 1
                    if done % 10 == 0 or done == N_TOTAL:
                        _progress(done, N_TOTAL, t0)

    print()  # newline after progress bar

    # Validate physical constraints
    n_bad_tau = np.sum(tau_lut < 0)
    n_bad_ssa = np.sum((ssa_lut < 0) | (ssa_lut > 1))
    n_bad_g = np.sum(np.abs(asm_lut) > 1)
    if n_bad_tau or n_bad_ssa or n_bad_g:
        print(
            f"WARNING: unphysical values — "
            f"τ<0: {n_bad_tau}, ω∉[0,1]: {n_bad_ssa}, |g|>1: {n_bad_g}"
        )
    else:
        print("All values physical (τ≥0, ω∈[0,1], |g|≤1).")

    np.savez_compressed(
        str(_OUT_PATH),
        T_grid=T_GRID,
        S_grid=S_GRID,
        rho_grid=RHO_GRID,
        bubble_radius_grid=BUBBLE_GRID,
        tau_per_m=tau_lut,
        ssa=ssa_lut,
        asm=asm_lut,
    )
    print(f"Saved: {_OUT_PATH}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    build_lut()
