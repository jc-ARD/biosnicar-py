#!/usr/bin/env python3
"""Pre-compute sea-ice optical property LUT and save to data/OP_data/480band/luts/sea_ice.npz.

Grid: T(7) × S(6) × density(4) × bubble_radius(4) = 672 combinations.
Each point stores (tau_per_m, ssa, g) at 480 wavelengths.

Run from the repo root:
    uv run python scripts/build_sea_ice_lut.py

Expected runtime: 2–5 minutes on a modern laptop.
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
RHO_GRID = np.array([870, 890, 910, 920], dtype=float)
BUBBLE_GRID = np.array([100, 300, 500, 1000], dtype=float)

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
