#!/usr/bin/env python3
"""Sea ice emulator — fast surrogate for the BioSNICAR sea ice forward model.

This example demonstrates:
  1. Loading the pre-built sea ice emulators
  2. Predicting spectral albedo at microsecond speed (vs ~3ms for the forward model)
  3. Full Outputs compatibility — .BBA, .to_platform(), .plot()
  4. Speed benchmark: emulator vs forward model
  5. Sensitivity sweeps using the emulator

Prerequisites
-------------
Build the emulators first (takes ~30–45 min):

    python scripts/build_sea_ice_emulators.py

Or build a fast version for testing:

    python scripts/build_sea_ice_emulators.py --fast
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from biosnicar import run_model
from biosnicar.drivers.run_emulator import run_emulator
from biosnicar.emulator import Emulator

PLOT = True
WAVELENGTHS = np.arange(0.205, 4.999, 0.01)

# ── Load emulators ───────────────────────────────────────────────────────────

print("Loading sea ice emulators…")
from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators

emulators = load_sea_ice_emulators()
for name, emu in emulators.items():
    print(f"  {name:14s}  R²={emu.training_score:.4f}  "
          f"params={emu.param_names}")


# ── Speed benchmark ──────────────────────────────────────────────────────────

print("\n── Speed benchmark: emulator vs forward model ──")

N_BENCH = 500
emu_bare = emulators["FYI_bare"]
params = dict(sea_ice_temperature=-10.0, sea_ice_bubble_radius=200.0,
              sea_ice_salinity=8.0, black_carbon=0.0, rho_DL=850.0,
              solzen=60, direct=1)

# Emulator speed
t0 = time.perf_counter()
for _ in range(N_BENCH):
    emu_bare.predict(**params)
t_emu = (time.perf_counter() - t0) / N_BENCH * 1e6  # µs per call

# Forward model speed (fewer calls — it's slower)
N_FWD = 20
t0 = time.perf_counter()
for _ in range(N_FWD):
    run_model(preset="FYI_WINTER_BARE", sea_ice_temperature=[-10, -10],
              sea_ice_bubble_radius=[200, 400], solzen=60, direct=1,
              black_carbon=[0, 0])
t_fwd = (time.perf_counter() - t0) / N_FWD * 1e3  # ms per call

print(f"  Emulator   :  {t_emu:.1f} µs / call")
print(f"  Forward model:  {t_fwd:.1f} ms / call")
print(f"  Speedup    :  {t_fwd * 1000 / t_emu:.0f}×")


# ── Full Outputs compatibility ───────────────────────────────────────────────

print("\n── Outputs compatibility ──")
out = run_emulator(emu_bare, **params)
print(f"  BBA        = {out.BBA:.3f}")
print(f"  BBAVIS     = {out.BBAVIS:.3f}")
print(f"  BBANIR     = {out.BBANIR:.3f}")
s2 = out.to_platform("sentinel2")
print(f"  Sentinel-2 B3={s2.B3:.3f}  B8={s2.B8:.3f}  B11={s2.B11:.3f}")


# ── Temperature sensitivity sweep ────────────────────────────────────────────

print("\n── Temperature sensitivity (FYI_bare) ──")
temps = np.linspace(-30, -2, 15)
bbas = []
for T in temps:
    a = emu_bare.predict(sea_ice_temperature=T, sea_ice_bubble_radius=200.0,
                         sea_ice_salinity=8.0, black_carbon=0.0,
                         rho_DL=850.0, solzen=60, direct=1)
    bbas.append(float(np.sum(emu_bare.flx_slr * a) / np.sum(emu_bare.flx_slr)))
print(f"  T=-30°C  BBA={bbas[0]:.3f}")
print(f"  T=-10°C  BBA={bbas[7]:.3f}")
print(f"  T= -2°C  BBA={bbas[-1]:.3f}")


# ── Bubble radius sensitivity (NIR control) ───────────────────────────────────

print("\n── Bubble radius sensitivity (FYI_bare) ──")
bubble_radii = [50, 100, 200, 500, 1000]
spectra_bbl = {}
for bbl in bubble_radii:
    spectra_bbl[bbl] = emu_bare.predict(
        sea_ice_temperature=-10.0, sea_ice_bubble_radius=float(bbl),
        sea_ice_salinity=8.0, black_carbon=0.0, rho_DL=850.0,
        solzen=60, direct=1,
    )
    nir = float(np.mean(spectra_bbl[bbl][50:]))
    print(f"  bbl={bbl:4d} µm  NIR={nir:.3f}")


# ── Snow depth sweep (FYI_snow) ────────────────────────────────────────────

print("\n── Snow depth sweep (FYI_snow emulator) ──")
emu_snow = emulators["FYI_snow"]
depths_cm = [2, 5, 10, 15, 20, 30]
for d_cm in depths_cm:
    a = emu_snow.predict(snow_depth=d_cm / 100, snow_grain_radius=300.0,
                         sea_ice_temperature=-15.0, black_carbon=0.0,
                         solzen=60, direct=1)
    bba = float(np.sum(emu_snow.flx_slr * a) / np.sum(emu_snow.flx_slr))
    print(f"  snow_depth={d_cm:2d} cm  BBA={bba:.3f}")


# ── Melt pond depth sweep (FYI_pond) ────────────────────────────────────────

print("\n── Pond depth sweep (FYI_pond emulator) ──")
emu_pond = emulators["FYI_pond"]
pond_depths = [0.02, 0.05, 0.10, 0.20, 0.40, 0.60]
for d in pond_depths:
    a = emu_pond.predict(pond_depth=d, sea_ice_temperature=-5.0,
                         black_carbon=1200.0, solzen=60, direct=1)
    nir = float(np.mean(a[50:]))
    bba = float(np.sum(emu_pond.flx_slr * a) / np.sum(emu_pond.flx_slr))
    print(f"  pond_depth={d:.2f} m  BBA={bba:.3f}  NIR={nir:.3f}")


# ── Plots ────────────────────────────────────────────────────────────────────

if PLOT:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: temperature sweep — FYI_bare
    ax = axes[0]
    cmap = plt.cm.coolwarm
    for i, T in enumerate(np.linspace(-30, -2, 7)):
        a = emu_bare.predict(sea_ice_temperature=T, sea_ice_bubble_radius=200.0,
                             sea_ice_salinity=8.0, black_carbon=0.0,
                             rho_DL=850.0, solzen=60, direct=1)
        ax.plot(WAVELENGTHS, a, color=cmap(i / 6), lw=1.2, label=f"{T:.0f}°C")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    ax.axvline(0.7, color="k", lw=0.5, ls=":", alpha=0.4)
    ax.set_title("FYI bare ice: temperature sweep\n(bubble_radius=200 µm, S=8 psu)")
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Spectral albedo")
    ax.legend(fontsize=7, title="Temperature")

    # Panel 2: bubble radius sweep — FYI_bare
    ax = axes[1]
    colors2 = plt.cm.viridis(np.linspace(0, 1, len(bubble_radii)))
    for bbl, col in zip(bubble_radii, colors2):
        a = spectra_bbl[bbl]
        ax.plot(WAVELENGTHS, a, color=col, lw=1.2, label=f"{bbl} µm")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    ax.axvline(0.7, color="k", lw=0.5, ls=":", alpha=0.4)
    ax.set_title("FYI bare ice: bubble radius sweep\n(T=−10°C, S=8 psu)")
    ax.set_xlabel("Wavelength (µm)")
    ax.legend(fontsize=7, title="Bubble radius")

    # Panel 3: all surface types at representative conditions
    ax = axes[2]
    spectra_types = {}
    for name, emu in emulators.items():
        if name == "FYI_bare":
            a = emu.predict(sea_ice_temperature=-10.0, sea_ice_bubble_radius=200.0,
                            sea_ice_salinity=8.0, black_carbon=0.0, rho_DL=850.0,
                            solzen=60, direct=1)
        elif name == "FYI_snow":
            a = emu.predict(snow_depth=0.15, snow_grain_radius=200.0,
                            sea_ice_temperature=-15.0, black_carbon=0.0,
                            solzen=60, direct=1)
        elif name == "FYI_summer":
            a = emu.predict(ssl_grain_radius=2000.0, sea_ice_temperature=-3.0,
                            sea_ice_bubble_radius=200.0, black_carbon=0.0,
                            solzen=60, direct=1)
        elif name == "MYI_bare":
            a = emu.predict(sea_ice_temperature=-10.0, sea_ice_bubble_radius=500.0,
                            sea_ice_salinity=2.0, black_carbon=0.0,
                            solzen=60, direct=1)
        elif name == "FYI_pond":
            a = emu.predict(pond_depth=0.15, sea_ice_temperature=-5.0,
                            black_carbon=1200.0, solzen=60, direct=1)
        spectra_types[name] = a
        ax.plot(WAVELENGTHS, a, lw=1.5, label=name)
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    ax.axvline(0.7, color="k", lw=0.5, ls=":", alpha=0.4)
    ax.set_title("All surface types (representative conditions)")
    ax.set_xlabel("Wavelength (µm)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
