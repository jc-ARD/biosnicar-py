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
params = dict(brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0,
              black_carbon=0.0, rho_DL=850.0, solzen=60, direct=1)

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

print("\n── Brine volume sensitivity (FYI_bare) ──")
from biosnicar.sea_ice.brine_volume import compute_brine_volume as _cvb
from biosnicar.sea_ice.emulator_configs import FYI_BARE_S_REF as _S_REF
# Sample Vb at representative temperatures (T in [-22, -2] at S_ref=6 psu)
vb_vals  = [_cvb(_S_REF, T) for T in [-20, -15, -10, -5, -3]]
t_labels = [-20, -15, -10, -5, -3]
bbas = []
for Vb in vb_vals:
    a = emu_bare.predict(brine_volume_fraction=Vb, sea_ice_bubble_radius=200.0,
                         black_carbon=0.0, rho_DL=850.0, solzen=60, direct=1)
    bbas.append(float(np.sum(emu_bare.flx_slr * a) / np.sum(emu_bare.flx_slr)))
for T_lbl, Vb, bba in zip(t_labels, vb_vals, bbas):
    print(f"  T≈{T_lbl:3d}°C  Vb={Vb:.4f}  BBA={bba:.3f}")


# ── Bubble radius sensitivity (NIR control) ───────────────────────────────────

print("\n── Bubble radius sensitivity (FYI_bare) ──")
bubble_radii = [50, 100, 200, 500, 1000]
spectra_bbl = {}
for bbl in bubble_radii:
    spectra_bbl[bbl] = emu_bare.predict(
        brine_volume_fraction=0.033, sea_ice_bubble_radius=float(bbl),
        black_carbon=0.0, rho_DL=850.0, solzen=60, direct=1,
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
    from biosnicar.sea_ice.brine_volume import compute_brine_volume, invert_brine_volume
    from biosnicar.sea_ice.emulator_configs import FYI_BARE_S_REF

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: brine_volume_fraction sweep — FYI_bare
    # Vb is the primary retrieved parameter; this shows how brine concentration
    # darkens the NIR.  Each Vb corresponds to a temperature at S_ref=6 psu —
    # shown in the legend for physical context.
    ax = axes[0]
    cmap = plt.cm.coolwarm_r
    vb_values = np.linspace(0.022, 0.130, 7)   # cold → near-melting
    for i, Vb in enumerate(vb_values):
        T_label = invert_brine_volume(Vb, FYI_BARE_S_REF)
        a = emu_bare.predict(brine_volume_fraction=Vb, sea_ice_bubble_radius=200.0,
                             black_carbon=0.0, rho_DL=850.0, solzen=60, direct=1)
        ax.plot(WAVELENGTHS, a, color=cmap(i / 6), lw=1.2,
                label=f"Vb={Vb:.3f}  (T≈{T_label:.0f}°C)")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    ax.axvline(0.7, color="k", lw=0.5, ls=":", alpha=0.4)
    ax.set_title("FYI bare ice: brine volume sweep\n"
                 "(bubble_radius=200 µm, S_ref=6 psu)")
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Spectral albedo")
    ax.legend(fontsize=6.5, title="Brine vol. fraction")

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
            a = emu.predict(brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0,
                            black_carbon=0.0, rho_DL=850.0, solzen=60, direct=1)
        elif name == "FYI_snow":
            a = emu.predict(snow_depth=0.15, snow_grain_radius=300.0,
                            sea_ice_temperature=-15.0, black_carbon=0.0,
                            solzen=60, direct=1)
        elif name == "FYI_summer":
            a = emu.predict(ssl_grain_radius=2000.0, sea_ice_temperature=-3.0,
                            sea_ice_bubble_radius=200.0, black_carbon=0.0,
                            solzen=60, direct=1)
        elif name == "MYI_bare":
            a = emu.predict(brine_volume_fraction=0.015, sea_ice_bubble_radius=500.0,
                            black_carbon=0.0, solzen=60, direct=1)
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

    # ── Figure 2: snow depth + SSL grain sweeps + emulator accuracy ───────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: snow depth sweep (FYI_snow)
    # Shows how NIR fills in as snow depth decreases — thin snow transmits
    # radiation to the underlying ice, which has lower NIR than snow.
    ax = axes2[0]
    depths_cm = [2, 5, 8, 12, 20, 30]
    emu_snow = emulators["FYI_snow"]
    snow_cmap = plt.cm.Blues_r
    for i, d_cm in enumerate(depths_cm):
        a = emu_snow.predict(snow_depth=d_cm / 100, snow_grain_radius=300.0,
                             sea_ice_temperature=-15.0, black_carbon=0.0,
                             solzen=60, direct=1)
        ax.plot(WAVELENGTHS, a, color=snow_cmap(0.2 + 0.65 * i / (len(depths_cm) - 1)),
                lw=1.3, label=f"{d_cm} cm")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    ax.axvline(0.7, color="k", lw=0.5, ls=":", alpha=0.4)
    ax.set_title("Snow-covered FYI: snow depth sweep\n"
                 "(grain=300 µm, T=−15°C, BC=0)")
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Spectral albedo")
    ax.legend(fontsize=7.5, title="Snow depth")

    # Panel 2: SSL grain radius sweep (FYI_summer)
    # The SSL grain radius is the primary control on summer bare ice NIR.
    # Coarser grains → less scattering → lower NIR.
    ax = axes2[1]
    ssl_radii = [500, 1000, 2000, 3000, 5000]
    emu_summer = emulators["FYI_summer"]
    summer_cmap = plt.cm.YlOrRd
    for i, ssl_r in enumerate(ssl_radii):
        a = emu_summer.predict(ssl_grain_radius=float(ssl_r), sea_ice_temperature=-3.0,
                               sea_ice_bubble_radius=200.0, black_carbon=0.0,
                               solzen=60, direct=1)
        ax.plot(WAVELENGTHS, a, color=summer_cmap(0.2 + 0.7 * i / (len(ssl_radii) - 1)),
                lw=1.3, label=f"{ssl_r} µm")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    ax.axvline(0.7, color="k", lw=0.5, ls=":", alpha=0.4)
    ax.set_title("Summer FYI: SSL grain radius sweep\n"
                 "(T=−3°C, bubble=200 µm, BC=0)")
    ax.set_xlabel("Wavelength (µm)")
    ax.legend(fontsize=7.5, title="SSL grain radius")

    # Panel 3: Emulator accuracy — BBA scatter vs forward model
    # 20 random points per surface type, coloured by type.
    # Should lie along the 1:1 line; deviations show emulator error.
    from biosnicar import run_model as _fwd
    ax = axes2[2]
    rng2 = np.random.default_rng(99)
    type_colors = {"FYI_bare": "#1b7837", "FYI_snow": "#4575b4",
                   "FYI_summer": "#d73027", "MYI_bare": "#762a83",
                   "FYI_pond": "#e08214"}
    all_true, all_pred = [], []
    for stype, col in type_colors.items():
        emu_t = emulators[stype]
        bba_true, bba_pred = [], []
        for _ in range(20):
            p = {k: float(rng2.uniform(lo, hi))
                 for k, (lo, hi) in emu_t.bounds.items()
                 if k not in ("direct",)}
            p["solzen"] = int(rng2.integers(25, 75))
            p["direct"] = int(rng2.integers(0, 2))
            # Forward model via transform
            from biosnicar.sea_ice.emulator_configs import SEA_ICE_EMULATOR_CONFIGS
            run_kw = SEA_ICE_EMULATOR_CONFIGS[stype]["transform_fn"](p)
            try:
                fwd = _fwd(**run_kw)
                bba_pred.append(float(np.sum(emu_t.flx_slr * emu_t.predict(**p))
                                      / np.sum(emu_t.flx_slr)))
                bba_true.append(fwd.BBA)
            except Exception:
                pass
        ax.scatter(bba_true, bba_pred, c=col, s=18, alpha=0.75, label=stype)
        all_true.extend(bba_true); all_pred.extend(bba_pred)
    lo_v = min(all_true + all_pred) - 0.02
    hi_v = max(all_true + all_pred) + 0.02
    ax.plot([lo_v, hi_v], [lo_v, hi_v], "k--", lw=0.8, alpha=0.5, label="1:1")
    mae = np.mean(np.abs(np.array(all_true) - np.array(all_pred)))
    ax.set_xlim(lo_v, hi_v); ax.set_ylim(lo_v, hi_v)
    ax.set_xlabel("Forward model BBA"); ax.set_ylabel("Emulator BBA")
    ax.set_title(f"Emulator accuracy — all surface types\n"
                 f"mean |BBA error| = {mae:.4f}  (n=20 per type)")
    ax.legend(fontsize=7.5, ncol=2)
    ax.set_aspect("equal")

    fig2.tight_layout()
    plt.show()
