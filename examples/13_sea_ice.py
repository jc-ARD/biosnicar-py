#!/usr/bin/env python3
"""Sea ice albedo — demonstration of the BioSNICAR sea ice extension.

Sea ice is accessed through the same run_model() entry point as
terrestrial ice, using layer_type=4 layers with three additional
per-layer parameters:
  - sea_ice_salinity     (psu)
  - sea_ice_temperature  (°C, in [-44, -2])
  - sea_ice_bubble_radius (µm)

Switching between glacier ice and sea ice is a single parameter change:

  # Glacier ice (layer_type=1)
  outputs = run_model(solzen=60, layer_type=1, rds=500, rho=700)

  # Sea ice (layer_type=4) — identical call style
  outputs = run_model(
      solzen=60,
      layer_type=[4, 4],
      dz=[0.05, 1.45],
      rds=[500, 500],         # not used for sea ice; any value works
      rho=[895, 895],
      sea_ice_salinity=[12, 8],
      sea_ice_temperature=[-25, -20],
      sea_ice_bubble_radius=[100, 200],
  )

  # Both return the same Outputs object with .BBA, .albedo, .to_platform() …

Built-in presets are available as dicts via:
  from biosnicar import run_model, FYI_WINTER_BARE, FYI_WINTER_SNOW, MYI_WINTER_BARE
  outputs = run_model(preset=FYI_WINTER_BARE, solzen=60)
"""

import numpy as np
import matplotlib.pyplot as plt

from biosnicar import run_model, FYI_WINTER_BARE, FYI_WINTER_SNOW, MYI_WINTER_BARE
from biosnicar.sea_ice.presets import ALL_PRESETS, FYI_SUMMER_BARE, FYI_POND_SHALLOW
from biosnicar.sea_ice.pond_fraction import blend_pond_fraction

WAVELENGTHS = np.arange(0.205, 4.999, 0.01)   # 480-band grid, µm
PLOT = True


# =============================================================================
# 1. Built-in presets — quickest way to get started
# =============================================================================
print("=" * 60)
print("1. Built-in presets (SZA=60°, sub-Arctic winter, clear sky)")
print("=" * 60)

for name in ["FYI_WINTER_BARE", "FYI_WINTER_SNOW", "MYI_WINTER_BARE"]:
    out = run_model(preset=name, solzen=60)
    print(f"  {name:<22}  BBA={out.BBA:.3f}  VIS={out.BBAVIS:.3f}  NIR={out.BBANIR:.3f}")


# =============================================================================
# 2. Preset as a dict — same object, can be merged with overrides
# =============================================================================
print("\n" + "=" * 60)
print("2. Preset + override (e.g. add black carbon to snow-covered FYI)")
print("=" * 60)

clean = run_model(preset=FYI_WINTER_SNOW, solzen=60)
dirty = run_model(preset=FYI_WINTER_SNOW, solzen=60, black_carbon=1000)
print(f"  Clean ice:  BBA={clean.BBA:.3f}")
print(f"  +1000 ppb BC: BBA={dirty.BBA:.3f}  (reduction: {clean.BBA - dirty.BBA:.3f})")


# =============================================================================
# 3. Fully explicit — identical syntax to terrestrial ice
# =============================================================================
print("\n" + "=" * 60)
print("3. Fully explicit flat kwargs — same style as terrestrial ice")
print("=" * 60)

# Snow on sea ice — compare with glacier ice on same call structure
glacier_outputs = run_model(
    solzen=60,
    layer_type=[0, 1],
    dz=[0.15, 1.5],
    rds=[200, 500],
    rho=[300, 700],
)

seaice_outputs = run_model(
    solzen=60,
    layer_type=[0, 4, 4],          # snow layer + two sea ice layers
    dz=[0.15, 0.05, 1.45],
    rds=[200, 500, 500],            # rds unused for layer_type=4
    rho=[300, 895, 895],
    sea_ice_salinity=[None, 12, 8], # None for snow layer
    sea_ice_temperature=[None, -25, -20],
    sea_ice_bubble_radius=[None, 100, 200],
)

print(f"  Snow on glacier ice:  BBA={glacier_outputs.BBA:.3f}")
print(f"  Snow on sea ice:      BBA={seaice_outputs.BBA:.3f}")
print(f"  Same Outputs object:  {type(glacier_outputs).__name__} == {type(seaice_outputs).__name__}")


# =============================================================================
# 4. Outputs attributes are identical for all ice types
# =============================================================================
print("\n" + "=" * 60)
print("4. Outputs — same attribute names for terrestrial and sea ice")
print("=" * 60)

terr = run_model(solzen=60, layer_type=1, rds=500, rho=700, dz=1.5)
seai = run_model(preset=FYI_WINTER_BARE, solzen=60)

print(f"  Terrestrial: BBA={terr.BBA:.3f}  (alias: broadband={terr.broadband:.3f})")
print(f"  Sea ice:     BBA={seai.BBA:.3f}  (alias: broadband={seai.broadband:.3f})")
print(f"  Albedo shapes: {terr.albedo.shape} == {seai.albedo.shape}")
print(f"  Spectrum alias: {np.array_equal(seai.albedo, seai.spectrum)}")
# Band convolution works identically
s2_terr = terr.to_platform("sentinel2")
s2_seai = seai.to_platform("sentinel2")
print(f"  Sentinel-2 B3: glacier={s2_terr.B3:.3f}  sea ice={s2_seai.B3:.3f}")


# =============================================================================
# 5. Sensitivity analysis
# =============================================================================
print("\n" + "=" * 60)
print("5. Sensitivity to sea ice salinity and temperature")
print("=" * 60)

print("  Salinity (T=−10°C, rho=910, bbl=300 µm):")
for S in [1, 4, 8, 12]:
    out = run_model(
        solzen=60,
        layer_type=4, dz=1.5, rds=500, rho=910,
        sea_ice_salinity=S, sea_ice_temperature=-10, sea_ice_bubble_radius=300,
    )
    print(f"    S={S:2d} psu  BBA={out.BBA:.3f}  VIS={out.BBAVIS:.3f}  NIR={out.BBANIR:.3f}")

print("  Temperature (S=8 psu, rho=910, bbl=300 µm):")
for T in [-2, -5, -10, -20, -30]:
    out = run_model(
        solzen=60,
        layer_type=4, dz=1.5, rds=500, rho=910,
        sea_ice_salinity=8, sea_ice_temperature=T, sea_ice_bubble_radius=300,
    )
    print(f"    T={T:3d}°C   BBA={out.BBA:.3f}  VIS={out.BBAVIS:.3f}  NIR={out.BBANIR:.3f}")


# =============================================================================
# 6. Solar zenith angle
# =============================================================================
print("\n" + "=" * 60)
print("6. Solar zenith angle sweep — FYI bare")
print("=" * 60)

for sza in [30, 45, 60, 75, 85]:
    out = run_model(preset=FYI_WINTER_BARE, solzen=sza)
    print(f"  SZA={sza:2d}°  BBA={out.BBA:.3f}  VIS={out.BBAVIS:.3f}  NIR={out.BBANIR:.3f}")


# =============================================================================
# 7. Snow cover thickness sweep
# =============================================================================
print("\n" + "=" * 60)
print("7. Snow thickness on FYI (SZA=60°)")
print("=" * 60)

for dz_snow in [0, 0.05, 0.10, 0.20, 0.40]:
    if dz_snow == 0:
        out = run_model(preset=FYI_WINTER_BARE, solzen=60)
        label = "  bare"
    else:
        out = run_model(
            solzen=60,
            layer_type=[0, 4, 4],
            dz=[dz_snow, 0.05, 1.45],
            rds=[200, 500, 500],
            rho=[250, 895, 895],
            sea_ice_salinity=[None, 12, 8],
            sea_ice_temperature=[None, -25, -20],
            sea_ice_bubble_radius=[None, 100, 200],
        )
        label = f"{int(dz_snow * 100):3d} cm"
    print(f"  Snow={label}  BBA={out.BBA:.3f}  VIS={out.BBAVIS:.3f}  NIR={out.BBANIR:.3f}")


# =============================================================================
# 8. Clear vs overcast sky
# =============================================================================
print("\n" + "=" * 60)
print("8. Clear (direct=1) vs overcast (direct=0)")
print("=" * 60)

for direct, label in [(1, "clear"), (0, "overcast")]:
    out = run_model(preset=FYI_WINTER_BARE, solzen=60, direct=direct)
    print(f"  {label:8s}  BBA={out.BBA:.3f}  VIS={out.BBAVIS:.3f}  NIR={out.BBANIR:.3f}")


# =============================================================================
# 9. Summary table
# =============================================================================
print("\n" + "=" * 60)
print("9. Summary: all presets × SZA")
print("=" * 60)
print(f"  {'Preset':<22}  {'SZA':>4}  {'BBA':>6}  {'VIS':>6}  {'NIR':>6}")
print("  " + "-" * 48)
for name in ["FYI_WINTER_BARE", "FYI_WINTER_SNOW", "MYI_WINTER_BARE"]:
    for sza in [45, 60, 75]:
        out = run_model(preset=name, solzen=sza)
        print(f"  {name:<22}  {sza:>4}°  {out.BBA:>6.3f}  {out.BBAVIS:>6.3f}  {out.BBANIR:>6.3f}")


# =============================================================================
# Plots
# =============================================================================
if PLOT:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Three presets
    ax = axes[0]
    for name, color in [("FYI_WINTER_BARE", "firebrick"),
                         ("FYI_WINTER_SNOW", "steelblue"),
                         ("MYI_WINTER_BARE", "darkorange")]:
        out = run_model(preset=name, solzen=60)
        ax.plot(WAVELENGTHS, out.albedo, lw=1.8, color=color,
                label=f"{name}  BBA={out.BBA:.2f}")
    ax.set_title("Built-in presets (SZA=60°)")
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Spectral albedo")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.axvline(0.75, color="k", lw=0.5, ls=":", alpha=0.4)

    # Panel 2: Salinity sensitivity
    ax = axes[1]
    import matplotlib.cm as cm
    cmap = cm.plasma_r
    for i, S in enumerate([1, 4, 8, 12]):
        out = run_model(solzen=60, layer_type=4, dz=1.5, rds=500, rho=910,
                        sea_ice_salinity=S, sea_ice_temperature=-10,
                        sea_ice_bubble_radius=300)
        ax.plot(WAVELENGTHS, out.albedo, lw=1.5, color=cmap(i / 3),
                label=f"S={S} psu  BBA={out.BBA:.2f}")
    ax.set_title("Salinity sensitivity (T=−10°C)")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)

    # Panel 3: Snow thickness
    ax = axes[2]
    cmap2 = cm.YlGnBu
    for i, dz_snow in enumerate([0, 0.05, 0.10, 0.20, 0.40]):
        if dz_snow == 0:
            out = run_model(preset=FYI_WINTER_BARE, solzen=60)
            label = "bare"
        else:
            out = run_model(solzen=60, layer_type=[0,4,4],
                            dz=[dz_snow,0.05,1.45], rds=[200,500,500],
                            rho=[250,895,895], sea_ice_salinity=[None,12,8],
                            sea_ice_temperature=[None,-25,-20],
                            sea_ice_bubble_radius=[None,100,200])
            label = f"{int(dz_snow*100)} cm"
        ax.plot(WAVELENGTHS, out.albedo, lw=1.5,
                color=cmap2(0.2 + 0.7 * i / 4),
                label=f"snow={label}  BBA={out.BBA:.2f}")
    ax.set_title("Snow thickness (SZA=60°)")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 10. Melt ponds (layer_type=5)
# =============================================================================
print("\n" + "=" * 60)
print("10. Melt ponds (layer_type=5)")
print("=" * 60)
print("  layer_type=5: liquid water layer. rho=1000 kg/m³.")
print("  Floor ice: rho=895 kg/m³ (calibrated FYI), BC=1000 ppb effective LAP.")
print("  BC represents the combined optical effect of cryoconite, black carbon,")
print("  mineral dust, and algae on summer Arctic pond floors (Makshtas &")
print("  Podgorny 1996; Briegleb & Light 2007; calibrated vs Morassutti 1995).")
print()

# 10a. Depth sweep
print("  Calibrated pond model: BC=1000 ppb in floor ice (effective LAP loading).")
print("  This matches Morassutti (1995) observations: NIR RMSE=0.028, VIS RMSE=0.068.")
print()
print("  Pond depth sweep (summer FYI below, T=-5°C, rho=895, BC=1000 ppb floor):")
print(f"  {'depth':>8}  {'BBA':>6}  {'VIS':>6}  {'NIR':>6}")
print("  " + "-" * 32)
pond_results = {}
for depth in [0, 0.05, 0.10, 0.20, 0.30, 0.50]:
    if depth == 0:
        out = run_model(
            solzen=60, layer_type=[4,4], dz=[0.05,1.45], rds=[500,500], rho=[895,895],
            sea_ice_salinity=[8,6], sea_ice_temperature=[-5,-5], sea_ice_bubble_radius=[100,200],
        )
        label = "bare ice"
    else:
        out = run_model(
            solzen=60,
            layer_type=[5, 4, 4],
            dz=[depth, 0.05, 1.45],
            rds=[500, 500, 500],
            rho=[1000, 895, 895],
            sea_ice_salinity=[None, 8, 6],
            sea_ice_temperature=[None, -5, -5],
            sea_ice_bubble_radius=[None, 100, 200],
            black_carbon=[0, 1000, 0],   # 1000 ppb effective LAP in floor ice
        )
        label = f"{int(depth*100)} cm"
    pond_results[label] = out
    print(f"  {label:>8}  {out.BBA:>6.3f}  {out.BBAVIS:>6.3f}  {out.BBANIR:>6.3f}")

print()
print("  Key: NIR drops sharply even for shallow ponds (water absorbs NIR strongly).")
print("       VIS stays high (transparent water; ice underneath reflects).")

# 10b. Preset shortcut
print()
print("  Using presets:")
for name in ["FYI_POND_SHALLOW", "FYI_POND_DEEP"]:
    out = run_model(preset=name, solzen=60)
    print(f"    {name:<20}  BBA={out.BBA:.3f}  VIS={out.BBAVIS:.3f}  NIR={out.BBANIR:.3f}")

# 10c. Comparison: bare ice → shallow pond → deep pond
print()
print("  Albedo profile: bare FYI → shallow pond → deep pond")
bare = run_model(preset="FYI_WINTER_BARE", solzen=60)
shallow = run_model(preset="FYI_POND_SHALLOW", solzen=60)
deep    = run_model(preset="FYI_POND_DEEP",    solzen=60)
for name, out in [("Bare FYI", bare), ("Shallow pond (10cm)", shallow), ("Deep pond (40cm)", deep)]:
    print(f"    {name:<22}  BBA={out.BBA:.3f}  VIS={out.BBAVIS:.3f}  NIR={out.BBANIR:.3f}")


if PLOT:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Panel 1: depth sweep spectra
    ax = axes[0]
    colors_pond = ["#555", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef"]
    for i, (label, out) in enumerate(pond_results.items()):
        ax.plot(WAVELENGTHS, out.albedo, lw=1.5, color=colors_pond[i],
                label=f"{label}  BBA={out.BBA:.2f}")
    ax.set_title("Melt pond depth sensitivity (T_ice=−5°C, SZA=60°)")
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Spectral albedo")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    ax.axvline(0.7, color="k", lw=0.5, ls=":", alpha=0.4)
    ax.text(0.72, 0.95, "NIR→", fontsize=8, color="gray")
    ax.legend(fontsize=8)

    # Panel 2: NIR and VIS BBA vs depth — compare model to Morassutti (1995) obs
    ax = axes[1]
    depths  = [0, 5, 10, 20, 30, 50]  # cm
    vis_mod = [bare.BBAVIS] + [pond_results.get(f"{d} cm", bare).BBAVIS for d in [5,10,20,30,50]]
    nir_mod = [bare.BBANIR] + [pond_results.get(f"{d} cm", bare).BBANIR for d in [5,10,20,30,50]]

    # Morassutti (1995) approximate observed values by depth bin
    obs_depths = [2.5, 7.5, 15, 25, 40]   # bin midpoints cm
    obs_vis    = [0.514, 0.363, 0.351, 0.333, 0.340]
    obs_nir    = [0.224, 0.095, 0.058, 0.032, 0.035]

    ax.plot(depths, vis_mod, "o-", color="#2171b5", lw=1.5, label="Model VIS")
    ax.plot(depths, nir_mod, "s-", color="#d62728", lw=1.5, label="Model NIR")
    ax.plot(obs_depths, obs_vis, "o--", color="#6baed6", lw=1.2, alpha=0.8, label="Obs VIS (Morassutti 1995)")
    ax.plot(obs_depths, obs_nir, "s--", color="#fc8d59", lw=1.2, alpha=0.8, label="Obs NIR (Morassutti 1995)")
    ax.set_xlabel("Pond depth (cm)"); ax.set_ylabel("Albedo (400-1000 nm)")
    ax.set_title("Model vs observations — NIR matches, VIS overestimated\n"
                 "(VIS gap: model assumes clear water + white ice bottom)")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    ax.set_xlim(-1, 52)

    plt.tight_layout()
    plt.show()


# ── Section 11: Melt pond areal fraction ──────────────────────────────────────
# Arctic summer surfaces are rarely pure white ice or pure melt pond.
# blend_pond_fraction() linearly mixes two Outputs objects by pond cover:
#
#   α_total(λ) = (1 − f) · α_ice(λ)  +  f · α_pond(λ)
#
# This is the standard surface albedo mixing model used in GCMs (Briegleb &
# Light 2007).  Two interfaces are available:
#
#   (A) Two-step: compute ice and pond separately, then blend.
#   (B) One-step: pass pond_fraction directly to run_model().

print("\n── Section 11: Melt pond areal fraction ──")

# (A) Manual blending
summer_ice   = run_model(preset=FYI_SUMMER_BARE,  solzen=60)
shallow_pond = run_model(preset=FYI_POND_SHALLOW, solzen=60)

print(f"  Pure white ice  BBA = {summer_ice.BBA:.3f}")
print(f"  Pure melt pond  BBA = {shallow_pond.BBA:.3f}")
print()
for f in [0.10, 0.20, 0.30, 0.40]:
    mixed = blend_pond_fraction(summer_ice, shallow_pond, f=f)
    print(f"  pond_fraction={f:.0%}  →  BBA={mixed.BBA:.3f}  "
          f"VIS={mixed.BBAVIS:.3f}  NIR={mixed.BBANIR:.3f}")

# (B) Convenience: pond_fraction kwarg on run_model()
# Internally runs FYI_POND_SHALLOW at pond_depth and blends automatically.
print()
mixed_auto = run_model(
    preset="FYI_SUMMER_BARE",
    solzen=60,
    pond_fraction=0.20,
    pond_depth=0.15,
)
print(f"  run_model(pond_fraction=0.20)  BBA = {mixed_auto.BBA:.3f}")

# Pond depth sensitivity at fixed fraction
print("\n  Pond depth effect at 20% cover:")
for depth_cm in [5, 10, 20, 40]:
    r = run_model(preset="FYI_SUMMER_BARE", solzen=60,
                  pond_fraction=0.20, pond_depth=depth_cm / 100)
    print(f"    pond_depth={depth_cm:2d} cm  BBA={r.BBA:.3f}  NIR={r.BBANIR:.3f}")

if PLOT:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Panel 1: blended spectra at different pond fractions
    ax = axes[0]
    fracs = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    colors_frac = ["#08306b", "#2171b5", "#4292c6", "#74c476", "#fd8d3c", "#d62728"]
    for f, col in zip(fracs, colors_frac):
        mixed = blend_pond_fraction(summer_ice, shallow_pond, f)
        ax.plot(WAVELENGTHS, mixed.albedo, color=col, lw=1.5,
                label=f"f={f:.0%}  BBA={mixed.BBA:.3f}")
    ax.axvline(0.7, color="k", lw=0.5, ls=":", alpha=0.4)
    ax.text(0.72, 0.92, "NIR→", fontsize=8, color="gray")
    ax.set_title("Summer Arctic: blended ice + pond spectra\n"
                 "FYI_SUMMER_BARE × (1−f) + FYI_POND_SHALLOW × f")
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Spectral albedo")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # Panel 2: BBA, VIS, NIR vs pond fraction
    ax = axes[1]
    bba_vals = [blend_pond_fraction(summer_ice, shallow_pond, f).BBA    for f in fracs]
    vis_vals = [blend_pond_fraction(summer_ice, shallow_pond, f).BBAVIS for f in fracs]
    nir_vals = [blend_pond_fraction(summer_ice, shallow_pond, f).BBANIR for f in fracs]
    pct = [f * 100 for f in fracs]
    ax.plot(pct, bba_vals, "o-", color="#2c2c2c", lw=2, label="BBA (400–5000 nm)")
    ax.plot(pct, vis_vals, "s-", color="#3a7dc9", lw=1.5, label="VIS (400–700 nm)")
    ax.plot(pct, nir_vals, "^-", color="#e07b39", lw=1.5, label="NIR (700–5000 nm)")
    ax.set_xlabel("Pond areal fraction (%)")
    ax.set_ylabel("Flux-weighted broadband albedo")
    ax.set_title("Albedo sensitivity to pond fraction\n"
                 "SZA=60°, FYI_SUMMER_BARE + FYI_POND_SHALLOW (15 cm)")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    ax.set_xlim(-1, 52); ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.show()
