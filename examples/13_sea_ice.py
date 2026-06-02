#!/usr/bin/env python3
"""Sea ice albedo — demo of the biosnicar.sea_ice extension.

Covers:
  1. Using built-in presets (FYI bare, FYI snow-covered, MYI bare)
  2. Building a custom column layer by layer
  3. Sensitivity to salinity and temperature
  4. Sensitivity to solar zenith angle
  5. Effect of snow cover thickness
  6. Comparing sea ice types side-by-side

Layer type mapping reminder
---------------------------
  0 — granular snow / ice grains
  1 — solid glacier ice with Fresnel reflection
  4 — sea ice (layer_type=4, NOT 2 — that was already taken)

Note: sea ice requires the adding-doubling solver (default). The Toon solver
does not handle the Fresnel air-ice interface needed for sea ice correctly.
"""

import numpy as np
import matplotlib.pyplot as plt

from biosnicar.sea_ice.api import SeaIceColumn, SeaIceLayer, SnowLayer
from biosnicar.sea_ice.presets import FYI_WINTER_BARE, FYI_WINTER_SNOW, MYI_WINTER_BARE

WAVELENGTHS = np.arange(0.205, 4.999, 0.01)  # 480-band grid, μm
PLOT = True  # set False to suppress figures


# =============================================================================
# 1. Built-in presets
# =============================================================================
print("=" * 60)
print("1. Built-in presets (SZA=60°, sub-Arctic winter, clear sky)")
print("=" * 60)

results = {}
for preset in [FYI_WINTER_BARE, FYI_WINTER_SNOW, MYI_WINTER_BARE]:
    col = SeaIceColumn.from_preset(preset)
    r = col.compute_albedo(sza_deg=60, atmosphere="sub_arctic_winter", sky="clear")
    results[preset.name] = r
    print(
        f"  {preset.name:<22}  BBA={r.broadband:.3f}"
        f"  VIS={r.visible:.3f}  NIR={r.nir:.3f}"
    )

if PLOT:
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {"FYI_WINTER_BARE": "firebrick", "FYI_WINTER_SNOW": "steelblue",
              "MYI_WINTER_BARE": "darkorange"}
    for name, r in results.items():
        ax.plot(WAVELENGTHS, r.spectrum, lw=1.8, label=f"{name}  BBA={r.broadband:.2f}",
                color=colors[name])
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Spectral albedo")
    ax.set_xlim(0.3, 2.5)
    ax.set_ylim(0, 1.05)
    ax.axvline(0.75, color="k", ls="--", lw=0.7, alpha=0.4, label="VIS/NIR boundary")
    ax.legend(fontsize=9)
    ax.set_title("Sea ice presets — SZA=60°")
    fig.tight_layout()
    plt.show()


# =============================================================================
# 2. Custom column
# =============================================================================
print("\n" + "=" * 60)
print("2. Custom column with snow + two sea-ice layers")
print("=" * 60)

custom_col = SeaIceColumn(layers=[
    # Thin snow layer (fresh water in MVP; salty snow deferred to v0.2)
    SnowLayer(thickness_m=0.08, density_kg_m3=280, grain_radius_um=150),
    # Cold, saline FYI surface
    SeaIceLayer(
        thickness_m=0.10,
        temperature_C=-22,
        salinity_psu=10,
        density_kg_m3=918,
        bubble_radius_um=120,
        layer_class="FYI",
    ),
    # Warmer bulk ice
    SeaIceLayer(
        thickness_m=1.80,
        temperature_C=-8,
        salinity_psu=5,
        density_kg_m3=912,
        bubble_radius_um=250,
        layer_class="FYI",
    ),
])

r_custom = custom_col.compute_albedo(sza_deg=65, sky="clear")
print(f"  Custom 3-layer column:   BBA={r_custom.broadband:.3f}"
      f"  VIS={r_custom.visible:.3f}  NIR={r_custom.nir:.3f}")


# =============================================================================
# 3. Sensitivity to salinity
# =============================================================================
print("\n" + "=" * 60)
print("3. Salinity sensitivity (T=−10°C, rho=910 kg/m³, bbl=300 µm)")
print("=" * 60)

salinities = [1, 4, 8, 12]
sal_results = {}
for S in salinities:
    col = SeaIceColumn(layers=[
        SeaIceLayer(thickness_m=1.5, temperature_C=-10, salinity_psu=S,
                    density_kg_m3=910, bubble_radius_um=300)
    ])
    r = col.compute_albedo(sza_deg=60)
    sal_results[S] = r
    print(f"  S={S:2d} psu  BBA={r.broadband:.3f}  VIS={r.visible:.3f}  NIR={r.nir:.3f}")


# =============================================================================
# 4. Sensitivity to temperature
# =============================================================================
print("\n" + "=" * 60)
print("4. Temperature sensitivity (S=8 psu, rho=910 kg/m³, bbl=300 µm)")
print("=" * 60)

temperatures = [-2, -5, -10, -20, -30]
temp_results = {}
for T in temperatures:
    col = SeaIceColumn(layers=[
        SeaIceLayer(thickness_m=1.5, temperature_C=T, salinity_psu=8,
                    density_kg_m3=910, bubble_radius_um=300)
    ])
    r = col.compute_albedo(sza_deg=60)
    temp_results[T] = r
    print(f"  T={T:3d} °C  BBA={r.broadband:.3f}  VIS={r.visible:.3f}  NIR={r.nir:.3f}")

if PLOT:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    cmap = plt.cm.plasma_r
    for idx, (S, r) in enumerate(sal_results.items()):
        ax.plot(WAVELENGTHS, r.spectrum, lw=1.5,
                color=cmap(idx / len(sal_results)),
                label=f"S={S} psu  (BBA={r.broadband:.2f})")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Spectral albedo")
    ax.set_title("Salinity sensitivity (T=−10°C)")
    ax.legend(fontsize=9)

    ax = axes[1]
    cmap2 = plt.cm.cool
    for idx, (T, r) in enumerate(temp_results.items()):
        ax.plot(WAVELENGTHS, r.spectrum, lw=1.5,
                color=cmap2(idx / len(temp_results)),
                label=f"T={T}°C  (BBA={r.broadband:.2f})")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Spectral albedo")
    ax.set_title("Temperature sensitivity (S=8 psu)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    plt.show()


# =============================================================================
# 5. Solar zenith angle
# =============================================================================
print("\n" + "=" * 60)
print("5. Solar zenith angle (FYI_WINTER_BARE)")
print("=" * 60)

sza_results = {}
for sza in [30, 45, 60, 75, 85]:
    col = SeaIceColumn.from_preset(FYI_WINTER_BARE)
    r = col.compute_albedo(sza_deg=sza)
    sza_results[sza] = r
    print(f"  SZA={sza:2d}°  BBA={r.broadband:.3f}  VIS={r.visible:.3f}  NIR={r.nir:.3f}")

if PLOT:
    fig, ax = plt.subplots(figsize=(9, 4))
    cmap3 = plt.cm.viridis
    for idx, (sza, r) in enumerate(sza_results.items()):
        ax.plot(WAVELENGTHS, r.spectrum, lw=1.5,
                color=cmap3(idx / len(sza_results)),
                label=f"SZA={sza}°  BBA={r.broadband:.2f}")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Spectral albedo")
    ax.set_title("SZA sensitivity — FYI bare")
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()


# =============================================================================
# 6. Snow cover thickness
# =============================================================================
print("\n" + "=" * 60)
print("6. Snow cover thickness on FYI (SZA=60°)")
print("=" * 60)

snow_thicknesses = [0, 0.05, 0.10, 0.20, 0.40]
snow_results = {}
for dz in snow_thicknesses:
    if dz == 0:
        col = SeaIceColumn.from_preset(FYI_WINTER_BARE)
    else:
        col = SeaIceColumn(layers=[
            SnowLayer(thickness_m=dz, density_kg_m3=300, grain_radius_um=200),
            SeaIceLayer(thickness_m=0.05, temperature_C=-25, salinity_psu=12,
                        density_kg_m3=920, bubble_radius_um=100),
            SeaIceLayer(thickness_m=1.45, temperature_C=-10, salinity_psu=8,
                        density_kg_m3=915, bubble_radius_um=200),
        ])
    r = col.compute_albedo(sza_deg=60)
    snow_results[dz] = r
    label = f"{int(dz*100):3d} cm"
    print(f"  Snow={label}  BBA={r.broadband:.3f}  VIS={r.visible:.3f}  NIR={r.nir:.3f}")

if PLOT:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    cmap4 = plt.cm.YlGnBu
    for idx, (dz, r) in enumerate(snow_results.items()):
        label = f"0 cm (bare)" if dz == 0 else f"{int(dz*100)} cm"
        ax.plot(WAVELENGTHS, r.spectrum, lw=1.5,
                color=cmap4(0.2 + 0.7 * idx / max(1, len(snow_results) - 1)),
                label=f"snow={label}  BBA={r.broadband:.2f}")
    ax.set_xlim(0.3, 2.5); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Spectral albedo")
    ax.set_title("Snow thickness sensitivity")
    ax.legend(fontsize=9)

    # BBA vs snow thickness summary bar
    ax = axes[1]
    labels = [f"{int(dz*100)} cm" if dz > 0 else "bare" for dz in snow_thicknesses]
    bbas = [r.broadband for r in snow_results.values()]
    ax.bar(labels, bbas, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Snow thickness")
    ax.set_ylabel("Broadband albedo")
    ax.set_ylim(0, 1.0)
    ax.set_title("BBA vs snow depth")
    ax.axhline(bbas[0], color="firebrick", ls="--", lw=1, label="bare ice BBA")
    ax.legend(fontsize=9)

    fig.tight_layout()
    plt.show()


# =============================================================================
# 7. Direct vs diffuse sky
# =============================================================================
print("\n" + "=" * 60)
print("7. Clear sky vs overcast — FYI bare, SZA=60°")
print("=" * 60)

for sky in ["clear", "cloudy"]:
    col = SeaIceColumn.from_preset(FYI_WINTER_BARE)
    r = col.compute_albedo(sza_deg=60, sky=sky)
    print(f"  {sky:7s}  BBA={r.broadband:.3f}  VIS={r.visible:.3f}  NIR={r.nir:.3f}")


# =============================================================================
# 8. Summary table
# =============================================================================
print("\n" + "=" * 60)
print("8. Summary: all presets × SZA")
print("=" * 60)
print(f"  {'Preset':<22}  {'SZA':>4}  {'BBA':>6}  {'VIS':>6}  {'NIR':>6}")
print("  " + "-" * 48)
for preset in [FYI_WINTER_BARE, FYI_WINTER_SNOW, MYI_WINTER_BARE]:
    for sza in [45, 60, 75]:
        col = SeaIceColumn.from_preset(preset)
        r = col.compute_albedo(sza_deg=sza)
        print(f"  {preset.name:<22}  {sza:>4}°  {r.broadband:>6.3f}  {r.visible:>6.3f}  {r.nir:>6.3f}")
