#!/usr/bin/env python3
"""PlanetScope SuperDove band convolution — forward and inverse modes.

Demonstrates the full workflow for PlanetScope SuperDove imagery:

  Part 1  Forward model  —  run_model() → PlanetScope bands and indices
  Part 2  Parameter sweep  —  explore how ice properties affect PS bands
  Part 3  Inverse retrieval  —  recover ice properties from synthetic PS observations

PlanetScope SuperDove has 8 bands (431–885 nm) with square (tophat) SRFs.
All bands fall within the visible–NIR range, so the retrieval is most
informative about grain size (via NIR), impurity loading (via blue/green),
and glacier algae (via red-edge NDRE using B7 at 697–713 nm).

The red-edge band B7 is particularly valuable: glacier algae (Sanguina) have
a chlorophyll absorption feature near 680 nm, causing a characteristic
red-edge reflectance signal that NDRE (B8−B7)/(B8+B7) captures well.

Usage
-----
    python examples/12_planetscope.py

Requires the pre-built emulator at data/emulators/glacier_ice_8_param_default.npz
(Part 3 only).  Parts 1 and 2 have no extra dependencies.
"""

import numpy as np

from biosnicar import DATA_DIR, run_model, to_platform
from biosnicar.drivers.sweep import parameter_sweep

PLOT = True

# ======================================================================
# Part 1: Forward model → PlanetScope bands
# ======================================================================
print("=" * 60)
print("Part 1: Forward model → PlanetScope bands")
print("=" * 60)

# Clean glacier ice at 50° solar zenith angle
outputs = run_model(solzen=50, rds=1000, layer_type=1)
ps = outputs.to_platform("planetscope")

print(f"\nPlatform:  {ps.platform}")
print(f"Bands:     {ps.band_names}")
print(f"Indices:   {ps.index_names}")
print()
print("Band albedos (clean ice, rds=1000 µm, SZA=50°):")
band_labels = {
    "B1": "Coastal Blue (431–452 nm)",
    "B2": "Blue         (465–515 nm)",
    "B3": "Green I      (513–549 nm)",
    "B4": "Green        (547–583 nm)",
    "B5": "Yellow       (600–620 nm)",
    "B6": "Red          (650–680 nm)",
    "B7": "Red-Edge     (697–713 nm)",
    "B8": "NIR          (845–885 nm)",
}
for band, label in band_labels.items():
    print(f"  {label}: {getattr(ps, band):.4f}")

print()
print("Spectral indices:")
print(f"  NDSI  (B4−B8)/(B4+B8): {ps.NDSI:+.4f}  (snow/ice detection)")
print(f"  NDVI  (B8−B6)/(B8+B6): {ps.NDVI:+.4f}  (vegetation proxy)")
print(f"  NDRE  (B8−B7)/(B8+B7): {ps.NDRE:+.4f}  (glacier algae proxy)")

# Dirty ice — heavy glacier algae loading
print()
print("Band albedos (algal bloom, glacier_algae=500000 ppb):")
ps_algae = run_model(
    solzen=50, rds=1000, layer_type=1, glacier_algae=500000
).to_platform("planetscope")
for band in ["B4", "B6", "B7", "B8"]:
    print(f"  {band}: {getattr(ps_algae, band):.4f}")
print(f"  NDRE (algal bloom): {ps_algae.NDRE:+.4f}")
print(f"  NDRE (clean ice):   {ps.NDRE:+.4f}")

# ======================================================================
# Part 2: Parameter sweep → PlanetScope bands
# ======================================================================
print()
print("=" * 60)
print("Part 2: Parameter sweep → PlanetScope bands")
print("=" * 60)

# Effect of grain radius and algae loading on key bands
df = parameter_sweep(
    params={
        "rds": [200, 500, 1000, 3000],
        "glacier_algae": [0, 50000, 200000, 500000],
        "solzen": [50],
        "layer_type": [1],
    },
    progress=False,
).to_platform("planetscope")

print()
print("Grain radius and glacier algae → PlanetScope bands")
print(
    df[["rds", "glacier_algae", "BBA", "B4", "B7", "B8", "NDSI", "NDRE"]].to_string(
        index=False, float_format="{:.4f}".format
    )
)

# Effect of solar zenith angle
print()
df_sza = parameter_sweep(
    params={"solzen": [30, 45, 60, 70], "rds": [1000], "layer_type": [1]},
    progress=False,
).to_platform("planetscope")

print("\nSolar zenith angle → PlanetScope bands (clean ice, rds=1000 µm)")
print(
    df_sza[["solzen", "BBA", "B1", "B4", "B8", "NDSI"]].to_string(
        index=False, float_format="{:.4f}".format
    )
)

# ======================================================================
# Part 3: Inverse retrieval from synthetic PlanetScope observations
# ======================================================================
print()
print("=" * 60)
print("Part 3: Inverse retrieval from PlanetScope observations")
print("=" * 60)

try:
    from biosnicar.emulator import Emulator
    from biosnicar.inverse import retrieve
    from biosnicar.inverse.result import _compute_ssa

    emu = Emulator.load(DATA_DIR / "emulators" / "glacier_ice_8_param_default.npz")

    # Fixed parameters that are known a priori
    fixed = {"solzen": 50, "direct": 1, "dust": 1000, "snow_algae": 0}

    # True ice surface — observation generated from the full forward model
    true_params = dict(rds=1000, rho=600, black_carbon=5000, glacier_algae=100000)
    true_outputs = run_model(**true_params, **fixed, layer_type=1)
    true_albedo = np.array(true_outputs.albedo, dtype=np.float64)
    true_ssa = _compute_ssa(true_params["rds"], true_params["rho"])

    # Simulate what PlanetScope would observe over this surface
    ps_obs = to_platform(true_albedo, "planetscope", flx_slr=emu.flx_slr)
    print(f"\nTrue SSA: {true_ssa:.4f} m²/kg  (rds={true_params['rds']}, rho={true_params['rho']})")
    print(f"True glacier_algae: {true_params['glacier_algae']} ppb")
    print(f"True NDRE: {ps_obs.NDRE:+.4f}")

    # PlanetScope bands to use in retrieval.
    # B1 (coastal blue) captures algae absorption; B4/B8 span the NDSI range;
    # B7 (red-edge) and B8 (NIR) together constrain glacier algae via NDRE.
    band_names = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
    obs_values = np.array([getattr(ps_obs, b) for b in band_names])

    print("\nSynthetic PlanetScope observation:")
    for name, val in zip(band_names, obs_values):
        label = band_labels[name].strip()
        print(f"  {name} ({label.split('(')[1].rstrip(')')}): {val:.4f}")

    # ── Example 3a: Basic SSA + impurity retrieval ──────────────────
    print("\n--- Example 3a: SSA + impurity retrieval ---\n")
    result = retrieve(
        observed=obs_values,
        parameters=["ssa", "black_carbon", "glacier_algae"],
        emulator=emu,
        platform="planetscope",
        observed_band_names=band_names,
        fixed_params=fixed,
    )
    print(result.summary())
    print(f"\n  True SSA:           {true_ssa:.4f}")
    print(f"  Retrieved SSA:      {result.best_fit['ssa']:.4f}")
    print(f"  True algae:         {true_params['glacier_algae']:.0f} ppb")
    print(f"  Retrieved algae:    {result.best_fit['glacier_algae']:.0f} ppb")

    # ── Example 3b: With per-band measurement uncertainty ───────────
    print("\n--- Example 3b: With per-band uncertainty ---\n")
    # PlanetScope SuperDove surface reflectance: ~2% in VIS, ~3% in NIR/red-edge
    obs_unc = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.025, 0.03])

    result2 = retrieve(
        observed=obs_values,
        parameters=["ssa", "black_carbon", "glacier_algae"],
        emulator=emu,
        platform="planetscope",
        observed_band_names=band_names,
        fixed_params=fixed,
        obs_uncertainty=obs_unc,
    )
    for name in result2.best_fit:
        print(f"  {name:25s} = {result2.best_fit[name]:10.4f}")

    # ── Optional: plot ───────────────────────────────────────────────
    if PLOT:
        import matplotlib.pyplot as plt

        wavelengths = np.arange(0.205, 4.999, 0.01)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: full spectrum with retrieved fit
        ax = axes[0]
        ax.plot(wavelengths, true_albedo, "k-", alpha=0.6, label="True spectrum")
        ax.plot(wavelengths, result.predicted_albedo, "r--", label="Retrieved (no unc.)")
        ax.plot(wavelengths, result2.predicted_albedo, "b:", label="Retrieved (with unc.)")
        # Mark PlanetScope bands
        band_centers = {
            "B1": 0.4415, "B2": 0.490, "B3": 0.531, "B4": 0.565,
            "B5": 0.610, "B6": 0.665, "B7": 0.705, "B8": 0.865,
        }
        for bname, val in zip(band_names, obs_values):
            ax.plot(band_centers[bname], val, "gs", ms=8, zorder=5)
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Albedo")
        ax.set_xlim(0.38, 1.0)
        ax.set_ylim(0, 1.05)
        ax.set_title("PlanetScope retrieval — observed bands (green squares)")
        ax.legend(fontsize=8)

        # Right: NDRE vs glacier_algae sensitivity
        ax2 = axes[1]
        algae_range = np.logspace(3, 6, 30)
        ndre_vals = []
        for alg in algae_range:
            out = run_model(solzen=50, rds=1000, layer_type=1, glacier_algae=int(alg))
            ndre_vals.append(out.to_platform("planetscope").NDRE)
        ax2.semilogx(algae_range, ndre_vals, "g-", linewidth=2)
        ax2.axhline(ps_obs.NDRE, color="k", linestyle="--", label=f"Observed NDRE ({ps_obs.NDRE:.3f})")
        ax2.set_xlabel("Glacier algae concentration (ppb)")
        ax2.set_ylabel("NDRE")
        ax2.set_title("NDRE sensitivity to glacier algae loading")
        ax2.legend()

        fig.tight_layout()
        plt.show()

except FileNotFoundError:
    print(
        "\n  Emulator file not found — skipping inverse retrieval.\n"
        "  Build the emulator first with:  python examples/04_emulator_build.py"
    )
