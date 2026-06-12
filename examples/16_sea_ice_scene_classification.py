#!/usr/bin/env python3
"""Sea ice scene classification — young ice, open water, quality flags, batch.

This example demonstrates the v0.4 retrieval capabilities:
  1. Young ice (layer_type=6): albedo vs thickness, classification, WMO codes
  2. Open water: analytical model, classification of a dark spectrum
  3. Quality flags on every retrieval
  4. Batch retrieval over a small synthetic "scene" with spatial exports

Prerequisites
-------------
    python scripts/build_sea_ice_emulators.py          # emulators
    pip install biosnicar[geo]                         # for part 4 exports
"""

import numpy as np

from biosnicar import run_model
from biosnicar.sea_ice.open_water import OpenWaterModel
from biosnicar.sea_ice.retrieve import retrieve_sea_ice, retrieve_sea_ice_batch

rng = np.random.default_rng(42)
NOISE = 0.004  # 1-sigma albedo noise added to every synthetic observation


# ── 1. Young ice ─────────────────────────────────────────────────────────────

print("=" * 70)
print("1. Young ice — albedo vs thickness (Grenfell & Maykut regime)")
print("=" * 70)
for d_cm in (1, 3, 8, 12, 25):
    out = run_model(layer_type=6, ice_thickness=d_cm / 100.0,
                    sea_ice_temperature=-10, sea_ice_salinity=25,
                    ocean_albedo=0.04, solzen=60, direct=1)
    print(f"  {d_cm:3d} cm  BBA = {out.BBA:.3f}")

obs = run_model(layer_type=6, ice_thickness=0.08, sea_ice_temperature=-8,
                sea_ice_salinity=25, ocean_albedo=0.04, solzen=70, direct=1)
noisy = np.clip(np.asarray(obs.albedo) + rng.normal(0, NOISE, 480), 0, 1)

# known_month=11: freeze-up — young ice is a candidate and gets a thin-ice prior
result = retrieve_sea_ice(observed=noisy, solzen=70, direct=1, known_month=11)
print(f"\n  classified: {result.surface_type}  (confidence {result.confidence:.2f})")
# NB the freeze-up prior (mean 5 cm) pulls the estimate low at this noise level
print(f"  retrieved thickness: {result.parameters['ice_thickness'] * 100:.1f} cm (truth 8.0)")
print(f"  WMO: {result.to_wmo().stage_of_development}")
print(f"  SIGRID-3: {result.to_sigrid3().full_code}")

# ── 2. Open water ────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("2. Open water — analytical model, no training data")
print("=" * 70)
ow = OpenWaterModel()
water = np.clip(ow.predict(solzen=60, wind_speed_ms=6.0)
                + rng.normal(0, NOISE, 480), 0, 1)
result = retrieve_sea_ice(observed=water, solzen=60, direct=1, known_month=9)
print(f"  classified: {result.surface_type}  (confidence {result.confidence:.2f})")
print(f"  retrieved wind speed: {result.parameters['wind_speed_ms']:.1f} m/s (truth 6.0)")

# ── 3. Quality flags ─────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("3. Quality flags — a deliberately bad observation")
print("=" * 70)
garbage = np.clip(0.5 + 0.3 * np.sin(np.linspace(0, 20, 480)), 0, 1)
result = retrieve_sea_ice(observed=garbage, solzen=60, direct=1)
print(f"  best-effort type: {result.surface_type}  cost={result.cost:.3f}")
for flag, on in result.quality_flag_description().items():
    if on:
        print(f"    FLAG: {flag}")

# ── 4. Batch retrieval over a tiny scene ─────────────────────────────────────

print("\n" + "=" * 70)
print("4. Batch retrieval — 2x3 pixel scene with one masked pixel")
print("=" * 70)
try:
    import xarray  # noqa: F401 — geo extras check
except ImportError:
    print("  (skipped — install with: pip install biosnicar[geo])")
    raise SystemExit(0)

pond = run_model(layer_type=[5, 4, 4], dz=[0.25, 0.05, 1.40],
                 rds=[500, 500, 500], rho=[1000, 850, 910],
                 sea_ice_salinity=[None, 8, 6], sea_ice_temperature=[None, -4, -4],
                 sea_ice_bubble_radius=[None, 200, 500],
                 black_carbon=[0, 100, 0], solzen=60, direct=1)
# Regenerate the young-ice pixel at the scene's solzen (observations and
# the fixed retrieval geometry must match).
yi60 = run_model(layer_type=6, ice_thickness=0.08, sea_ice_temperature=-8,
                 sea_ice_salinity=25, ocean_albedo=0.04, solzen=60, direct=1)
yi_pixel = np.clip(np.asarray(yi60.albedo) + rng.normal(0, NOISE, 480), 0, 1)
pixels = np.array([
    water, np.clip(np.asarray(pond.albedo) + rng.normal(0, NOISE, 480), 0, 1),
    yi_pixel, water,
    np.full(480, np.nan),  # masked (cloud) pixel
    yi_pixel,
]).reshape(2, 3, 480)
latlon = np.dstack(np.meshgrid(np.linspace(15.0, 15.002, 3),
                               np.linspace(78.0, 78.001, 2))[::-1])

# known_month matters here too: without it, dark young ice can be fitted by
# a warm/dirty FYI_bare configuration that the freeze-up priors rule out.
scene = retrieve_sea_ice_batch(
    pixels, n_jobs=2, chunksize=3,
    spatial_coords=latlon, crs="EPSG:4326",
    solzen=60, direct=1, known_month=11,
)
print(scene.summary())
scene.to_netcdf("/tmp/sea_ice_scene.nc")
scene.to_h3_geojson("/tmp/sea_ice_scene.geojson", resolution=7)
print("  wrote /tmp/sea_ice_scene.nc and /tmp/sea_ice_scene.geojson")
