#!/usr/bin/env python3
"""Platform band convolution examples.

Demonstrates ``.to_platform()`` chaining on ``run_model()`` and
``parameter_sweep()``, standalone ``to_platform()`` calls, and GCM bands.
"""

from biosnicar import run_model, to_platform
from biosnicar.drivers.sweep import parameter_sweep

# ======================================================================
# Example 1: Chain .to_platform() on run_model()
# ======================================================================
print("=== Example 1: run_model() -> Sentinel-2 bands ===\n")
s2 = run_model(solzen=50, rds=1000).to_platform("sentinel2")
print(f"  Platform: {s2.platform}")
print(f"  Bands:    {s2.band_names}")
print(f"  Indices:  {s2.index_names}")
print(f"  B3 (green):  {s2.B3:.4f}")
print(f"  B4 (red):    {s2.B4:.4f}")
print(f"  B11 (SWIR):  {s2.B11:.4f}")
print(f"  NDSI:        {s2.NDSI:.4f}")

# ======================================================================
# Example 2: Other satellite platforms
# ======================================================================
print("\n=== Example 2: Other platforms ===\n")
outputs = run_model(solzen=50, rds=1000)
for platform in ["sentinel2", "landsat8", "modis", "sentinel3", "planetscope"]:
    result = outputs.to_platform(platform)
    print(
        f"  {platform:12s}  bands={len(result.band_names):2d}  "
        f"indices={len(result.index_names)}"
    )

# ======================================================================
# Example 3: GCM broadband outputs
# ======================================================================
print("\n=== Example 3: GCM band outputs ===\n")
cesm = outputs.to_platform("cesm2band")
print(f"  CESM 2-band:  vis={cesm.vis:.4f}  nir={cesm.nir:.4f}")

mar = outputs.to_platform("mar")
print(
    f"  MAR 4-band:   sw1={mar.sw1:.4f}  sw2={mar.sw2:.4f}  "
    f"sw3={mar.sw3:.4f}  sw4={mar.sw4:.4f}"
)

# ======================================================================
# Example 4: Standalone to_platform()
# ======================================================================
print("\n=== Example 4: Standalone to_platform() call ===\n")
outputs = run_model(solzen=50, rds=500)
s2 = to_platform(outputs.albedo, "sentinel2", flx_slr=outputs.flx_slr)
print(f"  sentinel2:    B3={s2.B3:.4f}  B4={s2.B4:.4f}  NDSI={s2.NDSI:.4f}")
ps = to_platform(outputs.albedo, "planetscope", flx_slr=outputs.flx_slr)
print(f"  planetscope:  B4={ps.B4:.4f}  B7={ps.B7:.4f}  NDSI={ps.NDSI:.4f}  NDRE={ps.NDRE:.4f}")

# ======================================================================
# Example 5: Parameter sweep -> band convolution
# ======================================================================
print("\n=== Example 5: Sweep -> Sentinel-2 bands ===\n")
df = parameter_sweep(
    params={"rds": [200, 500, 1000, 5000], "solzen": [40, 60]},
    progress=False,
).to_platform("sentinel2")

print(
    df[["rds", "solzen", "BBA", "B3", "B11", "NDSI"]].to_string(
        index=False, float_format="{:.4f}".format
    )
)

# ======================================================================
# Example 6: Sweep -> multiple platforms
# ======================================================================
print("\n\n=== Example 6: Sweep -> multiple platforms ===\n")
df_multi = parameter_sweep(
    params={"rds": [500, 1000]},
    progress=False,
).to_platform("sentinel2", "modis", "planetscope")

# Columns are prefixed when using multiple platforms
print("Columns:", [c for c in df_multi.columns if "B3" in c or "NDSI" in c or "NDRE" in c])
print(
    df_multi[
        ["rds", "BBA", "sentinel2_B3", "modis_B4", "sentinel2_NDSI", "modis_NDSI",
         "planetscope_B4", "planetscope_NDSI", "planetscope_NDRE"]
    ].to_string(index=False, float_format="{:.4f}".format)
)
