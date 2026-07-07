#!/usr/bin/env python3
"""Using a pre-built emulator for fast predictions.

Demonstrates loading the default emulator, ``predict()``,
``predict_batch()``, ``run_emulator()``, speed comparison with the
forward model, and ``.to_platform()`` chaining.
"""

import time

import numpy as np

from biosnicar import run_model, run_emulator
from biosnicar.emulator import Emulator

PLOT = False

# ======================================================================
# Example 1: Load the default emulator
# ======================================================================
print("=== Example 1: Load default emulator ===\n")
from biosnicar import DATA_DIR

emu = Emulator.load(DATA_DIR / "emulators" / "glacier_ice_8_param_default.npz")
print(f"  {emu!r}")
print(f"  Parameters: {emu.param_names}")
print(f"  Bounds:     {emu.bounds}")

result = emu.verify(n_points=1000)
print(result.summary())

# ======================================================================
# Example 2: Single prediction with predict()
# ======================================================================
print("\n=== Example 2: predict() ===\n")
albedo = emu.predict(
    rds=1000, rho=600, black_carbon=5000, snow_algae=0, glacier_algae=50000,
    dust=10000, direct=1, solzen=50,
)
print(f"  Albedo shape: {albedo.shape}")
print(f"  Albedo at 0.5 um: {albedo[29]:.4f}")
print(f"  Albedo at 1.5 um: {albedo[129]:.4f}")

# ======================================================================
# Example 3: Batch prediction
# ======================================================================
print("\n=== Example 3: predict_batch() ===\n")
# param order matches emu.param_names
points = np.array(
    [   # rds, rho, BC, snow_algae, glacier_algae, dust, direct, solzen
        [500, 400, 0, 0, 0, 0, 1, 30],
        [1000, 600, 5000, 0, 50000, 10000, 1, 40],
        [2000, 800, 20000, 0, 200000, 10000, 1, 50],
        [5000, 917, 50000, 0, 500000, 10000, 1, 60],
    ]
)
batch_albedo = emu.predict_batch(points)
print(f"  Input shape:  {points.shape}")
print(f"  Output shape: {batch_albedo.shape}")
for i, row in enumerate(points):
    bba = float(np.sum(emu.flx_slr * batch_albedo[i]) / np.sum(emu.flx_slr))
    print(
        f"  Point {i}: rds={row[0]:.0f}, rho={row[1]:.0f}, "
        f"BC={row[2]:.0f}, GA={row[4]:.0f} -> BBA={bba:.4f}"
    )

# # ======================================================================
# # Example 4: run_emulator() — Outputs-compatible interface
# # ======================================================================
print("\n=== Example 4: run_emulator() ===\n")
outputs = run_emulator(
    emu, rds=1000, rho=600, black_carbon=5000, snow_algae=0, glacier_algae=50000,
    dust=10000, direct=1, solzen=50,
)
print(f"  BBA:    {outputs.BBA:.4f}")
print(f"  BBAVIS: {outputs.BBAVIS:.4f}")
print(f"  BBANIR: {outputs.BBANIR:.4f}")

# # ======================================================================
# # Example 5: Speed comparison
# # ======================================================================
print("\n=== Example 5: Speed comparison ===\n")
params = dict(
    rds=1000, rho=600, black_carbon=1000, snow_algae=0, glacier_algae=50000,
    dust=10000, direct=1, solzen=50,
)

# Emulator speed (100 calls)
t0 = time.time()
for _ in range(100):
    emu.predict(**params)
emu_time = (time.time() - t0) / 100

# Forward model speed (3 calls)
t0 = time.time()
for _ in range(3):
    run_model(layer_type=1, **params)
fm_time = (time.time() - t0) / 3

print(f"  Emulator:      {emu_time * 1e6:.0f} us per call")
print(f"  Forward model: {fm_time * 1e3:.0f} ms per call")
print(f"  Speedup:       {fm_time / emu_time:.0f}x")

# # ======================================================================
# # Example 6: Chain with .to_platform()
# # ======================================================================
print("\n=== Example 6: run_emulator() -> .to_platform() ===\n")
s2 = run_emulator(
    emu, rds=1000, rho=600, black_carbon=5000, snow_algae=0, glacier_algae=50000,
    dust=10000, direct=1, solzen=50,
).to_platform("sentinel2")
print(f"  B3={s2.B3:.4f}  B11={s2.B11:.4f}  NDSI={s2.NDSI:.4f}")

if PLOT:
    import matplotlib.pyplot as plt

    wavelengths = np.arange(0.205, 4.999, 0.01)
    fig, ax = plt.subplots(figsize=(8, 4))
    for row in points:
        alb = emu.predict(
            rds=row[0],
            rho=row[1],
            black_carbon=row[2],
            snow_algae=row[3],
            glacier_algae=row[4],
            dust=row[5],
            direct=row[6],
            solzen=row[7],
        )
        ax.plot(wavelengths, alb, label=f"rds={row[0]:.0f}, rho={row[1]:.0f}, BC={row[2]:.0f}, GA={row[4]:.0f}")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Albedo")
    ax.set_xlim(0.2, 2.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("Emulator spectral predictions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
