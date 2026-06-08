#!/usr/bin/env python3
"""Basic forward model usage.

Demonstrates ``run_model()`` with default inputs, keyword overrides,
multi-layer configuration, and accessing output fields.
"""

import numpy as np

from biosnicar import run_model

PLOT = True

# ======================================================================
# Example 1: Run with all defaults
# ======================================================================
print("=== Example 1: Default inputs ===\n")
outputs = run_model()
print(f"  Broadband albedo (BBA):     {outputs.BBA:.4f}")
print(f"  Visible BBA:                {outputs.BBAVIS:.4f}")
print(f"  NIR BBA:                    {outputs.BBANIR:.4f}")
print(f"  Spectral albedo shape:      {np.array(outputs.albedo).shape}")

# ======================================================================
# Example 2: Override a few parameters
# ======================================================================
print("\n=== Example 2: Override SZA and grain radius ===\n")
outputs = run_model(solzen=50, rds=1000)
print(f"  BBA (solzen=50, rds=1000):  {outputs.BBA:.4f}")

# ======================================================================
# Example 3: Add impurities
# ======================================================================
print("\n=== Example 3: Black carbon and glacier algae ===\n")
outputs_clean = run_model(solzen=50, rds=1000)
outputs_dirty = run_model(
    solzen=55,
    rds=1000,
    black_carbon=5000,  # ppb
    glacier_algae=50000,  # cells/mL
)
print(f"  Clean ice BBA:              {outputs_clean.BBA:.4f}")
print(f"  With impurities BBA:        {outputs_dirty.BBA:.4f}")
print(f"  Albedo reduction:           {outputs_clean.BBA - outputs_dirty.BBA:.4f}")

# ======================================================================
# Example 4: Multi-layer ice column
# ======================================================================
print("\n=== Example 4: Multi-layer configuration ===\n")
outputs = run_model(
    solzen=50,
    layer_type=[1, 1, 1, 1, 1],
    dz=[0.02, 0.05, 0.05, 0.05, 0.83],
    rds=[800, 900, 1000, 1100, 1200],
    rho=[500, 600, 700, 750, 800],
    glacier_algae=[40000, 10000, 0, 0, 0],  # algae in top 2 layers
)
print(f"  5-layer BBA:                {outputs.BBA:.4f}")

# ======================================================================
# Example 5: Access full spectral output
# ======================================================================
print("\n=== Example 5: Spectral output ===\n")
outputs = run_model(solzen=55, rds=500, layer_type=1)
albedo = np.array(outputs.albedo)
wavelengths = np.arange(0.205, 4.999, 0.01)

print(f"  Albedo at 0.5 um:           {albedo[29]:.4f}")
print(f"  Albedo at 1.0 um:           {albedo[79]:.4f}")
print(f"  Albedo at 1.5 um:           {albedo[129]:.4f}")
print(
    f"  Min albedo:                 {albedo.min():.4f} at {wavelengths[albedo.argmin()]:.2f} um"
)
print(
    f"  Max albedo:                 {albedo.max():.4f} at {wavelengths[albedo.argmax()]:.2f} um"
)

print(f"  Albedo array length:        {len(albedo)}")

if PLOT:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wavelengths, albedo)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Albedo")
    ax.set_xlim(0.2, 2)
    ax.set_ylim(0, 1.05)
    ax.set_title("Spectral albedo (solzen=50, rds=500)")
    fig.tight_layout()
    plt.show()
