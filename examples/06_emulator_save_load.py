#!/usr/bin/env python3
"""Emulator save/load roundtrip.

Demonstrates saving an emulator to ``.npz``, loading it back, verifying
predictions match, and inspecting the stored metadata.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from biosnicar import DATA_DIR
from biosnicar.emulator import Emulator

# ======================================================================
# Example 1: Build a small emulator and save it
# ======================================================================
print("=== Example 1: Build and save ===\n")
emu = Emulator.build(
    params={
        "rds": (100, 5000),
        "black_carbon": (0, 50000),
    },
    n_samples=200,
    layer_type=1,
    solzen=50,
    direct=1,
    progress=True,
)

save_path = Path(tempfile.mkdtemp()) / "test_emulator.npz"
emu.save(save_path)
size_kb = save_path.stat().st_size / 1024
print(f"\n  Saved to: {save_path}")
print(f"  File size: {size_kb:.1f} KB")

# ======================================================================
# Example 2: Load and verify predictions match
# ======================================================================
print("\n=== Example 2: Load and verify ===\n")
emu2 = Emulator.load(save_path)
print(f"  Loaded: {emu2!r}")
print(f"  Parameters match: {emu.param_names == emu2.param_names}")
print(f"  Bounds match:     {emu.bounds == emu2.bounds}")

# Compare predictions
test_params = {"rds": 1000, "black_carbon": 5000}
alb1 = emu.predict(**test_params)
alb2 = emu2.predict(**test_params)
max_diff = np.max(np.abs(alb1 - alb2))
print(f"  Max prediction difference: {max_diff:.2e}")

# ======================================================================
# Example 3: Inspect .npz contents
# ======================================================================
print("\n=== Example 3: Inspect .npz file ===\n")
data = np.load(str(save_path), allow_pickle=False)
print("  Keys in .npz file:")
for key in sorted(data.files):
    arr = data[key]
    if arr.shape == ():
        print(f"    {key:20s}  scalar (metadata JSON)")
    else:
        print(f"    {key:20s}  shape={arr.shape}  dtype={arr.dtype}")

# ======================================================================
# Example 4: Inspect metadata JSON
# ======================================================================
print("\n=== Example 4: Metadata ===\n")
meta = json.loads(str(data["metadata"]))
for key in [
    "param_names",
    "bounds",
    "n_samples",
    "n_pca_components",
    "training_r2",
    "build_timestamp",
    "fixed_overrides",
    "solver",
]:
    if key in meta:
        print(f"  {key}: {meta[key]}")

# ======================================================================
# Example 5: Load the pre-built default emulator
# ======================================================================
print("\n=== Example 5: Load default emulator ===\n")
default = Emulator.load(DATA_DIR / "emulators" / "glacier_ice_8_param_default.npz")
print(f"  {default!r}")

# Clean up
save_path.unlink()
