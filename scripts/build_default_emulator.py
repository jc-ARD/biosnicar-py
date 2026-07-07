#!/usr/bin/env python3
"""Build the default glacier-ice emulator shipped with the repository.

Creates an 8-parameter emulator (rds, rho, black_carbon, snow_algae,
glacier_algae, dust, direct, solzen) trained on 50,000 Latin hypercube
samples and saves it to ``data/emulators/glacier_ice_8_param_default.npz``.

Usage:
    python scripts/build_default_emulator.py
"""

from pathlib import Path

import biosnicar
from biosnicar.emulator import Emulator

OUTPUT = biosnicar.DATA_DIR / "emulators" / "glacier_ice_8_param_default.npz"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("Building default glacier-ice emulator (35,000 samples, 8 parameters)...")
    emu = Emulator.build(
        params={  # provide min and max for each param, then n_samples.
            "rds": (500, 10000),
            "rho": (300, 900),
            "black_carbon": (0, 5000),
            "dust": (0, 50000),
            "snow_algae": (0, 500000),
            "glacier_algae": (0, 500000),
            "direct": (0, 1),
            "solzen": (20, 80)
        },
        n_samples=35000,
        layer_type=1,
        direct=1,
        progress=True,
    )

    emu.save(OUTPUT)
    print(f"\nSaved to {OUTPUT}")
    print(f"  Parameters:     {emu.param_names}")
    print(f"  PCA components: {emu.n_pca_components}")
    print(f"  Training R²:    {emu.training_score:.6f}")


if __name__ == "__main__":
    main()
