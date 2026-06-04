#!/usr/bin/env python3
"""Build the five sea ice surface-type emulators.

Trains one MLP emulator per sea ice surface type and saves each to
``data/emulators/``.  Total training time is approximately 45–60 minutes
on a modern laptop (depends on CPU core count and n_samples).

Usage::

    python scripts/build_sea_ice_emulators.py              # build all five
    python scripts/build_sea_ice_emulators.py FYI_bare     # build one only
    python scripts/build_sea_ice_emulators.py --fast       # 2000 samples each (for testing)

Emulators built
---------------
  sea_ice_FYI_bare_7param.npz    Winter/spring bare first-year ice
  sea_ice_FYI_snow_6param.npz    Snow-covered first-year ice
  sea_ice_FYI_summer_6param.npz  Melt-season bare ice (with SSL)
  sea_ice_MYI_bare_6param.npz    Bare multiyear ice
  sea_ice_FYI_pond_5param.npz    Melt pond on first-year ice
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure repo root is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosnicar.emulator import Emulator
from biosnicar.sea_ice.emulator_configs import SEA_ICE_EMULATOR_CONFIGS


def build_one(name: str, n_samples_override: int = None, seed: int = 42) -> None:
    cfg = SEA_ICE_EMULATOR_CONFIGS[name]
    out_path = Path(cfg["emulator_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = n_samples_override or cfg["n_samples"]
    print(f"\n{'='*60}")
    print(f"  {name}  —  {cfg['description']}")
    print(f"  Parameters : {list(cfg['params'].keys())}")
    print(f"  Samples    : {n_samples}")
    print(f"  Output     : {out_path.name}")
    print(f"{'='*60}")

    hidden = cfg.get("hidden_layer_sizes", (128, 128, 64))
    print(f"  Network    : {hidden}")
    t0 = time.time()
    emu = Emulator.build(
        params=cfg["params"],
        n_samples=n_samples,
        transform_fn=cfg["transform_fn"],
        hidden_layer_sizes=hidden,
        seed=seed,
        progress=True,
    )
    elapsed = time.time() - t0

    emu.save(out_path)

    print(f"\n  Training R²      : {emu.training_score:.6f}")
    print(f"  PCA components   : {emu.n_pca_components}")
    print(f"  Build time       : {elapsed/60:.1f} min")
    print(f"  Saved to         : {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*",
                        help="Surface type names to build (default: all five)")
    parser.add_argument("--fast", action="store_true",
                        help="Use 2000 samples per emulator (for testing only)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Override n_samples for all emulators")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names = args.names or list(SEA_ICE_EMULATOR_CONFIGS)
    unknown = [n for n in names if n not in SEA_ICE_EMULATOR_CONFIGS]
    if unknown:
        print(f"Unknown surface type(s): {unknown}")
        print(f"Available: {list(SEA_ICE_EMULATOR_CONFIGS)}")
        sys.exit(1)

    n_override = 2000 if args.fast else args.samples

    t_total = time.time()
    for name in names:
        build_one(name, n_samples_override=n_override, seed=args.seed)

    print(f"\nAll done — {len(names)} emulator(s) built in "
          f"{(time.time()-t_total)/60:.1f} min total.")


if __name__ == "__main__":
    main()
