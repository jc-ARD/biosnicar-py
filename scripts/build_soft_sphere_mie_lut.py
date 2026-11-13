#!/usr/bin/env python3
"""Mie efficiencies of optically soft spheres, for scatterers inside sea ice.

Brine inclusions in ice, and salt crystals in brine, are spheres (after the
equivalent-sphere reduction) whose refractive index differs from their
surroundings by a few percent.  This tabulates their scattering efficiency
``Q_sca`` and asymmetry parameter ``g`` on a grid of

    m   relative real refractive index, 1.0005 - 1.12
    x   size parameter 2 pi r n_medium / lambda, 1 - 30000

by Mie theory (miepython), averaged over a +/-25% spread in size so that the
interference ripple of a single sphere — which no natural population of
inclusions shows — is removed.

Both indices are taken as real.  Brine, ice and the salts absorb almost
equally at any given wavelength, so an inclusion's *relative* absorption is
close to zero; the absorption itself is carried by the layer's
volume-weighted absorption coefficient, not here.

Two limits to recognise in the table: for large phase shift 2x(m - 1) >> 1,
``Q_sca -> 2`` and ``g`` depends on ``m`` alone (0.982 at m = 1.065, 0.993 at
m = 1.025, as in Light et al. 2004, Fig. 7); for small phase shift
``Q_sca`` falls as its square (Rayleigh-Gans), which is where micron-sized
salt crystals sit.

Writes data/OP_data/480band/luts/soft_sphere_mie.npz.  Needs the ``mie``
extra.  Runtime about five minutes.

Usage::

    uv run python scripts/build_soft_sphere_mie_lut.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import biosnicar  # noqa: E402
from biosnicar.utils.optional_deps import import_miepython  # noqa: E402

OUT = biosnicar.DATA_DIR / "OP_data" / "480band" / "luts" / "soft_sphere_mie.npz"

M_GRID = np.array([1.0005, 1.001, 1.002, 1.003, 1.005, 1.0075, 1.01, 1.015,
                   1.02, 1.025, 1.03, 1.035, 1.04, 1.045, 1.05, 1.055, 1.06,
                   1.065, 1.07, 1.08, 1.09, 1.10, 1.12])
X_GRID = np.geomspace(1.0, 3.0e4, 72)
_SPREAD = np.exp(np.linspace(-0.25, 0.25, 41))     # +/- 25% in size


def build():
    mie = import_miepython()
    q = np.empty((M_GRID.size, X_GRID.size))
    g = np.empty_like(q)
    t0 = time.time()
    for i, m in enumerate(M_GRID):
        for j, x0 in enumerate(X_GRID):
            xs = x0 * _SPREAD
            res = [mie.efficiencies_mx(float(m), float(x)) for x in xs]
            qs = np.array([r[1] for r in res])
            gs = np.array([r[3] for r in res])
            area = xs ** 2
            # Population means: scattering cross-section per unit projected
            # area, and the scattering-weighted asymmetry parameter.
            q[i, j] = (qs * area).sum() / area.sum()
            g[i, j] = (gs * qs * area).sum() / (qs * area).sum()
        print(f"  m = {m:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(OUT, m_grid=M_GRID, x_grid=X_GRID, q_sca=q, asm=g)
    print(f"saved {OUT}")


if __name__ == "__main__":
    build()
