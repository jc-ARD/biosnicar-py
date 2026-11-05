#!/usr/bin/env python3
"""Held-out spectral audit of the sea-ice emulators.

The ``Training R2`` printed by ``build_sea_ice_emulators.py`` is R2 on the PCA
*coefficients*, scored on the training data, and reads misleadingly low for
some emulators — FYI_bare reports ~0.70 while its held-out spectral R2 is
0.998.  This script measures the figure that actually matters: reconstructed
spectra scored against the forward model on fresh samples.

Sampling MUST match ``Emulator.build``: Latin hypercube, with
``_LOG_SAMPLE_PARAMS`` drawn in log10(x+1) space.  Sampling uniformly in linear
space instead over-weights the sparse high end of bubble radius and black
carbon, tests the emulator where it was barely trained, and understates
FYI_bare at R2 ~ 0.80 / RMSE 0.063.  That is an artefact of the measurement,
not of the emulator.

Error is decomposed into:

  PCA ceiling  best achievable given the retained components, by projecting the
               TRUE spectra onto the emulator's own stored basis.  Independent
               of the network.
  network gap  how far the MLP falls short of that ceiling.

Measured 2026-11-05 on the shipped fleet (250 samples each)::

    emulator        n | network R2    RMSE | PCA ceiling R2    RMSE
    FYI_bare      250 |     0.9981  0.0093 |         0.9997  0.0035
    MYI_bare      250 |     0.9977  0.0099 |         0.9997  0.0036
    FYI_summer    250 |     1.0000  0.0015 |         1.0000  0.0008
    FYI_snow      250 |     0.9999  0.0023 |         1.0000  0.0016
    FYI_pond      250 |     0.9999  0.0014 |         1.0000  0.0011
    young_ice     250 |     0.9990  0.0022 |         0.9996  0.0014

Companion script: ``scripts/experiments/fyi_bare_audit.py`` goes deeper on
FYI_bare alone — 2000 spectra, BBA error, and the rho_DL/brine-volume Jacobian
collinearity analysis.

Usage::

    uv run python scripts/audit_sea_ice_emulators.py

Reads the emulators' private PCA attributes (``_pca_components``,
``_pca_mean``) to compute the ceiling; that is deliberate and confined to this
audit script.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import numpy as np
from biosnicar.emulator import _latin_hypercube, _LOG_SAMPLE_PARAMS, _BINARY_PARAMS
from biosnicar.sea_ice.emulator_configs import (
    load_sea_ice_emulators, SEA_ICE_EMULATOR_CONFIGS as CFG, BAND_MASKS)
from biosnicar.drivers.run_model import run_model

WL = np.arange(0.205, 4.999, 0.01)
N = 250
SEED = 20261105          # different from the build seed (42)

def sample(cfg, n, seed):
    names = list(cfg["params"])
    lo = np.array([cfg["params"][k][0] for k in names], float)
    hi = np.array([cfg["params"][k][1] for k in names], float)
    u = _latin_hypercube(n, len(names), seed=seed)
    X = lo + u * (hi - lo)
    for j, k in enumerate(names):
        if k in _LOG_SAMPLE_PARAMS:
            a, b = np.log10(lo[j] + 1), np.log10(hi[j] + 1)
            X[:, j] = 10 ** (a + u[:, j] * (b - a)) - 1
        if k in _BINARY_PARAMS:
            X[:, j] = np.round(X[:, j])
    return names, X

def stats(T, P):
    r2 = 1 - ((T - P) ** 2).sum() / ((T - T.mean()) ** 2).sum()
    return r2, float(np.sqrt(((T - P) ** 2).mean()))

print(f"Held-out audit: {N} LHS samples, build-matched log-space sampling, seed {SEED}\n")
hdr = f"{'emulator':12} {'n':>4} | {'network R2':>10} {'RMSE':>7} | {'PCA ceiling R2':>14} {'RMSE':>7} | {'vis-only RMSE':>13}"
print(hdr); print("-" * len(hdr))

for name in ["FYI_bare", "MYI_bare", "FYI_summer", "FYI_snow", "FYI_pond", "young_ice"]:
    cfg = CFG[name]
    e = load_sea_ice_emulators([name])[name]
    names, X = sample(cfg, N, SEED)
    T, P = [], []
    for row in X:
        p = dict(zip(names, row))
        try:
            t = np.asarray(run_model(**cfg["transform_fn"](p)).albedo, float)
            q = np.asarray(e.predict(**p), float)
        except Exception:
            continue
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(q))):
            continue
        T.append(t); P.append(q)
    T = np.array(T); P = np.array(P)
    if len(T) < 10:
        print(f"{name:12} {len(T):>4} | insufficient samples"); continue

    # PCA ceiling: project the TRUE spectra onto the emulator's own stored basis
    comps, mean = e._pca_components, e._pca_mean
    recon = (T - mean) @ comps.T @ comps + mean

    r2n, rmn = stats(T, P)
    r2c, rmc = stats(T, recon)
    lo_um, hi_um = BAND_MASKS[cfg["band_mask"]]
    m = (WL >= lo_um) & (WL <= hi_um)
    _, rm_vis = stats(T[:, m], P[:, m])
    print(f"{name:12} {len(T):>4} | {r2n:10.4f} {rmn:7.4f} | {r2c:14.4f} {rmc:7.4f} | {rm_vis:13.4f}")
