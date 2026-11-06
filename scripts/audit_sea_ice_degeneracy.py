#!/usr/bin/env python3
"""Posterior parameter degeneracy across the whole sea-ice emulator fleet.

``scripts/experiments/fyi_bare_audit.py`` measures this for FYI_bare only.
That is how the dominant FYI_bare pair (``sea_ice_bubble_radius``/``rho_DL``,
posterior correlation 0.84) went unnoticed for so long: the diagnostic looked
at one hand-picked pair of one emulator.  This script measures *every* pair of
*every* emulator, so a degeneracy cannot hide by not being looked at.

Method
------
For each emulator: draw base states by Latin hypercube over the live config
(log-sampled parameters in log10(x+1) space, matching ``Emulator.build``),
generate a synthetic spectrum, add noise, and run the real OE retrieval with
the calibrated forward-model error covariance.  Read the posterior covariance
and convert to correlations.

Why the posterior and not a Jacobian cosine: the cosine is prior-free,
noise-free and blind to the other fitted parameters, so it is an upper bound on
the confusion rather than a measurement of it.  For FYI_bare's
``rho_DL``/``brine_volume_fraction`` the cosine says 0.73 and the posterior
says 0.31.

Reading the output
------------------
``|r|``        posterior correlation; 1 = interchangeable, 0 = independent.
``infl``       1/sqrt(1-r^2) -- how much wider the marginal 1-sigma is than it
               would be if the partner parameter were known exactly.
``sign``       positive/negative split across base points.  A *stable* sign
               means a reparameterisation onto the constrained combination can
               work; a sign that flips cannot be fixed by any global linear
               rotation, because the trade-off reverses.
``A``          averaging-kernel diagonal: 1 = determined by the measurement,
               0 = determined by the prior.

Usage::

    uv run python scripts/audit_sea_ice_degeneracy.py [n_points]
"""
import itertools
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import numpy as np

from biosnicar.emulator import _latin_hypercube, _LOG_SAMPLE_PARAMS
from biosnicar.sea_ice.emulator_configs import (
    load_sea_ice_emulators, SEA_ICE_EMULATOR_CONFIGS as CFG,
)
from biosnicar.inverse.optimize import retrieve
from biosnicar.inverse.model_error import ModelErrorCovariance

# solzen/direct are observation geometry, supplied by the caller at retrieval
# time rather than fitted, so they are held fixed here too.
GEOMETRY = {"solzen", "direct"}
SEED = 20261106


def sample_states(cfg, n, seed):
    names = [p for p in cfg["params"]]
    lo = np.array([cfg["params"][p][0] for p in names], float)
    hi = np.array([cfg["params"][p][1] for p in names], float)
    u = _latin_hypercube(n, len(names), seed=seed)
    X = lo + u * (hi - lo)
    for j, p in enumerate(names):
        if p in _LOG_SAMPLE_PARAMS:
            a, b = np.log10(lo[j] + 1), np.log10(hi[j] + 1)
            X[:, j] = 10 ** (a + u[:, j] * (b - a)) - 1
        if p == "direct":
            X[:, j] = 1.0
        if p == "solzen":
            X[:, j] = np.clip(np.round(X[:, j]), 20, 80)
    return names, X


def audit(name, n, sigma=0.01):
    cfg = CFG[name]
    emu = load_sea_ice_emulators([name])[name]
    try:
        ME = ModelErrorCovariance.load()
    except Exception:
        ME = None
    names, X = sample_states(cfg, n, SEED)
    fit = [p for p in names if p not in GEOMETRY]
    if len(fit) < 2:
        print(f"{name:12} {len(fit)} fitted parameter - no pairs to test")
        return None
    rng = np.random.default_rng(SEED)
    pairs = {pr: [] for pr in itertools.combinations(fit, 2)}
    akd = {p: [] for p in fit}
    pinned = {p: 0 for p in fit}
    n_ok = 0
    for row in X:
        st = dict(zip(names, row))
        geom = {g: st[g] for g in GEOMETRY if g in st}
        try:
            y = np.asarray(emu.predict(**st), float) + rng.normal(0, sigma, 480)
            r = retrieve(
                observed=y, parameters=fit, emulator=emu, fixed_params=geom,
                bounds={p: cfg["params"][p] for p in fit},
                x0={p: st[p] for p in fit},
                obs_uncertainty=np.full(480, sigma),
                method="oe", model_error=ME,
            )
            c = r.parameter_correlation()
            if not c:
                continue
            for pr in pairs:
                pairs[pr].append(float(c[pr]))
            for p in fit:
                akd[p].append(float(r.averaging_kernel_diag[p]))
                lo_, hi_ = cfg["params"][p]
                v = r.best_fit[p]
                if abs(v - lo_) < 1e-3 * (hi_ - lo_) or abs(v - hi_) < 1e-3 * (hi_ - lo_):
                    pinned[p] += 1
            n_ok += 1
        except Exception:
            continue
    if n_ok == 0:
        print(f"{name:12} no successful retrievals")
        return None

    print(f"\n{name}  ({n_ok} states, {len(fit)} fitted params, "
          f"{len(pairs)} pairs)")
    print(f"  {'pair':52} {'mean|r|':>8} {'max|r|':>8} {'sign +/-':>10} {'infl':>7} {'max':>7}")
    ranked = sorted(pairs.items(),
                    key=lambda kv: -np.mean(np.abs(kv[1])))
    for pr, v in ranked:
        a = np.abs(v)
        inf = 1.0 / np.sqrt(np.clip(1 - a ** 2, 1e-12, None))
        npos = int(sum(1 for x in v if x > 0))
        stable = "*" if npos == 0 or npos == len(v) else " "
        print(f" {stable}{pr[0] + ' / ' + pr[1]:52} {a.mean():8.3f} {a.max():8.3f} "
              f"{str(npos) + '/' + str(len(v) - npos):>10} {inf.mean():7.2f} {inf.max():7.2f}")
    print(f"  {'averaging kernel':52} {'mean':>8} {'min':>8}   bound-pinned")
    for p in fit:
        print(f"  {p:52} {np.mean(akd[p]):8.3f} {np.min(akd[p]):8.3f}   "
              f"{pinned[p]}/{n_ok}")
    return ranked


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    print(f"Fleet degeneracy audit: {n} states per emulator, seed {SEED}")
    print("'*' marks a pair whose correlation keeps the same sign at every "
          "state\n    (a reparameterisation can help; a flipping sign cannot "
          "be rotated away).")
    worst = {}
    for name in CFG:
        r = audit(name, n)
        if r:
            worst[name] = r[0]
    print("\n\n=== worst pair per emulator ===")
    print(f"{'emulator':12} {'pair':52} {'mean|r|':>8} {'infl':>7}")
    for name, (pr, v) in sorted(worst.items(),
                                key=lambda kv: -np.mean(np.abs(kv[1][1]))):
        a = np.abs(v)
        inf = float(np.mean(1 / np.sqrt(np.clip(1 - a ** 2, 1e-12, None))))
        print(f"{name:12} {pr[0] + ' / ' + pr[1]:52} {a.mean():8.3f} {inf:7.2f}")
