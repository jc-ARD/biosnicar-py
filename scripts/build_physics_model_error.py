#!/usr/bin/env python3
"""Build the physics-derived forward-model error budget.

For every surface type, runs the forward model across the emulator's
parameter box with each uncertain fixed input moved one sigma each way
(``biosnicar.inverse.forward_uncertainty.FACTORS``), and writes the
state-averaged ``K_b S_b K_b^T`` per type to
``data/model_error/physics_se_v1.npz``.

No field data is read.  To change a range, edit ``FACTORS`` and rerun.

Usage::

    uv run python scripts/build_physics_model_error.py [--n-states 300]
        [--jobs 8] [--types FYI_bare ...] [--out PATH] [--deltas DIR]

``--deltas DIR`` also saves the raw per-state responses (large; for
analysis, not shipped).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biosnicar.inverse import forward_uncertainty as fu  # noqa: E402
from biosnicar.inverse.model_error import (  # noqa: E402
    PHYSICS_ARTIFACT, ModelErrorCovariance, PhysicsModelError,
)
from biosnicar.sea_ice.emulator_configs import (  # noqa: E402
    SEA_ICE_EMULATOR_CONFIGS, load_sea_ice_emulators,
)

WVL = fu.WAVELENGTHS_UM
VIS = (WVL >= 0.4) & (WVL <= 1.0)
SWIR = (WVL > 1.0) & (WVL <= 2.5)


def report(name, cov, factor_var):
    print(f"\n{name}: rank {cov.meta['n_eofs']} "
          f"({cov.meta['trace_captured']:.4f} of trace), "
          f"effective DOF over the VIS {cov.effective_dof(VIS):.1f}")
    print(f"  {'input':30s} {'VIS':>8s} {'SWIR':>8s}   (RMS 1-sigma albedo)")
    rows = sorted(zip(fu.factor_names(), factor_var),
                  key=lambda kv: -kv[1][VIS].mean())
    for n, v in rows:
        if v.max() > 0:
            print(f"  {n:30s} {np.sqrt(v[VIS].mean()):8.4f} "
                  f"{np.sqrt(v[SWIR].mean()):8.4f}")
    tot = factor_var.sum(axis=0)
    print(f"  {'total':30s} {np.sqrt(tot[VIS].mean()):8.4f} "
          f"{np.sqrt(tot[SWIR].mean()):8.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-states", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--max-rank", type=int, default=40)
    ap.add_argument("--energy", type=float, default=0.999)
    ap.add_argument("--out", default=PHYSICS_ARTIFACT)
    ap.add_argument("--deltas", default=None,
                    help="directory for raw per-state responses")
    ap.add_argument("--from-deltas", default=None,
                    help="rebuild from responses saved with --deltas "
                         "(no forward-model runs)")
    args = ap.parse_args()

    types = list(args.types or SEA_ICE_EMULATOR_CONFIGS)
    fleet = load_sea_ice_emulators(types)
    by_type, factor_var = {}, {}
    t_all = time.time()
    for i, name in enumerate(types):
        t0 = time.time()
        if args.from_deltas:
            saved = np.load(Path(args.from_deltas) / f"{name}.npz")
            if list(saved["factor_names"]) != fu.factor_names():
                raise SystemExit(f"{name}: saved responses were built from a "
                                 "different factor table; rerun without "
                                 "--from-deltas")
            deltas, states = saved["deltas"], None
            args.n_states = deltas.shape[0]
        else:
            deltas, states = fu.type_deltas(
                name, n_states=args.n_states, seed=args.seed + i,
                n_jobs=args.jobs, emulator=fleet[name])
        S, fv = fu.covariance_from_deltas(deltas)
        by_type[name] = ModelErrorCovariance.from_dense(
            S, max_rank=args.max_rank, energy=args.energy,
            meta=dict(surface_type=name, n_states=int(args.n_states)))
        factor_var[name] = fv
        report(name, by_type[name], fv)
        print(f"  ({time.time() - t0:.0f} s)")
        if args.deltas and states is not None:
            out = Path(args.deltas)
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out / f"{name}.npz", deltas=deltas,
                factor_names=np.array(fu.factor_names()),
                state_names=np.array(list(states[0])),
                states=np.array([[s[k] for k in states[0]] for s in states]))

    model = PhysicsModelError(
        by_type=by_type, factor_names=fu.factor_names(),
        factor_var=factor_var,
        meta=dict(source="physics", n_states=int(args.n_states),
                  seed=int(args.seed), factors=fu.factor_table()),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"\nsaved: {args.out}  ({time.time() - t_all:.0f} s total)")


if __name__ == "__main__":
    main()
