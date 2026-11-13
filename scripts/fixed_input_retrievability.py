#!/usr/bin/env python3
"""Could a fixed input of the forward model be retrieved instead?

The physics error budget (``biosnicar.inverse.forward_uncertainty``) says
which fixed inputs cost the most.  The obvious response to a costly one is
to retrieve it.  That only helps if the spectrum can tell it apart from the
parameters already retrieved; otherwise it becomes one more member of a
degenerate group, takes information from its neighbours and returns nothing.

This script answers that before anything is retrained.  For each surface
type and each continuous fixed input ``b`` it forms, at every sampled state,
the linear retrieval with ``b`` appended to the state vector:

* Jacobian of the retrieved parameters from the emulator (finite
  differences, in the optimiser's space, scaled by the prior width);
* Jacobian of ``b`` from the budget's one-sigma response at that state, so
  ``b`` is in units of its own prior sigma (the range in ``FACTORS``);
* ``S_e`` = instrument noise + the budget with ``b``'s own term removed
  (once retrieved, ``b`` is no longer a source of forward-model error).

and reports ``b``'s averaging kernel ``A = 1 - var_posterior / var_prior``:
1 means the spectrum fixes it, 0 means the retrieval returns the prior.
Also: which retrieved parameter it is most correlated with, and what adding
it does to the averaging kernels of the parameters already retrieved.

The same is computed with every *other* fixed input taken as known exactly
("A alone"), which separates two reasons for a low A: the input is
degenerate with a retrieved parameter (A alone is low too, |r| near 1), or
it is hidden behind the other fixed inputs, whose spectral signatures look
like its own (A alone is high).

Reading the table: A above ~0.5 in most states is a candidate to retrieve.
A low but A alone high: retrieving this input by itself gains nothing — the
fixed inputs have to be dealt with as a group.  Both low: leave it in the
error budget and look for outside information.

Usage::

    uv run python scripts/build_physics_model_error.py --deltas DIR
    uv run python scripts/fixed_input_retrievability.py --deltas DIR
        [--sigma-inst 0.02] [--types ...] [--json out.json]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biosnicar.inverse import forward_uncertainty as fu  # noqa: E402
from biosnicar.inverse.optimize import (  # noqa: E402
    _LOG_SPACE_PARAMS, _from_log, _to_log,
)
from biosnicar.sea_ice.emulator_configs import (  # noqa: E402
    BAND_MASKS, SEA_ICE_EMULATOR_CONFIGS, load_sea_ice_emulators,
    trained_emulator_names,
)
from biosnicar.sky import mixed_sky  # noqa: E402

WVL = fu.WAVELENGTHS_UM


def emulator_jacobian(emu, state, free, rel_step=2e-3):
    """d(albedo)/d(parameter) for the retrieved parameters, each column
    scaled by that parameter's prior width (its range in optimiser space) —
    the weak prior the retrieval uses when it has no other."""
    K = np.empty((WVL.size, len(free)))
    for j, name in enumerate(free):
        lo, hi = emu.bounds[name]
        is_log = name in _LOG_SPACE_PARAMS
        if is_log:
            lo, hi = float(_to_log(lo)), float(_to_log(hi))
            x0 = float(_to_log(state[name]))
        else:
            x0 = float(state[name])
        width = hi - lo
        h = rel_step * width
        xs = [min(x0 + h, hi), max(x0 - h, lo)]
        ys = []
        for x in xs:
            p = dict(state)
            p[name] = float(_from_log(x)) if is_log else x
            ys.append(np.asarray(emu.predict(**p), dtype=float))
        K[:, j] = (ys[0] - ys[1]) / (xs[0] - xs[1]) * width
    return K


def averaging_kernel(K, S_e):
    """Diagonal of the averaging kernel and the posterior correlation matrix
    for a unit-prior linear retrieval with Jacobian *K*."""
    Si_K = np.linalg.solve(S_e, K)
    post = np.linalg.inv(K.T @ Si_K + np.eye(K.shape[1]))
    sd = np.sqrt(np.diag(post))
    return 1.0 - np.diag(post), post / np.outer(sd, sd)


def analyse_type(type_name, deltas_dir, emu, sigma_inst):
    saved = np.load(Path(deltas_dir) / f"{type_name}.npz")
    names = [str(n) for n in saved["factor_names"]]
    if names != fu.factor_names():
        raise SystemExit(f"{type_name}: saved responses do not match FACTORS")
    D = np.asarray(saved["deltas"], dtype=float)           # (n, F, 480)
    state_names = [str(n) for n in saved["state_names"]]
    states = [dict(zip(state_names, row)) for row in saved["states"]]
    free = [p for p in emu.param_names if p not in ("solzen", "direct")]
    lo, hi = BAND_MASKS[SEA_ICE_EMULATOR_CONFIGS[type_name]["band_mask"]]
    mask = (WVL >= lo - 1e-9) & (WVL <= hi + 1e-9)
    m = int(mask.sum())
    Dm = D[:, :, mask]
    per_factor = np.einsum("sfi,sfj->fij", Dm, Dm) / D.shape[0]   # (F, m, m)
    total = per_factor.sum(axis=0)
    noise = np.eye(m) * sigma_inst ** 2

    candidates = [f for f in fu.applicable_factors(type_name)
                  if f.kind != "alternative" and f.where != "state"]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Ks = [emulator_jacobian(emu, {k: s[k] for k in emu.param_names},
                                free)[mask] for s in states]
    a_base = np.array([averaging_kernel(K, noise + total)[0] for K in Ks])
    for f in candidates:
        i = names.index(f.name)
        if np.sqrt((Dm[:, i] ** 2).mean()) < 1e-4:
            continue
        S_e = noise + total - per_factor[i]
        a_b, a_alone, a_others, corr, corr_alone = [], [], [], [], []
        for s, K in enumerate(Ks):
            Kb = np.column_stack([K, Dm[s, i]])
            a, r = averaging_kernel(Kb, S_e)
            a_b.append(a[-1])
            a_others.append(a[:-1])
            corr.append(r[-1, :-1])
            # the same with every other fixed input known exactly: is it the
            # retrieved parameters that hide b, or the other unknowns?
            a0, r0 = averaging_kernel(Kb, noise)
            a_alone.append(a0[-1])
            corr_alone.append(r0[-1, :-1])
        a_b, corr = np.array(a_b), np.abs(np.array(corr))
        corr_alone = np.abs(np.array(corr_alone))
        j0 = int(np.argmax(np.median(corr_alone, axis=0)))
        j = int(np.argmax(np.median(corr, axis=0)))
        out[f.name] = dict(
            sigma_albedo=float(np.sqrt((Dm[:, i] ** 2).mean())),
            ak_median=float(np.median(a_b)),
            ak_frac_measured=float((a_b > 0.5).mean()),
            ak_alone_median=float(np.median(a_alone)),
            alone_partner=free[j0],
            alone_partner_abs_r=float(np.median(corr_alone[:, j0])),
            partner=free[j],
            partner_abs_r=float(np.median(corr[:, j])),
            others_ak_before=float(np.median(a_base)),
            others_ak_after=float(np.median(np.array(a_others))),
            per_param_after={p: float(np.median(np.array(a_others)[:, k]))
                             for k, p in enumerate(free)},
            per_param_before={p: float(np.median(a_base[:, k]))
                              for k, p in enumerate(free)},
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--deltas", required=True,
                    help="directory written by build_physics_model_error.py "
                         "--deltas")
    ap.add_argument("--sigma-inst", type=float, default=0.02)
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    types = args.types or trained_emulator_names()
    fleet = load_sea_ice_emulators(types)
    results = {}
    print(f"instrument sigma {args.sigma_inst}; each type's own band window\n")
    print("A        averaging kernel of the input if retrieved, the other "
          "fixed inputs staying uncertain\n"
          "A alone  the same if every other fixed input were known exactly\n")
    print(f"{'type':11s} {'fixed input':26s} {'cost':>6s} {'A':>5s} "
          f"{'A>0.5':>6s} {'A alone':>8s}  {'alone: most correlated with':28s} "
          f"{'|r|':>5s}")
    for t in types:
        res = analyse_type(t, args.deltas, mixed_sky(fleet[t]), args.sigma_inst)
        results[t] = res
        for name, r in sorted(res.items(), key=lambda kv: -kv[1]["sigma_albedo"]):
            if r["sigma_albedo"] < 0.003:
                continue
            print(f"{t:11s} {name:26s} {r['sigma_albedo']:6.3f} "
                  f"{r['ak_median']:5.2f} {r['ak_frac_measured']:6.0%} "
                  f"{r['ak_alone_median']:8.2f}  {r['alone_partner']:28s} "
                  f"{r['alone_partner_abs_r']:5.2f}")
        print()
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
