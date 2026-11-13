#!/usr/bin/env python3
"""Closure test of the physics error budget — no field data involved.

The budget claims: *if* the forward model's fixed inputs are wrong by the
amounts listed in ``biosnicar.inverse.forward_uncertainty.FACTORS``, then
this covariance describes the resulting error.  That claim can be tested
exactly.  For each surface type:

1. draw a true state across the parameter box;
2. draw every fixed input from its stated distribution, all at once, and run
   the full forward model there — the physics, not the lookup table, with a
   clear sky that has its diffuse component;
3. add instrument noise;
4. retrieve with the emulator at the neutral physics, the way a user would;
5. compare the error of every retrieved parameter with the uncertainty the
   retrieval reported.

If the covariance is right, the standardised errors ``z = (x_hat - x_true) /
sigma`` have unit spread: 68% inside one sigma, 95% inside two; and the
residual's reduced chi-squared is near 1.  Each case is retrieved twice,
with instrument noise only and with the physics budget, so the table shows
what the budget buys.

What this test cannot see: anything not in ``FACTORS``.  It checks that the
arithmetic from stated input uncertainty to retrieval uncertainty is sound
(linearisation, state averaging, low-rank truncation), not that the list of
inputs is complete.  A discrete alternative (grain shape) is drawn as a coin
flip between the two choices, which is a little harsher than the budget's
half-difference.

The budget is averaged over states, so it is not the covariance of any one
state.  Evaluating the covariance at the true state instead gives coverage
closer to nominal and leaves more parameters measured; the averaged budget
errs wide.  Parameters the spectrum does not constrain are always "covered",
because their posterior is the prior and the truth is drawn inside it — read
the ``A > 0.5`` column for the parameters that are actually measured.

Usage::

    uv run python tests/validation_data/physics_budget_closure.py [--n 150]
        [--sigma-inst 0.005] [--types ...] [--json out.json]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from biosnicar.inverse import forward_uncertainty as fu  # noqa: E402
from biosnicar.inverse.model_error import PhysicsModelError  # noqa: E402
from biosnicar.inverse.optimize import (  # noqa: E402
    _LOG_SPACE_PARAMS, _to_log, retrieve,
)
from biosnicar.sea_ice.emulator_configs import (  # noqa: E402
    BAND_MASKS, SEA_ICE_EMULATOR_CONFIGS, load_sea_ice_emulators,
    trained_emulator_names,
)

WVL = fu.WAVELENGTHS_UM


def draw_truth(type_name, params, rng):
    """Spectrum of *type_name* at *params* with every fixed input drawn from
    its stated distribution.  Returns the 480-band albedo."""
    cfg = SEA_ICE_EMULATOR_CONFIGS[type_name]
    switches, fixed, p = {"sea_ice_direct": True}, {}, dict(params)
    for f in fu.applicable_factors(type_name):
        z = float(rng.standard_normal())
        if f.kind == "alternative":
            if f.where == "physics":
                # continuous: one sigma is half the distance to the alternative
                switches[f.key] = 0.5 * z * f.sigma
            elif z > 0:
                # discrete (grain shape): a coin flip between the two choices
                fixed[f.key] = f.sigma
            continue
        if f.where == "physics":
            base = fu.perturbation.NEUTRAL[f.key]
        elif f.where == "fixed":
            base = cfg["fixed"][f.key]
        else:
            base = p[f.key]
        value = base + z * f.sigma if f.kind == "additive" else base * f.sigma ** z
        if f.where == "physics":
            switches[f.key] = value
        elif f.where == "fixed":
            fixed[f.key] = value
        else:
            p[f.key] = int(round(np.clip(value, 1, 89)))
    y = fu.forward(type_name, p, fixed=fixed or None, **switches)
    if int(p.get("direct", 1)) == 1:
        frac = fu.clear_sky_diffuse_fraction(p["solzen"])
        y_dif = fu.forward(type_name, dict(p, direct=0), fixed=fixed or None,
                           **switches)
        y = (1.0 - frac) * y + frac * y_dif
    return y


def _mask(type_name):
    lo, hi = BAND_MASKS[SEA_ICE_EMULATOR_CONFIGS[type_name]["band_mask"]]
    return (WVL >= lo - 1e-9) & (WVL <= hi + 1e-9)


def run_type(type_name, emu, budget, n, sigma_inst, seed):
    rng = np.random.default_rng(seed)
    states = fu.sample_states(type_name, n, seed=seed)
    free = [p for p in emu.param_names if p not in ("solzen", "direct")]
    mask = _mask(type_name)
    out = {"instrument": [], "physics": []}
    for p in states:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = draw_truth(type_name, p, rng)
            y = y + rng.normal(0.0, sigma_inst, y.size)
            for label, me in (("instrument", None), ("physics", budget)):
                r = retrieve(
                    observed=y, parameters=free, emulator=emu, method="oe",
                    obs_uncertainty=np.full(480, sigma_inst),
                    wavelength_mask=mask, model_error=me,
                    fixed_params={"solzen": p["solzen"], "direct": p["direct"]},
                )
                z, ak = {}, {}
                for name in free:
                    is_log = name in _LOG_SPACE_PARAMS
                    xh = _to_log(r.best_fit[name]) if is_log else r.best_fit[name]
                    xt = _to_log(p[name]) if is_log else p[name]
                    sg = np.sqrt(max(r.posterior_covariance[(name, name)], 0.0))
                    z[name] = float((xh - xt) / max(sg, 1e-12))
                    ak[name] = float(r.averaging_kernel_diag[name])
                resid = (y - r.predicted_albedo)[mask]
                out[label].append(dict(z=z, ak=ak, resid=resid))
    return out, free, mask


def summarise(rows, free, mask, sigma_inst, cov):
    z = np.array([[r["z"][p] for p in free] for r in rows])
    ak = np.array([[r["ak"][p] for p in free] for r in rows])
    S = np.diag(np.full(int(mask.sum()), sigma_inst ** 2))
    if cov is not None:
        S = S + cov.dense(mask)
    Sinv = np.linalg.inv(S)
    chi2 = np.array([r["resid"] @ Sinv @ r["resid"] / mask.sum() for r in rows])
    meas = ak > 0.5
    return dict(
        cover1=float((np.abs(z) < 1).mean()),
        cover2=float((np.abs(z) < 2).mean()),
        z_rms=float(np.sqrt(np.median(z ** 2) / 0.455)),   # robust: median chi2_1
        cover1_measured=float((np.abs(z[meas]) < 1).mean()) if meas.any() else None,
        frac_measured=float(meas.mean()),
        chi2_median=float(np.median(chi2)),
        per_param={p: dict(cover1=float((np.abs(z[:, i]) < 1).mean()),
                           cover2=float((np.abs(z[:, i]) < 2).mean()),
                           mean_ak=float(ak[:, i].mean()))
                   for i, p in enumerate(free)},
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--sigma-inst", type=float, default=0.005)
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    types = args.types or trained_emulator_names()
    fleet = load_sea_ice_emulators(types)
    budget = PhysicsModelError.load()

    print(f"closure test: {args.n} states per type, instrument sigma "
          f"{args.sigma_inst}\ntarget: 68% within 1 sigma, 95% within 2, "
          f"z spread 1, reduced chi2 ~1\n")
    print(f"{'type':11s} {'S_e':11s} {'<1s':>6s} {'<2s':>6s} {'z rms':>7s} "
          f"{'<1s (A>0.5)':>12s} {'chi2':>8s}")
    results = {}
    for i, t in enumerate(types):
        cov = budget.for_type(t)
        rows, free, mask = run_type(t, fleet[t], cov, args.n, args.sigma_inst,
                                    args.seed + i)
        results[t] = {}
        for label, c in (("instrument", None), ("physics", cov)):
            s = summarise(rows[label], free, mask, args.sigma_inst, c)
            results[t][label] = s
            cm = (f"{s['cover1_measured']:.0%}"
                  if s["cover1_measured"] is not None else "-")
            print(f"{t:11s} {label:11s} {s['cover1']:6.0%} {s['cover2']:6.0%} "
                  f"{s['z_rms']:7.2f} {cm:>12s} {s['chi2_median']:8.2f}")
        for p, d in results[t]["physics"]["per_param"].items():
            d0 = results[t]["instrument"]["per_param"][p]
            print(f"      {p:24s} <1s {d0['cover1']:4.0%} -> {d['cover1']:4.0%}   "
                  f"<2s {d0['cover2']:4.0%} -> {d['cover2']:4.0%}   "
                  f"A {d0['mean_ak']:.2f} -> {d['mean_ak']:.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
