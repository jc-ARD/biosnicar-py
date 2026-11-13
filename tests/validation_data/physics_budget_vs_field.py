#!/usr/bin/env python3
"""Is the physics error budget complete?  A check against field residuals.

The physics budget (``data/model_error/physics_se_v1.npz``) is built without
field data: it is the uncertainty of the forward model's fixed inputs,
propagated to the spectrum.  It contains what is on the list and nothing
else.  Field residuals — observed minus best-fit, from
``residual_library.npz`` — contain everything: the listed uncertainties,
physics the model lacks, and the field measurements' own error.

So this script does not calibrate anything and nothing it prints is fed
back.  It asks one question per surface type and spectral region: are field
misfits inside the budget?

* Inside (ratio near or below 1): the listed input uncertainties are enough
  to explain what is seen.
* Outside (ratio well above 1): something is unaccounted for — missing
  physics, or measurement error.  The wavelengths where it happens say
  which is more likely.

It is evaluation only, so the frozen test campaign may be read
(``corpus.require_role(..., "evaluate")``).  Residuals are post-fit, so a
budget that was exactly right would give a reduced chi-squared slightly
below 1.

Rows used: every fit of an *expected* surface type, whether or not it won
the classification — conditioning on the winner keeps the well-fitting
cases.

Usage::

    uv run python tests/validation_data/physics_budget_vs_field.py
        [--library tests/validation_data/residual_library.npz]
        [--sigma-inst 0.02] [--json out.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from biosnicar.inverse.model_error import (  # noqa: E402
    ModelErrorCovariance, PhysicsModelError,
)
from build_residual_library import load_residual_library  # noqa: E402
from corpus import require_role  # noqa: E402

WVL_NM = 205.0 + 10.0 * np.arange(480)
REGIONS = {
    "400-500": (400, 500), "500-700": (500, 700), "700-1000": (700, 1000),
    "1000-1350": (1000, 1350), "1450-1800": (1450, 1800),
    "2000-2400": (2000, 2400),
}


def reduced_chi2(resid, mask, sigma_inst, cov):
    y = resid[mask]
    S = np.diag(np.full(y.size, sigma_inst ** 2))
    if cov is not None:
        S = S + cov.dense(mask)
    return float(y @ np.linalg.solve(S, y) / y.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--library", default=str(HERE / "residual_library.npz"))
    ap.add_argument("--sigma-inst", type=float, default=0.02)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    lib = load_residual_library(args.library)
    campaigns = sorted(set(lib["campaign"].tolist()))
    require_role(campaigns, "evaluate")
    physics = PhysicsModelError.load()
    field = ModelErrorCovariance.load()
    sel = lib["is_expected"].astype(bool)
    out = {}

    vis = (WVL_NM >= 400) & (WVL_NM <= 1000)
    for title, window in (("400-1000 nm", vis),
                          ("all bands each spectrum covers",
                           np.ones(480, dtype=bool))):
        print(f"Reduced chi-squared of field residuals, {title} "
              f"(median over spectra; expected-type fits).\n")
        print(f"{'campaign':10s} {'fit type':11s} {'n':>4s} {'instrument':>11s} "
              f"{'+ physics':>10s} {'+ field-fitted':>15s}")
        for camp in campaigns:
            for t in sorted(set(lib["fit_type"][sel & (lib["campaign"] == camp)])):
                rows = np.flatnonzero(sel & (lib["campaign"] == camp)
                                      & (lib["fit_type"] == t))
                if rows.size < 3:
                    continue
                cov = physics.for_type(t)
                c = np.array([[reduced_chi2(lib["residual"][i],
                                            lib["mask"][i] & window,
                                            args.sigma_inst, m)
                               for m in (None, cov, field)] for i in rows])
                med = np.median(c, axis=0)
                out.setdefault(f"{camp}/{t}", {})[title] = dict(
                    n=int(rows.size), instrument=med[0], physics=med[1],
                    field=med[2])
                print(f"{camp:10s} {t:11s} {rows.size:4d} {med[0]:11.2f} "
                      f"{med[1]:10.2f} {med[2]:15.2f}")
        print()

    print("RMS field residual / budget one-sigma (instrument + physics), "
          "by region.\nNear 1: inside the budget.  Well above: unaccounted for.\n")
    head = " ".join(f"{r:>10s}" for r in REGIONS)
    print(f"{'campaign':10s} {'fit type':11s} {head}")
    for camp in campaigns:
        for t in sorted(set(lib["fit_type"][sel & (lib["campaign"] == camp)])):
            rows = np.flatnonzero(sel & (lib["campaign"] == camp)
                                  & (lib["fit_type"] == t))
            if rows.size < 3:
                continue
            sig = np.sqrt(args.sigma_inst ** 2 + physics.for_type(t).sigma() ** 2)
            R, M = lib["residual"][rows], lib["mask"][rows]
            cells = []
            for lo, hi in REGIONS.values():
                band = (WVL_NM >= lo) & (WVL_NM <= hi)
                ok = M[:, band]
                if ok.sum() < 5:
                    cells.append(f"{'-':>10s}")
                    continue
                rms = np.sqrt(np.mean(R[:, band][ok] ** 2))
                bud = np.sqrt(np.mean(np.broadcast_to(sig[band] ** 2,
                                                      ok.shape)[ok]))
                cells.append(f"{rms / bud:10.2f}")
                out.setdefault(f"{camp}/{t}", {})[f"ratio_{lo}_{hi}"] = rms / bud
            print(f"{camp:10s} {t:11s} {' '.join(cells)}")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
