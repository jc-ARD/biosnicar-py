#!/usr/bin/env python3
"""OE-mode SHEBA SWIR classification experiment (audit item B6).

Decision experiment: ``method="oe"`` classifies by Laplace model evidence over
the full observation and cannot apply the per-type classification band masks
(evidences are only comparable when every candidate scores the same data).
The default method's masks are what protect summer bare-ice accuracy when
SWIR is present. Question: **does OE classification degrade on VIS+SWIR the
way the unmasked chi-squared did (100% -> 42%), and is OE fine on VIS-only?**

Design — a 2x2 on the SHEBA summer white-ice dates with paired ALBI SWIR
columns (the same data as ``sheba_classification_validation.py`` tests 2-3):

    method  in {default (L-BFGS-B + classification band masks), oe}
    window  in {VIS (400-1000 nm), VIS+SWIR (400-2000 nm, ALBI overwrite)}

Expected surface types: FYI_bare or FYI_summer. All runs use known_month
seasonal priors and direct=1, matching the reference suite. OE uses its
default obs_uncertainty (0.02).

Usage::

    python tests/validation_data/oe_swir_classification_experiment.py [--json out.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

GRENFELL_DIR = Path(__file__).parent / "Grenfell_light_2007"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from biosnicar.sea_ice.retrieve import retrieve_sea_ice  # noqa: E402
from tests.validation_data.Grenfell_light_2007.validate_grenfell_light_2007 import (  # noqa: E402
    catalogue, parse_albv, parse_albi,
)

WL_NM = np.arange(205, 4999, 10)
MASK_VIS  = (WL_NM >= 400) & (WL_NM <= 1000)
MASK_VSW  = (WL_NM >= 400) & (WL_NM <= 2000)
EXPECTED = ("FYI_bare", "FYI_summer")


def _interp(wl_obs, alb_obs):
    f = interp1d(wl_obs, alb_obs, kind="linear", bounds_error=False,
                 fill_value=np.nan)
    return f(WL_NM)


def _paired_summer_obs():
    """Summer ALBV spectra with the ALBI white-ice SWIR column spliced in.

    Returns [(date, sza, month, obs)] — identical construction to
    sheba_classification_validation.test_summer_white_ice_vis_swir.
    """
    _, summer = catalogue()
    out = []
    for e in summer:
        tag = e["date"].replace("1998-", "").replace("-", "")
        albi = GRENFELL_DIR / f"ICEDATA_OPTICS_SPECALB_ALBI{tag}.CSV"
        if not albi.exists():
            continue
        wl_v, alb_v, _ = parse_albv(e["path"])
        wl_i, wi_alb, _, _ = parse_albi(albi)
        if wl_v is None or wl_i is None:
            continue
        obs = _interp(wl_v, alb_v)
        obs_wi = _interp(wl_i, wi_alb)
        ir = (WL_NM >= 1100) & (WL_NM <= 2000) & ~np.isnan(obs_wi)
        obs[ir] = obs_wi[ir]
        out.append((e["date"], e["sza"], int(e["date"][5:7]), obs))
    return out


def _classify(obs, window_mask, sza, month, method, model_error=None):
    m = window_mask & ~np.isnan(obs)
    r = retrieve_sea_ice(
        observed=obs, wavelength_mask=m, solzen=sza, direct=1,
        known_month=month, method=method, model_error=model_error,
    )
    return r


# (label, method kwarg, model_error kwarg).  "oe+Se" adds the A7 field-
# calibrated forward-model error covariance to the OE measurement covariance
# (biosnicar.inverse.model_error) — the acceptance test for whether honest
# S_e repairs the evidence-based classification.
VARIANTS = (
    ("L-BFGS-B", "L-BFGS-B", None),
    ("oe", "oe", None),
    ("oe+Se", "oe", True),
)


def run():
    cases = _paired_summer_obs()
    print(f"\nOE-mode SHEBA SWIR experiment (B6/A7) — {len(cases)} summer "
          f"dates with paired ALBI SWIR\n")
    print(f"  {'date':<12} {'window':<9} {'variant':<9} {'winner':<12} "
          f"{'conf':>6}  ok")
    print("  " + "-" * 60)

    results = []
    tally = {}
    for label, method, me in VARIANTS:
        for wname, wmask in (("VIS", MASK_VIS), ("VIS+SWIR", MASK_VSW)):
            key = (label, wname)
            tally[key] = [0, 0]
            for date, sza, month, obs in cases:
                r = _classify(obs, wmask, sza, month, method, model_error=me)
                ok = r.surface_type in EXPECTED
                tally[key][0] += ok
                tally[key][1] += 1
                results.append(dict(
                    date=date, window=wname, method=label,
                    winner=r.surface_type, confidence=float(r.confidence),
                    correct=bool(ok),
                    class_probabilities={k: float(v) for k, v
                                         in r.class_probabilities.items()},
                ))
                print(f"  {date:<12} {wname:<9} {label:<9} "
                      f"{r.surface_type:<12} {r.confidence:6.2f}  "
                      f"{'Y' if ok else 'N'}")

    print("\n  Summary (correct / n):")
    for (label, wname), (c, n) in tally.items():
        print(f"    {label:<9} {wname:<9} {c}/{n}  ({c/n:.0%})")

    return results, {f"{m}|{w}": f"{c}/{n}" for (m, w), (c, n) in tally.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    results, summary = run()
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"summary": summary, "rows": results}, indent=2))
        print(f"\n  json: {args.json}")


if __name__ == "__main__":
    main()
