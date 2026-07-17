#!/usr/bin/env python3
"""OE band-mode classification experiment (B6 follow-up, the IceNav path).

The A7 model-error covariance repaired OE classification in *spectral* mode
(docs/OE_MODEL_ERROR_EXPERIMENT.md), but the production consumer (IceNav's
mile-ahead inversion) runs OE in *band mode* — Sentinel-2 surface reflectance,
9 VIS-NIR bands (B1..B8A), no SWIR — where the model-error term does not apply
(band-mode S_e is the atmospheric-correction budget, roadmap B-MS1). That path
was never measured. This experiment measures it.

Design: the 23 SHEBA ALBV dates (spring snow + summer bare ice, 400-1000 nm),
convolved to satellite bands and classified with default vs OE, mirroring
IceNav's exact call — `retrieve_sea_ice(platform=..., observed_band_names=...,
method="oe", obs_uncertainty=default_obs_uncertainty(...))`.

Two band sets:
  * S2-9  : B1,B2,B3,B4,B5,B6,B7,B8,B8A — IceNav's INVERSION_BANDS (VIS-NIR)
  * S2-4  : B2,B3,B4,B8 — the SHEBA-suite subset, for continuity

Expected: spring -> FYI_snow; summer -> FYI_bare or FYI_summer.

Usage::

    python tests/validation_data/oe_bandmode_classification_experiment.py [--json out.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from biosnicar.bands import to_platform  # noqa: E402
from biosnicar.sea_ice.retrieve import retrieve_sea_ice  # noqa: E402
from biosnicar.sea_ice.sensor_config import default_obs_uncertainty  # noqa: E402
from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators  # noqa: E402
from Grenfell_light_2007.validate_grenfell_light_2007 import (  # noqa: E402
    catalogue, parse_albv,
)

WL_NM = np.arange(205, 4999, 10)
VIS = (WL_NM >= 400) & (WL_NM <= 1000)
S2_9 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A"]  # IceNav config
S2_4 = ["B2", "B3", "B4", "B8"]

_FLX = None


def _flx():
    global _FLX
    if _FLX is None:
        _FLX = load_sea_ice_emulators(["FYI_pond"])["FYI_pond"].flx_slr
    return _FLX


def _bands(obs, band_names):
    """Convolve a spectrum to Sentinel-2 bands (VIS-NIR gaps zero-filled;
    all requested bands lie within 400-1000 nm ALBV coverage)."""
    filled = np.where(np.isnan(obs), 0.0, obs)
    br = to_platform(filled, "sentinel2", flx_slr=_flx())
    return np.array([float(getattr(br, b)) for b in band_names])


def _cases():
    spring, summer = catalogue()
    out = []
    for entries, expected, label in (
        (spring, ("FYI_snow",), "spring_snow"),
        (summer, ("FYI_bare", "FYI_summer"), "summer_bare_ice"),
    ):
        for e in entries:
            wl, alb, _ = parse_albv(e["path"])
            if wl is None:
                continue
            f = interp1d(wl, alb, kind="linear", bounds_error=False,
                         fill_value=np.nan)
            obs = f(WL_NM)
            if (VIS & np.isfinite(obs)).sum() < 30:
                continue
            out.append((e["date"], e["sza"], int(e["date"][5:7]),
                        expected, label, obs))
    return out


VARIANTS = (("default", "L-BFGS-B"), ("oe", "oe"))
BANDSETS = (("S2-9(IceNav)", S2_9), ("S2-4", S2_4))


def run():
    cases = _cases()
    print(f"\nOE band-mode classification experiment (IceNav path) — "
          f"{len(cases)} SHEBA dates\n")
    print(f"  {'date':<12} {'label':<16} {'bands':<13} {'variant':<8} "
          f"{'winner':<12} {'conf':>6} ok")
    print("  " + "-" * 74)

    results, tally = [], {}
    for bname, bands in BANDSETS:
        unc = default_obs_uncertainty("sentinel2", bands)
        for vlabel, method in VARIANTS:
            for grp in ("spring_snow", "summer_bare_ice"):
                tally[(bname, vlabel, grp)] = [0, 0]
            for date, sza, month, expected, label, obs in cases:
                y = _bands(obs, bands)
                r = retrieve_sea_ice(
                    observed=y, platform="sentinel2",
                    observed_band_names=bands, obs_uncertainty=unc,
                    solzen=float(np.clip(sza, 20, 89)), direct=1,
                    known_month=month, method=method,
                )
                ok = r.surface_type in expected
                t = tally[(bname, vlabel, label)]
                t[0] += ok
                t[1] += 1
                results.append(dict(
                    date=date, label=label, bands=bname, variant=vlabel,
                    winner=r.surface_type, confidence=float(r.confidence),
                    correct=bool(ok),
                    class_probabilities={k: float(v) for k, v
                                         in r.class_probabilities.items()},
                ))
                print(f"  {date:<12} {label:<16} {bname:<13} {vlabel:<8} "
                      f"{r.surface_type:<12} {r.confidence:6.2f} "
                      f"{'Y' if ok else 'N'}")

    print("\n  Summary (correct / n):")
    for (bname, vlabel, grp), (c, n) in tally.items():
        if n:
            print(f"    {bname:<13} {vlabel:<8} {grp:<16} {c}/{n}  ({c/n:.0%})")

    summary = {f"{b}|{v}|{g}": f"{c}/{n}"
               for (b, v, g), (c, n) in tally.items() if n}
    return results, summary


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
