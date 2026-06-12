#!/usr/bin/env python3
"""Synthetic parameter-retrieval validation (E1).

For every surface type, generates 50 test spectra with the FORWARD MODEL
(``transform_fn`` → ``run_model``, never the emulator) at parameters drawn
from seed 2026 — disjoint from the training seed (42) and the emulator-audit
holdout seed (777).  Each spectrum is retrieved single-type with geometry
(solzen, direct) fixed to truth and no seasonal priors (``known_month=None``)
so the numbers measure parameter identifiability, not prior pull.

Reports per parameter: bias, RMSE, R², and the fraction of retrievals at a
training bound (1% tolerance — same as the AT_BOUNDS quality flag).

Usage::

    python tests/validation_data/parameter_retrieval_validation.py
    python tests/validation_data/parameter_retrieval_validation.py --types FYI_pond
    python tests/validation_data/parameter_retrieval_validation.py --n 20

Writes a markdown table (for docs/sea_ice_validation.md) and a JSON dump to
tests/validation_data/parameter_retrieval_results.{md,json}.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from biosnicar.drivers.run_model import run_model
from biosnicar.emulator import _LOG_SAMPLE_PARAMS
from biosnicar.sea_ice.emulator_configs import (
    SEA_ICE_EMULATOR_CONFIGS,
    load_sea_ice_emulators,
    trained_emulator_names,
)
from biosnicar.sea_ice.retrieve import retrieve_sea_ice

SEED = 2026   # document: held-out seed, disjoint from training (42) / audit (777)
N_DEFAULT = 50
OUT_MD = Path(__file__).parent / "parameter_retrieval_results.md"
OUT_JSON = Path(__file__).parent / "parameter_retrieval_results.json"


def _sample(rng, name, lo, hi):
    if name == "direct":
        return int(rng.integers(0, 2))
    if name == "solzen":
        return int(rng.integers(int(lo), int(hi) + 1))
    if name in _LOG_SAMPLE_PARAMS:
        return float(10 ** rng.uniform(np.log10(lo + 1), np.log10(hi + 1)) - 1)
    return float(rng.uniform(lo, hi))


def _forward(stype, params):
    if stype == "open_water":
        from biosnicar.sea_ice.open_water import OpenWaterModel
        return OpenWaterModel().predict(**params)
    cfg = SEA_ICE_EMULATOR_CONFIGS[stype]
    return np.asarray(run_model(**cfg["transform_fn"](params)).albedo)


def validate_type(stype, emulators, n, rng):
    cfg = SEA_ICE_EMULATOR_CONFIGS[stype]
    bounds = cfg["params"]
    free = [p for p in bounds if p not in ("solzen", "direct")]
    truths, fits = [], []
    t0 = time.time()
    for i in range(n):
        truth = {p: _sample(rng, p, *bounds[p]) for p in bounds}
        try:
            obs = _forward(stype, truth)
        except Exception as exc:  # noqa: BLE001 — skip unphysical corners
            warnings.warn(f"{stype}: forward model failed ({exc}); skipping")
            continue
        if not np.all((obs >= 0) & (obs <= 1.01)):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = retrieve_sea_ice(
                observed=obs,
                emulators=emulators,
                surface_types=[stype],
                solzen=truth["solzen"],
                direct=truth.get("direct", 1),
            )
        truths.append(truth)
        fits.append(result.parameters)
    print(f"  {stype}: {len(truths)}/{n} retrievals in {time.time()-t0:.0f}s",
          flush=True)

    stats = {}
    for p in free:
        t = np.array([tr[p] for tr in truths])
        r = np.array([f[p] for f in fits])
        lo, hi = float(bounds[p][0]), float(bounds[p][1])
        tol = 0.01 * (hi - lo)
        at_bound = float(np.mean((r <= lo + tol) | (r >= hi - tol)))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        r2 = 1.0 - float(np.sum((r - t) ** 2)) / ss_tot if ss_tot > 0 else np.nan
        stats[p] = {
            "bias": float(np.mean(r - t)),
            "rmse": float(np.sqrt(np.mean((r - t) ** 2))),
            "r2": r2,
            "at_bound_fraction": at_bound,
            "n": len(t),
        }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    args = ap.parse_args()

    types = args.types or trained_emulator_names() + ["open_water"]
    fleet = load_sea_ice_emulators(types)
    rng = np.random.default_rng(SEED)

    all_stats = {}
    for stype in types:
        all_stats[stype] = validate_type(stype, fleet, args.n, rng)

    lines = [
        "## Parameter retrieval validation (E1)",
        "",
        f"{args.n} forward-model spectra per type, seed {SEED}; single-type",
        "retrieval, geometry fixed to truth, no seasonal priors.",
        "",
        "| Surface type | Parameter | Bias | RMSE | R² | At-bound |",
        "|---|---|---|---|---|---|",
    ]
    for stype, stats in all_stats.items():
        for p, s in stats.items():
            lines.append(
                f"| {stype} | {p} | {s['bias']:+.3g} | {s['rmse']:.3g} "
                f"| {s['r2']:.3f} | {s['at_bound_fraction']:.0%} |"
            )
    md = "\n".join(lines)
    OUT_MD.write_text(md + "\n")
    OUT_JSON.write_text(json.dumps(all_stats, indent=2))
    print(md)
    print(f"\nSaved {OUT_MD} and {OUT_JSON}")


if __name__ == "__main__":
    main()
