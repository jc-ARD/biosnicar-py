#!/usr/bin/env python3
"""A1 audit: FYI_bare emulator accuracy investigation.

Measures held-out spectral accuracy of the FYI_bare emulator and runs the
four investigation steps from the build guide:

  1. rho_DL / brine_volume_fraction degeneracy analysis (Jacobian collinearity)
  2. Training sample density (30k -> 60k)
  3. PCA dimensionality (0.999 -> 0.9999 retained variance)
  4. Architecture depth ((256,256,128,64) -> (256,256,256,128,64))

Usage::

    python scripts/experiments/fyi_bare_audit.py baseline     # fast: holdout + eval + degeneracy
    python scripts/experiments/fyi_bare_audit.py experiments  # slow: training runs

Results accumulate in data/emulators/experiments/fyi_bare_audit_results.json.
The held-out set (seed 777, distinct from training seed 42) is cached in
data/emulators/experiments/fyi_bare_holdout.npz.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from biosnicar.emulator import Emulator, _latin_hypercube
from biosnicar.sea_ice.emulator_configs import SEA_ICE_EMULATOR_CONFIGS

EXP_DIR = Path(__file__).resolve().parents[2] / "data" / "emulators" / "experiments"
HOLDOUT_PATH = EXP_DIR / "fyi_bare_holdout.npz"
RESULTS_PATH = EXP_DIR / "fyi_bare_audit_results.json"

CFG = SEA_ICE_EMULATOR_CONFIGS["FYI_bare"]
PARAM_NAMES = list(CFG["params"].keys())
N_HOLDOUT = 2000
HOLDOUT_SEED = 777


def _sample_params(n, seed):
    """LHS sample of FYI_bare params matching the training distribution."""
    lo = np.array([CFG["params"][p][0] for p in PARAM_NAMES])
    hi = np.array([CFG["params"][p][1] for p in PARAM_NAMES])
    lhs = _latin_hypercube(n, len(PARAM_NAMES), seed=seed)
    scaled = lo + lhs * (hi - lo)
    for j, name in enumerate(PARAM_NAMES):
        if name == "sea_ice_bubble_radius":  # log-sampled during training
            llo, lhi = np.log10(lo[j] + 1), np.log10(hi[j] + 1)
            scaled[:, j] = 10 ** (llo + lhs[:, j] * (lhi - llo)) - 1
        elif name == "solzen":
            scaled[:, j] = np.clip(np.round(scaled[:, j]), 1, 89)
        elif name == "direct":
            scaled[:, j] = np.round(scaled[:, j])
    return scaled


def make_holdout():
    """Generate (or load) the held-out forward-model test set."""
    if HOLDOUT_PATH.exists():
        d = np.load(HOLDOUT_PATH)
        return d["params"], d["albedos"], d["flx_slr"]

    from biosnicar.drivers.run_model import run_model

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    scaled = _sample_params(N_HOLDOUT, HOLDOUT_SEED)
    albedos, flx = [], None
    t0 = time.time()
    for i in range(N_HOLDOUT):
        p = dict(zip(PARAM_NAMES, scaled[i]))
        p["solzen"] = int(p["solzen"])
        p["direct"] = int(p["direct"])
        out = run_model(**CFG["transform_fn"](p))
        albedos.append(np.asarray(out.albedo, dtype=np.float64))
        if flx is None:
            flx = np.asarray(out.flx_slr, dtype=np.float64)
        if (i + 1) % 500 == 0:
            print(f"  holdout {i+1}/{N_HOLDOUT}  ({time.time()-t0:.0f}s)", flush=True)
    albedos = np.array(albedos)
    physical = np.all((albedos >= 0) & (albedos <= 1.01), axis=1)
    print(f"  {int((~physical).sum())} unphysical spectra dropped")
    np.savez_compressed(HOLDOUT_PATH, params=scaled[physical],
                        albedos=albedos[physical], flx_slr=flx)
    return scaled[physical], albedos[physical], flx


def evaluate(emu, params, ref, flx):
    pred = emu.predict_batch(params)
    res = ref - pred
    ss_tot = np.sum((ref - ref.mean()) ** 2)
    r2_pooled = float(1 - np.sum(res**2) / ss_tot)
    per_r2 = 1 - np.sum(res**2, axis=1) / np.maximum(
        np.sum((ref - ref.mean(axis=1, keepdims=True)) ** 2, axis=1), 1e-30)
    bba_ref = ref @ flx / flx.sum()
    bba_pred = pred @ flx / flx.sum()
    return {
        "r2_pooled_albedo": r2_pooled,
        "r2_per_spectrum_median": float(np.median(per_r2)),
        "r2_per_spectrum_p5": float(np.percentile(per_r2, 5)),
        "mae_spectral": float(np.mean(np.abs(res))),
        "max_err_spectral": float(np.max(np.abs(res))),
        "bba_mae": float(np.mean(np.abs(bba_ref - bba_pred))),
        "bba_max_err": float(np.max(np.abs(bba_ref - bba_pred))),
        "n_holdout": int(len(ref)),
    }


def degeneracy_analysis():
    """Jacobian collinearity between rho_DL and brine_volume_fraction.

    |cosine| near 1 at most base points means the two parameters push the
    spectrum along the same direction (a degenerate ridge for inversion).
    """
    from biosnicar.drivers.run_model import run_model

    rng = np.random.default_rng(123)
    cosines, base_points = [], []
    for _ in range(20):
        p = {
            "brine_volume_fraction": rng.uniform(0.03, 0.12),
            "sea_ice_bubble_radius": 10 ** rng.uniform(np.log10(60), np.log10(900)),
            "black_carbon": 0.0,
            "rho_DL": rng.uniform(830, 890),
            "solzen": int(rng.integers(25, 75)),
            "direct": 1,
        }
        def alb(q):
            return np.asarray(run_model(**CFG["transform_fn"](q)).albedo)

        dvb = 0.005 * (0.141 - 0.019)
        drho = 0.005 * (900 - 820)
        p_vb = dict(p); p_vb["brine_volume_fraction"] += dvb
        p_rho = dict(p); p_rho["rho_DL"] += drho
        a0 = alb(p)
        j_vb = (alb(p_vb) - a0) / dvb * (0.141 - 0.019)   # range-normalised
        j_rho = (alb(p_rho) - a0) / drho * (900 - 820)
        denom = np.linalg.norm(j_vb) * np.linalg.norm(j_rho)
        cos = float(j_vb @ j_rho / denom) if denom > 0 else 0.0
        cosines.append(cos)
        base_points.append({k: float(v) for k, v in p.items()})
    return {
        "jacobian_cosine_mean": float(np.mean(np.abs(cosines))),
        "jacobian_cosine_min": float(np.min(np.abs(cosines))),
        "jacobian_cosine_max": float(np.max(np.abs(cosines))),
        "cosines": [float(c) for c in cosines],
        "note": "|cos|~1 => rho_DL and Vb degenerate; <0.9 => separable",
    }


def _save_result(key, value):
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    results[key] = value
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"[saved] {key}: {json.dumps(value)[:200]}", flush=True)


def run_experiment(key, n_samples, hidden, pca, params, ref, flx):
    print(f"\n=== {key}: n={n_samples} hidden={hidden} pca={pca} ===", flush=True)
    emu_path = EXP_DIR / f"fyi_bare_{key}.npz"
    t0 = time.time()
    if emu_path.exists():
        emu = Emulator.load(emu_path)
    else:
        emu = Emulator.build(
            params=CFG["params"], n_samples=n_samples,
            transform_fn=CFG["transform_fn"], hidden_layer_sizes=hidden,
            pca_components=pca, seed=42, progress=False,
        )
        emu.save(emu_path)
    metrics = evaluate(emu, params, ref, flx)
    metrics.update({
        "n_samples": int(n_samples), "hidden": list(hidden), "pca": float(pca),
        "n_pca_components": int(emu.n_pca_components),
        "training_r2_pca_space": float(emu.training_score),
        "build_minutes": round((time.time() - t0) / 60, 1),
    })
    _save_result(key, metrics)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    print("Building/loading held-out set...", flush=True)
    params, ref, flx = make_holdout()

    if mode == "baseline":
        emu = Emulator.load(CFG["emulator_file"])
        m = evaluate(emu, params, ref, flx)
        m["training_r2_pca_space"] = emu.training_score
        m["n_pca_components"] = emu.n_pca_components
        _save_result("baseline_current", m)
        print("Running degeneracy analysis...", flush=True)
        _save_result("degeneracy_rho_vb", degeneracy_analysis())
    elif mode == "experiments":
        run_experiment("exp1_60k", 60000, (256, 256, 128, 64), 0.999, params, ref, flx)
        run_experiment("exp2_deeper", 30000, (256, 256, 256, 128, 64), 0.999, params, ref, flx)
        run_experiment("exp3_pca9999", 30000, (256, 256, 128, 64), 0.9999, params, ref, flx)
        run_experiment("exp4_combo", 60000, (256, 256, 256, 128, 64), 0.9999, params, ref, flx)
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
