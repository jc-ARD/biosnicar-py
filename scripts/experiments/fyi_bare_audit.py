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

import itertools
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
    """rho_DL / brine_volume_fraction collinearity, two ways.

    The Jacobian cosine is the prior-free, noise-free proxy: it asks whether
    the two parameters push the spectrum in the same direction.  The quantity
    that actually governs a retrieval is the **posterior correlation**, which
    additionally accounts for the other fitted parameters, the measurement
    noise and the calibrated forward-model error.  Both are reported, because
    they differ materially and only the second is decision-relevant.

    Marginal 1-sigma is inflated over its value-if-the-partner-were-known by
    exactly 1/sqrt(1 - r**2), with r the posterior correlation.

    Ranges come from the live config rather than being hardcoded, so this
    cannot silently drift out of date when the emulator is reparameterised.
    """
    from biosnicar.drivers.run_model import run_model
    from biosnicar.inverse.optimize import retrieve
    from biosnicar.inverse.model_error import ModelErrorCovariance

    P = CFG["params"]
    vb_lo, vb_hi = P["brine_volume_fraction"]
    rho_lo, rho_hi = P["rho_DL"]
    bbl_lo, bbl_hi = P["sea_ice_bubble_radius"]

    rng = np.random.default_rng(123)
    cosines, corrs, inflations, base_points = [], [], [], []
    all_pairs = {}

    emu = None
    try:
        from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
        emu = load_sea_ice_emulators(["FYI_bare"])["FYI_bare"]
        ME = ModelErrorCovariance.load()
    except Exception as exc:                      # pragma: no cover
        print(f"  posterior correlation unavailable ({exc}); cosine only")

    FIT = ["brine_volume_fraction", "sea_ice_bubble_radius",
           "black_carbon", "rho_DL"]

    bc_lo, bc_hi = P["black_carbon"]
    for _ in range(24):
        p = {
            "brine_volume_fraction": rng.uniform(vb_lo, vb_hi),
            "sea_ice_bubble_radius": 10 ** rng.uniform(np.log10(bbl_lo + 1),
                                                       np.log10(bbl_hi + 1)) - 1,
            # Sampled, not pinned at zero: black_carbon sits on its lower
            # bound otherwise, which makes its Jacobian a boundary case and
            # its correlations unrepresentative of a real retrieval.
            "black_carbon": 10 ** rng.uniform(np.log10(bc_lo + 1),
                                              np.log10(bc_hi + 1)) - 1,
            "rho_DL": rng.uniform(rho_lo, rho_hi),
            "solzen": int(rng.integers(25, 75)),
            "direct": 1,
        }

        def alb(q):
            return np.asarray(run_model(**CFG["transform_fn"](q)).albedo)

        # --- 1. Jacobian cosine (central differences, clipped to bounds) ---
        dvb = 0.005 * (vb_hi - vb_lo)
        drho = 0.005 * (rho_hi - rho_lo)
        p_vb = dict(p); p_vb["brine_volume_fraction"] = min(p["brine_volume_fraction"] + dvb, vb_hi)
        p_rho = dict(p); p_rho["rho_DL"] = min(p["rho_DL"] + drho, rho_hi)
        a0 = alb(p)
        j_vb = (alb(p_vb) - a0) / (p_vb["brine_volume_fraction"] - p["brine_volume_fraction"])
        j_rho = (alb(p_rho) - a0) / (p_rho["rho_DL"] - p["rho_DL"])
        denom = np.linalg.norm(j_vb) * np.linalg.norm(j_rho)
        cosines.append(float(j_vb @ j_rho / denom) if denom > 0 else 0.0)

        # --- 2. Posterior correlation from the real OE path ---
        if emu is not None:
            y = np.asarray(emu.predict(**p), float)
            try:
                r = retrieve(
                    observed=y, parameters=FIT, emulator=emu,
                    fixed_params={"solzen": p["solzen"], "direct": p["direct"]},
                    bounds={k: P[k] for k in FIT},
                    x0={k: p[k] for k in FIT},
                    obs_uncertainty=np.full(480, 0.01),
                    method="oe", model_error=ME,
                )
                cm = r.parameter_correlation()
                c = cm[("brine_volume_fraction", "rho_DL")]
                corrs.append(float(c))
                inflations.append(float(1.0 / np.sqrt(max(1.0 - c ** 2, 1e-12))))
                for a, b in itertools.combinations(FIT, 2):
                    all_pairs.setdefault((a, b), []).append(float(cm[(a, b)]))
            except Exception:                     # pragma: no cover
                pass

        base_points.append({k: float(v) for k, v in p.items()})

    out = {
        "jacobian_cosine_mean": float(np.mean(np.abs(cosines))),
        "jacobian_cosine_min": float(np.min(np.abs(cosines))),
        "jacobian_cosine_max": float(np.max(np.abs(cosines))),
        "cosines": [float(c) for c in cosines],
        "note": ("Jacobian cosine is a prior-free proxy and runs HIGHER than "
                 "the posterior correlation, which is what governs retrieval. "
                 "Quote posterior_corr_*, not the cosine."),
    }
    if corrs:
        a = np.abs(corrs)
        out.update({
            "posterior_corr_mean": float(a.mean()),
            "posterior_corr_min": float(a.min()),
            "posterior_corr_max": float(a.max()),
            "posterior_corr_signs": {"positive": int(sum(1 for c in corrs if c > 0)),
                                     "negative": int(sum(1 for c in corrs if c < 0))},
            "frac_abs_corr_gt_0.8": float(np.mean(a > 0.8)),
            "sigma_inflation_mean": float(np.mean(inflations)),
            "sigma_inflation_max": float(np.max(inflations)),
        })
    if all_pairs:
        # Every pair, not just the one historically tracked -- the largest
        # degeneracy in this emulator is bubble_radius/rho_DL, which went
        # unmeasured for as long as only rho_DL/Vb was looked at.
        pairs = {}
        for (a, b), v in all_pairs.items():
            av = np.abs(v)
            pairs[f"{a}|{b}"] = {
                "mean_abs": float(av.mean()),
                "max_abs": float(av.max()),
                "n_positive": int(sum(1 for x in v if x > 0)),
                "n_negative": int(sum(1 for x in v if x < 0)),
                "sigma_inflation_mean": float(np.mean(
                    1.0 / np.sqrt(np.clip(1.0 - av ** 2, 1e-12, None)))),
                "sigma_inflation_max": float(np.max(
                    1.0 / np.sqrt(np.clip(1.0 - av ** 2, 1e-12, None)))),
            }
        out["posterior_corr_all_pairs"] = dict(
            sorted(pairs.items(), key=lambda kv: -kv[1]["mean_abs"]))
    return out


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
