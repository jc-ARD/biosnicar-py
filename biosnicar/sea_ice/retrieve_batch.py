"""Vectorised optimal-estimation retrieval over a whole scene (roadmap G3).

``retrieve_sea_ice_batch(method="oe", engine="vectorized")`` dispatches here.
Instead of running one Python optimiser per pixel (the ``engine="loop"`` /
joblib path), this solves all N pixels of each fleet emulator simultaneously
with the batched Gauss-Newton engine
(:func:`biosnicar.inverse.optimal_estimation_batch.optimal_estimation_batch`),
exploiting that the emulator MLP and the band convolution are matmuls that
handle ``(N, ...)`` natively. Per-pixel priors, model-error S_e and class
priors are the *shared* inputs the batch engine assumes.

The per-emulator OE setup (parameters, log-space handling, x_a / S_a / S_e) is
pixel-independent, built once per emulator and reused for the whole scene; the
per-emulator regularization is the same helper the loop path uses
(``retrieve._emulator_regularization``), so the two paths cannot diverge — a
machine-precision equivalence test (``tests/test_retrieve_batch.py``) enforces
it. Classification uses the OE Laplace evidence combined with class priors,
exactly as the scalar OE path (evidence over the full observation, no per-type
band masks — see audit B6 / SEA_ICE_RETRIEVAL.md §2.5).

Scope: ``method="oe"`` only; spectral and SRF-convolution band mode; scalar or
per-pixel solzen/direct; scene-level priors / model_error. Per-pixel
known_month, SSA retrieval, L-BFGS-B/MCMC and flag_prior_influence keep the
loop path.
"""

import numpy as np

from biosnicar.inverse.optimize import (
    DEFAULT_BOUNDS, _LOG_SPACE_PARAMS, _from_log, _to_log,
)
from biosnicar.inverse.optimal_estimation_batch import optimal_estimation_batch

_LN10 = np.log(10.0)
_band_matrix_cache: dict = {}


def _predict_batch(model, points, all_names, fixed_extra):
    """(N, n_all) params -> (N, 480). Uses the emulator's vectorised
    predict_batch when available; falls back to a per-row loop for analytical
    duck-typed models (open_water), which also need fixed params that are not
    represented as columns (e.g. `direct`)."""
    if hasattr(model, "predict_batch"):
        return model.predict_batch(points)
    N = points.shape[0]

    def _extra(i):
        # per-pixel array fixed params (e.g. a per-pixel `direct`) index by row
        return {k: (v[i] if (isinstance(v, np.ndarray) and v.shape[:1] == (N,))
                    else v) for k, v in fixed_extra.items()}

    return np.stack([
        model.predict(**{**dict(zip(all_names, row)), **_extra(i)})
        for i, row in enumerate(points)
    ])


def _band_weight_matrix(platform, band_names, flx_slr):
    """(480, k) matrix W such that band_values = albedo @ W.

    Band convolution is linear in albedo, so W[:, b] = to_platform(e_i)[b]
    over the 480 unit spectra. Built once per (platform, bands, flux) and
    cached — amortised over all pixels and all fleet emulators.
    """
    from biosnicar.bands import to_platform

    key = (platform, tuple(band_names), flx_slr.tobytes())
    W = _band_matrix_cache.get(key)
    if W is not None:
        return W
    nwvl = flx_slr.size
    W = np.empty((nwvl, len(band_names)))
    e = np.zeros(nwvl)
    for i in range(nwvl):
        e[i] = 1.0
        br = to_platform(e, platform, flx_slr=flx_slr)
        W[i, :] = [getattr(br, b) for b in band_names]
        e[i] = 0.0
    _band_matrix_cache[key] = W
    return W


def _emulator_setup(emu, retrieve_params, emu_fixed, emu_reg, obs_uncertainty,
                    sel, model_error, W):
    """Pixel-independent OE inputs for one emulator (mirrors optimize._run_oe).

    Returns a dict with the parameter list, log mask, opt-space bounds/x0,
    prior mean/covariance x_a/S_a, measurement covariance S_e, and a batched
    forward ``F(X (N,n)) -> (N,m)``.
    """
    params = list(retrieve_params)
    n = len(params)
    emu_bounds = emu.bounds
    log_mask = np.array([p in _LOG_SPACE_PARAMS for p in params])

    # Linear bounds, then opt-space bounds and x0 (log midpoint for log params).
    lin_bounds = [emu_bounds.get(p, DEFAULT_BOUNDS.get(p)) for p in params]
    opt_bounds = np.array([
        (_to_log(lo), _to_log(hi)) if log_mask[i] else (lo, hi)
        for i, (lo, hi) in enumerate(lin_bounds)
    ], dtype=float)
    opt_x0 = np.array([float(np.mean(opt_bounds[i])) for i in range(n)])

    # x_a / S_a in opt space (mirrors _run_oe exactly).
    x_a = np.empty(n)
    sig_a = np.empty(n)
    for i, p in enumerate(params):
        lo, hi = opt_bounds[i]
        if p in emu_reg:
            mu_lin, sg_lin = emu_reg[p]
            if log_mask[i]:
                x_a[i] = float(_to_log(mu_lin))
                sig_a[i] = sg_lin / ((mu_lin + 1.0) * _LN10)
            else:
                x_a[i] = mu_lin
                sig_a[i] = sg_lin
        else:
            x_a[i] = opt_x0[i]
            sig_a[i] = hi - lo
        x_a[i] = float(np.clip(x_a[i], lo, hi))
        sig_a[i] = float(max(sig_a[i], 1e-6))
    S_a = sig_a ** 2

    # S_e: instrument noise (+ model error in spectral mode).
    m = int(sel.sum()) if W is None else W.shape[1]
    if obs_uncertainty is not None:
        sig_e = np.asarray(obs_uncertainty, dtype=float)
        sig_e = sig_e[sel] if (W is None and sig_e.size == sel.size) else sig_e
    else:
        sig_e = np.full(m, 0.02)
    if model_error is not None:
        if W is not None:
            raise ValueError("model_error is spectral-mode only (band-mode S_e "
                             "is the atmospheric-correction budget, B-MS1).")
        from biosnicar.inverse.model_error import ModelErrorCovariance
        M = (model_error if hasattr(model_error, "dense")
             else ModelErrorCovariance.load())
        S_e = np.diag(sig_e ** 2) + M.dense(sel)
    else:
        S_e = sig_e ** 2

    # Batched forward: opt-space X (N,n) -> measurement (N,m).
    all_names = list(emu.param_names)
    n_all = len(all_names)
    free_pos = [all_names.index(p) for p in params]
    fixed_cols = {all_names.index(k): v for k, v in emu_fixed.items()
                  if k in all_names}
    # Fixed params not represented as columns (e.g. `direct` for open_water) —
    # passed to the analytical fallback in _predict_batch.
    fixed_extra = {k: v for k, v in emu_fixed.items() if k not in all_names}

    def forward(X):
        N = X.shape[0]
        Xlin = X.copy()
        if log_mask.any():
            Xlin[:, log_mask] = _from_log(X[:, log_mask])
        pts = np.empty((N, n_all))
        for col, v in fixed_cols.items():
            pts[:, col] = v                      # scalar or (N,) per-pixel
        for k, pos in enumerate(free_pos):
            pts[:, pos] = Xlin[:, k]
        alb = _predict_batch(emu, pts, all_names, fixed_extra)   # (N, 480)
        return alb @ W if W is not None else alb[:, sel]

    return dict(params=params, log_mask=log_mask, opt_bounds=opt_bounds,
                opt_x0=opt_x0, x_a=x_a, S_a=S_a, S_e=S_e, forward=forward)


def _backmap(params, log_mask, batch):
    """OE batch result (opt space) -> linear best_fit / uncertainty (N, n)."""
    X = batch.x
    sig = batch.sigma
    lin = X.copy()
    unc = sig.copy()
    if log_mask.any():
        lin[:, log_mask] = _from_log(X[:, log_mask])
        unc[:, log_mask] = (lin[:, log_mask] + 1.0) * _LN10 * sig[:, log_mask]
    return lin, unc


def vectorized_oe_fleet(observed, emulators, emu_regs, shared_fixed,
                        obs_uncertainty, wavelength_mask, platform,
                        observed_band_names, model_error, class_log_prior,
                        surface_descriptions, max_iter=20):
    """Run the vectorised OE fleet over N pixels; return per-pixel records.

    ``observed`` is (N, obs_dim): 480-band spectra (spectral) or band values
    (band mode). Returns a list of record dicts compatible with
    ``SeaIceSceneResult.from_records`` (via ``_result_to_record``'s schema).
    """
    observed = np.asarray(observed, dtype=float)
    N = observed.shape[0]

    if platform is not None:
        flx = next(iter(emulators.values())).flx_slr
        W = _band_weight_matrix(platform, observed_band_names, flx)
        sel = np.ones(480, dtype=bool)          # unused in band mode
        Y = observed
    else:
        W = None
        sel = (np.ones(480, dtype=bool) if wavelength_mask is None
               else np.asarray(wavelength_mask, dtype=bool))
        Y = np.nan_to_num(observed[:, sel])

    # Per-emulator batched retrieval.
    names = list(emulators)
    per_emu = {}
    ev_stack = np.full((N, len(names)), -np.inf)
    for j, name in enumerate(names):
        emu = emulators[name]
        params = [p for p in emu.param_names if p not in shared_fixed]
        emu_fixed = {k: v for k, v in shared_fixed.items()
                     if k in emu.param_names or k in ("solzen", "direct")}
        setup = _emulator_setup(emu, params, emu_fixed, emu_regs[name],
                                obs_uncertainty, sel, model_error, W)
        batch = optimal_estimation_batch(
            setup["forward"], Y, setup["x_a"], setup["S_a"], setup["S_e"],
            bounds=setup["opt_bounds"], x0=setup["opt_x0"], max_iter=max_iter,
        )
        lin, unc = _backmap(setup["params"], setup["log_mask"], batch)
        per_emu[name] = dict(setup=setup, batch=batch, lin=lin, unc=unc)
        # class prior enters the evidence (B6/C1): posterior ∝ P(y|c)·P(c)
        ev_stack[:, j] = batch.log_evidence + class_log_prior.get(name, 0.0)

    # Fleet combine: softmax over (evidence + class prior), winner per pixel.
    ev_stack = np.where(np.isfinite(ev_stack), ev_stack, -np.inf)
    mx = ev_stack.max(axis=1, keepdims=True)
    w = np.exp(ev_stack - mx)
    probs = w / w.sum(axis=1, keepdims=True)     # (N, n_emu)
    winner_idx = np.argmax(probs, axis=1)        # (N,)

    from biosnicar.sea_ice.quality_flags import compute_quality_flags

    records = []
    for i in range(N):
        j = int(winner_idx[i])
        name = names[j]
        pe = per_emu[name]
        setup = pe["setup"]
        pnames = setup["params"]
        pvals = {p: float(pe["lin"][i, k]) for k, p in enumerate(pnames)}
        punc = {p: float(pe["unc"][i, k]) for k, p in enumerate(pnames)}
        akd = {p: float(pe["batch"].parameter_dfs[i, k])
               for k, p in enumerate(pnames)}
        # Derived physical parameters (match retrieve_sea_ice).
        if "tau_snow" in pvals:
            grain = pvals.get("snow_grain_radius",
                              shared_fixed.get("snow_grain_radius"))
            if grain is not None:
                pvals["snow_depth"] = float(
                    np.clip(pvals["tau_snow"] * grain * 1e-6, 0.003, 1.5))
        if "ice_thickness_cm" in pvals:
            pvals["ice_thickness"] = pvals["ice_thickness_cm"] / 100.0

        class_probs = {names[k]: float(probs[i, k]) for k in range(len(names))}
        conf = float(probs[i, j])
        cost = float(pe["batch"].cost[i])
        flags = compute_quality_flags(
            cost=cost, rms_residual=float("nan"), confidence=min(conf, 1.0),
            converged=bool(pe["batch"].converged[i]),
            parameters={p: pe["lin"][i, k] for k, p in enumerate(pnames)},
            bounds=emulators[name].bounds, surface_type=name,
            cost_per_type={n2: 0.0 for n2 in names},
        )
        records.append({
            "surface_type": name,
            "confidence": conf,
            "cost": cost,
            "quality_flags": int(flags),
            "parameters": pvals,
            "uncertainty": punc,
            "dfs": float(pe["batch"].dfs[i]),
            "class_probabilities": class_probs,
            "prior_resolved": None,
            "spectrum_only_surface_type": None,
            "averaging_kernel_diag": akd,
        })
    return records
