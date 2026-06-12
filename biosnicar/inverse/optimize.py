"""Retrieval dispatcher for spectral and band-mode inversion.

Provides :func:`retrieve`, the main entry point for estimating ice
physical properties from observed albedo using an emulator or direct
forward model.

Supported optimisation methods:

* ``"L-BFGS-B"`` — fast quasi-Newton (default)
* ``"Nelder-Mead"`` — derivative-free simplex
* ``"differential_evolution"`` — global search
* ``"mcmc"`` — Bayesian posterior sampling via emcee
"""

import warnings

import numpy as np
from scipy.optimize import minimize, differential_evolution

from biosnicar.inverse.cost import spectral_cost, band_cost
from biosnicar.inverse.result import RetrievalResult


# Parameters that must be binary (0 or 1) — cannot be continuously optimised.
# Pass these via ``fixed_params`` instead.
_BINARY_PARAMS = {"direct"}

# Parameters optimised in log10(x + 1) space for better conditioning.
# Impurity concentrations span orders of magnitude (0–500 000); log-space
# makes the cost surface much better conditioned and prevents the optimizer
# from getting stuck at high concentrations.
_LOG_SPACE_PARAMS = {
    "black_carbon", "snow_algae", "glacier_algae", "dust", "ssa",
    # Sea ice: bubble radius and depth parameters span orders of magnitude;
    # log-space conditioning substantially improves optimiser convergence.
    "sea_ice_bubble_radius", "pond_depth", "snow_depth", "snow_grain_radius",
    "ssl_grain_radius", "tau_snow",
    # Young ice: NIR transmittance is exponential in thickness (cm scale
    # so the log10(x+1) transform has effect).
    "ice_thickness_cm",
}


def _to_log(x):
    """Transform linear value(s) to log10(x + 1) space."""
    return np.log10(np.asarray(x, dtype=float) + 1.0)


def _from_log(x):
    """Transform log10(x + 1) space back to linear."""
    return 10.0 ** np.asarray(x, dtype=float) - 1.0


# Default parameter bounds for glacier ice retrieval
DEFAULT_BOUNDS = {
    # Glacier / terrestrial ice
    "rds": (100.0, 5000.0),
    "rho": (100.0, 917.0),
    "solzen": (20.0, 80.0),
    "direct": (0, 1),
    "black_carbon": (0.0, 100000.0),
    "snow_algae": (0.0, 500000.0),
    "glacier_algae": (0.0, 100000.0),
    "dust": (0.0, 500000.0),
    "ssa": (0.01, 300.0),
    # Sea ice physical parameters
    "brine_volume_fraction": (0.005, 0.15),   # FYI; use (0.005, 0.05) for MYI
    "sea_ice_temperature":   (-30.0, -2.0),   # legacy — prefer brine_volume_fraction
    "sea_ice_bubble_radius": (50.0,  2000.0),
    "sea_ice_salinity":      (0.0,   20.0),   # legacy — prefer brine_volume_fraction
    "rho_DL":                (820.0, 900.0),
    # Surface structural parameters
    "snow_depth":            (0.02,  0.30),
    "tau_snow":              (50.0,  3000.0),   # snow depth / grain radius
    "snow_grain_radius":     (50.0,  2000.0),
    "ssl_grain_radius":      (500.0, 5000.0),
    "pond_depth":            (0.02,  0.60),
    # Young ice (layer_type=6)
    "ice_thickness_cm":      (0.5,   30.0),
    "ocean_albedo":          (0.03,  0.08),
}



def retrieve(
    observed,
    parameters,
    emulator=None,
    forward_fn=None,
    platform=None,
    observed_band_names=None,
    obs_uncertainty=None,
    bounds=None,
    x0=None,
    regularization=None,
    wavelength_mask=None,
    method="L-BFGS-B",
    mcmc_walkers=32,
    mcmc_steps=2000,
    mcmc_burn=500,
    fixed_params=None,
    ssa_rho=None,
):
    """Retrieve ice physical properties from observed albedo.

    Supports two observation modes:

    * **Spectral mode** (default): *observed* is a 480-element array of
      spectral albedo.  Compared directly to the emulator/forward-model
      prediction.
    * **Band mode**: *observed* is an N-element array of satellite band
      albedo values.  Set *platform* and *observed_band_names* to
      activate this mode.

    And two forward-model modes:

    * **Emulator mode** (recommended): pass an :class:`Emulator` instance.
      ~microsecond evaluations enable fast optimisation and MCMC.
    * **Direct mode**: pass a callable *forward_fn* that returns a
      480-element albedo array.  Slower but exact.

    Parameters
    ----------
    observed : np.ndarray
        Observed albedo — 480-band spectral array or N-band satellite array.
    parameters : list of str
        Names of parameters to retrieve (e.g.
        ``["rds", "black_carbon"]``).
    emulator : Emulator, optional
        Trained emulator for fast forward evaluation.
    forward_fn : callable, optional
        Direct forward function: ``forward_fn(**params) -> np.ndarray (480,)``.
        Either *emulator* or *forward_fn* must be provided.
    platform : str, optional
        Satellite platform key (e.g. ``"sentinel2"``).  Activates band mode.
    observed_band_names : list of str, optional
        Band names matching *observed* elements (e.g.
        ``["B3", "B4", "B11"]``).  Required when *platform* is set.
    obs_uncertainty : np.ndarray, optional
        Per-element 1-sigma measurement uncertainty.
    bounds : dict, optional
        ``{param_name: (lo, hi)}``.  Overrides :data:`DEFAULT_BOUNDS`.
    x0 : dict, optional
        ``{param_name: value}``.  Default: midpoint of the active bounds
        (log-space midpoint for log-conditioned parameters).
    regularization : dict, optional
        ``{param_name: (prior_mean, prior_sigma)}`` for Gaussian priors.
    wavelength_mask : np.ndarray of bool, optional
        Wavelength mask for spectral mode (True = include).
    method : str
        ``"L-BFGS-B"`` (default), ``"Nelder-Mead"``,
        ``"differential_evolution"``, or ``"mcmc"``.
    mcmc_walkers : int
        Number of MCMC walkers (only for ``method="mcmc"``).
    mcmc_steps : int
        Number of MCMC steps.
    mcmc_burn : int
        Number of burn-in steps to discard.
    fixed_params : dict, optional
        ``{param_name: value}`` for parameters that are known and should
        not be optimised.  The parameter must exist in the emulator's
        input space.
    ssa_rho : float, optional
        Reference density (kg m-3) used to decompose SSA into (rds, rho)
        when ``"ssa"`` is in *parameters*.  Falls back to
        ``fixed_params["rho"]``, then the midpoint of the emulator's rho
        training range, then 500.0 kg m-3.

    Returns
    -------
    RetrievalResult
    """
    observed = np.asarray(observed, dtype=float)

    # --- Validate inputs ---
    if emulator is None and forward_fn is None:
        raise ValueError("Provide either `emulator` or `forward_fn`.")
    if platform is not None and observed_band_names is None:
        raise ValueError("`observed_band_names` is required when `platform` is set.")
    binary_in_params = _BINARY_PARAMS.intersection(parameters)
    if binary_in_params:
        raise ValueError(
            "Binary parameters {} cannot be continuously optimised. "
            "Pass them via `fixed_params` instead.".format(binary_in_params)
        )
    if "dust" in parameters:
        warnings.warn(
            "Mineral dust has very low spectral sensitivity at typical "
            "environmental concentrations (<5000 ppb).  The retrieval will "
            "likely be unreliable — consider fixing dust via `fixed_params` "
            "unless you expect very high concentrations (>10 000 ppb).",
            stacklevel=2,
        )

    # --- SSA validation ---
    use_ssa = "ssa" in parameters
    if use_ssa:
        if "rds" in parameters or "rho" in parameters:
            raise ValueError(
                "Cannot retrieve 'ssa' alongside 'rds' or 'rho'. "
                "SSA replaces both — the emulator decomposes SSA into "
                "(rds, rho) internally using a reference density."
            )
        # Determine reference density for SSA → (rds, rho) decomposition
        _ref_rho = ssa_rho
        if _ref_rho is None and fixed_params and "rho" in fixed_params:
            _ref_rho = float(fixed_params["rho"])
        if _ref_rho is None and emulator is not None:
            emu_b = emulator.bounds
            if "rho" in emu_b:
                _ref_rho = float(np.mean(emu_b["rho"]))
        if _ref_rho is None:
            _ref_rho = 500.0

    # --- Build forward function from emulator ---
    if emulator is not None and forward_fn is None:
        # Copy: bounds is filled in from the emulator below, and mutating the
        # caller's dict would leak one emulator's training bounds into the
        # next fleet member in retrieve_sea_ice().
        bounds = dict(bounds) if bounds else {}
        if use_ssa:
            forward_fn = _make_ssa_emulator_fn(
                emulator, parameters, fixed_params, _ref_rho
            )
            # Auto-compute SSA bounds using ref_rho and emulator rds range.
            # This ensures the internal rds decomposition stays within the
            # emulator's training bounds.
            if "ssa" not in bounds:
                from biosnicar.inverse.result import _compute_ssa
                emu_b = emulator.bounds
                rds_lo, rds_hi = emu_b.get("rds", (100.0, 5000.0))
                ssa_hi = _compute_ssa(rds_lo, _ref_rho)
                ssa_lo = max(_compute_ssa(rds_hi, _ref_rho), 0.001)
                bounds["ssa"] = (ssa_lo, ssa_hi)
        else:
            forward_fn = _make_emulator_fn(emulator, parameters, fixed_params)
        # Fill bounds from emulator where not overridden
        emu_bounds = emulator.bounds
        for p in parameters:
            if p not in bounds:
                if p in emu_bounds:
                    bounds[p] = emu_bounds[p]
                elif p in DEFAULT_BOUNDS:
                    bounds[p] = DEFAULT_BOUNDS[p]

    # --- Resolve bounds and x0 ---
    if bounds is None:
        bounds = {}
    active_bounds = []
    for p in parameters:
        if p in bounds:
            active_bounds.append(bounds[p])
        elif p in DEFAULT_BOUNDS:
            active_bounds.append(DEFAULT_BOUNDS[p])
        else:
            raise ValueError(
                "No bounds for parameter {!r}. "
                "Provide via `bounds` argument.".format(p)
            )

    # Default x0 to midpoint of active bounds.
    x0_dict = {p: float(np.mean(active_bounds[i]))
               for i, p in enumerate(parameters)}
    # Override with user-provided x0 values
    if x0 is not None:
        x0_dict.update(x0)
    x0_vec = np.array([x0_dict[p] for p in parameters])

    # --- Build cost function ---
    flx_slr = getattr(emulator, "flx_slr", None) if emulator else None

    if platform is not None:
        # Band mode
        if flx_slr is None:
            raise ValueError(
                "Band mode requires `flx_slr`.  Use an emulator or "
                "provide a forward_fn that doesn't need band convolution."
            )

        def cost_fn(params):
            return band_cost(
                params, parameters, observed, observed_band_names,
                forward_fn, flx_slr, platform,
                obs_uncertainty=obs_uncertainty,
                regularization=regularization,
            )
    else:
        # Spectral mode
        def cost_fn(params):
            return spectral_cost(
                params, parameters, observed, forward_fn,
                obs_uncertainty=obs_uncertainty,
                wavelength_mask=wavelength_mask,
                regularization=regularization,
            )

    # --- Log-space transformation for impurity parameters ---
    log_mask = np.array([p in _LOG_SPACE_PARAMS for p in parameters])
    use_log = np.any(log_mask)

    if use_log:
        # Transform bounds and x0 to log10(x+1) space
        opt_bounds = []
        for i, (lo, hi) in enumerate(active_bounds):
            if log_mask[i]:
                opt_bounds.append((float(_to_log(lo)), float(_to_log(hi))))
            else:
                opt_bounds.append((lo, hi))

        # Compute x0 in log space as midpoint of log bounds (not
        # transform of linear midpoint, which would be near the upper bound).
        opt_x0 = x0_vec.copy()
        for i in range(len(parameters)):
            if log_mask[i]:
                opt_x0[i] = float(np.mean(opt_bounds[i]))
        # Override with user-provided x0 (transformed to log)
        if x0 is not None:
            for p, val in x0.items():
                idx = parameters.index(p) if p in parameters else -1
                if idx >= 0 and log_mask[idx]:
                    opt_x0[idx] = float(_to_log(val))

        # Wrap cost function: optimizer works in log space, cost needs linear
        _orig_cost = cost_fn

        def opt_cost_fn(params):
            p = params.copy()
            p[log_mask] = _from_log(params[log_mask])
            return _orig_cost(p)

        # Wrap forward function similarly
        _orig_fwd = forward_fn

        def opt_forward_fn(**kw):
            linear_kw = {}
            for name, val in kw.items():
                if name in _LOG_SPACE_PARAMS:
                    linear_kw[name] = float(_from_log(val))
                else:
                    linear_kw[name] = val
            return _orig_fwd(**linear_kw)
    else:
        opt_bounds = active_bounds
        opt_x0 = x0_vec
        opt_cost_fn = cost_fn
        opt_forward_fn = forward_fn

    # --- Dispatch to optimiser ---
    if method == "mcmc":
        result = _run_mcmc(
            opt_cost_fn, parameters, opt_bounds, opt_x0,
            opt_forward_fn, observed, method,
            mcmc_walkers, mcmc_steps, mcmc_burn,
        )
    elif method == "differential_evolution":
        result = _run_differential_evolution(
            opt_cost_fn, parameters, opt_bounds, opt_forward_fn, observed, method,
        )
    else:
        result = _run_scipy_minimize(
            opt_cost_fn, parameters, opt_bounds, opt_x0,
            opt_forward_fn, observed, method,
        )

    # --- Attach solar flux for to_outputs() / to_platform() ---
    result.flx_slr = flx_slr

    # --- Transform results back from log space ---
    if use_log:
        for p in parameters:
            if p in _LOG_SPACE_PARAMS:
                log_val = result.best_fit[p]
                linear_val = float(_from_log(log_val))
                result.best_fit[p] = linear_val
                # sigma_linear ≈ (x + 1) * ln(10) * sigma_log10
                if p in result.uncertainty:
                    result.uncertainty[p] = float(
                        (linear_val + 1.0) * np.log(10) * result.uncertainty[p]
                    )
        if result.chains is not None:
            for i, p in enumerate(parameters):
                if p in _LOG_SPACE_PARAMS:
                    result.chains[:, :, i] = _from_log(result.chains[:, :, i])
        # Re-predict with correct linear values
        result.predicted_albedo = forward_fn(**result.best_fit)

    # --- Populate derived quantities for SSA mode ---
    if use_ssa and "ssa" in result.best_fit:
        ssa_val = result.best_fit["ssa"]
        if ssa_val > 0:
            phi = 1.0 - _ref_rho / 917.0
            rds_internal = 3.0 * phi / (ssa_val * _ref_rho) / 1e-6
            result.derived = {
                "rds_internal": float(rds_internal),
                "rho_ref": float(_ref_rho),
            }

    return result


def _make_emulator_fn(emulator, parameters, fixed_params):
    """Build a forward function that fills in fixed params for the emulator."""
    fixed = dict(fixed_params) if fixed_params else {}

    def fn(**active_params):
        full = dict(fixed)
        full.update(active_params)
        return emulator.predict(**full)

    return fn


_RHO_ICE = 917.0


def _make_ssa_emulator_fn(emulator, parameters, fixed_params, ref_rho):
    """Forward function that converts SSA to (rds, rho) internally."""
    fixed = dict(fixed_params) if fixed_params else {}

    def fn(**active_params):
        full = dict(fixed)
        ssa_val = active_params.get("ssa")
        if ssa_val is not None:
            phi = 1.0 - ref_rho / _RHO_ICE
            rds_um = 3.0 * phi / (ssa_val * ref_rho) / 1e-6
            full["rds"] = float(rds_um)
            full["rho"] = float(ref_rho)
            full.update((k, v) for k, v in active_params.items() if k != "ssa")
        else:
            full.update(active_params)
        return emulator.predict(**full)

    return fn


def _run_scipy_minimize(cost_fn, parameters, active_bounds, x0_vec,
                        forward_fn, observed, method):
    """Run scipy.optimize.minimize and return RetrievalResult.

    For L-BFGS-B with 2+ parameters, a quick differential-evolution
    pre-search is used to escape local minima before polishing.
    """
    options = {"maxiter": 500}
    if method == "L-BFGS-B":
        options["ftol"] = 1e-9
    elif method == "Nelder-Mead":
        options["fatol"] = 1e-9
        options["xatol"] = 1e-8

    n_params = len(parameters)
    total_nfev = 0
    lo = np.array([b[0] for b in active_bounds])
    hi = np.array([b[1] for b in active_bounds])

    # Enforce bounds for methods that don't support them natively.  The
    # Hessian below uses the unwrapped cost so finite-difference steps that
    # cross a bound don't see the 1e20 penalty wall.
    hessian_cost_fn = cost_fn
    if method == "Nelder-Mead":
        _inner = cost_fn

        def cost_fn(params):
            if np.any(params < lo) or np.any(params > hi):
                return 1e20
            return _inner(params)

    # For L-BFGS-B, seed with a lightweight DE pass to escape local minima.
    # maxiter=20/popsize=5 gives ~300 evals vs ~4000 for the old 100/10 settings
    # while producing identical classifications on all tested surface types.
    if method == "L-BFGS-B" and n_params >= 2:
        de_result = differential_evolution(
            cost_fn,
            bounds=active_bounds,
            maxiter=20,
            popsize=5,
            tol=1e-8,
            seed=42,
            polish=False,
        )
        total_nfev += int(de_result.nfev)
        # Use DE result if it's better than the provided x0
        if de_result.fun < cost_fn(x0_vec):
            x0_vec = de_result.x

    result = minimize(
        cost_fn,
        x0_vec,
        method=method,
        bounds=active_bounds if method != "Nelder-Mead" else None,
        options=options,
    )
    total_nfev += int(result.nfev)

    best_fit = dict(zip(parameters, result.x))
    predicted = forward_fn(**best_fit)

    # Hessian uncertainty
    uncertainty = _hessian_uncertainty(
        hessian_cost_fn, result.x, parameters, active_bounds
    )

    return RetrievalResult(
        best_fit=best_fit,
        cost=float(result.fun),
        uncertainty=uncertainty,
        predicted_albedo=predicted,
        observed=observed,
        converged=bool(result.success),
        method=method,
        n_function_evals=total_nfev,
    )


def _run_differential_evolution(cost_fn, parameters, active_bounds,
                                forward_fn, observed, method):
    """Run differential_evolution and return RetrievalResult."""
    result = differential_evolution(
        cost_fn,
        bounds=active_bounds,
        maxiter=1000,
        tol=1e-10,
        seed=42,
    )

    best_fit = dict(zip(parameters, result.x))
    predicted = forward_fn(**best_fit)

    uncertainty = _hessian_uncertainty(
        cost_fn, result.x, parameters, active_bounds
    )

    return RetrievalResult(
        best_fit=best_fit,
        cost=float(result.fun),
        uncertainty=uncertainty,
        predicted_albedo=predicted,
        observed=observed,
        converged=bool(result.success),
        method=method,
        n_function_evals=int(result.nfev),
    )


def _run_mcmc(cost_fn, parameters, active_bounds, x0_vec,
              forward_fn, observed, method,
              n_walkers, n_steps, n_burn):
    """Run emcee MCMC sampler and return RetrievalResult."""
    try:
        import emcee
    except ImportError:
        raise ImportError(
            "emcee is required for MCMC.  Install with:  pip install emcee>=3"
        )

    n_params = len(parameters)
    lo = np.array([b[0] for b in active_bounds])
    hi = np.array([b[1] for b in active_bounds])

    def log_prob(params):
        # Uniform prior: -inf if outside bounds
        if np.any(params < lo) or np.any(params > hi):
            return -np.inf
        return -0.5 * cost_fn(params)

    # Initialise walkers as Gaussian ball around x0
    spread = (hi - lo) * 0.01
    p0 = x0_vec + spread * np.random.randn(n_walkers, n_params)
    # Clip to bounds
    p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)

    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_prob)
    sampler.run_mcmc(p0, n_steps + n_burn, progress=False)

    chains = sampler.get_chain(discard=n_burn)  # (n_steps, n_walkers, n_params)
    flat = chains.reshape(-1, n_params)

    best_fit = dict(zip(parameters, np.median(flat, axis=0)))
    uncertainty = dict(zip(parameters, np.std(flat, axis=0)))
    predicted = forward_fn(**best_fit)

    # Acceptance fraction
    acc = float(np.mean(sampler.acceptance_fraction))

    # Autocorrelation time (may fail if chains are too short)
    try:
        autocorr = sampler.get_autocorr_time(quiet=True)
    except Exception:
        autocorr = np.full(n_params, np.nan)

    cost_at_median = cost_fn(np.array([best_fit[p] for p in parameters]))

    return RetrievalResult(
        best_fit=best_fit,
        cost=float(cost_at_median),
        uncertainty=uncertainty,
        predicted_albedo=predicted,
        observed=observed,
        converged=acc > 0.1,  # crude convergence check
        method=method,
        n_function_evals=n_walkers * (n_steps + n_burn),
        chains=chains,
        acceptance_fraction=acc,
        autocorr_time=autocorr,
    )


def _hessian_uncertainty(cost_fn, x_opt, parameters, active_bounds):
    """Compute parameter uncertainties from finite-difference Hessian.

    Returns ``{param_name: 1_sigma}`` from the diagonal of the
    inverse Hessian at the optimum.  Parameters whose Hessian row
    is singular get ``inf`` uncertainty.
    """
    n = len(x_opt)
    # Step sizes: 1e-4 of parameter range
    steps = np.array([(b[1] - b[0]) * 1e-4 for b in active_bounds])
    steps = np.maximum(steps, 1e-10)

    hessian = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            x_pp = x_opt.copy()
            x_pm = x_opt.copy()
            x_mp = x_opt.copy()
            x_mm = x_opt.copy()

            x_pp[i] += steps[i]
            x_pp[j] += steps[j]

            x_pm[i] += steps[i]
            x_pm[j] -= steps[j]

            x_mp[i] -= steps[i]
            x_mp[j] += steps[j]

            x_mm[i] -= steps[i]
            x_mm[j] -= steps[j]

            hessian[i, j] = (
                cost_fn(x_pp) - cost_fn(x_pm) - cost_fn(x_mp) + cost_fn(x_mm)
            ) / (4.0 * steps[i] * steps[j])
            hessian[j, i] = hessian[i, j]

    # Invert to get covariance; fall back to inf for singular Hessian
    try:
        cov = np.linalg.inv(hessian)
        sigmas = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        sigmas = np.full(n, float("inf"))

    return dict(zip(parameters, sigmas))
