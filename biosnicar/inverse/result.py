"""RetrievalResult dataclass for inversion outputs.

Holds the best-fit parameters, uncertainties, predicted spectrum, convergence
diagnostics, and (optionally) MCMC chains from a ``retrieve()`` call.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# Pure-ice density (kg m-3)
_RHO_ICE = 917.0


def _compute_ssa(rds_um, rho):
    """Compute specific surface area for bubbly ice.

    For spherical air bubbles of effective radius *rds* (um) embedded in
    ice of bulk density *rho* (kg m-3):

        SSA = 3 * phi / (r * rho)

    where phi = 1 - rho/917 is the air volume fraction and r is the
    bubble radius in metres.  Units: m2 kg-1.

    This is the quantity that the spectral inversion actually constrains:
    the scattering optical depth per unit mass scales with SSA in the
    geometric-optics limit.  Different (rds, rho) pairs with the same SSA
    produce nearly identical spectra.
    """
    phi = 1.0 - rho / _RHO_ICE
    r_m = rds_um * 1e-6
    if r_m <= 0 or rho <= 0:
        return float("nan")
    return 3.0 * phi / (r_m * rho)


def _ssa_uncertainty(rds, rho, sigma_rds, sigma_rho, cov_rds_rho=0.0):
    """Propagate 1-sigma uncertainties in rds and rho to SSA.

    Uses first-order error propagation:

        sigma_SSA^2 = (dSSA/drds)^2 * sigma_rds^2
                    + (dSSA/drho)^2 * sigma_rho^2
                    + 2 (dSSA/drds)(dSSA/drho) cov(rds, rho)

    The covariance term is not optional here.  It is supplied from the OE
    posterior when available (``method="oe"``); omitting it assumes the two
    are independent, which they emphatically are not.

    Partial derivatives of SSA = 3(1 - rho/917) / (r * rho):

        dSSA/drds = -SSA / rds          [m2 kg-1 per um]
        dSSA/drho = -3 / (r * rho^2)    [m2 kg-1 per kg m-3]

    The rho derivative follows from rewriting SSA = 3/(r) * (1/rho - 1/917),
    so d/drho = 3/r * (-1/rho^2).
    """
    r_m = rds * 1e-6
    if r_m <= 0 or rho <= 0:
        return float("inf")
    phi = 1.0 - rho / _RHO_ICE
    ssa_val = 3.0 * phi / (r_m * rho)

    # dSSA/d(rds_um) = dSSA/dr * dr/d(rds_um) = (-SSA / r) * 1e-6
    dSSA_drds = -ssa_val / rds  # in m2 kg-1 per um

    # dSSA/drho = -3 / (r * rho^2)
    dSSA_drho = -3.0 / (r_m * rho ** 2)

    var = (dSSA_drds * sigma_rds) ** 2 + (dSSA_drho * sigma_rho) ** 2
    if cov_rds_rho:
        # Radius and density are strongly anti-correlated along the scattering
        # ridge (posterior r ~ -0.99 for FYI_bare).  Both partials are
        # negative, so with cov < 0 this term SUBTRACTS: SSA is far better
        # determined than either component, which is the whole reason to
        # report it.  Dropping the term overstates sigma_SSA several-fold.
        var += 2.0 * dSSA_drds * dSSA_drho * cov_rds_rho
    return float(np.sqrt(max(var, 0.0)))


@dataclass
class RetrievalResult:
    """Result container returned by :func:`~biosnicar.inverse.optimize.retrieve`.

    Attributes
    ----------
    best_fit : dict
        ``{param_name: best_value}`` at the cost-function minimum.
    cost : float
        Final cost-function value (chi-squared).
    uncertainty : dict
        ``{param_name: 1_sigma}`` from Hessian or MCMC posterior.
    predicted_albedo : np.ndarray
        480-band spectral albedo at the best-fit point.
    observed : np.ndarray
        Input observations (480-band array or N-band satellite array).
    converged : bool
        Whether the optimiser reported convergence.
    method : str
        Optimisation method used (e.g. ``"L-BFGS-B"``, ``"mcmc"``).
    n_function_evals : int
        Number of forward-model (or emulator) evaluations.
    flx_slr : np.ndarray or None
        480-band spectral solar irradiance used for flux-weighted BBA
        and platform band convolution.  Populated when an emulator was
        used; ``None`` when only a ``forward_fn`` was provided.
    chains : np.ndarray or None
        MCMC chains of shape ``(n_steps, n_walkers, n_params)``.
        Only populated when ``method="mcmc"``.
    acceptance_fraction : float or None
        Mean MCMC acceptance fraction.  Only for ``method="mcmc"``.
    autocorr_time : np.ndarray or None
        Integrated autocorrelation time per parameter.  Only for ``method="mcmc"``.
    """

    best_fit: Dict[str, float]
    cost: float
    uncertainty: Dict[str, float]
    predicted_albedo: np.ndarray
    observed: np.ndarray
    converged: bool
    method: str
    n_function_evals: int

    # Auxiliary derived quantities (e.g. internal rds/rho decomposition in SSA mode)
    derived: Dict[str, float] = field(default_factory=dict)

    # Solar flux spectrum — populated from emulator.flx_slr when available
    flx_slr: Optional[np.ndarray] = None

    # MCMC-specific fields (None unless method="mcmc")
    chains: Optional[np.ndarray] = None
    acceptance_fraction: Optional[float] = None
    autocorr_time: Optional[np.ndarray] = None

    # Optimal-estimation fields (None unless method="oe").
    # posterior_covariance: {(p_i, p_j): cov} over retrieved params (linear space).
    # averaging_kernel_diag / dfs: information content — how much each parameter
    #   (and the retrieval overall) came from the measurement vs the prior.
    # log_evidence: Laplace ln p(y|model), used for surface-type classification.
    posterior_covariance: Optional[Dict] = None
    averaging_kernel_diag: Optional[Dict[str, float]] = None
    dfs: Optional[float] = None
    log_evidence: Optional[float] = None

    def parameter_correlation(self) -> Dict:
        """Posterior correlation matrix as ``{(p_i, p_j): r}`` (``method="oe"``).

        Empty unless the OE engine populated ``posterior_covariance``.

        A correlation near +/-1 means the two parameters trade off against each
        other: the measurement constrains a combination of them better than
        either alone.  The marginal 1-sigma on each is inflated over its
        value-if-the-other-were-known by ``1 / sqrt(1 - r**2)`` — so r = 0.5
        costs 15%, r = 0.9 costs 2.3x, r = 0.99 costs 7x.

        For sea-ice retrievals this lives on the per-type fits, e.g.
        ``result.all_fits["FYI_bare"].parameter_correlation()``.
        """
        pc = self.posterior_covariance
        if not pc:
            return {}
        names = [p for p in self.best_fit if (p, p) in pc]
        sig = {p: float(np.sqrt(max(pc[(p, p)], 0.0))) for p in names}
        out = {}
        for a in names:
            for b in names:
                d = sig[a] * sig[b]
                out[(a, b)] = float(pc[(a, b)] / d) if d > 0 else 0.0
        return out

    def degenerate_parameters(self, threshold: float = 0.8):
        """Parameter pairs whose posterior correlation exceeds *threshold*.

        Returns ``[(p_i, p_j, r), ...]`` sorted by ``|r|`` descending, each
        pair listed once.  Empty unless ``method="oe"``.

        Use this instead of assuming a given pair is degenerate: collinearity
        in sea ice is strongly state-dependent, and for the FYI_bare
        ``rho_DL``/``brine_volume_fraction`` pair the correlation both varies
        in magnitude and *changes sign* across the parameter space.
        """
        corr = self.parameter_correlation()
        seen, out = set(), []
        for (a, b), r in corr.items():
            if a == b or (b, a) in seen:
                continue
            seen.add((a, b))
            if abs(r) >= threshold:
                out.append((a, b, r))
        return sorted(out, key=lambda t: -abs(t[2]))

    def constrained_combinations(self):
        """Principal axes of the posterior — what the measurement actually
        constrained, when it is a combination rather than a parameter.

        Returns a list of ``{"weights": {param: w}, "variance": lam,
        "sigma": sqrt(lam)}``, sorted from best- to worst-determined.  Empty
        unless ``method="oe"``.

        Each entry is an eigenvector of the posterior *correlation* matrix,
        i.e. a unit-length combination of the parameters measured in units
        of their own marginal 1-sigma::

            c = sum_i  w_i * (x_i - x_hat_i) / sigma_i

        and ``sigma`` is the posterior standard deviation of ``c``.  If the
        parameters were independently determined every ``sigma`` would be 1.
        A value well below 1 is a combination the spectrum pins down far
        better than the marginal error bars suggest; a value above 1 is the
        direction the retrieval is free to slide along.  For a single
        degenerate pair with correlation ``r`` the two values are
        ``sqrt(1 - |r|)`` and ``sqrt(1 + |r|)`` — at r = -0.98 the
        constrained combination is known 7x better than either component.

        This is the general form of what :attr:`ssa` does for the bubble
        radius / density pair, and it needs no retraining: rotating an
        emulator's inputs onto these axes would re-express the same
        posterior, not add information to it.  The axes are local — the
        posterior is linearised at the solution — so they describe this
        retrieval, not the emulator.
        """
        corr = self.parameter_correlation()
        if not corr:
            return []
        names = [p for p in self.best_fit if (p, p) in corr]
        C = np.array([[corr[(a, b)] for b in names] for a in names])
        lam, vec = np.linalg.eigh(0.5 * (C + C.T))
        out = []
        for k in np.argsort(lam):
            w = vec[:, k]
            # Fix the arbitrary eigenvector sign: largest weight positive.
            if w[np.argmax(np.abs(w))] < 0:
                w = -w
            v = float(max(lam[k], 0.0))
            out.append({
                "weights": {p: float(w[i]) for i, p in enumerate(names)},
                "variance": v,
                "sigma": float(np.sqrt(v)),
            })
        return out

    def _degeneracy_lines(self, threshold: float = 0.8):
        """Summary lines naming degenerate pairs; empty when there are none."""
        pairs = self.degenerate_parameters(threshold)
        if not pairs:
            return []
        lines = [
            f"  Degenerate pairs (posterior |r| >= {threshold:g}) — the "
            "spectrum fixes a combination,",
            "  not the components; each marginal sigma is inflated as shown:",
        ]
        for a, b, r in pairs:
            infl = 1.0 / np.sqrt(max(1.0 - r * r, 1e-12))
            lines.append(f"    {a} / {b}  r = {r:+.2f}  ({infl:.1f}x)")
        return lines

    # ── Information content: measurement or prior? ──────────────────────

    def information_content(self) -> Dict[str, float]:
        """``{param: averaging-kernel diagonal}`` — 1 = measured, 0 = prior.

        Empty unless ``method="oe"``.  This is the diagnostic that
        ``uncertainty`` cannot give you: a small sigma means either the
        spectrum pinned the value or the prior was narrow and the spectrum
        said nothing, and only the averaging kernel tells the two apart.

        See :mod:`biosnicar.inverse.information` for the threshold
        convention shared across the package.
        """
        return dict(self.averaging_kernel_diag or {})

    def parameter_provenance(self, **kw) -> Dict[str, str]:
        """``{param: "measured" | "prior-influenced" | "not retrieved"}``.

        Parameters with no averaging kernel (non-OE methods, or derived
        quantities) are labelled ``"unknown"`` rather than dropped.
        """
        from biosnicar.inverse import information as _info
        return _info.provenance(self.averaging_kernel_diag, self.best_fit, **kw)

    def prior_dominated_parameters(self, threshold: float = None) -> Dict[str, bool]:
        """``{param: True}`` where the prior, not the spectrum, set the value.

        *threshold* defaults to
        :data:`biosnicar.inverse.information.MEASURED_THRESHOLD` (0.5).
        Empty unless ``method="oe"``.  Mirrors the method of the same name
        on :class:`~biosnicar.sea_ice.retrieve.SeaIceRetrievalResult`, which
        shares this threshold.
        """
        from biosnicar.inverse import information as _info
        if threshold is None:
            threshold = _info.MEASURED_THRESHOLD
        return {p: (a < threshold) for p, a in (self.averaging_kernel_diag or {}).items()}

    def measured_parameters(self, threshold: float = None) -> Dict[str, float]:
        """The subset of ``best_fit`` the measurement actually determined.

        Use this when the retrieved values are going somewhere they will be
        read as measurements — a publication table, a downstream fusion
        step — rather than filtering ``best_fit`` by eye.  Returns the whole
        of ``best_fit`` when no averaging kernel is available, since in that
        case nothing has been shown either way.
        """
        from biosnicar.inverse import information as _info
        akd = self.averaging_kernel_diag or {}
        if not akd:
            return dict(self.best_fit)
        if threshold is None:
            threshold = _info.MEASURED_THRESHOLD
        return {p: v for p, v in self.best_fit.items()
                if akd.get(p, float("nan")) >= threshold}

    def _ssa_param_names(self):
        """Which (radius, density) pair in ``best_fit`` defines SSA, if any.

        Glacier ice names them ``rds``/``rho``; the sea-ice emulators name the
        same physical quantities ``sea_ice_bubble_radius``/``rho_DL``.  Before
        this lookup existed, ``ssa`` silently returned None for every sea-ice
        retrieval even though the formula below is derived for bubbly ice.
        """
        for r_name, d_name in (("rds", "rho"),
                               ("sea_ice_bubble_radius", "rho_DL")):
            if self.best_fit.get(r_name) is not None and \
                    self.best_fit.get(d_name) is not None:
                return r_name, d_name
        return None, None

    @property
    def ssa(self):
        """Specific surface area (m2 kg-1) of the air/ice interface.

        For bubbly ice: SSA = 3 * (1 - rho/917) / (rds_m * rho), from either
        ``rds``/``rho`` (glacier ice) or
        ``sea_ice_bubble_radius``/``rho_DL`` (sea ice).  None if the pair was
        not retrieved.

        **Prefer this over the two components separately.**  Radius and
        density are individually degenerate — the measurement constrains
        their combination.  Measured on the forward model along true iso-SSA
        curves (radius swept ~5x, density 820-895 kg/m3), holding SSA fixed
        removes **88% of the NIR spectral variation and 61% of the VIS**.

        The residual matters too, which is why this is offered alongside the
        components rather than replacing them: the single-scattering albedo
        is a function of SSA alone, but the asymmetry parameter depends on
        radius alone, and in the VIS — where ice barely absorbs and omega -> 1
        — that second handle carries much of the signal.
        """
        r_name, d_name = self._ssa_param_names()
        if r_name is None:
            return None
        return _compute_ssa(self.best_fit[r_name], self.best_fit[d_name])

    @property
    def ssa_uncertainty(self):
        """1-sigma uncertainty on SSA (m2 kg-1).

        Uses the OE posterior *covariance* between radius and density when it
        is available (``method="oe"``), not just the two marginal sigmas.
        That matters: the pair is strongly anti-correlated along the
        scattering ridge, and both partials of SSA are negative, so the
        cross-term subtracts.  Assuming independence overstates this several-
        fold and hides the very point of reporting SSA.

        Returns None if the (radius, density) pair was not retrieved.
        """
        r_name, d_name = self._ssa_param_names()
        if r_name is None:
            return None
        sig_r = self.uncertainty.get(r_name, 0.0)
        sig_d = self.uncertainty.get(d_name, 0.0)
        # Rebuild the covariance from the CORRELATION times the linear-space
        # sigmas.  posterior_covariance is stored in optimiser space, and
        # sea_ice_bubble_radius / rds are log-sampled there, so its raw entry
        # is in log10(x+1) units while `uncertainty` is linear -- mixing them
        # silently reduces the cross-term to nothing.  Correlation is
        # invariant under the per-parameter rescaling, so this is exact to
        # first order and needs no Jacobian factor.
        cov = 0.0
        corr = self.parameter_correlation()
        if corr:
            r_rd = corr.get((r_name, d_name), corr.get((d_name, r_name), 0.0))
            cov = float(r_rd) * sig_r * sig_d
        return _ssa_uncertainty(
            self.best_fit[r_name], self.best_fit[d_name],
            sig_r, sig_d, cov_rds_rho=cov,
        )

    def to_outputs(self):
        """Return the predicted spectrum as an :class:`~biosnicar.classes.outputs.Outputs` object.

        Wraps ``predicted_albedo`` and ``flx_slr`` into a full ``Outputs``
        instance, giving access to ``.to_platform()``, ``.plot()``, and all
        other ``Outputs`` methods.

        Returns
        -------
        Outputs
            BBA, BBAVIS, BBANIR, albedo, and flx_slr are populated.
            Subsurface flux fields (F_up, F_dwn, heat_rt) are None.

        Raises
        ------
        ValueError
            If ``predicted_albedo`` is not available.

        Notes
        -----
        ``to_platform()`` on the returned object requires ``flx_slr`` to be
        present (i.e. the retrieval used an emulator rather than a bare
        ``forward_fn``).  If ``flx_slr`` is None a uniform weighting is used
        for BBA computation, and ``to_platform()`` will raise.
        """
        from biosnicar.classes.outputs import Outputs

        if self.predicted_albedo is None:
            raise ValueError("predicted_albedo is not set on this RetrievalResult.")

        _VIS_MAX = 50
        _NIR_MAX = 480

        out = Outputs()
        out.albedo = np.array(self.predicted_albedo, dtype=float)
        out.flx_slr = self.flx_slr

        flx = self.flx_slr if self.flx_slr is not None else np.ones(480)
        out.BBA = float(np.sum(flx * out.albedo) / np.sum(flx))
        out.BBAVIS = float(
            np.sum(flx[:_VIS_MAX] * out.albedo[:_VIS_MAX])
            / np.sum(flx[:_VIS_MAX])
        )
        out.BBANIR = float(
            np.sum(flx[_VIS_MAX:_NIR_MAX] * out.albedo[_VIS_MAX:_NIR_MAX])
            / np.sum(flx[_VIS_MAX:_NIR_MAX])
        )
        return out

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [f"RetrievalResult (method={self.method}, converged={self.converged})"]
        lines.append(f"  Cost: {self.cost:.4f}  |  Function evaluations: {self.n_function_evals}")
        lines.append("  Best-fit parameters:")
        from biosnicar.inverse import information as _info
        lines.extend(_info.format_parameter_lines(
            self.best_fit, self.uncertainty, self.averaging_kernel_diag,
            name_width=25,
        ))
        lines.extend(self._degeneracy_lines())
        if self.ssa is not None:
            lines.append("  Derived quantities:")
            lines.append(
                f"    {'SSA':25s} = {self.ssa:12.4f}  ±  {self.ssa_uncertainty:.4f}  m2/kg"
            )
        if self.chains is not None:
            n_steps, n_walkers, n_params = self.chains.shape
            lines.append(
                f"  MCMC: {n_steps} steps × {n_walkers} walkers, "
                f"acceptance = {self.acceptance_fraction:.3f}"
            )
        return "\n".join(lines)

    def plot(self, true_values=None, save=None, show=False, **kwargs):
        """Plot spectral fit and retrieved parameters.

        See :func:`biosnicar.plotting.plot_retrieval`.

        Parameters
        ----------
        true_values : dict, optional
            ``{param_name: true_value}`` for comparison.
        save : str or Path, optional
            Save figure to this path.
        show : bool
            If True, display in an interactive window.
        **kwargs
            Passed to :func:`~biosnicar.plotting.plot_retrieval`.

        Returns
        -------
        fig, (ax_spec, ax_params) : matplotlib Figure and Axes
        """
        from biosnicar.plotting import plot_retrieval
        return plot_retrieval(self, true_values=true_values,
                              save=save, show=show, **kwargs)
