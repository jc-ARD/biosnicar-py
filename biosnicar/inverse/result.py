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


def _ssa_uncertainty(rds, rho, sigma_rds, sigma_rho):
    """Propagate 1-sigma uncertainties in rds and rho to SSA.

    Uses first-order error propagation:

        sigma_SSA^2 = (dSSA/drds)^2 * sigma_rds^2
                    + (dSSA/drho)^2 * sigma_rho^2

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
    return float(np.sqrt(var))


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

    @property
    def ssa(self):
        """Specific surface area (m2 kg-1) derived from rds and rho.

        For bubbly ice: SSA = 3 * (1 - rho/917) / (rds_m * rho).
        Returns None if rds or rho was not retrieved (or fixed).

        SSA is the quantity the inversion actually constrains —
        different (rds, rho) pairs with the same SSA produce nearly
        identical spectra.
        """
        rds = self.best_fit.get("rds")
        rho = self.best_fit.get("rho")
        if rds is None or rho is None:
            return None
        return _compute_ssa(rds, rho)

    @property
    def ssa_uncertainty(self):
        """1-sigma uncertainty on SSA (m2 kg-1), propagated from rds and rho.

        Returns None if either parameter was not retrieved.
        """
        rds = self.best_fit.get("rds")
        rho = self.best_fit.get("rho")
        if rds is None or rho is None:
            return None
        sigma_rds = self.uncertainty.get("rds", 0.0)
        sigma_rho = self.uncertainty.get("rho", 0.0)
        return _ssa_uncertainty(rds, rho, sigma_rds, sigma_rho)

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
        for name, val in self.best_fit.items():
            unc = self.uncertainty.get(name, float("nan"))
            lines.append(f"    {name:25s} = {val:12.4f}  ±  {unc:.4f}")
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
