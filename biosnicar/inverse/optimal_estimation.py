"""Optimal estimation (OE) retrieval engine — Rodgers (2000) framework.

Pure, forward-model-agnostic Bayesian retrieval of a continuous state vector
from a measurement, returning not just a best estimate but its posterior
covariance, averaging kernel, and degrees of freedom for signal (DFS), plus a
Laplace approximation to the model evidence (for surface-type classification).

This module is deliberately self-contained and knows nothing about sea ice,
emulators, or log-space transforms — those belong to the integration layer
(:mod:`biosnicar.inverse.optimize`).  It operates in whatever "retrieval space"
the caller's ``forward_fn`` expects; the caller is responsible for any
parameter transforms and for building the prior/covariance inputs.

Notation (Rodgers):
    x     state vector              (n,)
    y     measurement               (m,)
    F(x)  forward model             x -> y
    x_a   prior mean                (n,)
    S_a   prior covariance          (n, n)
    S_e   measurement covariance    (m, m) or (m,) diagonal
    K     Jacobian dF/dx            (m, n)
    x̂     posterior (MAP) estimate
    Ŝ     posterior covariance = (K^T S_e^-1 K + S_a^-1)^-1
    A     averaging kernel = Ŝ K^T S_e^-1 K        (n, n)
    DFS   degrees of freedom for signal = trace(A)
"""

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

ArrayLike = Union[np.ndarray, float]


@dataclass
class OEResult:
    """Result of an optimal-estimation retrieval."""

    x: np.ndarray                 # posterior (MAP) state estimate (n,)
    S: np.ndarray                 # posterior covariance (n, n)
    averaging_kernel: np.ndarray  # A (n, n)
    dfs: float                    # degrees of freedom for signal = trace(A)
    gain: np.ndarray              # G = Ŝ K^T S_e^-1 (n, m)
    jacobian: np.ndarray          # K at the solution (m, n)
    y_pred: np.ndarray            # F(x̂) (m,)
    cost: float                   # J(x̂) = data misfit + prior misfit
    chi2_reduced: float           # data misfit / m  (≈1 when S_e is right)
    log_evidence: float           # Laplace ln p(y | model)
    n_iter: int
    converged: bool

    # Per-parameter 1-sigma (sqrt of diagonal of S), convenience
    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.S), 0.0, np.inf))

    # Per-parameter information content: diagonal of the averaging kernel.
    # ~1 => measurement-determined; ~0 => prior-determined.
    @property
    def parameter_dfs(self) -> np.ndarray:
        return np.clip(np.diag(self.averaging_kernel), 0.0, 1.0 + 1e-9)


# ── covariance helpers (support diagonal-as-1D or full 2D) ───────────────────

def _is_diag(C) -> bool:
    return np.ndim(C) == 1


def _inv_cov_matmul(C, M):
    """Return S^-1 @ M for covariance C given as 1D (diagonal) or 2D."""
    if _is_diag(C):
        return (M.T / C).T if M.ndim > 1 else M / C
    return np.linalg.solve(C, M)


def _logdet(C) -> float:
    if _is_diag(C):
        return float(np.sum(np.log(C)))
    sign, ld = np.linalg.slogdet(C)
    if sign <= 0:
        raise np.linalg.LinAlgError("covariance is not positive-definite")
    return float(ld)


def _cost(forward_fn, x, y, S_e, x_a, S_a):
    r = y - forward_fn(x)
    e = x - x_a
    data = float(r @ _inv_cov_matmul(S_e, r))
    prior = float(e @ _inv_cov_matmul(S_a, e))
    return data + prior, data, r


def _jacobian(forward_fn, x, y_at_x, steps):
    """Central-difference Jacobian dF/dx, shape (m, n). 2n forward evals."""
    n = x.size
    m = y_at_x.size
    K = np.empty((m, n))
    for j in range(n):
        dxj = steps[j]
        xp = x.copy(); xp[j] += dxj
        xm = x.copy(); xm[j] -= dxj
        K[:, j] = (forward_fn(xp) - forward_fn(xm)) / (2.0 * dxj)
    return K


def optimal_estimation(
    forward_fn: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    x_a: np.ndarray,
    S_a: ArrayLike,
    S_e: ArrayLike,
    bounds: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    fd_step: Optional[np.ndarray] = None,
    max_iter: int = 20,
    tol: float = 1e-3,
) -> OEResult:
    """Run a Gauss-Newton / Levenberg-Marquardt optimal-estimation retrieval.

    Parameters
    ----------
    forward_fn : callable
        ``x (n,) -> y_pred (m,)``.  Operates in the caller's retrieval space.
    y : np.ndarray (m,)
        Observed measurement.
    x_a : np.ndarray (n,)
        Prior mean (also the default starting point).
    S_a : np.ndarray (n, n) or (n,)
        Prior covariance (full matrix, or 1D diagonal).
    S_e : np.ndarray (m, m) or (m,)
        Measurement covariance (full matrix, or 1D diagonal).  Should include
        instrument noise **and** forward-model error; under-stating it makes
        the posterior over-confident.
    bounds : np.ndarray (n, 2), optional
        ``[[lo, hi], ...]`` in retrieval space; iterates are clipped into them.
    x0 : np.ndarray (n,), optional
        Starting state (default: ``x_a``).
    fd_step : np.ndarray (n,), optional
        Finite-difference steps for the Jacobian (default: 1% of the prior
        standard deviation, floored to a small absolute value).
    max_iter, tol : int, float
        Iteration cap and convergence tolerance on the normalised step
        ``dx^T H dx / n``.

    Returns
    -------
    OEResult
    """
    y = np.asarray(y, dtype=float)
    x_a = np.asarray(x_a, dtype=float)
    n = x_a.size
    x = (x_a if x0 is None else np.asarray(x0, dtype=float)).copy()

    Sa_diag = np.asarray(S_a, dtype=float)
    sigma_a = np.sqrt(Sa_diag) if _is_diag(Sa_diag) else np.sqrt(np.diag(Sa_diag))
    if fd_step is None:
        fd_step = np.maximum(0.01 * sigma_a, 1e-6)

    def _clip(v):
        if bounds is None:
            return v
        return np.clip(v, bounds[:, 0], bounds[:, 1])

    x = _clip(x)
    if _is_diag(Sa_diag):
        Sa_inv = np.diag(1.0 / Sa_diag)
    else:
        Sa_inv = np.linalg.inv(Sa_diag)

    cost, _, _ = _cost(forward_fn, x, y, S_e, x_a, Sa_diag)
    lam = 1e-3  # Levenberg-Marquardt damping
    converged = False
    it = 0
    K = None
    for it in range(1, max_iter + 1):
        y_at_x = forward_fn(x)
        K = _jacobian(forward_fn, x, y_at_x, fd_step)
        KtSe_inv = _inv_cov_matmul(S_e, K).T            # (n, m) = K^T S_e^-1
        H = KtSe_inv @ K + Sa_inv                       # GN Hessian / 2  (n, n)
        r = y - y_at_x
        grad_half = KtSe_inv @ r - Sa_inv @ (x - x_a)   # -0.5 ∇J

        # Levenberg-Marquardt: try a step, accept if cost drops, else damp more.
        accepted = False
        for _ in range(12):
            try:
                dx = np.linalg.solve(H + lam * np.diag(np.diag(H)), grad_half)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            x_new = _clip(x + dx)
            cost_new, _, _ = _cost(forward_fn, x_new, y, S_e, x_a, Sa_diag)
            if cost_new < cost:
                step_size = float(dx @ H @ dx) / n
                x, cost = x_new, cost_new
                lam = max(lam * 0.4, 1e-8)
                accepted = True
                break
            lam *= 4.0
        if not accepted:
            converged = True   # cannot reduce further -> at a minimum
            break
        if step_size < tol:
            converged = True
            break

    # Final diagnostics at the solution
    y_pred = forward_fn(x)
    K = _jacobian(forward_fn, x, y_pred, fd_step)
    KtSe_inv = _inv_cov_matmul(S_e, K).T
    H = KtSe_inv @ K + Sa_inv
    S_hat = np.linalg.inv(H)
    gain = S_hat @ KtSe_inv
    A = S_hat @ (KtSe_inv @ K)
    dfs = float(np.trace(A))

    cost, data_misfit, _ = _cost(forward_fn, x, y, S_e, x_a, Sa_diag)
    m = y.size
    chi2_reduced = data_misfit / m

    # Laplace approximation to the log-evidence ln p(y | model):
    #   ln Z ≈ -0.5 [ J(x̂) + ln|S_e| + ln|S_a| - ln|Ŝ| + m ln 2π ]
    # (full/absolute form so it is comparable across models with different m,
    # S_e, or n — needed when surface types use different band sets.)
    log_evidence = -0.5 * (
        cost + _logdet(S_e) + _logdet(Sa_diag) - _logdet(S_hat)
        + m * np.log(2.0 * np.pi)
    )

    return OEResult(
        x=x, S=S_hat, averaging_kernel=A, dfs=dfs, gain=gain, jacobian=K,
        y_pred=y_pred, cost=cost, chi2_reduced=chi2_reduced,
        log_evidence=log_evidence, n_iter=it, converged=converged,
    )
