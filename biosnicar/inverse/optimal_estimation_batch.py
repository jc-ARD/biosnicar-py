"""Batched optimal estimation — N independent retrievals sharing priors.

A vectorised Gauss-Newton / Levenberg-Marquardt retrieval that solves N pixels
at once, reproducing the per-pixel math of
:func:`biosnicar.inverse.optimal_estimation.optimal_estimation` exactly but
replacing N Python optimiser loops with batched linear algebra. The forward
model is evaluated for all N pixels in one call (``forward_batch``), which is
where the speed comes from: the emulator MLP is a matmul that handles
``(N, n) -> (N, m)`` natively.

Assumptions that make the batch well-posed (all true in the sea-ice fleet):

* the prior mean ``x_a``, prior covariance ``S_a`` and measurement covariance
  ``S_e`` are **shared** across pixels — only the measurement ``Y`` differs;
* the GN Hessian ``H = K^T S_e^-1 K + S_a^-1`` is positive-definite because
  ``S_a^-1`` is (a proper prior covariance), so ``H + lam·diag(H)`` is never
  singular — the batched solve does not need the scalar engine's per-pixel
  LinAlgError guard.

Each pixel keeps its **own** Levenberg-Marquardt damping and its own
accept/converge state (a ``done`` mask freezes a pixel once it converges), so
the per-pixel optimisation path matches the scalar engine; only the arithmetic
is shared. See docs — this is roadmap G3 (vectorised inverse) and the engine
behind ``retrieve_sea_ice_batch(method="oe", engine="vectorized")``.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class OEBatchResult:
    """Batched OE result — arrays share the leading pixel axis N."""
    x: np.ndarray            # posterior state (N, n)
    S: np.ndarray            # posterior covariance (N, n, n)
    averaging_kernel: np.ndarray  # (N, n, n)
    dfs: np.ndarray          # (N,)
    y_pred: np.ndarray       # (N, m)
    cost: np.ndarray         # (N,)
    chi2_reduced: np.ndarray  # (N,)
    log_evidence: np.ndarray  # (N,)
    n_iter: np.ndarray       # (N,)
    converged: np.ndarray    # (N,) bool

    @property
    def sigma(self):
        """Per-parameter 1-sigma, (N, n)."""
        d = np.diagonal(self.S, axis1=1, axis2=2)
        return np.sqrt(np.clip(d, 0.0, np.inf))

    @property
    def parameter_dfs(self):
        """Averaging-kernel diagonal, (N, n)."""
        d = np.diagonal(self.averaging_kernel, axis1=1, axis2=2)
        return np.clip(d, 0.0, 1.0 + 1e-9)


def _is_diag(C):
    return np.ndim(C) == 1


def _make_se_ops(S_e, m):
    """Return (apply_inv, logdet) for the shared measurement covariance.

    apply_inv(M) computes S_e^-1 @ M for M of shape (N, m, k).
    """
    if _is_diag(S_e):
        se = np.asarray(S_e, dtype=float)
        inv = 1.0 / se

        def apply_inv(M):                       # (N, m, k)
            return M * inv[None, :, None]
        logdet = float(np.sum(np.log(se)))
    else:
        Se = np.asarray(S_e, dtype=float)
        Se_inv = np.linalg.inv(Se)
        sign, ld = np.linalg.slogdet(Se)
        if sign <= 0:
            raise np.linalg.LinAlgError("S_e is not positive-definite")
        logdet = float(ld)

        def apply_inv(M):                       # (N, m, k)
            return np.einsum("mk,Nkj->Nmj", Se_inv, M, optimize=True)
    return apply_inv, logdet


def optimal_estimation_batch(forward_batch, Y, x_a, S_a, S_e,
                             bounds=None, x0=None, fd_step=None,
                             max_iter=20, tol=1e-3):
    """Batched Gauss-Newton/LM OE over N pixels sharing (x_a, S_a, S_e).

    Parameters
    ----------
    forward_batch : callable
        ``X (N, n) -> Y_pred (N, m)`` in the caller's retrieval space.
    Y : (N, m) array
        Observations (one row per pixel).
    x_a : (n,) array
        Shared prior mean (also default start).
    S_a : (n,) or (n, n) array
        Shared prior covariance (1D diagonal or full).
    S_e : (m,) or (m, m) array
        Shared measurement covariance (1D diagonal or full).
    bounds : (n, 2) array, optional
        Retrieval-space bounds; iterates are clipped into them.
    x0 : (n,) or (N, n) array, optional
        Start state (default: x_a broadcast to all pixels).
    fd_step : (n,) array, optional
        Finite-difference steps (default 1% of prior sigma, floored 1e-6).
    max_iter, tol : int, float
        Outer-iteration cap and step-size tolerance (dx^T H dx / n).
    """
    Y = np.asarray(Y, dtype=float)
    N, m = Y.shape
    x_a = np.asarray(x_a, dtype=float)
    n = x_a.size

    X = (np.broadcast_to(x_a, (N, n)).copy() if x0 is None
         else np.broadcast_to(np.asarray(x0, dtype=float), (N, n)).copy())

    Sa = np.asarray(S_a, dtype=float)
    sigma_a = np.sqrt(Sa) if _is_diag(Sa) else np.sqrt(np.diag(Sa))
    if fd_step is None:
        fd_step = np.maximum(0.01 * sigma_a, 1e-6)
    fd_step = np.asarray(fd_step, dtype=float)

    Sa_inv = np.diag(1.0 / Sa) if _is_diag(Sa) else np.linalg.inv(Sa)
    sign, logdet_Sa = np.linalg.slogdet(np.diag(Sa) if _is_diag(Sa) else Sa)
    logdet_Sa = float(logdet_Sa)
    apply_se_inv, logdet_Se = _make_se_ops(S_e, m)

    lo = None if bounds is None else np.asarray(bounds, float)[:, 0]
    hi = None if bounds is None else np.asarray(bounds, float)[:, 1]

    def clip(V):
        return V if bounds is None else np.clip(V, lo, hi)

    def cost_of(Xc):
        """Batched cost J = data misfit + prior misfit, (N,)."""
        r = Y - forward_batch(Xc)                       # (N, m)
        Sein_r = apply_se_inv(r[:, :, None])[:, :, 0]   # S_e^-1 r  (N, m)
        data = np.einsum("Nm,Nm->N", r, Sein_r, optimize=True)
        e = Xc - x_a
        prior = np.einsum("Nn,nk,Nk->N", e, Sa_inv, e, optimize=True)
        return data + prior, r

    def jacobian(Xc):
        """Central-difference K, (N, m, n). 2n forward_batch calls."""
        K = np.empty((N, m, n))
        for j in range(n):
            dj = fd_step[j]
            Xp = Xc.copy(); Xp[:, j] += dj
            Xm = Xc.copy(); Xm[:, j] -= dj
            K[:, :, j] = (forward_batch(Xp) - forward_batch(Xm)) / (2.0 * dj)
        return K

    X = clip(X)
    cost, _ = cost_of(X)
    lam = np.full(N, 1e-3)
    done = np.zeros(N, dtype=bool)
    n_iter = np.zeros(N, dtype=int)
    step_size = np.full(N, np.inf)

    for it in range(1, max_iter + 1):
        if done.all():
            break
        n_iter[~done] = it

        K = jacobian(X)                                 # (N, m, n)
        SeinvK = apply_se_inv(K)                        # (N, m, n)
        KtSe_inv = np.transpose(SeinvK, (0, 2, 1))      # (N, n, m) = K^T S_e^-1
        H = np.matmul(KtSe_inv, K) + Sa_inv[None]       # (N, n, n)
        _, r = cost_of(X)
        grad = (np.matmul(KtSe_inv, r[:, :, None])[:, :, 0]
                - np.einsum("nk,Nk->Nn", Sa_inv, X - x_a, optimize=True))
        diagH = np.diagonal(H, axis1=1, axis2=2)        # (N, n)
        eye = np.eye(n)

        accepted = np.zeros(N, dtype=bool)
        for _ in range(12):
            trying = (~done) & (~accepted)
            if not trying.any():
                break
            Hdamp = H + lam[:, None, None] * (diagH[:, :, None] * eye[None])
            dx = np.linalg.solve(Hdamp, grad[:, :, None])[:, :, 0]   # (N, n)
            X_new = clip(X + dx)
            cost_new, _ = cost_of(X_new)
            improved = trying & (cost_new < cost)
            # dx^T H dx / n  (only meaningful where improved)
            step = np.einsum("Nn,Nnk,Nk->N", dx, H, dx, optimize=True) / n

            X[improved] = X_new[improved]
            cost[improved] = cost_new[improved]
            step_size[improved] = step[improved]
            lam[improved] = np.maximum(lam[improved] * 0.4, 1e-8)
            accepted |= improved
            worse = trying & ~improved
            lam[worse] *= 4.0

        # Pixels that could not reduce cost this outer iter are at a minimum.
        done |= (~done) & (~accepted)
        # Accepted pixels whose step is below tol have converged.
        done |= accepted & (step_size < tol)

    # ── final diagnostics at the solution (all pixels) ──
    y_pred = forward_batch(X)
    K = jacobian(X)
    SeinvK = apply_se_inv(K)
    KtSe_inv = np.transpose(SeinvK, (0, 2, 1))
    H = np.matmul(KtSe_inv, K) + Sa_inv[None]
    S_hat = np.linalg.inv(H)                            # (N, n, n)
    A = np.matmul(S_hat, np.matmul(KtSe_inv, K))        # (N, n, n)
    dfs = np.trace(A, axis1=1, axis2=2)                 # (N,)

    r = Y - y_pred
    Sein_r = apply_se_inv(r[:, :, None])[:, :, 0]
    data_misfit = np.einsum("Nm,Nm->N", r, Sein_r, optimize=True)
    e = X - x_a
    prior_misfit = np.einsum("Nn,nk,Nk->N", e, Sa_inv, e, optimize=True)
    cost = data_misfit + prior_misfit
    chi2_reduced = data_misfit / m

    sign_sh, logdet_Shat = np.linalg.slogdet(S_hat)     # (N,)
    log_evidence = -0.5 * (
        cost + logdet_Se + logdet_Sa - logdet_Shat + m * np.log(2.0 * np.pi)
    )

    return OEBatchResult(
        x=X, S=S_hat, averaging_kernel=A, dfs=dfs, y_pred=y_pred,
        cost=cost, chi2_reduced=chi2_reduced, log_evidence=log_evidence,
        n_iter=n_iter, converged=done,
    )
