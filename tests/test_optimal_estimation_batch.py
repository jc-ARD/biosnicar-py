"""Correctness gate for the batched OE engine: it must reproduce the scalar
:func:`optimal_estimation` per pixel (same math, batched arithmetic)."""

import numpy as np
import pytest

from biosnicar.inverse.optimal_estimation import optimal_estimation
from biosnicar.inverse.optimal_estimation_batch import optimal_estimation_batch


def _linear_problem(rng, n=3, m=40):
    K0 = rng.normal(size=(m, n))
    x_true = rng.uniform(-1, 1, n)
    x_a = np.zeros(n)
    S_a = np.full(n, 4.0)
    S_e = np.full(m, 0.05 ** 2)
    return K0, x_true, x_a, S_a, S_e


def test_batch_matches_scalar_linear():
    """Linear forward: batched == N scalar runs, to tight tolerance."""
    rng = np.random.default_rng(0)
    K0, x_true, x_a, S_a, S_e = _linear_problem(rng)
    n, m = x_a.size, S_e.size
    N = 12
    Y = np.stack([K0 @ x_true + rng.normal(0, 0.05, m) for _ in range(N)])

    def fb(X):                      # (N,n) -> (N,m)
        return X @ K0.T

    batch = optimal_estimation_batch(fb, Y, x_a, S_a, S_e, max_iter=30)

    for i in range(N):
        sol = optimal_estimation(lambda x: K0 @ x, Y[i], x_a, S_a, S_e,
                                 max_iter=30)
        assert np.allclose(batch.x[i], sol.x, rtol=1e-5, atol=1e-7)
        assert np.allclose(batch.sigma[i], sol.sigma, rtol=1e-4, atol=1e-7)
        assert abs(batch.dfs[i] - sol.dfs) < 1e-5
        assert abs(batch.log_evidence[i] - sol.log_evidence) < 1e-4
        assert np.allclose(batch.parameter_dfs[i], sol.parameter_dfs,
                           rtol=1e-4, atol=1e-6)


def test_batch_matches_scalar_nonlinear_with_bounds():
    """Mildly non-linear forward + bounds + full S_e."""
    rng = np.random.default_rng(1)
    n, m = 2, 25
    A = rng.normal(size=(m, n))
    x_true = np.array([0.4, -0.3])
    x_a = np.zeros(n)
    S_a = np.diag([1.0, 1.0])
    # full (correlated) S_e
    B = rng.normal(size=(m, m)) * 0.02
    S_e = B @ B.T + np.eye(m) * 0.01
    bounds = np.array([[-2.0, 2.0]] * n)

    def f(x):
        return A @ x + 0.3 * (A @ x) ** 2

    def fb(X):
        lin = X @ A.T
        return lin + 0.3 * lin ** 2

    N = 8
    Y = np.stack([f(x_true) + rng.multivariate_normal(np.zeros(m), S_e)
                  for _ in range(N)])
    batch = optimal_estimation_batch(fb, Y, x_a, S_a, S_e, bounds=bounds,
                                     max_iter=40)
    for i in range(N):
        sol = optimal_estimation(f, Y[i], x_a, S_a, S_e, bounds=bounds,
                                 max_iter=40)
        assert np.allclose(batch.x[i], sol.x, rtol=1e-4, atol=1e-6)
        assert abs(batch.log_evidence[i] - sol.log_evidence) < 1e-3
        assert np.allclose(np.diag(batch.S[i]), np.diag(sol.S),
                           rtol=1e-3, atol=1e-7)


def test_pixels_are_independent():
    """Row i of the batch depends only on Y[i] (no cross-pixel leakage)."""
    rng = np.random.default_rng(2)
    K0, x_true, x_a, S_a, S_e = _linear_problem(rng, n=2, m=20)
    n, m = x_a.size, S_e.size

    def fb(X):
        return X @ K0.T

    Y = np.stack([K0 @ x_true + rng.normal(0, 0.05, m) for _ in range(6)])
    full = optimal_estimation_batch(fb, Y, x_a, S_a, S_e)
    subset = optimal_estimation_batch(fb, Y[2:4], x_a, S_a, S_e)
    assert np.allclose(full.x[2:4], subset.x, rtol=1e-8, atol=1e-10)
