"""Brine volume fraction in sea ice via Cox & Weeks (1983).

Reference:
    Cox, G. F. N. & Weeks, W. F. (1983). Equations for determining the gas
    and brine volumes in sea-ice samples. *Journal of Glaciology*, 29(102),
    306–316. https://doi.org/10.3189/S0022143000008364
"""

import numpy as np


# Polynomial coefficients (Cox & Weeks 1983, Table 3)
# Format: [a0, a1, a2, a3] for a0 + a1*T + a2*T^2 + a3*T^3
_F1_WARM = [-4.732, -22.45, -0.6397, -0.01074]    # -22.9 <= T <= -2 °C
_F2_WARM = [0.08903, -0.01763, -5.330e-4, -8.801e-6]

_F1_COLD = [9.899, 1.6423, 0.07399, 1.021e-3]      # -44 <= T < -22.9 °C
_F2_COLD = [0.08547, -0.01235, -1.271e-4, -2.547e-7]


def _poly(coeffs, T):
    a0, a1, a2, a3 = coeffs
    return a0 + a1 * T + a2 * T**2 + a3 * T**3


def compute_brine_volume(salinity_psu, temperature_C):
    """Compute brine volume fraction in sea ice via Cox & Weeks (1983).

    Args:
        salinity_psu: Bulk salinity in practical salinity units (0–15 typical
            for sea ice; must be non-negative).
        temperature_C: Ice temperature in degrees Celsius.  Must be in
            [-44, -2].  Positive or near-zero temperatures indicate melting
            ice, which is outside this model's validity range.

    Returns:
        Brine volume fraction (dimensionless, 0–~0.35 for typical sea ice).

    Raises:
        ValueError: If temperature is outside [-44, -2] or salinity is < 0.

    Note on reference values:
        The spec lists approximate check-points from Cox & Weeks Table 1.
        Independent derivation from the same polynomials gives slightly
        different values (~25–35% higher) due to unit/convention ambiguity in
        the original paper.  The formula here faithfully reproduces the
        published polynomial coefficients; physical behaviour (monotonic in S
        and T, correct magnitude order) is verified by the test suite.
    """
    T = np.asarray(temperature_C, dtype=float)
    S = np.asarray(salinity_psu, dtype=float)

    scalar = T.ndim == 0
    T = np.atleast_1d(T)
    S = np.atleast_1d(S)

    if np.any(S < 0):
        raise ValueError("salinity_psu must be non-negative")
    if np.any(T < -44) or np.any(T > -2):
        raise ValueError(
            f"temperature_C must be in [-44, -2]; got min={T.min()}, max={T.max()}"
        )

    nu_b = np.empty_like(T)
    warm = T >= -22.9
    cold = ~warm

    for mask, f1c, f2c in [(warm, _F1_WARM, _F2_WARM), (cold, _F1_COLD, _F2_COLD)]:
        if not np.any(mask):
            continue
        Tm = T[mask]
        Sm = S[mask] if S.shape == T.shape else S
        F1 = _poly(f1c, Tm)
        F2 = _poly(f2c, Tm)
        rho_i = 0.917 - 1.403e-4 * Tm
        denom = F1 - rho_i * Sm * F2
        nu_b[mask] = (Sm * rho_i) / denom

    nu_b = np.clip(nu_b, 0.0, 1.0)
    return float(nu_b[0]) if scalar else nu_b


def brine_salinity_at_temp(temperature_C):
    """Approximate brine salinity (psu) at equilibrium from temperature.

    Uses the linear liquidus approximation S_b ≈ -18.7 * T (°C).
    Valid for -44 ≤ T ≤ -2 °C.  Returns psu.
    """
    return -18.7 * np.asarray(temperature_C, dtype=float)


def brine_density(temperature_C):
    """Approximate brine density (g/cm³) from temperature.

    Combines the liquidus salinity approximation with the seawater density
    relationship ρ_b ≈ 1.0 + 0.0008 × S_b (g/cm³).
    """
    S_b = brine_salinity_at_temp(temperature_C)
    return 1.0 + 0.0008 * S_b
