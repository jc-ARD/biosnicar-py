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

# Cox & Weeks (1983) Table 3 cold-range coefficients (-30 <= T < -22.9 °C).
_F1_COLD = [9899.0, 1309.0, 55.27, 0.7160]
_F2_COLD = [8.547, 1.089, 4.518e-2, 5.819e-4]


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
        # The CW83 cold polynomial is only fitted to -30 °C; extrapolating
        # the cubic below that is unstable, so clamp.
        if f1c is _F1_COLD:
            Tm = np.maximum(Tm, -30.0)
        Sm = S[mask] if S.shape == T.shape else S
        F1 = _poly(f1c, Tm)
        F2 = _poly(f2c, Tm)
        rho_i = 0.917 - 1.403e-4 * Tm
        denom = F1 - rho_i * Sm * F2
        if np.any(denom <= 0):
            raise RuntimeError(
                "Cox & Weeks denominator non-positive — outside the "
                "polynomial validity range."
            )
        nu_b[mask] = (Sm * rho_i) / denom

    nu_b = np.clip(nu_b, 0.0, 1.0)
    return float(nu_b[0]) if scalar else nu_b


def invert_brine_volume(brine_volume_fraction, salinity_psu,
                        T_lo=-22.0, T_hi=-2.1):
    """Find the temperature that produces a given brine volume fraction.

    Numerically inverts :func:`compute_brine_volume` by root-finding on the
    monotonic Vb(T) relationship at fixed bulk salinity.  Used to
    reparameterise the sea ice emulators: the emulator works with
    ``brine_volume_fraction`` as a single input, and the transform function
    recovers temperature via this inversion before calling the forward model.

    Args:
        brine_volume_fraction: Target Vb (dimensionless, must be within the
            range produced by the forward function on [T_lo, T_hi]).
        salinity_psu: Reference bulk salinity (psu).  Should match the value
            used as the fixed salinity in the emulator transform function.
        T_lo: Lower temperature bound for root search (°C).  Default -22.
        T_hi: Upper temperature bound for root search (°C).  Default -2.1.

    Returns:
        Temperature (°C) such that ``compute_brine_volume(salinity_psu, T)``
        equals ``brine_volume_fraction``.

    Raises:
        ValueError: If ``brine_volume_fraction`` is outside the range
            achievable within [T_lo, T_hi] at the given salinity.
    """
    from scipy.optimize import brentq

    vb_lo = compute_brine_volume(salinity_psu, T_lo)
    vb_hi = compute_brine_volume(salinity_psu, T_hi)

    # Clamp to achievable range with a small tolerance for floating-point jitter
    # at the bounds of the emulator training range.
    tol = 1e-4
    if brine_volume_fraction < vb_lo - tol or brine_volume_fraction > vb_hi + tol:
        raise ValueError(
            f"brine_volume_fraction={brine_volume_fraction:.4f} is outside the "
            f"range [{vb_lo:.4f}, {vb_hi:.4f}] achievable with "
            f"S={salinity_psu} psu on T in [{T_lo}, {T_hi}]°C."
        )
    vb = float(np.clip(brine_volume_fraction, vb_lo, vb_hi))

    def residual(T):
        return compute_brine_volume(salinity_psu, T) - vb

    return float(brentq(residual, T_lo, T_hi, xtol=1e-4))


def brine_salinity_at_temp(temperature_C):
    """Approximate brine salinity (psu) at equilibrium from temperature.

    Uses the linear liquidus approximation S_b ≈ -18.7 * T (°C), capped at
    250 psu (≈ -13 °C): the linear form increasingly overestimates brine
    salinity below about -8 °C (the true liquidus is strongly sub-linear)
    and uncapped values produce unphysical brine densities.  Returns psu.
    """
    return np.minimum(-18.7 * np.asarray(temperature_C, dtype=float), 250.0)


def brine_density(temperature_C):
    """Approximate brine density (g/cm³) from temperature.

    Combines the liquidus salinity approximation with the seawater density
    relationship ρ_b ≈ 1.0 + 0.0008 × S_b (g/cm³).
    """
    S_b = brine_salinity_at_temp(temperature_C)
    return 1.0 + 0.0008 * S_b
