"""Maxwell-Garnett effective medium theory for sea ice.

Computes the effective complex permittivity of a composite material
consisting of spherical inclusions (brine) embedded in a host matrix (ice).

Reference:
    Sihvola, A. (1999). *Electromagnetic Mixing Formulas and Applications*.
    IEE Electromagnetic Waves Series 47, London.

Note:
    This module implements the sphere-inclusion variant of Maxwell-Garnett.
    The formula is valid for dilute inclusions (volume fraction f < ~0.3).
    For higher brine fractions (rare in sea ice outside melt season), the
    Bruggeman symmetric mixing rule would be more accurate — that extension
    is deferred to a future version.
"""

import numpy as np


def ri_to_eps(ri_complex: np.ndarray) -> np.ndarray:
    """Convert complex refractive index (n + ik) to complex permittivity.

    ε = (n + ik)² = (n² − k²) + i(2nk)

    Args:
        ri_complex: Complex array of shape (...,).  Real part = n, imaginary
            part = k (both ≥ 0 for physical media).

    Returns:
        Complex permittivity of same shape.
    """
    return ri_complex**2


def eps_to_ri(eps: np.ndarray) -> np.ndarray:
    """Convert complex permittivity to complex refractive index (n + ik).

    Uses the principal square root so that n ≥ 0 and k ≥ 0.

    Args:
        eps: Complex permittivity array.

    Returns:
        Complex refractive index with non-negative real and imaginary parts.
    """
    ri = np.sqrt(eps.astype(complex))
    # Ensure physical sign conventions (n ≥ 0, k ≥ 0)
    ri = np.where(np.real(ri) < 0, -ri, ri)
    return ri


def maxwell_garnett(
    eps_host: np.ndarray,
    eps_inclusion: np.ndarray,
    volume_fraction: float,
) -> np.ndarray:
    """Maxwell-Garnett effective complex permittivity (sphere inclusions).

    Formula (Sihvola 1999, Eq. 5.62 in terms of permittivities):

        ε_eff = ε_h × [ε_i(1 + 2f) + 2ε_h(1 − f)] /
                      [ε_i(1 − f) + ε_h(2 + f)]

    where f is the inclusion volume fraction, ε_h is the host permittivity,
    and ε_i is the inclusion permittivity.

    Note:
        This is the sphere-inclusion variant. The spheroid generalisation
        (Niccolai/Sihvola) is deferred; for typical sea ice brine fractions
        (< 0.15) the sphere approximation introduces < 5% error in ε_eff.

    Args:
        eps_host: Complex permittivity of host medium (e.g. ice).
            Shape (n_wavelengths,) or broadcastable.
        eps_inclusion: Complex permittivity of inclusion (e.g. brine).
            Same shape as eps_host.
        volume_fraction: Volume fraction of inclusions (0 to 1).
            Maxwell-Garnett is most accurate for f < 0.3.

    Returns:
        Effective complex permittivity, same shape as inputs.

    Examples:
        >>> import numpy as np
        >>> eps_h = np.array([1.7 + 0j] * 5)   # ice-like
        >>> eps_i = np.array([1.78 + 0.01j] * 5)  # brine-like
        >>> eps_eff = maxwell_garnett(eps_h, eps_i, 0.05)
        >>> # Result should be between host and inclusion
    """
    f = float(volume_fraction)
    eps_h = np.asarray(eps_host, dtype=complex)
    eps_i = np.asarray(eps_inclusion, dtype=complex)

    numerator = eps_i * (1.0 + 2.0 * f) + 2.0 * eps_h * (1.0 - f)
    denominator = eps_i * (1.0 - f) + eps_h * (2.0 + f)

    return eps_h * numerator / denominator


def effective_ri_ice_brine(
    ri_ice: np.ndarray,
    ri_brine: np.ndarray,
    brine_volume_fraction: float,
) -> np.ndarray:
    """Effective complex refractive index of ice containing brine inclusions.

    Convenience wrapper: converts RI → ε, applies Maxwell-Garnett, converts
    back to RI.

    Args:
        ri_ice: Complex RI of pure ice, shape (480,).
        ri_brine: Complex RI of brine, shape (480,).
        brine_volume_fraction: Fraction of total volume occupied by brine
            (from compute_brine_volume).

    Returns:
        Effective complex RI, shape (480,).
    """
    eps_ice = ri_to_eps(ri_ice)
    eps_brine = ri_to_eps(ri_brine)
    eps_eff = maxwell_garnett(eps_ice, eps_brine, brine_volume_fraction)
    return eps_to_ri(eps_eff)
