"""Effective-medium mixing of ice and brine.

Two rules for the effective complex permittivity of a two-phase mixture:

* :func:`maxwell_garnett` — spherical inclusions in a host.  Asymmetric: it
  treats one phase as the continuous matrix, and is only accurate while the
  inclusions are dilute (volume fraction below ~0.3).
* :func:`bruggeman` — the symmetric rule.  Neither phase is privileged, so it
  holds across the whole range of fractions and reduces to each pure phase at
  the ends.  This is what the sea-ice optics use
  (:func:`effective_ri_ice_brine`): a melting layer can be a third brine or
  more, where "brine inclusions in an ice host" stops being a description of
  it and Maxwell-Garnett with ice as host is applied outside its validity.

For ice and brine the choice changes little in practice, because their
permittivities differ by only ~5% and both absorb weakly: the two rules agree
to within 0.04% in the imaginary index over the whole 0-1 range of brine
fraction (and both sit within 0.5% of a plain volume average).  The
symmetric rule is used because it is the one that is valid there, not because
it moves the answer.

Reference:
    Sihvola, A. (1999). *Electromagnetic Mixing Formulas and Applications*.
    IEE Electromagnetic Waves Series 47, London.
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


def bruggeman(
    eps_a: np.ndarray,
    eps_b: np.ndarray,
    volume_fraction_b: float,
) -> np.ndarray:
    """Bruggeman symmetric effective complex permittivity (spherical grains).

    The effective permittivity is the root of

        (1 - f) (eps_a - eps) / (eps_a + 2 eps)
            + f (eps_b - eps) / (eps_b + 2 eps) = 0

    i.e. of ``2 eps^2 - B eps - eps_a eps_b = 0`` with
    ``B = (2 - 3f) eps_a + (3f - 1) eps_b``; the physical root has a
    non-negative imaginary part (a passive medium).

    Args:
        eps_a, eps_b: Complex permittivities of the two phases.
        volume_fraction_b: Volume fraction of phase b, 0 to 1.

    Returns:
        Effective complex permittivity, same shape as the inputs.
    """
    f = float(np.clip(volume_fraction_b, 0.0, 1.0))
    ea = np.asarray(eps_a, dtype=complex)
    eb = np.asarray(eps_b, dtype=complex)
    B = (2.0 - 3.0 * f) * ea + (3.0 * f - 1.0) * eb
    root = np.sqrt(B * B + 8.0 * ea * eb)
    e1 = (B + root) / 4.0
    e2 = (B - root) / 4.0
    # Passive medium: Im(eps) >= 0; between the two phases: Re(eps) > 0.
    pick1 = (np.imag(e1) >= -1e-15) & (np.real(e1) > 0)
    return np.where(pick1, e1, e2)


def effective_ri_ice_brine(
    ri_ice: np.ndarray,
    ri_brine: np.ndarray,
    brine_volume_fraction: float,
) -> np.ndarray:
    """Effective complex refractive index of ice containing brine inclusions.

    Convenience wrapper: converts RI → ε, applies the symmetric Bruggeman
    rule, converts back to RI.  Valid for any brine fraction from 0 to 1.

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
    eps_eff = bruggeman(eps_ice, eps_brine, brine_volume_fraction)
    return eps_to_ri(eps_eff)
