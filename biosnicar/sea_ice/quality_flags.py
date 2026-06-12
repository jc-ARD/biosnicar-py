"""Quality flags for sea ice retrievals.

A ``uint8`` bitmask attached to every :class:`SeaIceRetrievalResult` (and to
batch outputs) marking conditions under which the classification or the
retrieved parameters should be treated with caution.
"""

from typing import Dict


class QualityFlag:
    """Bitmask values for retrieval quality conditions."""

    POOR_FIT             = 0x01  # best cost > POOR_FIT_THRESHOLD
    LOW_CONFIDENCE       = 0x02  # confidence < LOW_CONFIDENCE_THRESHOLD
    AT_BOUNDS            = 0x04  # a retrieved parameter within 1% of training bound
    SPECTRALLY_AMBIGUOUS = 0x08  # best and second-best cost within 10%
    NO_CONVERGENCE       = 0x10  # optimizer reported convergence=False
    OPEN_WATER_LIKELY    = 0x20  # classified as open_water (informational)
    YOUNG_ICE_LIKELY     = 0x40  # classified as young_ice (informational)


# Empirical thresholds from the SHEBA validation suite.
POOR_FIT_THRESHOLD = 0.05       # chi-squared
LOW_CONFIDENCE_THRESHOLD = 0.20  # confidence metric (see retrieve.py)

# Fraction of the parameter range within which a retrieved value counts
# as "at" a training bound.
_BOUND_TOLERANCE = 0.01

_FLAG_NAMES = {
    "poor_fit":             QualityFlag.POOR_FIT,
    "low_confidence":       QualityFlag.LOW_CONFIDENCE,
    "at_bounds":            QualityFlag.AT_BOUNDS,
    "spectrally_ambiguous": QualityFlag.SPECTRALLY_AMBIGUOUS,
    "no_convergence":       QualityFlag.NO_CONVERGENCE,
    "open_water_likely":    QualityFlag.OPEN_WATER_LIKELY,
    "young_ice_likely":     QualityFlag.YOUNG_ICE_LIKELY,
}


def compute_quality_flags(
    cost,
    confidence,
    converged,
    parameters,
    bounds,
    surface_type,
    cost_per_type=None,
) -> int:
    """Compute the uint8 quality bitmask for a completed retrieval.

    Parameters
    ----------
    cost : float
        Chi-squared of the winning fit.
    confidence : float
        Classification confidence (0-1).
    converged : bool
        Winning optimiser convergence flag.
    parameters : dict
        ``{name: value}`` retrieved (free) parameters of the winner.
    bounds : dict
        ``{name: (lo, hi)}`` training bounds of the winning emulator.
    surface_type : str
        Winning surface type label.
    cost_per_type : dict or None
        ``{surface_type: cost}`` for ambiguity checking.
    """
    flags = 0
    if cost > POOR_FIT_THRESHOLD:
        flags |= QualityFlag.POOR_FIT
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        flags |= QualityFlag.LOW_CONFIDENCE
    for name, val in parameters.items():
        b = bounds.get(name)
        if b is None:
            continue
        lo, hi = float(b[0]), float(b[1])
        tol = _BOUND_TOLERANCE * (hi - lo)
        if val <= lo + tol or val >= hi - tol:
            flags |= QualityFlag.AT_BOUNDS
            break
    if cost_per_type and len(cost_per_type) > 1:
        ranked = sorted(cost_per_type.values())
        best, second = ranked[0], ranked[1]
        if second > 0 and (second - best) <= 0.10 * second:
            flags |= QualityFlag.SPECTRALLY_AMBIGUOUS
    if not converged:
        flags |= QualityFlag.NO_CONVERGENCE
    if surface_type == "open_water":
        flags |= QualityFlag.OPEN_WATER_LIKELY
    if surface_type == "young_ice":
        flags |= QualityFlag.YOUNG_ICE_LIKELY
    return flags


def describe_quality_flags(flags: int) -> Dict[str, bool]:
    """Unpack a quality bitmask into ``{flag_name: bool}``."""
    return {name: bool(flags & bit) for name, bit in _FLAG_NAMES.items()}
