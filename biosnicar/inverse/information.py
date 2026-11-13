"""Per-parameter information content: did the measurement determine this, or the prior?

Optimal estimation returns a value and a 1-sigma for every free parameter,
and nothing in those two numbers distinguishes

    "the spectrum pinned this tightly"

from

    "the prior was narrow and the spectrum said nothing".

Both produce a small sigma.  The quantity that separates them is the
diagonal of the averaging kernel,

    A = dx_retrieved / dx_true,

which is the fraction of each retrieved value that came from the
measurement rather than from the prior (Rodgers 2000, ch. 3).  ``A = 1``
means the measurement determined the value outright; ``A = 0`` means the
retrieval handed the prior straight back, uncertainty included.

This module defines one convention for reading that number, so that
``RetrievalResult``, ``SeaIceRetrievalResult``, the scene rasters, the
summaries and the plots all draw the same line in the same place.

Why these thresholds
--------------------
``MEASURED_THRESHOLD = 0.5`` is the half-and-half point: below it, more of
the reported value comes from the prior than from the spectrum.

``NOT_RETRIEVED_THRESHOLD = 0.1`` marks values that are not retrievals at
all.  It is set from measurement, not taste: ``sea_ice_temperature`` scores
A = 0.048 across 22 of 23 SHEBA spectra, because temperature acts only
through brine volume and brine salinity while the bare-ice and SSL
emulators are ``vis_only``-masked, where bubble scattering swamps brine
absorption.  There is no signal there to retrieve, and a parameter in that
state should not be quoted with its uncertainty as though there were.

What A does *not* tell you
--------------------------
A is measured **relative to the prior you supplied**, not against truth.
Loosen the prior and A rises for free: on one synthetic pond spectrum,
``sea_ice_temperature`` scores A = 0.86 under ``retrieve()``'s wide default
prior while running to the edge of its training range (-9.99 degC against a
truth of -4), and A = 0.13 under the seasonal prior that
``retrieve_sea_ice()`` actually applies.  The spectrum did not become less
informative; the prior became honest.  So a high A means "the measurement
moved this away from my prior", which is only a claim about accuracy when
the prior was a fair one.

Two companions close that gap, and all three should be read together:

* ``QualityFlag.AT_BOUNDS`` -- a parameter pinned to its training bound
  *with* high A is a binding constraint, and the reported sigma is the
  width of a wall, not of a posterior.
* ``RetrievalResult.parameter_correlation()`` -- the companion degeneracy
  diagnostic.  A tells you whether a parameter was measured;
  correlation tells you whether it was measured *separately from another
  one*.  A parameter can score A = 0.98 and
  still be individually meaningless if it is 0.98-correlated with its
  neighbour -- exactly the ``sea_ice_bubble_radius`` / ``rho_DL`` case that
  SSA reporting exists to handle.
"""

from typing import Dict, Mapping, Optional

# Averaging-kernel diagonal at or above which the measurement, not the
# prior, dominates the reported value.
MEASURED_THRESHOLD = 0.5

# Below this the value is the prior restated; it should not be reported as
# a retrieval, with or without its uncertainty.
NOT_RETRIEVED_THRESHOLD = 0.1

MEASURED = "measured"
PRIOR_INFLUENCED = "prior-influenced"
NOT_RETRIEVED = "not retrieved"
UNKNOWN = "unknown"

#: Ordered worst-to-best, for sorting and for choosing a headline label.
PROVENANCE_ORDER = (NOT_RETRIEVED, PRIOR_INFLUENCED, MEASURED, UNKNOWN)

_DESCRIPTIONS = {
    MEASURED: "the spectrum determined this value",
    PRIOR_INFLUENCED: "part prior, part spectrum — quote with care",
    NOT_RETRIEVED: "the value and its uncertainty are essentially the prior",
    UNKNOWN: "no averaging kernel available (not an OE retrieval)",
}


def classify(akd_value: Optional[float],
             measured_threshold: float = MEASURED_THRESHOLD,
             not_retrieved_threshold: float = NOT_RETRIEVED_THRESHOLD) -> str:
    """Label one averaging-kernel diagonal.

    Returns one of :data:`MEASURED`, :data:`PRIOR_INFLUENCED`,
    :data:`NOT_RETRIEVED` or :data:`UNKNOWN` (for None / NaN, i.e. a
    retrieval method that produces no averaging kernel).
    """
    if akd_value is None:
        return UNKNOWN
    a = float(akd_value)
    if a != a:                                   # NaN
        return UNKNOWN
    if a < not_retrieved_threshold:
        return NOT_RETRIEVED
    if a < measured_threshold:
        return PRIOR_INFLUENCED
    return MEASURED


def describe(label: str) -> str:
    """One-line explanation of a provenance label."""
    return _DESCRIPTIONS.get(label, "")


def provenance(akd: Optional[Mapping[str, float]],
               parameters: Optional[Mapping[str, float]] = None,
               **kw) -> Dict[str, str]:
    """``{param: label}`` for every parameter.

    *akd* is the averaging-kernel diagonal mapping; *parameters* is the
    reported parameter set.  Parameters present in *parameters* but absent
    from *akd* are labelled :data:`UNKNOWN` rather than omitted — a derived
    quantity such as ``snow_depth`` has no averaging kernel of its own, and
    silently dropping it from the provenance report would read as an
    endorsement.
    """
    akd = akd or {}
    names = list(parameters) if parameters is not None else list(akd)
    for p in akd:
        if p not in names:
            names.append(p)
    return {p: classify(akd.get(p), **kw) for p in names}


def format_parameter_lines(parameters: Mapping[str, float],
                           uncertainty: Mapping[str, float],
                           akd: Optional[Mapping[str, float]] = None,
                           name_width: int = 25,
                           indent: str = "    ") -> list:
    """Render the parameter block of a summary, with provenance when known.

    Falls back to the plain ``name = value ± sigma`` form when *akd* is
    empty, so non-OE retrievals read exactly as before rather than growing
    a column of "unknown".
    """
    akd = dict(akd or {})
    lines = []
    if not akd:
        for name, val in parameters.items():
            unc = uncertainty.get(name, float("nan"))
            lines.append(f"{indent}{name:{name_width}s} = {val:12.4f}  ±  {unc:.4f}")
        return lines

    labels = provenance(akd, parameters)
    # 30 = len(" = ") + 12 (value) + len("  \u00b1  ") + 8 (sigma) + 2, so that
    # "info" sits exactly over the averaging-kernel column.
    lines.append(f"{indent}{' ' * (name_width + 30)}  info  provenance")
    for name, val in parameters.items():
        unc = uncertainty.get(name, float("nan"))
        a = akd.get(name)
        a_txt = "   -  " if a is None or a != a else f"{float(a):6.2f}"
        lines.append(
            f"{indent}{name:{name_width}s} = {val:12.4f}  ±  {unc:8.4f}  "
            f"{a_txt}  {labels[name]}"
        )

    flagged = [p for p, lab in labels.items()
               if lab == NOT_RETRIEVED and p in parameters]
    if flagged:
        lines.append(
            f"{indent}! not retrieved (averaging kernel < "
            f"{NOT_RETRIEVED_THRESHOLD:.2f}): {', '.join(flagged)}"
        )
        lines.append(
            f"{indent}  {describe(NOT_RETRIEVED)}; do not quote as a measurement."
        )
    return lines
