"""Brine and gas volume fractions in sea ice via Cox & Weeks (1983).

Cox & Weeks give brine volume and gas (air) volume as the two halves of a
single density balance.  Both are provided here and both route through
:func:`_cw_terms`, so they cannot drift onto different coefficient sets:

* :func:`compute_brine_volume` — nu_b, from (S, T) and *either* a porosity or
  a bulk density.  With neither it falls back to the gas-free form, which
  overestimates nu_b by the gas fraction (1–9% across the shipped presets).
* :func:`compute_gas_volume` — nu_air, from (S, T) and bulk density.
* :func:`brine_salinity_at_temp` — the liquidus, taken from the same F1 so
  that brine salinity and brine volume conserve salt.
* :func:`mirabilite_mass_fraction`, :func:`hydrohalite_mass_fraction` — the
  salts that precipitate below -8.2 and -22.9 degC.
* :func:`invert_brine_volume` — T from a target nu_b, used by the emulator
  transforms.  Takes the layer density too, so the sampled
  ``brine_volume_fraction`` names the brine volume the forward model will
  actually compute.

Verified against SMRT v1.7.3's independent implementation: the porosity path
agrees to 0.033% (the residual is the pure-ice density constant, 0.917 vs
0.9167 g/cm3), the bulk-density path exactly.  ``tests/test_brine_volume.py``
pins both against hard-coded SMRT values, with no SMRT dependency added.

The melt season (-2 < T <= 0 degC) uses the Leppäranta & Manninen (1988)
coefficients in the same equations, so the model no longer stops at -2 degC.

Reference:
    Cox, G. F. N. & Weeks, W. F. (1983). Equations for determining the gas
    and brine volumes in sea-ice samples. *Journal of Glaciology*, 29(102),
    306–316. https://doi.org/10.3189/S0022143000008364
"""

import warnings

import numpy as np


# Polynomial coefficients (Cox & Weeks 1983, Table 3)
# Format: [a0, a1, a2, a3] for a0 + a1*T + a2*T^2 + a3*T^3
_F1_WARM = [-4.732, -22.45, -0.6397, -0.01074]    # -22.9 <= T <= -2 °C
_F2_WARM = [0.08903, -0.01763, -5.330e-4, -8.801e-6]

# Cox & Weeks (1983) Table 3 cold-range coefficients (-30 <= T < -22.9 °C).
_F1_COLD = [9899.0, 1309.0, 55.27, 0.7160]
_F2_COLD = [8.547, 1.089, 4.518e-2, 5.819e-4]

# Leppäranta & Manninen (1988) coefficients for -2 < T <= 0 °C, the melt
# season, which the Cox & Weeks fits do not reach.  Same functional form and
# same density balance; verified coefficient-for-coefficient against SMRT
# v1.7.3 (``brine_volume_cox83_lepparanta88``).
_F1_MELT = [-4.1221e-2, -1.8407e1, 5.8402e-1, 2.1454e-1]
_F2_MELT = [9.0312e-2, -1.6111e-2, 1.2291e-4, 1.3603e-4]

_T_MELT_BRANCH = -2.0     # above this the Leppäranta & Manninen fit applies
_T_MAX = 0.0              # upper limit of the phase model


def _poly(coeffs, T):
    a0, a1, a2, a3 = coeffs
    return a0 + a1 * T + a2 * T**2 + a3 * T**3


def _cw_terms(T):
    """Cox & Weeks F1(T), F2(T) and pure-ice density, elementwise over *T*.

    Selects the warm (-22.9 <= T <= -2) or cold (T < -22.9) coefficient set
    per element, applies the -30 degC clamp to the cold branch, and returns
    ``(F1, F2, rho_i)`` with ``rho_i`` in g/cm3.

    Shared by :func:`compute_brine_volume` and :func:`compute_gas_volume` so
    the two cannot drift apart: in Cox & Weeks they are the brine and gas
    halves of one density balance, and they must see identical coefficients.
    """
    F1 = np.empty_like(T)
    F2 = np.empty_like(T)
    rho_i = np.empty_like(T)

    melt = T > _T_MELT_BRANCH
    cold = T < -22.9
    warm = ~melt & ~cold
    for mask, f1c, f2c in [(melt, _F1_MELT, _F2_MELT),
                           (warm, _F1_WARM, _F2_WARM),
                           (cold, _F1_COLD, _F2_COLD)]:
        if not np.any(mask):
            continue
        Tm = T[mask]
        # The CW83 cold polynomial is only fitted to -30 °C; extrapolating
        # the cubic below that is unstable, so clamp.  The clamp means values
        # returned below -30 °C are NOT Cox & Weeks values — they are the
        # -30 °C value held constant, and they diverge from an unclamped
        # evaluation by up to ~35% by -38 °C (verified against SMRT v1.7.3,
        # which refuses this range below -38 °C as unphysical).  Warn rather
        # than fail: callers in the configured retrieval ranges (T >= -30,
        # see inverse/optimize.py) never reach it.
        if f1c is _F1_COLD and np.any(Tm < -30.0):
            warnings.warn(
                f"Temperature below -30 °C (min={float(Tm.min()):.1f}) is "
                "outside the Cox & Weeks (1983) fitted range; brine volume "
                "is clamped to its -30 °C value and is not a Cox & Weeks "
                "result.",
                RuntimeWarning,
                stacklevel=3,
            )
            Tm = np.maximum(Tm, -30.0)
        elif f1c is _F1_COLD:
            Tm = np.maximum(Tm, -30.0)

        F1[mask] = _poly(f1c, Tm)
        F2[mask] = _poly(f2c, Tm)
        rho_i[mask] = 0.917 - 1.403e-4 * Tm

    return F1, F2, rho_i


def _validate_TS(salinity_psu, temperature_C):
    """Shared input validation and promotion to 1-D arrays."""
    T = np.asarray(temperature_C, dtype=float)
    S = np.asarray(salinity_psu, dtype=float)
    scalar = T.ndim == 0 and S.ndim == 0

    T = np.atleast_1d(T)
    S = np.atleast_1d(S)

    if np.any(S < 0):
        raise ValueError("salinity_psu must be non-negative")
    if np.any(T < -44) or np.any(T > _T_MAX):
        raise ValueError(
            f"temperature_C must be in [-44, {_T_MAX:g}]; "
            f"got min={T.min()}, max={T.max()}"
        )
    return T, S, scalar


def compute_brine_volume(salinity_psu, temperature_C, porosity=0.0,
                         bulk_density_kg_m3=None):
    """Compute brine volume fraction in sea ice via Cox & Weeks (1983).

    The air (gas) content of the ice may be supplied either as *porosity* or,
    equivalently and more usefully, as a measured *bulk_density_kg_m3*; the two
    are alternative statements of the same quantity and may not both be given.
    Leaving both at their defaults assumes bubble-free ice, which overestimates
    brine volume by the gas fraction — about 1-9% for the shipped presets.
    Prefer the bulk-density form wherever a density is already known, since
    that is the quantity an ice core actually yields.

    Args:
        salinity_psu: Bulk salinity in practical salinity units (0–15 typical
            for sea ice; must be non-negative).
        temperature_C: Ice temperature in degrees Celsius.  Must be in
            [-44, -2].  Positive or near-zero temperatures indicate melting
            ice, which is outside this model's validity range.  Note that
            Cox & Weeks is only fitted to [-30, -2]; inputs below -30 are
            clamped (see ``Known limitations``) and emit a RuntimeWarning.
        porosity: Gas (air) volume fraction of the ice, 0 to <1.  Default 0,
            i.e. bubble-free.  Enters as a simple ``(1 - porosity)`` scaling.
            Mutually exclusive with *bulk_density_kg_m3*.
        bulk_density_kg_m3: Measured bulk sea-ice density in kg/m3 (typically
            820-920).  When given, brine volume is taken straight from
            ``Vb = S * rho_si / F1`` -- the Cox & Weeks form for ice of known
            density -- and the gas content is implicit in that density.
            Mutually exclusive with *porosity*.

    Returns:
        Brine volume fraction (dimensionless, 0–~0.35 for typical sea ice).

    Raises:
        ValueError: If temperature is outside [-44, -2] or salinity is < 0.

    Verification:
        Cross-checked 2026-11-05 against SMRT v1.7.3's independent
        implementation of the same paper
        (``smrt.permittivity.brine.brine_volume_cox83_lepparanta88``, at zero
        gas porosity) over T in [-3, -30] degC and S in [1, 15] psu.  Agreement
        is 0.0326-0.0331% throughout, entirely accounted for by the pure-ice
        density constant: this module uses rho_i(0 degC) = 0.917 g/cm3 where
        SMRT uses 0.9167, a 0.0327% difference.  The implementation is
        therefore confirmed faithful to Cox & Weeks (1983).

        The published Table 1 check-points are reproduced to within 3%, and
        SMRT departs from them by the same margin -- the residual is rounding
        in the published table, not an error here.  (An earlier version of this
        note claimed a ~25-35% discrepancy attributed to "unit/convention
        ambiguity in the original paper"; the cross-check shows no such
        discrepancy and no such ambiguity.  Do not re-introduce a correction
        factor on the strength of it.)

    Known limitation:
        **Clamped below -30 degC.**  See the clamp note in ``_cw_terms``.
        Results between -30 and -44 degC are held constant and are not
        Cox & Weeks values; SMRT refuses this range outright below -38 degC.

    See also:
        :func:`compute_gas_volume` -- the gas half of the same Cox & Weeks
        density balance.  Given a bulk density, the two are consistent by
        construction; computing one with a density and the other without is
        the inconsistency this signature exists to remove.
    """
    T, S, scalar = _validate_TS(salinity_psu, temperature_C)

    if bulk_density_kg_m3 is not None:
        if np.any(np.asarray(porosity, dtype=float) != 0.0):
            raise ValueError(
                "pass either porosity or bulk_density_kg_m3, not both — one "
                "is deduced from the other"
            )
        rho_si = np.atleast_1d(np.asarray(bulk_density_kg_m3, dtype=float))
        if np.any(rho_si <= 0):
            raise ValueError("bulk_density_kg_m3 must be positive")
        T, S, rho_si = np.broadcast_arrays(T, S, rho_si)
        F1, _F2, _rho_i = _cw_terms(T)
        # Cox & Weeks (1983): Vb = S * rho_si / F1, with rho_si in g/cm3.
        # F1 -> 0 as T -> 0: the ice is past its own melting point and is
        # all brine.  _all_melted marks those elements.
        with np.errstate(divide="ignore", invalid="ignore"):
            nu_b = np.where(_all_melted(S, rho_si / 1e3, F1), 1.0,
                            S * (rho_si / 1e3) / F1)
    else:
        por = np.atleast_1d(np.asarray(porosity, dtype=float))
        if np.any(por < 0) or np.any(por >= 1):
            raise ValueError("porosity must be in [0, 1)")
        T, S, por = np.broadcast_arrays(T, S, por)
        F1, F2, rho_i = _cw_terms(T)
        denom = F1 - rho_i * S * F2
        melted = denom <= rho_i * S          # gas-free Vb would reach 1
        if np.any((denom <= 0) & (T <= _T_MELT_BRANCH)):
            raise RuntimeError(
                "Cox & Weeks denominator non-positive — outside the "
                "polynomial validity range."
            )
        # The gas term enters as a simple (1 - Va) scaling: the Cox & Weeks
        # bulk density carries the (1 - Va) factor and F1 does not, so it
        # survives into Vb unchanged.  Verified identical to the
        # bulk-density path to machine precision.
        with np.errstate(divide="ignore", invalid="ignore"):
            nu_b = np.where(melted, 1.0 - por,
                            (1.0 - por) * (S * rho_i) / denom)

    nu_b = np.clip(nu_b, 0.0, 1.0)
    return float(nu_b.ravel()[0]) if scalar else nu_b


def _all_melted(S, rho_g_cm3, F1):
    """True where ice of this salinity holds no solid phase at this
    temperature — i.e. the Cox & Weeks brine volume ``S * rho / F1`` would
    reach or pass 1, which happens as ``F1 -> 0`` on approach to 0 °C."""
    return (F1 <= 0) | (S * rho_g_cm3 >= F1)


def melting_temperature(salinity_psu):
    """Temperature (°C) at which ice of bulk salinity *salinity_psu* is fully
    melted — the liquidus temperature of water of that salinity.

    Uses the same Leppäranta & Manninen relation as the brine volume, with no
    solid salts: brine of salinity S_b has ``rho_b * S_b = F1(T)`` and the
    ice is all brine when S_b has fallen to the bulk salinity.  For S = 5 psu
    this is -0.27 °C, i.e. the familiar ~ -0.054 °C per psu.
    """
    from scipy.optimize import brentq

    S = float(salinity_psu)
    if S <= 0:
        return 0.0
    target = S * (1.0 + _BRINE_RHO_SLOPE * 1e-3 * S)
    f = lambda T: _poly(_F1_MELT, T) - target          # noqa: E731
    if f(_T_MELT_BRANCH) < 0:
        return _T_MELT_BRANCH       # saltier than this branch covers
    return float(brentq(f, _T_MELT_BRANCH, 0.0, xtol=1e-6))


def compute_gas_volume(salinity_psu, temperature_C, bulk_density_kg_m3):
    """Gas (air) volume fraction of sea ice via Cox & Weeks (1983).

    The companion to :func:`compute_brine_volume`.  Cox & Weeks give

        Va = 1 - rho_si / rho_i + rho_si * S * F2(T) / F1(T)

    with densities in g/cm3.  This is the same density balance that yields
    brine volume, so a (Vb, Va) pair computed here from one bulk density is
    mutually consistent by construction -- unlike deriving Va from a separate
    balance that uses a different brine-density model.

    Args:
        salinity_psu: Bulk salinity in psu (non-negative).
        temperature_C: Ice temperature in degrees Celsius, in [-44, -2].
        bulk_density_kg_m3: Measured bulk sea-ice density in kg/m3.  Densities
            above the gas-free Cox & Weeks density give a negative Va, which
            is clipped to 0 (ice cannot hold negative air).

    Returns:
        Gas volume fraction (dimensionless, 0-1).  Scalar in, scalar out.

    Raises:
        ValueError: If temperature or salinity are out of range, or the
            density is non-positive.
    """
    T, S, scalar = _validate_TS(salinity_psu, temperature_C)
    rho_si = np.atleast_1d(np.asarray(bulk_density_kg_m3, dtype=float))
    if np.any(rho_si <= 0):
        raise ValueError("bulk_density_kg_m3 must be positive")

    T, S, rho_si = np.broadcast_arrays(T, S, rho_si)
    F1, F2, rho_i = _cw_terms(T)

    rho_g = rho_si / 1e3            # kg/m3 -> g/cm3
    with np.errstate(divide="ignore", invalid="ignore"):
        nu_a = 1.0 - rho_g / rho_i + rho_g * S * F2 / F1
    # Fully melted: no ice matrix left to hold gas.
    nu_a = np.where(_all_melted(S, rho_g, F1), 0.0, nu_a)
    nu_a = np.clip(nu_a, 0.0, 1.0)
    return float(nu_a.ravel()[0]) if scalar else nu_a


def invert_brine_volume(brine_volume_fraction, salinity_psu,
                        T_lo=-22.0, T_hi=-2.1, bulk_density_kg_m3=None):
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
        bulk_density_kg_m3: Bulk sea-ice density in kg/m³, passed through to
            :func:`compute_brine_volume`.  **Pass the same density the forward
            model will use for this layer.**  Omitting it inverts the
            bubble-free relation, so the recovered temperature reproduces the
            requested Vb only in ice with no air — in porous ice the forward
            model would then compute a different Vb from the one asked for,
            and ``brine_volume_fraction`` would stop naming the brine volume.

    Returns:
        Temperature (°C) such that ``compute_brine_volume(salinity_psu, T)``
        equals ``brine_volume_fraction``.

    Raises:
        ValueError: If ``brine_volume_fraction`` is outside the range
            achievable within [T_lo, T_hi] at the given salinity.
    """
    from scipy.optimize import brentq

    def _fwd(T):
        return compute_brine_volume(
            salinity_psu, T, bulk_density_kg_m3=bulk_density_kg_m3
        )

    vb_lo = _fwd(T_lo)
    vb_hi = _fwd(T_hi)

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
        return _fwd(T) - vb

    return float(brentq(residual, T_lo, T_hi, xtol=1e-4))


# ── Brine salinity (liquidus) ────────────────────────────────────────────────
# Brine in sea ice sits at phase equilibrium, so its salinity is a function of
# temperature alone (the liquidus).  It has to come from the SAME phase model
# as the brine volume, because the two are tied by conservation of salt:
#
#       S_i * rho_si  =  S_b * rho_b * V_b * (1 + k)
#
# (S_i, rho_si bulk; S_b, rho_b brine; k the mass of precipitated salts per
# unit mass of salt left in the brine).  The Cox & Weeks brine volume is
# V_b = S_i * rho_si / F1(T), so their F1 *is* the liquidus:
#
#       F1(T) = rho_b * S_b * (1 + k),        rho_b = 1 + 0.0008 * S_b  g/cm3
#
# Above -8.2 degC nothing has precipitated (k = 0), and S_b follows from F1 by
# solving that quadratic — which makes brine salinity and brine volume
# conserve salt identically.  That is the branch every melt-season and
# most winter retrievals sit on.
#
# Below -8.2 degC mirabilite, and below -22.9 degC hydrohalite, take salt out
# of the brine, k > 0, and F1 alone no longer fixes S_b.  There the liquidus
# is the fit to Assur's (1958) phase data:
#
#   -22.9 <= T < -8.2   S_b = 57.041 - 9.929 T - 0.16204 T^2 - 0.002396 T^3
#                       (Cox & Weeks 1975, after Assur; the form SMRT uses)
#   T < -22.9           S_b = 206.24 - 1.8907 T - 0.060868 T^2 - 0.0010247 T^3
#                       (Cox & Weeks 1986)
#
# The pieces join well: at -8.2 degC the F1-derived value is 128.97 and the
# Assur fit 128.89 (0.06%).  At -22.9 degC the two published fits differ by
# 0.7% (228.2 vs 229.9); the cold fit is shifted by that constant so the
# liquidus is continuous.  At -2 degC the Cox & Weeks and Leppäranta &
# Manninen F1 polynomials themselves differ by 0.8%, and S_b inherits that
# step — it is the brine volume's own step, so salt is still conserved on
# both sides of it.
#
# This replaces a liquidus derived from the Frankenstein & Garner (1967)
# brine-volume relation, which disagreed with the Cox & Weeks volume it was
# used alongside by 1.7% at -2 degC rising to 13.9% at -22.9 degC: brine
# volume and brine salinity were drawn from two different phase models.
#
# References:
#   Assur, A. (1958). Composition of sea ice and its tensile strength.
#     Arctic Sea Ice, NAS-NRC Publ. 598, 106-138.
#   Cox, G. F. N. & Weeks, W. F. (1975). Brine drainage and initial salt
#     entrapment in sodium chloride ice. CRREL Research Report 345.
#   Cox, G. F. N. & Weeks, W. F. (1986). Changes in the salinity and porosity
#     of sea-ice samples during shipping and storage. J. Glaciol. 32(112).
#   Leppäranta, M. & Manninen, T. (1988). The brine and gas content of sea
#     ice with attention to low salinities and high temperatures. Finnish
#     Institute of Marine Research Internal Report 88-2.
_BRINE_RHO_SLOPE = 0.8    # rho_b = 1000 + slope * S_b  (kg m^-3), Cox & Weeks
_T_MIRABILITE = -8.2      # degC — onset of mirabilite precipitation
_T_HYDROHALITE = -22.9    # degC — onset of hydrohalite precipitation
_SB_ASSUR = [57.041, -9.929, -0.16204, -0.002396]
_SB_COLD = [206.24, -1.8907, -0.060868, -0.0010247]


def _sb_from_f1(F1):
    """Brine salinity from ``rho_b * S_b = F1`` with rho_b = 1 + 0.0008 S_b."""
    c = _BRINE_RHO_SLOPE * 1e-3
    return (-1.0 + np.sqrt(1.0 + 4.0 * c * np.maximum(F1, 0.0))) / (2.0 * c)


_SB_COLD_OFFSET = (_poly(_SB_ASSUR, _T_HYDROHALITE)
                   - _poly(_SB_COLD, _T_HYDROHALITE))


def brine_salinity_at_temp(temperature_C):
    """Brine (liquidus) salinity in psu as a function of ice temperature (°C).

    Consistent with :func:`compute_brine_volume`: above -8.2 °C it is the
    salinity for which the Cox & Weeks / Leppäranta & Manninen brine volume
    conserves salt exactly; below, Assur's phase data.  See the module notes
    above for the derivation, the joins and the references.  Valid 0 to
    -30 °C; held at its -30 °C value below that, as the brine volume is.
    """
    T = np.atleast_1d(np.asarray(temperature_C, dtype=float))
    T = np.clip(T, -30.0, _T_MAX)
    S_b = np.empty_like(T)

    melt = T > _T_MELT_BRANCH
    warm = (T <= _T_MELT_BRANCH) & (T >= _T_MIRABILITE)
    mid = (T < _T_MIRABILITE) & (T >= _T_HYDROHALITE)
    cold = T < _T_HYDROHALITE
    S_b[melt] = _sb_from_f1(_poly(_F1_MELT, T[melt]))
    S_b[warm] = _sb_from_f1(_poly(_F1_WARM, T[warm]))
    S_b[mid] = _poly(_SB_ASSUR, T[mid])
    S_b[cold] = _poly(_SB_COLD, T[cold]) + _SB_COLD_OFFSET
    S_b = np.maximum(S_b, 0.0)
    return float(S_b[0]) if np.ndim(temperature_C) == 0 else S_b


# ── Precipitated salts ───────────────────────────────────────────────────────
# Two salts matter optically (Light, Maykut & Grenfell 2004): mirabilite,
# Na2SO4.10H2O, below -8.2 degC, and hydrohalite, NaCl.2H2O, below -22.9 degC.
#
# _MIRABILITE_MAX: kg of mirabilite per kg of sea salt if all the sulphate
# precipitates — 7.7% SO4 by mass is 11.4% as Na2SO4, times 322.2/142.0 as
# the decahydrate.
_MIRABILITE_MAX = 0.259
_MIRABILITE_EFOLD = 5.05  # K
_HYDROHALITE_PER_NACL = 94.47 / 58.44    # kg NaCl.2H2O per kg NaCl


def mirabilite_mass_fraction(temperature_C):
    """Mass of precipitated mirabilite per unit mass of sea salt.

    Zero above -8.2 °C, rising towards the total-sulphate limit of 0.259 as
    the ice cools.  The shape is a saturating exponential with one anchor
    from the literature: 9e-4 kg of mirabilite per kg of 4.7 ppt sea ice at
    -15 °C (Light et al. 2004, from Richardson 1976), i.e. 0.191 per kg of
    salt, 74% of the sulphate.  It reaches 94% by -22.9 °C.

    This is an interpolation, not a phase calculation: Assur's and
    Richardson's tables were not available to fit.  It matters little —
    mirabilite in brine has almost no refractive-index contrast and
    contributes under 5% of the similarity parameter at any temperature.
    """
    T = np.asarray(temperature_C, dtype=float)
    dT = np.maximum(_T_MIRABILITE - T, 0.0)
    out = _MIRABILITE_MAX * (1.0 - np.exp(-dT / _MIRABILITE_EFOLD))
    return float(out) if np.ndim(temperature_C) == 0 else out


def _salt_fraction_in_brine_cold(T):
    """Fraction of the bulk salt still dissolved, on the Cox & Weeks cold
    branch: ``rho_b * S_b / F1`` = 1 / (1 + k)."""
    T = np.clip(T, -30.0, _T_HYDROHALITE)
    S_b = _poly(_SB_COLD, T)
    return (1.0 + _BRINE_RHO_SLOPE * 1e-3 * S_b) * S_b / _poly(_F1_COLD, T)


def hydrohalite_mass_fraction(temperature_C):
    """Mass of precipitated hydrohalite per unit mass of sea salt.

    Zero above -22.9 °C.  Below it, the salt that leaves the brine as the
    Cox & Weeks cold-branch brine volume collapses is sodium chloride, and
    it precipitates as the dihydrate.  Computed entirely from the cold-branch
    fits (brine volume and liquidus), so it is consistent with the brine
    volume the model uses and is exactly zero at the onset.  By -30 °C about
    60% of the salt has left the brine and the hydrohalite outweighs the
    original salt.  Held at its -30 °C value below that.
    """
    T = np.asarray(temperature_C, dtype=float)
    lost = (_salt_fraction_in_brine_cold(np.full_like(T, _T_HYDROHALITE))
            - _salt_fraction_in_brine_cold(T))
    out = np.where(T < _T_HYDROHALITE,
                   np.maximum(lost, 0.0) * _HYDROHALITE_PER_NACL, 0.0)
    return float(out) if np.ndim(temperature_C) == 0 else out


def brine_density(temperature_C):
    """Approximate brine density (g/cm³) from temperature.

    Uses the linear brine-density model ρ_b ≈ 1.0 + 0.0008 × S_b (g/cm³) — the
    same model used self-consistently inside :func:`brine_salinity_at_temp`'s
    salt-balance derivation — with S_b the liquidus salinity at *temperature_C*.

    Note:
        Not used by the forward model (the air fraction comes from
        :func:`compute_gas_volume`).  Retained as a public utility.  S_b is
        the Cox & Weeks-consistent liquidus of
        :func:`brine_salinity_at_temp`, so this is the brine density of the
        same phase model as the brine volume.
    """
    S_b = brine_salinity_at_temp(temperature_C)
    return 1.0 + 0.0008 * S_b
