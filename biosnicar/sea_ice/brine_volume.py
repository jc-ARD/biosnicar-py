"""Brine and gas volume fractions in sea ice via Cox & Weeks (1983).

Cox & Weeks give brine volume and gas (air) volume as the two halves of a
single density balance.  Both are provided here and both route through
:func:`_cw_terms`, so they cannot drift onto different coefficient sets:

* :func:`compute_brine_volume` — nu_b, from (S, T) and *either* a porosity or
  a bulk density.  With neither it falls back to the gas-free form, which
  overestimates nu_b by the gas fraction (1–9% across the shipped presets).
* :func:`compute_gas_volume` — nu_air, from (S, T) and bulk density.
* :func:`invert_brine_volume` — T from a target nu_b, used by the emulator
  transforms.  Takes the layer density too, so the sampled
  ``brine_volume_fraction`` names the brine volume the forward model will
  actually compute.

Verified against SMRT v1.7.3's independent implementation: the porosity path
agrees to 0.033% (the residual is the pure-ice density constant, 0.917 vs
0.9167 g/cm3), the bulk-density path exactly.  ``tests/test_brine_volume.py``
pins both against hard-coded SMRT values, with no SMRT dependency added.

:func:`brine_density` is a separate Frankenstein & Garner (1967) liquidus and
is **no longer used by the forward model** — the air fraction now comes from
the Cox & Weeks gas equation instead.  See its docstring.

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

    warm = T >= -22.9
    for mask, f1c, f2c in [(warm, _F1_WARM, _F2_WARM), (~warm, _F1_COLD, _F2_COLD)]:
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
    if np.any(T < -44) or np.any(T > -2):
        raise ValueError(
            f"temperature_C must be in [-44, -2]; got min={T.min()}, max={T.max()}"
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
        nu_b = S * (rho_si / 1e3) / F1
    else:
        por = np.atleast_1d(np.asarray(porosity, dtype=float))
        if np.any(por < 0) or np.any(por >= 1):
            raise ValueError("porosity must be in [0, 1)")
        T, S, por = np.broadcast_arrays(T, S, por)
        F1, F2, rho_i = _cw_terms(T)
        denom = F1 - rho_i * S * F2
        if np.any(denom <= 0):
            raise RuntimeError(
                "Cox & Weeks denominator non-positive — outside the "
                "polynomial validity range."
            )
        # The gas term enters as a simple (1 - Va) scaling: the Cox & Weeks
        # bulk density carries the (1 - Va) factor and F1 does not, so it
        # survives into Vb unchanged.  Verified identical to the
        # bulk-density path to machine precision.
        nu_b = (1.0 - por) * (S * rho_i) / denom

    nu_b = np.clip(nu_b, 0.0, 1.0)
    return float(nu_b.ravel()[0]) if scalar else nu_b


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
    nu_a = 1.0 - rho_g / rho_i + rho_g * S * F2 / F1
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
# temperature alone (the liquidus).  We derive S_b(T) from the Frankenstein &
# Garner (1967) brine-VOLUME relation by salt mass balance, rather than the
# linear "warm-ice" law S_b ≈ -T/0.05411 ≈ -18.5·T of Notz et al. (2005), which
# overestimates brine salinity by ~30-40% below about -8 °C.
#
# Reference (brine volume):
#   Frankenstein, G. & Garner, R. (1967). Equations for determining the brine
#   volume of sea ice from -0.5° to -22.9 °C. Journal of Glaciology, 6(48),
#   943–944. https://doi.org/10.3189/S0022143000020244
#
# Derivation (honest about its assumptions — this is a *derived* liquidus, not
# a direct fit to Assur (1958) phase data such as Notz & Worster (2009)):
#   F&G give the relative brine volume of ice of bulk salinity S_i:
#       Vb = 1e-3 · S_i · (-49.185/T + 0.532)          [-22.9 ≤ T ≤ -0.5 °C]
#   All salt resides in the brine, so salt mass balance gives
#       S_i · ρ_si = S_b · ρ_b · Vb
#   in which S_i cancels (S_b is independent of bulk salinity, as physics
#   requires), leaving, with bulk/pure ice density ρ_si = 917 kg m⁻³ and the
#   linear brine-density model ρ_b = 1000 + 0.8·S_b kg m⁻³ (see brine_density),
#       0.8·c·S_b² + 1000·c·S_b − 1000·ρ_si = 0,   c = (-49.185/T + 0.532)
#   solved for the physical (positive) root.
#
# Accuracy (verified against accepted Assur-based brine salinities):
#   -2 °C → ~36, -5 °C → ~83, -10 °C → ~150 psu — within a few psu of the
#   accepted liquidus (the linear law gives 37, 94, 187: ~30% high at -10 °C).
#   Below ~-18 °C the derived value saturates against the NaCl·2H₂O eutectic
#   (23.3 wt% ≈ 233 psu near -22.9 °C, the F&G lower validity limit), where
#   inverting a volume fit is least reliable; it is capped and held there.
_FG_A = -49.185       # Frankenstein & Garner (1967) coefficient (1/°C)
_FG_B = 0.532         # Frankenstein & Garner (1967) coefficient
_RHO_ICE = 917.0      # pure ice density, kg m⁻³
_BRINE_RHO_SLOPE = 0.8  # ρ_b = 1000 + slope·S_b  (kg m⁻³); see brine_density()
_S_EUTECTIC = 233.0   # psu — NaCl·2H₂O eutectic brine salinity (~23.3 wt%, -22.9 °C)
_T_FG_MIN = -22.9     # °C — F&G lower validity / hydrohalite eutectic
_T_FG_MAX = -0.5      # °C — F&G upper validity


def brine_salinity_at_temp(temperature_C):
    """Brine (liquidus) salinity in psu as a function of ice temperature (°C).

    Derived from the Frankenstein & Garner (1967) brine-volume relation by
    salt mass balance (see module notes above for the full derivation,
    citation, assumptions, and accuracy).  Valid -0.5 to -22.9 °C; held at the
    ~233 psu NaCl·2H₂O eutectic below -22.9 °C.  Replaces the previous linear
    warm-ice approximation, removing its ~30 % cold-ice overestimate.
    """
    T = np.asarray(temperature_C, dtype=float)
    # Clamp into F&G validity; below the eutectic the brine is held constant.
    T_eval = np.clip(T, _T_FG_MIN, _T_FG_MAX)
    c = _FG_A / T_eval + _FG_B                      # > 0 for T_eval < 0
    a = _BRINE_RHO_SLOPE * c
    b = 1000.0 * c
    disc = b * b + 4.0 * a * 1000.0 * _RHO_ICE
    S_b = (-b + np.sqrt(disc)) / (2.0 * a)
    S_b = np.clip(S_b, 0.0, _S_EUTECTIC)
    return float(S_b) if np.ndim(temperature_C) == 0 else S_b


def brine_density(temperature_C):
    """Approximate brine density (g/cm³) from temperature.

    Uses the linear brine-density model ρ_b ≈ 1.0 + 0.0008 × S_b (g/cm³) — the
    same model used self-consistently inside :func:`brine_salinity_at_temp`'s
    salt-balance derivation — with S_b the liquidus salinity at *temperature_C*.

    Note:
        No longer used by the forward model.  It previously supplied the brine
        density for the sea-ice air-fraction balance; that balance has been
        replaced by :func:`compute_gas_volume`, which is pure Cox & Weeks and
        needs no separate brine density.  Because S_b here comes from
        Frankenstein & Garner (1967) while brine volume comes from
        Cox & Weeks (1983), mixing this into a Cox & Weeks calculation
        reintroduces two phase models for one quantity.  Retained as a public
        utility and for :mod:`brine_optics`, which uses the same liquidus.
    """
    S_b = brine_salinity_at_temp(temperature_C)
    return 1.0 + 0.0008 * S_b
