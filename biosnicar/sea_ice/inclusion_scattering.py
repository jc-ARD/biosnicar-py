"""Scattering by brine inclusions and precipitated salts in sea ice.

Air bubbles are not the only thing in sea ice that scatters light.  The brine
itself sits in pockets and tubes whose refractive index differs from the ice
around them by 2-6%, and below -8.2 °C and -22.9 °C salts crystallise out of
that brine.  This module supplies those terms, following the
structural-optical model of Light, Maykut & Grenfell (2004), which was built
from simultaneous measurements of microstructure and optical properties on
first-year ice between -33 and -1 °C.

How scattering is computed
--------------------------
Every scatterer here is much larger than the wavelength or optically soft, and
is reduced to *equivalent spheres* (Grenfell & Warren 1999): a collection of
spheres with the same total volume and the same total surface area.  For such
a collection the number-weighted cross-sectional area per unit volume is

    psi = A / 4                    (m^-1; A = surface area per unit volume)

— the mean projected area of randomly oriented convex bodies — and the
equivalent radius is ``r_eq = 3 V / A``.  The scattering coefficient is
``sigma = Q_sca * psi`` and the asymmetry parameter ``g`` is that of a sphere
of radius ``r_eq``, both from the soft-sphere Mie table
(``scripts/build_soft_sphere_mie_lut.py``) at the relative refractive index
of the scatterer in its surroundings.  Light et al. take ``Q_sca = 2``; the
table gives that for brine inclusions and less for micron-sized crystals,
whose phase shift is small.

Brine inclusions
----------------
The reference microstructure is the size distribution Light et al. (2003a)
measured at -15 °C, in inclusion length ``l`` (mm):

    N(l) = 0.28 l^-1.96  mm^-3 per mm,        l >= 0.01 mm
    aspect ratio  gamma(l) = 10.3 l^0.67  (1 <= gamma <= 70),  1 for l <= 0.03

Inclusions shorter than 0.5 mm are *pockets* (prolate ellipsoids), longer
ones *tubes* (cylinders).  This module reproduces their published numbers
from that distribution: 24 inclusions per mm^3, and for l up to 8 mm a brine
volume of 1.2% with psi = 28 m^-1 from pockets and 82 m^-1 from tubes
(published: 30 and 80).

The distribution is applied to other ice as they applied it:

* **Other salinities** — the upper length limit is extended (or reduced)
  until the distribution holds the brine volume that ice has at -15 °C.  More
  brine means longer tubes, with the population of small inclusions
  unchanged.
* **Other temperatures** — every inclusion's volume changes in proportion to
  the bulk brine volume, as freezing equilibrium requires.  Pockets keep
  their aspect ratio; tubes keep their length and change diameter.
* **Merging** — above -14 °C neighbouring inclusions coalesce.  The number
  of inclusions of volume ``v`` is reduced by ``eta(v, T)``, log-linear in
  ``v`` from 1 at the smallest inclusion to
  ``eta_max = 0.0307 - 0.0693 T`` at the largest, and the distribution is
  extended to larger sizes along its own slope until brine volume is
  conserved.

Limits of validity: the model was fitted to interior first-year ice of 4.7
and 15 ppt.  It is used here for any ice, including multi-year ice whose
microstructure is not known to follow it, and it is a description of
inclusions *in an ice matrix* — above a brine fraction of 0.5 the geometry is
held at its 0.5 form, since past that point the ice is the inclusion.

Precipitated salts
------------------
Mirabilite (below -8.2 °C) and hydrohalite (below -22.9 °C) are treated as
crystals of fixed edge length inside the brine, with ``psi = 1.5 V / a`` for
a cube of edge ``a``.  Their masses come from
:mod:`biosnicar.sea_ice.brine_volume`.  The edge lengths are Light et al.'s:
10 um for mirabilite and 4.5 um for hydrohalite, the latter chosen by them to
match observed scattering in cold ice under ``Q_sca = 2``.  With the Mie
efficiency used here hydrohalite scatters less than their model at the same
size, so the cold regime (T < -22.9 °C) is the least certain part of this
module.  Refractive indices: mirabilite 1.396; hydrohalite 1.416, which is
not well determined in the literature and is chosen to reproduce the relative
index of 1.02-1.03 in brine shown in their Figure 7.

References
----------
Light, B., Maykut, G. A. & Grenfell, T. C. (2003a). Effects of temperature on
    the microstructure of first-year Arctic sea ice. J. Geophys. Res. 108(C2),
    3051. doi:10.1029/2001JC000887
Light, B., Maykut, G. A. & Grenfell, T. C. (2004). A temperature-dependent,
    structural-optical model of first-year sea ice. J. Geophys. Res. 109,
    C06013. doi:10.1029/2003JC002164
Grenfell, T. C. & Warren, S. G. (1999). Representation of a nonspherical ice
    particle by a collection of independent spheres for scattering and
    absorption of radiation. J. Geophys. Res. 104(D24), 31697-31709.
"""

from functools import lru_cache

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import biosnicar
from biosnicar.sea_ice.brine_volume import (
    compute_brine_volume, hydrohalite_mass_fraction, mirabilite_mass_fraction,
)

_MIE_PATH = (biosnicar.DATA_DIR / "OP_data" / "480band" / "luts"
             / "soft_sphere_mie.npz")

# ── Light et al. (2003a, 2004) microstructure ────────────────────────────────
T_REFERENCE = -15.0        # degC at which the size distribution was measured
_N0, _N_EXP = 0.28, -1.96  # N(l) = N0 * l**N_EXP, mm^-3 per mm
_L_MIN = 0.01              # mm
_L_POCKET = 0.5            # mm — pockets below, tubes above
_T_MERGE = -14.0           # degC — merging observed above this
_NU_B_TOPOLOGY_LIMIT = 0.5

# Salt crystals
MIRABILITE_EDGE_UM = 10.0
HYDROHALITE_EDGE_UM = 4.5
MIRABILITE_DENSITY = 1464.0     # kg m^-3
HYDROHALITE_DENSITY = 1630.0    # kg m^-3
N_MIRABILITE = 1.396
N_HYDROHALITE = 1.416

# Master grid of inclusion lengths (mm) for the reference distribution.
_L_GRID = np.geomspace(_L_MIN, 1.0e5, 24000)


def _aspect_ratio(l_mm):
    return np.where(l_mm <= 0.03, 1.0,
                    np.clip(10.3 * l_mm ** 0.67, 1.0, 70.0))


def _ellipsoid_area(a, b):
    """Surface area of a prolate ellipsoid, semi-axes a >= b."""
    e = np.sqrt(np.clip(1.0 - (b / a) ** 2, 0.0, 1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        prolate = 2.0 * np.pi * b ** 2 * (1.0 + (a / (b * e)) * np.arcsin(e))
    return np.where(e > 1e-6, prolate, 4.0 * np.pi * a ** 2)


def _inclusion_geometry(l_mm, volume_ratio=1.0):
    """Volume (mm^3) and surface area (mm^2) of inclusions of reference
    length *l_mm* after their volume has changed by *volume_ratio*.

    Pockets keep their aspect ratio (all dimensions scale as ratio^(1/3));
    tubes keep their length and change diameter (as ratio^(1/2)).
    """
    d = l_mm / _aspect_ratio(l_mm)
    pocket = l_mm < _L_POCKET

    s = volume_ratio ** (1.0 / 3.0)
    a, b = 0.5 * l_mm * s, 0.5 * d * s
    v_pocket = 4.0 / 3.0 * np.pi * a * b ** 2
    a_pocket = _ellipsoid_area(a, b)

    d_t = d * np.sqrt(volume_ratio)
    v_tube = 0.25 * np.pi * d_t ** 2 * l_mm
    a_tube = np.pi * d_t * l_mm + 0.5 * np.pi * d_t ** 2

    return (np.where(pocket, v_pocket, v_tube),
            np.where(pocket, a_pocket, a_tube), pocket)


def _trapz_cumulative(y, x):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


_V_REF, _A_REF, _ = _inclusion_geometry(_L_GRID)
_CUM_V_REF = _trapz_cumulative(_N0 * _L_GRID ** _N_EXP * _V_REF, _L_GRID)


def _reference_grid(nu_b_ref):
    """Inclusion lengths (mm) of the -15 °C distribution holding *nu_b_ref*."""
    nu = min(float(nu_b_ref), float(_CUM_V_REF[-1]))
    l_max = float(np.exp(np.interp(nu, _CUM_V_REF, np.log(_L_GRID))))
    l_max = max(l_max, _L_MIN * 1.0001)
    grid = _L_GRID[_L_GRID < l_max]
    return np.append(grid, l_max)


def brine_inclusion_area(nu_b, nu_b_ref, temperature_C, merging=True):
    """Cross-sectional area and equivalent radius of the brine inclusions.

    Args:
        nu_b: Brine volume fraction of the layer at *temperature_C*.
        nu_b_ref: Brine volume fraction the same ice has at -15 °C
            (:data:`T_REFERENCE`) — this fixes the size distribution.
        temperature_C: Layer temperature, used for merging.
        merging: Apply the inclusion-merging rule above -14 °C.

    Returns:
        ``(psi, r_eq)``: number-weighted cross-sectional area per unit volume
        (m^-1) and equivalent-sphere radius (um).  ``(0, 0)`` without brine.
    """
    nu_b = min(float(nu_b), _NU_B_TOPOLOGY_LIMIT)
    nu_b_ref = float(nu_b_ref)
    if nu_b <= 0.0 or nu_b_ref <= 0.0:
        return 0.0, 0.0

    l = _reference_grid(nu_b_ref)
    n = _N0 * l ** _N_EXP
    # Normalise on the grid actually used, so the population holds exactly
    # nu_b at this temperature.
    v0, _, _ = _inclusion_geometry(l)
    ratio = nu_b / np.trapezoid(n * v0, l)
    v, area, _ = _inclusion_geometry(l, ratio)

    vol = np.trapezoid(n * v, l)
    surf = np.trapezoid(n * area, l)

    T = float(temperature_C)
    if merging and T > _T_MERGE and l.size > 2 and v[-1] > v[0]:
        eta_max = float(np.clip(0.0307 - 0.0693 * T, 0.01, 1.0))
        eta = np.exp(np.log(eta_max) * np.log(v / v[0]) / np.log(v[-1] / v[0]))
        n_m = eta * n
        deficit = np.trapezoid((1.0 - eta) * n * v, l)

        # Extend the merged distribution past l_max along its own slope
        # until the brine volume removed by merging is restored.
        slope = np.log(n_m[-1] / n_m[-2]) / np.log(l[-1] / l[-2])
        l_x = np.geomspace(l[-1], l[-1] * 1.0e4, 4000)
        n_x = n_m[-1] * (l_x / l[-1]) ** slope
        v_x, a_x, _ = _inclusion_geometry(l_x, ratio)
        cum_v = _trapz_cumulative(n_x * v_x, l_x)
        cum_a = _trapz_cumulative(n_x * a_x, l_x)
        extra_a = float(np.interp(min(deficit, cum_v[-1]), cum_v, cum_a))

        surf = np.trapezoid(n_m * area, l) + extra_a

    psi = 0.25 * surf * 1.0e3                 # mm^-1 -> m^-1
    r_eq = 3.0 * vol / surf * 1.0e3           # mm -> um
    return float(psi), float(r_eq)


def salt_crystal_area(mass_per_kg_salt, salinity_psu, bulk_density_kg_m3,
                      crystal_density, edge_um):
    """Cross-sectional area per unit volume (m^-1) of precipitated crystals.

    The crystals are cubes of edge *edge_um*; for a cube the surface area is
    6 a^2 and the mean projected area A / 4, so ``psi = 1.5 V / a`` with V the
    crystal volume fraction.
    """
    vol = (float(mass_per_kg_salt) * float(salinity_psu) * 1e-3
           * float(bulk_density_kg_m3) / float(crystal_density))
    return 1.5 * vol / (float(edge_um) * 1e-6)


# ── Soft-sphere Mie table ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _mie_table():
    data = np.load(str(_MIE_PATH))
    axes = (data["m_grid"], np.log(data["x_grid"]))
    kw = dict(method="linear", bounds_error=False, fill_value=None)
    return (RegularGridInterpolator(axes, data["q_sca"], **kw),
            RegularGridInterpolator(axes, data["asm"], **kw),
            (float(data["m_grid"][0]), float(data["m_grid"][-1])),
            (float(data["x_grid"][0]), float(data["x_grid"][-1])))


def soft_sphere_scattering(m_rel, x):
    """``(Q_sca, g)`` of a sphere of relative index *m_rel*, size parameter *x*.

    Depends on the index contrast ``|m_rel - 1|`` only, so an inclusion
    optically thinner than its surroundings is evaluated at ``2 - m_rel``.
    Below the table's smallest contrast ``Q_sca`` is continued with its
    Rayleigh-Gans scaling, as the square of the contrast, so that a
    scatterer with no contrast does not scatter.
    """
    q_tab, g_tab, (m_lo, m_hi), (x_lo, x_hi) = _mie_table()
    contrast = np.abs(np.asarray(m_rel, dtype=float) - 1.0)
    x = np.asarray(x, dtype=float)
    contrast, x = np.broadcast_arrays(contrast, x)

    m_eval = np.clip(1.0 + contrast, m_lo, m_hi)
    pts = np.stack([m_eval, np.log(np.clip(x, x_lo, x_hi))], axis=-1)
    q, g = q_tab(pts), g_tab(pts)

    weak = contrast < (m_lo - 1.0)
    q = np.where(weak, q * (contrast / (m_lo - 1.0)) ** 2, q)
    return np.clip(q, 0.0, None), np.clip(g, -1.0, 1.0)


# ── Layer-level terms ────────────────────────────────────────────────────────

def inclusion_scattering(salinity_psu, temperature_C, density_kg_m3, nu_b,
                         n_ice, n_brine, wavelengths_um):
    """Scattering by brine inclusions and precipitated salts in one layer.

    Args:
        salinity_psu, temperature_C, density_kg_m3: Layer state.
        nu_b: Brine volume fraction of the layer.
        n_ice, n_brine: Real refractive indices on *wavelengths_um*.
        wavelengths_um: Wavelength grid (um).

    Returns:
        dict ``{name: (sigma, g)}`` for ``"brine"``, ``"mirabilite"`` and
        ``"hydrohalite"`` — scattering coefficient (m^-1) and asymmetry
        parameter per wavelength.
    """
    lam = np.asarray(wavelengths_um, dtype=float)
    n_ice = np.asarray(n_ice, dtype=float)
    n_brine = np.asarray(n_brine, dtype=float)
    zeros = np.zeros_like(lam)
    out = {}

    # Brine inclusions in ice
    S, rho = float(salinity_psu), float(density_kg_m3)
    nu_ref = (compute_brine_volume(S, T_REFERENCE, bulk_density_kg_m3=rho)
              if S > 0 else 0.0)
    psi, r_eq = brine_inclusion_area(nu_b, nu_ref, temperature_C)
    if psi > 0:
        q, g = soft_sphere_scattering(n_brine / n_ice,
                                      2.0 * np.pi * r_eq * n_ice / lam)
        out["brine"] = (q * psi, g)
    else:
        out["brine"] = (zeros, zeros)

    # Salt crystals in brine
    for name, mass, dens, edge, n_x in (
        ("mirabilite", mirabilite_mass_fraction(temperature_C),
         MIRABILITE_DENSITY, MIRABILITE_EDGE_UM, N_MIRABILITE),
        ("hydrohalite", hydrohalite_mass_fraction(temperature_C),
         HYDROHALITE_DENSITY, HYDROHALITE_EDGE_UM, N_HYDROHALITE),
    ):
        psi = salt_crystal_area(mass, S, rho, dens, edge) if mass > 0 else 0.0
        if psi > 0:
            # Equivalent radius of a cube: 3V/A = a/2.
            q, g = soft_sphere_scattering(
                n_x / n_brine, 2.0 * np.pi * (0.5 * edge) * n_brine / lam)
            out[name] = (q * psi, g)
        else:
            out[name] = (zeros, zeros)
    return out
