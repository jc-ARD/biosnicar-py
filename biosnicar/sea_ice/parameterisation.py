"""Optical parameterisations as alternatives to the physical ones.

Radius and density are individually degenerate: the measurement constrains
their combination, the specific surface area of the air/ice interface.
Likewise temperature and salinity act on the optics only through the brine
volume fraction.  This module lets a caller work in whichever set is natural
and converts to the canonical model inputs.

Two SSA conventions, and the distinction matters
------------------------------------------------
"Specific surface area" means interface area per unit mass in both cases, but
the geometry differs by topology and so does the formula:

* **granular** (``layer_type=0`` — grains of ice in air, e.g. snow and the
  SSL)::

      SSA = 3 / (r * rho_ice)

  Independent of the layer's *bulk* density: the mass is the mass of the
  grains themselves.  This is the standard snow-science definition.

* **bubbly** (``layer_type`` 1, 2 and 4 — air inclusions in a continuous ice
  matrix)::

      SSA = 3 * phi / (r * rho)      with  phi = 1 - rho / rho_ice

  Here the surface belongs to the bubbles and the mass to the surrounding
  ice, so bulk density enters twice.

The two differ by a factor ``phi * rho_ice / rho`` — about 2x at a snow
density of 300 kg/m3, and the absolute values differ by two orders of
magnitude between a snow layer (~10 m2/kg) and dense sea ice (~0.5 m2/kg).
Applying the wrong one silently mis-converts the layer, which is why these
functions require the topology rather than guessing it.

Why SSA is the better coordinate
--------------------------------
In ``sea_ice_optics`` scattering is formed as ``sca_cff_vlm(r) * nu_air`` and
absorption as ``(1 - nu_air) * alpha_ice``.  Form the single-scattering
albedo and the density cancels, leaving omega a function of SSA alone — and
omega governs the albedo of an optically thick layer.  Measured on the forward
model along true iso-SSA curves, holding SSA fixed removes 88% of the NIR
spectral variation and 61% of the VIS.

The VIS shortfall is not noise: the asymmetry parameter depends on radius
*alone*, and where ice barely absorbs (omega -> 1) that is what carries the
signal.  So SSA is a better-conditioned coordinate, not a complete one.
"""

from typing import Optional

import numpy as np

RHO_ICE = 917.0        # kg m-3, consistent with inverse.result._RHO_ICE

#: ``layer_type`` values whose scattering elements are air inclusions in ice.
BUBBLY_LAYER_TYPES = frozenset({1, 2, 4})
#: ``layer_type`` values whose scattering elements are discrete ice grains.
GRANULAR_LAYER_TYPES = frozenset({0})


def topology_for_layer_type(layer_type: int) -> str:
    """``'granular'`` or ``'bubbly'`` for *layer_type*.

    Raises ValueError for layer types where SSA is not defined: 3 (ice and
    water spheres mixed by volume), 5 (melt pond — liquid water) and 6 (young
    ice, a two-stream slab with no explicit inclusion size).
    """
    lt = int(layer_type)
    if lt in GRANULAR_LAYER_TYPES:
        return "granular"
    if lt in BUBBLY_LAYER_TYPES:
        return "bubbly"
    raise ValueError(
        f"specific surface area is not defined for layer_type={lt}. "
        "It applies to granular layers (0) and bubbly-ice layers (1, 2, 4); "
        "layer_type 3 (ice+water spheres), 5 (melt pond) and 6 (young-ice "
        "slab) have no single inclusion radius to convert."
    )


def specific_surface_area(radius_um: float, topology: str,
                          density_kg_m3: Optional[float] = None) -> float:
    """Specific surface area (m2 kg-1) of the air/ice interface.

    Args:
        radius_um: Grain radius (granular) or bubble radius (bubbly), in um.
        topology: ``'granular'`` or ``'bubbly'`` — see the module docstring;
            use :func:`topology_for_layer_type` to derive it.
        density_kg_m3: Layer bulk density.  Required for ``'bubbly'``,
            ignored for ``'granular'`` (where SSA does not depend on it).

    Returns:
        SSA in m2 kg-1.
    """
    r_m = float(radius_um) * 1e-6
    if r_m <= 0:
        raise ValueError(f"radius_um must be > 0, got {radius_um!r}")
    if topology == "granular":
        return 3.0 / (r_m * RHO_ICE)
    if topology == "bubbly":
        if density_kg_m3 is None:
            raise ValueError(
                "density_kg_m3 is required for bubbly topology: SSA = "
                "3*(1 - rho/917)/(r*rho) depends on it twice."
            )
        rho = float(density_kg_m3)
        _check_bubbly_density(rho)
        return 3.0 * (1.0 - rho / RHO_ICE) / (r_m * rho)
    raise ValueError(
        f"topology must be 'granular' or 'bubbly', got {topology!r}")


def radius_from_ssa(ssa: float, topology: str,
                    density_kg_m3: Optional[float] = None) -> float:
    """Invert :func:`specific_surface_area` for the radius, in um.

    The inverse is exact and unique *given the density* — SSA alone is one
    equation in two unknowns, which is why the bubbly form takes a density
    rather than returning a pair.
    """
    ssa = float(ssa)
    if not np.isfinite(ssa) or ssa <= 0:
        raise ValueError(f"ssa must be finite and > 0, got {ssa!r}")
    if topology == "granular":
        return 3.0 / (ssa * RHO_ICE) * 1e6
    if topology == "bubbly":
        if density_kg_m3 is None:
            raise ValueError(
                "density_kg_m3 is required to convert SSA to a bubble "
                "radius: SSA fixes only the combination 3*phi/(r*rho)."
            )
        rho = float(density_kg_m3)
        _check_bubbly_density(rho)
        phi = 1.0 - rho / RHO_ICE
        return 3.0 * phi / (ssa * rho) * 1e6
    raise ValueError(
        f"topology must be 'granular' or 'bubbly', got {topology!r}")


def _check_bubbly_density(rho: float) -> None:
    if not np.isfinite(rho) or rho <= 0 or rho >= RHO_ICE:
        raise ValueError(
            f"bubbly-ice density must be in (0, {RHO_ICE}) kg/m3, got "
            f"{rho!r}. At rho -> 917 the air fraction goes to zero and SSA "
            "with it."
        )


def ssa_density_sensitivity(density_kg_m3: float) -> float:
    """``|d ln SSA / d ln rho|`` for a bubbly layer — the error amplification.

    SSA is a small difference of large numbers at sea-ice densities:
    ``phi = 1 - rho/917`` is 0.03-0.11 over 820-900 kg/m3, and the sensitivity
    is ``1/phi``, i.e. **9-30x**.  A 2% density error becomes a ~30% SSA
    error.  That is physical rather than a coordinate artefact, but it means
    SSA is only a useful coordinate where density is genuinely constrained.
    For snow (``phi ~ 0.67``) the amplification is ~1.5x and does not bite.
    """
    rho = float(density_kg_m3)
    _check_bubbly_density(rho)
    return 1.0 / (1.0 - rho / RHO_ICE)
