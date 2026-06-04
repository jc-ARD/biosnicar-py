"""Melt pond areal fraction blending for BioSNICAR.

Arctic sea ice surfaces are rarely homogeneous: a pixel or survey transect
typically contains a mixture of bare/snow-covered white ice and melt ponds
in varying proportions.  This module provides a linear area-weighted mixing
model that combines a white-ice albedo spectrum with a melt-pond albedo
spectrum according to a pond areal fraction *f*:

    α_total(λ) = (1 − f) · α_ice(λ)  +  f · α_pond(λ)

This is the standard surface albedo mixing model used in GCMs for Arctic
mixed surfaces (Briegleb & Light 2007; Notz & Marotzke 2009).  It is valid
when the horizontal scale of individual ice/pond features (metres to hundreds
of metres) is much larger than the photon transport mean free path in the
atmosphere, so radiative adjacency effects between neighbouring patches are
negligible.

Usage::

    from biosnicar import run_model
    from biosnicar.sea_ice.pond_fraction import blend_pond_fraction
    from biosnicar.sea_ice.presets import FYI_SUMMER_BARE, FYI_POND_SHALLOW

    ice  = run_model(preset=FYI_SUMMER_BARE,  solzen=60)
    pond = run_model(preset=FYI_POND_SHALLOW, solzen=60)

    mixed = blend_pond_fraction(ice, pond, f=0.30)
    print(mixed.BBA)   # blended broadband albedo

Or via the run_model() convenience interface::

    mixed = run_model(
        preset="FYI_SUMMER_BARE",
        solzen=60,
        pond_fraction=0.30,   # 30 % pond cover
        pond_depth=0.15,      # companion pond water depth (m)
    )
"""

import copy

import numpy as np

from biosnicar.classes.outputs import Outputs

# Band boundary indices on the 480-band 0.205–4.995 µm grid.
# VIS_MAX = 50 corresponds to 0.705 µm (from inputs.yaml RTM.VIS_MAX_IDX).
_VIS_MAX = 50
_NIR_MAX = 480


def blend_pond_fraction(ice: Outputs, pond: Outputs, f: float) -> Outputs:
    """Linear area-weighted blend of white-ice and melt-pond albedo spectra.

    Args:
        ice:  Outputs from a white-ice model run (no pond water layer).
        pond: Outputs from a melt-pond model run (layer_type=5 on top).
        f:    Pond areal fraction, 0 ≤ f ≤ 1.
              f=0 → pure white ice (returns ice unchanged).
              f=1 → pure melt pond (returns pond albedo with ice BBAs).

    Returns:
        Outputs with blended spectral albedo and recomputed BBA/BBAVIS/BBANIR.
        The solar flux weighting uses ``ice.flx_slr`` (identical for both
        runs given the same SZA and atmospheric profile).
        Subsurface flux fields (F_up, F_dwn, heat_rt) are set to None — they
        are not physically meaningful for a spatially mixed surface.

    Raises:
        ValueError: if *f* is outside [0, 1].

    References:
        Briegleb, B. P. & Light, B. (2007). A Delta-Eddington multiple
        scattering parameterization for solar radiation in the sea ice
        component of the Community Climate System Model. NCAR Tech. Note
        TN-472+STR.

        Notz, D. & Marotzke, J. (2009). Observations reveal external driver
        for Arctic sea-ice retreat. Geophys. Res. Lett., 36, L15502.
    """
    if not 0.0 <= f <= 1.0:
        raise ValueError(
            f"pond_fraction must be in [0, 1], got {f!r}"
        )

    result = copy.copy(ice)
    result.albedo = (1.0 - f) * ice.albedo + f * pond.albedo

    # Recompute flux-weighted broadband values from the blended spectrum.
    # Both runs share the same SZA and atmospheric profile, so flx_slr is
    # identical; we use ice.flx_slr as the authoritative weights.
    flx = ice.flx_slr
    result.BBA = float(
        np.sum(flx * result.albedo) / np.sum(flx)
    )
    result.BBAVIS = float(
        np.sum(flx[:_VIS_MAX] * result.albedo[:_VIS_MAX])
        / np.sum(flx[:_VIS_MAX])
    )
    result.BBANIR = float(
        np.sum(flx[_VIS_MAX:_NIR_MAX] * result.albedo[_VIS_MAX:_NIR_MAX])
        / np.sum(flx[_VIS_MAX:_NIR_MAX])
    )

    # Subsurface fluxes are not meaningful for a spatially mixed surface:
    # each patch has independent subsurface light fields that cannot be
    # averaged without knowing the geometric arrangement of ice and pond.
    result.F_up = None
    result.F_dwn = None
    result.heat_rt = None

    return result
