"""Switches for forward-model assumptions that are normally held fixed.

The forward model is ``y = F(x, b)``: ``x`` is what a retrieval solves for and
``b`` is everything else — refractive-index datasets, microstructure
constants, interface tables.  ``b`` is not known exactly, and the part of the
measurement error that comes from that is ``K_b S_b K_b^T`` (Rodgers 2000,
section 3.2).  To evaluate it the model has to be run with ``b`` moved, which
is what this module allows::

    from biosnicar import perturbation

    with perturbation.perturbed(brine_index_offset=0.002):
        albedo = run_model(...).albedo

Outside a ``perturbed`` block every switch sits at its neutral value and the
forward model is bit-for-bit what it is without this module.  The switches
are consumed where the assumption is made (search for ``perturbation.get``);
their uncertainties live in :mod:`biosnicar.inverse.forward_uncertainty`.

The state is module-global and so per-process: set it inside the worker when
running in parallel.
"""

from contextlib import contextmanager

# name -> neutral value
NEUTRAL = {
    # Compute layer_type=4 optics from the physics rather than the LUT.
    # Implied by every sea-ice switch below (the LUT cannot express them).
    "sea_ice_direct": False,
    # Ice absorption: k = k_Pic16^(1-t) * k_Wrn08^t.  t=0 is the model's
    # dataset (Picard et al. 2016), t=1 is Warren & Brandt (2008).
    "ice_absorption_blend": 0.0,
    # Added to the real refractive index of brine.
    "brine_index_offset": 0.0,
    # Multiplies the brine-inclusion scattering cross-section per volume
    # (number density of inclusions at fixed brine volume).
    "brine_inclusion_scale": 1.0,
    # Multiply the salt-crystal scattering cross-sections (crystal size).
    "mirabilite_scale": 1.0,
    "hydrohalite_scale": 1.0,
    # Multiplies (1 - g) of air bubbles in sea ice.
    "bubble_backscatter_scale": 1.0,
    # Added to the diffuse Fresnel reflectance of an ice interface seen from
    # below (fl_r_dif_b).  Not applied to a water interface.
    "fresnel_diffuse_offset": 0.0,
    # Multiplies the absorption coefficient of melt-pond water.
    "water_absorption_scale": 1.0,
    # Multiplies the frazil scattering coefficient of young ice.
    "young_ice_scattering_scale": 1.0,
    # Multiplies the ocean's subsurface reflectance (lower boundary of a
    # sea-ice column; water-leaving term of open water).
    "ocean_reflectance_scale": 1.0,
}

_SEA_ICE_SWITCHES = frozenset({
    "sea_ice_direct", "ice_absorption_blend", "brine_index_offset",
    "brine_inclusion_scale", "mirabilite_scale", "hydrohalite_scale",
    "bubble_backscatter_scale",
})

_active = {}


def get(name):
    """Current value of switch *name* (its neutral value when not perturbed)."""
    try:
        return _active[name]
    except KeyError:
        return NEUTRAL[name]


def active():
    """True inside a ``perturbed`` block that moved at least one switch."""
    return bool(_active)


def sea_ice_direct():
    """True when layer_type=4 optics must be computed rather than looked up."""
    return any(k in _active for k in _SEA_ICE_SWITCHES)


@contextmanager
def perturbed(**values):
    """Run the forward model with some switches moved off neutral."""
    unknown = set(values) - set(NEUTRAL)
    if unknown:
        raise KeyError(
            f"unknown perturbation switch(es) {sorted(unknown)}; "
            f"known: {sorted(NEUTRAL)}"
        )
    saved = dict(_active)
    for k, v in values.items():
        if v == NEUTRAL[k]:
            _active.pop(k, None)
        else:
            _active[k] = v
    try:
        yield
    finally:
        _active.clear()
        _active.update(saved)
