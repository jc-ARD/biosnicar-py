"""Single entry point for running the BioSNICAR forward model.

Provides :func:`run_model`, which sets up the model from a YAML config,
applies optional keyword overrides, runs the radiative transfer solver,
and returns an :class:`~biosnicar.classes.outputs.Outputs` object.

Example::

    from biosnicar.drivers.run_model import run_model

    outputs = run_model(solzen=50, rds=1000)
    print(outputs.BBA)
"""

import re
import warnings

from biosnicar.drivers.setup_snicar import setup_snicar
from biosnicar.optical_properties.column_OPs import get_layer_OPs, mix_in_impurities
from biosnicar.rt_solvers.adding_doubling_solver import adding_doubling_solver
from biosnicar.rt_solvers.toon_rt_solver import toon_solver
from biosnicar.utils.display import plot_albedo
from biosnicar.utils.validate_inputs import validate_inputs

# Legacy regex for impurity concentration keys like "impurity_0_conc"
_IMPURITY_CONC_RE = re.compile(r"^impurity_(\d+)_conc$")

# Parameter keys that require recalculating irradiance
_ILLUMINATION_KEYS = {"solzen", "direct", "incoming"}

# Parameter keys that apply to the ice object (broadcast to all layers)
_ICE_BROADCAST_KEYS = {
    "rds", "rho", "dz", "lwc", "layer_type",
    "cdom", "shp", "water", "hex_side", "hex_length", "shp_fctr", "grain_ar",
    # sea-ice per-layer fields (layer_type=4)
    "sea_ice_salinity", "sea_ice_temperature", "sea_ice_bubble_radius",
}

# Young-ice (layer_type=6) scalar keys — stored as attributes on the ice
# object and consumed by the thin-slab boundary-condition path.
_YOUNG_ICE_KEYS = {"ice_thickness", "ocean_albedo"}

# Placement of the refractive (Fresnel) boundary — a column-level scalar:
# "auto" (default rule), a layer index, or None to remove the interface.
_FRESNEL_KEYS = {"fresnel_layer"}

# All per-layer ice list attributes (used to resize when nbr_lyr changes)
_ICE_ALL_LIST_ATTRS = [
    "dz", "layer_type", "cdom", "rho", "rds", "shp", "water",
    "hex_side", "hex_length", "shp_fctr", "grain_ar", "lwc",
    "sea_ice_salinity", "sea_ice_temperature", "sea_ice_bubble_radius",
]


def run_model(
    input_file="default",
    solver="adding-doubling",
    validate=False,
    plot=False,
    preset=None,
    pond_fraction=None,
    pond_depth=0.15,
    pond_preset=None,
    **overrides,
):
    """Run the BioSNICAR forward model and return outputs.

    Single entry point for both terrestrial and sea-ice configurations.
    Builds model objects from *input_file*, applies any keyword *overrides*
    (and optional *preset*) to ice/illumination/impurity parameters, then
    runs the full pipeline (optical properties → impurity mixing →
    radiative transfer).

    Args:
        input_file: Path to YAML config or ``"default"`` for the bundled
            ``inputs.yaml``.
        solver: ``"adding-doubling"`` (default) or ``"toon"``.
        validate: If True, run input validation before the forward model.
        plot: If True, display a spectral albedo plot after the run.
        preset: Sea-ice preset to use as a base configuration. Either a
            string name (``"FYI_WINTER_BARE"``, ``"FYI_WINTER_SNOW"``,
            ``"MYI_WINTER_BARE"``) or a preset dict imported from
            ``biosnicar.sea_ice.presets``.  Any additional ``**overrides``
            are merged on top of the preset, taking precedence.
            Example::

                outputs = run_model(preset="FYI_WINTER_BARE", solzen=70)
                outputs = run_model(preset=FYI_WINTER_SNOW, solzen=60,
                                    black_carbon=500)

        pond_fraction: Melt pond areal fraction, 0–1. When set, a companion
            melt-pond spectrum is computed and linearly blended with the main
            (white-ice) result:
            ``α_total = (1 − f) · α_ice + f · α_pond``.
            Requires the main configuration to be a bare/snow-covered ice
            preset; the companion pond uses *pond_preset* (default:
            ``"FYI_POND_SHALLOW"``) at depth *pond_depth*.
            Example::

                mixed = run_model(
                    preset="FYI_SUMMER_BARE",
                    solzen=60,
                    pond_fraction=0.30,
                    pond_depth=0.15,
                )

        pond_depth: Water depth (m) of the companion melt pond used when
            *pond_fraction* is set. Default 0.15 m. Ignored when
            *pond_fraction* is None.
        pond_preset: Preset name or dict for the companion melt-pond model.
            Default ``"FYI_POND_SHALLOW"``. The pond water layer thickness
            is overridden to *pond_depth*; all other pond parameters
            (floor ice density, salinity, LAP) come from this preset.
            Ignored when *pond_fraction* is None.

        **overrides: Parameter overrides applied before running the model.

            Illumination:

            - **solzen** (*float*) — solar zenith angle (degrees, 1–89)
            - **direct** (*int*) — 1 for direct beam, 0 for diffuse/cloudy
            - **incoming** (*int*) — irradiance spectrum index (0–6):
              0=mid-lat winter, 1=mid-lat summer, 2=sub-Arctic winter,
              3=sub-Arctic summer, 4=summit, 5=high mountain, 6=tropical

            Ice structure (all accept scalar or per-layer list):

            - **layer_type** — 0=granular snow/ice, 1=solid glacier ice
              (Fresnel), 2=solid ice (no Fresnel), 3=mixed water/ice
              spheres, **4=sea ice** (brine and salt inclusions, air bubbles),
              **5=melt pond** (liquid water, set rho=1000 kg/m³)
            - **dz** — layer thickness (m)
            - **rho** — bulk density (kg/m³)
            - **rds** — grain/bubble radius (µm); not used for
              ``layer_type=4`` (sea ice), pass ``None`` or any value
            - **lwc** — liquid water content (volume fraction)
            - **shp** — grain shape (0=sphere, 1=spheroid, 2=hex plate,
              3=Koch snowflake, 4=hex prism)
            - **grain_ar**, **shp_fctr** — asphericity parameters
            - **cdom** — CDOM flag (0/1, types 1/2 only)
            - **water** — liquid water coating radius (µm, type 0 only)
            - **hex_side**, **hex_length** — hexagonal prism dimensions

            Refractive boundary (adding-doubling solver only):

            - **fresnel_layer** (*"auto" | int | None*) — where the smooth
              air/medium interface sits.  ``"auto"`` (default) puts it on
              top of the first solid-ice, sea-ice or melt-pond layer
              (types 1, 4, 5).  An integer puts it on top of that layer
              instead; ``None`` removes it.  The medium is water when that
              layer is a melt pond and ice otherwise.

            Sea ice (required when ``layer_type=4``; use ``None`` for
            non-sea-ice layers in the same column):

            - **sea_ice_salinity** (*float | list*) — bulk salinity (psu)
            - **sea_ice_temperature** (*float | list*) — temperature (°C,
              must be in [−44, 0]; the optics LUT spans −30 to −1)
            - **sea_ice_bubble_radius** (*float | list*) — effective air-
              bubble radius (µm, typically 100–1000)

            Optical parameterisation (alternatives to the above — supply
            one set or the other, never both for the same layer):

            - **ssa** (*float | list*) — specific surface area of the
              air/ice interface (m² kg⁻¹), replacing ``rds`` for granular
              and glacier-ice layers and ``sea_ice_bubble_radius`` for
              ``layer_type=4``.  Requires ``layer_type``, and ``rho`` for
              bubbly layers.  Radius and density are individually degenerate
              — the spectrum constrains their combination — so SSA is the
              better-conditioned coordinate.  **Two conventions**: granular
              layers use ``3/(r·ρ_ice)`` and bubbly layers
              ``3(1 − ρ/917)/(r·ρ)``; the layer type selects, since they
              differ by ~2× at snow densities.  See
              :mod:`biosnicar.sea_ice.parameterisation`.
            - **brine_volume_fraction** (*float | list*) — replaces
              ``sea_ice_temperature`` for ``layer_type=4``, inverting
              Cox & Weeks (1983) at that layer's salinity and bulk density.
              Requires ``sea_ice_salinity`` and ``rho``.

            Impurities (scalar = apply to first layer only; list = per-layer):

            - **black_carbon** (*float | list*) — black carbon (ppb)
            - **snow_algae** (*float | list*) — snow algae (cells/mL)
            - **glacier_algae** (*float | list*) — glacier algae (cells/mL)

    Returns:
        :class:`~biosnicar.classes.outputs.Outputs` with ``.BBA``,
        ``.BBAVIS``, ``.BBANIR``, ``.albedo`` (480-band spectrum), and
        convenience aliases ``.broadband``, ``.visible``, ``.nir``,
        ``.spectrum``, ``.wavelengths``.  Also provides
        ``.to_platform()``, ``.plot()``, and subsurface-light methods.

    Examples::

        # Terrestrial glacier ice
        outputs = run_model(solzen=50, layer_type=1, rds=500, rho=700)

        # Snow on sea ice — identical style, just add sea-ice params
        outputs = run_model(
            solzen=60,
            layer_type=[0, 4, 4],
            dz=[0.15, 0.05, 1.45],
            rds=[200, None, None],
            rho=[300, 895, 895],
            sea_ice_salinity=[None, 12, 8],
            sea_ice_temperature=[None, -25, -20],
            sea_ice_bubble_radius=[None, 100, 200],
            black_carbon=500,
        )

        # The same column in the optical parameterisation
        from biosnicar.sea_ice import specific_surface_area
        outputs = run_model(
            solzen=60,
            layer_type=[0, 4, 4],
            dz=[0.15, 0.05, 1.45],
            rho=[300, 895, 895],
            ssa=[specific_surface_area(200, "granular"),
                 specific_surface_area(100, "bubbly", 895),
                 specific_surface_area(200, "bubbly", 895)],
            sea_ice_salinity=[None, 12, 8],
            brine_volume_fraction=[None, 0.082, 0.049],
            black_carbon=500,
        )

        # Both outputs use the same attribute names
        print(outputs.BBA)          # or outputs.broadband
        print(outputs.albedo)       # or outputs.spectrum
        outputs.to_platform("sentinel2")

    Raises:
        ValueError: If *solver* or *preset* name is not recognised, or an
            override key is unknown.
    """
    # Expand preset into base overrides (explicit kwargs take precedence)
    if preset is not None:
        from biosnicar.sea_ice.presets import _resolve_preset
        overrides = {**_resolve_preset(preset), **overrides}

    # Resolve "default" to actual path so it can be reused in recalculations
    if input_file == "default":
        from pathlib import Path

        input_file = Path(__file__).resolve().parent.joinpath(
            "../inputs.yaml"
        ).as_posix()

    ice, illumination, rt_config, model_config, plot_config, impurities = (
        setup_snicar(input_file)
    )

    if overrides:
        overrides = _resolve_parameterisations(overrides)
        _apply_overrides(overrides, ice, illumination, impurities, input_file)

    _set_lower_boundary(ice)

    if validate:
        validate_inputs(ice, illumination, impurities)

    # Young ice (layer_type=6): thin-slab boundary-condition albedo —
    # bypasses the τ/ω/g pipeline and the RT solvers entirely.
    if 6 in ice.layer_type:
        return _run_young_ice(ice, illumination)

    # Optical properties
    ssa_snw, g_snw, mac_snw = get_layer_OPs(ice, model_config)
    tau, ssa, g, L_snw = mix_in_impurities(
        ssa_snw, g_snw, mac_snw, ice, impurities, model_config
    )

    # Radiative transfer
    if solver == "adding-doubling":
        outputs = adding_doubling_solver(
            tau, ssa, g, L_snw, ice, illumination, model_config
        )
    elif solver == "toon":
        outputs = toon_solver(
            tau, ssa, g, L_snw, ice, illumination, model_config, rt_config
        )
    else:
        raise ValueError(
            f"Unknown solver {solver!r}; use 'adding-doubling' or 'toon'"
        )

    if pond_fraction is not None:
        outputs = _blend_with_pond(
            outputs, pond_fraction, pond_depth, pond_preset,
            input_file, solver, overrides,
        )

    if plot:
        plot_albedo(plot_config, model_config, outputs.albedo)

    return outputs


def _set_lower_boundary(ice):
    """Put ocean, not glacier ice, under a sea-ice column.

    ``ice.sfc`` — the reflectance of whatever lies beneath the bottom layer —
    comes from the YAML config and defaults to a measured blue glacier-ice
    spectrum (16% at 450 nm), which is the right thing under a glacier
    weathering crust and the wrong thing under floating ice.  When the bottom
    layer is sea ice the boundary is the ocean's subsurface reflectance.
    The effect is small for thick ice (under 0.001 in broadband albedo at
    1.5 m) and grows as the ice thins (0.002 at 30 cm).
    """
    if ice.layer_type and ice.layer_type[-1] == 4:
        from biosnicar.sea_ice.open_water import ocean_subsurface_reflectance
        from biosnicar import perturbation

        ice.sfc = (ocean_subsurface_reflectance()
                   * perturbation.get("ocean_reflectance_scale"))


def _run_young_ice(ice, illumination):
    """Evaluate the layer_type=6 thin-slab model and wrap as Outputs."""
    import numpy as np

    from biosnicar.classes.outputs import Outputs
    from biosnicar.optical_properties.column_OPs import _compute_young_ice_ops

    if any(lt != 6 for lt in ice.layer_type):
        raise ValueError(
            "layer_type=6 (young ice) must be the only layer in the column — "
            f"got layer_type={ice.layer_type}.  Young ice is a boundary "
            "condition, not a stackable scattering layer."
        )
    thickness = getattr(ice, "ice_thickness", None)
    if thickness is None:
        raise ValueError("layer_type=6 requires the `ice_thickness` override (m).")
    T = next((t for t in ice.sea_ice_temperature if t is not None), None)
    S = next((s for s in ice.sea_ice_salinity if s is not None), None)
    if T is None or S is None:
        raise ValueError(
            "layer_type=6 requires `sea_ice_temperature` (degC) and "
            "`sea_ice_salinity` (psu)."
        )

    albedo = _compute_young_ice_ops(
        thickness_m=thickness,
        temperature_C=T,
        salinity_psu=S,
        ocean_albedo=getattr(ice, "ocean_albedo", 0.04),
        solzen=illumination.solzen,
        direct=illumination.direct,
    )

    flx_slr = np.asarray(illumination.flx_slr, dtype=float)
    outputs = Outputs()
    outputs.albedo = albedo
    outputs.flx_slr = flx_slr
    outputs.BBA = float(np.sum(flx_slr * albedo) / np.sum(flx_slr))
    outputs.BBAVIS = float(np.sum(flx_slr[:50] * albedo[:50]) / np.sum(flx_slr[:50]))
    outputs.BBANIR = float(
        np.sum(flx_slr[50:] * albedo[50:]) / np.sum(flx_slr[50:])
    )
    return outputs


def _blend_with_pond(ice_outputs, f, pond_depth, pond_preset_arg,
                     input_file, solver, ice_overrides):
    """Run a companion pond model and blend with the ice result."""
    from biosnicar.sea_ice.pond_fraction import blend_pond_fraction
    from biosnicar.sea_ice.presets import _resolve_preset

    if not 0.0 <= f <= 1.0:
        raise ValueError(f"pond_fraction must be in [0, 1], got {f!r}")

    # Resolve companion pond preset (default: FYI_POND_SHALLOW)
    if pond_preset_arg is None:
        pond_preset_arg = "FYI_POND_SHALLOW"
    pond_kwargs = _resolve_preset(pond_preset_arg)

    # Override the pond water depth (first layer is the water layer)
    dz = list(pond_kwargs.get("dz", [0.15, 0.05, 1.40]))
    dz[0] = pond_depth
    pond_kwargs["dz"] = dz

    # Carry over illumination settings from the ice run
    for key in _ILLUMINATION_KEYS:
        if key in ice_overrides:
            pond_kwargs[key] = ice_overrides[key]

    pond_outputs = run_model(input_file=input_file, solver=solver, **pond_kwargs)
    return blend_pond_fraction(ice_outputs, pond_outputs, f)


# Layer types whose optics include brine: sea ice (4) and young ice (6).
_BRINE_LAYER_TYPES = {4, 6}


def _warn_ignored_brine(ice):
    """Warn when salinity is given for a layer whose optics carry no brine.

    Only ``layer_type`` 4 and 6 model brine.  Granular (0, 3) and bubbly-ice
    (1, 2) layers are treated as fresh, so a saline snow cover or a saline
    weathering crust has no correct representation: the salinity would be
    accepted and silently dropped.  Drained crust is largely desalinated, so
    the approximation is usually tolerable — but it should be a stated one.
    """
    sal = getattr(ice, "sea_ice_salinity", None) or []
    ignored = [
        (i, lt, s) for i, (lt, s) in enumerate(zip(ice.layer_type, sal))
        if s is not None and float(s) > 0 and lt not in _BRINE_LAYER_TYPES
    ]
    if ignored:
        detail = ", ".join(
            f"layer {i} (layer_type={lt}, S={s:g} psu)" for i, lt, s in ignored
        )
        warnings.warn(
            f"sea_ice_salinity is ignored for {detail}: only layer_type 4 "
            "(sea ice) and 6 (young ice) model brine, so these layers are "
            "computed as fresh ice. Pass None (or 0) for them to state that "
            "explicitly, or use layer_type=4 for saline ice.",
            UserWarning, stacklevel=4,
        )


# Alternative ("optical") parameterisations, resolved to canonical model
# inputs before _apply_overrides sees them.  Keyed by the alias the caller
# supplies; the value is the canonical key(s) it replaces.
_ALIAS_PARAMETERISATIONS = {
    "ssa": ("rds", "sea_ice_bubble_radius"),
    "brine_volume_fraction": ("sea_ice_temperature",),
}


def _as_layer_list(value, n):
    """Broadcast a scalar to *n* layers, or validate an existing list."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(
                f"expected {n} per-layer values to match layer_type, got "
                f"{len(value)}"
            )
        return list(value)
    return [value] * n


def _resolve_parameterisations(overrides):
    """Convert ``ssa`` / ``brine_volume_fraction`` to canonical model inputs.

    Radius and density are individually degenerate and temperature acts on the
    optics only through brine volume, so a caller may prefer to specify the
    combination the measurement actually constrains.  Either set is accepted;
    giving both for the same layer is an error rather than a silent
    precedence rule.

    ``ssa`` becomes ``sea_ice_bubble_radius`` for ``layer_type=4`` and ``rds``
    otherwise, using the granular or bubbly convention as the layer type
    dictates (see :mod:`biosnicar.sea_ice.parameterisation` — the two formulas
    differ by ~2x at snow densities, so the topology is not guessed).

    ``brine_volume_fraction`` becomes ``sea_ice_temperature`` by inverting
    Cox & Weeks (1983) at that layer's salinity and bulk density.
    """
    aliases = [k for k in _ALIAS_PARAMETERISATIONS if k in overrides]
    if not aliases:
        return overrides

    from biosnicar.sea_ice.parameterisation import (
        radius_from_ssa, topology_for_layer_type,
    )

    out = dict(overrides)
    layer_type = out.get("layer_type")
    if layer_type is None:
        raise ValueError(
            f"{aliases[0]!r} requires `layer_type` to be given in the same "
            "call: the conversion depends on the layer's topology "
            "(granular vs bubbly ice), which cannot be inferred otherwise."
        )
    layer_type = layer_type if isinstance(layer_type, (list, tuple)) else [layer_type]
    n = len(layer_type)

    for alias in aliases:
        values = _as_layer_list(out.pop(alias), n)
        for i, v in enumerate(values):
            if v is None:
                continue
            lt = int(layer_type[i])
            if alias == "ssa":
                target = "sea_ice_bubble_radius" if lt == 4 else "rds"
                topology = topology_for_layer_type(lt)
                rho = None
                if topology == "bubbly":
                    rho_list = out.get("rho")
                    if rho_list is None:
                        raise ValueError(
                            f"ssa for layer {i} (layer_type={lt}) needs `rho` "
                            "in the same call: SSA fixes only the combination "
                            "3*(1 - rho/917)/(r*rho), so density closes it."
                        )
                    rho = _as_layer_list(rho_list, n)[i]
                new_value = radius_from_ssa(v, topology, rho)
            else:   # brine_volume_fraction
                if lt != 4:
                    raise ValueError(
                        f"brine_volume_fraction is only meaningful for "
                        f"layer_type=4 (sea ice); layer {i} is "
                        f"layer_type={lt}."
                    )
                target = "sea_ice_temperature"
                sal = out.get("sea_ice_salinity")
                rho_list = out.get("rho")
                if sal is None or rho_list is None:
                    raise ValueError(
                        f"brine_volume_fraction for layer {i} needs both "
                        "`sea_ice_salinity` and `rho` in the same call: brine "
                        "volume is a function of salinity, temperature and "
                        "bulk density together (Cox & Weeks 1983)."
                    )
                from biosnicar.sea_ice.brine_volume import invert_brine_volume
                new_value = float(invert_brine_volume(
                    v, _as_layer_list(sal, n)[i],
                    bulk_density_kg_m3=_as_layer_list(rho_list, n)[i],
                ))

            existing = out.get(target)
            if existing is not None:
                at_i = _as_layer_list(existing, n)[i]
                if at_i is not None:
                    raise ValueError(
                        f"layer {i}: both {alias!r} and {target!r} were given "
                        f"({alias}={v!r}, {target}={at_i!r}). They are two "
                        "ways of saying the same thing — supply one."
                    )
                slot = _as_layer_list(existing, n)
            else:
                slot = [None] * n
            slot[i] = new_value
            out[target] = slot

    return out


def _apply_overrides(overrides, ice, illumination, impurities, input_file):
    """Mutate model objects in-place according to keyword overrides."""
    needs_irradiance = False
    needs_refractive = False

    # Determine the new layer count from any list-valued ice override
    new_nbr_lyr = None
    for key, value in overrides.items():
        if key in _ICE_BROADCAST_KEYS and isinstance(value, list):
            new_nbr_lyr = len(value)
            break

    # Sea-ice fields default to None — use None rather than last-value fill.
    _SEA_ICE_ATTRS = {"sea_ice_salinity", "sea_ice_temperature", "sea_ice_bubble_radius"}

    # If layer count is changing, resize all per-layer attributes first
    if new_nbr_lyr is not None and new_nbr_lyr != ice.nbr_lyr:
        for attr in _ICE_ALL_LIST_ATTRS:
            old = getattr(ice, attr)
            fill = None if attr in _SEA_ICE_ATTRS else old[-1]
            if len(old) < new_nbr_lyr:
                setattr(ice, attr, old + [fill] * (new_nbr_lyr - len(old)))
            elif len(old) > new_nbr_lyr:
                setattr(ice, attr, old[:new_nbr_lyr])
        # Resize impurity concentrations too
        for imp in impurities:
            old = imp.conc
            if len(old) < new_nbr_lyr:
                imp.conc = old + [0] * (new_nbr_lyr - len(old))
            elif len(old) > new_nbr_lyr:
                imp.conc = old[:new_nbr_lyr]
        ice.nbr_lyr = new_nbr_lyr

    # Build name→index mapping from impurities list
    _imp_name_map = {imp.name: i for i, imp in enumerate(impurities)}

    for key, value in overrides.items():
        # Illumination scalars
        if key in _ILLUMINATION_KEYS:
            setattr(illumination, key, value)
            needs_irradiance = True

        # Young-ice scalars (layer_type=6)
        elif key in _YOUNG_ICE_KEYS:
            setattr(ice, key, float(value))

        # Fresnel-boundary placement (validated by the solver)
        elif key in _FRESNEL_KEYS:
            ice.fresnel_layer = value

        # Ice broadcast keys
        elif key in _ICE_BROADCAST_KEYS:
            # rds=None is valid for layer_type=4 (sea ice) layers where rds
            # is unused.  Substitute a harmless dummy so downstream code
            # never receives None in ice.rds.
            if key == "rds":
                if isinstance(value, list):
                    value = [500 if v is None else v for v in value]
                elif value is None:
                    value = 500
            if isinstance(value, list):
                setattr(ice, key, value)
            else:
                setattr(ice, key, [value] * ice.nbr_lyr)
            needs_refractive = True

        # Named impurity keys (e.g. black_carbon, glacier_algae)
        elif key in _imp_name_map:
            idx = _imp_name_map[key]
            if isinstance(value, list):
                impurities[idx].conc = value
            else:
                impurities[idx].conc = [value] + [0] * (ice.nbr_lyr - 1)

        # Legacy impurity_0_conc syntax (deprecated)
        else:
            m = _IMPURITY_CONC_RE.match(key)
            if m:
                idx = int(m.group(1))
                imp_name = impurities[idx].name if idx < len(impurities) else f"impurity {idx}"
                warnings.warn(
                    f"{key!r} is deprecated; use {imp_name!r} instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                if isinstance(value, list):
                    impurities[idx].conc = value
                else:
                    impurities[idx].conc = [value] + [0] * (ice.nbr_lyr - 1)
            else:
                imp_names = sorted(_imp_name_map.keys())
                raise ValueError(
                    f"Unknown override key {key!r}. Supported: "
                    f"{sorted(_ILLUMINATION_KEYS | _ICE_BROADCAST_KEYS | _YOUNG_ICE_KEYS | _FRESNEL_KEYS)} "
                    f"and impurity names {imp_names}."
                )

    _warn_ignored_brine(ice)

    if needs_refractive:
        ice.calculate_refractive_index(input_file)
    if needs_irradiance:
        illumination.calculate_irradiance()
