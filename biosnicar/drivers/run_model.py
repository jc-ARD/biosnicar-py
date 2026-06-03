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
              spheres, **4=sea ice** (brine inclusions, Maxwell-Garnett)
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

            Sea ice (required when ``layer_type=4``; use ``None`` for
            non-sea-ice layers in the same column):

            - **sea_ice_salinity** (*float | list*) — bulk salinity (psu)
            - **sea_ice_temperature** (*float | list*) — temperature (°C,
              must be in [−44, −2])
            - **sea_ice_bubble_radius** (*float | list*) — effective air-
              bubble radius (µm, typically 100–1000)

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
        _apply_overrides(overrides, ice, illumination, impurities, input_file)

    if validate:
        validate_inputs(ice, illumination, impurities)

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

    if plot:
        plot_albedo(plot_config, model_config, outputs.albedo)

    return outputs


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
                    f"{sorted(_ILLUMINATION_KEYS | _ICE_BROADCAST_KEYS)} "
                    f"and impurity names {imp_names}."
                )

    if needs_refractive:
        ice.calculate_refractive_index(input_file)
    if needs_irradiance:
        illumination.calculate_irradiance()
