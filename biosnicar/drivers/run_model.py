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
    **overrides,
):
    """Run the BioSNICAR forward model and return outputs.

    Builds model objects from *input_file*, applies any keyword *overrides*
    to ice/illumination/impurity parameters, then runs the full pipeline
    (optical properties -> impurity mixing -> radiative transfer).

    Args:
        input_file: Path to YAML config or ``"default"`` for the bundled
            ``inputs.yaml``.
        solver: ``"adding-doubling"`` (default) or ``"toon"``.
        validate: If True, run input validation before the forward model.
        plot: If True, display a spectral albedo plot after the run.
        **overrides: Parameter overrides applied before running the model.
            Supported keys:

            - **solzen** (*float*) — solar zenith angle (degrees)
            - **direct** (*int*) — 1 for direct beam, 0 for diffuse
            - **incoming** (*int*) — irradiance spectrum index (0–6)
            - **rds** (*int | list*) — grain/bubble radius (µm)
            - **rho** (*int | list*) — layer density (kg/m³)
            - **dz** (*float | list*) — layer thickness (m)
            - **lwc** (*float | list*) — liquid water content
            - **layer_type** (*int | list*) — 0=grains, 1=solid ice, etc.
            - **shp** (*int | list*) — grain shape
            - **grain_ar** (*float | list*) — grain aspect ratio
            - **cdom** (*float | list*) — CDOM concentration
            - **water** (*float | list*) — liquid water coating radius
            - **hex_side** (*int | list*) — hexagonal column side length
            - **hex_length** (*int | list*) — hexagonal column length
            - **shp_fctr** (*float | list*) — shape factor
            - **black_carbon** (*float | list*) — black carbon conc (ppb)
            - **snow_algae** (*float | list*) — snow algae conc (cells/mL)
            - **glacier_algae** (*float | list*) — glacier algae conc
              (cells/mL)

            Impurity names correspond to YAML keys in ``inputs.yaml``.
            Scalar values are broadcast to ``[value, 0, ...]``.

            Legacy ``impurity_{i}_conc`` syntax is still accepted but
            deprecated.

            All ice parameters are broadcast to all layers if a scalar is
            passed, or applied directly if a list.

    Returns:
        :class:`~biosnicar.classes.outputs.Outputs` with albedo, BBA,
        absorbed fluxes, etc.

    Raises:
        ValueError: If *solver* is not recognised or an override key is
            unknown.
    """
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
