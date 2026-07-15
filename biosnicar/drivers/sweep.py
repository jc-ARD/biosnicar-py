"""Parameter sweep API for running biosnicar across ranges of inputs.

Example usage::

    from biosnicar.drivers.sweep import parameter_sweep

    df = parameter_sweep(
        params={
            "solzen": [30, 40, 50, 60, 70],
            "rds": [100, 200, 500, 1000],
        }
    )
    df.pivot_table(values="BBA", index="solzen", columns="rds").plot()

    # Chain with platform band convolution:
    df = parameter_sweep(
        params={"rds": [500, 1000], "solzen": [50, 60]},
    ).to_platform("sentinel2")

"""

import itertools
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from biosnicar.drivers.setup_snicar import setup_snicar, get_impurity_names
from biosnicar.optical_properties.column_OPs import get_layer_OPs, mix_in_impurities
from biosnicar.rt_solvers.adding_doubling_solver import adding_doubling_solver
from biosnicar.rt_solvers.toon_rt_solver import toon_solver


class SweepResult(pd.DataFrame):
    """DataFrame subclass returned by :func:`parameter_sweep`.

    Adds a :meth:`to_platform` method for chaining band convolution::

        parameter_sweep(params={...}).to_platform("sentinel2")
    """

    _metadata = ["_spectral"]

    @property
    def _constructor(self):
        return SweepResult

    def to_platform(self, *platforms):
        """Convolve spectral albedo onto platform bands for every row.

        Args:
            *platforms: One or more platform keys (e.g. ``"sentinel2"``,
                ``"modis"``).  When a single platform is given, band columns
                are unprefixed (``B3``, ``NDSI``).  When multiple platforms
                are given, columns are prefixed (``sentinel2_B3``,
                ``modis_B1``).

        Returns:
            :class:`pandas.DataFrame` with the original sweep columns plus
            band albedo and index columns appended.
        """
        from biosnicar.bands import to_platform as _to_platform

        if not platforms:
            raise ValueError("Provide at least one platform name.")

        if not hasattr(self, "_spectral") or self._spectral is None:
            raise RuntimeError(
                "No spectral data stored on this SweepResult. "
                "This is a bug -- please report it."
            )

        # Look spectra up by row index so derived frames (sorted, filtered,
        # head()) get each row's own spectrum rather than positional zip.
        if isinstance(self._spectral, pd.Series):
            aligned = self._spectral.reindex(self.index)
            if aligned.isna().any():
                missing = list(self.index[aligned.isna()])[:5]
                raise KeyError(
                    f"No stored spectra for rows {missing} — was this frame "
                    "built from rows outside the original sweep?"
                )
            pairs = list(aligned)
        else:  # legacy plain-list storage: only positional order is valid
            if len(self._spectral) != len(self):
                raise RuntimeError(
                    "Spectral store length does not match this frame — "
                    "re-run the sweep (legacy list storage cannot follow "
                    "filtered/reordered frames)."
                )
            pairs = self._spectral

        prefix = len(platforms) > 1
        rows = []
        for albedo, flx_slr in pairs:
            band_data = {}
            for plat in platforms:
                r = _to_platform(albedo, plat, flx_slr=flx_slr)
                pfx = "{}_".format(plat) if prefix else ""
                for name in r.band_names:
                    band_data["{}{}".format(pfx, name)] = getattr(r, name)
                for name in r.index_names:
                    band_data["{}{}".format(pfx, name)] = getattr(r, name)
            rows.append(band_data)

        band_df = pd.DataFrame(rows, index=self.index)
        return pd.concat([self, band_df], axis=1)

    def plot_sensitivity(self, y="BBA", save=None, show=False, **kwargs):
        """Plot parameter sensitivity.

        See :func:`biosnicar.plotting.plot_sensitivity`.

        Parameters
        ----------
        y : str
            Column to plot (default ``"BBA"``).
        save : str or Path, optional
            Save figure to this path.
        show : bool
            If True, display in an interactive window.
        **kwargs
            Passed to :func:`~biosnicar.plotting.plot_sensitivity`.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        from biosnicar.plotting import plot_sensitivity
        return plot_sensitivity(self, y=y, save=save, show=show, **kwargs)

# Legacy regex for impurity concentration keys like "impurity.0.conc"
_IMPURITY_CONC_RE = re.compile(r"^impurity\.(\d+)\.conc$")

# Parameter keys that require recalculating irradiance
_ILLUMINATION_KEYS = {"solzen", "direct", "incoming"}

# Parameter keys that apply to the ice object (broadcast to all layers)
_ICE_BROADCAST_KEYS = {
    "rds", "rho", "dz", "lwc", "layer_type",
    "cdom", "shp", "water", "hex_side", "hex_length", "shp_fctr", "grain_ar",
}

_VALID_KEYS = _ILLUMINATION_KEYS | _ICE_BROADCAST_KEYS

_OUTPUT_SCALARS = [
    "BBA",
    "BBAVIS",
    "BBANIR",
    "abs_slr_tot",
    "abs_vis_tot",
    "abs_nir_tot",
    "abs_slr_btm",
    "total_insolation",
]


def parameter_sweep(
    params,
    solver="adding-doubling",
    input_file="default",
    return_spectral=False,
    progress=True,
):
    """Run biosnicar over the Cartesian product of parameter values.

    Args:
        params: Dict mapping parameter names to lists of values.
            Supported keys: ``solzen``, ``direct``, ``incoming``, ``rds``,
            ``rho``, ``dz``, ``layer_type``, and impurity names
            (``black_carbon``, ``snow_algae``, ``glacier_algae``).
        solver: ``"adding-doubling"`` (default) or ``"toon"``.
        input_file: Path to YAML config or ``"default"``.
        return_spectral: If True, include an ``albedo`` column with 480-element
            numpy arrays.
        progress: If True, display a tqdm progress bar.

    Returns:
        :class:`SweepResult` (a DataFrame subclass) with one row per
        parameter combination.  Call ``.to_platform("sentinel2")`` on the
        result to append band-convolved albedo columns.

    Raises:
        ValueError: If an unrecognised parameter key is supplied.
    """
    _validate_keys(params, input_file)

    # Resolve "default" to actual path so it can be reused in recalculations
    if input_file == "default":
        input_file = Path(__file__).resolve().parent.joinpath("../inputs.yaml").as_posix()

    # Build baseline objects once
    ice, illumination, rt_config, model_config, plot_config, impurities = setup_snicar(
        input_file
    )

    # Resolve solver function
    if solver == "adding-doubling":
        solve = _run_adding_doubling
    elif solver == "toon":
        solve = _run_toon
    else:
        raise ValueError(f"Unknown solver {solver!r}; use 'adding-doubling' or 'toon'")

    # Build Cartesian product
    keys = list(params.keys())
    value_lists = [params[k] for k in keys]
    combos = list(itertools.product(*value_lists))

    # Optional progress bar
    iterator = combos
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(combos, desc="parameter_sweep")
        except ImportError:
            pass

    results = []
    spectral = []
    for combo in iterator:
        combo_dict = dict(zip(keys, combo))

        # Apply parameter mutations
        _apply_params(combo_dict, ice, illumination, impurities, input_file)

        # Forward model
        ssa_snw, g_snw, mac_snw = get_layer_OPs(ice, model_config)
        tau, ssa, g, L_snw = mix_in_impurities(
            ssa_snw, g_snw, mac_snw, ice, impurities, model_config
        )
        outputs = solve(tau, ssa, g, L_snw, ice, illumination, model_config, rt_config)

        # Always capture spectral data for .to_platform() chaining
        spectral.append((np.array(outputs.albedo), np.array(outputs.flx_slr)))

        # Collect results
        row = dict(combo_dict)
        for attr in _OUTPUT_SCALARS:
            row[attr] = getattr(outputs, attr)
        if return_spectral:
            row["albedo"] = spectral[-1][0]
            row["flx_slr"] = spectral[-1][1]
        results.append(row)

    result = SweepResult(results)
    # Keyed by row index so to_platform() stays aligned after sort_values(),
    # head(), or boolean filtering (a plain list zipped against a reordered
    # frame silently paired band values with the wrong rows).
    result._spectral = pd.Series(spectral, index=result.index)
    return result


def _validate_keys(params, input_file="default"):
    """Raise ValueError for any unrecognised parameter key."""
    imp_names = set(get_impurity_names(input_file))
    for key in params:
        if key in _VALID_KEYS:
            continue
        if key in imp_names:
            continue
        if _IMPURITY_CONC_RE.match(key):
            warnings.warn(
                f"{key!r} is deprecated; use the impurity name directly "
                f"(available: {sorted(imp_names)}).",
                DeprecationWarning,
                stacklevel=3,
            )
            continue
        raise ValueError(
            f"Unknown parameter key {key!r}. Supported keys: "
            f"{sorted(_VALID_KEYS)} and impurity names {sorted(imp_names)}."
        )


def _apply_params(combo_dict, ice, illumination, impurities, input_file):
    """Mutate model objects in-place according to a single parameter combination."""
    needs_irradiance = False
    needs_refractive = False

    # Build name→index mapping from impurities list
    imp_name_map = {imp.name: i for i, imp in enumerate(impurities)}

    for key, value in combo_dict.items():
        # Illumination scalars
        if key == "solzen":
            illumination.solzen = value
            needs_irradiance = True
        elif key == "direct":
            illumination.direct = value
            needs_irradiance = True
        elif key == "incoming":
            illumination.incoming = value
            needs_irradiance = True

        # Ice broadcast keys
        elif key in _ICE_BROADCAST_KEYS:
            broadcast = [value] * ice.nbr_lyr
            setattr(ice, key, broadcast)
            needs_refractive = True

        # Named impurity keys (e.g. black_carbon, glacier_algae).
        # First-layer convention, matching run_model(): a scalar
        # concentration goes in the surface layer only.  (This previously
        # broadcast to every layer, so sweep results were not comparable to
        # run_model() for the same nominal inputs on multi-layer columns.)
        elif key in imp_name_map:
            idx = imp_name_map[key]
            impurities[idx].conc = [value] + [0] * (ice.nbr_lyr - 1)

        # Legacy impurity.0.conc syntax (deprecated)
        else:
            m = _IMPURITY_CONC_RE.match(key)
            if m:
                idx = int(m.group(1))
                if idx >= len(impurities):
                    raise IndexError(
                        f"Impurity index {idx} out of range "
                        f"(only {len(impurities)} impurities configured)."
                    )
                # First-layer convention, matching run_model() (see above)
                impurities[idx].conc = [value] + [0] * (ice.nbr_lyr - 1)

    if needs_refractive:
        ice.calculate_refractive_index(input_file)
    if needs_irradiance:
        illumination.calculate_irradiance()


def _run_adding_doubling(tau, ssa, g, L_snw, ice, illumination, model_config, rt_config):
    return adding_doubling_solver(tau, ssa, g, L_snw, ice, illumination, model_config)


def _run_toon(tau, ssa, g, L_snw, ice, illumination, model_config, rt_config):
    return toon_solver(tau, ssa, g, L_snw, ice, illumination, model_config, rt_config)
