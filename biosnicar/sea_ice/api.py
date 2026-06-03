"""Sea ice API — compatibility shims only.

The primary interface for sea ice is :func:`biosnicar.run_model`, using
``layer_type=4`` layers with ``sea_ice_salinity``, ``sea_ice_temperature``,
and ``sea_ice_bubble_radius`` parameters, or the ``preset`` shortcut::

    from biosnicar import run_model, FYI_WINTER_BARE, FYI_WINTER_SNOW

    # By preset name
    outputs = run_model(preset="FYI_WINTER_BARE", solzen=60)

    # By preset dict
    outputs = run_model(preset=FYI_WINTER_BARE, solzen=60, black_carbon=500)

    # Fully explicit — same style as terrestrial ice
    outputs = run_model(
        solzen=60, direct=1, incoming=2,
        layer_type=[0, 4, 4],
        dz=[0.15, 0.05, 1.45],
        rds=[200, None, None],
        rho=[300, 895, 895],
        sea_ice_salinity=[None, 12, 8],
        sea_ice_temperature=[None, -25, -20],
        sea_ice_bubble_radius=[None, 100, 200],
        black_carbon=500,
    )

    # All outputs have the same attributes as terrestrial ice
    print(outputs.BBA)           # broadband albedo
    print(outputs.albedo)        # 480-band spectral albedo
    outputs.to_platform("sentinel2")

The classes in this module (``SeaIceColumn``, ``SeaIceLayer``,
``SnowLayer``, ``AlbedoResult``) are **deprecated** and will be removed
in v0.3.  They delegate internally to ``run_model()`` and exist only to
keep existing code working during the transition.
"""

import warnings
from typing import List, Optional, Union

import numpy as np

from biosnicar.sea_ice.presets import (
    FYI_WINTER_BARE,
    FYI_WINTER_SNOW,
    MYI_WINTER_BARE,
    _resolve_preset,
)


# ---------------------------------------------------------------------------
# Deprecated layer descriptors
# ---------------------------------------------------------------------------

class SnowLayer:
    """Deprecated. Pass dz/rho/rds directly to run_model()."""

    def __init__(self, thickness_m: float, density_kg_m3: float, grain_radius_um: float):
        warnings.warn(
            "SnowLayer is deprecated. Pass dz, rho, rds directly to run_model(). "
            "See docs/sea_ice.md for the updated API.",
            DeprecationWarning, stacklevel=2,
        )
        self.thickness_m     = thickness_m
        self.density_kg_m3   = density_kg_m3
        self.grain_radius_um = grain_radius_um


class SeaIceLayer:
    """Deprecated. Pass sea_ice_* kwargs directly to run_model()."""

    def __init__(
        self,
        thickness_m: float,
        temperature_C: float,
        salinity_psu: float,
        density_kg_m3: float = 895.0,
        bubble_radius_um: float = 200.0,
        layer_class: str = "FYI",
    ):
        warnings.warn(
            "SeaIceLayer is deprecated. Pass sea_ice_temperature, "
            "sea_ice_salinity, sea_ice_bubble_radius directly to run_model(). "
            "See docs/sea_ice.md for the updated API.",
            DeprecationWarning, stacklevel=2,
        )
        self.thickness_m     = thickness_m
        self.temperature_C   = temperature_C
        self.salinity_psu    = salinity_psu
        self.density_kg_m3   = density_kg_m3
        self.bubble_radius_um = bubble_radius_um
        self.layer_class     = layer_class


# ---------------------------------------------------------------------------
# Deprecated AlbedoResult — run_model() now returns Outputs directly
# ---------------------------------------------------------------------------

class AlbedoResult:
    """Deprecated wrapper. run_model() now returns Outputs directly.

    Outputs has all the same attributes: .BBA / .broadband, .BBAVIS /
    .visible, .BBANIR / .nir, .albedo / .spectrum, .wavelengths,
    .to_platform(), .plot(), etc.
    """

    def __init__(self, outputs):
        warnings.warn(
            "AlbedoResult is deprecated. run_model() and "
            "SeaIceColumn.compute_albedo() now return an Outputs object "
            "with .BBA / .broadband, .albedo / .spectrum attributes. "
            "See docs/sea_ice.md.",
            DeprecationWarning, stacklevel=2,
        )
        self._outputs = outputs

    @property
    def spectrum(self):
        return self._outputs.albedo

    @property
    def wavelengths(self):
        return np.arange(0.205, 4.999, 0.01)

    @property
    def broadband(self):
        return self._outputs.BBA

    @property
    def visible(self):
        return self._outputs.BBAVIS

    @property
    def nir(self):
        return self._outputs.BBANIR

    @property
    def outputs(self):
        return self._outputs

    def __repr__(self):
        return (
            f"AlbedoResult(broadband={self.broadband:.3f}, "
            f"visible={self.visible:.3f}, nir={self.nir:.3f}) [deprecated]"
        )


# ---------------------------------------------------------------------------
# Deprecated SeaIceColumn
# ---------------------------------------------------------------------------

class SeaIceColumn:
    """Deprecated. Use run_model(preset=...) or run_model(**flat_kwargs).

    This class is kept for backwards compatibility only and will be
    removed in v0.3.  It delegates internally to run_model().

    Migration guide
    ---------------
    Old::

        col = SeaIceColumn.from_preset(FYI_WINTER_BARE)
        result = col.compute_albedo(sza_deg=60)
        print(result.broadband)

    New::

        from biosnicar import run_model, FYI_WINTER_BARE
        outputs = run_model(preset=FYI_WINTER_BARE, solzen=60)
        print(outputs.BBA)   # or outputs.broadband
    """

    def __init__(self, layers: List[Union[SnowLayer, SeaIceLayer]]):
        warnings.warn(
            "SeaIceColumn is deprecated. Use run_model(preset=...) or "
            "run_model(**flat_kwargs) instead. See docs/sea_ice.md.",
            DeprecationWarning, stacklevel=2,
        )
        self.layers = layers

    @classmethod
    def from_preset(cls, preset) -> "SeaIceColumn":
        """Deprecated. Use run_model(preset=...) instead."""
        warnings.warn(
            "SeaIceColumn.from_preset() is deprecated. "
            "Use run_model(preset=...) instead.",
            DeprecationWarning, stacklevel=2,
        )
        # Build a SeaIceColumn from a preset dict without triggering the
        # SnowLayer/SeaIceLayer deprecation warnings internally.
        obj = object.__new__(cls)
        obj._preset_kwargs = _resolve_preset(preset)
        obj.layers = []   # unused, but kept for attribute access
        return obj

    def compute_albedo(
        self,
        solzen: float = 60.0,
        direct: int = 1,
        incoming: int = 2,
        snow_grain_radius_um: Optional[float] = None,
        # Legacy parameter names — accepted with deprecation warnings
        sza_deg: Optional[float] = None,
        sky: Optional[str] = None,
        atmosphere: Optional[str] = None,
        # Impurities
        black_carbon: float = 0,
        snow_algae: float = 0,
        glacier_algae: float = 0,
        dust: float = 0,
    ):
        """Deprecated. Use run_model() directly.

        Returns an Outputs object (same as run_model).
        """
        warnings.warn(
            "SeaIceColumn.compute_albedo() is deprecated. "
            "Use run_model(preset=..., solzen=...) instead.",
            DeprecationWarning, stacklevel=2,
        )

        # Resolve legacy parameter names
        if sza_deg is not None:
            warnings.warn("sza_deg is deprecated; use solzen", DeprecationWarning, stacklevel=2)
            solzen = sza_deg
        if sky is not None:
            warnings.warn(
                "sky='clear/cloudy' is deprecated; use direct=1 or direct=0",
                DeprecationWarning, stacklevel=2,
            )
            direct = 1 if sky == "clear" else 0
        if atmosphere is not None:
            _atm_map = {
                "mid_lat_winter": 0, "mid_lat_summer": 1,
                "sub_arctic_winter": 2, "sub_arctic_summer": 3,
                "summit": 4, "high_mountain": 5, "tropical": 6,
            }
            warnings.warn(
                "atmosphere='...' is deprecated; use incoming=<int 0-6>",
                DeprecationWarning, stacklevel=2,
            )
            incoming = _atm_map.get(atmosphere, 2)

        from biosnicar.drivers.run_model import run_model as _run_model

        # Build kwargs: start from preset (if from_preset was used) or layers
        if hasattr(self, "_preset_kwargs"):
            kwargs = self._preset_kwargs.copy()
        else:
            kwargs = _layers_to_kwargs(self.layers)

        # Apply snow grain radius override if provided
        if snow_grain_radius_um is not None and "rds" in kwargs:
            rds = kwargs["rds"]
            lt  = kwargs.get("layer_type", [0])
            kwargs["rds"] = [
                int(snow_grain_radius_um) if t == 0 else r
                for r, t in zip(rds, lt)
            ]

        # Illumination + impurities
        kwargs.update(solzen=solzen, direct=direct, incoming=incoming)
        if black_carbon:   kwargs["black_carbon"]  = black_carbon
        if snow_algae:     kwargs["snow_algae"]    = snow_algae
        if glacier_algae:  kwargs["glacier_algae"] = glacier_algae
        if dust:           kwargs["dust"]          = dust

        return _run_model(**kwargs)


# ---------------------------------------------------------------------------
# Internal helper: convert deprecated layer objects to run_model kwargs
# ---------------------------------------------------------------------------

def _layers_to_kwargs(layers: List[Union[SnowLayer, SeaIceLayer]]) -> dict:
    """Convert a list of SnowLayer/SeaIceLayer to flat run_model kwargs."""
    dz, lt, rho, rds = [], [], [], []
    s_sal, s_temp, s_bbl = [], [], []

    for lyr in layers:
        if isinstance(lyr, SnowLayer):
            dz.append(lyr.thickness_m)
            lt.append(0)
            rho.append(lyr.density_kg_m3)
            rds.append(int(lyr.grain_radius_um))
            s_sal.append(None)
            s_temp.append(None)
            s_bbl.append(None)
        elif isinstance(lyr, SeaIceLayer):
            dz.append(lyr.thickness_m)
            lt.append(4)
            rho.append(lyr.density_kg_m3)
            rds.append(500)   # unused for layer_type=4
            s_sal.append(lyr.salinity_psu)
            s_temp.append(lyr.temperature_C)
            s_bbl.append(lyr.bubble_radius_um)
        else:
            raise TypeError(f"Unknown layer type: {type(lyr)}")

    return dict(
        layer_type=lt, dz=dz, rds=rds, rho=rho,
        sea_ice_salinity=s_sal,
        sea_ice_temperature=s_temp,
        sea_ice_bubble_radius=s_bbl,
    )
