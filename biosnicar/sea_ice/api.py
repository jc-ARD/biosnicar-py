"""High-level Python API for constructing and running sea-ice columns.

Quickstart::

    from biosnicar.sea_ice.api import SeaIceColumn, SeaIceLayer
    from biosnicar.sea_ice.presets import FYI_WINTER_BARE

    # From a preset
    col = SeaIceColumn.from_preset(FYI_WINTER_BARE)
    result = col.compute_albedo(sza_deg=60)
    print(f"Broadband albedo: {result.broadband:.3f}")

    # Custom column
    col = SeaIceColumn(layers=[
        SeaIceLayer(thickness_m=0.05, temperature_C=-20, salinity_psu=10,
                    density_kg_m3=920, bubble_radius_um=100),
        SeaIceLayer(thickness_m=1.5, temperature_C=-10, salinity_psu=6,
                    density_kg_m3=910, bubble_radius_um=200),
    ])
    result = col.compute_albedo(sza_deg=55, sky="cloudy")

Note on snow layers (MVP limitation)
--------------------------------------
SnowLayer is treated as fresh (salinity = 0) snow in the MVP. Sea ice with
salty snow is deferred to v0.2.  The adding-doubling solver requires
layer_type=1 (granular snow) for snow layers; the snow is placed on top of
the sea ice column.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from biosnicar.sea_ice.presets import SeaIcePreset, SeaIceLayerSpec, SnowLayerSpec

# Atmosphere index → YAML incoming index mapping
_ATMOSPHERE_MAP = {
    "mid_lat_winter": 0,
    "mid_lat_summer": 1,
    "sub_arctic_winter": 2,
    "sub_arctic_summer": 3,
    "summit": 4,
    "high_mountain": 5,
    "tropical": 6,
}


@dataclass
class SnowLayer:
    """Fresh snow layer above sea ice.

    MVP limitation: treated as pure fresh-water snow (no brine wicking).
    Salty snow is deferred to v0.2.
    """
    thickness_m: float
    density_kg_m3: float
    grain_radius_um: float


@dataclass
class SeaIceLayer:
    """Single sea-ice layer with brine inclusions.

    Attributes:
        thickness_m:     Layer thickness in metres.
        temperature_C:   Temperature (°C, must be in [-44, -2]).
        salinity_psu:    Bulk salinity (psu, 0–15 typical).
        density_kg_m3:   Bulk density (kg/m³, 870–920 typical).
        bubble_radius_um: Effective air-bubble radius (μm, 100–1000).
        layer_class:     Informational tag ('FYI' or 'MYI').
    """
    thickness_m: float
    temperature_C: float
    salinity_psu: float
    density_kg_m3: float = 915.0
    bubble_radius_um: float = 200.0
    layer_class: str = "FYI"


@dataclass
class AlbedoResult:
    """Result of a sea-ice albedo calculation.

    Attributes:
        spectrum:     Spectral albedo array of length 480 (0.205–4.995 μm).
        wavelengths:  Wavelength grid in μm, length 480.
        broadband:    Flux-weighted broadband albedo.
        visible:      Flux-weighted visible (0.205–0.75 μm) albedo.
        nir:          Flux-weighted NIR (0.75–4.995 μm) albedo.
        outputs:      Raw BioSNICAR Outputs object (for advanced users).
    """
    spectrum: np.ndarray
    wavelengths: np.ndarray
    broadband: float
    visible: float
    nir: float
    outputs: object   # biosnicar.classes.outputs.Outputs

    def __repr__(self):
        return (
            f"AlbedoResult(broadband={self.broadband:.3f}, "
            f"visible={self.visible:.3f}, nir={self.nir:.3f})"
        )


class SeaIceColumn:
    """A sea-ice column with optional snow cover.

    Layers are ordered top-to-bottom.  A snow layer (if present) is prepended
    as a granular-snow (layer_type=0) layer above the sea-ice layers
    (layer_type=4).

    Args:
        layers: List of SeaIceLayer (and optionally a leading SnowLayer).
    """

    def __init__(self, layers: List[Union[SeaIceLayer, SnowLayer]]):
        self.layers = layers

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(cls, preset: SeaIcePreset) -> "SeaIceColumn":
        """Build a SeaIceColumn from a SeaIcePreset."""
        layer_list: List[Union[SeaIceLayer, SnowLayer]] = []
        if preset.snow_layer is not None:
            sl = preset.snow_layer
            layer_list.append(
                SnowLayer(
                    thickness_m=sl.thickness_m,
                    density_kg_m3=sl.density_kg_m3,
                    grain_radius_um=sl.grain_radius_um,
                )
            )
        for spec in preset.ice_layers:
            layer_list.append(
                SeaIceLayer(
                    thickness_m=spec.thickness_m,
                    temperature_C=spec.temperature_C,
                    salinity_psu=spec.salinity_psu,
                    density_kg_m3=spec.density_kg_m3,
                    bubble_radius_um=spec.bubble_radius_um,
                    layer_class=spec.layer_class,
                )
            )
        return cls(layer_list)

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------

    def compute_albedo(
        self,
        sza_deg: float = 60.0,
        atmosphere: str = "sub_arctic_winter",
        sky: str = "clear",
    ) -> AlbedoResult:
        """Run the BioSNICAR adding-doubling RT model and return albedo.

        Args:
            sza_deg:    Solar zenith angle in degrees (0–89).
            atmosphere: Atmospheric profile name.  One of:
                        'mid_lat_winter', 'mid_lat_summer',
                        'sub_arctic_winter' (default), 'sub_arctic_summer',
                        'summit', 'high_mountain', 'tropical'.
            sky:        'clear' (direct beam) or 'cloudy' (diffuse only).

        Returns:
            AlbedoResult with .spectrum, .broadband, .visible, .nir.
        """
        from biosnicar.drivers.setup_snicar import setup_snicar
        from biosnicar.optical_properties.column_OPs import get_layer_OPs, mix_in_impurities
        from biosnicar.rt_solvers.adding_doubling_solver import adding_doubling_solver

        incoming = _ATMOSPHERE_MAP.get(atmosphere, 2)
        direct = 1 if sky == "clear" else 0

        # ---- Build lists for BioSNICAR Ice object ----
        dz, layer_type, rho = [], [], []
        rds, shp, cdom, water = [], [], [], []
        hex_side, hex_length, shp_fctr, grain_ar, lwc = [], [], [], [], []
        sea_salinity, sea_temperature, sea_bubble = [], [], []

        for lyr in self.layers:
            if isinstance(lyr, SnowLayer):
                dz.append(lyr.thickness_m)
                layer_type.append(0)            # granular snow
                rho.append(lyr.density_kg_m3)
                rds.append(int(lyr.grain_radius_um))
                shp.append(0)
                cdom.append(0)
                water.append(0)
                hex_side.append(10000)
                hex_length.append(10000)
                shp_fctr.append(0)
                grain_ar.append(0)
                lwc.append(0)
                sea_salinity.append(None)
                sea_temperature.append(None)
                sea_bubble.append(None)
            elif isinstance(lyr, SeaIceLayer):
                dz.append(lyr.thickness_m)
                layer_type.append(4)            # sea ice
                rho.append(lyr.density_kg_m3)
                rds.append(500)                 # not used for layer_type=4
                shp.append(0)
                cdom.append(0)
                water.append(0)
                hex_side.append(10000)
                hex_length.append(10000)
                shp_fctr.append(0)
                grain_ar.append(0)
                lwc.append(0)
                sea_salinity.append(lyr.salinity_psu)
                sea_temperature.append(lyr.temperature_C)
                sea_bubble.append(lyr.bubble_radius_um)
            else:
                raise TypeError(f"Unknown layer type: {type(lyr)}")

        # ---- Setup BioSNICAR objects from default config ----
        input_file = str(
            Path(__file__).resolve().parent.parent / "inputs.yaml"
        )
        ice, illumination, rt_config, model_config, plot_config, impurities = (
            setup_snicar(input_file)
        )

        # Overwrite ice attributes
        ice.nbr_lyr = len(dz)
        ice.dz = dz
        ice.layer_type = layer_type
        ice.rho = rho
        ice.rds = rds
        ice.shp = shp
        ice.cdom = cdom
        ice.water = water
        ice.hex_side = hex_side
        ice.hex_length = hex_length
        ice.shp_fctr = shp_fctr
        ice.grain_ar = grain_ar
        ice.lwc = lwc
        ice.lwc_pct_bbl = [0] * len(dz)
        ice.sea_ice_salinity = sea_salinity
        ice.sea_ice_temperature = sea_temperature
        ice.sea_ice_bubble_radius = sea_bubble

        # Zero all impurity concentrations
        for imp in impurities:
            imp.conc = [0] * len(dz)

        # Override illumination
        illumination.solzen = sza_deg
        illumination.direct = direct
        illumination.incoming = incoming
        illumination.calculate_irradiance()

        # Recalculate refractive indices (needed after layer_type change)
        ice.calculate_refractive_index(input_file)

        # ---- Run RT ----
        ssa_snw, g_snw, mac_snw = get_layer_OPs(ice, model_config)
        tau, ssa, g, L_snw = mix_in_impurities(
            ssa_snw, g_snw, mac_snw, ice, impurities, model_config
        )
        outputs = adding_doubling_solver(tau, ssa, g, L_snw, ice, illumination, model_config)

        wvl = model_config.wavelengths
        return AlbedoResult(
            spectrum=np.array(outputs.albedo),
            wavelengths=wvl,
            broadband=float(outputs.BBA),
            visible=float(outputs.BBAVIS),
            nir=float(outputs.BBANIR),
            outputs=outputs,
        )
