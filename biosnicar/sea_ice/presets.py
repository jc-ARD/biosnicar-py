"""Pre-defined sea-ice column configurations for common scenarios.

All presets represent winter conditions (no surface melt).  Broadband
albedo for winter bare sea ice is expected in the range [0.4, 0.9].

Reference conditions:
  FYI_WINTER_BARE       — bare first-year ice, no snow cover
  FYI_WINTER_SNOW       — first-year ice with 15 cm snow layer
  MYI_WINTER_BARE       — bare multiyear ice, no snow cover

FYI (First-Year Ice) properties:
  - Higher salinity (6–12 psu surface, 4–8 psu bulk)
  - Higher density (910–920 kg/m³)
  - Small air bubbles (100–300 μm)

MYI (Multiyear Ice) properties:
  - Low salinity (1–3 psu; brine has drained over multiple summers)
  - Lower density (850–890 kg/m³; more air inclusions)
  - Larger air bubbles (500–1000 μm)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class SeaIceLayerSpec:
    """Physical specification of a single sea-ice layer."""
    thickness_m: float
    temperature_C: float
    salinity_psu: float
    density_kg_m3: float = 915.0
    bubble_radius_um: float = 200.0
    layer_class: str = "FYI"   # informational


@dataclass
class SnowLayerSpec:
    """Physical specification of a snow layer above sea ice."""
    thickness_m: float
    density_kg_m3: float
    grain_radius_um: float
    # v0.2: add salinity_psu for salty snow; MVP treats as fresh snow only


@dataclass
class SeaIcePreset:
    """Named preset configuration for a sea-ice column.

    Attributes:
        name:        Human-readable identifier.
        ice_layers:  List of SeaIceLayerSpec (top to bottom).
        snow_layer:  Optional snow layer above the ice (fresh snow in MVP).
    """
    name: str
    ice_layers: List[SeaIceLayerSpec]
    snow_layer: Optional[SnowLayerSpec] = None


# ---------------------------------------------------------------------------
# Concrete preset instances
# ---------------------------------------------------------------------------

FYI_WINTER_BARE = SeaIcePreset(
    name="FYI_WINTER_BARE",
    ice_layers=[
        SeaIceLayerSpec(
            thickness_m=0.05,
            temperature_C=-25.0,
            salinity_psu=12.0,
            density_kg_m3=920.0,
            bubble_radius_um=100.0,
            layer_class="FYI",
        ),
        SeaIceLayerSpec(
            thickness_m=1.45,
            temperature_C=-10.0,
            salinity_psu=8.0,
            density_kg_m3=915.0,
            bubble_radius_um=200.0,
            layer_class="FYI",
        ),
    ],
    snow_layer=None,
)

FYI_WINTER_SNOW = SeaIcePreset(
    name="FYI_WINTER_SNOW",
    ice_layers=[
        SeaIceLayerSpec(
            thickness_m=0.05,
            temperature_C=-25.0,
            salinity_psu=12.0,
            density_kg_m3=920.0,
            bubble_radius_um=100.0,
            layer_class="FYI",
        ),
        SeaIceLayerSpec(
            thickness_m=1.45,
            temperature_C=-10.0,
            salinity_psu=8.0,
            density_kg_m3=915.0,
            bubble_radius_um=200.0,
            layer_class="FYI",
        ),
    ],
    snow_layer=SnowLayerSpec(
        thickness_m=0.15,
        density_kg_m3=300.0,
        grain_radius_um=200.0,
    ),
)

MYI_WINTER_BARE = SeaIcePreset(
    name="MYI_WINTER_BARE",
    ice_layers=[
        SeaIceLayerSpec(
            thickness_m=0.30,
            temperature_C=-20.0,
            salinity_psu=1.0,
            density_kg_m3=870.0,
            bubble_radius_um=500.0,
            layer_class="MYI",
        ),
        SeaIceLayerSpec(
            thickness_m=2.70,
            temperature_C=-8.0,
            salinity_psu=3.0,
            density_kg_m3=880.0,
            bubble_radius_um=700.0,
            layer_class="MYI",
        ),
    ],
    snow_layer=None,
)

#: All built-in presets, accessible by name.
ALL_PRESETS: Dict[str, SeaIcePreset] = {
    p.name: p for p in [FYI_WINTER_BARE, FYI_WINTER_SNOW, MYI_WINTER_BARE]
}
