"""Tests for layer_type=5 (melt pond) optical properties and albedo.

Physical expectations
---------------------
A melt pond is a shallow layer of liquid water on sea ice.  Its optical
properties are dominated by water absorption:

  - Visible (400–700 nm): water is nearly transparent (k_water ≈ 10⁻⁹).
    Most visible light passes through and is reflected by the ice below.
    VIS albedo ≈ ice_VIS × exp(-2 × α × d) — still high for shallow ponds.

  - NIR (700–1000 nm): water absorbs strongly (k ≈ 10⁻⁸ – 10⁻³).
    Even a 5 cm pond reduces NIR albedo dramatically.

  - SWIR (>1000 nm): completely opaque even at < 1 cm depth.

Validation reference: Morassutti (1995), NSIDC G01169.
  Observed BBA by depth bin:
    0–5 cm:   ~0.37,  VIS ~0.51,  NIR ~0.22
    5–10 cm:  ~0.23,  VIS ~0.36,  NIR ~0.10
    10–30 cm: ~0.18,  VIS ~0.33,  NIR ~0.03–0.06

The model overestimates VIS for deeper ponds because it assumes a clear
white-ice pond bottom; real ponds often have dark or algae-covered bottoms.
NIR comparisons are more diagnostic and are well-matched.
"""

import numpy as np
import pytest

from biosnicar import run_model
from biosnicar.sea_ice.presets import FYI_POND_SHALLOW, FYI_POND_DEEP


# ---------------------------------------------------------------------------
# Helper: build a pond + sea ice column
# ---------------------------------------------------------------------------

def _pond_on_fyi(pond_depth_m: float, sza: int = 60) -> object:
    """Return run_model outputs for a pond of given depth on FYI."""
    return run_model(
        solzen=sza,
        layer_type=[5, 4, 4],
        dz=[pond_depth_m, 0.05, 1.45],
        rds=[500, 500, 500],
        rho=[1000, 895, 895],
        sea_ice_salinity=[None, 12, 8],
        sea_ice_temperature=[None, -5, -5],    # summer-ish FYI
        sea_ice_bubble_radius=[None, 100, 200],
    )


# ---------------------------------------------------------------------------
# Physical correctness
# ---------------------------------------------------------------------------

class TestPondLayerPhysics:
    def test_pond_reduces_bba_vs_bare_ice(self):
        bare = run_model(preset="FYI_WINTER_BARE", solzen=60)
        pond = _pond_on_fyi(0.20)
        assert pond.BBA < bare.BBA, "Pond must be darker than bare ice"

    def test_nir_drops_more_than_vis(self):
        """NIR is more strongly absorbed by water than visible."""
        bare = run_model(preset="FYI_WINTER_BARE", solzen=60)
        pond = _pond_on_fyi(0.20)
        vis_reduction = bare.BBAVIS - pond.BBAVIS
        nir_reduction = bare.BBANIR - pond.BBANIR
        assert nir_reduction > vis_reduction, (
            f"NIR reduction ({nir_reduction:.3f}) should exceed "
            f"VIS reduction ({vis_reduction:.3f})"
        )

    def test_deeper_pond_darker(self):
        """Albedo decreases monotonically with pond depth."""
        bbas = [_pond_on_fyi(d).BBA for d in [0.05, 0.10, 0.20, 0.40]]
        assert all(bbas[i] > bbas[i+1] for i in range(len(bbas)-1)), (
            "BBA must decrease monotonically with depth"
        )

    def test_nir_nearly_zero_deep_pond(self):
        """A 40 cm pond is essentially opaque in NIR."""
        out = _pond_on_fyi(0.40)
        assert out.BBANIR < 0.05, (
            f"NIR should be near 0 for 40cm pond, got {out.BBANIR:.3f}"
        )

    def test_vis_still_significant_shallow(self):
        """Shallow pond transmits most visible light to ice below."""
        out = _pond_on_fyi(0.05)
        assert out.BBAVIS > 0.50, (
            f"VIS should be > 0.5 for 5 cm pond (transparent), got {out.BBAVIS:.3f}"
        )

    def test_physical_range(self):
        for d in [0.05, 0.10, 0.20, 0.40]:
            out = _pond_on_fyi(d)
            assert 0.0 <= out.BBA <= 1.0
            assert np.all(out.albedo >= 0.0)
            assert np.all(out.albedo <= 1.0)


class TestPondNIRComparison:
    """Compare model NIR against Morassutti (1995) observations.

    The model uses clear water + white ice, so VIS is expected to be higher
    than observations (which include dark-bottom ponds). NIR is diagnostic.
    """

    def test_nir_10cm_pond_plausible(self):
        """10 cm pond NIR: model ~0.08–0.12, obs ~0.10 (Morassutti 5–10cm)."""
        out = _pond_on_fyi(0.10)
        # Observed NIR (700-1000 nm) for 5-10cm depth bin ≈ 0.10
        assert 0.04 < out.BBANIR < 0.20, (
            f"NIR for 10cm pond should be ~0.10, got {out.BBANIR:.3f}"
        )

    def test_nir_20cm_pond_plausible(self):
        """20 cm pond NIR: model ~0.04–0.08, obs ~0.03–0.06 (Morassutti 10–20cm)."""
        out = _pond_on_fyi(0.20)
        assert 0.02 < out.BBANIR < 0.12, (
            f"NIR for 20cm pond should be ~0.04–0.06, got {out.BBANIR:.3f}"
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class TestPondPresets:
    def test_shallow_preset_physical(self):
        out = run_model(preset=FYI_POND_SHALLOW, solzen=60)
        assert 0.0 <= out.BBA <= 1.0
        assert out.BBANIR < 0.20   # NIR absorbed by 10cm water

    def test_deep_preset_darker(self):
        shallow = run_model(preset=FYI_POND_SHALLOW, solzen=60)
        deep    = run_model(preset=FYI_POND_DEEP,    solzen=60)
        assert deep.BBA < shallow.BBA

    def test_deep_pond_nir_near_zero(self):
        out = run_model(preset=FYI_POND_DEEP, solzen=60)
        assert out.BBANIR < 0.05

    def test_preset_by_name(self):
        for name in ["FYI_POND_SHALLOW", "FYI_POND_DEEP"]:
            out = run_model(preset=name, solzen=60)
            assert 0.0 <= out.BBA <= 1.0

    def test_returns_outputs_object(self):
        from biosnicar.classes.outputs import Outputs
        out = run_model(preset=FYI_POND_SHALLOW, solzen=60)
        assert isinstance(out, Outputs)
        assert out.albedo.shape == (480,)
        # BBA alias works
        assert out.BBA == out.broadband


# ---------------------------------------------------------------------------
# Regression: terrestrial and sea ice unaffected
# ---------------------------------------------------------------------------

class TestPondRegressionExistingIce:
    def test_glacier_ice_unchanged(self):
        out = run_model(layer_type=1, dz=1.5, rds=500, rho=700, solzen=60)
        assert 0.3 < out.BBA < 1.0

    def test_sea_ice_unchanged(self):
        out = run_model(preset="FYI_WINTER_BARE", solzen=60)
        assert 0.4 <= out.BBA <= 0.9
