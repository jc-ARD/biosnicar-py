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

    def test_nir_of_deep_pond_is_surface_reflection_only(self):
        """A 40 cm pond is opaque in the NIR, so nothing returns from below
        and NIR albedo falls to the specular reflection off the water
        surface (~6% at SZA 60) — not to zero, which is what a pond with no
        surface interface would give."""
        out = _pond_on_fyi(0.40)
        assert 0.03 < out.BBANIR < 0.10, (
            f"NIR of a 40 cm pond should be ~ the water-surface Fresnel "
            f"reflectance, got {out.BBANIR:.3f}"
        )
        beyond_1p2um = np.asarray(out.albedo)[100:230]
        assert beyond_1p2um.max() - beyond_1p2um.min() < 0.02

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

    def test_deep_pond_nir_is_surface_reflection_only(self):
        out = run_model(preset=FYI_POND_DEEP, solzen=60)
        assert 0.03 < out.BBANIR < 0.10

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


# ── Refractive boundary of a ponded column ──────────────────────────────────

class TestPondSurfaceInterface:
    """The smooth interface of a pond is its water surface, not the ice floor.

    For a column [5, 4, 4] the Fresnel boundary used to land on the ice under
    the pond and be computed as air->ice, as though the water were absent.
    The signature of that error is unambiguous: wherever the pond is
    optically thick the only light that can come back is the specular
    reflection off the water, so a pond with no surface interface has zero
    NIR albedo.  Real water reflects ~2% at normal incidence.
    """

    _POND = dict(
        layer_type=[5, 4, 4], dz=[0.30, 0.05, 1.40], rds=[500, 500, 500],
        rho=[1000, 850, 910], sea_ice_salinity=[None, 8, 6],
        sea_ice_temperature=[None, -3, -3],
        sea_ice_bubble_radius=[None, 300, 750],
    )
    # 1.6-1.8 um: 30 cm of water is opaque, so albedo == surface reflectance.
    _OPAQUE = slice(140, 160)

    @staticmethod
    def _fresnel_unpolarised(n, sza_deg):
        mu = np.cos(np.radians(sza_deg))
        mut = np.sqrt(1.0 - (1.0 - mu ** 2) / n ** 2)
        rs = ((mu - n * mut) / (mu + n * mut)) ** 2
        rp = ((n * mu - mut) / (n * mu + mut)) ** 2
        return 0.5 * (rs + rp)

    @pytest.mark.parametrize("sza", [30, 50, 70])
    def test_opaque_band_albedo_is_water_fresnel(self, sza):
        import pandas as pd
        import biosnicar
        n_w = pd.read_csv(
            biosnicar.DATA_DIR / "OP_data" / "480band"
            / "refractive_index_water_273K_Rowe2020.csv"
        ).n.values[self._OPAQUE]
        out = run_model(solzen=sza, direct=1, **self._POND)
        alb = np.asarray(out.albedo)[self._OPAQUE]
        assert alb == pytest.approx(self._fresnel_unpolarised(n_w, sza),
                                    abs=3e-3)

    def test_interface_is_on_the_pond_layer(self):
        from biosnicar.rt_solvers.adding_doubling_solver import (
            _first_fresnel_layer,
        )

        class _Ice:
            layer_type = [5, 4, 4]
        assert _first_fresnel_layer(_Ice) == 0

    def test_no_spurious_total_reflection_near_3um(self):
        # Water's real index stays above 1.11, so air->water never totally
        # reflects.  The complex-arcsin criterion used for ice would fire at
        # SZA >~ 69 deg and paint albedo = 1 over 2.9-3.2 um.
        out = run_model(solzen=75, direct=1, **self._POND)
        assert np.asarray(out.albedo)[250:310].max() < 0.5

    def test_diffuse_fresnel_is_exact(self):
        from biosnicar.classes.ice import diffuse_fresnel
        # CICE's constants for n = 1.31 (Briegleb & Light 2007).
        ra, rb = diffuse_fresnel(np.array([1.31]))
        assert ra[0] == pytest.approx(0.063, abs=1e-3)
        assert rb[0] == pytest.approx(0.455, abs=2e-3)
        # Radiance conservation ties the two directions together exactly.
        n = np.linspace(1.2, 1.45, 11)
        ra, rb = diffuse_fresnel(n)
        assert (1 - rb) * n ** 2 == pytest.approx(1 - ra, rel=1e-12)
        # No interface when the media match.
        ra, rb = diffuse_fresnel(np.array([1.0 + 1e-9]))
        assert ra[0] == pytest.approx(0.0, abs=1e-6)

    def test_from_above_agrees_with_the_inherited_ice_table(self):
        from pathlib import Path
        import biosnicar
        from biosnicar.classes.ice import diffuse_fresnel
        from biosnicar.drivers.setup_snicar import setup_snicar
        ice = setup_snicar(str(Path(biosnicar.__file__).parent / "inputs.yaml"))[0]
        vis_nir = slice(20, 200)
        ra, _ = diffuse_fresnel(ice.ref_idx_re[vis_nir])
        assert ra == pytest.approx(ice.fl_r_dif_a[vis_nir], abs=1e-6)
        # Water is optically denser than ice across the solar spectrum.
        assert (ice.fl_r_dif_a_water[vis_nir] > ice.fl_r_dif_a[vis_nir]).all()


class TestFresnelLayerOverride:
    _COL = dict(
        layer_type=[0, 4, 4], dz=[0.05, 0.05, 1.40], rds=[1000, 500, 500],
        rho=[300, 850, 910], sea_ice_salinity=[None, 4, 2],
        sea_ice_temperature=[None, -3, -3],
        sea_ice_bubble_radius=[None, 300, 600], solzen=60,
    )

    def test_auto_is_the_default(self):
        a = run_model(**self._COL).albedo
        b = run_model(fresnel_layer="auto", **self._COL).albedo
        assert np.array_equal(a, b)

    def test_explicit_index_matches_auto_rule(self):
        # auto puts the boundary on the first sea-ice layer: index 1 here.
        a = run_model(**self._COL).albedo
        b = run_model(fresnel_layer=1, **self._COL).albedo
        assert np.array_equal(a, b)

    def test_none_removes_the_interface(self):
        with_ifc = run_model(**self._COL)
        without = run_model(fresnel_layer=None, **self._COL)
        assert not np.allclose(with_ifc.albedo, without.albedo)
        # No internal reflection trapping light under the grains: brighter.
        assert without.BBA > with_ifc.BBA

    def test_solid_ice_type1_equals_type2_with_interface_on_top(self):
        col = dict(self._COL, layer_type=[1, 4, 4], rho=[700, 850, 910])
        a = run_model(**col).albedo
        b = run_model(fresnel_layer=0,
                      **dict(col, layer_type=[2, 4, 4])).albedo
        assert np.allclose(a, b)

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError, match="fresnel_layer"):
            run_model(fresnel_layer=7, **self._COL)


class TestBrineOnlyInSeaIceLayers:
    def test_salinity_on_granular_layer_warns(self):
        with pytest.warns(UserWarning, match="sea_ice_salinity is ignored"):
            run_model(
                layer_type=[0, 4], dz=[0.1, 1.0], rds=[300, None],
                rho=[300, 900], sea_ice_salinity=[5, 6],
                sea_ice_temperature=[None, -10],
                sea_ice_bubble_radius=[None, 200],
            )

    def test_none_or_zero_salinity_is_silent(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run_model(preset="FYI_WINTER_SNOW")
            run_model(preset="FYI_POND_SHALLOW")
            run_model(
                layer_type=[0, 4], dz=[0.1, 1.0], rds=[300, None],
                rho=[300, 900], sea_ice_salinity=[0, 6],
                sea_ice_temperature=[None, -10],
                sea_ice_bubble_radius=[None, 200],
            )
