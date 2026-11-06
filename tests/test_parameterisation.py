"""Alternative (optical) parameterisations: SSA and brine volume fraction.

Radius and density are individually degenerate; temperature acts on the optics
only through brine volume.  A caller may supply either set, and the two must
produce the same spectrum.
"""
import numpy as np
import pytest

from biosnicar.drivers.run_model import run_model, _resolve_parameterisations
from biosnicar.sea_ice.brine_volume import compute_brine_volume
from biosnicar.sea_ice.parameterisation import (
    RHO_ICE,
    radius_from_ssa,
    specific_surface_area,
    ssa_density_sensitivity,
    topology_for_layer_type,
)


class TestSSAConventions:
    def test_granular_is_independent_of_bulk_density(self):
        """Snow SSA = 3/(r*rho_ice): the mass is the grains' own."""
        a = specific_surface_area(100, "granular")
        b = specific_surface_area(100, "granular", density_kg_m3=500)
        assert a == pytest.approx(b)
        assert a == pytest.approx(3.0 / (100e-6 * RHO_ICE))

    def test_bubbly_requires_density(self):
        with pytest.raises(ValueError, match="density_kg_m3 is required"):
            specific_surface_area(400, "bubbly")

    def test_conventions_differ(self):
        """The two formulas are not interchangeable — ~2x apart at rho=300."""
        gran = specific_surface_area(500, "granular")
        bub = specific_surface_area(500, "bubbly", density_kg_m3=300)
        assert bub / gran == pytest.approx(
            (1 - 300 / RHO_ICE) * RHO_ICE / 300)
        assert not np.isclose(gran, bub, rtol=0.5)

    @pytest.mark.parametrize("r,topo,rho", [
        (800, "granular", None), (400, "bubbly", 860.0), (700, "bubbly", 910.0),
    ])
    def test_round_trip_exact(self, r, topo, rho):
        ssa = specific_surface_area(r, topo, rho)
        assert radius_from_ssa(ssa, topo, rho) == pytest.approx(r, rel=1e-12)

    def test_layer_type_mapping(self):
        assert topology_for_layer_type(0) == "granular"
        for lt in (1, 2, 4):
            assert topology_for_layer_type(lt) == "bubbly"
        for lt in (3, 5, 6):
            with pytest.raises(ValueError, match="not defined for layer_type"):
                topology_for_layer_type(lt)

    def test_density_amplification_is_large_for_sea_ice(self):
        """SSA amplifies density error by 1/phi — 9-30x at sea-ice density,
        but barely at all for snow.  Documented so it is not forgotten."""
        assert ssa_density_sensitivity(820) == pytest.approx(9.45, rel=0.02)
        assert ssa_density_sensitivity(900) == pytest.approx(53.9, rel=0.02)
        assert ssa_density_sensitivity(300) < 2.0


class TestResolver:
    BASE = dict(layer_type=[0, 4, 4], dz=[0.05, 0.05, 1.40],
                rho=[300.0, 860.0, 910.0], sea_ice_salinity=[None, 6.0, 3.0])

    def test_ssa_resolves_to_the_right_key_per_layer(self):
        ssa = [specific_surface_area(800, "granular"),
               specific_surface_area(400, "bubbly", 860.0),
               specific_surface_area(700, "bubbly", 910.0)]
        out = _resolve_parameterisations(dict(self.BASE, ssa=ssa))
        assert "ssa" not in out
        # granular layer -> rds; sea-ice layers -> sea_ice_bubble_radius
        assert out["rds"][0] == pytest.approx(800.0)
        assert out["rds"][1] is None and out["rds"][2] is None
        assert out["sea_ice_bubble_radius"][0] is None
        assert out["sea_ice_bubble_radius"][1] == pytest.approx(400.0)
        assert out["sea_ice_bubble_radius"][2] == pytest.approx(700.0)

    def test_brine_volume_resolves_to_temperature(self):
        vb = [None,
              float(compute_brine_volume(6.0, -8.0, bulk_density_kg_m3=860.0)),
              float(compute_brine_volume(3.0, -8.0, bulk_density_kg_m3=910.0))]
        out = _resolve_parameterisations(
            dict(self.BASE, brine_volume_fraction=vb))
        assert "brine_volume_fraction" not in out
        assert out["sea_ice_temperature"][1] == pytest.approx(-8.0, abs=1e-3)
        assert out["sea_ice_temperature"][2] == pytest.approx(-8.0, abs=1e-3)

    def test_both_parameterisations_for_one_layer_is_an_error(self):
        with pytest.raises(ValueError, match="two ways of saying the same"):
            _resolve_parameterisations(dict(
                self.BASE, ssa=[None, 0.5, None],
                sea_ice_bubble_radius=[None, 400.0, None]))

    def test_ssa_without_layer_type_is_an_error(self):
        with pytest.raises(ValueError, match="requires `layer_type`"):
            _resolve_parameterisations({"ssa": [0.5], "rho": [860.0]})

    def test_bubbly_ssa_without_density_is_an_error(self):
        with pytest.raises(ValueError, match="needs `rho`"):
            _resolve_parameterisations(
                {"layer_type": [4], "ssa": [0.5]})

    def test_brine_volume_on_non_sea_ice_layer_is_an_error(self):
        with pytest.raises(ValueError, match="only meaningful for"):
            _resolve_parameterisations({
                "layer_type": [0], "rho": [300.0],
                "sea_ice_salinity": [6.0], "brine_volume_fraction": [0.05]})

    def test_absent_aliases_pass_through_untouched(self):
        d = {"layer_type": [4], "rho": [860.0]}
        assert _resolve_parameterisations(d) is d


class TestEquivalence:
    def test_optical_and_physical_give_the_same_spectrum(self):
        """The whole point: either set, same answer."""
        lt, rho = [0, 4, 4], [300.0, 860.0, 910.0]
        sal, T = [None, 6.0, 3.0], [None, -8.0, -8.0]
        common = dict(layer_type=lt, dz=[0.05, 0.05, 1.40], rho=rho,
                      sea_ice_salinity=sal, solzen=55, direct=1)
        phys = run_model(rds=[800.0, None, None],
                         sea_ice_temperature=T,
                         sea_ice_bubble_radius=[None, 400.0, 700.0], **common)
        ssa = [specific_surface_area(800, "granular"),
               specific_surface_area(400, "bubbly", 860.0),
               specific_surface_area(700, "bubbly", 910.0)]
        vb = [None] + [float(compute_brine_volume(
            sal[i], T[i], bulk_density_kg_m3=rho[i])) for i in (1, 2)]
        opt = run_model(ssa=ssa, brine_volume_fraction=vb, **common)
        assert np.max(np.abs(np.asarray(phys.albedo)
                             - np.asarray(opt.albedo))) < 1e-7
        assert opt.BBA == pytest.approx(phys.BBA, abs=1e-7)


class TestLUTRadiusRounding:
    def test_radius_rounds_rather_than_truncates(self):
        """A radius computed from a formula can land a hair under an integer.
        int() would silently pick the 799 um entry instead of 800."""
        from biosnicar.optical_properties.op_lookup import get_lut
        import biosnicar
        path = str(biosnicar.DATA_DIR / "OP_data" / "480band" / "luts"
                   / "bubbly_air.npz")
        lut = get_lut(path)
        r = int(lut.data["radii"][5])
        exact = lut.get(r, "asm_prm")
        nudged = lut.get(r - 1e-13, "asm_prm")
        assert np.array_equal(exact, nudged)
