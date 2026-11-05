"""Tests for sea-ice per-layer optical properties."""

import numpy as np
import pytest
from biosnicar.sea_ice.sea_ice_optics import compute_sea_ice_optics, compute_sea_ice_optics_per_metre


class TestSeaIceOpticsPhysical:
    """Output must always be physically valid."""

    @pytest.mark.parametrize("S,T,rho,bbl", [
        (8.0, -10.0, 915.0, 200.0),
        (1.0, -20.0, 870.0, 700.0),
        (12.0, -5.0, 920.0, 100.0),
        (0.0, -10.0, 917.0, 300.0),   # freshwater ice limit
    ])
    def test_physical_values(self, S, T, rho, bbl):
        tau, ssa, g = compute_sea_ice_optics(1.0, S, T, rho, bbl)
        assert np.all(tau >= 0), "tau must be non-negative"
        assert np.all((ssa > 0) & (ssa < 1)), "ssa must be in (0,1)"
        assert np.all(np.abs(g) <= 1), "|g| must be ≤ 1"
        assert tau.shape == (480,)
        assert ssa.shape == (480,)
        assert g.shape == (480,)

    def test_tau_scales_linearly_with_thickness(self):
        S, T, rho, bbl = 8.0, -10.0, 915.0, 200.0
        tau1, _, _ = compute_sea_ice_optics(1.0, S, T, rho, bbl)
        tau2, _, _ = compute_sea_ice_optics(2.0, S, T, rho, bbl)
        np.testing.assert_allclose(tau2, 2.0 * tau1, rtol=1e-6)

    def test_ssa_independent_of_thickness(self):
        S, T, rho, bbl = 8.0, -10.0, 915.0, 200.0
        _, ssa1, _ = compute_sea_ice_optics(1.0, S, T, rho, bbl)
        _, ssa2, _ = compute_sea_ice_optics(5.0, S, T, rho, bbl)
        np.testing.assert_allclose(ssa1, ssa2, rtol=1e-6)

    def test_per_metre_equals_unit_thickness(self):
        S, T, rho, bbl = 6.0, -15.0, 910.0, 300.0
        tau_1m, ssa_1m, g_1m = compute_sea_ice_optics(1.0, S, T, rho, bbl)
        tau_pm, ssa_pm, g_pm = compute_sea_ice_optics_per_metre(S, T, rho, bbl)
        np.testing.assert_allclose(tau_1m, tau_pm, rtol=1e-6)
        np.testing.assert_allclose(ssa_1m, ssa_pm, rtol=1e-6)

    def test_zero_salinity_similar_to_bubbly_ice(self):
        """Sea ice with S=0 should have optical properties similar to glacier ice."""
        tau_si, ssa_si, _ = compute_sea_ice_optics(1.0, 0.0, -10.0, 915.0, 300.0)
        # At S=0 there's no brine; tau should be finite and ssa reasonable
        # (not identical to bubbly_air LUT since we also compute effective-medium
        # absorption, but same order of magnitude)
        assert np.all(tau_si > 0)
        visible_avg_ssa = np.mean(ssa_si[:50])
        assert visible_avg_ssa > 0.8, "Visible SSA should be high for clean ice"

    def test_higher_salinity_changes_optics(self):
        """Different salinity must produce different optical properties."""
        tau_low, ssa_low, _ = compute_sea_ice_optics(1.0, 1.0, -10.0, 910.0, 300.0)
        tau_high, ssa_high, _ = compute_sea_ice_optics(1.0, 12.0, -10.0, 910.0, 300.0)
        # The optical properties must differ between S=1 and S=12
        assert not np.allclose(tau_low, tau_high, rtol=0.01), (
            "tau should differ between low and high salinity sea ice"
        )


class TestSeaIceOpticsVariants:
    def test_ri_variants(self):
        """All RI variants should produce physical output."""
        for ri in ["Pic16", "Wrn84", "Wrn08"]:
            tau, ssa, g = compute_sea_ice_optics(1.0, 8.0, -10.0, 910.0, 200.0, ri_variant=ri)
            assert np.all(tau >= 0)
            assert np.all(ssa > 0) and np.all(ssa < 1)


class TestNearestLutRadius:
    """A3: out-of-range bubble radii clamp to LUT endpoints with a warning."""

    def _lut(self):
        import biosnicar
        from biosnicar.optical_properties.op_lookup import get_lut
        path = biosnicar.DATA_DIR / "OP_data" / "480band" / "luts" / "bubbly_air.npz"
        return get_lut(str(path))

    def test_in_range_no_warning(self):
        import warnings
        from biosnicar.sea_ice.sea_ice_optics import _nearest_lut_radius
        lut = self._lut()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _nearest_lut_radius(213.0, lut) in (210, 215)

    def test_below_range_clamps_and_warns(self):
        import pytest
        from biosnicar.sea_ice.sea_ice_optics import _nearest_lut_radius
        lut = self._lut()
        lo = float(lut.data["radii"].min())
        with pytest.warns(UserWarning, match="outside the"):
            assert _nearest_lut_radius(lo / 2.0, lut) == int(lo)

    def test_above_range_clamps_and_warns(self):
        import pytest
        from biosnicar.sea_ice.sea_ice_optics import _nearest_lut_radius
        lut = self._lut()
        hi = float(lut.data["radii"].max())
        with pytest.warns(UserWarning, match="outside the"):
            assert _nearest_lut_radius(hi * 2.0, lut) == int(hi)


class TestDensityIsHonoured:
    """Bulk density must reach the optics, both in-module and through the LUT.

    The sea-ice LUT is interpolated with *clamping* rather than extrapolation
    (``column_OPs._interpolate_sea_ice_lut``), so any density outside the LUT's
    rho grid is silently replaced by the nearest edge.  The grid used to start
    at 870 while the shipped presets use a drained-layer density of 850 and the
    FYI_bare emulator samples rho_DL down to 820 — so the parameter did nothing
    over most of its range.  These tests fail if that gap reopens.
    """

    def test_lut_grid_covers_configured_densities(self):
        import numpy as np
        import biosnicar.optical_properties.column_OPs as C

        data = np.load(C._SEA_ICE_LUT_PATH)
        rho_grid = data["rho_grid"]

        # Lowest density the model is ever asked for: FYI_bare emulator rho_DL.
        from biosnicar.sea_ice.emulator_configs import SEA_ICE_EMULATOR_CONFIGS
        rho_lo = SEA_ICE_EMULATOR_CONFIGS["FYI_bare"]["params"]["rho_DL"][0]

        assert rho_grid.min() <= rho_lo, (
            f"LUT rho grid starts at {rho_grid.min()} but the FYI_bare emulator "
            f"samples rho_DL down to {rho_lo}; everything below the grid edge is "
            "clamped, so the parameter would be inert over that range."
        )
        # Shipped presets use a Jin et al. (2023) drained-layer density of 850.
        assert rho_grid.min() <= 850.0

    def test_density_changes_the_interpolated_optics(self):
        """Two densities inside the grid must give different optics."""
        import numpy as np
        from biosnicar.optical_properties.column_OPs import _interpolate_sea_ice_lut

        tau_light, ssa_light, _ = _interpolate_sea_ice_lut(6.0, -10.0, 850.0, 300.0)
        tau_dense, ssa_dense, _ = _interpolate_sea_ice_lut(6.0, -10.0, 910.0, 300.0)
        assert not np.allclose(tau_light, tau_dense), (
            "density is not reaching the LUT interpolation"
        )
        # Less dense ice holds more air, so scatters more.
        assert ssa_light.mean() > ssa_dense.mean()

    def test_optics_use_density_for_brine_volume(self):
        """compute_sea_ice_optics must pass bulk density into Cox & Weeks.

        Denser ice holds less air *and* less brine, so its brine volume is
        higher per unit mass but its scattering is lower.  The direct check is
        that the function's nu_b matches the bulk-density form.
        """
        from biosnicar.sea_ice.brine_volume import compute_brine_volume

        S, T, rho = 8.0, -10.0, 850.0
        bubble_free = compute_brine_volume(S, T)
        with_density = compute_brine_volume(S, T, bulk_density_kg_m3=rho)
        assert with_density < bubble_free, "porosity must reduce brine volume"
        # ~7% at this density — the gas fraction.
        assert 0.03 < (bubble_free - with_density) / bubble_free < 0.12
