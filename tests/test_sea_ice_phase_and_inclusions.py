"""Melt-season phase model, consistent liquidus, and scattering by brine
inclusions and precipitated salts (Light, Maykut & Grenfell 2004)."""

import numpy as np
import pytest

from biosnicar.sea_ice import brine_volume as bv
from biosnicar.sea_ice import effective_medium as em
from biosnicar.sea_ice import inclusion_scattering as inc
from biosnicar.sea_ice.brine_optics import WAVELENGTHS, compute_brine_rfidx
from biosnicar.sea_ice.sea_ice_optics import _load_ice_ri, compute_sea_ice_optics

I589 = int(np.argmin(np.abs(WAVELENGTHS - 0.589)))


# ── Phase model ──────────────────────────────────────────────────────────────

class TestMeltSeasonBrineVolume:
    # brine_volume_cox83_lepparanta88 from SMRT v1.7.3, bulk density 900.
    @pytest.mark.parametrize("S,T,expected", [
        (5.0, -1.0, 0.2402), (5.0, -0.5, 0.4848), (2.0, -1.5, 0.0639),
    ])
    def test_lepparanta_manninen_branch(self, S, T, expected):
        # V_b = S * rho / F1 with the L&M F1 — check against the closed form.
        F1 = np.polyval([0.21454, 0.58402, -18.407, -0.041221], T)
        assert bv.compute_brine_volume(S, T, bulk_density_kg_m3=900.0) == \
            pytest.approx(S * 0.9 / F1, rel=1e-12)
        assert bv.compute_brine_volume(S, T, bulk_density_kg_m3=900.0) == \
            pytest.approx(expected, abs=5e-4)

    def test_monotonic_through_minus_two(self):
        T = np.linspace(-6.0, -0.6, 60)
        v = bv.compute_brine_volume(4.0, T, bulk_density_kg_m3=900.0)
        assert np.all(np.diff(v) > 0)

    def test_join_at_minus_two_is_small(self):
        lo = bv.compute_brine_volume(5.0, -2.0, bulk_density_kg_m3=900.0)
        hi = bv.compute_brine_volume(5.0, -1.999, bulk_density_kg_m3=900.0)
        assert hi == pytest.approx(lo, rel=0.015)

    def test_fully_melted_above_the_melting_point(self):
        assert bv.compute_brine_volume(12.0, -0.3, bulk_density_kg_m3=900.0) == 1.0
        assert bv.compute_gas_volume(12.0, -0.3, 900.0) == 0.0

    def test_melting_temperature_of_sea_water(self):
        """34 psu water freezes at -1.86 °C."""
        assert bv.melting_temperature(34.0) == pytest.approx(-1.86, abs=0.02)
        assert bv.melting_temperature(0.0) == 0.0


class TestLiquidusIsTheSamePhaseModel:
    @pytest.mark.parametrize("T", [-0.5, -1.0, -1.9, -2.0, -3.0, -5.0, -8.0])
    @pytest.mark.parametrize("S,rho", [(2.0, 870.0), (5.0, 900.0), (12.0, 850.0)])
    def test_salt_is_conserved_where_nothing_has_precipitated(self, T, S, rho):
        """S_i rho_si = S_b rho_b V_b, identically, above -8.2 °C."""
        vb = bv.compute_brine_volume(S, T, bulk_density_kg_m3=rho)
        if vb >= 1.0:
            pytest.skip("fully melted")
        sb = bv.brine_salinity_at_temp(T)
        in_brine = sb * (1.0 + 0.0008 * sb) * vb
        assert in_brine == pytest.approx(S * rho / 1000.0, rel=1e-9)

    def test_salt_leaves_the_brine_below_the_precipitation_points(self):
        def frac(T):
            sb = bv.brine_salinity_at_temp(T)
            vb = bv.compute_brine_volume(5.0, T, bulk_density_kg_m3=900.0)
            return sb * (1.0 + 0.0008 * sb) * vb / 4.5
        assert frac(-8.0) == pytest.approx(1.0, abs=1e-9)
        assert 0.85 < frac(-15.0) < 0.95
        assert frac(-30.0) < 0.35

    def test_continuous_at_the_joins(self):
        for T in (-8.2, -22.9):
            assert bv.brine_salinity_at_temp(T - 1e-6) == pytest.approx(
                bv.brine_salinity_at_temp(T + 1e-6), rel=2e-3)

    def test_monotonic(self):
        T = np.linspace(-30.0, 0.0, 301)
        assert np.all(np.diff(bv.brine_salinity_at_temp(T)) < 1e-9)

    def test_pure_water_at_zero(self):
        assert bv.brine_salinity_at_temp(0.0) == pytest.approx(0.0, abs=1e-6)

    def test_accepted_values(self):
        """Assur-based liquidus: ~37 at -2, ~87 at -5, ~143 at -10 °C."""
        assert bv.brine_salinity_at_temp(-2.0) == pytest.approx(37.0, abs=1.0)
        assert bv.brine_salinity_at_temp(-5.0) == pytest.approx(87.0, abs=2.0)
        assert bv.brine_salinity_at_temp(-10.0) == pytest.approx(143.0, abs=3.0)


class TestPrecipitatedSalts:
    def test_onsets(self):
        assert bv.mirabilite_mass_fraction(-8.2) == 0.0
        assert bv.mirabilite_mass_fraction(-8.3) > 0.0
        assert bv.hydrohalite_mass_fraction(-22.9) == 0.0
        assert bv.hydrohalite_mass_fraction(-23.5) > 0.0

    def test_mirabilite_anchor(self):
        """9e-4 kg per kg of 4.7 ppt sea ice at -15 °C (Light et al. 2004)."""
        assert bv.mirabilite_mass_fraction(-15.0) * 4.7e-3 == \
            pytest.approx(9.0e-4, rel=0.02)

    def test_mirabilite_bounded_by_the_sulphate_available(self):
        assert bv.mirabilite_mass_fraction(-30.0) < 0.259

    def test_hydrohalite_grows_as_brine_collapses(self):
        T = np.array([-23.0, -25.0, -28.0, -30.0])
        h = bv.hydrohalite_mass_fraction(T)
        assert np.all(np.diff(h) < 0) or np.all(np.diff(h[::-1]) < 0)
        assert h[-1] > h[0]


# ── Brine refractive index ───────────────────────────────────────────────────

class TestBrineIndexAgainstMaykutAndLight:
    """Measured 589 nm index of freezing brine: 1.341 at -2 °C, 1.397 at
    -32 °C (Maykut & Light 1995)."""

    def test_warm_end(self):
        assert compute_brine_rfidx(-2.0).real[I589] == pytest.approx(1.341, abs=0.002)

    def test_cold_end(self):
        assert compute_brine_rfidx(-30.0).real[I589] == pytest.approx(1.397, abs=0.004)

    def test_pure_water_at_zero(self):
        assert compute_brine_rfidx(0.0).real[I589] == pytest.approx(1.3340, abs=0.001)

    def test_relative_index_in_ice(self):
        """Light et al. (2004) Fig. 7a: ~1.025 at -2 °C to ~1.065 at -33 °C."""
        n_ice = _load_ice_ri().real[I589]
        assert compute_brine_rfidx(-2.0).real[I589] / n_ice == \
            pytest.approx(1.025, abs=0.003)
        assert compute_brine_rfidx(-30.0).real[I589] / n_ice == \
            pytest.approx(1.065, abs=0.004)

    def test_spectrum_is_continuous_at_the_window_edges(self):
        n = compute_brine_rfidx(-10.0).real
        for edge in (0.40, 0.70):
            i = int(np.argmin(np.abs(WAVELENGTHS - edge)))
            assert abs(n[i + 1] - n[i - 1]) < 0.003


# ── Effective medium ─────────────────────────────────────────────────────────

class TestBruggeman:
    EA, EB = np.array([1.72 + 1e-8j]), np.array([1.80 + 3e-8j])

    def test_endpoints_are_the_pure_phases(self):
        assert em.bruggeman(self.EA, self.EB, 0.0) == pytest.approx(self.EA)
        assert em.bruggeman(self.EA, self.EB, 1.0) == pytest.approx(self.EB)

    def test_symmetric_in_its_phases(self):
        for f in (0.1, 0.4, 0.7):
            assert em.bruggeman(self.EA, self.EB, f) == pytest.approx(
                em.bruggeman(self.EB, self.EA, 1.0 - f))

    def test_solves_its_own_equation(self):
        for f in (0.05, 0.3, 0.6, 0.95):
            e = em.bruggeman(self.EA, self.EB, f)
            res = ((1 - f) * (self.EA - e) / (self.EA + 2 * e)
                   + f * (self.EB - e) / (self.EB + 2 * e))
            assert abs(res[0]) < 1e-12

    def test_passive_medium(self):
        for f in np.linspace(0, 1, 11):
            assert em.bruggeman(self.EA, self.EB, f).imag[0] >= 0

    def test_agrees_with_maxwell_garnett_when_dilute(self):
        mg = em.maxwell_garnett(self.EA, self.EB, 0.03)
        assert em.bruggeman(self.EA, self.EB, 0.03) == pytest.approx(mg, rel=1e-5)


# ── Brine inclusions ─────────────────────────────────────────────────────────

class TestLightMicrostructure:
    """Numbers published in Light et al. (2003a, 2004) for -15 °C."""

    @staticmethod
    def _reference(l_max):
        l = np.geomspace(0.01, l_max, 200000)
        n = 0.28 * l ** -1.96
        v, a, pocket = inc._inclusion_geometry(l)
        return (np.trapezoid(n, l), np.trapezoid(n * v, l),
                np.trapezoid(n * a * pocket, l) / 4 * 1e3,
                np.trapezoid(n * a * ~pocket, l) / 4 * 1e3)

    def test_number_density(self):
        assert self._reference(8.0)[0] == pytest.approx(24.0, abs=0.5)

    def test_brine_volume_of_the_thin_section(self):
        assert self._reference(8.0)[1] == pytest.approx(0.012, abs=0.0005)

    def test_cross_sections_of_pockets_and_tubes(self):
        _, _, psi_p, psi_t = self._reference(8.0)
        assert psi_p == pytest.approx(30.0, abs=3.0)
        assert psi_t == pytest.approx(80.0, abs=4.0)

    def test_extended_distribution_holds_the_optical_samples_brine(self):
        """Tubes to 14.6 mm: 1.9% brine, tube cross-section 110 m^-1."""
        _, vol, _, psi_t = self._reference(14.6)
        assert vol == pytest.approx(0.019, abs=0.001)
        assert psi_t == pytest.approx(110.0, abs=6.0)

    def test_scattering_coefficient_at_minus_15(self):
        """sigma_bp + sigma_bt = 60 + 220 = 280 m^-1 for the 4.7 ppt sample."""
        nu = bv.compute_brine_volume(4.7, -15.0, bulk_density_kg_m3=915.0)
        psi, _ = inc.brine_inclusion_area(nu, nu, -15.0)
        assert 2.0 * psi == pytest.approx(280.0, rel=0.06)


class TestBrineInclusionArea:
    NU_REF = 0.0192

    def test_no_brine_no_scattering(self):
        assert inc.brine_inclusion_area(0.0, self.NU_REF, -5.0) == (0.0, 0.0)
        assert inc.brine_inclusion_area(0.01, 0.0, -5.0) == (0.0, 0.0)

    def test_area_grows_more_slowly_than_volume(self):
        """Inclusions swell: volume x4 gives less than x4 the area."""
        p1, r1 = inc.brine_inclusion_area(0.02, self.NU_REF, -20.0)
        p4, r4 = inc.brine_inclusion_area(0.08, self.NU_REF, -20.0)
        assert p1 < p4 < 4.0 * p1
        assert r4 > r1

    def test_equivalent_radius_conserves_volume_to_area(self):
        nu = 0.03
        psi, r_eq = inc.brine_inclusion_area(nu, self.NU_REF, -20.0)
        # psi = A/4 and r_eq = 3V/A  =>  psi * r_eq = 3V/4
        assert psi * r_eq * 1e-6 == pytest.approx(0.75 * nu, rel=1e-6)

    def test_merging_only_above_minus_14(self):
        cold = inc.brine_inclusion_area(0.02, self.NU_REF, -15.0)
        assert cold == inc.brine_inclusion_area(0.02, self.NU_REF, -15.0,
                                                merging=False)
        warm = inc.brine_inclusion_area(0.05, self.NU_REF, -5.0)
        unmerged = inc.brine_inclusion_area(0.05, self.NU_REF, -5.0,
                                            merging=False)
        assert warm[0] < unmerged[0]       # less interface
        assert warm[1] > unmerged[1]       # larger inclusions

    def test_merging_conserves_brine_volume(self):
        nu = 0.05
        psi, r_eq = inc.brine_inclusion_area(nu, self.NU_REF, -5.0)
        assert psi * r_eq * 1e-6 == pytest.approx(0.75 * nu, rel=0.02)

    def test_more_salt_means_more_interface(self):
        lo = inc.brine_inclusion_area(0.02, 0.02, -15.0)[0]
        hi = inc.brine_inclusion_area(0.06, 0.06, -15.0)[0]
        assert hi > lo

    def test_held_fixed_past_the_topology_limit(self):
        assert inc.brine_inclusion_area(0.6, self.NU_REF, -1.0) == \
            inc.brine_inclusion_area(0.9, self.NU_REF, -1.0)


class TestSoftSphereMie:
    def test_large_sphere_limit(self):
        """Light et al. (2004) Fig. 7b: g = 0.98 at m = 1.065, ~0.995 at
        m = 1.025; Q_sca = 2."""
        q, g = inc.soft_sphere_scattering(np.array([1.065, 1.025]), 1000.0)
        assert q == pytest.approx([2.0, 2.0], abs=0.1)
        assert g[0] == pytest.approx(0.982, abs=0.004)
        assert g[1] == pytest.approx(0.994, abs=0.003)

    def test_depends_on_contrast_not_its_sign(self):
        a = inc.soft_sphere_scattering(1.03, 300.0)
        b = inc.soft_sphere_scattering(0.97, 300.0)
        assert a[0] == pytest.approx(b[0]) and a[1] == pytest.approx(b[1])

    def test_no_contrast_no_scattering(self):
        q, _ = inc.soft_sphere_scattering(1.0, 100.0)
        assert q == pytest.approx(0.0, abs=1e-12)

    def test_small_phase_shift_scatters_weakly(self):
        """Micron-sized crystals are not in the Q = 2 regime."""
        q_small, _ = inc.soft_sphere_scattering(1.02, 20.0)
        q_large, _ = inc.soft_sphere_scattering(1.02, 2000.0)
        assert q_small < 0.5 * q_large


class TestSaltCrystalArea:
    def test_cube_formula(self):
        # 1e-3 volume fraction of 10 um cubes: psi = 1.5 * V / a = 150 m^-1
        psi = inc.salt_crystal_area(1.0, 1.0, 1000.0, 1000.0, 10.0)
        assert psi == pytest.approx(150.0)

    def test_mirabilite_cross_section_at_minus_15(self):
        """Light et al. (2004): psi = 76.3 m^-1 for the 4.7 ppt sample."""
        psi = inc.salt_crystal_area(bv.mirabilite_mass_fraction(-15.0), 4.7,
                                    915.0, inc.MIRABILITE_DENSITY, 10.0)
        assert psi == pytest.approx(76.3, rel=0.15)


# ── Layer optics ─────────────────────────────────────────────────────────────

class TestLayerOptics:
    def test_light_sample_similarity_parameter(self):
        """Interior first-year ice at -15 °C, 4.7 ppt, 0.915 Mg/m3: Light et
        al. report s = 8.7-9.3 m^-1 and g = 0.978-0.982 (Table 1).  The gas
        here is 300 um bubbles, which they did not have, so this is an
        order-of-magnitude check on the sum, and a real one on g."""
        tau, ssa, g = compute_sea_ice_optics(1.0, 4.7, -15.0, 915.0, 300.0)
        s = tau[I589] * ssa[I589] * (1.0 - g[I589])
        assert 5.0 < s < 20.0
        assert g[I589] == pytest.approx(0.98, abs=0.01)

    def test_brine_adds_scattering_to_bubble_free_ice(self):
        """With no air at all, saline ice still scatters."""
        rho_gas_free = 925.0
        tau, ssa, g = compute_sea_ice_optics(1.0, 6.0, -5.0, rho_gas_free, 300.0)
        tau0, ssa0, _ = compute_sea_ice_optics(1.0, 0.0, -5.0, 917.0, 300.0)
        assert tau[I589] * ssa[I589] > 50.0
        assert tau0[I589] * ssa0[I589] < 5.0
        assert g[I589] > 0.98

    def test_fresh_ice_is_unchanged_bubbly_ice(self):
        """S = 0: no brine, no salts — only the bubbles scatter."""
        tau, ssa, g = compute_sea_ice_optics(1.0, 0.0, -10.0, 880.0, 300.0)
        assert g[I589] == pytest.approx(0.86, abs=0.02)

    def test_warm_ice_is_accepted(self):
        tau, ssa, g = compute_sea_ice_optics(1.0, 4.0, -1.0, 880.0, 300.0)
        assert np.all(np.isfinite(tau)) and np.all((ssa > 0) & (ssa < 1))

    def test_hydrohalite_brightens_cold_ice(self):
        warm = compute_sea_ice_optics(1.0, 8.0, -22.0, 915.0, 300.0)
        cold = compute_sea_ice_optics(1.0, 8.0, -28.0, 915.0, 300.0)
        s = lambda r: r[0][I589] * r[1][I589] * (1.0 - r[2][I589])   # noqa: E731
        assert s(cold) > s(warm)


class TestLowerBoundary:
    def test_ocean_under_sea_ice(self):
        from biosnicar.drivers.run_model import run_model
        from biosnicar.sea_ice.open_water import ocean_subsurface_reflectance
        r = ocean_subsurface_reflectance()
        assert r.shape == (480,)
        assert r[25] < 0.06 and r[60] < 1e-3      # dark, black past the red
        thin = dict(layer_type=[4], dz=[0.10], rho=[900], rds=[500],
                    sea_ice_salinity=[6], sea_ice_temperature=[-5],
                    sea_ice_bubble_radius=[300], solzen=60)
        # Thin ice over ocean must be darker in the blue than the same ice
        # over the default blue-glacier-ice surface used for glacier columns.
        import sys
        rm = sys.modules["biosnicar.drivers.run_model"]
        over_ocean = np.asarray(run_model(**thin).albedo)
        saved = rm._set_lower_boundary
        try:
            rm._set_lower_boundary = lambda ice: None
            over_glacier = np.asarray(run_model(**thin).albedo)
        finally:
            rm._set_lower_boundary = saved
        assert over_ocean[25] < over_glacier[25]

    def test_glacier_columns_keep_their_surface(self):
        import sys
        from biosnicar.drivers.run_model import run_model
        rm = sys.modules["biosnicar.drivers.run_model"]
        a = run_model(layer_type=1, rds=500, rho=700, solzen=60).BBA
        saved = rm._set_lower_boundary
        try:
            rm._set_lower_boundary = lambda ice: None
            b = run_model(layer_type=1, rds=500, rho=700, solzen=60).BBA
        finally:
            rm._set_lower_boundary = saved
        assert a == b
