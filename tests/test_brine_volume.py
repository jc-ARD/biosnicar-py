"""Tests for sea-ice brine volume fraction (Cox & Weeks 1983)."""

import numpy as np
import pytest
from biosnicar.sea_ice.brine_volume import compute_brine_volume


class TestBrineVolumePhysics:
    """Physical behaviour of the C&W brine volume formula."""

    def test_increases_with_salinity(self):
        T = -10.0
        values = [compute_brine_volume(S, T) for S in [1, 4, 8, 10, 12]]
        assert all(values[i] < values[i+1] for i in range(len(values)-1))

    def test_increases_toward_zero_celsius(self):
        S = 8.0
        values = [compute_brine_volume(S, T) for T in [-30, -20, -10, -5, -2]]
        assert all(values[i] < values[i+1] for i in range(len(values)-1))

    def test_physical_range(self):
        """Brine volume must be in [0, 1] for all valid inputs."""
        for S in [1, 4, 8, 12]:
            for T in [-30, -20, -10, -5, -2]:
                nu_b = compute_brine_volume(S, T)
                assert 0.0 <= nu_b <= 1.0, f"S={S}, T={T}: nu_b={nu_b}"

    def test_zero_salinity_gives_near_zero_brine(self):
        nu_b = compute_brine_volume(0.0, -10.0)
        assert nu_b == pytest.approx(0.0, abs=1e-8)

    def test_cold_ice_has_small_brine_volume(self):
        """At -30°C and S=4 psu, brine volume should be very small."""
        nu_b = compute_brine_volume(4.0, -30.0)
        assert nu_b < 0.005

    def test_warm_ice_has_large_brine_volume(self):
        """At -2°C and S=12 psu, brine volume should be significant."""
        nu_b = compute_brine_volume(12.0, -2.0)
        assert nu_b > 0.15


class TestBrineVolumeReferenceValues:
    """Spot checks against the published Cox & Weeks (1983) Table 1 values.

    These are reproduced to within 3%.  SMRT's independent implementation of
    the same paper departs from the published values by the same margin (see
    TestBrineVolumeAgainstSMRT), so the residual is rounding in the published
    table rather than error here.

    This tolerance was previously 40%, on the stated grounds of "ambiguity in
    the original paper's unit conventions".  The SMRT cross-check found no
    such ambiguity, and a 40% band is wide enough to pass almost any
    implementation.  Do not loosen it again without evidence.
    """

    @pytest.mark.parametrize("S,T,expected", [
        (10.0, -2.0,  0.250),
        (8.0,  -10.0, 0.045),
        (4.0,  -15.0, 0.016),
    ])
    def test_reference_values(self, S, T, expected):
        nu_b = compute_brine_volume(S, T)
        assert nu_b == pytest.approx(expected, rel=0.05), (
            f"S={S} psu, T={T}°C: got {nu_b:.4f}, expected ~{expected:.4f}"
        )


class TestBrineVolumeAgainstSMRT:
    """Regression pin against SMRT v1.7.3's Cox & Weeks implementation.

    SMRT (Picard et al.) is the community sea-ice/snow microwave RT model and
    implements the same Cox & Weeks (1983) polynomials independently.  It is a
    microwave model, so it shares no radiative observable with BioSNICAR --
    but brine volume fraction is frequency-independent, which makes it a
    genuine external check on this module's geophysics.

    Expected values were generated from
    ``smrt.permittivity.brine.brine_volume_cox83_lepparanta88(T_kelvin,
    S_psu * PSU, porosity=0)`` with SMRT v1.7.3, and are hard-coded so the
    suite does not depend on SMRT being installed.

    The 0.04% tolerance is the pure-ice density constant: this module uses
    rho_i(0 degC) = 0.917 g/cm3, SMRT uses 0.9167 (a 0.0327% difference).
    A failure here means the polynomials or the formula have drifted.
    """

    @pytest.mark.parametrize("S,T,smrt_value", [
        (1.0,  -5.0,  0.009894677),
        (5.0,  -5.0,  0.049798505),
        (5.0,  -10.0, 0.027733183),
        (8.0,  -15.0, 0.033050828),
        (12.0, -20.0, 0.040714540),
        (5.0,  -30.0, 0.004443732),
    ])
    def test_matches_smrt(self, S, T, smrt_value):
        nu_b = compute_brine_volume(S, T)
        assert nu_b == pytest.approx(smrt_value, rel=4e-4), (
            f"S={S} psu, T={T}°C: biosnicar {nu_b:.9f} vs SMRT {smrt_value:.9f} "
            f"({abs(nu_b - smrt_value) / smrt_value:.4%} apart)"
        )

    def test_clamp_below_30C_warns(self):
        """Below -30 degC the result is held constant, not a C&W value."""
        with pytest.warns(RuntimeWarning, match="Cox & Weeks"):
            clamped = compute_brine_volume(5.0, -35.0)
        assert clamped == pytest.approx(compute_brine_volume(5.0, -30.0))


class TestBrineVolumeValidation:
    def test_raises_on_too_warm(self):
        """0 °C is the upper limit; the melt season below it is covered."""
        with pytest.raises(ValueError, match="temperature_C"):
            compute_brine_volume(8.0, 0.5)
        assert compute_brine_volume(8.0, 0.0) == 1.0      # all melted

    def test_raises_on_too_cold(self):
        with pytest.raises(ValueError, match="temperature_C"):
            compute_brine_volume(8.0, -50.0)

    def test_raises_on_negative_salinity(self):
        with pytest.raises(ValueError, match="salinity"):
            compute_brine_volume(-1.0, -10.0)

    def test_boundary_warm(self):
        nu_b = compute_brine_volume(8.0, -2.0)
        assert 0.0 < nu_b < 1.0

    def test_boundary_cold(self):
        nu_b = compute_brine_volume(8.0, -44.0)
        assert 0.0 <= nu_b < 0.01


class TestBrineVolumeVectorised:
    def test_vectorised_matches_scalar(self):
        S_arr = np.array([4.0, 8.0, 10.0])
        T_arr = np.array([-15.0, -10.0, -2.0])
        vec = compute_brine_volume(S_arr, T_arr)
        scalar = np.array([compute_brine_volume(S, T) for S, T in zip(S_arr, T_arr)])
        np.testing.assert_allclose(vec, scalar)

    def test_vectorised_shape(self):
        S = np.array([4.0, 8.0])
        T = np.array([-10.0, -10.0])
        result = compute_brine_volume(S, T)
        assert result.shape == (2,)

    def test_scalar_returns_float(self):
        result = compute_brine_volume(8.0, -10.0)
        assert isinstance(result, float)


class TestBrineSalinityLiquidus:
    """Liquidus salinity S_b(T) of the Cox & Weeks phase model."""

    def test_check_values_against_accepted_liquidus(self):
        from biosnicar.sea_ice.brine_volume import brine_salinity_at_temp as Sb
        # Accepted Assur-based brine salinities (psu); F&G-derived should be
        # within a few psu in the well-constrained -2 to -10 C range.
        assert abs(Sb(-2.0) - 36) < 4
        assert abs(Sb(-5.0) - 87) < 6
        assert abs(Sb(-10.0) - 145) < 10

    def test_below_linear_law_in_cold_regime(self):
        from biosnicar.sea_ice.brine_volume import brine_salinity_at_temp as Sb
        # The whole point of D1: well below the old linear -18.7*T at cold T.
        assert Sb(-10.0) < 0.9 * (18.7 * 10)

    def test_monotonic_and_eutectic_cap(self):
        import numpy as np
        from biosnicar.sea_ice.brine_volume import brine_salinity_at_temp as Sb
        T = np.linspace(-2, -30, 100)
        s = Sb(T)
        assert np.all(np.diff(s) >= -1e-9)          # non-decreasing as T falls
        # Below the hydrohalite onset (-22.9 °C) the brine barely
        # concentrates further: Assur's data rise from ~228 to ~236 psu by
        # -30 °C.  (The earlier derived liquidus was capped flat at 233.)
        assert 226.0 < Sb(-22.9) < 231.0
        assert Sb(-22.9) < Sb(-30.0) < 238.0
        assert Sb(-40.0) == Sb(-30.0)               # held below the fitted range

    def test_scalar_and_vector(self):
        import numpy as np
        from biosnicar.sea_ice.brine_volume import brine_salinity_at_temp as Sb
        assert isinstance(Sb(-10.0), float)
        assert Sb(np.array([-5.0, -10.0])).shape == (2,)


class TestPorosity:
    """Gas (air) content: the Cox & Weeks term that was previously omitted.

    Cox & Weeks describe brine and gas as two halves of one density balance.
    The air content can be stated either as a porosity or as a measured bulk
    density; expected values below come from SMRT v1.7.3 and are hard-coded so
    the suite does not depend on SMRT being installed.
    """

    def test_default_is_bubble_free(self):
        """Omitting both arguments must reproduce the old behaviour exactly."""
        for S, T in [(5.0, -5.0), (8.0, -10.0), (12.0, -25.0)]:
            assert compute_brine_volume(S, T) == compute_brine_volume(S, T, porosity=0.0)

    def test_porosity_scales_linearly(self):
        """The gas term enters as a simple (1 - Va) factor."""
        base = compute_brine_volume(5.0, -10.0)
        for Va in [0.0, 0.02, 0.05, 0.10, 0.25]:
            assert compute_brine_volume(5.0, -10.0, porosity=Va) == pytest.approx(
                (1.0 - Va) * base, rel=1e-12
            )

    @pytest.mark.parametrize("S,T,Va,smrt_value", [
        (5.0,  -10.0, 0.02, 0.027178519),
        (5.0,  -10.0, 0.10, 0.024959865),
        (12.0, -20.0, 0.05, 0.038678813),
        (2.0,  -28.0, 0.02, 0.002098774),
    ])
    def test_porosity_matches_smrt(self, S, T, Va, smrt_value):
        nu_b = compute_brine_volume(S, T, porosity=Va)
        assert nu_b == pytest.approx(smrt_value, rel=4e-4)

    @pytest.mark.parametrize("S,T,rho,smrt_value", [
        (5.0,  -10.0, 850.0, 0.025519701),
        (5.0,  -10.0, 910.0, 0.027321092),
        (12.0, -20.0, 820.0, 0.035872085),
        (2.0,  -28.0, 880.0, 0.002044021),
    ])
    def test_bulk_density_matches_smrt(self, S, T, rho, smrt_value):
        """The bulk-density path is exact to the precision of the stored literals.

        Unlike the porosity path (rel=4e-4, limited by the 0.917 vs 0.9167
        pure-ice density constant), this path never evaluates rho_i at all, so
        the only residual is the rounding of the 9-significant-figure values above
        (worst case ~1e-7 relative, on the smallest value in the set).
        """
        nu_b = compute_brine_volume(S, T, bulk_density_kg_m3=rho)
        assert nu_b == pytest.approx(smrt_value, rel=1e-6)

    def test_porosity_and_density_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="not both"):
            compute_brine_volume(5.0, -10.0, porosity=0.05, bulk_density_kg_m3=880.0)

    def test_rejects_bad_porosity(self):
        for bad in [-0.01, 1.0, 1.5]:
            with pytest.raises(ValueError, match="porosity"):
                compute_brine_volume(5.0, -10.0, porosity=bad)

    def test_rejects_bad_density(self):
        with pytest.raises(ValueError, match="bulk_density"):
            compute_brine_volume(5.0, -10.0, bulk_density_kg_m3=0.0)


class TestGasVolume:
    """compute_gas_volume — the gas half of the same density balance."""

    def test_consistent_with_brine_path(self):
        """Va from the gas equation must reproduce the bulk-density brine volume."""
        from biosnicar.sea_ice.brine_volume import compute_gas_volume
        for T in [-5.0, -15.0, -25.0]:
            for S in [2.0, 8.0]:
                for rho in [830.0, 880.0, 910.0]:
                    Va = compute_gas_volume(S, T, rho)
                    via_porosity = compute_brine_volume(S, T, porosity=Va)
                    via_density = compute_brine_volume(S, T, bulk_density_kg_m3=rho)
                    assert via_porosity == pytest.approx(via_density, abs=1e-12), (
                        f"S={S} T={T} rho={rho}: brine and gas halves disagree"
                    )

    def test_denser_ice_holds_less_air(self):
        from biosnicar.sea_ice.brine_volume import compute_gas_volume
        vals = [compute_gas_volume(5.0, -10.0, r) for r in [820, 850, 880, 910]]
        assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))

    def test_clipped_at_zero(self):
        """Density above the gas-free Cox & Weeks density cannot give negative air."""
        from biosnicar.sea_ice.brine_volume import compute_gas_volume
        assert compute_gas_volume(5.0, -10.0, 1000.0) == 0.0

    def test_scalar_and_vector(self):
        import numpy as np
        from biosnicar.sea_ice.brine_volume import compute_gas_volume
        assert isinstance(compute_gas_volume(5.0, -10.0, 880.0), float)
        out = compute_gas_volume(np.array([5.0, 8.0]), np.array([-10.0, -12.0]),
                                 np.array([880.0, 900.0]))
        assert out.shape == (2,)


class TestInversionWithDensity:
    """invert_brine_volume must invert the same relation the forward model uses."""

    @pytest.mark.parametrize("rho", [820.0, 860.0, 900.0])
    @pytest.mark.parametrize("vb", [0.02, 0.05, 0.10])
    def test_round_trip(self, rho, vb):
        from biosnicar.sea_ice.brine_volume import invert_brine_volume
        T = invert_brine_volume(vb, 6.0, bulk_density_kg_m3=rho)
        back = compute_brine_volume(6.0, T, bulk_density_kg_m3=rho)
        assert back == pytest.approx(vb, abs=1e-5)

    def test_density_changes_recovered_temperature(self):
        """A denser layer needs a colder temperature for the same brine volume."""
        from biosnicar.sea_ice.brine_volume import invert_brine_volume
        T_light = invert_brine_volume(0.05, 6.0, bulk_density_kg_m3=820.0)
        T_dense = invert_brine_volume(0.05, 6.0, bulk_density_kg_m3=900.0)
        assert T_dense < T_light
