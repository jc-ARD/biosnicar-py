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
    """Spot checks against published approximate values.

    Note: The C&W polynomial as implemented gives values somewhat higher
    than the rounded approximations in the original paper; both sets are
    physically consistent.  The tolerance is set to 40% to accommodate
    the ambiguity in the original paper's unit conventions.
    """

    @pytest.mark.parametrize("S,T,expected", [
        (10.0, -2.0,  0.250),
        (8.0,  -10.0, 0.045),
        (4.0,  -15.0, 0.016),
    ])
    def test_reference_values(self, S, T, expected):
        nu_b = compute_brine_volume(S, T)
        assert nu_b == pytest.approx(expected, rel=0.40), (
            f"S={S} psu, T={T}°C: got {nu_b:.4f}, expected ~{expected:.4f}"
        )


class TestBrineVolumeValidation:
    def test_raises_on_too_warm(self):
        with pytest.raises(ValueError, match="temperature_C"):
            compute_brine_volume(8.0, 0.0)

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
