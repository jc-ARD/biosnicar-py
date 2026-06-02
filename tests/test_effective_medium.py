"""Tests for Maxwell-Garnett effective medium theory."""

import numpy as np
import pytest
from biosnicar.sea_ice.effective_medium import (
    maxwell_garnett, ri_to_eps, eps_to_ri, effective_ri_ice_brine
)


class TestRiEpsConversions:
    def test_ri_to_eps_real(self):
        n = np.array([1.3, 1.31])
        eps = ri_to_eps(n.astype(complex))
        np.testing.assert_allclose(np.real(eps), n**2)
        np.testing.assert_allclose(np.imag(eps), 0.0, atol=1e-12)

    def test_eps_to_ri_roundtrip(self):
        ri_orig = np.array([1.3 + 0.01j, 1.31 + 0.001j, 1.5 + 0.1j])
        eps = ri_to_eps(ri_orig)
        ri_back = eps_to_ri(eps)
        np.testing.assert_allclose(np.real(ri_back), np.real(ri_orig), rtol=1e-10)
        np.testing.assert_allclose(np.imag(ri_back), np.imag(ri_orig), rtol=1e-10)

    def test_eps_to_ri_nonnegative(self):
        eps = np.array([1.7 + 0.01j, 2.0 + 0.1j])
        ri = eps_to_ri(eps)
        assert np.all(np.real(ri) >= 0)
        assert np.all(np.imag(ri) >= 0)


class TestMaxwellGarnett:
    def test_zero_inclusion_equals_host(self):
        eps_h = np.array([1.7 + 0j] * 5)
        eps_i = np.array([2.0 + 0.1j] * 5)
        eps_eff = maxwell_garnett(eps_h, eps_i, 0.0)
        np.testing.assert_allclose(eps_eff, eps_h)

    def test_small_inclusion_closer_to_host(self):
        eps_h = np.array([1.7 + 0j] * 5)
        eps_i = np.array([2.0 + 0.1j] * 5)
        eps_01 = maxwell_garnett(eps_h, eps_i, 0.1)
        eps_03 = maxwell_garnett(eps_h, eps_i, 0.3)
        # More inclusion → further from host
        diff_01 = np.abs(eps_01 - eps_h).mean()
        diff_03 = np.abs(eps_03 - eps_h).mean()
        assert diff_01 < diff_03

    def test_real_input_stays_real(self):
        eps_h = np.array([1.7] * 5)
        eps_i = np.array([1.0] * 5)
        eps_eff = maxwell_garnett(eps_h, eps_i, 0.3)
        np.testing.assert_allclose(np.imag(eps_eff), 0.0, atol=1e-12)

    def test_result_between_host_and_inclusion(self):
        """For real permittivities, result must be between host and inclusion."""
        eps_h = np.array([1.7])
        eps_i = np.array([2.5])
        for f in [0.1, 0.2, 0.3]:
            eps_eff = maxwell_garnett(eps_h, eps_i, f)
            assert float(np.real(eps_eff[0])) >= float(np.real(eps_h[0]))
            assert float(np.real(eps_eff[0])) <= float(np.real(eps_i[0]))

    def test_f05(self):
        """Smoke test at f=0.5 — result is physically plausible."""
        eps_h = np.array([1.7 + 0j] * 10)
        eps_i = np.array([2.0 + 0.05j] * 10)
        eps_eff = maxwell_garnett(eps_h, eps_i, 0.5)
        # Imaginary part should be between 0 and eps_i imaginary
        assert np.all(np.imag(eps_eff) >= 0.0)
        assert np.all(np.imag(eps_eff) <= np.imag(eps_i) * 1.1)


class TestEffectiveRiIceBrine:
    def test_zero_brine_near_pure_ice(self):
        ri_ice = np.array([1.31 + 1e-9j] * 480)
        ri_brine = np.array([1.34 + 0.001j] * 480)
        ri_eff = effective_ri_ice_brine(ri_ice, ri_brine, 0.0)
        np.testing.assert_allclose(np.real(ri_eff), np.real(ri_ice), rtol=1e-6)

    def test_nonnegative_imaginary(self):
        ri_ice = np.array([1.31 + 1e-9j] * 480)
        ri_brine = np.array([1.34 + 0.001j] * 480)
        ri_eff = effective_ri_ice_brine(ri_ice, ri_brine, 0.05)
        assert np.all(np.imag(ri_eff) >= 0.0)

    def test_more_brine_increases_absorption(self):
        ri_ice = np.array([1.31 + 1e-9j] * 480)
        ri_brine = np.array([1.34 + 0.01j] * 480)
        ri_low = effective_ri_ice_brine(ri_ice, ri_brine, 0.01)
        ri_high = effective_ri_ice_brine(ri_ice, ri_brine, 0.10)
        assert np.mean(np.imag(ri_high)) > np.mean(np.imag(ri_low))
