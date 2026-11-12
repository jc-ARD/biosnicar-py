"""FYI_summer's surface scattering layer is parameterised by optical depth.

``tau_ssl`` = SSL thickness / grain radius.  The layer used to be fixed at
5 cm; these tests pin the reasons the replacement has the form it has.
"""

from pathlib import Path

import numpy as np
import pytest

from biosnicar.drivers.run_model import run_model
from biosnicar.sea_ice.emulator_configs import (
    SEA_ICE_EMULATOR_CONFIGS, SSL_REFERENCE_DENSITY, load_sea_ice_emulators,
    ssl_thickness_m, trained_emulator_names,
)

CFG = SEA_ICE_EMULATOR_CONFIGS["FYI_summer"]
VIS = slice(20, 80)        # 400-1000 nm

_BUILT = all(Path(SEA_ICE_EMULATOR_CONFIGS[n]["emulator_file"]).exists()
             for n in trained_emulator_names())


def _state(**over):
    p = dict(tau_ssl=20.0, ssl_grain_radius=1000.0, sea_ice_temperature=-3.0,
             sea_ice_bubble_radius=300.0, black_carbon=0.0, solzen=60, direct=1)
    p.update(over)
    return p


def _albedo(**over):
    return np.asarray(run_model(**CFG["transform_fn"](_state(**over))).albedo)


class TestTransform:
    def test_tau_ssl_is_a_parameter_and_thickness_is_not(self):
        assert "tau_ssl" in CFG["params"]
        assert "ssl_grain_radius" in CFG["params"]

    def test_thickness_is_tau_times_radius(self):
        kw = CFG["transform_fn"](_state(tau_ssl=20.0, ssl_grain_radius=1000.0))
        assert kw["dz"][0] == pytest.approx(0.020)
        assert kw["rho"][0] == SSL_REFERENCE_DENSITY
        assert ssl_thickness_m(20.0, 1000.0) == pytest.approx(0.020)

    def test_no_clipping_anywhere_in_the_box(self):
        """A clip would make tau_ssl inert over part of its range."""
        (t_lo, t_hi), (r_lo, r_hi) = (CFG["params"]["tau_ssl"],
                                      CFG["params"]["ssl_grain_radius"])
        for tau in (t_lo, t_hi):
            for r in (r_lo, r_hi):
                kw = CFG["transform_fn"](_state(tau_ssl=tau, ssl_grain_radius=r))
                assert kw["dz"][0] == pytest.approx(tau * r * 1e-6)

    def test_old_fixed_layer_is_inside_the_new_box(self):
        """5 cm of 500 um grains — the brightest state the fixed-thickness
        model could reach — is tau_ssl = 100, the upper bound."""
        assert CFG["params"]["tau_ssl"][1] == pytest.approx(0.05 / 500e-6)


class TestPhysics:
    def test_thicker_layer_is_brighter(self):
        a = [_albedo(tau_ssl=t)[VIS].mean() for t in (1, 3, 10, 30, 100)]
        assert all(b > c for b, c in zip(a[1:], a[:-1]))

    def test_response_is_not_saturated_at_the_upper_bound(self):
        hi = CFG["params"]["tau_ssl"][1]
        assert (_albedo(tau_ssl=hi)[VIS].mean()
                - _albedo(tau_ssl=hi / 2)[VIS].mean()) > 0.01

    def test_lower_bound_is_the_bare_ice_limit(self):
        """Below tau_ssl ~ 1 the layer no longer matters."""
        lo = CFG["params"]["tau_ssl"][0]
        assert abs(_albedo(tau_ssl=lo)[VIS].mean()
                   - _albedo(tau_ssl=lo / 2)[VIS].mean()) < 0.01

    def test_density_is_exactly_interchangeable_with_thickness(self):
        """Only rho * dz enters, which is why density is not a parameter."""
        kw = CFG["transform_fn"](_state())
        a = np.asarray(run_model(**kw).albedo)
        kw2 = dict(kw)
        kw2["rho"] = [450.0] + list(kw["rho"][1:])
        kw2["dz"] = [kw["dz"][0] * kw["rho"][0] / 450.0] + list(kw["dz"][1:])
        assert np.allclose(a, np.asarray(run_model(**kw2).albedo), atol=1e-12)

    def test_grain_radius_keeps_an_independent_handle(self):
        """At fixed tau_ssl, grain size still changes the spectrum (through
        absorption per scattering event) — so it stays a parameter."""
        d = _albedo(ssl_grain_radius=1000.0) - _albedo(ssl_grain_radius=2000.0)
        assert np.sqrt(np.mean(d[VIS] ** 2)) > 0.01


@pytest.mark.skipif(not _BUILT, reason="pre-built emulators absent")
class TestRetrieval:
    @pytest.fixture(scope="class")
    def fleet(self):
        return load_sea_ice_emulators()

    def test_emulator_has_the_parameter(self, fleet):
        assert "tau_ssl" in fleet["FYI_summer"].param_names

    def test_thickness_is_reported_as_a_derived_quantity(self, fleet):
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        obs = _albedo(tau_ssl=25.0, ssl_grain_radius=1500.0)
        r = retrieve_sea_ice(observed=obs, emulators=fleet,
                             surface_types=["FYI_summer"], solzen=60, direct=1,
                             known_month=8, method="oe")
        p = r.parameters
        assert p["ssl_thickness"] == pytest.approx(
            p["tau_ssl"] * p["ssl_grain_radius"] * 1e-6)
        # Derived, so it has no averaging kernel of its own.
        assert "ssl_thickness" not in r.averaging_kernel_diag
        assert r.parameter_provenance()["ssl_thickness"] == "unknown"

    def test_batch_engine_reports_it_too(self, fleet):
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch
        obs = np.array([_albedo(tau_ssl=25.0, ssl_grain_radius=1500.0)])
        ds = retrieve_sea_ice_batch(
            obs, emulators={"FYI_summer": fleet["FYI_summer"]},
            engine="vectorized", solzen=60, direct=1, known_month=8,
            method="oe").to_xarray()
        assert float(ds["param_ssl_thickness"].values[0]) == pytest.approx(
            float(ds["param_tau_ssl"].values[0])
            * float(ds["param_ssl_grain_radius"].values[0]) * 1e-6, rel=1e-5)
