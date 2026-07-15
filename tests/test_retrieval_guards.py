"""Regression tests for retrieval input-validation bugs (2026-07 audit).

Covers:
- B1: omitting `direct` must NOT silently fail every trained emulator and
  classify as open_water (the only model without a `direct` param).
- B4: parameter/fixed-param names unknown to the emulator must raise instead
  of being silently dropped (a typo previously produced a flat-cost garbage
  retrieval with no error).
- B3: Hessian-derived 1-sigma must use the chi-squared convention
  (posterior ~ exp(-J/2), covariance = 2 H^-1), matching the MCMC path.
"""

from pathlib import Path

import numpy as np
import pytest

from biosnicar.sea_ice.emulator_configs import (
    SEA_ICE_EMULATOR_CONFIGS,
    trained_emulator_names,
)

_BUILT = all(Path(SEA_ICE_EMULATOR_CONFIGS[n]["emulator_file"]).exists()
             for n in trained_emulator_names())


def test_hessian_sigma_uses_chi2_convention():
    """For J(x) = ((x-mu)/sigma)^2 the posterior 1-sigma is exactly sigma."""
    from biosnicar.inverse.optimize import _hessian_uncertainty
    sigma_true = 0.5
    cost_fn = lambda x: ((x[0] - 2.0) / sigma_true) ** 2  # noqa: E731
    out = _hessian_uncertainty(cost_fn, np.array([2.0]), ["p"], [(0.0, 10.0)])
    # Before the fix this returned sigma/sqrt(2) ~ 0.354.
    assert out["p"] == pytest.approx(sigma_true, rel=1e-3)


@pytest.mark.skipif(not _BUILT, reason="pre-built sea ice emulators not found")
class TestRetrievalEntryGuards:
    def _bare_ice_obs(self):
        from biosnicar.drivers.run_model import run_model
        cfg = SEA_ICE_EMULATOR_CONFIGS["FYI_bare"]
        p = {k: (lo + hi) / 2 for k, (lo, hi) in cfg["params"].items()}
        p["solzen"], p["direct"] = 60, 1
        a = np.asarray(run_model(**cfg["transform_fn"](p)).albedo)
        return np.clip(a + np.random.default_rng(3).normal(0, 0.005, 480), 0, 1)

    def test_missing_direct_fits_all_emulators(self):
        """Omitting `direct` defaults to 1 — every emulator must fit.

        Regression: `direct` used to land in each trained emulator's retrieve
        list, fail the fit via a swallowed ValueError, and leave open_water
        as the silent winner for any spectrum.
        """
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        r = retrieve_sea_ice(observed=self._bare_ice_obs(), solzen=60)
        assert len(r.all_fits) == len(SEA_ICE_EMULATOR_CONFIGS)
        assert r.surface_type != "open_water"

    def test_typoed_fixed_param_raises_fleet(self):
        """A fixed_params key unknown to every emulator raises, not drops."""
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        with pytest.raises(ValueError, match="bogus_param"):
            retrieve_sea_ice(observed=self._bare_ice_obs(), solzen=60,
                             fixed_params={"bogus_param": 1})

    def test_unknown_parameter_raises_in_retrieve(self):
        """optimize.retrieve validates `parameters` against the emulator."""
        from biosnicar.inverse.optimize import retrieve
        from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
        emu = load_sea_ice_emulators(["FYI_pond"])["FYI_pond"]
        with pytest.raises(ValueError, match="not in the emulator's input"):
            retrieve(observed=self._bare_ice_obs(),
                     parameters=["pond_depht"],  # typo
                     emulator=emu,
                     fixed_params={"solzen": 60, "direct": 1})

    def test_unknown_fixed_param_raises_in_retrieve(self):
        """optimize.retrieve validates fixed_params against the emulator."""
        from biosnicar.inverse.optimize import retrieve
        from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
        emu = load_sea_ice_emulators(["FYI_pond"])["FYI_pond"]
        with pytest.raises(ValueError, match="not in the emulator's input"):
            retrieve(observed=self._bare_ice_obs(),
                     parameters=["pond_depth"],
                     emulator=emu,
                     fixed_params={"solzen": 60, "direct": 1, "bogus": 2.0})
