"""Tests for MCMC retrieval through retrieve_sea_ice (E2)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("emcee")

from biosnicar.sea_ice.emulator_configs import SEA_ICE_EMULATOR_CONFIGS
from biosnicar.sea_ice.retrieve import retrieve_sea_ice

POND_FILE = Path(SEA_ICE_EMULATOR_CONFIGS["FYI_pond"]["emulator_file"])


@pytest.mark.skipif(not POND_FILE.exists(), reason="FYI_pond emulator not built")
def test_mcmc_single_type_posterior():
    from biosnicar.emulator import Emulator

    emu = Emulator.load(POND_FILE)
    truth = dict(pond_depth=0.25, sea_ice_temperature=-4.0,
                 black_carbon=100.0, solzen=60, direct=1)
    rng = np.random.default_rng(3)
    obs = np.clip(emu.predict(**truth) + rng.normal(0, 0.005, 480), 0, 1)

    result = retrieve_sea_ice(
        observed=obs,
        emulators={"FYI_pond": emu},
        solzen=60, direct=1,
        fixed_params={"sea_ice_temperature": -4.0, "black_carbon": 100.0},
        method="mcmc",
        mcmc_walkers=16, mcmc_steps=300, mcmc_burn=100,
        obs_uncertainty=np.full(480, 0.005),
    )

    assert result.surface_type == "FYI_pond"
    assert abs(result.parameters["pond_depth"] - 0.25) < 0.08
    # Posterior std (not Hessian) — must be finite and positive
    unc = result.uncertainty["pond_depth"]
    assert np.isfinite(unc) and unc > 0
    fit = result.all_fits["FYI_pond"]
    assert fit.method == "mcmc"
    assert fit.chains is not None
    assert fit.chains.shape[2] == 1  # one free parameter
    assert fit.acceptance_fraction > 0.1
