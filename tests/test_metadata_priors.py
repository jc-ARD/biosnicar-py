"""Tests for the C1 metadata->prior adapter (biosnicar.sea_ice.metadata_priors)
and its integration into retrieve_sea_ice."""

from pathlib import Path

import numpy as np
import pytest

from biosnicar.sea_ice.metadata_priors import (
    SOFT_NAT, build_prior_set, season_provider,
)
from biosnicar.sea_ice.emulator_configs import (
    SEA_ICE_EMULATOR_CONFIGS, trained_emulator_names,
)

_BUILT = all(Path(SEA_ICE_EMULATOR_CONFIGS[n]["emulator_file"]).exists()
             for n in trained_emulator_names())


class TestFramework:
    def test_season_param_priors_preserved(self):
        """The season provider reproduces the original known_month priors."""
        assert season_provider({"month": 7}).param_prior["sea_ice_temperature"] == (-4.0, 3.0)
        assert season_provider({"month": 1}).param_prior["sea_ice_temperature"] == (-15.0, 8.0)
        oct_ = season_provider({"month": 10})
        assert oct_.param_prior["ice_thickness_cm"] == (5.0, 8.0)
        assert season_provider({"month": 4}) is None or \
            season_provider({"month": 4}).param_prior == {}

    def test_hard_exclusions_are_physical(self):
        assert "young_ice" in build_prior_set({"month": 7}).excluded
        assert build_prior_set({"month": 1}).excluded >= {"FYI_pond", "FYI_summer"}
        assert build_prior_set({"month": 4}).excluded == frozenset()

    def test_soft_snow_prior_only_in_peak_melt(self):
        # peak/late melt disfavours fresh snow; spring (May) does not
        assert build_prior_set({"month": 8}).class_log_prior["FYI_snow"] == -SOFT_NAT
        assert "FYI_snow" not in build_prior_set({"month": 5}).class_log_prior
        assert "FYI_snow" not in build_prior_set({"month": 4}).class_log_prior

    def test_caller_channel_composes_and_is_attributed(self):
        ps = build_prior_set({"month": 8},
                             extra_class_priors={"MYI_bare": 1.5, "FYI_bare": -1.5})
        assert ps.class_log_prior["MYI_bare"] == 1.5
        assert ps.class_log_prior["FYI_bare"] == -1.5
        assert set(ps.sources) == {"season", "caller"}

    def test_source_toggle_withholds_providers(self):
        assert build_prior_set({"month": 7}, sources=()).is_empty()
        # caller priors still apply even with built-ins withheld
        ps = build_prior_set({"month": 7}, sources=(),
                             extra_class_priors={"open_water": 2.0})
        assert ps.class_log_prior == {"open_water": 2.0}
        assert ps.excluded == frozenset()

    def test_unknown_surface_type_rejected(self):
        with pytest.raises(ValueError, match="unknown surface type"):
            build_prior_set({"month": 7}, extra_class_priors={"glacier": 1.0})

    def test_no_context_is_empty(self):
        assert build_prior_set().is_empty()


@pytest.mark.skipif(not _BUILT, reason="pre-built sea ice emulators not found")
class TestIntegration:
    def _obs(self, stype="FYI_bare", seed=3):
        from biosnicar.drivers.run_model import run_model
        cfg = SEA_ICE_EMULATOR_CONFIGS[stype]
        p = {k: (lo + hi) / 2 for k, (lo, hi) in cfg["params"].items()}
        p["solzen"], p["direct"] = 60, 1
        a = np.asarray(run_model(**cfg["transform_fn"](p)).albedo)
        return np.clip(a + np.random.default_rng(seed).normal(0, 0.006, 480), 0, 1)

    def test_class_prior_shifts_probability(self):
        """A caller class prior moves the class probability in band mode."""
        from biosnicar.bands import to_platform
        from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        fleet = load_sea_ice_emulators()
        bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A"]
        br = to_platform(self._obs(), "sentinel2", flx_slr=fleet["FYI_pond"].flx_slr)
        y = np.array([getattr(br, b) for b in bands])
        common = dict(observed=y, platform="sentinel2", observed_band_names=bands,
                      emulators=fleet, solzen=60, known_month=6, method="oe")
        base = retrieve_sea_ice(**common)
        pushed = retrieve_sea_ice(class_priors={"MYI_bare": 4.0}, **common)
        assert (pushed.class_probabilities.get("MYI_bare", 0)
                > base.class_probabilities.get("MYI_bare", 0))

    def test_use_priors_false_ignores_class_priors(self):
        """Spectrum-only drops the caller class priors too (A3 semantics)."""
        from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        fleet = load_sea_ice_emulators()
        obs = self._obs()
        a = retrieve_sea_ice(observed=obs, emulators=fleet, solzen=60,
                             known_month=7, method="oe", use_priors=False,
                             class_priors={"open_water": 8.0})
        b = retrieve_sea_ice(observed=obs, emulators=fleet, solzen=60,
                             known_month=7, method="oe", use_priors=False)
        assert a.surface_type == b.surface_type   # class prior had no effect

    def test_winter_excludes_pond_from_fleet(self):
        """A January retrieval must not consider FYI_pond / FYI_summer."""
        from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        fleet = load_sea_ice_emulators()
        r = retrieve_sea_ice(observed=self._obs(), emulators=fleet, solzen=60,
                             known_month=1)
        assert "FYI_pond" not in r.all_fits
        assert "FYI_summer" not in r.all_fits
