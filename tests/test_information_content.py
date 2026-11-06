"""Per-parameter information content: measurement or prior?

Optimal estimation reports a value and a 1-sigma for every free parameter,
and those two numbers alone cannot distinguish "the spectrum pinned this"
from "the prior was narrow and the spectrum said nothing".  The averaging
kernel does.  These tests pin the convention that reads it, and the places
it has to reach: the two result classes, the per-pixel record, the scene
rasters and the summaries.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from biosnicar.inverse import information as info
from biosnicar.inverse.result import RetrievalResult
from biosnicar.sea_ice.retrieve import SeaIceRetrievalResult


def _hs_result(akd=None, best_fit=None, method="oe"):
    bf = best_fit or {"pond_depth": 0.25, "sea_ice_temperature": -4.0}
    return RetrievalResult(
        best_fit=bf, cost=0.01,
        uncertainty={k: 0.1 for k in bf},
        predicted_albedo=np.zeros(480), observed=np.zeros(480),
        converged=True, method=method, n_function_evals=10,
        averaging_kernel_diag=akd,
    )


def _si_result(akd=None, params=None):
    p = params or {"pond_depth": 0.25, "sea_ice_temperature": -4.0}
    return SeaIceRetrievalResult(
        surface_type="FYI_pond", surface_description="", confidence=1.0,
        parameters=p, uncertainty={k: 0.1 for k in p},
        predicted_albedo=np.zeros(480), observed=np.zeros(480),
        cost=0.0, converged=True, flx_slr=None,
        averaging_kernel_diag=akd or {},
    )


# ── The convention itself ────────────────────────────────────────────────

class TestClassify:
    def test_three_bands(self):
        assert info.classify(0.95) == info.MEASURED
        assert info.classify(0.30) == info.PRIOR_INFLUENCED
        assert info.classify(0.02) == info.NOT_RETRIEVED

    def test_boundaries_are_inclusive_upward(self):
        """A parameter exactly at a threshold takes the better label, so the
        bands partition [0, 1] with no value falling between them."""
        assert info.classify(info.NOT_RETRIEVED_THRESHOLD) == info.PRIOR_INFLUENCED
        assert info.classify(info.MEASURED_THRESHOLD) == info.MEASURED
        assert info.classify(info.MEASURED_THRESHOLD - 1e-12) == info.PRIOR_INFLUENCED

    def test_missing_kernel_is_unknown_not_measured(self):
        """No averaging kernel must never read as an endorsement."""
        assert info.classify(None) == info.UNKNOWN
        assert info.classify(float("nan")) == info.UNKNOWN

    def test_sheba_temperature_is_not_retrieved(self):
        """The measured case the 0.1 threshold is set from: sea_ice_temperature
        scores A = 0.048 across 22 of 23 SHEBA spectra."""
        assert info.classify(0.048) == info.NOT_RETRIEVED

    def test_thresholds_are_ordered(self):
        assert 0.0 < info.NOT_RETRIEVED_THRESHOLD < info.MEASURED_THRESHOLD < 1.0


class TestProvenanceMapping:
    def test_derived_parameters_are_labelled_unknown_not_dropped(self):
        """snow_depth is derived from tau_snow and has no kernel of its own.
        Omitting it from the provenance report would read as approval."""
        prov = info.provenance({"tau_snow": 0.9}, {"tau_snow": 12.0,
                                                   "snow_depth": 0.1})
        assert prov == {"tau_snow": info.MEASURED, "snow_depth": info.UNKNOWN}

    def test_kernel_entries_absent_from_parameters_still_reported(self):
        prov = info.provenance({"hidden": 0.9}, {"shown": 1.0})
        assert set(prov) == {"shown", "hidden"}


class TestFormatting:
    def test_falls_back_to_plain_form_without_a_kernel(self):
        """Non-OE retrievals must read exactly as before — no column of
        'unknown' where there was nothing to say."""
        lines = info.format_parameter_lines({"a": 1.0}, {"a": 0.1}, None)
        assert lines == ["    a                         =       1.0000  ±  0.1000"]

    def test_info_column_aligns_with_its_header(self):
        lines = info.format_parameter_lines({"a": 1.0}, {"a": 0.1}, {"a": 0.93})
        assert lines[0].index("info") == lines[1].index("0.93")

    def test_not_retrieved_parameters_get_an_explicit_warning(self):
        text = "\n".join(info.format_parameter_lines(
            {"good": 1.0, "bad": 2.0}, {"good": 0.1, "bad": 0.2},
            {"good": 0.95, "bad": 0.03}))
        assert "not retrieved" in text and "bad" in text
        assert "do not quote as a measurement" in text

    def test_no_warning_when_everything_is_measured(self):
        text = "\n".join(info.format_parameter_lines(
            {"good": 1.0}, {"good": 0.1}, {"good": 0.95}))
        assert "do not quote" not in text


# ── Both result classes expose the same thing the same way ───────────────

@pytest.mark.parametrize("make", [_hs_result, _si_result])
class TestResultAccessors:
    _AKD = {"pond_depth": 0.99, "sea_ice_temperature": 0.048}

    def test_information_content_round_trips(self, make):
        assert make(self._AKD).information_content() == self._AKD

    def test_information_content_empty_without_oe(self, make):
        assert make(None).information_content() == {}

    def test_parameter_provenance_labels_each_parameter(self, make):
        prov = make(self._AKD).parameter_provenance()
        assert prov["pond_depth"] == info.MEASURED
        assert prov["sea_ice_temperature"] == info.NOT_RETRIEVED

    def test_prior_dominated_uses_the_shared_threshold(self, make):
        """Both classes must draw the line in the same place; they used to be
        free to drift because each hard-coded its own 0.5."""
        flags = make(self._AKD).prior_dominated_parameters()
        assert flags == {"pond_depth": False, "sea_ice_temperature": True}

    def test_prior_dominated_threshold_is_overridable(self, make):
        assert make(self._AKD).prior_dominated_parameters(threshold=0.999)[
            "pond_depth"] is True

    def test_measured_parameters_drops_the_prior_dominated_ones(self, make):
        m = make(self._AKD).measured_parameters()
        assert set(m) == {"pond_depth"}

    def test_measured_parameters_returns_all_when_nothing_was_shown(self, make):
        """Without a kernel nothing has been demonstrated either way, so
        silently returning an empty dict would be a false negative."""
        r = make(None)
        reported = getattr(r, "best_fit", None) or r.parameters
        assert r.measured_parameters() == reported

    def test_summary_flags_the_prior_dominated_parameter(self, make):
        text = make(self._AKD).summary()
        assert "not retrieved" in text
        assert "0.05" in text          # the averaging kernel is shown

    def test_summary_is_unchanged_without_a_kernel(self, make):
        text = make(None).summary()
        assert "provenance" not in text
        assert "not retrieved" not in text


# ── The information has to survive the trip to batch and scene output ────

def test_record_carries_the_averaging_kernel():
    """The vectorised batch path always carried this; the per-pixel loop
    path dropped it, so scene results silently lost provenance depending on
    which engine produced them."""
    from biosnicar.sea_ice.retrieve import _result_to_record
    rec = _result_to_record(_si_result({"pond_depth": 0.99,
                                        "sea_ice_temperature": 0.048}))
    assert rec["averaging_kernel_diag"]["sea_ice_temperature"] == pytest.approx(0.048)


def _record(akd):
    return {
        "surface_type": "FYI_pond", "confidence": 0.9, "cost": 0.01,
        "quality_flags": 0,
        "parameters": {"pond_depth": 0.25, "sea_ice_temperature": -4.0},
        "uncertainty": {"pond_depth": 0.01, "sea_ice_temperature": 1.2},
        "dfs": 2.0, "class_probabilities": {"FYI_pond": 0.9},
        "prior_resolved": None, "spectrum_only_surface_type": None,
        "averaging_kernel_diag": akd,
    }


class TestSceneRasters:
    def test_akd_rasters_written_when_kernels_exist(self):
        pytest.importorskip("xarray")
        from biosnicar.sea_ice.scene_result import SeaIceSceneResult
        akd = {"pond_depth": 0.99, "sea_ice_temperature": 0.048}
        ds = SeaIceSceneResult.from_records([_record(akd), _record(akd)]).to_xarray()
        assert float(ds["akd_sea_ice_temperature"].values[0]) == pytest.approx(0.048)
        assert float(ds["akd_pond_depth"].values[0]) == pytest.approx(0.99)

    def test_no_akd_rasters_when_no_kernels(self):
        """An all-NaN akd_ raster is indistinguishable from 'measured
        nothing', so the variable must be absent rather than empty."""
        pytest.importorskip("xarray")
        from biosnicar.sea_ice.scene_result import SeaIceSceneResult
        ds = SeaIceSceneResult.from_records([_record({}), _record({})]).to_xarray()
        assert not [v for v in ds.data_vars if str(v).startswith("akd_")]

    def test_scene_summary_names_the_unretrieved_parameter(self):
        pytest.importorskip("xarray")
        from biosnicar.sea_ice.scene_result import SeaIceSceneResult
        akd = {"pond_depth": 0.99, "sea_ice_temperature": 0.048}
        text = SeaIceSceneResult.from_records([_record(akd)]).summary()
        assert "information content" in text
        assert "sea_ice_temperature" in text and "not retrieved" in text


# ── Against a real retrieval ─────────────────────────────────────────────

from biosnicar.sea_ice.emulator_configs import (  # noqa: E402
    SEA_ICE_EMULATOR_CONFIGS, trained_emulator_names,
)

_BUILT = all(Path(SEA_ICE_EMULATOR_CONFIGS[n]["emulator_file"]).exists()
             for n in trained_emulator_names())


@pytest.mark.skipif(not _BUILT, reason="pre-built sea ice emulators not found")
class TestAgainstARealRetrieval:
    """``retrieve_sea_ice`` is the path that applies the real seasonal priors,
    so it is the one whose averaging kernels mean anything."""

    def _pond_obs(self):
        from biosnicar.drivers.run_model import run_model
        truth = dict(pond_depth=0.25, sea_ice_temperature=-4.0,
                     sea_ice_bubble_radius=200.0, black_carbon=100.0,
                     solzen=60, direct=1)
        albedo = np.asarray(run_model(
            **SEA_ICE_EMULATOR_CONFIGS["FYI_pond"]["transform_fn"](truth)).albedo)
        return np.clip(albedo + np.random.default_rng(1).normal(0, 0.005, 480), 0, 1)

    def _retrieve(self):
        from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        return retrieve_sea_ice(
            observed=self._pond_obs(),
            emulators=load_sea_ice_emulators(), solzen=60, direct=1,
            known_month=7, method="oe", obs_uncertainty=np.full(480, 0.005))

    def test_separates_measured_from_prior_dominated(self):
        """Pond depth is measured — water absorption is unmistakable — while
        temperature is not: it acts only through brine volume and salinity,
        which the VIS barely sees.

        Asserted as an ordering, not against a fixed kernel value, so a
        retrain that shifts the absolute numbers is not a false alarm.
        """
        r = self._retrieve()
        prov = r.parameter_provenance()
        assert prov["pond_depth"] == info.MEASURED
        assert prov["sea_ice_temperature"] != info.MEASURED
        akd = r.information_content()
        assert akd["sea_ice_temperature"] < akd["pond_depth"]
        assert "pond_depth" in r.measured_parameters()
        assert "sea_ice_temperature" not in r.measured_parameters()

    def test_visible_without_the_caller_going_looking(self):
        """The whole point: the default human-readable output has to say so."""
        assert "provenance" in self._retrieve().summary()

    def test_a_weak_prior_inflates_the_averaging_kernel(self):
        """A is relative to the prior, not to truth.

        Under ``retrieve()``'s wide default prior the same temperature scores
        a high kernel while running to the edge of its training range; under
        the seasonal prior ``retrieve_sea_ice`` applies it is correctly
        marked prior-influenced.  This pins the caveat so it cannot be
        quietly lost — a high A is not on its own a claim of accuracy.
        """
        from biosnicar.inverse.optimize import retrieve
        from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # runs outside training bounds
            weak = retrieve(
                observed=self._pond_obs(),
                parameters=["pond_depth", "sea_ice_temperature", "black_carbon"],
                emulator=load_sea_ice_emulators(["FYI_pond"])["FYI_pond"],
                fixed_params={"solzen": 60, "direct": 1,
                              "sea_ice_bubble_radius": 200.0},
                obs_uncertainty=np.full(480, 0.005), method="oe")
        strong = self._retrieve()
        assert weak.information_content()["sea_ice_temperature"] > \
            strong.information_content()["sea_ice_temperature"]
        # and the weak-prior value is the less trustworthy of the two
        assert abs(weak.best_fit["sea_ice_temperature"] - (-4.0)) > \
            abs(strong.parameters["sea_ice_temperature"] - (-4.0))
