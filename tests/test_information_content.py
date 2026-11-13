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
        # model_error=True: the forward-model error budget.  With instrument
        # noise alone every one of the 480 bands counts as independent evidence,
        # and the small visible signal temperature does have (through the
        # brine volume of the pond floor, amplified since the pond-surface
        # interface fix by light recycling under the water surface) is
        # over-counted into an averaging kernel of ~0.6.  Under the honest
        # covariance it is ~0.2.
        return retrieve_sea_ice(
            observed=self._pond_obs(),
            emulators=load_sea_ice_emulators(), solzen=60, direct=1,
            known_month=7, method="oe", model_error=True)

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
        a higher kernel; under
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
        # (This test used to add that the weak-prior value was also the
        # further from truth.  That is one noise draw's outcome, not a
        # property of the kernel, and it flipped on a retrain.  The claim
        # being pinned is only that a wide prior raises A.)


# ── Degenerate pairs: report the combination, not the components ──────────

class TestConstrainedCombinations:
    @staticmethod
    def _result(cov, names=("a", "b", "c")):
        from biosnicar.inverse.result import RetrievalResult
        n = len(names)
        return RetrievalResult(
            best_fit={p: 1.0 for p in names}, cost=0.0,
            uncertainty={p: float(np.sqrt(cov[i][i])) for i, p in enumerate(names)},
            predicted_albedo=np.zeros(480), observed=np.zeros(480),
            converged=True, method="oe", n_function_evals=0,
            posterior_covariance={(names[i], names[j]): float(cov[i][j])
                                  for i in range(n) for j in range(n)},
        )

    def test_empty_without_a_posterior(self):
        from biosnicar.inverse.result import RetrievalResult
        r = RetrievalResult(best_fit={"a": 1.0}, cost=0.0, uncertainty={},
                            predicted_albedo=np.zeros(480),
                            observed=np.zeros(480), converged=True,
                            method="L-BFGS-B", n_function_evals=0)
        assert r.constrained_combinations() == []
        assert r._degeneracy_lines() == []

    def test_independent_parameters_have_unit_sigma(self):
        r = self._result(np.diag([4.0, 9.0, 0.25]))
        assert [c["sigma"] for c in r.constrained_combinations()] == \
            pytest.approx([1.0, 1.0, 1.0])
        assert r._degeneracy_lines() == []

    def test_a_degenerate_pair_gives_sqrt_one_minus_r(self):
        rho, sa, sb = -0.98, 2.0, 50.0
        cov = np.array([[sa * sa, rho * sa * sb, 0.0],
                        [rho * sa * sb, sb * sb, 0.0],
                        [0.0, 0.0, 1.0]])
        combos = self._result(cov).constrained_combinations()
        best, worst = combos[0], combos[-1]
        assert best["sigma"] == pytest.approx(np.sqrt(1 - abs(rho)))
        assert worst["sigma"] == pytest.approx(np.sqrt(1 + abs(rho)))
        # Negatively correlated pair: the constrained combination is the sum
        # of the normalised parameters, the free direction their difference.
        assert best["weights"]["a"] == pytest.approx(best["weights"]["b"])
        assert worst["weights"]["a"] == pytest.approx(-worst["weights"]["b"])
        assert abs(best["weights"]["c"]) < 1e-9

    def test_variances_sum_to_the_number_of_parameters(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(3, 3))
        combos = self._result(A @ A.T + 0.1 * np.eye(3)).constrained_combinations()
        assert sum(c["variance"] for c in combos) == pytest.approx(3.0)
        assert [c["sigma"] for c in combos] == sorted(c["sigma"] for c in combos)

    def test_summary_names_the_pair(self):
        cov = np.array([[1.0, 0.95, 0.0], [0.95, 1.0, 0.0], [0.0, 0.0, 1.0]])
        text = self._result(cov).summary()
        assert "Degenerate pairs" in text and "a / b" in text and "+0.95" in text


@pytest.mark.skipif(not _BUILT, reason="pre-built sea ice emulators not found")
class TestOEDefaultsToTheForwardModelErrorBudget:
    """model_error="auto": optimal estimation carries the physics budget in
    S_e, in spectral and band mode; other methods are untouched."""

    def test_resolution_rule(self):
        from biosnicar.inverse.model_error import (
            ModelErrorCovariance, PhysicsModelError,
        )
        from biosnicar.sea_ice.retrieve import _resolve_model_error as f
        assert isinstance(f("auto", "oe", None), PhysicsModelError)
        assert isinstance(f("auto", "oe", "sentinel2"), PhysicsModelError)
        assert f("auto", "L-BFGS-B", None) is None
        assert f(None, "oe", None) is None
        assert isinstance(f(True, "oe", None), PhysicsModelError)
        assert isinstance(f("field", "oe", None), ModelErrorCovariance)
        with pytest.raises(ValueError, match="model_error"):
            f("yes", "oe", None)

    def test_default_equals_explicit_true_and_differs_from_none(self):
        from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        fleet = load_sea_ice_emulators(["FYI_pond", "FYI_bare"])
        obs = TestAgainstARealRetrieval()._pond_obs()
        kw = dict(observed=obs, emulators=fleet, solzen=60, direct=1,
                  known_month=7, method="oe")
        default = retrieve_sea_ice(**kw)
        explicit = retrieve_sea_ice(model_error=True, **kw)
        off = retrieve_sea_ice(model_error=None, **kw)
        assert default.parameters == explicit.parameters
        assert default.uncertainty == explicit.uncertainty
        # Honest covariance: fewer independent pieces of evidence.
        assert default.dfs < off.dfs

    def test_band_mode_carries_the_physics_budget_and_refuses_the_field_one(self):
        from biosnicar.bands import to_platform
        from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        fleet = load_sea_ice_emulators(["FYI_pond", "FYI_bare"])
        obs = TestAgainstARealRetrieval()._pond_obs()
        bands = ["B2", "B3", "B4", "B8"]
        br = to_platform(obs, "sentinel2",
                         flx_slr=fleet["FYI_pond"].solar_flux(60, 1))
        y = [getattr(br, b) for b in bands]
        kw = dict(observed=y, emulators=fleet, platform="sentinel2",
                  observed_band_names=bands, solzen=60, direct=1,
                  known_month=7, method="oe")
        default = retrieve_sea_ice(**kw)
        assert default.uncertainty == \
            retrieve_sea_ice(model_error="physics", **kw).uncertainty
        assert default.uncertainty != \
            retrieve_sea_ice(model_error=None, **kw).uncertainty
        # The field-fitted covariance carries field-spectrometer error; it
        # has no business in a satellite band.
        with pytest.raises(ValueError, match="spectral"):
            retrieve_sea_ice(model_error="field", **kw)
