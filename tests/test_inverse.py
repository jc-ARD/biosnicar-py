"""Tests for the biosnicar.inverse module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from biosnicar.inverse.result import RetrievalResult
from biosnicar.emulator import Emulator, _latin_hypercube
from biosnicar.inverse.cost import spectral_cost, band_cost
from biosnicar.inverse.optimize import retrieve, DEFAULT_BOUNDS
from biosnicar.drivers.run_emulator import run_emulator


# ── RetrievalResult ──────────────────────────────────────────────────


class TestRetrievalResult:
    """Unit tests for the RetrievalResult dataclass."""

    @pytest.fixture
    def basic_result(self):
        return RetrievalResult(
            best_fit={"rds": 800.0, "black_carbon": 5000.0},
            cost=0.0012,
            uncertainty={"rds": 25.0, "black_carbon": 120.0},
            predicted_albedo=np.full(480, 0.5),
            observed=np.full(480, 0.5),
            converged=True,
            method="L-BFGS-B",
            n_function_evals=42,
        )

    def test_fields(self, basic_result):
        assert basic_result.best_fit["rds"] == 800.0
        assert basic_result.cost == pytest.approx(0.0012)
        assert basic_result.converged is True
        assert basic_result.method == "L-BFGS-B"
        assert basic_result.n_function_evals == 42

    def test_mcmc_fields_default_none(self, basic_result):
        assert basic_result.chains is None
        assert basic_result.acceptance_fraction is None
        assert basic_result.autocorr_time is None

    def test_mcmc_fields_populated(self):
        chains = np.random.randn(500, 16, 2)
        result = RetrievalResult(
            best_fit={"rds": 800.0},
            cost=0.01,
            uncertainty={"rds": 25.0},
            predicted_albedo=np.full(480, 0.5),
            observed=np.full(480, 0.5),
            converged=True,
            method="mcmc",
            n_function_evals=16000,
            chains=chains,
            acceptance_fraction=0.35,
            autocorr_time=np.array([20.0, 15.0]),
        )
        assert result.chains.shape == (500, 16, 2)
        assert result.acceptance_fraction == pytest.approx(0.35)

    def test_summary_contains_key_info(self, basic_result):
        s = basic_result.summary()
        assert "L-BFGS-B" in s
        assert "converged=True" in s
        assert "rds" in s
        assert "800.0" in s
        assert "25.0" in s

    def test_summary_mcmc_section(self):
        result = RetrievalResult(
            best_fit={"rds": 800.0},
            cost=0.01,
            uncertainty={"rds": 25.0},
            predicted_albedo=np.full(480, 0.5),
            observed=np.full(480, 0.5),
            converged=True,
            method="mcmc",
            n_function_evals=16000,
            chains=np.random.randn(500, 16, 1),
            acceptance_fraction=0.35,
            autocorr_time=np.array([20.0]),
        )
        s = result.summary()
        assert "MCMC" in s
        assert "500 steps" in s
        assert "16 walkers" in s

    def test_predicted_albedo_shape(self, basic_result):
        assert basic_result.predicted_albedo.shape == (480,)

    def test_observed_shape(self, basic_result):
        assert basic_result.observed.shape == (480,)

    def test_flx_slr_defaults_none(self, basic_result):
        assert basic_result.flx_slr is None

    # ── to_outputs() ──────────────────────────────────────────────────

    def test_to_outputs_returns_outputs(self, basic_result):
        from biosnicar.classes.outputs import Outputs
        out = basic_result.to_outputs()
        assert isinstance(out, Outputs)

    def test_to_outputs_albedo_matches(self, basic_result):
        out = basic_result.to_outputs()
        np.testing.assert_array_equal(out.albedo, basic_result.predicted_albedo)

    def test_to_outputs_bba_physical(self, basic_result):
        out = basic_result.to_outputs()
        assert 0.0 <= out.BBA <= 1.0
        assert 0.0 <= out.BBAVIS <= 1.0
        assert 0.0 <= out.BBANIR <= 1.0

    def test_to_outputs_aliases(self, basic_result):
        out = basic_result.to_outputs()
        assert out.broadband == out.BBA
        assert out.visible == out.BBAVIS
        assert out.nir == out.BBANIR

    def test_to_outputs_with_flx_slr_uses_flux_weighting(self):
        flx = np.ones(480)
        flx[:50] = 10.0   # weight VIS heavily
        result = RetrievalResult(
            best_fit={}, cost=0.0, uncertainty={},
            predicted_albedo=np.linspace(0.9, 0.1, 480),
            observed=np.zeros(480), converged=True,
            method="L-BFGS-B", n_function_evals=1,
            flx_slr=flx,
        )
        out = result.to_outputs()
        assert out.flx_slr is flx
        # BBAVIS should be weighted toward high-albedo end
        assert out.BBAVIS > out.BBANIR

    def test_to_outputs_no_platform_without_flx_slr(self, basic_result):
        # flx_slr is None → to_platform() should raise
        out = basic_result.to_outputs()
        assert out.flx_slr is None
        with pytest.raises(Exception):
            out.to_platform("sentinel2")

    def test_to_outputs_to_platform_with_flx_slr(self):
        flx = np.ones(480)
        result = RetrievalResult(
            best_fit={}, cost=0.0, uncertainty={},
            predicted_albedo=np.full(480, 0.6),
            observed=np.zeros(480), converged=True,
            method="L-BFGS-B", n_function_evals=1,
            flx_slr=flx,
        )
        out = result.to_outputs()
        bands = out.to_platform("sentinel2")
        assert hasattr(bands, "B3")


# ── Latin Hypercube Sampling ─────────────────────────────────────────


class TestLatinHypercube:
    """Unit tests for the LHS implementation."""

    def test_shape(self):
        samples = _latin_hypercube(100, 3, seed=0)
        assert samples.shape == (100, 3)

    def test_range(self):
        samples = _latin_hypercube(1000, 5, seed=0)
        assert np.all(samples >= 0.0)
        assert np.all(samples < 1.0)

    def test_stratification(self):
        """Each column should have exactly one sample per stratum."""
        n = 50
        samples = _latin_hypercube(n, 3, seed=0)
        for j in range(3):
            strata = np.floor(samples[:, j] * n).astype(int)
            assert len(np.unique(strata)) == n

    def test_reproducibility(self):
        a = _latin_hypercube(50, 2, seed=42)
        b = _latin_hypercube(50, 2, seed=42)
        np.testing.assert_array_equal(a, b)


# ── Emulator ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_emulator():
    """Build a small 2-parameter emulator for fast testing.

    Uses 500 samples to reliably achieve >0.8 R² across different
    scikit-learn versions and platforms.  Module-scoped so it
    is built once and shared across all tests in this class.
    """
    emu = Emulator.build(
        params={"rds": (500, 3000), "black_carbon": (0, 50000)},
        n_samples=500,
        progress=False,
        seed=42,
        layer_type=1,
        solzen=50,
    )
    return emu


class TestEmulatorBuild:
    """Tests for Emulator.build() and basic properties."""

    def test_param_names(self, tiny_emulator):
        assert tiny_emulator.param_names == ["rds", "black_carbon"]

    def test_bounds(self, tiny_emulator):
        b = tiny_emulator.bounds
        assert b["rds"] == (500.0, 3000.0)
        assert b["black_carbon"] == (0.0, 50000.0)

    def test_n_pca_components(self, tiny_emulator):
        assert tiny_emulator.n_pca_components > 0
        assert tiny_emulator.n_pca_components < 480

    def test_training_score(self, tiny_emulator):
        # R² should be reasonable even with a small training set.
        # Threshold is 0.8 (not 0.9) for robustness across scikit-learn
        # versions and platforms where early stopping can halt sooner.
        assert tiny_emulator.training_score > 0.8

    def test_flx_slr(self, tiny_emulator):
        assert tiny_emulator.flx_slr is not None
        assert len(tiny_emulator.flx_slr) == 480

    def test_repr(self, tiny_emulator):
        r = repr(tiny_emulator)
        assert "rds" in r
        assert "black_carbon" in r


class TestEmulatorPredict:
    """Tests for Emulator.predict() and predict_batch()."""

    def test_predict_shape(self, tiny_emulator):
        alb = tiny_emulator.predict(rds=1000, black_carbon=5000)
        assert alb.shape == (480,)

    def test_predict_physical_range(self, tiny_emulator):
        alb = tiny_emulator.predict(rds=1000, black_carbon=0)
        assert np.all(alb >= 0.0)
        assert np.all(alb <= 1.0)

    def test_predict_batch_shape(self, tiny_emulator):
        points = np.array([[1000, 5000], [2000, 10000]])
        result = tiny_emulator.predict_batch(points)
        assert result.shape == (2, 480)

    def test_predict_batch_matches_single(self, tiny_emulator):
        single = tiny_emulator.predict(rds=1000, black_carbon=5000)
        batch = tiny_emulator.predict_batch(np.array([[1000, 5000]]))
        np.testing.assert_allclose(single, batch[0], atol=1e-10)

    def test_missing_param_raises(self, tiny_emulator):
        with pytest.raises(ValueError, match="Missing parameters"):
            tiny_emulator.predict(rds=1000)

    def test_out_of_bounds_warns(self, tiny_emulator):
        with pytest.warns(UserWarning, match="outside training bounds"):
            tiny_emulator.predict(rds=10000, black_carbon=5000)

    def test_impurity_increases_darken_spectrum(self, tiny_emulator):
        """More impurities should produce lower albedo (at least broadband)."""
        clean = tiny_emulator.predict(rds=1000, black_carbon=0)
        dirty = tiny_emulator.predict(rds=1000, black_carbon=40000)
        assert np.mean(clean) > np.mean(dirty)


class TestEmulatorSaveLoad:
    """Tests for save/load round-trip."""

    def test_save_load_roundtrip(self, tiny_emulator):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_emu.npz"
            tiny_emulator.save(path)

            loaded = Emulator.load(path)
            assert loaded.param_names == tiny_emulator.param_names
            assert loaded.bounds == tiny_emulator.bounds
            assert loaded.n_pca_components == tiny_emulator.n_pca_components
            assert loaded.training_score == pytest.approx(
                tiny_emulator.training_score
            )

    def test_predictions_identical_after_load(self, tiny_emulator):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_emu.npz"
            tiny_emulator.save(path)
            loaded = Emulator.load(path)

            original = tiny_emulator.predict(rds=1500, black_carbon=10000)
            reloaded = loaded.predict(rds=1500, black_carbon=10000)
            np.testing.assert_allclose(original, reloaded, atol=1e-10)

    def test_flx_slr_preserved(self, tiny_emulator):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_emu.npz"
            tiny_emulator.save(path)
            loaded = Emulator.load(path)
            np.testing.assert_allclose(
                loaded.flx_slr, tiny_emulator.flx_slr, atol=1e-10
            )


# ── Cost Functions ───────────────────────────────────────────────────


class TestSpectralCost:
    """Unit tests for spectral_cost()."""

    def _make_forward_fn(self, offset=0.0):
        """Return a trivial forward function for testing."""
        def fn(**params):
            return np.full(480, 0.5 + offset)
        return fn

    def test_zero_cost_perfect_match(self):
        observed = np.full(480, 0.5)
        cost = spectral_cost(
            params=[1000, 5000],
            param_names=["rds", "black_carbon"],
            observed=observed,
            forward_fn=self._make_forward_fn(0.0),
        )
        assert cost == pytest.approx(0.0)

    def test_positive_cost_mismatch(self):
        observed = np.full(480, 0.5)
        cost = spectral_cost(
            params=[1000, 5000],
            param_names=["rds", "black_carbon"],
            observed=observed,
            forward_fn=self._make_forward_fn(0.1),
        )
        # 480 * (0.1)^2 = 4.8
        assert cost == pytest.approx(4.8)

    def test_uncertainty_weighting(self):
        observed = np.full(480, 0.5)
        sigma = np.full(480, 0.1)
        cost = spectral_cost(
            params=[1000],
            param_names=["rds"],
            observed=observed,
            forward_fn=self._make_forward_fn(0.1),
            obs_uncertainty=sigma,
        )
        # 480 * (0.1 / 0.1)^2 = 480
        assert cost == pytest.approx(480.0)

    def test_wavelength_mask_excludes_bins(self):
        observed = np.full(480, 0.5)
        mask = np.zeros(480, dtype=bool)
        mask[:10] = True  # only use first 10 bins
        cost = spectral_cost(
            params=[1000],
            param_names=["rds"],
            observed=observed,
            forward_fn=self._make_forward_fn(0.1),
            wavelength_mask=mask,
        )
        # 10 * (0.1)^2 = 0.1
        assert cost == pytest.approx(0.1)

    def test_regularization_adds_penalty(self):
        observed = np.full(480, 0.5)
        cost_no_reg = spectral_cost(
            params=[1000],
            param_names=["rds"],
            observed=observed,
            forward_fn=self._make_forward_fn(0.0),
        )
        cost_with_reg = spectral_cost(
            params=[1000],
            param_names=["rds"],
            observed=observed,
            forward_fn=self._make_forward_fn(0.0),
            regularization={"rds": (500, 100)},  # prior=500, sigma=100
        )
        # Regularization adds ((1000 - 500) / 100)^2 = 25
        assert cost_no_reg == pytest.approx(0.0)
        assert cost_with_reg == pytest.approx(25.0)


class TestBandCost:
    """Unit tests for band_cost()."""

    def test_zero_cost_perfect_match(self, tiny_emulator):
        """Band cost should be zero when prediction matches observation."""
        # Generate a prediction, then use it as the observation
        predicted = tiny_emulator.predict(rds=1000, black_carbon=5000)
        flx_slr = tiny_emulator.flx_slr

        from biosnicar.bands import to_platform
        s2 = to_platform(predicted, "sentinel2", flx_slr=flx_slr)
        observed_bands = np.array([s2.B3, s2.B4, s2.B11])

        cost = band_cost(
            params=[1000, 5000],
            param_names=["rds", "black_carbon"],
            observed=observed_bands,
            observed_band_names=["B3", "B4", "B11"],
            forward_fn=tiny_emulator.predict,
            flx_slr=flx_slr,
            platform="sentinel2",
        )
        assert cost == pytest.approx(0.0, abs=1e-10)

    def test_positive_cost_mismatch(self, tiny_emulator):
        """Band cost should be positive when parameters differ."""
        predicted = tiny_emulator.predict(rds=1000, black_carbon=5000)
        flx_slr = tiny_emulator.flx_slr

        from biosnicar.bands import to_platform
        s2 = to_platform(predicted, "sentinel2", flx_slr=flx_slr)
        observed_bands = np.array([s2.B3, s2.B4, s2.B11])

        # Evaluate cost at a different point
        cost = band_cost(
            params=[2000, 20000],
            param_names=["rds", "black_carbon"],
            observed=observed_bands,
            observed_band_names=["B3", "B4", "B11"],
            forward_fn=tiny_emulator.predict,
            flx_slr=flx_slr,
            platform="sentinel2",
        )
        assert cost > 0.0

    def test_band_cost_with_uncertainty(self, tiny_emulator):
        """Uncertainty weighting should change the cost value."""
        predicted = tiny_emulator.predict(rds=1000, black_carbon=5000)
        flx_slr = tiny_emulator.flx_slr

        from biosnicar.bands import to_platform
        s2 = to_platform(predicted, "sentinel2", flx_slr=flx_slr)
        observed_bands = np.array([s2.B3, s2.B4, s2.B11])

        # Cost with tight uncertainty should differ from unweighted
        cost_unweighted = band_cost(
            params=[2000, 20000],
            param_names=["rds", "black_carbon"],
            observed=observed_bands,
            observed_band_names=["B3", "B4", "B11"],
            forward_fn=tiny_emulator.predict,
            flx_slr=flx_slr,
            platform="sentinel2",
        )
        cost_weighted = band_cost(
            params=[2000, 20000],
            param_names=["rds", "black_carbon"],
            observed=observed_bands,
            observed_band_names=["B3", "B4", "B11"],
            forward_fn=tiny_emulator.predict,
            flx_slr=flx_slr,
            platform="sentinel2",
            obs_uncertainty=np.array([0.01, 0.01, 0.01]),
        )
        assert cost_weighted != pytest.approx(cost_unweighted)


# ── Retrieve ─────────────────────────────────────────────────────────


class TestRetrieve:
    """Tests for the retrieve() dispatcher."""

    def test_spectral_roundtrip_single_param(self, tiny_emulator):
        """Retrieve rds from emulator-generated spectrum."""
        true_rds = 1500
        obs = tiny_emulator.predict(rds=true_rds, black_carbon=0)

        result = retrieve(
            observed=obs,
            parameters=["rds"],
            emulator=tiny_emulator,
            fixed_params={"black_carbon": 0},
        )
        assert result.converged
        assert abs(result.best_fit["rds"] - true_rds) < 200
        assert result.predicted_albedo.shape == (480,)
        assert result.method == "L-BFGS-B"

    def test_spectral_roundtrip_two_params(self, tiny_emulator):
        """Retrieve rds + black_carbon from emulator spectrum."""
        true_rds = 1500
        true_bc = 10000
        obs = tiny_emulator.predict(rds=true_rds, black_carbon=true_bc)

        result = retrieve(
            observed=obs,
            parameters=["rds", "black_carbon"],
            emulator=tiny_emulator,
        )
        # With a tiny emulator (100 samples) the 2-param fit may not
        # recover exact values, but should produce a valid result
        assert result.predicted_albedo.shape == (480,)
        assert "rds" in result.best_fit
        assert "black_carbon" in result.best_fit
        assert result.n_function_evals > 0

    def test_uncertainty_is_finite(self, tiny_emulator):
        """Uncertainty should be finite and positive for well-constrained params."""
        obs = tiny_emulator.predict(rds=1000, black_carbon=5000)
        result = retrieve(
            observed=obs,
            parameters=["rds", "black_carbon"],
            emulator=tiny_emulator,
        )
        for name in ["rds", "black_carbon"]:
            unc = result.uncertainty[name]
            assert np.isfinite(unc), f"{name} uncertainty is not finite"
            assert unc > 0, f"{name} uncertainty should be positive"

    def test_band_mode_sentinel2(self, tiny_emulator):
        """Retrieve from Sentinel-2 band values."""
        predicted = tiny_emulator.predict(rds=1000, black_carbon=5000)
        flx_slr = tiny_emulator.flx_slr

        from biosnicar.bands import to_platform
        s2 = to_platform(predicted, "sentinel2", flx_slr=flx_slr)
        obs_bands = np.array([s2.B3, s2.B4, s2.B11])

        result = retrieve(
            observed=obs_bands,
            parameters=["rds", "black_carbon"],
            emulator=tiny_emulator,
            platform="sentinel2",
            observed_band_names=["B3", "B4", "B11"],
        )
        assert result.converged
        assert result.predicted_albedo.shape == (480,)

    def test_fixed_params(self, tiny_emulator):
        """fixed_params should not appear in best_fit."""
        obs = tiny_emulator.predict(rds=1000, black_carbon=5000)
        result = retrieve(
            observed=obs,
            parameters=["rds"],
            emulator=tiny_emulator,
            fixed_params={"black_carbon": 5000},
        )
        assert "rds" in result.best_fit
        assert "black_carbon" not in result.best_fit

    def test_differential_evolution(self, tiny_emulator):
        """Differential evolution should converge."""
        obs = tiny_emulator.predict(rds=1500, black_carbon=5000)
        result = retrieve(
            observed=obs,
            parameters=["rds"],
            emulator=tiny_emulator,
            fixed_params={"black_carbon": 5000},
            method="differential_evolution",
        )
        assert result.converged
        assert abs(result.best_fit["rds"] - 1500) < 300
        assert result.method == "differential_evolution"

    def test_nelder_mead(self, tiny_emulator):
        """Nelder-Mead should run and produce a result."""
        obs = tiny_emulator.predict(rds=1500, black_carbon=5000)
        result = retrieve(
            observed=obs,
            parameters=["rds"],
            emulator=tiny_emulator,
            fixed_params={"black_carbon": 5000},
            method="Nelder-Mead",
        )
        assert result.method == "Nelder-Mead"
        assert result.predicted_albedo.shape == (480,)

    def test_no_emulator_no_forward_fn_raises(self):
        """Should raise if neither emulator nor forward_fn provided."""
        with pytest.raises(ValueError, match="emulator.*forward_fn"):
            retrieve(
                observed=np.zeros(480),
                parameters=["rds"],
            )

    def test_binary_param_raises(self, tiny_emulator):
        """Binary parameters like 'direct' cannot be continuously optimised."""
        with pytest.raises(ValueError, match="Binary parameters"):
            retrieve(
                observed=np.zeros(480),
                parameters=["rds", "direct"],
                emulator=tiny_emulator,
            )

    def test_band_mode_without_band_names_raises(self, tiny_emulator):
        """Band mode requires observed_band_names."""
        with pytest.raises(ValueError, match="observed_band_names"):
            retrieve(
                observed=np.array([0.5, 0.4, 0.1]),
                parameters=["rds"],
                emulator=tiny_emulator,
                platform="sentinel2",
            )

    def test_flx_slr_stored_on_result(self, tiny_emulator):
        obs = tiny_emulator.predict(rds=1000, black_carbon=500)
        result = retrieve(
            observed=obs, parameters=["rds"], emulator=tiny_emulator,
            fixed_params={"black_carbon": 500},
        )
        assert result.flx_slr is not None
        assert result.flx_slr.shape == (480,)

    def test_to_outputs_from_retrieve(self, tiny_emulator):
        from biosnicar.classes.outputs import Outputs
        obs = tiny_emulator.predict(rds=1000, black_carbon=500)
        result = retrieve(
            observed=obs, parameters=["rds"], emulator=tiny_emulator,
            fixed_params={"black_carbon": 500},
        )
        out = result.to_outputs()
        assert isinstance(out, Outputs)
        assert out.albedo.shape == (480,)
        assert 0.0 <= out.BBA <= 1.0

    def test_to_platform_from_retrieve(self, tiny_emulator):
        obs = tiny_emulator.predict(rds=1000, black_carbon=500)
        result = retrieve(
            observed=obs, parameters=["rds"], emulator=tiny_emulator,
            fixed_params={"black_carbon": 500},
        )
        bands = result.to_outputs().to_platform("sentinel2")
        assert hasattr(bands, "B3")
        assert 0.0 <= bands.B3 <= 1.0

    def test_flx_slr_none_without_emulator(self, tiny_emulator):
        # When using forward_fn only (no emulator), flx_slr should be None
        def fwd(rds):
            return tiny_emulator.predict(rds=rds, black_carbon=0)
        result = retrieve(
            observed=np.full(480, 0.5),
            parameters=["rds"],
            forward_fn=fwd,
            bounds={"rds": (500, 2000)},
        )
        assert result.flx_slr is None

    def test_summary_output(self, tiny_emulator):
        """result.summary() should return a non-empty string."""
        obs = tiny_emulator.predict(rds=1000, black_carbon=5000)
        result = retrieve(
            observed=obs,
            parameters=["rds", "black_carbon"],
            emulator=tiny_emulator,
        )
        s = result.summary()
        assert "rds" in s
        assert "black_carbon" in s
        assert "L-BFGS-B" in s

    def test_ssa_with_rds_raises(self, tiny_emulator):
        """Cannot retrieve SSA alongside rds."""
        with pytest.raises(ValueError, match="Cannot retrieve 'ssa' alongside"):
            retrieve(
                observed=np.zeros(480),
                parameters=["ssa", "rds"],
                emulator=tiny_emulator,
            )

    def test_ssa_with_rho_raises(self, tiny_emulator):
        """Cannot retrieve SSA alongside rho."""
        with pytest.raises(ValueError, match="Cannot retrieve 'ssa' alongside"):
            retrieve(
                observed=np.zeros(480),
                parameters=["ssa", "rho"],
                emulator=tiny_emulator,
            )

    def test_direct_forward_fn_mode(self, tiny_emulator):
        """Direct forward function mode (no emulator) should work."""
        # Use the tiny_emulator.predict as a "direct" forward function
        true_rds = 1500

        def my_forward(rds):
            return tiny_emulator.predict(rds=rds, black_carbon=0)

        obs = my_forward(rds=true_rds)
        result = retrieve(
            observed=obs,
            parameters=["rds"],
            forward_fn=my_forward,
            bounds={"rds": (500, 3000)},
        )
        assert result.converged
        assert abs(result.best_fit["rds"] - true_rds) < 200


# ── run_emulator ────────────────────────────────────────────────────


class TestRunEmulator:
    """Tests for the run_emulator() driver function."""

    def test_returns_outputs(self, tiny_emulator):
        """run_emulator should return an Outputs object."""
        from biosnicar.classes.outputs import Outputs

        outputs = run_emulator(tiny_emulator, rds=1000, black_carbon=5000)
        assert isinstance(outputs, Outputs)

    def test_albedo_shape(self, tiny_emulator):
        outputs = run_emulator(tiny_emulator, rds=1000, black_carbon=5000)
        assert outputs.albedo.shape == (480,)

    def test_bba_is_scalar(self, tiny_emulator):
        outputs = run_emulator(tiny_emulator, rds=1000, black_carbon=5000)
        assert isinstance(outputs.BBA, float)
        assert 0.0 < outputs.BBA < 1.0

    def test_bbavis_and_bbanir(self, tiny_emulator):
        outputs = run_emulator(tiny_emulator, rds=1000, black_carbon=5000)
        assert isinstance(outputs.BBAVIS, float)
        assert isinstance(outputs.BBANIR, float)
        assert 0.0 < outputs.BBAVIS < 1.0
        assert 0.0 < outputs.BBANIR < 1.0

    def test_flx_slr_populated(self, tiny_emulator):
        outputs = run_emulator(tiny_emulator, rds=1000, black_carbon=5000)
        assert outputs.flx_slr is not None
        assert len(outputs.flx_slr) == 480

    def test_to_platform_works(self, tiny_emulator):
        """Outputs.to_platform() should work with emulator output."""
        outputs = run_emulator(tiny_emulator, rds=1000, black_carbon=5000)
        s2 = outputs.to_platform("sentinel2")
        assert hasattr(s2, "B3")
        assert hasattr(s2, "B4")
        assert 0.0 < s2.B3 < 1.0

    def test_matches_raw_predict(self, tiny_emulator):
        """Albedo from run_emulator should match emulator.predict."""
        outputs = run_emulator(tiny_emulator, rds=1000, black_carbon=5000)
        raw = tiny_emulator.predict(rds=1000, black_carbon=5000)
        np.testing.assert_allclose(outputs.albedo, raw, atol=1e-10)

    def test_non_rt_fields_are_none(self, tiny_emulator):
        """Fields only produced by the full RT solver should be None."""
        outputs = run_emulator(tiny_emulator, rds=1000, black_carbon=5000)
        assert outputs.heat_rt is None
        assert outputs.absorbed_flux_per_layer is None


# ── Backward compatibility ──────────────────────────────────────────


class TestBackwardCompatImports:
    """Verify that old import paths still work."""

    def test_inverse_emulator_import(self):
        from biosnicar.inverse.emulator import Emulator as E1
        from biosnicar.emulator import Emulator as E2
        assert E1 is E2

    def test_inverse_init_import(self):
        from biosnicar.inverse import Emulator as E1
        from biosnicar.emulator import Emulator as E2
        assert E1 is E2


# ── SSA Retrieval ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ssa_emulator():
    """Build a 3-parameter emulator (rds, rho, black_carbon) for SSA tests.

    Module-scoped — built once and shared across all SSA tests.
    """
    emu = Emulator.build(
        params={"rds": (500, 3000), "rho": (200, 800), "black_carbon": (0, 50000)},
        n_samples=300,
        progress=False,
        seed=42,
        layer_type=1,
        solzen=50,
    )
    return emu


class TestSSARetrieval:
    """Tests for SSA as a retrievable parameter."""

    def test_ssa_retrieval_smoke(self, ssa_emulator):
        """SSA retrieval should converge and return a positive SSA."""
        from biosnicar.inverse.result import _compute_ssa

        true_rds, true_rho = 1500.0, 500.0
        true_ssa = _compute_ssa(true_rds, true_rho)
        obs = ssa_emulator.predict(rds=true_rds, rho=true_rho, black_carbon=0)

        result = retrieve(
            observed=obs,
            parameters=["ssa"],
            emulator=ssa_emulator,
            fixed_params={"black_carbon": 0},
        )
        assert result.converged
        assert result.best_fit["ssa"] > 0
        # SSA should be within 50% of true value
        assert abs(result.best_fit["ssa"] - true_ssa) / true_ssa < 0.5

    def test_ssa_derived_populated(self, ssa_emulator):
        """result.derived should contain rds_internal and rho_ref."""
        obs = ssa_emulator.predict(rds=1500, rho=500, black_carbon=0)
        result = retrieve(
            observed=obs,
            parameters=["ssa"],
            emulator=ssa_emulator,
            fixed_params={"black_carbon": 0},
        )
        assert "rds_internal" in result.derived
        assert "rho_ref" in result.derived
        assert result.derived["rds_internal"] > 0
        assert result.derived["rho_ref"] > 0

    def test_ssa_rho_override(self, ssa_emulator):
        """ssa_rho kwarg should override the default reference density."""
        obs = ssa_emulator.predict(rds=1500, rho=500, black_carbon=0)
        result = retrieve(
            observed=obs,
            parameters=["ssa"],
            emulator=ssa_emulator,
            fixed_params={"black_carbon": 0},
            ssa_rho=700.0,
        )
        assert result.derived["rho_ref"] == 700.0
        assert result.converged

    def test_ssa_with_impurity(self, ssa_emulator):
        """SSA can be retrieved alongside impurity parameters."""
        obs = ssa_emulator.predict(rds=1500, rho=500, black_carbon=5000)
        result = retrieve(
            observed=obs,
            parameters=["ssa", "black_carbon"],
            emulator=ssa_emulator,
        )
        assert "ssa" in result.best_fit
        assert "black_carbon" in result.best_fit
        assert result.predicted_albedo.shape == (480,)

    def test_ssa_uncertainty_finite(self, ssa_emulator):
        """SSA uncertainty should be finite and positive."""
        obs = ssa_emulator.predict(rds=1500, rho=500, black_carbon=0)
        result = retrieve(
            observed=obs,
            parameters=["ssa"],
            emulator=ssa_emulator,
            fixed_params={"black_carbon": 0},
        )
        assert np.isfinite(result.uncertainty["ssa"])
        assert result.uncertainty["ssa"] > 0

    def test_derived_empty_without_ssa(self, tiny_emulator):
        """result.derived should be empty when SSA is not retrieved."""
        obs = tiny_emulator.predict(rds=1000, black_carbon=5000)
        result = retrieve(
            observed=obs,
            parameters=["rds"],
            emulator=tiny_emulator,
            fixed_params={"black_carbon": 5000},
        )
        assert result.derived == {}
