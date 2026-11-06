"""Tests for the optimal-estimation engine (biosnicar.inverse.optimal_estimation).

The anchor is the linear-Gaussian case, where OE has a closed-form answer and
the engine must reproduce it to machine precision — that proves the maths.
"""

import numpy as np
import pytest

from biosnicar.inverse.optimal_estimation import (
    OEResult,
    optimal_estimation,
)


def _analytic_linear(K0, y, x_a, S_a, S_e):
    """Closed-form MAP estimate and posterior covariance for F(x)=K0 x."""
    Se_inv = np.linalg.inv(np.diag(S_e) if S_e.ndim == 1 else S_e)
    Sa_inv = np.linalg.inv(np.diag(S_a) if S_a.ndim == 1 else S_a)
    H = K0.T @ Se_inv @ K0 + Sa_inv
    S_hat = np.linalg.inv(H)
    x_hat = x_a + S_hat @ K0.T @ Se_inv @ (y - K0 @ x_a)
    A = S_hat @ (K0.T @ Se_inv @ K0)
    return x_hat, S_hat, A


class TestLinearGaussianExactness:
    """F(x) = K0 x : OE must recover the analytic posterior exactly."""

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.m, self.n = 12, 3
        self.K0 = rng.normal(size=(self.m, self.n))
        self.x_true = np.array([1.0, -2.0, 0.5])
        self.S_e = np.full(self.m, 0.04**2)        # diagonal noise
        self.S_a = np.array([1.0, 1.0, 1.0])       # broad prior
        self.x_a = np.zeros(self.n)
        self.y = self.K0 @ self.x_true + rng.normal(0, 0.04, self.m)

    def _run(self):
        return optimal_estimation(
            lambda x: self.K0 @ x, self.y, self.x_a, self.S_a, self.S_e,
            fd_step=np.full(self.n, 1e-4),
        )

    def test_state_matches_closed_form(self):
        r = self._run()
        x_hat, _, _ = _analytic_linear(self.K0, self.y, self.x_a,
                                       self.S_a, self.S_e)
        np.testing.assert_allclose(r.x, x_hat, rtol=1e-7, atol=1e-9)

    def test_posterior_covariance_matches(self):
        r = self._run()
        _, S_hat, _ = _analytic_linear(self.K0, self.y, self.x_a,
                                       self.S_a, self.S_e)
        np.testing.assert_allclose(r.S, S_hat, rtol=1e-6, atol=1e-10)

    def test_averaging_kernel_and_dfs(self):
        r = self._run()
        _, _, A = _analytic_linear(self.K0, self.y, self.x_a, self.S_a, self.S_e)
        np.testing.assert_allclose(r.averaging_kernel, A, rtol=1e-6, atol=1e-9)
        assert r.dfs == pytest.approx(np.trace(A), rel=1e-6)
        # 3 well-observed params under a broad prior -> DFS near 3
        assert 2.5 < r.dfs <= 3.0 + 1e-6

    def test_full_matrix_Se_matches_diagonal(self):
        r_diag = self._run()
        r_full = optimal_estimation(
            lambda x: self.K0 @ x, self.y, self.x_a, self.S_a,
            np.diag(self.S_e), fd_step=np.full(self.n, 1e-4),
        )
        np.testing.assert_allclose(r_full.x, r_diag.x, rtol=1e-8)
        np.testing.assert_allclose(r_full.S, r_diag.S, rtol=1e-8)


class TestInformationContent:
    """DFS and the averaging kernel must respond to information sensibly."""

    def test_tight_prior_lowers_dfs(self):
        rng = np.random.default_rng(1)
        K0 = rng.normal(size=(10, 2))
        y = K0 @ np.array([1.0, 1.0])
        broad = optimal_estimation(lambda x: K0 @ x, y, np.zeros(2),
                                   np.array([10.0, 10.0]), np.full(10, 0.05**2),
                                   fd_step=np.full(2, 1e-4))
        tight = optimal_estimation(lambda x: K0 @ x, y, np.zeros(2),
                                   np.array([1e-6, 1e-6]), np.full(10, 0.05**2),
                                   fd_step=np.full(2, 1e-4))
        # A near-fixed prior leaves almost nothing for the data to determine.
        assert broad.dfs > 1.8
        assert tight.dfs < 0.2

    def test_noisier_data_lowers_dfs(self):
        rng = np.random.default_rng(2)
        K0 = rng.normal(size=(8, 2))
        y = K0 @ np.array([0.5, -0.5])
        clean = optimal_estimation(lambda x: K0 @ x, y, np.zeros(2),
                                   np.full(2, 1.0), np.full(8, 0.01**2),
                                   fd_step=np.full(2, 1e-4))
        noisy = optimal_estimation(lambda x: K0 @ x, y, np.zeros(2),
                                   np.full(2, 1.0), np.full(8, 1.0**2),
                                   fd_step=np.full(2, 1e-4))
        assert clean.dfs > noisy.dfs

    def test_uninformative_parameter_flagged(self):
        # Second parameter has zero effect on the measurement -> A[1,1]≈0.
        K0 = np.array([[1.0, 0.0], [2.0, 0.0], [1.5, 0.0]])
        y = K0 @ np.array([1.0, 99.0])
        r = optimal_estimation(lambda x: K0 @ x, y, np.zeros(2),
                               np.full(2, 4.0), np.full(3, 0.02**2),
                               fd_step=np.full(2, 1e-4))
        pd = r.parameter_dfs
        assert pd[0] > 0.9          # determined by data
        assert pd[1] < 0.05         # determined by prior (no sensitivity)


class TestNonlinearRecovery:
    """A mildly non-linear forward model still recovers the truth."""

    def test_quadratic_forward(self):
        # F_i(x) = a_i * x0 + b_i * x1^2  -> non-linear in x1
        rng = np.random.default_rng(3)
        a = rng.uniform(0.5, 1.5, 15)
        b = rng.uniform(0.5, 1.5, 15)
        x_true = np.array([2.0, 1.3])

        def F(x):
            return a * x[0] + b * x[1] ** 2

        y = F(x_true) + rng.normal(0, 1e-3, 15)
        r = optimal_estimation(F, y, np.array([0.0, 0.5]), np.full(2, 5.0),
                               np.full(15, 1e-3**2), bounds=np.array([[-5, 5], [0, 5]]))
        assert r.converged
        np.testing.assert_allclose(r.x, x_true, atol=0.02)
        assert r.chi2_reduced < 5.0


class TestEvidenceForModelSelection:
    """Log-evidence must prefer the model that genuinely generated the data."""

    def test_correct_model_wins(self):
        rng = np.random.default_rng(4)
        # Two candidate linear models; data generated by model A.
        KA = rng.normal(size=(20, 2))
        KB = rng.normal(size=(20, 2))
        x_true = np.array([1.0, -1.0])
        y = KA @ x_true + rng.normal(0, 0.02, 20)
        Se = np.full(20, 0.02**2)
        Sa = np.full(2, 4.0)
        rA = optimal_estimation(lambda x: KA @ x, y, np.zeros(2), Sa, Se,
                                fd_step=np.full(2, 1e-4))
        rB = optimal_estimation(lambda x: KB @ x, y, np.zeros(2), Sa, Se,
                                fd_step=np.full(2, 1e-4))
        assert rA.log_evidence > rB.log_evidence
        assert rA.chi2_reduced < 2.0     # A explains the data
        assert rB.chi2_reduced > rA.chi2_reduced


class TestResultObject:
    def test_sigma_and_repr(self):
        K0 = np.eye(3)
        r = optimal_estimation(lambda x: K0 @ x, np.ones(3), np.zeros(3),
                               np.full(3, 1.0), np.full(3, 0.1**2),
                               fd_step=np.full(3, 1e-4))
        assert isinstance(r, OEResult)
        assert r.sigma.shape == (3,)
        assert np.all(r.sigma > 0)


# ── Integration with the sea-ice retrieval (requires built emulators) ────────

from pathlib import Path  # noqa: E402

from biosnicar.sea_ice.emulator_configs import (  # noqa: E402
    SEA_ICE_EMULATOR_CONFIGS, trained_emulator_names,
)

_BUILT = all(Path(SEA_ICE_EMULATOR_CONFIGS[n]["emulator_file"]).exists()
             for n in trained_emulator_names())


@pytest.mark.skipif(not _BUILT, reason="pre-built sea ice emulators not found")
class TestOEIntegration:
    def _pond_obs(self):
        from biosnicar.drivers.run_model import run_model
        truth = dict(pond_depth=0.25, sea_ice_temperature=-4.0,
                     black_carbon=100.0, solzen=60, direct=1)
        a = np.asarray(run_model(
            **SEA_ICE_EMULATOR_CONFIGS["FYI_pond"]["transform_fn"](truth)).albedo)
        return np.clip(a + np.random.default_rng(1).normal(0, 0.005, 480), 0, 1)

    def test_retrieve_oe_method(self):
        from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
        from biosnicar.inverse.optimize import retrieve
        emu = load_sea_ice_emulators(["FYI_pond"])["FYI_pond"]
        r = retrieve(observed=self._pond_obs(),
                     parameters=["pond_depth", "sea_ice_temperature", "black_carbon"],
                     emulator=emu, fixed_params={"solzen": 60, "direct": 1},
                     obs_uncertainty=np.full(480, 0.005), method="oe")
        assert r.method == "oe"
        assert abs(r.best_fit["pond_depth"] - 0.25) < 0.03
        assert r.dfs is not None and 0 < r.dfs <= 3 + 1e-6
        assert r.uncertainty["pond_depth"] > 0           # has an error bar
        assert r.log_evidence is not None
        # pond depth is well constrained by water absorption
        assert r.averaging_kernel_diag["pond_depth"] > 0.7

    def test_retrieve_sea_ice_oe_classification(self):
        from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        fleet = load_sea_ice_emulators()
        r = retrieve_sea_ice(observed=self._pond_obs(), emulators=fleet,
                             solzen=60, direct=1, known_month=7, method="oe",
                             obs_uncertainty=np.full(480, 0.005))
        assert r.surface_type == "FYI_pond"
        # class probabilities are a normalised distribution
        assert r.class_probabilities
        assert abs(sum(r.class_probabilities.values()) - 1.0) < 1e-6
        assert 0.0 <= r.confidence <= 1.0
        assert r.confidence == pytest.approx(max(r.class_probabilities.values()))
        assert r.dfs is not None

    def test_oe_agrees_with_lbfgsb_on_well_constrained_case(self):
        """Validation: on a well-constrained spectrum, the OE point estimate
        should agree with the established L-BFGS-B optimiser. (If it didn't,
        the OE integration would be suspect.)"""
        from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
        from biosnicar.inverse.optimize import retrieve
        emu = load_sea_ice_emulators(["FYI_pond"])["FYI_pond"]
        obs = self._pond_obs()
        common = dict(observed=obs, parameters=["pond_depth", "black_carbon"],
                      emulator=emu,
                      fixed_params={"sea_ice_temperature": -4.0,
                                    "solzen": 60, "direct": 1},
                      obs_uncertainty=np.full(480, 0.005))
        oe = retrieve(method="oe", **common)
        lb = retrieve(method="L-BFGS-B", **common)
        assert abs(oe.best_fit["pond_depth"] - lb.best_fit["pond_depth"]) < 0.02

    def test_dfs_drops_with_fewer_bands(self):
        """Validation of the headline OE behaviour: a full spectrum carries
        more information (higher DFS) than a few satellite bands."""
        from biosnicar.bands import to_platform
        from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
        from biosnicar.inverse.optimize import retrieve
        emu = load_sea_ice_emulators(["FYI_pond"])["FYI_pond"]
        obs = self._pond_obs()
        free = ["pond_depth", "sea_ice_temperature", "black_carbon"]
        fixed = {"solzen": 60, "direct": 1}
        hs = retrieve(observed=obs, parameters=free, emulator=emu,
                      fixed_params=fixed, obs_uncertainty=np.full(480, 0.005),
                      method="oe")
        bands = ["B2", "B3", "B4", "B8"]
        br = to_platform(obs, "sentinel2", flx_slr=emu.flx_slr)
        y = np.array([getattr(br, b) for b in bands])
        s2 = retrieve(observed=y, parameters=free, emulator=emu,
                      platform="sentinel2", observed_band_names=bands,
                      fixed_params=fixed, obs_uncertainty=np.full(len(bands), 0.02),
                      method="oe")
        assert hs.dfs > s2.dfs
        # pond depth stays measured in both; temperature is the one that
        # collapses onto the prior with only 4 VIS-NIR bands
        assert hs.averaging_kernel_diag["pond_depth"] > 0.7
        assert s2.averaging_kernel_diag["sea_ice_temperature"] < \
            hs.averaging_kernel_diag["sea_ice_temperature"]


def test_prior_dominated_parameters_unit():
    """prior_dominated_parameters() thresholds the averaging-kernel diagonal."""
    from biosnicar.sea_ice.retrieve import SeaIceRetrievalResult
    r = SeaIceRetrievalResult(
        surface_type="FYI_pond", surface_description="", confidence=1.0,
        parameters={}, uncertainty={}, predicted_albedo=np.zeros(1),
        observed=np.zeros(1), cost=0.0, converged=True, flx_slr=None,
        averaging_kernel_diag={"measured": 0.92, "prior_bound": 0.15},
    )
    flags = r.prior_dominated_parameters()           # default threshold 0.5
    assert flags == {"measured": False, "prior_bound": True}
    # threshold is adjustable
    assert r.prior_dominated_parameters(threshold=0.95)["measured"] is True


@pytest.mark.skipif(not _BUILT, reason="pre-built sea ice emulators not found")
class TestA3PriorProvenance:
    """A3: priors-on/priors-off (spectrum-only) toggle and prior_resolved flag."""

    def _young_ice_obs(self):
        from biosnicar.drivers.run_model import run_model
        truth = dict(ice_thickness_cm=6.0, sea_ice_temperature=-8.0,
                     sea_ice_salinity=20.0, ocean_albedo=0.05,
                     solzen=60, direct=1)
        a = np.asarray(run_model(
            **SEA_ICE_EMULATOR_CONFIGS["young_ice"]["transform_fn"](truth)).albedo)
        return np.clip(a + np.random.default_rng(2).normal(0, 0.004, 480), 0, 1)

    def _pond_obs(self):
        from biosnicar.drivers.run_model import run_model
        truth = dict(pond_depth=0.25, sea_ice_temperature=-4.0,
                     black_carbon=100.0, solzen=60, direct=1)
        a = np.asarray(run_model(
            **SEA_ICE_EMULATOR_CONFIGS["FYI_pond"]["transform_fn"](truth)).albedo)
        return np.clip(a + np.random.default_rng(1).normal(0, 0.005, 480), 0, 1)

    def _common(self, obs):
        from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
        return dict(observed=obs, emulators=load_sea_ice_emulators(),
                    solzen=60, direct=1, method="oe",
                    obs_uncertainty=np.full(480, 0.004))

    def test_use_priors_toggle_changes_candidate_set(self):
        """known_month excludes young_ice in melt season; use_priors=False keeps it."""
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        common = self._common(self._young_ice_obs())
        on = retrieve_sea_ice(known_month=7, **common)
        off = retrieve_sea_ice(known_month=7, use_priors=False, **common)
        assert "young_ice" not in on.class_probabilities      # excluded by prior
        assert "young_ice" in off.class_probabilities          # spectrum-only keeps it

    def test_prior_resolved_true_when_prior_flips_class(self):
        """A young-ice spectrum: spectrum alone says young_ice, the melt-season
        prior forbids it -> the classification is resolved by the prior."""
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        r = retrieve_sea_ice(known_month=7, flag_prior_influence=True,
                             **self._common(self._young_ice_obs()))
        assert r.spectrum_only_surface_type == "young_ice"
        assert r.surface_type != "young_ice"
        assert r.prior_resolved is True
        assert r.spectrum_only_class_probabilities                # populated

    def test_prior_resolved_false_when_spectrum_agrees(self):
        """A clean pond: prior and spectrum agree -> not prior-resolved."""
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        r = retrieve_sea_ice(known_month=7, flag_prior_influence=True,
                             **self._common(self._pond_obs()))
        assert r.prior_resolved is False
        assert r.spectrum_only_surface_type == r.surface_type

    def test_prior_resolved_none_without_metadata_prior(self):
        """No known_month -> no active prior -> nothing to diagnose."""
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        r = retrieve_sea_ice(flag_prior_influence=True,
                             **self._common(self._pond_obs()))
        assert r.prior_resolved is None


class TestParameterCorrelation:
    """`parameter_correlation` / `degenerate_parameters` on RetrievalResult.

    These exist so callers can read the *measured* collinearity of their own
    retrieval instead of relying on a blanket documented warning.  For the
    FYI_bare rho_DL/brine_volume_fraction pair the correlation is strongly
    state-dependent and changes sign, so a blanket claim would be wrong.
    """

    def _result(self):
        from biosnicar.inverse.result import RetrievalResult
        # Hand-built 2-parameter posterior with a known correlation of 0.6.
        s1, s2, r = 2.0, 5.0, 0.6
        pc = {
            ("a", "a"): s1 ** 2,
            ("b", "b"): s2 ** 2,
            ("a", "b"): r * s1 * s2,
            ("b", "a"): r * s1 * s2,
        }
        return RetrievalResult(
            best_fit={"a": 1.0, "b": 2.0}, cost=0.0,
            uncertainty={"a": s1, "b": s2},
            predicted_albedo=np.zeros(3), observed=np.zeros(3),
            converged=True, method="oe", n_function_evals=1,
            posterior_covariance=pc,
        )

    def test_correlation_matches_covariance(self):
        c = self._result().parameter_correlation()
        assert c[("a", "a")] == pytest.approx(1.0)
        assert c[("b", "b")] == pytest.approx(1.0)
        assert c[("a", "b")] == pytest.approx(0.6)
        assert c[("a", "b")] == pytest.approx(c[("b", "a")])

    def test_degenerate_pairs_respect_threshold(self):
        r = self._result()
        assert r.degenerate_parameters(threshold=0.5) == [("a", "b", pytest.approx(0.6))]
        assert r.degenerate_parameters(threshold=0.8) == []

    def test_empty_without_oe(self):
        from biosnicar.inverse.result import RetrievalResult
        r = RetrievalResult(
            best_fit={"a": 1.0}, cost=0.0, uncertainty={"a": 1.0},
            predicted_albedo=np.zeros(3), observed=np.zeros(3),
            converged=True, method="L-BFGS-B", n_function_evals=1,
        )
        assert r.parameter_correlation() == {}
        assert r.degenerate_parameters() == []

    def test_variance_inflation_identity(self):
        """sigma_marginal / sigma_conditional == 1/sqrt(1-r^2) exactly.

        This is the identity the docs quote when turning a correlation into a
        cost, so it is pinned here rather than asserted in prose.
        """
        c = self._result().parameter_correlation()[("a", "b")]
        pc = self._result().posterior_covariance
        S = np.array([[pc[("a", "a")], pc[("a", "b")]],
                      [pc[("b", "a")], pc[("b", "b")]]])
        H = np.linalg.inv(S)
        sigma_marginal = np.sqrt(S[0, 0])
        sigma_conditional = 1.0 / np.sqrt(H[0, 0])
        assert sigma_marginal / sigma_conditional == pytest.approx(
            1.0 / np.sqrt(1.0 - c ** 2)
        )


class TestSSAFromSeaIceParameters:
    """SSA must be reachable from the sea-ice parameter names, and its
    uncertainty must use the radius/density covariance.

    Before this, `ssa` keyed only off `rds`/`rho` and silently returned None
    for every sea-ice retrieval, even though `_compute_ssa`'s formula is
    derived for bubbly ice.  And `_ssa_uncertainty` assumed independence,
    which overstates sigma_SSA ~3x for FYI_bare because radius and density
    are strongly anti-correlated along the scattering ridge.
    """

    def _result(self, names, corr):
        from biosnicar.inverse.result import RetrievalResult
        r_name, d_name = names
        rad, rho = 400.0, 860.0
        s_r, s_d = 150.0, 18.0
        pc = {
            (r_name, r_name): s_r ** 2,
            (d_name, d_name): s_d ** 2,
            (r_name, d_name): corr * s_r * s_d,
            (d_name, r_name): corr * s_r * s_d,
        }
        return RetrievalResult(
            best_fit={r_name: rad, d_name: rho}, cost=0.0,
            uncertainty={r_name: s_r, d_name: s_d},
            predicted_albedo=np.zeros(3), observed=np.zeros(3),
            converged=True, method="oe", n_function_evals=1,
            posterior_covariance=pc,
        )

    def test_ssa_reachable_from_sea_ice_names(self):
        sea = self._result(("sea_ice_bubble_radius", "rho_DL"), -0.85)
        glacier = self._result(("rds", "rho"), -0.85)
        assert sea.ssa is not None
        assert sea.ssa == pytest.approx(glacier.ssa)
        assert sea.ssa_uncertainty == pytest.approx(glacier.ssa_uncertainty)

    def test_ssa_formula(self):
        r = self._result(("sea_ice_bubble_radius", "rho_DL"), 0.0)
        expected = 3.0 * (1.0 - 860.0 / 917.0) / (400e-6 * 860.0)
        assert r.ssa == pytest.approx(expected)

    def test_anticorrelation_shrinks_ssa_uncertainty(self):
        """Both partials of SSA are negative, so a negative covariance
        subtracts.  This is the whole reason SSA is worth reporting."""
        indep = self._result(("sea_ice_bubble_radius", "rho_DL"), 0.0)
        anti = self._result(("sea_ice_bubble_radius", "rho_DL"), -0.85)
        assert anti.ssa_uncertainty < indep.ssa_uncertainty
        assert anti.ssa_uncertainty == pytest.approx(
            indep.ssa_uncertainty * 0.35, rel=0.2)

    def test_none_when_pair_absent(self):
        from biosnicar.inverse.result import RetrievalResult
        r = RetrievalResult(
            best_fit={"brine_volume_fraction": 0.05}, cost=0.0,
            uncertainty={"brine_volume_fraction": 0.01},
            predicted_albedo=np.zeros(3), observed=np.zeros(3),
            converged=True, method="oe", n_function_evals=1,
            posterior_covariance={("brine_volume_fraction",
                                   "brine_volume_fraction"): 1e-4},
        )
        assert r.ssa is None and r.ssa_uncertainty is None

    def test_falls_back_to_independence_without_oe(self):
        """A non-OE fit has no covariance; SSA must still be computable."""
        from biosnicar.inverse.result import RetrievalResult
        r = RetrievalResult(
            best_fit={"sea_ice_bubble_radius": 400.0, "rho_DL": 860.0},
            cost=0.0,
            uncertainty={"sea_ice_bubble_radius": 150.0, "rho_DL": 18.0},
            predicted_albedo=np.zeros(3), observed=np.zeros(3),
            converged=True, method="L-BFGS-B", n_function_evals=1,
        )
        assert r.ssa is not None
        assert np.isfinite(r.ssa_uncertainty)
