"""The physics-derived forward-model error budget.

Covers the perturbation switches (neutral means untouched), the fixed-input
refactor of the emulator transforms, the arithmetic from one-sigma responses
to a covariance, the shipped artifact, and its use in both retrieval engines.
"""

import warnings

import numpy as np
import pytest

from biosnicar import perturbation
from biosnicar.inverse import forward_uncertainty as fu
from biosnicar.inverse.model_error import (
    ModelErrorCovariance, PhysicsModelError, measurement_space, resolve,
)
from biosnicar.sea_ice.emulator_configs import (
    SEA_ICE_EMULATOR_CONFIGS, load_sea_ice_emulators,
)

WVL = fu.WAVELENGTHS_UM
VIS = (WVL >= 0.4) & (WVL <= 1.0)


def _mid(type_name, **over):
    p = {k: 0.5 * (lo + hi)
         for k, (lo, hi) in SEA_ICE_EMULATOR_CONFIGS[type_name]["params"].items()}
    p.update(solzen=60, direct=1)
    p.update(over)
    return p


# ── perturbation switches ────────────────────────────────────────────────

class TestSwitches:
    def test_unknown_switch_is_refused(self):
        with pytest.raises(KeyError, match="unknown perturbation"):
            with perturbation.perturbed(not_a_switch=1.0):
                pass

    def test_state_is_restored_on_exit_and_on_error(self):
        with perturbation.perturbed(brine_index_offset=0.01):
            assert perturbation.get("brine_index_offset") == 0.01
            with perturbation.perturbed(brine_inclusion_scale=2.0):
                assert perturbation.get("brine_index_offset") == 0.01
                assert perturbation.get("brine_inclusion_scale") == 2.0
            assert perturbation.get("brine_inclusion_scale") == 1.0
        assert not perturbation.active()
        with pytest.raises(RuntimeError):
            with perturbation.perturbed(ocean_reflectance_scale=2.0):
                raise RuntimeError
        assert not perturbation.active()

    def test_neutral_values_do_not_activate_anything(self):
        with perturbation.perturbed(brine_inclusion_scale=1.0,
                                    brine_index_offset=0.0):
            assert not perturbation.active()
            assert not perturbation.sea_ice_direct()

    def test_table_lookup_agrees_with_the_physics_it_tabulates(self):
        """sea_ice_direct swaps the LUT for the calculation behind it; the
        two must be the same model to interpolation accuracy."""
        p = _mid("FYI_bare")
        lut = fu.forward("FYI_bare", p)
        direct = fu.forward("FYI_bare", p, sea_ice_direct=True)
        assert np.abs(lut - direct)[VIS].max() < 0.01

    @pytest.mark.parametrize("type_name,switch,value", [
        ("FYI_bare", "brine_inclusion_scale", 2.0),
        ("FYI_bare", "brine_index_offset", 0.005),
        ("FYI_bare", "bubble_backscatter_scale", 1.2),
        ("FYI_bare", "ice_absorption_blend", 1.0),
        ("FYI_bare", "fresnel_diffuse_offset", 0.02),
        ("FYI_pond", "water_absorption_scale", 1.3),
        ("young_ice", "young_ice_scattering_scale", 1.5),
        ("open_water", "ocean_reflectance_scale", 2.0),
    ])
    def test_each_switch_moves_the_spectrum(self, type_name, switch, value):
        p = _mid(type_name)
        if type_name == "open_water":
            p = dict(solzen=60, wind_speed_ms=5.0, direct=1)
        base = fu.forward(type_name, p, sea_ice_direct=True)
        moved = fu.forward(type_name, p, sea_ice_direct=True, **{switch: value})
        assert np.abs(moved - base).max() > 1e-4

    def test_more_brine_scattering_brightens_the_ice(self):
        p = _mid("FYI_bare")
        base = fu.forward("FYI_bare", p, sea_ice_direct=True)
        more = fu.forward("FYI_bare", p, brine_inclusion_scale=2.0)
        assert (more - base)[VIS].mean() > 0

    def test_weaker_ice_absorption_brightens_only_where_the_data_differ(self):
        """Warren & Brandt (2008) absorbs less than Picard (2016) in the
        blue and is identical beyond the visible."""
        p = _mid("FYI_bare")
        base = fu.forward("FYI_bare", p, sea_ice_direct=True)
        alt = fu.forward("FYI_bare", p, ice_absorption_blend=1.0)
        blue = (WVL >= 0.35) & (WVL <= 0.5)
        swir = (WVL >= 1.2) & (WVL <= 2.5)
        assert (alt - base)[blue].mean() > 0
        assert np.abs(alt - base)[swir].max() < 1e-6


# ── fixed inputs of the transforms ───────────────────────────────────────

class TestFixedInputs:
    def test_every_trained_type_declares_its_fixed_inputs(self):
        for name, cfg in SEA_ICE_EMULATOR_CONFIGS.items():
            if "transform_fn" in cfg:
                assert "fixed" in cfg, name

    def test_unknown_fixed_input_is_refused(self):
        tf = SEA_ICE_EMULATOR_CONFIGS["FYI_bare"]["transform_fn"]
        with pytest.raises(KeyError, match="unknown fixed input"):
            tf(_mid("FYI_bare"), fixed={"dl_density": 800.0})

    def test_default_fixed_inputs_reproduce_the_transform(self):
        for name, cfg in SEA_ICE_EMULATOR_CONFIGS.items():
            if "transform_fn" not in cfg:
                continue
            p = _mid(name)
            assert cfg["transform_fn"](dict(p)) == \
                cfg["transform_fn"](dict(p), fixed=dict(cfg["fixed"]))

    def test_salinity_moves_temperature_not_brine_volume(self):
        """Where brine volume is the retrieved parameter, perturbing the
        fixed salinity must leave the brine volume where the state put it."""
        from biosnicar.sea_ice.brine_volume import compute_brine_volume
        cfg = SEA_ICE_EMULATOR_CONFIGS["FYI_bare"]
        p = _mid("FYI_bare")
        for scale in (1.0, 1.3, 1 / 1.3):
            kw = cfg["transform_fn"](dict(p), fixed={"salinity_scale": scale})
            vb = compute_brine_volume(kw["sea_ice_salinity"][0],
                                      kw["sea_ice_temperature"][0],
                                      bulk_density_kg_m3=kw["rho"][0])
            assert vb == pytest.approx(p["brine_volume_fraction"], rel=1e-4)


# ── from responses to a covariance ───────────────────────────────────────

class TestArithmetic:
    def test_factor_variances_sum_to_the_diagonal(self):
        rng = np.random.default_rng(0)
        d = rng.normal(size=(7, 4, 480)) * 0.01
        S, fv = fu.covariance_from_deltas(d)
        assert np.allclose(np.diag(S), fv.sum(axis=0))
        assert np.linalg.eigvalsh(S).min() > -1e-12

    def test_low_rank_form_keeps_every_band_variance(self):
        rng = np.random.default_rng(1)
        d = rng.normal(size=(40, 3, 480)) * 0.01
        S, _ = fu.covariance_from_deltas(d)
        cov = ModelErrorCovariance.from_dense(S, max_rank=5)
        full = cov.dense(np.ones(480, dtype=bool))
        assert np.allclose(np.diag(full), np.diag(S))
        assert np.linalg.eigvalsh(full).min() > -1e-12
        assert np.allclose(cov.sigma() ** 2, np.diag(S))

    def test_a_rank_one_covariance_is_recovered_exactly(self):
        v = np.sin(np.linspace(0, 3, 480)) * 0.03
        cov = ModelErrorCovariance.from_dense(np.outer(v, v))
        assert cov.meta["n_eofs"] == 1
        assert np.allclose(cov.dense(np.ones(480, dtype=bool)),
                           np.outer(v, v), atol=1e-12)

    def test_one_sigma_response_is_linear_in_the_range(self):
        """The budget is first order.  For a continuous factor the response
        to a half-size perturbation should be about half."""
        p = _mid("FYI_pond", black_carbon=0.0)
        full = next(f for f in fu.FACTORS if f.name == "drained_layer_density")
        half = fu.Factor(**{**full.__dict__, "sigma": full.sigma / 2})
        d1 = fu.factor_delta("FYI_pond", p, full)
        d2 = fu.factor_delta("FYI_pond", p, half)
        assert np.abs(d1[VIS]).max() > 1e-3
        assert np.allclose(d2[VIS], 0.5 * d1[VIS], rtol=0.2, atol=2e-4)

    def test_factors_that_do_not_exist_for_a_type_are_skipped(self):
        names = {f.name for f in fu.applicable_factors("FYI_bare")}
        assert "drained_layer_density" not in names     # retrieved as rho_DL
        assert "grain_shape" not in names               # no granular layer
        assert "interior_density" in names
        names = {f.name for f in fu.applicable_factors("FYI_summer")}
        assert {"drained_layer_density", "grain_shape"} <= names

    def test_clear_sky_diffuse_fraction_is_rayleigh_like(self):
        f60 = fu.clear_sky_diffuse_fraction(60)
        f80 = fu.clear_sky_diffuse_fraction(80)
        i400, i800, i1600 = (np.argmin(np.abs(WVL - w)) for w in (0.4, 0.8, 1.6))
        assert 0.2 < f60[i400] < 0.6
        assert f60[i400] > f60[i800] > f60[i1600]
        assert f60[i1600] < 0.03
        assert (f80 >= f60).all()

    def test_band_space_covariance_is_the_projection(self):
        rng = np.random.default_rng(2)
        d = rng.normal(size=(60, 2, 480)) * 0.01
        S, _ = fu.covariance_from_deltas(d)
        cov = ModelErrorCovariance.from_dense(S, max_rank=480, energy=1.0)
        W = np.abs(rng.normal(size=(480, 5)))
        W /= W.sum(axis=0)
        got = measurement_space(cov, None, W, band_mode=True)
        banded = d.reshape(-1, 480) @ W
        assert np.allclose(got, banded.T @ banded / 60, atol=1e-10)


# ── the shipped artifact ─────────────────────────────────────────────────

class TestArtifact:
    @pytest.fixture(scope="class")
    def budget(self):
        return PhysicsModelError.load()

    def test_covers_every_surface_type(self, budget):
        assert set(budget.by_type) == set(SEA_ICE_EMULATOR_CONFIGS)

    def test_was_built_from_the_table_in_the_code(self, budget):
        """The artifact records the ranges it was built with; they must be
        the ranges in FACTORS, or the file is stale."""
        built = {f["name"]: f["sigma"] for f in budget.meta["factors"]
                 if f["where"] != "realised"}
        assert built == {f.name: f.sigma for f in fu.FACTORS}
        assert budget.factor_names == fu.factor_names()

    def test_reads_no_field_data(self, budget):
        assert budget.meta["source"] == "physics"
        assert "calibration_campaigns" not in budget.meta

    def test_each_covariance_is_valid_and_correlated(self, budget):
        for name, cov in budget.by_type.items():
            S = cov.dense(VIS)
            assert np.linalg.eigvalsh(S).min() > -1e-12, name
            # far fewer independent directions than bands
            assert cov.effective_dof(VIS) < 15, name

    def test_budget_rows_add_up(self, budget):
        for name in budget.by_type:
            b = budget.budget(name, VIS)
            total = b.pop("total")
            assert total == pytest.approx(
                np.sqrt(sum(v ** 2 for v in b.values())), rel=1e-6)
            assert total == pytest.approx(
                np.sqrt((budget.for_type(name).sigma(VIS) ** 2).mean()),
                rel=0.02)

    def test_magnitudes_are_physical(self, budget):
        """A few hundredths of albedo in the visible for ice; open water is
        an analytical model with little held fixed."""
        for name in budget.by_type:
            total = budget.budget(name, VIS)["total"]
            if name == "open_water":
                assert total < 0.03
            else:
                assert 0.01 < total < 0.12, (name, total)

    def test_unknown_type_is_refused(self, budget):
        with pytest.raises(KeyError, match="no physics error budget"):
            budget.for_type("glacier")


# ── in the retrievals ────────────────────────────────────────────────────

class TestRetrieval:
    def test_resolve(self):
        assert resolve(None) is None
        assert isinstance(resolve(True), PhysicsModelError)
        assert isinstance(resolve("physics"), PhysicsModelError)
        assert isinstance(resolve("field"), ModelErrorCovariance)
        assert isinstance(resolve("physics", "FYI_pond"), ModelErrorCovariance)
        field = resolve("field")
        assert resolve(field, "FYI_pond") is field
        with pytest.raises(ValueError, match="model_error"):
            resolve("empirical")
        # an emulator the budget does not know: said, not silently dropped
        with pytest.warns(UserWarning, match="no forward-model error budget"):
            assert resolve("physics", "my_custom_emulator") is None

    def test_auto_is_the_physics_budget_for_oe_only(self):
        from biosnicar.sea_ice.retrieve import _resolve_model_error as f
        assert isinstance(f("auto", "oe", None), PhysicsModelError)
        assert isinstance(f("auto", "oe", "sentinel2"), PhysicsModelError)
        assert f("auto", "L-BFGS-B", None) is None
        assert f(None, "oe", None) is None

    @pytest.fixture(scope="class")
    def spectrum(self):
        emu = load_sea_ice_emulators(["FYI_snow"])["FYI_snow"]
        p = _mid("FYI_snow")
        return emu, emu.predict(**p)

    def test_default_spectral_oe_uses_it(self, spectrum):
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        _, y = spectrum
        kw = dict(observed=y, solzen=60, known_month=4, method="oe")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto = retrieve_sea_ice(**kw)
            phys = retrieve_sea_ice(model_error="physics", **kw)
            none = retrieve_sea_ice(model_error=None, **kw)
            field = retrieve_sea_ice(model_error="field", **kw)
        assert auto.uncertainty == phys.uncertainty
        assert auto.uncertainty != none.uncertainty
        assert auto.uncertainty != field.uncertainty
        # correlated error: fewer independent pieces of evidence
        assert auto.dfs < none.dfs

    def test_band_mode_carries_the_budget(self, spectrum):
        from biosnicar.bands import to_platform
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        emu, y = spectrum
        bands = ["B2", "B3", "B4", "B8", "B11", "B12"]
        br = to_platform(y, "sentinel2", flx_slr=emu.solar_flux(60, 1))
        obs = np.array([getattr(br, b) for b in bands])
        kw = dict(observed=obs, platform="sentinel2", observed_band_names=bands,
                  solzen=60, known_month=4, method="oe")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto = retrieve_sea_ice(**kw)
            none = retrieve_sea_ice(model_error=None, **kw)
            with pytest.raises(ValueError, match="spectral-mode only"):
                retrieve_sea_ice(model_error="field", **kw)
        assert auto.surface_type == none.surface_type == "FYI_snow"
        assert auto.dfs < none.dfs

    def test_engines_agree_under_the_budget(self, spectrum):
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch
        _, y = spectrum
        Y = np.tile(y, (3, 1))
        kw = dict(solzen=60, known_month=4, method="oe")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loop = retrieve_sea_ice_batch(Y, engine="loop", **kw)
            vec = retrieve_sea_ice_batch(Y, engine="vectorized", **kw)
        a = loop.to_xarray()["surface_type_code"].values
        b = vec.to_xarray()["surface_type_code"].values
        assert (a == b).all() and (a != 255).all()

    def test_closure_on_a_small_sample(self):
        """Spectra generated with every fixed input drawn from its stated
        range are covered far better with the budget than without it."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "validation_data"))
        import physics_budget_closure as closure

        emu = load_sea_ice_emulators(["FYI_bare"])["FYI_bare"]
        cov = PhysicsModelError.load().for_type("FYI_bare")
        rows, free, mask = closure.run_type("FYI_bare", emu, cov, 40, 0.005,
                                            seed=11)
        inst = closure.summarise(rows["instrument"], free, mask, 0.005, None)
        phys = closure.summarise(rows["physics"], free, mask, 0.005, cov)
        assert inst["chi2_median"] > 5 * phys["chi2_median"]
        assert phys["chi2_median"] < 3.0
        assert phys["cover2"] > inst["cover2"] + 0.1
        assert phys["cover2"] > 0.9
