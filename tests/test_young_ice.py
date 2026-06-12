"""Tests for the young ice (layer_type=6) extension (B2 / Y-1..Y-6)."""

from pathlib import Path

import numpy as np
import pytest

import biosnicar
from biosnicar.drivers.run_model import run_model
from biosnicar.optical_properties.column_OPs import _compute_young_ice_ops
from biosnicar.sea_ice.emulator_configs import SEA_ICE_EMULATOR_CONFIGS


@pytest.fixture(scope="module")
def flx():
    d = np.load(str(biosnicar.DATA_DIR / "OP_data" / "480band" / "fsds.npz"))
    f = d["swnb_480bnd_mlw_clr_SZA60"].copy()
    f[f <= 0] = 1e-30
    return f


def _bba(albedo, flx):
    return float(albedo @ flx / flx.sum())


class TestYoungIceForwardModel:
    """Y-1/Y-6: thin-slab albedo against Grenfell & Maykut (1977) brackets."""

    @pytest.mark.parametrize("d,lo,hi", [
        (0.01, 0.05, 0.10),   # grease ice (Y-1 acceptance bracket)
        (0.03, 0.08, 0.12),   # dark nilas
        (0.08, 0.10, 0.18),   # light nilas
        (0.12, 0.15, 0.22),   # grey ice
    ])
    def test_bba_vs_thickness_brackets(self, flx, d, lo, hi):
        a = _compute_young_ice_ops(d, -10.0, 25.0, 0.04, solzen=60, direct=1)
        bba = _bba(a, flx)
        # ±0.03 tolerance per young-ice-build-spec.md acceptance table
        assert lo - 0.03 <= bba <= hi + 0.03, f"d={d}: BBA={bba:.3f}"

    def test_albedo_increases_with_thickness(self, flx):
        bbas = [
            _bba(_compute_young_ice_ops(d, -10, 25, 0.04, 60, 1), flx)
            for d in (0.005, 0.02, 0.05, 0.10, 0.20, 0.30)
        ]
        assert all(b2 > b1 for b1, b2 in zip(bbas, bbas[1:]))

    def test_warmer_ice_is_darker(self, flx):
        cold = _bba(_compute_young_ice_ops(0.10, -15, 25, 0.04, 60, 1), flx)
        warm = _bba(_compute_young_ice_ops(0.10, -3, 25, 0.04, 60, 1), flx)
        assert warm < cold  # more brine at the liquidus → more absorption

    def test_thin_ice_tracks_ocean_albedo(self, flx):
        dark = _bba(_compute_young_ice_ops(0.005, -10, 25, 0.03, 60, 1), flx)
        bright = _bba(_compute_young_ice_ops(0.005, -10, 25, 0.08, 60, 1), flx)
        assert bright > dark

    def test_physical_range(self):
        a = _compute_young_ice_ops(0.15, -5, 15, 0.05, 45, 1)
        assert a.shape == (480,)
        assert np.all(a >= 0) and np.all(a <= 1)

    def test_invalid_thickness_raises(self):
        with pytest.raises(ValueError, match="ice_thickness"):
            _compute_young_ice_ops(0.0, -10, 25, 0.04)


class TestRunModelIntegration:
    """Y-2: layer_type=6 boundary-condition path in run_model."""

    def test_acceptance_bba(self):
        out = run_model(
            layer_type=6, ice_thickness=0.05, sea_ice_temperature=-10,
            sea_ice_salinity=20, ocean_albedo=0.04, solzen=60, direct=1,
        )
        assert 0.07 <= float(out.BBA) <= 0.15  # spec: ≈ 0.10
        assert out.albedo.shape == (480,)
        bands = out.to_platform("sentinel2")
        assert 0.0 <= float(bands.B3) <= 1.0

    def test_missing_thickness_raises(self):
        with pytest.raises(ValueError, match="ice_thickness"):
            run_model(layer_type=6, sea_ice_temperature=-10,
                      sea_ice_salinity=20, solzen=60)

    def test_mixed_column_raises(self):
        with pytest.raises(ValueError, match="only layer"):
            run_model(
                layer_type=[6, 4], dz=[0.05, 1.0], ice_thickness=0.05,
                sea_ice_temperature=[-10, -10], sea_ice_salinity=[20, 8],
                sea_ice_bubble_radius=[None, 300], solzen=60,
            )


class TestIceChartMapping:
    """Y-4: thickness-resolved WMO / SIGRID-3 codes."""

    def _result(self, thickness_m):
        from biosnicar.sea_ice.retrieve import SeaIceRetrievalResult
        return SeaIceRetrievalResult(
            surface_type="young_ice", surface_description="", confidence=0.9,
            parameters={"ice_thickness": thickness_m},
            uncertainty={}, predicted_albedo=np.zeros(480),
            observed=np.zeros(480), cost=0.0, converged=True, flx_slr=None,
        )

    @pytest.mark.parametrize("d,code", [
        (0.005, "SA"), (0.03, "SB"), (0.12, "SI"), (0.20, "SJ"),
    ])
    def test_sigrid3_codes(self, d, code):
        sig = self._result(d).to_sigrid3()
        assert sig.ice_type_code == code
        assert sig.stage_of_melt == 1

    @pytest.mark.parametrize("d,term", [
        (0.005, "Grease"), (0.03, "Dark nilas"), (0.07, "Light nilas"),
        (0.12, "Grey ice"), (0.20, "Grey-white"),
    ])
    def test_wmo_terms(self, d, term):
        wmo = self._result(d).to_wmo()
        assert term in wmo.stage_of_development


YOUNG_ICE_BUILT = Path(
    SEA_ICE_EMULATOR_CONFIGS["young_ice"]["emulator_file"]
).exists()


@pytest.mark.skipif(not YOUNG_ICE_BUILT, reason="young_ice emulator not built")
class TestYoungIceEmulator:
    """Y-3/Y-5: emulator accuracy and classification integration."""

    @pytest.fixture(scope="class")
    def emu(self):
        from biosnicar.emulator import Emulator
        return Emulator.load(SEA_ICE_EMULATOR_CONFIGS["young_ice"]["emulator_file"])

    def test_heldout_accuracy(self, emu):
        cfg = SEA_ICE_EMULATOR_CONFIGS["young_ice"]
        rng = np.random.default_rng(99)
        preds, refs = [], []
        for _ in range(25):
            p = {
                "ice_thickness_cm": float(10 ** rng.uniform(np.log10(0.5), np.log10(30))),
                "sea_ice_temperature": float(rng.uniform(-20, -2)),
                "sea_ice_salinity": float(rng.uniform(10, 35)),
                "ocean_albedo": float(rng.uniform(0.03, 0.08)),
                "solzen": int(rng.integers(20, 81)),
                "direct": int(rng.integers(0, 2)),
            }
            preds.append(emu.predict(**p))
            refs.append(run_model(**cfg["transform_fn"](p)).albedo)
        preds, refs = np.array(preds), np.array(refs)
        res = refs - preds
        r2 = 1 - res.var() / refs.var()
        assert r2 > 0.98, f"held-out R2={r2:.4f}"
        assert np.abs(res).mean() < 0.01

    def test_classification_freezeup(self, emu):
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        cfg = SEA_ICE_EMULATOR_CONFIGS["young_ice"]
        p = {"ice_thickness_cm": 8.0, "sea_ice_temperature": -8.0,
             "sea_ice_salinity": 25.0, "ocean_albedo": 0.04,
             "solzen": 70, "direct": 1}
        obs = run_model(**cfg["transform_fn"](p)).albedo
        result = retrieve_sea_ice(observed=obs, solzen=70, direct=1,
                                  known_month=11)
        assert result.surface_type == "young_ice"
        assert 0.02 < result.parameters["ice_thickness"] < 0.20
        d = result.quality_flag_description()
        assert d["young_ice_likely"]

    def test_summer_exclusion(self, emu):
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        cfg = SEA_ICE_EMULATOR_CONFIGS["young_ice"]
        p = {"ice_thickness_cm": 8.0, "sea_ice_temperature": -8.0,
             "sea_ice_salinity": 25.0, "ocean_albedo": 0.04,
             "solzen": 70, "direct": 1}
        obs = run_model(**cfg["transform_fn"](p)).albedo
        result = retrieve_sea_ice(observed=obs, solzen=70, direct=1,
                                  known_month=7)
        assert result.surface_type != "young_ice"
