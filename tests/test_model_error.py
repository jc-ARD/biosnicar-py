"""Tests for the A7 forward-model error covariance (biosnicar.inverse.model_error)."""

from pathlib import Path

import numpy as np
import pytest

from biosnicar.inverse.model_error import (
    DEFAULT_ARTIFACT,
    ModelErrorCovariance,
    fit_model_error,
    fit_model_error_gap_tolerant,
)

_ARTIFACT = Path(DEFAULT_ARTIFACT).exists()


def _synthetic_residuals(n=60, seed=0):
    """Rows = smooth correlated curve + white noise, full VIS coverage."""
    rng = np.random.default_rng(seed)
    wl = np.arange(480)
    mask = np.zeros((n, 480), dtype=bool)
    mask[:, 20:80] = True                     # 60-band "VIS" window
    shape = np.sin(wl / 15.0)                 # one dominant smooth mode
    resid = np.full((n, 480), np.nan)
    amp = rng.normal(0, 0.03, n)
    for i in range(n):
        resid[i, 20:80] = (amp[i] * shape[20:80]
                           + rng.normal(0, 0.004, 60))
    return resid, mask


class TestFit:
    def test_recovers_low_rank_structure(self):
        resid, mask = _synthetic_residuals()
        m = fit_model_error(resid, mask, n_eofs=3)
        # dominant EOF variance >> residual diagonal
        vis = mask[0]
        assert m.eof_var[0] > 1e-4
        assert m.domain[vis].all()
        # correlated residuals -> few effective DOF over the 60 bands
        assert m.effective_dof(vis) < 10

    def test_dense_shape_and_symmetry(self):
        resid, mask = _synthetic_residuals()
        m = fit_model_error(resid, mask, n_eofs=2)
        sub = np.zeros(480, dtype=bool)
        sub[25:65] = True
        S = m.dense(sub)
        assert S.shape == (40, 40)
        assert np.allclose(S, S.T)
        assert np.linalg.eigvalsh(S).min() > 0   # positive definite

    def test_uncovered_bands_get_fallback_floor(self):
        resid, mask = _synthetic_residuals(n=20)
        m = fit_model_error(resid, mask, n_eofs=2)
        outside = ~m.domain
        assert outside.any()
        assert (m.diag_var[outside] >= 0.005**2 - 1e-12).all()

    def test_round_trip(self, tmp_path):
        resid, mask = _synthetic_residuals()
        m = fit_model_error(resid, mask, n_eofs=2, meta={"note": "test"})
        p = tmp_path / "se.npz"
        m.save(p)
        m2 = ModelErrorCovariance.load(p)
        vis = mask[0]
        assert np.allclose(m.dense(vis), m2.dense(vis), atol=1e-6)
        assert m2.meta["note"] == "test"


class TestGapTolerant:
    """Gap-tolerant fit must use partially-covering rows and stay PSD."""

    def _mixed_rows(self, n=80, seed=1):
        """Half the rows cover only the first block ('VIS'); half cover both
        blocks ('VIS+SWIR'). A full-coverage SVD would keep only the VIS+SWIR
        half; the gap-tolerant fit must use all rows."""
        rng = np.random.default_rng(seed)
        wl = np.arange(480)
        blockA = (wl >= 20) & (wl < 70)     # ~VIS
        blockB = (wl >= 120) & (wl < 170)   # ~SWIR
        sA = np.sin(wl / 12.0)
        sB = np.cos(wl / 9.0)
        resid = np.full((n, 480), np.nan)
        masks = np.zeros((n, 480), dtype=bool)
        for i in range(n):
            a = rng.normal(0, 0.03)
            full = i % 2 == 0               # even rows cover both blocks
            cover = blockA | blockB if full else blockA
            masks[i] = cover
            val = a * sA + (a * sB if full else 0.0)
            resid[i, cover] = (val[cover]
                               + rng.normal(0, 0.004, cover.sum()))
        return resid, masks, blockA, blockB

    def test_uses_partial_rows_and_is_psd(self):
        resid, masks, blockA, blockB = self._mixed_rows()
        m = fit_model_error_gap_tolerant(resid, masks, n_eofs=3,
                                         min_band_rows=10, min_pair_rows=8)
        # both blocks enter the domain despite half the rows missing block B
        assert m.domain[blockA].all() and m.domain[blockB].all()
        # correlation captured (low effective DOF over the covered bands)
        cov_bands = blockA | blockB
        assert m.effective_dof(cov_bands) < 12
        # valid PSD covariance
        S = m.dense(cov_bands)
        assert np.allclose(S, S.T)
        assert np.linalg.eigvalsh(S).min() > -1e-10
        assert m.meta["fit"] == "gap-tolerant-pairwise"

    def test_undetermined_pairs_are_zeroed(self):
        # min_pair_rows very high -> no cross terms trusted -> near-diagonal
        resid, masks, *_ = self._mixed_rows()
        m = fit_model_error_gap_tolerant(resid, masks, n_eofs=3,
                                         min_band_rows=10, min_pair_rows=999)
        # with no admissible pairs the EOF term vanishes
        assert np.allclose(m.eof_var, 0.0)


@pytest.mark.skipif(not _ARTIFACT, reason="shipped model-error artifact absent")
class TestShippedArtifact:
    def test_loads_and_is_sane(self):
        m = ModelErrorCovariance.load()
        wl = np.arange(480) * 10 + 205
        vis = (wl >= 400) & (wl <= 1000)
        assert m.domain[vis].all()               # VIS fully calibrated
        dof = m.effective_dof(vis)
        assert 1.0 < dof < 15.0                  # strongly correlated
        assert m.meta["holdout_campaign"] == "istomina"
        # per-band model sigma in a physically plausible range
        sig = np.sqrt(np.diag(m.dense(vis)) )
        assert (sig > 0.003).all() and (sig < 0.2).all()

    def test_second_load_is_cached(self):
        assert ModelErrorCovariance.load() is ModelErrorCovariance.load()
