"""Fast guards for the field-residual library (A7 step 1).

The full build (tests/validation_data/build_residual_library.py) runs ~190
fleet retrievals and stays a manual script, like the campaign validations.
These tests only exercise the save/load round-trip and row assembly with
synthetic data, so schema regressions are caught in CI at zero cost.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

VAL = Path(__file__).resolve().parent / "validation_data"
sys.path.insert(0, str(VAL))

from build_residual_library import (  # noqa: E402
    LIBRARY_VERSION, load_residual_library, save_library,
)


def _synthetic_rows(n=3):
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        mask = np.zeros(480, dtype=bool)
        mask[20:80] = True
        resid = np.full(480, np.nan, dtype=np.float32)
        resid[mask] = rng.normal(0, 0.02, mask.sum()).astype(np.float32)
        rows.append(dict(
            campaign="sheba", spectrum_id=f"s{i}", date="1998-04-08",
            label="spring_snow", expected_set="FYI_snow",
            fit_type="FYI_snow", is_winner=(i == 0), is_expected=True,
            winner_type="FYI_snow", confidence=0.9, sza=69.0, direct=1,
            month=4, n_bands=int(mask.sum()),
            rms_vis=float(np.sqrt(np.nanmean(resid[mask] ** 2))),
            residual=resid, mask=mask,
        ))
    return rows


def test_round_trip(tmp_path):
    rows = _synthetic_rows()
    path = tmp_path / "lib.npz"
    save_library(rows, path)
    lib = load_residual_library(path)

    assert int(lib["library_version"]) == LIBRARY_VERSION
    assert lib["residual"].shape == (3, 480)
    assert lib["mask"].dtype == bool
    assert lib["wavelength_nm"].shape == (480,)
    assert list(lib["fit_type"]) == ["FYI_snow"] * 3
    assert lib["is_winner"].sum() == 1
    # residuals are NaN exactly off-mask, finite exactly on-mask
    on = lib["mask"][0]
    assert np.isfinite(lib["residual"][0][on]).all()
    assert np.isnan(lib["residual"][0][~on]).all()
    # no pickled objects — allow_pickle=False must be sufficient
    assert lib["rms_vis"].dtype.kind == "f"


def test_rejects_empty_library(tmp_path):
    with pytest.raises(Exception):
        save_library([], tmp_path / "empty.npz")
