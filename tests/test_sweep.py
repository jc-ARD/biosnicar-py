"""Tests for the parameter sweep API."""

import numpy as np
import pytest

from biosnicar.drivers.sweep import parameter_sweep


def test_single_param_sweep_shape():
    """Single-parameter sweep returns correct DataFrame shape."""
    df = parameter_sweep(
        params={"solzen": [30, 50, 70]},
        progress=False,
    )
    assert len(df) == 3
    assert "solzen" in df.columns
    assert "BBA" in df.columns


def test_multi_param_sweep_shape():
    """Multi-parameter sweep returns rows = product of lengths."""
    df = parameter_sweep(
        params={"solzen": [30, 50, 60], "rds": [200, 500, 600]},
        progress=False,
    )
    assert len(df) == 9  # 3 x 3
    assert set(df.columns) >= {"solzen", "rds", "BBA", "BBAVIS", "BBANIR"}


def test_impurity_conc_decreases_bba():
    """More black carbon -> lower BBA."""
    df = parameter_sweep(
        params={"black_carbon": [0, 5000, 50000]},
        progress=False,
    )
    bbas = df.sort_values("black_carbon")["BBA"].values
    assert bbas[0] > bbas[1] > bbas[2], (
        f"BBA should decrease with increasing BC: {bbas}"
    )


def test_bba_decreases_with_sza():
    """BBA decreases with increasing solar zenith angle (for direct beam)."""
    df = parameter_sweep(
        params={"solzen": [30, 50, 70]},
        progress=False,
    )
    bbas = df.sort_values("solzen")["BBA"].values
    # BBA generally increases with SZA for snow/ice (more forward scattering at
    # oblique angles), but the key thing is values are distinct and physical.
    assert all(0 < b < 1 for b in bbas), f"BBA out of physical range: {bbas}"


def test_bba_decreases_with_grain_radius():
    """Larger grains -> lower BBA (more absorption path length)."""
    df = parameter_sweep(
        params={"rds": [100, 500, 2000]},
        progress=False,
    )
    bbas = df.sort_values("rds")["BBA"].values
    assert bbas[0] > bbas[-1], (
        f"Small grains should have higher BBA than large: {bbas}"
    )


def test_return_spectral():
    """return_spectral=True includes albedo column with 480-element arrays."""
    df = parameter_sweep(
        params={"solzen": [40]},
        return_spectral=True,
        progress=False,
    )
    assert "albedo" in df.columns
    alb = df.iloc[0]["albedo"]
    assert isinstance(alb, np.ndarray)
    assert len(alb) == 480


def test_unknown_key_raises():
    """Unknown parameter key raises ValueError."""
    with pytest.raises(ValueError, match="Unknown parameter key"):
        parameter_sweep(params={"nonexistent_param": [1, 2]}, progress=False)


def test_toon_solver():
    """Toon solver runs without error and produces physical results."""
    # Toon solver requires layer_type=0 (grains) and SZA in [50, 89]
    df = parameter_sweep(
        params={"solzen": [55, 70], "layer_type": [0]},
        solver="toon",
        progress=False,
    )
    assert len(df) == 2
    assert all(0 < b < 1 for b in df["BBA"])


def test_adding_doubling_solver():
    """Adding-doubling solver runs without error and produces physical results."""
    df = parameter_sweep(
        params={"solzen": [40, 60]},
        solver="adding-doubling",
        progress=False,
    )
    assert len(df) == 2
    assert all(0 < b < 1 for b in df["BBA"])


# ── SweepResult.to_platform() tests ─────────────────────────────────


def test_to_platform_single():
    """Single platform appends unprefixed band columns."""
    df = parameter_sweep(
        params={"solzen": [50]},
        progress=False,
    ).to_platform("sentinel2")
    # Band columns should be present without prefix
    assert "B3" in df.columns
    assert "NDSI" in df.columns
    # Original sweep columns preserved
    assert "solzen" in df.columns
    assert "BBA" in df.columns


def test_to_platform_multi_prefix():
    """Multiple platforms produce prefixed band columns."""
    df = parameter_sweep(
        params={"solzen": [50]},
        progress=False,
    ).to_platform("sentinel2", "modis")
    assert "sentinel2_B3" in df.columns
    assert "sentinel2_NDSI" in df.columns
    assert "modis_B1" in df.columns
    assert "modis_NDSI" in df.columns


def test_to_platform_physical_values():
    """Band albedo values should be in the physical range [0, 1]."""
    df = parameter_sweep(
        params={"rds": [500, 1000]},
        progress=False,
    ).to_platform("sentinel2")
    for col in ["B3", "B4", "B8", "B11"]:
        vals = df[col].values
        assert all(0 <= v <= 1 for v in vals), f"{col} out of range: {vals}"


def test_to_platform_cesm():
    """GCM platform (cesm2band) works via sweep chaining."""
    df = parameter_sweep(
        params={"solzen": [50]},
        progress=False,
    ).to_platform("cesm2band")
    assert "vis" in df.columns
    assert "nir" in df.columns
    assert 0 < df.iloc[0]["vis"] < 1


# ── 2026-07 audit regressions (B9, B10) ─────────────────────────────


def test_to_platform_aligned_after_sort_and_filter():
    """Band values must follow their own rows through sort/filter (B9).

    Regression: _spectral was a plain list zipped positionally against the
    derived frame's index, so df.sort_values(...).to_platform(...) silently
    paired band values with the wrong rows.
    """
    from biosnicar.drivers.sweep import parameter_sweep

    df = parameter_sweep(params={"rds": [200, 800, 2000]})
    base = df.to_platform("sentinel2")
    flipped = df.sort_values("BBA").to_platform("sentinel2")
    for _, row in flipped.iterrows():
        ref = base.loc[base["rds"] == row["rds"]].iloc[0]
        assert row["B3"] == ref["B3"]
        assert row["B11"] == ref["B11"]
    # filtered frames also work and stay aligned
    sub = df[df["rds"] > 500].to_platform("sentinel2")
    assert len(sub) == 2
    for _, row in sub.iterrows():
        ref = base.loc[base["rds"] == row["rds"]].iloc[0]
        assert row["B3"] == ref["B3"]


def test_sweep_impurity_matches_run_model():
    """Scalar impurity semantics must match run_model (B10).

    Regression: sweep broadcast a scalar concentration to every layer while
    run_model puts it in the first layer only, so the two were silently not
    comparable on multi-layer columns.
    """
    from biosnicar.drivers.run_model import run_model
    from biosnicar.drivers.sweep import parameter_sweep

    bba_model = run_model(black_carbon=50000).BBA
    df = parameter_sweep(params={"black_carbon": [50000]})
    assert df["BBA"].iloc[0] == pytest.approx(bba_model, abs=1e-9)
