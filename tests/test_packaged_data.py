"""Regression tests for packaged data (optical properties + emulators).

These guard the packaging invariant broken historically: ``data/`` lived as a
sibling of the package, so a wheel installed into site-packages shipped none of
it and the model failed at runtime with a missing-file error. The data now
lives *inside* the ``biosnicar`` package and is declared as package-data.

Note: an editable install (``pip install -e .``, as CI uses) always finds the
source-tree data, so these tests assert the *location* is inside the package —
which is what makes a built wheel correct too.
"""

from pathlib import Path

import biosnicar
from biosnicar.sea_ice.emulator_configs import (
    SEA_ICE_EMULATOR_CONFIGS,
    trained_emulator_names,
)


def test_data_dir_is_inside_package():
    """DATA_DIR must resolve within the installed biosnicar package."""
    pkg_dir = Path(biosnicar.__file__).resolve().parent
    data_dir = biosnicar.DATA_DIR.resolve()
    assert data_dir == pkg_dir / "data"
    assert pkg_dir in data_dir.parents
    assert data_dir.is_dir()


def test_optical_property_data_present():
    """A representative optical-property file must ship with the package."""
    fsds = biosnicar.DATA_DIR / "OP_data" / "480band" / "fsds.npz"
    assert fsds.is_file(), f"missing packaged optical-property data: {fsds}"


def test_sea_ice_emulators_present_on_disk():
    """Every sea-ice emulator config's .npz must exist at its resolved path.

    This is the exact failure mode that crashed staging inversions: the config
    pointed at a path that did not exist for a pip-installed package.
    """
    missing = []
    for name in trained_emulator_names():
        path = Path(SEA_ICE_EMULATOR_CONFIGS[name]["emulator_file"])
        if not path.is_file():
            missing.append((name, str(path)))
    assert not missing, f"emulator files not found on disk: {missing}"
