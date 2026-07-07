from pathlib import Path

__version__ = "2.2.0"

# Data (optical properties, band SRFs, pigments, pre-built emulators) ships
# *inside* the package as package-data, so these resolve correctly whether
# biosnicar is run from a source checkout, an editable install, or a wheel
# installed into site-packages.  Prior to this, ``data/`` was a sibling of the
# package and paths were derived from ``parent.parent``, which pointed at
# ``site-packages/data`` (nonexistent) for a normal ``pip install``.
#
# ``PROJECT_ROOT`` now means "the biosnicar package root" — the directory that
# contains ``data/``.  Consumers that join ``PROJECT_ROOT / "data" / ...`` or
# use ``DATA_DIR`` keep working unchanged.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def run_model(*args, **kwargs):
    """Run the BioSNICAR forward model (terrestrial or sea ice).

    See :func:`biosnicar.drivers.run_model.run_model` for full docs.

    Quick examples::

        from biosnicar import run_model

        # Glacier ice
        outputs = run_model(solzen=50, layer_type=1, rds=500, rho=700)

        # Sea ice via preset
        outputs = run_model(preset="FYI_WINTER_BARE", solzen=60)

        # Snow on sea ice, fully explicit
        outputs = run_model(
            solzen=60,
            layer_type=[0, 4, 4],
            dz=[0.15, 0.05, 1.45],
            rds=[200, None, None],
            rho=[300, 895, 895],
            sea_ice_salinity=[None, 12, 8],
            sea_ice_temperature=[None, -25, -20],
            sea_ice_bubble_radius=[None, 100, 200],
        )
    """
    from biosnicar.drivers.run_model import run_model as _run_model

    return _run_model(*args, **kwargs)


def run_emulator(*args, **kwargs):
    """Evaluate a trained emulator. See :func:`biosnicar.drivers.run_emulator.run_emulator`."""
    from biosnicar.drivers.run_emulator import run_emulator as _run_emulator

    return _run_emulator(*args, **kwargs)


def to_platform(*args, **kwargs):
    """Convolve spectral albedo onto platform bands. See :func:`biosnicar.bands.to_platform`."""
    from biosnicar.bands import to_platform as _to_platform

    return _to_platform(*args, **kwargs)


# Sea ice preset dicts — importable from the top level for convenience
def __getattr__(name):
    _PRESETS = ("FYI_WINTER_BARE", "FYI_WINTER_SNOW", "MYI_WINTER_BARE")
    if name in _PRESETS:
        from biosnicar.sea_ice import presets as _p
        return getattr(_p, name)
    raise AttributeError(f"module 'biosnicar' has no attribute {name!r}")
