from pathlib import Path

__version__ = "2.2.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
