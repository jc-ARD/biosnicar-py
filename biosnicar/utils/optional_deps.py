"""Helpers for dependencies that are only needed by a subset of the model.

`miepython` is the only one at present. It is needed to *generate* single
scattering property files (the bio-optical model and the sphere SSP
generator), but not to run the forward model, which reads the precomputed
files shipped in `data/`. It pulls in numba/llvmlite, which have no wheels
for some platforms (notably Intel macOS), so it lives in the `mie` extra
rather than the core dependency set.
"""


def import_miepython():
    """Import and return the `miepython` module, with an install hint on failure.

    Returns:
        the imported miepython module

    Raises:
        ImportError: if miepython is not installed
    """

    try:
        import miepython
    except ImportError as err:
        raise ImportError(
            "miepython is required to generate single scattering properties, "
            "but is not installed. Install it with `pip install biosnicar[mie]` "
            "(or `pip install miepython`). It is not needed to run the forward "
            "model, which uses the precomputed optical property files in data/."
        ) from err

    return miepython
