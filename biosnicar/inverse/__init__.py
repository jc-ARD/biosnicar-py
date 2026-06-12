"""Inverse modelling for BioSNICAR: emulator-based spectral retrieval.

Provides optimisation routines to retrieve ice physical properties from
observed spectral or satellite-band albedo using an :class:`Emulator`.

Quick start
-----------
>>> from biosnicar.emulator import Emulator
>>> from biosnicar.inverse import retrieve
>>> emu = Emulator.load("glacier_ice.npz")
>>> result = retrieve(observed=obs, parameters=["rds", "rho"], emulator=emu)
>>> print(result.summary())
"""

from biosnicar.emulator import Emulator  # re-export for convenience
from biosnicar.inverse.optimize import retrieve, DEFAULT_BOUNDS
from biosnicar.inverse.result import RetrievalResult

__all__ = [
    "Emulator",
    "retrieve",
    "RetrievalResult",
    "DEFAULT_BOUNDS",
]
