"""Fast emulator-based forward model evaluation.

Provides :func:`run_emulator`, a drop-in companion to :func:`run_model` that
uses a pre-trained :class:`~biosnicar.emulator.Emulator` for ~microsecond
spectral albedo predictions.

Example::

    from biosnicar.emulator import Emulator
    from biosnicar.drivers.run_emulator import run_emulator

    emu = Emulator.load("glacier_ice.npz")
    outputs = run_emulator(emu, rds=1000, black_carbon=500)
    print(outputs.BBA)
"""

import numpy as np

from biosnicar.classes.outputs import Outputs


# Standard 480-band model: VIS = 0:50, NIR = 50:480
_VIS_MAX_IDX = 50
_NIR_MAX_IDX = 480


def run_emulator(emulator, **params):
    """Evaluate the emulator and return an Outputs object.

    This mirrors the interface of :func:`~biosnicar.drivers.run_model.run_model`
    but runs in ~microseconds rather than ~50 ms.

    Parameters
    ----------
    emulator : :class:`~biosnicar.emulator.Emulator`
        A trained emulator (from :meth:`Emulator.build` or
        :meth:`Emulator.load`).
    **params
        Keyword arguments for the emulator parameters
        (e.g. ``rds=1000, black_carbon=500``).  Must match the
        parameter names the emulator was built with.

    Returns
    -------
    :class:`~biosnicar.classes.outputs.Outputs`
        Object with ``.albedo``, ``.BBA``, ``.BBAVIS``, ``.BBANIR``,
        ``.flx_slr``, and ``.to_platform()`` populated.  Fields that
        only the full radiative transfer solver produces (e.g.
        ``heat_rt``, ``absorbed_flux_per_layer``) are ``None``.
    """
    albedo = emulator.predict(**params)
    # Flux of the illumination these parameters describe, not of whichever
    # training sample the emulator happened to store.
    if hasattr(emulator, "solar_flux"):
        flx_slr = emulator.solar_flux(params.get("solzen"), params.get("direct"))
    else:
        flx_slr = emulator.flx_slr

    outputs = Outputs()
    outputs.albedo = albedo
    outputs.flx_slr = flx_slr

    # Broadband albedo — same formula as the RT solvers
    outputs.BBA = float(
        np.sum(flx_slr * albedo) / np.sum(flx_slr)
    )
    outputs.BBAVIS = float(
        np.sum(flx_slr[:_VIS_MAX_IDX] * albedo[:_VIS_MAX_IDX])
        / np.sum(flx_slr[:_VIS_MAX_IDX])
    )
    outputs.BBANIR = float(
        np.sum(flx_slr[_VIS_MAX_IDX:_NIR_MAX_IDX] * albedo[_VIS_MAX_IDX:_NIR_MAX_IDX])
        / np.sum(flx_slr[_VIS_MAX_IDX:_NIR_MAX_IDX])
    )

    return outputs
