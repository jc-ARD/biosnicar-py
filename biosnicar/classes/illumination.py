import math
import numpy as np
import biosnicar
from biosnicar.utils.load_inputs import load_inputs

# Module-level cache: loaded once on first use
_fsds_data = None


def _get_fsds_data():
    global _fsds_data
    if _fsds_data is None:
        npz_path = str(biosnicar.DATA_DIR / "OP_data" / "480band" / "fsds.npz")
        _fsds_data = np.load(npz_path)
    return _fsds_data


# Irradiance-file stubs, indexed by the ``incoming`` atmosphere code (0-6).
# Mirrors PATHS.ILLUMINATION_FILE_STUBS in inputs.yaml.
DEFAULT_ILLUMINATION_STUBS = (
    "swnb_480bnd_mlw", "swnb_480bnd_mls", "swnb_480bnd_saw",
    "swnb_480bnd_sas", "swnb_480bnd_smm", "swnb_480bnd_hmn",
    "swnb_480bnd_trp",
)


def solar_flux(solzen, direct, incoming=0, stubs=None):
    """Spectral downwelling solar flux (480 bands) for an illumination state.

    This is the flux the forward model itself uses, exposed so that anything
    weighting a spectrum by sunlight — broadband albedo, satellite band
    convolution — can use the flux of the illumination it is actually
    describing.

    Parameters
    ----------
    solzen : float
        Solar zenith angle in degrees.  Rounded to the nearest integer: the
        archive is tabulated per degree, 0-89.  Ignored for diffuse sky.
    direct : int
        1 for clear-sky direct beam, 0 for overcast/diffuse.
    incoming : int
        Atmosphere profile, 0-6 (see ``run_model``).
    stubs : sequence of str, optional
        Irradiance-file stubs indexed by *incoming*.
    """
    if incoming < 0 or incoming > 6:
        raise ValueError("Irradiance type out of range - between 0 and 6 only")
    stubs = DEFAULT_ILLUMINATION_STUBS if stubs is None else stubs
    if direct:
        # Zero-padded integer degrees: a float solzen (50.0) must not leak
        # into the key ("SZA50.0" -> KeyError).
        key = stubs[incoming] + "_clr_" + f"SZA{int(np.rint(solzen)):02d}"
    else:
        key = stubs[incoming] + "_cld"
    flx_slr = _get_fsds_data()[key].copy()
    flx_slr[flx_slr <= 0] = 1e-30
    return flx_slr


class Illumination:
    """Properties of incoming irradiance.

    Instances of Illumination contain all data relating to the incoming irradiance.

    Attributes:
        direct: Boolean toggling between direct and diffuse irradiance
        diffuse_fraction: diffuse part of a clear sky's irradiance — None
            (pure direct beam), "clear_sky" (modelled, see biosnicar.sky),
            a number or a 480-band array.  Ignored when ``direct`` is 0.
        solzen: solar zenith angle in degrees from the vertical
        incoming: choice of spectral distribution from file 0-6
        stubs: array of stub strings for selecting irradiance files
        nbr_wvl: number fo wavelengths (default 480)
    """

    def __init__(self, input_file):
        inputs = load_inputs(input_file)

        self.direct = inputs["RTM"]["DIRECT"]
        self.diffuse_fraction = inputs["RTM"].get("DIFFUSE_FRACTION")
        self.solzen = inputs["RTM"]["SOLZEN"]
        self.incoming = inputs["RTM"]["INCOMING"]
        self.stubs = inputs["PATHS"]["ILLUMINATION_FILE_STUBS"]
        self.nbr_wvl = inputs["RTM"]["NBR_WVL"]

        self.calculate_irradiance()

    def calculate_irradiance(self):
        """Calculates irradiance from initialized attributes.

        Takes mu_not, incoming and file stubs from self and calculates irradiance.

        Args:
            self

        Returns:
            flx_slr: incoming flux from file
            Fd: diffuse irradiance
            Fs: direct irradiance


        Raises:
            ValueError is incoming is out of range
        """

        if self.incoming < 0 or self.incoming > 6:
            raise ValueError("Irradiance type out of range - between 0 and 6 only")

        # update mu_not from solzen
        self.mu_not = np.cos(math.radians(np.rint(self.solzen)))

        flx_slr = solar_flux(self.solzen, self.direct, self.incoming,
                             stubs=self.stubs)
        self.flx_slr = flx_slr
        out = flx_slr / (self.mu_not * np.pi)

        if self.direct:
            from biosnicar.sky import resolve_diffuse_fraction

            frac = resolve_diffuse_fraction(
                getattr(self, "diffuse_fraction", None), self.solzen)
            if frac is None:
                self.Fs = out
                self.Fd = np.zeros(self.nbr_wvl)
            else:
                # A clear sky with its scattered part.  The solvers take the
                # incident flux as  mu0*pi*Fs  (beam) +  Fd  (diffuse), so
                # this splits flx_slr between the two and leaves the total
                # where it was.
                frac = np.broadcast_to(frac, (self.nbr_wvl,))
                self.Fs = (1.0 - frac) * out
                self.Fd = frac * flx_slr
        else:
            self.Fd = out
            self.Fs = np.zeros(self.nbr_wvl)
        return
