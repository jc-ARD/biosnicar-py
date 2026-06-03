import numpy as np


class Outputs:
    """output data from radiative transfer calculations.

    Attributes:
        heat_rt: heating rate in each layer
        BBAVIS: broadband albedo in visible range
        BBANIR: broadband albedo in NIR range
        BBA: broadband albedo across solar spectrum
        abs_slr_btm: absorbed solar energy at bottom surface
        abs_vis_btm: absorbed visible energy at bottom surface
        abs_nir_btm: absorbed NIR energy at bottom surface
        albedo: albedo of ice column
        total_insolation: energy arriving from atmosphere
        abs_slr_tot: total absorbed energy across solar spectrum
        abs_vis_tot: total absorbed energy across visible spectrum
        abs_nir_tot: total absorbed energy across NIR spectrum
        absorbed_flux_per_layer: total absorbed flux per layer
        F_up: spectral upwelling flux at layer interfaces [W/m²],
            shape (nbr_wvl, nbr_lyr+1)
        F_dwn: spectral downwelling flux at layer interfaces [W/m²],
            shape (nbr_wvl, nbr_lyr+1)
    """

    def __init__(self):
        self.heat_rt = None
        self.BBAVIS = None
        self.BBANIR = None
        self.BBA = None
        self.abs_slr_btm = None
        self.abs_vis_btm = None
        self.abs_nir_btm = None
        self.albedo = None
        self.total_insolation = None
        self.abs_slr_tot = None
        self.abs_vis_tot = None
        self.abs_nir_tot = None
        self.absorbed_flux_per_layer = None
        self.flx_slr = None
        self.F_up = None
        self.F_dwn = None
        self._dz = None
        self._wavelengths = None
        self._L_snw = None

    # ------------------------------------------------------------------
    # Aliases so sea-ice and terrestrial code use the same attribute names.
    # These are read-only properties; the canonical names (BBA, albedo, …)
    # remain the authoritative storage.
    # ------------------------------------------------------------------

    @property
    def spectrum(self):
        """Spectral albedo array (480 bands, 0.205–4.995 µm). Alias for .albedo."""
        return self.albedo

    @property
    def broadband(self):
        """Flux-weighted broadband albedo. Alias for .BBA."""
        return self.BBA

    @property
    def visible(self):
        """Flux-weighted visible broadband albedo. Alias for .BBAVIS."""
        return self.BBAVIS

    @property
    def nir(self):
        """Flux-weighted NIR broadband albedo. Alias for .BBANIR."""
        return self.BBANIR

    @property
    def wavelengths(self):
        """Wavelength grid in µm (0.205–4.995, step 0.01). Shape (480,)."""
        return np.arange(0.205, 4.999, 0.01)

    def to_platform(self, platform):
        """Convolve spectral albedo onto platform bands.

        Args:
            platform: Platform key (e.g. ``"sentinel2"``, ``"cesm2band"``).

        Returns:
            :class:`~biosnicar.bands.BandResult`
        """
        from biosnicar.bands import to_platform as _to_platform

        return _to_platform(self.albedo, platform, flx_slr=self.flx_slr)

    def subsurface_flux(self, depth_m):
        """Spectral fluxes at an arbitrary depth by linear interpolation.

        Args:
            depth_m: depth below surface in metres (scalar or 1-D array).

        Returns:
            dict with keys ``'F_up'``, ``'F_dwn'``, ``'F_net'``.
            Each is shape ``(nbr_wvl,)`` for scalar depth, or
            ``(len(depth_m), nbr_wvl)`` for array depth.

        Raises:
            RuntimeError: if subsurface flux data is not available.
        """
        if self.F_up is None:
            raise RuntimeError(
                "Subsurface flux data not available. "
                "Re-run the model to populate F_up / F_dwn."
            )

        z_interfaces = np.concatenate(([0.0], np.cumsum(self._dz)))
        z_max = z_interfaces[-1]

        scalar_input = np.ndim(depth_m) == 0
        depths = np.atleast_1d(np.asarray(depth_m, dtype=float))

        F_up_out = np.empty((len(depths), self.F_up.shape[0]))
        F_dwn_out = np.empty_like(F_up_out)

        for k, d in enumerate(depths):
            d_clipped = np.clip(d, 0.0, z_max)
            # find bracketing interface index
            idx = np.searchsorted(z_interfaces, d_clipped, side="right") - 1
            idx = min(idx, len(z_interfaces) - 2)
            dz_local = z_interfaces[idx + 1] - z_interfaces[idx]
            if dz_local == 0:
                frac = 0.0
            else:
                frac = (d_clipped - z_interfaces[idx]) / dz_local
            F_up_out[k, :] = (1 - frac) * self.F_up[:, idx] + frac * self.F_up[:, idx + 1]
            F_dwn_out[k, :] = (1 - frac) * self.F_dwn[:, idx] + frac * self.F_dwn[:, idx + 1]

        if scalar_input:
            F_up_out = F_up_out[0]
            F_dwn_out = F_dwn_out[0]

        return {
            "F_up": F_up_out,
            "F_dwn": F_dwn_out,
            "F_net": F_dwn_out - F_up_out,
        }

    def par(self, depth_m=0.0):
        """Photosynthetically Active Radiation (400-700 nm) at a given depth.

        Returns the downwelling planar irradiance summed over the PAR
        band (400--700 nm).

        **Flux enhancement near the surface:** In a highly scattering,
        low-absorption medium (ice in the visible has ssa ~0.99998),
        backscattered light from below adds to the downwelling stream,
        so PAR just below the surface can *exceed* the incoming value.
        This is real physics (radiation trapping), not a numerical
        artefact.  The enhancement is strongest in the top ~1 transport
        mean-free-path and diminishes with depth.

        Units follow the model convention: fluxes are normalised so that
        total incoming irradiance sums to 1 across all bands.  Multiply
        by actual total incoming irradiance (W/m²) to obtain absolute PAR.

        Args:
            depth_m: depth in metres (scalar or array), default 0 (surface).

        Returns:
            PAR (scalar for scalar depth, array for array depth).
        """
        if self.F_up is None:
            raise RuntimeError(
                "Subsurface flux data not available. "
                "Re-run the model to populate F_up / F_dwn."
            )

        wvl = self._wavelengths
        par_mask = (wvl >= 0.4) & (wvl <= 0.7)

        flux = self.subsurface_flux(depth_m)
        F_dwn = flux["F_dwn"]

        scalar_input = np.ndim(depth_m) == 0

        if scalar_input:
            F_dwn = F_dwn[np.newaxis, :]

        par_values = np.sum(F_dwn[:, par_mask], axis=1)

        if scalar_input:
            return float(par_values[0])
        return par_values

    def spectral_heating_rate(self):
        """Spectral radiative heating rate per layer [K/hr per band].

        Returns:
            array shape ``(nbr_wvl, nbr_lyr)`` — spectral heating rate.

        Raises:
            RuntimeError: if subsurface flux data is not available.
        """
        if self.F_up is None:
            raise RuntimeError(
                "Subsurface flux data not available. "
                "Re-run the model to populate F_up / F_dwn."
            )

        F_net = self.F_up - self.F_dwn
        F_abs = F_net[:, 1:] - F_net[:, :-1]
        L_snw = np.asarray(self._L_snw)
        # K/s per band, then convert to K/hr
        return (F_abs / (L_snw[np.newaxis, :] * 2117)) * 3600

    def plot(self, *others, labels=None, platform=None,
             save=None, show=False, **kwargs):
        """Plot spectral albedo. See :func:`biosnicar.plotting.plot_albedo`.

        Parameters
        ----------
        *others : Outputs
            Additional Outputs objects to overlay.
        labels : list of str, optional
            Legend labels.
        platform : str, optional
            Overlay platform band values (e.g. ``"sentinel2"``).
        save : str or Path, optional
            Save figure to this path.
        show : bool
            If True, display in an interactive window.
        **kwargs
            Passed to :func:`~biosnicar.plotting.plot_albedo`.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        from biosnicar.plotting import plot_albedo
        return plot_albedo(self, *others, labels=labels, platform=platform,
                           save=save, show=show, **kwargs)

    def plot_subsurface(self, save=None, show=False, **kwargs):
        """Plot subsurface PAR depth profiles: normalised and absolute.

        See :func:`biosnicar.plotting.plot_subsurface`.

        Parameters
        ----------
        save : str or Path, optional
            Save figure to this path.
        show : bool
            If True, display in an interactive window.
        **kwargs
            Passed to :func:`~biosnicar.plotting.plot_subsurface`.

        Returns
        -------
        fig, (ax_norm, ax_abs) : matplotlib Figure and Axes
        """
        from biosnicar.plotting import plot_subsurface
        return plot_subsurface(self, save=save, show=show, **kwargs)
