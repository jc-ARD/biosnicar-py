"""Built-in plotting functions for BioSNICAR.

All functions return ``(fig, axes)`` so users can further customise.
Matplotlib is imported lazily — this module has no effect on import
time and does not add a hard dependency.

Usage
-----
These functions are available as methods on the core result objects::

    outputs = run_model(solzen=50, rds=1000)
    outputs.plot()
    outputs.plot(platform="sentinel2")
    outputs.plot_subsurface()

    result = retrieve(observed=obs, parameters=[...], emulator=emu)
    result.plot()
    result.plot(true_values={"ssa": 0.67, "glacier_algae": 20000})

    sweep_df = parameter_sweep(params={"rds": [500, 1000, 2000]})
    sweep_df.plot_sensitivity()

They can also be called directly::

    from biosnicar.plotting import plot_albedo, plot_retrieval
    fig, ax = plot_albedo(outputs)
"""

import numpy as np

# Wavelength grid (must match _core.py)
_WVL = np.arange(0.205, 4.999, 0.01)

# Approximate Sentinel-2 band centres and half-widths (um) for overlays.
# Only used for visual display; convolution uses the full SRF.
_S2_BANDS = {
    "B1": (0.443, 0.010), "B2": (0.490, 0.033), "B3": (0.560, 0.018),
    "B4": (0.665, 0.015), "B5": (0.705, 0.008), "B6": (0.740, 0.008),
    "B7": (0.783, 0.010), "B8": (0.842, 0.058), "B8A": (0.865, 0.010),
    "B9": (0.945, 0.010), "B11": (1.610, 0.045), "B12": (2.190, 0.090),
}

_LANDSAT_BANDS = {
    "B1": (0.443, 0.008), "B2": (0.482, 0.030), "B3": (0.562, 0.028),
    "B4": (0.655, 0.019), "B5": (0.865, 0.021), "B6": (1.609, 0.042),
    "B7": (2.201, 0.094),
}

_MODIS_BANDS = {
    "B1": (0.645, 0.025), "B2": (0.858, 0.018), "B3": (0.469, 0.010),
    "B4": (0.555, 0.010), "B5": (1.240, 0.010), "B6": (1.640, 0.012),
    "B7": (2.130, 0.025),
}

_PLATFORM_BAND_INFO = {
    "sentinel2": _S2_BANDS,
    "landsat8": _LANDSAT_BANDS,
    "modis": _MODIS_BANDS,
}


def _import_mpl():
    """Lazy-import matplotlib and return (plt, mpl)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        return plt, matplotlib
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )


def _finalize(fig, save=None, show=False, dpi=300):
    """Save and/or show a figure, then optionally close it.

    Parameters
    ----------
    fig : matplotlib Figure
    save : str or Path, optional
        If given, save the figure to this path.
    show : bool
        If True, call ``plt.show()`` to display in an interactive window.
    dpi : int
        Resolution for saved figures.
    """
    plt, _ = _import_mpl()
    if save is not None:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    elif save is not None:
        # Close the figure after saving to free memory (unless showing).
        plt.close(fig)


# ── 1. Spectral albedo plot ──────────────────────────────────────────────

def plot_albedo(*outputs_list, labels=None, platform=None, xlim=(0.3, 2.5),
               title=None, colors=None, figsize=(10, 5),
               save=None, show=False, dpi=300):
    """Plot spectral albedo for one or more Outputs objects.

    Parameters
    ----------
    *outputs_list : Outputs
        One or more ``Outputs`` objects to plot.
    labels : list of str, optional
        Legend labels.  Defaults to ``"Run 1"``, ``"Run 2"``, etc.
    platform : str, optional
        If given, overlay convolved band values as horizontal bars.
        Supported: ``"sentinel2"``, ``"landsat8"``, ``"modis"``.
    xlim : tuple
        Wavelength axis limits in um.
    title : str, optional
        Plot title.
    colors : list, optional
        Line colours.
    figsize : tuple
        Figure size.
    save : str or Path, optional
        Save figure to this path.
    show : bool
        If True, display in an interactive window.
    dpi : int
        Resolution for saved figures (default 300).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    plt, _ = _import_mpl()

    if len(outputs_list) == 0:
        raise ValueError("Provide at least one Outputs object.")

    fig, ax = plt.subplots(figsize=figsize)

    if labels is None:
        if len(outputs_list) == 1:
            labels = [None]
        else:
            labels = [f"Run {i+1}" for i in range(len(outputs_list))]

    if colors is None:
        cmap = plt.cm.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(outputs_list))]

    for i, outputs in enumerate(outputs_list):
        albedo = outputs.albedo
        ax.plot(_WVL[:len(albedo)], albedo, color=colors[i],
                linewidth=1.2, label=labels[i])

    # Overlay platform bands if requested.
    if platform is not None and platform in _PLATFORM_BAND_INFO:
        band_info = _PLATFORM_BAND_INFO[platform]
        # Use the last outputs object for convolution.
        out = outputs_list[-1]
        from biosnicar.bands import to_platform as _to_platform
        br = _to_platform(out.albedo, platform, flx_slr=out.flx_slr)

        for j, (bname, (centre, half_w)) in enumerate(band_info.items()):
            val = getattr(br, bname, None)
            if val is None:
                continue
            ax.plot([centre - half_w, centre + half_w], [val, val],
                    color="#ff7f0e", linewidth=3, alpha=0.7,
                    label=platform.capitalize() if j == 0 else None)
            ax.plot(centre, val, "o", color="#ff7f0e", markersize=3)

    ax.set_xlabel("Wavelength ($\\mu$m)")
    ax.set_ylabel("Albedo")
    ax.set_xlim(xlim)
    ax.set_ylim(0, None)
    if title:
        ax.set_title(title)
    if any(l is not None for l in labels) or platform:
        ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _finalize(fig, save=save, show=show, dpi=dpi)
    return fig, ax


# ── 2. Subsurface light field ────────────────────────────────────────────

def plot_subsurface(outputs, irradiance=1000.0, figsize=(12, 5),
                    save=None, show=False, dpi=300):
    """Plot subsurface PAR depth profiles: normalised and absolute.

    Parameters
    ----------
    outputs : Outputs
        Must have subsurface flux data (F_up, F_dwn) populated,
        i.e. from the adding-doubling solver.
    irradiance : float
        Total incoming solar irradiance in W m-2, used to convert
        the model's normalised fluxes to absolute PAR for panel (b).
        Default 1000 W m-2 (approximate clear-sky value).
    figsize : tuple
        Figure size.
    save : str or Path, optional
        Save figure to this path.
    show : bool
        If True, display in an interactive window.
    dpi : int
        Resolution for saved figures (default 300).

    Returns
    -------
    fig, (ax_norm, ax_abs) : matplotlib Figure and Axes
    """
    plt, _ = _import_mpl()

    if outputs.F_up is None:
        raise RuntimeError(
            "Subsurface flux data not available. "
            "Re-run with the adding-doubling solver to populate F_up / F_dwn."
        )

    dz = np.asarray(outputs._dz)
    z_interfaces = np.concatenate(([0.0], np.cumsum(dz)))

    fig, (ax_norm, ax_abs) = plt.subplots(1, 2, figsize=figsize)

    depths = np.linspace(0, z_interfaces[-1], 100)
    par_vals = outputs.par(depths)
    par_surface = outputs.par(0.0)
    depths_cm = depths * 100

    # ── Panel (a): PAR normalised to surface ──
    ax_norm.plot(par_vals / par_surface, depths_cm, color="#1f77b4",
                 linewidth=1.5)
    ax_norm.axvline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_norm.set_xlabel("PAR (normalised to surface)")
    ax_norm.set_ylabel("Depth (cm)")
    ax_norm.invert_yaxis()
    ax_norm.set_title("(a) Normalised PAR")
    ax_norm.spines["top"].set_visible(False)
    ax_norm.spines["right"].set_visible(False)

    # ── Panel (b): Absolute PAR in W m-2 ──
    par_abs = par_vals * irradiance
    ax_abs.plot(par_abs, depths_cm, color="#d62728", linewidth=1.5)
    ax_abs.set_xlabel("PAR (W m$^{-2}$)")
    ax_abs.set_ylabel("Depth (cm)")
    ax_abs.invert_yaxis()
    ax_abs.set_title("(b) Absolute PAR")
    ax_abs.annotate(
        f"Assumes {irradiance:.0f} W m$^{{-2}}$ total irradiance",
        xy=(0.95, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8, color="grey",
    )
    ax_abs.spines["top"].set_visible(False)
    ax_abs.spines["right"].set_visible(False)

    fig.tight_layout()
    _finalize(fig, save=save, show=show, dpi=dpi)
    return fig, (ax_norm, ax_abs)


# ── 3. Retrieval result plot ─────────────────────────────────────────────

def plot_retrieval(result, true_values=None, wvl_range=(0.35, 2.5),
                   figsize=(12, 5), save=None, show=False, dpi=300):
    """Plot inversion results: spectral fit and retrieved parameters.

    Parameters
    ----------
    result : RetrievalResult
        Output from ``retrieve()``.
    true_values : dict, optional
        ``{param_name: true_value}`` for comparison.  If provided,
        true values are shown as horizontal dashed lines.
    wvl_range : tuple
        Wavelength range for the spectral panel.
    figsize : tuple
        Figure size.
    save : str or Path, optional
        Save figure to this path.
    show : bool
        If True, display in an interactive window.
    dpi : int
        Resolution for saved figures (default 300).

    Returns
    -------
    fig, (ax_spec, ax_params) : matplotlib Figure and Axes
    """
    plt, _ = _import_mpl()

    fig, (ax_spec, ax_params) = plt.subplots(1, 2, figsize=figsize,
                                              gridspec_kw={"width_ratios": [1.3, 1]})

    # ── Panel (a): Observed vs fitted spectrum ──
    observed = result.observed
    predicted = result.predicted_albedo
    is_band_mode = len(observed) < 100  # band mode typically has < 30 values

    if is_band_mode:
        x = np.arange(len(observed))
        ax_spec.bar(x - 0.15, observed, 0.3, label="Observed", color="#aec7e8",
                    edgecolor="k", linewidth=0.5)
        ax_spec.bar(x + 0.15, predicted[:len(observed)], 0.3, label="Retrieved",
                    color="#ffbb78", edgecolor="k", linewidth=0.5)
        ax_spec.set_xlabel("Band index")
        ax_spec.set_ylabel("Albedo")
    else:
        wvl = _WVL[:len(observed)]
        mask = (wvl >= wvl_range[0]) & (wvl <= wvl_range[1])
        ax_spec.plot(wvl[mask], observed[mask], color="#bbbbbb", linewidth=0.6,
                     label="Observed")
        ax_spec.plot(wvl[mask], predicted[mask], color="#1f77b4", linewidth=1.2,
                     label="Retrieved fit")

        # Residual subplot inset.
        residual = predicted[mask] - observed[mask]
        ax_inset = ax_spec.inset_axes([0.55, 0.55, 0.42, 0.38])
        ax_inset.plot(wvl[mask], residual, color="#d62728", linewidth=0.5)
        ax_inset.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax_inset.set_xlabel("$\\lambda$ ($\\mu$m)", fontsize=7)
        ax_inset.set_ylabel("Residual", fontsize=7)
        ax_inset.tick_params(labelsize=6)
        ax_inset.set_title("Fit residual", fontsize=7)

        ax_spec.set_xlabel("Wavelength ($\\mu$m)")
        ax_spec.set_ylabel("Albedo")

    ax_spec.set_title("(a) Spectral fit")
    ax_spec.legend(fontsize=8)
    ax_spec.spines["top"].set_visible(False)
    ax_spec.spines["right"].set_visible(False)

    # ── Panel (b): Retrieved parameters with uncertainties ──
    params = list(result.best_fit.keys())
    values = [result.best_fit[p] for p in params]
    uncertainties = [result.uncertainty.get(p, 0) for p in params]

    x = np.arange(len(params))
    colors = plt.cm.get_cmap("tab10")

    # Per-parameter information content.  A bar drawn identically for a
    # measured parameter and for one the retrieval copied from its prior
    # invites the reader to believe both equally, so prior-dominated bars
    # are hatched and labelled.  Absent for non-OE methods, in which case
    # this reduces to the plain bar chart.
    from biosnicar.inverse import information as _info
    akd = getattr(result, "averaging_kernel_diag", None) or {}
    labels = _info.provenance(akd, result.best_fit) if akd else {}
    # "not retrieved" drops the series colour entirely: a coloured bar reads
    # as a result, and this one is the prior drawn back at the reader.
    _STYLE = {
        _info.PRIOR_INFLUENCED: dict(hatch="///", alpha=0.55,
                                     edgecolor="#555555", facecolor=None),
        _info.NOT_RETRIEVED: dict(hatch="xxx", alpha=0.9,
                                  edgecolor="#999999", facecolor="#e0e0e0"),
    }

    for i, (p, v, u) in enumerate(zip(params, values, uncertainties)):
        style = _STYLE.get(labels.get(p), {})
        ax_params.barh(i, v, xerr=u, height=0.5,
                       color=style.get("facecolor") or colors(i),
                       alpha=style.get("alpha", 0.8), hatch=style.get("hatch"),
                       edgecolor=style.get("edgecolor", "white"),
                       linewidth=0.6, capsize=3)
        if true_values and p in true_values:
            ax_params.plot(true_values[p], i, "kx", markersize=10, markeredgewidth=2,
                           label="True" if i == 0 else None)

    ax_params.set_yticks(x)
    if labels:
        ticks = []
        for p in params:
            lab = labels.get(p)
            a = akd.get(p)
            if lab in _STYLE and a is not None:
                ticks.append(f"{p}\n({lab}, A={float(a):.2f})")
            else:
                ticks.append(p)
        ax_params.set_yticklabels(ticks, fontsize=7)
    else:
        ax_params.set_yticklabels(params)
    ax_params.set_xlabel("Parameter value")
    ax_params.set_title("(b) Retrieved parameters")
    ax_params.spines["top"].set_visible(False)
    ax_params.spines["right"].set_visible(False)

    handles = []
    if true_values:
        handles, _ = ax_params.get_legend_handles_labels()
    import matplotlib.patches as _mpatches
    for lab in (_info.PRIOR_INFLUENCED, _info.NOT_RETRIEVED):
        if any(v == lab for v in labels.values()):
            st = _STYLE[lab]
            handles.append(_mpatches.Patch(
                facecolor=st["facecolor"] or "grey", alpha=st["alpha"],
                hatch=st["hatch"], edgecolor=st["edgecolor"], label=lab))
    if handles:
        ax_params.legend(handles=handles, fontsize=7)

    fig.tight_layout()
    _finalize(fig, save=save, show=show, dpi=dpi)
    return fig, (ax_spec, ax_params)


# ── 4. Sweep sensitivity plot ────────────────────────────────────────────

def plot_sensitivity(sweep_df, y="BBA", figsize=None, save=None, show=False,
                     dpi=300):
    """Plot parameter sensitivity from a SweepResult DataFrame.

    Automatically detects the swept parameters (columns that vary) and
    chooses a line plot (1 swept param), heatmap (2 swept params), or
    multi-line plot (2 params, using the second as hue).

    Parameters
    ----------
    sweep_df : SweepResult or DataFrame
        Output from ``parameter_sweep()``.
    y : str
        Column to plot on the y-axis (default ``"BBA"``).
    figsize : tuple, optional
        Figure size.  Auto-selected if None.
    save : str or Path, optional
        Save figure to this path.
    show : bool
        If True, display in an interactive window.
    dpi : int
        Resolution for saved figures (default 300).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    plt, _ = _import_mpl()

    # Identify swept parameters: columns with more than one unique value,
    # excluding output columns.
    output_cols = {"BBA", "BBAVIS", "BBANIR", "abs_slr_tot", "abs_vis_tot",
                   "abs_nir_tot", y}
    param_cols = [c for c in sweep_df.columns
                  if c not in output_cols and sweep_df[c].nunique() > 1]

    if len(param_cols) == 0:
        raise ValueError("No swept parameters found in DataFrame.")

    if len(param_cols) == 1:
        # ── 1D: line plot ──
        p = param_cols[0]
        if figsize is None:
            figsize = (8, 5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(sweep_df[p], sweep_df[y], "o-", color="#1f77b4",
                markersize=4, linewidth=1.2)
        ax.set_xlabel(p)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {p}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    elif len(param_cols) == 2:
        p1, p2 = param_cols[0], param_cols[1]
        u1 = sorted(sweep_df[p1].unique())
        u2 = sorted(sweep_df[p2].unique())

        if len(u1) >= 4 and len(u2) >= 4:
            # ── 2D: heatmap ──
            if figsize is None:
                figsize = (8, 6)
            fig, ax = plt.subplots(figsize=figsize)
            pivot = sweep_df.pivot_table(values=y, index=p2, columns=p1)
            im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values,
                               cmap="viridis", shading="auto")
            ax.set_xlabel(p1)
            ax.set_ylabel(p2)
            ax.set_title(y)
            fig.colorbar(im, ax=ax, label=y)
        else:
            # ── 2D: multi-line (use p2 as hue) ──
            if figsize is None:
                figsize = (8, 5)
            fig, ax = plt.subplots(figsize=figsize)
            cmap = plt.cm.get_cmap("viridis", len(u2))
            for i, v2 in enumerate(u2):
                sub = sweep_df[sweep_df[p2] == v2].sort_values(p1)
                ax.plot(sub[p1], sub[y], "o-", color=cmap(i), markersize=4,
                        linewidth=1.2, label=f"{p2}={v2}")
            ax.set_xlabel(p1)
            ax.set_ylabel(y)
            ax.set_title(f"{y} vs {p1}")
            ax.legend(fontsize=8, title=p2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    else:
        # ── 3+D: plot first param, group by second, ignore rest ──
        p1, p2 = param_cols[0], param_cols[1]
        if figsize is None:
            figsize = (8, 5)
        fig, ax = plt.subplots(figsize=figsize)
        u2 = sorted(sweep_df[p2].unique())
        cmap = plt.cm.get_cmap("viridis", len(u2))
        for i, v2 in enumerate(u2):
            sub = sweep_df[sweep_df[p2] == v2].sort_values(p1)
            ax.plot(sub[p1], sub[y], "o-", color=cmap(i), markersize=3,
                    linewidth=1, alpha=0.7, label=f"{p2}={v2}")
        ax.set_xlabel(p1)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {p1} (grouped by {p2})")
        ax.legend(fontsize=7, title=p2, ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _finalize(fig, save=save, show=show, dpi=dpi)
    return fig, ax
