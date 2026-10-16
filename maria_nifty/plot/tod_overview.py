"""Overview plots that combine TOD traces, residuals, and spectra."""

import os

import matplotlib.pyplot as plt
import nifty.re as jft
import numpy as np
import scipy as sp


def _as_component_mean(model_fn, samples):
    """Evaluate a component model on samples and return the mean value."""
    if model_fn is None:
        return None
    if isinstance(samples, jft.Samples):
        return jft.mean(tuple(model_fn(s) for s in samples))
    if isinstance(samples, jft.Vector):
        return model_fn(samples)
    return model_fn(samples)


def _to_2d(array):
    """Convert 1D arrays to a single-row 2D representation."""
    arr = np.asarray(array)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        return arr.reshape(-1, arr.shape[-1])
    return arr


def _extract_components(tod_data, model, samples, tods=None):
    """Extract atmosphere, map, residual, and time-axis data for plotting."""
    tod_data = _to_2d(tod_data)

    full_rec = _to_2d(_as_component_mean(model, samples))
    atm_rec = _as_component_mean(getattr(model, "atm_model", None), samples)
    map_rec = _as_component_mean(getattr(model, "map_response", None), samples)

    atm_rec = np.zeros_like(tod_data) if atm_rec is None else _to_2d(atm_rec)

    map_rec = np.zeros_like(tod_data) if map_rec is None else _to_2d(map_rec)

    residual = full_rec - tod_data

    n_t = tod_data.shape[-1]
    if tods is not None and hasattr(tods, "time"):
        time_axis = np.asarray(tods.time[:n_t])
    else:
        time_axis = np.arange(n_t, dtype=float)

    return atm_rec, map_rec, residual, time_axis


def _power_spectrum_binned(comp, fs, n_freq_bins=256):
    """Compute a log-binned mean periodogram for the overview plots."""
    # Match spectrum.py power_spectrum_plot: Tukey window + log-binned mean PSD.
    f, ps = sp.signal.periodogram(comp, fs=fs, axis=-1, window="tukey")
    ps_mean = np.mean(ps, axis=0)

    if f.size < 2:
        return np.array([]), np.array([])

    f_bins = np.geomspace(f[1], f[-1], n_freq_bins)
    f_mids = np.sqrt(f_bins[1:] * f_bins[:-1])
    binned_ps = sp.stats.binned_statistic(f, ps_mean, bins=f_bins, statistic="mean")[0]
    use = np.isfinite(binned_ps) & (binned_ps > 0)
    return f_mids[use], binned_ps[use]


def _power_spectrum_unbinned(comp, fs):
    """Compute an unbinned mean periodogram for the overview plots."""
    f, ps = sp.signal.periodogram(comp, fs=fs, axis=-1, window="tukey")
    ps_mean = np.mean(ps, axis=0)
    use = (f > 0) & np.isfinite(ps_mean) & (ps_mean > 0)
    return f[use], ps_mean[use]


def _plot_panel(
    components,
    labels,
    time_axis,
    title,
    out_file,
    fs=1.0,
    alpha=0.35,
    n_freq_bins=256,
    binned=True,
):
    """Render one overview panel with traces on the left and PSD on the right."""
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0], wspace=0.04)

    left_axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_ps = fig.add_subplot(gs[:, 1])

    cmap = plt.colormaps.get_cmap("cmb")
    # Keep atmosphere cool, but swap map/noise colors.
    comp_colors = [cmap(0.1), cmap(0.9), cmap(0.5)]
    ps_ymins = []
    ps_ymaxs = []

    for i, (comp, label, comp_color) in enumerate(zip(components, labels, comp_colors)):
        ax = left_axes[i]
        if comp.shape[0] > 1:
            row_colors = cmap(np.linspace(0.0, 1.0, comp.shape[0]))
            for row, row_color in zip(comp, row_colors):
                ax.plot(time_axis, row, color=row_color, alpha=alpha, lw=1.0)
        else:
            ax.plot(time_axis, comp[0], color=comp_color, alpha=1.0, lw=1.2)
        ax.set_ylabel(label)
        if i < 2:
            ax.set_xticklabels([])

        if binned:
            f_plot, ps_plot = _power_spectrum_binned(comp, fs=fs, n_freq_bins=n_freq_bins)
        else:
            f_plot, ps_plot = _power_spectrum_unbinned(comp, fs=fs)
        if f_plot.size > 0 and ps_plot.size > 0:
            ax_ps.plot(f_plot, ps_plot, lw=1.6, color=comp_color, label=label)
            ps_ymins.append(float(np.nanmin(ps_plot)))
            ps_ymaxs.append(float(np.nanmax(ps_plot)))

    left_axes[-1].set_xlabel("Timestamp [s]")

    ax_ps.set_xlabel("Frequency [Hz]")
    ax_ps.set_ylabel("Power spectral density")
    ax_ps.set_xscale("log")
    ax_ps.set_yscale("log")
    ax_ps.yaxis.tick_right()
    ax_ps.yaxis.set_label_position("right")
    if ps_ymins and ps_ymaxs:
        y_min = min(ps_ymins)
        y_max = max(ps_ymaxs)
        if np.isfinite(y_min) and np.isfinite(y_max) and y_min > 0 and y_max > y_min:
            ax_ps.set_ylim(y_min, y_max)
    ax_ps.legend(loc="best")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def tod_overview_plot(
    tod_data=None,
    model=None,
    samples=None,
    tods=None,
    odir=".",
    fs=1.0,
    title_prefix=None,
    n_freq_bins=256,
):
    """Plot atmosphere, map, residual, and spectrum overviews for a TOD set."""
    if tod_data is None or model is None:
        raise ValueError("tod_data and model are required")
    if samples is None:
        raise ValueError("samples is required")

    if tods is not None and hasattr(tods, "fs"):
        fs = float(tods.fs)

    atm_rec, map_rec, residual, time_axis = _extract_components(tod_data, model, samples, tods=tods)

    os.makedirs(odir, exist_ok=True)
    prefix = "" if title_prefix is None else f"{title_prefix}: "

    labels = ["atm. rec", "map rec", "residual (model - data)"]
    comps_all = [atm_rec, map_rec, residual]
    _plot_panel(
        comps_all,
        labels,
        time_axis,
        title=f"{prefix}TOD Overview (all TODs, binned)",
        out_file=os.path.join(odir, "tod_overview_all_binned.png"),
        fs=fs,
        alpha=0.35,
        n_freq_bins=n_freq_bins,
        binned=True,
    )

    _plot_panel(
        comps_all,
        labels,
        time_axis,
        title=f"{prefix}TOD Overview (all TODs)",
        out_file=os.path.join(odir, "tod_overview_all.png"),
        fs=fs,
        alpha=0.35,
        n_freq_bins=n_freq_bins,
        binned=False,
    )

    comps_0 = [c[0:1] for c in comps_all]
    _plot_panel(
        comps_0,
        labels,
        time_axis,
        title=f"{prefix}TOD Overview (TOD 0, binned)",
        out_file=os.path.join(odir, "tod_overview_0_binned.png"),
        fs=fs,
        alpha=1.0,
        n_freq_bins=n_freq_bins,
        binned=True,
    )

    _plot_panel(
        comps_0,
        labels,
        time_axis,
        title=f"{prefix}TOD Overview (TOD 0)",
        out_file=os.path.join(odir, "tod_overview_0.png"),
        fs=fs,
        alpha=1.0,
        n_freq_bins=n_freq_bins,
        binned=False,
    )
