"""Power-spectrum comparison plots for TOD datasets."""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from maria_nifty.plot.util import psd_unit_label


def _power_spectrum_binned(comp, fs, n_freq_bins=256):
    """Compute a log-binned mean periodogram for a TOD component."""
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
    """Compute an unbinned mean periodogram for a TOD component."""
    f, ps = sp.signal.periodogram(comp, fs=fs, axis=-1, window="tukey")
    ps_mean = np.mean(ps, axis=0)
    use = (f > 0) & np.isfinite(ps_mean) & (ps_mean > 0)
    return f[use], ps_mean[use]


def tod_spectrum_plot(
    datasets=None,
    odir=None,
    fs=1.0,
    n_freq_bins=256,
    labels=None,
    grid=True,
    raw_fname="tod_spectra.png",
    write_binned=True,
    binned_fname="tod_spectra_binned.png",
):
    """Plot mean TOD power spectra for one or more datasets.

    ``labels``, if given, is a per-dataset list of legend strings (e.g. from
    :func:`maria_nifty.fits_label`) used instead of the default "Dataset {i}".
    ``grid`` toggles the dotted major gridlines on both figures. The unbinned
    figure is always written, as ``raw_fname``; the log-binned figure is only
    computed/written (as ``binned_fname``) when ``write_binned`` is True.
    """
    if datasets is None or len(datasets) == 0:
        raise ValueError("datasets must be a non-empty list")
    if labels is not None and len(labels) != len(datasets):
        raise ValueError(f"labels must match datasets in length, got {len(labels)} vs {len(datasets)}")

    if write_binned:
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
    fig_raw, ax_raw = plt.subplots(figsize=(7.2, 7.2))
    cmap = plt.colormaps.get_cmap("cmb")
    n_datasets = len(datasets)

    all_freq = []
    all_power = []
    all_freq_raw = []
    all_power_raw = []
    tod_units = next(
        (getattr(dct["tods"], "units", None) for dct in datasets if dct.get("tods") is not None),
        None,
    )
    unit_label = f" {psd_unit_label(tod_units)}" if tod_units else ""

    for i, dct in enumerate(datasets):
        if "tod_data" not in dct:
            raise KeyError(f"datasets[{i}] is missing key 'tod_data'")

        ds_fs = fs
        if "tods" in dct and dct["tods"] is not None and hasattr(dct["tods"], "fs"):
            ds_fs = float(dct["tods"].fs)

        tod_data = np.asarray(dct["tod_data"])
        if tod_data.ndim == 1:
            tod_data = tod_data[None, :]
        if tod_data.ndim != 2:
            raise ValueError(f"datasets[{i}]['tod_data'] must be 1D or 2D, got shape {tod_data.shape}")

        color = cmap(i / max(1, n_datasets - 1))
        label = labels[i] if labels is not None else f"Dataset {i}"

        if write_binned:
            f_plot, ps_plot = _power_spectrum_binned(tod_data, fs=ds_fs, n_freq_bins=n_freq_bins)
            if f_plot.size > 0 and ps_plot.size > 0:
                all_freq.append(f_plot)
                all_power.append(ps_plot)
                ax.plot(f_plot, ps_plot, lw=1.7, alpha=0.8, color=color, label=label)

        f_raw, ps_raw = _power_spectrum_unbinned(tod_data, fs=ds_fs)
        if f_raw.size > 0 and ps_raw.size > 0:
            all_freq_raw.append(f_raw)
            all_power_raw.append(ps_raw)
            ax_raw.plot(f_raw, ps_raw, lw=1.4, alpha=0.8, color=color, label=label)

    if write_binned:
        if len(all_freq) == 0:
            raise ValueError("No positive finite spectrum values found to plot")

        freq_min = min(np.min(freq) for freq in all_freq)
        freq_max = max(np.max(freq) for freq in all_freq)
        power_min = min(np.min(power) for power in all_power)
        power_max = max(np.max(power) for power in all_power)

        ax.set_xlabel("Frequency [Hz]", fontsize=14)
        ax.set_ylabel(f"Power spectral density{unit_label}", fontsize=14)
        ax.loglog()

        ax.set_xlim(freq_min, freq_max)
        ax.set_ylim(power_min, power_max)
        ax.tick_params(axis="both", which="both", labelsize=12)
        ax.legend(fontsize=10, loc="best")
        if grid:
            ax.grid(True, which="major", axis="both", linestyle=":", color="0.55", alpha=0.5)
        else:
            ax.grid(False)

    if len(all_freq_raw) == 0:
        raise ValueError("No positive finite unbinned spectrum values found to plot")

    freq_min_raw = min(np.min(freq) for freq in all_freq_raw)
    freq_max_raw = max(np.max(freq) for freq in all_freq_raw)
    power_min_raw = min(np.min(power) for power in all_power_raw)
    power_max_raw = max(np.max(power) for power in all_power_raw)

    ax_raw.set_xlabel("Frequency [Hz]", fontsize=14)
    ax_raw.set_ylabel(f"Power spectral density{unit_label}", fontsize=14)
    ax_raw.loglog()
    ax_raw.set_xlim(freq_min_raw, freq_max_raw)
    ax_raw.set_ylim(power_min_raw, power_max_raw)
    ax_raw.tick_params(axis="both", which="both", labelsize=12)
    ax_raw.legend(fontsize=10, loc="best")
    if grid:
        ax_raw.grid(True, which="major", axis="both", linestyle=":", color="0.55", alpha=0.5)
    else:
        ax_raw.grid(False)

    if write_binned:
        fig.tight_layout()
    fig_raw.tight_layout()

    if odir:
        import os

        os.makedirs(odir, exist_ok=True)
        ps_file_raw = f"{odir}/{raw_fname}"
        fig_raw.savefig(ps_file_raw, dpi=150, bbox_inches="tight")
        plt.close(fig_raw)
        if write_binned:
            ps_file = f"{odir}/{binned_fname}"
            fig.savefig(ps_file, dpi=150, bbox_inches="tight")
            plt.close(fig)
    else:
        plt.show()
        plt.close(fig_raw)
        if write_binned:
            plt.show()
            plt.close(fig)
