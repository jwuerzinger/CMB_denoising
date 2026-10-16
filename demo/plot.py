"""Figures for the map-reconstruction demo (see ``demo.py``).

All figures are drawn from the ``results.npz`` that ``demo.py`` writes:

    python demo/plot.py demo/runs/<run>/files/results.npz [output folder]

The output folder defaults to ``plots/`` next to ``files/``. Maps are shown in
mK_RJ against the sky offset from the map center in arcmin, TODs against time
in minutes.
"""

import logging
import os
import sys

import maria.map  # noqa: F401 -- registers the "cmb" colormap
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

# The math-text fallback fonts (STIX, Computer Modern) are not installed; matplotlib
# falls back to DejaVu Sans, so the "findfont" warnings are harmless.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

RAD2ARCMIN = 180 * 60 / np.pi
CMAP = "cmb"
LABEL_FS = 15.0
TICK_FS = 12.5
SIGNAL_LABEL = r"Signal [mK$_\mathrm{RJ}$]"
K_RJ = r"[K$_\mathrm{RJ}$]"


# =============================================================================
# Helpers
# =============================================================================


def map_extent(fov):
    """``imshow`` extent in arcmin of a map centered on zero offset."""
    hx, hy = 0.5 * np.asarray(fov) * RAD2ARCMIN
    return (-hx, hx, -hy, hy)


def symmetric_limits(*maps):
    """Symmetric color limits in mK covering all ``maps`` (in K)."""
    vmax = 1e3 * max(float(np.abs(m).max()) for m in maps)
    return -vmax, vmax


def show_map(ax, sky, extent, vlim, ylabel=True):
    """Draw a map (axis 0 = x, axis 1 = y, in K) in mK on offset axes."""
    img = ax.imshow(1e3 * sky.T, cmap=CMAP, origin="lower", extent=extent, vmin=vlim[0], vmax=vlim[1])
    ax.set_xlabel(r"$\Delta x$ [arcmin]")
    if ylabel:
        ax.set_ylabel(r"$\Delta y$ [arcmin]")
    return img


def detector_colors(n):
    """One color per detector, spread over the colormap."""
    cmap = plt.colormaps.get_cmap(CMAP)
    norm = mcolors.Normalize(vmin=0, vmax=max(n - 1, 1))
    return [cmap(norm(i)) for i in range(n)]


def plot_tods(ax, time_min, tods, scale=1.0):
    """One line per detector, colored as in the detector layout."""
    for tod, color in zip(tods, detector_colors(len(tods))):
        ax.plot(time_min, scale * tod, color=color, alpha=0.45, linewidth=0.9)
    ax.set_xlim(time_min[0], time_min[-1])
    ax.tick_params(labelsize=TICK_FS)


def save(fig, fname):
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Figures
# =============================================================================


def plot_detector_and_tods(det_offsets, data, time_min, fname):
    """Detector layout (left) next to every detector's data (right), colored alike."""
    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 3.0], wspace=0.01)
    ax_det, ax_tod = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    x, y = det_offsets.T * RAD2ARCMIN
    ax_det.scatter(x, y, c=detector_colors(len(det_offsets)), s=88, linewidths=0.35, edgecolors="black")
    ax_det.set_aspect("equal", adjustable="datalim")
    ax_det.set_xlabel("Detector x [arcmin]", fontsize=LABEL_FS)
    ax_det.set_ylabel("Detector y [arcmin]", fontsize=LABEL_FS)
    ax_det.tick_params(labelsize=TICK_FS)

    plot_tods(ax_tod, time_min, data)
    ax_tod.set_xlabel("Time [minutes]", fontsize=LABEL_FS)
    ax_tod.set_ylabel(f"Observed data {K_RJ}", fontsize=LABEL_FS)
    ax_tod.yaxis.set_label_position("right")
    ax_tod.yaxis.tick_right()
    save(fig, fname)


def plot_scan_pattern(offsets, det_offsets, true_map, fov, fname):
    """Sky track of the central detector (black) over the true map.

    The central detector's track is centered on the map; an off-axis detector's
    track is shifted by its focal-plane offset.
    """
    det = int(np.argmin(np.hypot(*det_offsets.T)))
    extent = map_extent(fov)
    x, y = offsets[det].T * RAD2ARCMIN

    fig, ax = plt.subplots(figsize=(7, 6))
    img = show_map(ax, true_map, extent, symmetric_limits(true_map))
    ax.plot(x, y, color="black", linewidth=0.5)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_title("Scan pattern of the central detector")
    fig.colorbar(img, ax=ax, label=SIGNAL_LABEL, fraction=0.046, pad=0.02)
    save(fig, fname)


def plot_map_comparison(mean_map, true_map, fov, sigma_pixels, fname):
    """Reconstruction, beam-smoothed true map and true map, sharing one colorbar."""
    panels = {
        "Reconstruction (posterior mean)": mean_map,
        "True map, beam-smoothed": gaussian_filter(true_map, sigma_pixels, mode="constant"),
        "True map": true_map,
    }
    extent, vlim = map_extent(fov), symmetric_limits(true_map, mean_map)

    fig, axes = plt.subplots(1, len(panels), figsize=(17.5, 5.6), sharey=True, constrained_layout=True)
    for i, (ax, (title, sky)) in enumerate(zip(axes, panels.items())):
        img = show_map(ax, sky, extent, vlim, ylabel=i == 0)
        ax.set_title(title)
    fig.colorbar(img, ax=axes, label=SIGNAL_LABEL, fraction=0.046, pad=0.02)
    save(fig, fname)


def plot_tod_components(data, tod_mean, time_min, fname):
    """Observed data, reconstructed map response and residuals, one panel each."""
    panels = [
        (f"Observed data {K_RJ}", data, 1.0),
        (r"Reconstructed map response [mK$_\mathrm{RJ}$]", tod_mean, 1e3),
        (f"Residuals (model - data) {K_RJ}", tod_mean - data, 1.0),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(16, 4.0 * len(panels)), sharex=True)
    for ax, (label, tods, scale) in zip(axes, panels):
        plot_tods(ax, time_min, tods, scale)
        ax.set_ylabel(label, fontsize=LABEL_FS)
    axes[-1].set_xlabel("Time [minutes]", fontsize=LABEL_FS)
    fig.tight_layout(h_pad=0.4)
    save(fig, fname)


def plot_samples(map_samples, true_map, fov, fname, ncols=4):
    """Grid of posterior samples with one shared colorbar, scaled to the true map."""
    n = len(map_samples)
    nrows = int(np.ceil(n / ncols))
    extent, vlim = map_extent(fov), symmetric_limits(true_map)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols + 1, 3.6 * nrows), sharex=True, sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue
        img = show_map(ax, map_samples[i], extent, vlim, ylabel=i % ncols == 0)
        if i < n - ncols:  # x labels on the bottom row only
            ax.set_xlabel("")
    fig.colorbar(img, ax=axes, label=SIGNAL_LABEL, fraction=0.03, pad=0.01)
    save(fig, fname)


# =============================================================================
# Main
# =============================================================================


def plot_all(results, odir):
    """Write every demo figure from ``results`` (path to ``results.npz``) into ``odir``."""
    os.makedirs(odir, exist_ok=True)
    r = np.load(results)
    time_min = np.arange(r["data"].shape[-1]) / float(r["sample_rate"]) / 60.0
    mean_map = r["map_samples"].mean(axis=0)

    def out(name):
        return os.path.join(odir, name)

    plot_detector_and_tods(r["det_offsets"], r["data"], time_min, out("detector_and_tods.png"))
    plot_scan_pattern(r["offsets"], r["det_offsets"], r["true_map"], r["fov"], out("scan_pattern.png"))
    plot_map_comparison(mean_map, r["true_map"], r["fov"], float(r["sigma_pixels"]), out("map_comparison.png"))
    plot_tod_components(r["data"], r["tod_mean"], time_min, out("tod_components.png"))
    plot_samples(r["map_samples"], r["true_map"], r["fov"], out("map_posterior_samples.png"))
    print(f"figures written to {odir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    results = os.path.abspath(sys.argv[1])
    default_odir = os.path.join(os.path.dirname(os.path.dirname(results)), "plots")
    plot_all(results, sys.argv[2] if len(sys.argv) > 2 else default_odir)
