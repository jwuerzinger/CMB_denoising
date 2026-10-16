"""Residual and per-detector diagnostic plots."""

import os

import matplotlib.pyplot as plt
import nifty.re as jft
import numpy as np
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist

from ..detector import Detector


def _compute_plot_data(tod_array, model, samples):
    """Compute what to plot: raw TODs or residuals after model subtraction.

    Returns
    -------
    plot_data : numpy.ndarray, shape (n_detectors, n_timesamples)
    data_label : str
    file_suffix : str
    """
    if model is None:
        return tod_array, "TODs", "tods"

    if isinstance(samples, jft.Samples):
        sum_rec = jft.mean(tuple(model(s) for s in samples))
    elif isinstance(samples, jft.Vector):
        sum_rec = model(samples)
    else:
        sum_rec = jft.mean(tuple(model(s) for s in samples))

    return tod_array - np.asarray(sum_rec, dtype=float), "Residuals", "residuals"


def _cluster_plot_data(plot_data, n_clusters):
    """Cluster detectors by TOD similarity using Ward linkage.

    Returns
    -------
    cluster_indices : numpy.ndarray of int, shape (n_detectors,)
        1-based cluster labels.
    n_clusters_eff : int
        Actual number of clusters used (capped to n_detectors).
    group_colors : dict
        Mapping from 0-based cluster index to RGBA colour.
    """
    n_items = plot_data.shape[0]
    n_clusters_eff = max(1, min(n_clusters, n_items))

    if n_items < 2:
        cluster_indices = np.ones(n_items, dtype=int)
    else:
        dist = pdist(plot_data, metric="euclidean")
        if dist.size == 0:
            cluster_indices = np.ones(n_items, dtype=int)
        else:
            Z = linkage(dist, method="ward")
            cluster_indices = fcluster(Z, t=n_clusters_eff, criterion="maxclust")

    colors_map = plt.colormaps.get_cmap("cmb")
    color_denom = max(n_clusters_eff - 1, 1)
    group_colors = {i: colors_map(i / color_denom) for i in range(n_clusters_eff)}

    return cluster_indices, n_clusters_eff, group_colors


def _get_detector_layout(detector):
    """Extract unmasked detector coordinates, normalised to [0, 1].

    Returns
    -------
    detector_coords : numpy.ndarray, shape (n_unmasked, 2)
    coords_normalized : numpy.ndarray, shape (n_unmasked, 2)
    hex_group_indices : numpy.ndarray or None
    draw_group_boundaries : bool
    """
    coords_full = np.asarray(detector.coordinates)
    det_mask = np.asarray(
        getattr(detector, "mask", np.zeros(coords_full.shape[0], dtype=bool)),
        dtype=bool,
    )
    detector_coords = coords_full[~det_mask]

    coords_min = detector_coords.min(axis=0)
    coords_max = detector_coords.max(axis=0)
    coords_range = np.where((coords_max - coords_min) == 0, 1.0, coords_max - coords_min)
    coords_normalized = (detector_coords - coords_min) / coords_range

    hex_group_indices = None
    if detector.groups is not None:
        hex_group_indices = np.asarray(detector.groups.indices)[~det_mask]
    draw_group_boundaries = hex_group_indices is not None and getattr(detector, "n_groups", 1) > 1

    return detector_coords, coords_normalized, coords_min, coords_range, hex_group_indices, draw_group_boundaries


def _stacked_plot(plot_data, cluster_indices, group_colors, data_label, legend_labels=None):
    """Return a figure with all detector traces stacked vertically.

    When ``legend_labels`` is given (one label per colour index in
    ``group_colors``), a legend mapping colour -> label is drawn.
    """
    n_detectors = plot_data.shape[0]
    fig_height = max(8, 0.25 * n_detectors)
    fig, ax = plt.subplots(figsize=(20, fig_height))

    residual_spans = np.ptp(plot_data, axis=1)
    global_span = float(np.max(residual_spans)) if residual_spans.size > 0 else 0.0
    padding = 0.02 * global_span if global_span > 0 else 1e-3

    trace_offsets = np.zeros(n_detectors, dtype=float)
    for i in range(1, n_detectors):
        prev_bottom = np.min(plot_data[i - 1] + trace_offsets[i - 1])
        curr_top = np.max(plot_data[i])
        trace_offsets[i] = prev_bottom - curr_top - padding

    tick_positions = np.zeros(n_detectors, dtype=float)
    for i in range(n_detectors):
        color = group_colors[cluster_indices[i] - 1]
        shifted = plot_data[i] + trace_offsets[i]
        ax.plot(shifted, color=color, linewidth=1)
        tick_positions[i] = np.median(shifted)

    y_margin = global_span if global_span > 0 else 1e-3
    ax.set_xlim(0, plot_data.shape[1] - 1)
    ax.set_ylim(tick_positions[-1] - y_margin, tick_positions[0] + y_margin)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(i) for i in range(n_detectors)], fontsize=8)
    ax.set_ylabel("Detector Number", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_title(f"TOD {data_label} per Detector", fontsize=16, fontweight="bold")
    ax.grid(False)

    if legend_labels is not None:
        handles = [Line2D([0], [0], color=group_colors[i], lw=2, label=label) for i, label in enumerate(legend_labels)]
        ax.legend(handles=handles, loc="upper right", fontsize=11, framealpha=0.9)

    plt.tight_layout()
    return fig


def _spatial_plot(
    plot_data,
    cluster_indices,
    group_colors,
    detector_coords,
    coords_normalized,
    coords_min,
    coords_range,
    hex_group_indices,
    draw_group_boundaries,
    vmin,
    vmax,
    data_label,
):
    """Return a figure with each detector's TOD placed at its sky position."""
    subplot_width = 0.04
    subplot_height = 0.04
    margin = 0.05

    fig = plt.figure(figsize=(20, 20))
    ax_bg = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)

    if draw_group_boundaries:
        n_hex_groups = hex_group_indices.max() + 1
        for group_id in range(n_hex_groups):
            group_coords = detector_coords[hex_group_indices == group_id]
            if len(group_coords) > 2:
                try:
                    hull = ConvexHull(group_coords)
                    hull_norm = (group_coords[hull.vertices] - coords_min) / coords_range
                    x = margin + subplot_width / 2 + hull_norm[:, 0] * (1 - 2 * margin - subplot_width)
                    y = margin + subplot_height / 2 + hull_norm[:, 1] * (1 - 2 * margin - subplot_height)
                    ax_bg.plot(
                        np.hstack([x, x[0]]),
                        np.hstack([y, y[0]]),
                        "k-",
                        linewidth=3.5,
                        alpha=1.0,
                        zorder=0,
                    )
                except QhullError:
                    pass

    n_detectors = plot_data.shape[0]
    for i in range(n_detectors):
        left = margin + coords_normalized[i, 0] * (1 - 2 * margin - subplot_width)
        bottom = margin + coords_normalized[i, 1] * (1 - 2 * margin - subplot_height)
        ax = fig.add_axes([left, bottom, subplot_width, subplot_height])

        color = group_colors[cluster_indices[i] - 1]
        ax.plot(plot_data[i], color=color, linewidth=0.8)
        ax.fill_between(range(len(plot_data[i])), plot_data[i], alpha=0.3, color=color)
        ax.set_ylim(vmin, vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            0.95,
            f"{i}",
            transform=ax.transAxes,
            fontsize=7,
            ha="center",
            va="top",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    ax_bg.set_xlabel("Detector X Coordinate", fontsize=14, labelpad=20)
    ax_bg.set_ylabel("Detector Y Coordinate", fontsize=14, labelpad=20)
    ax_bg.set_title(
        f"TOD {data_label} Arranged by Detector Position",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax_bg.grid(True, alpha=0.2)
    ax_bg.set_xticks([])
    ax_bg.set_yticks([])

    return fig


def detector_residual_plots(
    detector=None,
    tod_data=None,
    model=None,
    samples=None,
    n_clusters=5,
    vmin=-0.1,
    vmax=0.1,
    odir=None,
    categories=None,
    category_colors=None,
    category_labels=None,
    plots=("grid", "spatial"),
    dpi=150,
    fname=None,
):
    """Create stacked and spatial detector plots for TODs or residuals.

    Parameters
    ----------
    detector : Detector
        Detector geometry and grouping metadata.
    tod_data : array-like, shape (n_detectors, n_timesamples)
        Raw time-ordered data.
    model : callable or None
        Forward model; required when ``samples`` is provided.
    samples : jft.Samples or jft.Vector or None
        Posterior samples used to compute residuals. Must be paired with
        ``model``; omit both to plot raw TODs.
    n_clusters : int, optional
        Number of detector clusters for colour-coding (ignored when
        ``categories`` is given).
    vmin, vmax : float, optional
        Colour / y-axis limits for the spatial plot.
    odir : str or None, optional
        Output directory; figures are saved there when provided, otherwise
        displayed interactively.
    categories : array-like of int or None, optional
        Per-detector 0-based category label. When given, detectors are coloured
        by category (with a legend) instead of by similarity clustering.
    category_colors : sequence, optional
        Colour per category index; defaults to the ``tab10`` colormap.
    category_labels : sequence of str, optional
        Legend label per category index.
    plots : sequence of str, optional
        Which figures to produce: any of ``"grid"`` (stacked) and ``"spatial"``.
    dpi : int, optional
        Save resolution (default 150). Lower it (e.g. 100) to render/save
        faster at the cost of sharpness; the figures are large in inches.
    fname : str or None, optional
        Exact output path for the grid figure (the spatial figure, if produced,
        is saved alongside with a ``_spatial`` suffix). Overrides the default
        ``{odir}/grid_{suffix}.png`` naming; ``odir`` may be omitted.
    """
    if not isinstance(detector, Detector):
        raise TypeError("detector must be a Detector object")
    if tod_data is None:
        raise ValueError("tod_data must not be None")

    tod_array = np.asarray(tod_data, dtype=float)
    if tod_array.ndim != 2:
        raise ValueError("tod_data must be a 2D array with shape (n_detectors, n_timesamples)")

    if (model is None) != (samples is None):
        raise ValueError("Provide both model and samples for residual mode, or neither for direct TOD mode")

    plot_data, data_label, file_suffix = _compute_plot_data(tod_array, model, samples)

    legend_labels = None
    if categories is not None:
        # Colour by externally supplied category instead of similarity clusters.
        cluster_indices = np.asarray(categories, dtype=int) + 1  # 1-based, like fcluster labels
        n_cat = len(category_colors) if category_colors is not None else int(cluster_indices.max())
        if category_colors is None:
            cmap = plt.colormaps.get_cmap("tab10")
            category_colors = [cmap(i) for i in range(n_cat)]
        group_colors = {i: category_colors[i] for i in range(n_cat)}
        legend_labels = category_labels
    else:
        cluster_indices, _n_clusters_eff, group_colors = _cluster_plot_data(plot_data, n_clusters)

    detector_coords, coords_normalized, coords_min, coords_range, hex_group_indices, draw_group_boundaries = (
        _get_detector_layout(detector)
    )

    # Trim to the shorter of the two axes when shape mismatch occurs from upstream.
    n_det = min(plot_data.shape[0], detector_coords.shape[0])
    plot_data = plot_data[:n_det]
    cluster_indices = cluster_indices[:n_det]
    detector_coords = detector_coords[:n_det]
    coords_normalized = coords_normalized[:n_det]
    if hex_group_indices is not None:
        hex_group_indices = hex_group_indices[:n_det]

    # When ``fname`` is given it sets the exact grid output path (spatial gets a
    # "_spatial" sibling); otherwise figures are named under ``odir``.
    save = (odir is not None) or (fname is not None)
    if fname is not None:
        os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    elif odir is not None:
        os.makedirs(odir, exist_ok=True)

    if "grid" in plots:
        fig1 = _stacked_plot(plot_data, cluster_indices, group_colors, data_label, legend_labels=legend_labels)
        if save:
            grid_path = fname if fname is not None else f"{odir}/grid_{file_suffix}.png"
            fig1.savefig(grid_path, dpi=dpi, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig1)

    if "spatial" in plots:
        fig2 = _spatial_plot(
            plot_data,
            cluster_indices,
            group_colors,
            detector_coords,
            coords_normalized,
            coords_min,
            coords_range,
            hex_group_indices,
            draw_group_boundaries,
            vmin,
            vmax,
            data_label,
        )
        if save:
            if fname is not None:
                root, ext = os.path.splitext(fname)
                spatial_path = f"{root}_spatial{ext}"
            else:
                spatial_path = f"{odir}/spatial_{file_suffix}.png"
            fig2.savefig(spatial_path, dpi=dpi, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig2)
