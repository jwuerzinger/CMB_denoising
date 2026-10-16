"""Scatter plotting helpers for detector layouts and coordinate clouds."""

import matplotlib.pyplot as plt
import numpy as np

from .util import plot_figure, set_cbar


def scatter_plot(
    coordinates,
    color=0,
    size=100,
    fname="",
    title="",
    *,
    axes=None,
    cmap="cmb",
    cbar=True,
    vmin=None,
    vmax=None,
    limits=None,
    n_ticks=5,
    cbar_kwargs=None,
    callback=None,
    figsize=(5, 5),
    dpi=100,
    **kwargs,
):
    """Plot 2D coordinates as a scatter map with an optional colorbar."""
    if cbar_kwargs is None:
        cbar_kwargs = {}
    plot_now = False
    if axes is None:
        figure = plt.figure(figsize=figsize, dpi=dpi)
        axes = []
        axes.append(figure.add_subplot(1, 1, 1))
        plot_now = True

    coordinates = coordinates.reshape((-1, 2))
    sizes = np.broadcast_to(size, coordinates.shape[:1])
    colors = np.broadcast_to(color, coordinates.shape[:1])

    img = plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        sizes,
        colors,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    set_cbar(axes[-1], img, cbar, **cbar_kwargs)

    if title != "":
        axes[-1].set_title(title)

    if limits is None:
        margin = (coordinates.max(axis=0) - coordinates.min(axis=0)) * 0.1
        limits = np.array([coordinates.min(axis=0) - margin, coordinates.max(axis=0) + margin]).T
    axes[-1].set_xlim(limits[0])
    axes[-1].set_ylim(limits[1])

    if n_ticks == 0:
        axes[-1].axis("off")

    if callable(callback):
        callback(figure, axes)

    if plot_now:
        plot_figure(fname)
