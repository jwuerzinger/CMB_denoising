"""Image-style plotting helpers for 2D map-like data."""

import matplotlib.pyplot as plt
import numpy as np

from .util import plot_figure, set_cbar


def image_plot(
    array,
    fname="",
    title="",
    *,
    axes=None,
    cmap="cmb",
    norm="linear",
    cbar=True,
    vmin=None,
    vmax=None,
    n_ticks=5,
    origin="upper",
    extent=None,
    transpose=False,
    cbar_kwargs=None,
    marker_kwargs=None,
    contour_kwargs=None,
    callback=None,
    figsize=(5, 5),
    dpi=100,
    **kwargs,
):
    """Plot a 2D array as an image with optional colorbar and overlays."""
    if contour_kwargs is None:
        contour_kwargs = {}
    if marker_kwargs is None:
        marker_kwargs = {}
    if cbar_kwargs is None:
        cbar_kwargs = {}
    plot_now = False
    if axes is None:
        figure = plt.figure(figsize=figsize, dpi=dpi)
        axes = []
        axes.append(figure.add_subplot(1, 1, 1))
        plot_now = True

    array = np.array(array, dtype="float64")

    img = axes[-1].imshow(
        X=array.T if transpose else array,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        origin=origin,
        extent=extent,
    )

    set_cbar(axes[-1], img, cbar, **cbar_kwargs)

    if title != "":
        axes[-1].set_title(title)

    marker = {"m0": marker_kwargs} if all(k in marker_kwargs for k in ["x", "y"]) else marker_kwargs
    for mrk in marker.values():
        if isinstance(mrk, dict) and all(k in mrk for k in ["x", "y"]):
            axes[-1].scatter(**mrk)
        else:
            raise TypeError("`marker` has to be a dictionary with keys `x`, `y`.")

    if contour_kwargs:
        contour_array = contour_kwargs.copy().pop("array")
        contour_array = contour_array.T if transpose else contour_array
        axes[-1].contour(contour_array, origin=origin, **contour_kwargs)

    if n_ticks == 0:
        axes[-1].axis("off")

    if callable(callback):
        callback(figure, axes)

    if plot_now:
        plot_figure(fname)
