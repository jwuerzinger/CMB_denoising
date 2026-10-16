"""Line plotting helpers for 1D or stacked time-series data."""

import matplotlib.pyplot as plt
import numpy as np

from .util import plot_figure, to_shape


def line_plot(
    array,
    color=None,
    fname="",
    label="",
    title="",
    *,
    axes=None,
    cmap="cmb",
    linestyle="-",
    linewidth=1,
    alpha=1.0,
    vmin=None,
    vmax=None,
    n_ticks=5,
    callback=None,
    figsize=(5, 2.5),
    dpi=100,
    legend_loc="best",
    **kwargs,
):
    """Plot one or more 1D series with shared axis styling."""
    plot_now = False
    if axes is None:
        figure = plt.figure(figsize=figsize, dpi=dpi)
        axes = []
        axes.append(figure.add_subplot(1, 1, 1))
        plot_now = True

    array = np.array(array, dtype="float64")

    cmap = plt.colormaps.get_cmap(cmap)

    if array.ndim == 1:
        array = array.reshape((1, -1))
    elif array.ndim != 2:
        raise ValueError("Array must be 1D or 2D.")

    if isinstance(color, np.ndarray) and color.ndim == 2 and color.shape[1] in (3, 4):
        # Explicit per-row RGB(A), used as-is (no categorical palette lookup).
        colors = color
    elif isinstance(color, np.ndarray):
        n_colors, inv = np.unique(color, return_inverse=True)
        palette = cmap(np.linspace(0, 1, len(n_colors)))
        colors = np.array([palette[i] for i in inv])
    elif isinstance(color, str):
        colors = np.array([color] * array.shape[0], dtype=object)
    else:
        n_colors = color if isinstance(color, int) else array.shape[0]
        palette = cmap(np.linspace(0, 1, n_colors))
        colors = np.array([palette[i % n_colors] for i in range(array.shape[0])])

    labels = to_shape(label, array.shape[:1], default="")
    linestyles = to_shape(np.asarray(linestyle).squeeze(), array.shape[:1])
    linewidths = to_shape(np.asarray(linewidth).squeeze(), array.shape[:1])
    alphas = to_shape(np.asarray(alpha).squeeze(), array.shape[:1])
    xs = np.arange(array.shape[1])

    for i in range(array.shape[0]):
        axes[-1].plot(
            xs,
            array[i],
            label=labels[i],
            color=colors[i],
            ls=linestyles[i],
            lw=linewidths[i],
            alpha=alphas[i],
        )

    if np.any(labels != ""):
        axes[-1].legend(loc=legend_loc)

    if vmin is not None or vmax is not None:
        axes[-1].set_ylim(vmin, vmax)

    if title != "":
        axes[-1].set_title(title)

    if n_ticks == 0:
        axes[-1].tick_params(labelbottom=False, labelleft=False)

    if callable(callback):
        callback(figure, axes)

    if plot_now:
        plot_figure(fname)
