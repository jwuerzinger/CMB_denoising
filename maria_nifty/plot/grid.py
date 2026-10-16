"""Grid-based wrappers that arrange multiple line or image plots."""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .image import image_plot
from .line import line_plot
from .util import plot_figure, to_shape


def grid_plot(
    array,
    fname="",
    title="",
    label="",
    mode="image",
    rows=None,
    cols=None,
    flip=False,
    *,
    n_ticks=5,
    callback=None,
    figsize=(5, 5),
    dpi=100,
    grid_kwargs=None,
    tight_layout_kwargs=None,
    repeat_kwargs="all",
    **kwargs,
):
    """Plot a grid of subplots with per-cell modes, labels, and titles.

    Note that the final ``tight_layout()`` call recomputes subplot spacing
    from scratch to avoid label/tick overlap, which overrides any ``wspace``/
    ``hspace`` given via ``grid_kwargs`` at GridSpec-construction time -- to
    control the actual final gap between panels, pass ``w_pad``/``h_pad``
    (fractions of fontsize, matplotlib's ``tight_layout`` units) via
    ``tight_layout_kwargs`` instead.
    """
    if grid_kwargs is None:
        grid_kwargs = {}
    if tight_layout_kwargs is None:
        tight_layout_kwargs = {}
    arrays = to_shape(array, None, rows, cols, 0.0, None, 2, flip)
    shape = arrays.shape[:2]
    rows, cols = shape

    shape_f = shape[::-1] if flip else shape
    titles = to_shape(title, shape_f, default="", flip=flip)
    labels = to_shape(label, shape_f, default="", flip=flip)
    modes = to_shape(mode, shape_f, default="line", flip=flip)

    if repeat_kwargs == "all":
        kwargs = to_shape(kwargs, shape_f, default={}, flip=flip)
    elif repeat_kwargs == "one":
        for key, val in kwargs.items():
            if isinstance(val, dict):
                kwargs[key] = to_shape(val, shape_f, default={}, flip=flip)
            else:
                kwargs[key] = to_shape(val, shape_f, default=-1, flip=flip)

    figsize = to_shape(figsize, (2,), dtype="float64") * np.array(shape_f[::-1])
    figure = plt.figure(figsize=figsize, dpi=dpi)
    axes = []
    grid = GridSpec(rows, cols, figure=figure, **grid_kwargs)

    for _i, (x, y) in enumerate(product(range(rows), range(cols))):
        axes.append(figure.add_subplot(grid[x, y]))

        array_i = arrays[x, y]

        if np.asarray(array_i).ndim != 2:
            continue

        mode_i = modes[x, y]
        match mode_i:
            case "line":
                plot_func = line_plot
            case "image":
                plot_func = image_plot
            case _:
                raise ValueError(f"Invalid mode: {mode_i}")

        kwargs_i = kwargs[x, y] if repeat_kwargs == "all" else {key: val[x, y] for key, val in kwargs.items()}

        plot_func(
            array=array_i,
            title=titles[x, y],
            label=labels[x, y],
            axes=axes,
            n_ticks=n_ticks,
            **kwargs_i,
        )

    if callable(callback):
        callback(figure, axes)

    grid.tight_layout(figure, **tight_layout_kwargs)

    plot_figure(fname)
