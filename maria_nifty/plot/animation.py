"""Animation helpers for time-dependent scatter visualizations."""

import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

from .util import set_cbar


def scatter_animation(
    coordinates,
    colors,
    size=100,
    fname="",
    title="",
    *,
    axes=None,
    cmap="cmb",
    vmin=None,
    vmax=None,
    limits=None,
    n_ticks=5,
    cbar=True,
    cbar_kwargs=None,
    callback=None,
    figsize=(5, 5),
    dpi=100,
    stride=10,
    length=250,
    fps=25,
    background=None,
    **kwargs,
):
    """Create and optionally save an animated scatter plot as a GIF.

    ``background`` selects the canvas color: ``None`` keeps the active
    matplotlib style, ``"black"`` switches to ``dark_background`` (white text
    and axes on black), ``"white"`` forces the default light style. Any other
    string is used directly as a facecolor. The color is also passed through to
    the GIF writer, since ``FuncAnimation.save`` otherwise renders frames on
    matplotlib's default white ``savefig.facecolor``.
    """
    if cbar_kwargs is None:
        cbar_kwargs = {}
    bg = None
    if background is not None:
        bg = str(background).lower()
        if bg == "black":
            plt.style.use("dark_background")
        elif bg == "white":
            plt.style.use("default")
    figure = plt.figure(figsize=figsize, dpi=dpi)
    if bg is not None:
        figure.patch.set_facecolor(bg)
    axes = []
    axes.append(figure.add_subplot(1, 1, 1))

    if coordinates.ndim == 2:
        coordinates = coordinates[:, None]
    if np.isscalar(colors) or colors.ndim < 2:
        colors = np.broadcast_to(colors, coordinates.shape[:2])
    if np.isscalar(size) or size.ndim < 2:
        sizes = np.broadcast_to(size, coordinates.shape[:2])

    coordinates = coordinates[:, ::stride][:, :length]
    colors = colors[:, ::stride][:, :length]
    sizes = sizes[:, ::stride][:, :length]

    # Both are indexed per frame below, so a scalar limit (a fixed color scale
    # for the whole animation, which is usually what you want -- a per-frame
    # scale makes the colors flicker) has to be broadcast to one value per frame.
    n_frames = colors.shape[1]
    vmin = colors.min(axis=0) if vmin is None else np.broadcast_to(np.asarray(vmin, dtype=float), (n_frames,))
    vmax = colors.max(axis=0) if vmax is None else np.broadcast_to(np.asarray(vmax, dtype=float), (n_frames,))

    scat = plt.scatter(
        coordinates[:, 0, 0],
        coordinates[:, 0, 1],
        sizes[:, 0],
        colors[:, 0],
        cmap=cmap,
        vmin=vmin[0],
        vmax=vmax[0],
        **kwargs,
    )

    set_cbar(axes[-1], scat, cbar, **cbar_kwargs)

    if title:
        axes[-1].set_title(title)

    if n_ticks == 0:
        axes[-1].axis("off")

    if bg is not None:
        axes[-1].set_facecolor(bg)

    if limits is None:
        limits = [(c.min() - 0.25 * np.abs(c.min()), c.max() + 0.25 * np.abs(c.max())) for c in coordinates.T]
    axes[-1].set_xlim(limits[0])
    axes[-1].set_ylim(limits[1])

    if callable(callback):
        callback(figure, axes)

    def init():
        return (scat,)

    def update(i):
        if coordinates.shape[1] > 1:
            scat.set_offsets(coordinates[:, i])
        if colors.shape[1] > 1:
            scat.set_array(colors[:, i])
            scat.set_clim(vmin=vmin[i], vmax=vmax[i])
        if sizes.shape[1] > 1:
            scat.set_sizes(sizes[:, i])
        return (scat,)

    ani = animation.FuncAnimation(figure, update, length, init, blit=True)

    if fname != "":
        odir = os.path.dirname(fname)
        odir = "." if odir == "" else odir
        os.makedirs(odir, exist_ok=True)
        if not fname.endswith(".gif"):
            fname += ".gif"
        save_kwargs = {} if bg is None else {"savefig_kwargs": {"facecolor": bg}}
        ani.save(fname, writer="pillow", fps=fps, **save_kwargs)
    else:
        display(HTML(ani.to_jshtml(fps=fps)))
    plt.close()
