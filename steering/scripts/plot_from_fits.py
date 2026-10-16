"""Regenerate ``talk_plots.py``'s figures from ``save_fits.py``'s FITS exports.

Same figures, styling and options as ``steering/talk_plots.py``, but without
``last.pkl``, the raw TOD files or rebuilding the model: everything is read
from the files ``scripts/save_fits.py`` writes to ``plot_{run}/save_fits/``:

* ``map_samples.fits`` -- posterior states and prior draws of the full map and
  each component (means, stds, rel-std and pull maps are computed from these).
* ``map_mean.fits`` -- header only: beam sigma (``BEAMSIG``) for
  ``map_smoothed.png``.
* ``tod_mean.fits`` -- per dataset: data, posterior-mean atmosphere/map/model
  TODs, detector table, one prior draw of the atmosphere/map TODs, and the
  detector pointing in map pixels (to project ``--fits_map`` into TOD space).
* ``config.yaml`` -- the fit's config, for the scan-label legend.

Differences from ``talk_plots.py``:

* No bin-mapper. The map comparison figures (``map_comparison*.png``,
  ``map_diff*.png``) are only drawn when an external
  reference map is passed via ``--fits_map``; without it they are skipped.
* The map prior draws are fixed at export time (``save_fits.py``'s
  ``--n_prior``/``--prior_vmin_mk``/``--prior_vmax_mk``).

Figures go to ``<fits_dir>/../<subdir>/`` (i.e. ``plot_{run}/<subdir>/``)
unless ``--out_dir`` is given.

Example::

    pixi run -e cpu python scripts/plot_from_fits.py \\
        --fits_dir runs/RXJ1347/AGBT17A_358_test/10tods_10amin_8s_nl/plot_4/save_fits \\
        --fits_map runs/RXJ1347_20.fits --map_vmin -2.75e-3 --map_vmax 2.75e-3
"""

import os
from types import SimpleNamespace

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from astropy.io import fits
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from nifty.re import logger

from maria_nifty.fits import from_fits_data
from maria_nifty.model.response import build_pointing_matrix, map2tods
from maria_nifty.model.util import ARCMIN2RAD
from maria_nifty.plot.util import _format_unit_label, apply_sky_ticks, psd_unit_label


def load_map_fits(fits_dir):
    """Read ``map_samples.fits`` (+ ``map_mean.fits``'s header) into a namespace.

    ``src_samples``/``comp_samples`` hold the *source* map (background
    excluded, as in ``talk_plots.py``) per posterior state, ``prior_src`` the
    prior draws; all ``(n, nx, ny)`` in model orientation. The grid fields
    (``shape``, ``distances``, ``center`` in rad) mimic ``GridSpace`` so
    ``extent_arcmin`` works unchanged.
    """
    with fits.open(os.path.join(fits_dir, "map_samples.fits")) as hdul:
        hdr = hdul[0].header
        names = [hdr[f"COMP{i}"] for i in range(1, int(hdr["NCOMP"]) + 1)]
        cubes = {h.name: from_fits_data(h.data) for h in hdul[1:]}
    bg = "BG" if "BG" in cubes else None
    sources = [n for n in names if n.upper() != "BG"]

    def source_sum(prefix):
        full = cubes.get(f"{prefix}FULL")
        if full is None:
            return None
        return full - cubes[f"{prefix}{bg}"] if bg else full

    with fits.open(os.path.join(fits_dir, "map_mean.fits")) as hdul:
        beam_sigma = float(hdul[0].header.get("BEAMSIG", 0.0))

    shape = cubes["FULL"].shape[1:]
    return SimpleNamespace(
        names=sources,
        src_samples=source_sum(""),
        comp_samples=[cubes[n.upper()] for n in sources],
        prior_src=source_sum("PRIOR_"),
        beam_sigma=beam_sigma,
        units=str(hdr.get("BUNIT", "")) or None,
        center_phi_theta=(float(hdr["CPHI0"]), float(hdr["CTHETA0"])),
        space=SimpleNamespace(
            shape=shape,
            distances=(float(hdr["PIXX"]), float(hdr["PIXY"])),
            fov=np.asarray(shape, dtype=float) * np.array([float(hdr["PIXX"]), float(hdr["PIXY"])]),
            cen=np.array([float(hdr["CENX"]), float(hdr["CENY"])]),
        ),
    )


def mean_std(cube):
    """Mean and std over the leading (sample) axis, matching ``jft.mean_and_std``
    (``correct_bias=True``, i.e. ``ddof=1``); a single state gets std 0."""
    cube = np.asarray(cube, dtype=float)
    if cube.shape[0] < 2:
        return cube[0], np.zeros_like(cube[0])
    return cube.mean(axis=0), cube.std(axis=0, ddof=1)


def load_tod_fits(fits_dir):
    """Read ``tod_mean.fits`` into ``talk_plots``-style dataset dicts.

    Each dict carries ``tod_data``, the posterior means ``atm``/``map``/
    ``model`` (``None`` if the fit had no such component), the prior draws
    ``prior_atm``/``prior_map``, the map response's ``pointing`` (pixel
    ``coords`` + projection settings, ``None`` without a map), a ``detector``
    with ``coordinates``/``full_size``, a ``tods`` with ``fs``/``units``, the
    duration in seconds, and ``mask = None`` (as after ``build_detector``).
    """
    with fits.open(os.path.join(fits_dir, "tod_mean.fits")) as hdul:
        primary = hdul[0].header
        units = str(primary.get("BUNIT", "")) or None
        meta = SimpleNamespace(
            noise_std=float(primary.get("NOISESTD", np.nan)),
            key=int(primary.get("KEY", 123)),
        )
        datasets = []
        for k in range(1, int(primary["NDATA"]) + 1):
            ext = f"D{k:02d}"
            cube_hdu = hdul[ext]
            chans = {
                cube_hdu.header[f"CHAN{i}"]: np.asarray(cube_hdu.data[i], dtype=float)
                for i in range(cube_hdu.data.shape[0])
            }
            prior = {}
            if f"{ext}_PRIOR" in hdul:
                p_hdu = hdul[f"{ext}_PRIOR"]
                prior = {
                    p_hdu.header[f"CHAN{i}"]: np.asarray(p_hdu.data[i], dtype=float) for i in range(p_hdu.data.shape[0])
                }
            pointing = None
            if f"{ext}_PNT" in hdul:
                ph = hdul[f"{ext}_PNT"]
                pointing = SimpleNamespace(
                    coords=np.asarray(ph.data, dtype=np.float64),
                    kind=str(ph.header["KIND"]),
                    order=int(ph.header["ORDER"]),
                    mode=str(ph.header["MODE"]),
                    cval=float(ph.header["CVAL"]),
                )
            dets = hdul[f"{ext}_DETS"]
            coords = np.stack([np.asarray(dets.data["X"]), np.asarray(dets.data["Y"])], axis=-1)
            dh = dets.header
            datasets.append(
                {
                    "tod_data": chans["DATA"],
                    "atm": chans.get("ATM_MEAN"),
                    "map": chans.get("MAP_MEAN"),
                    "model": chans["MODEL_MEAN"],
                    "prior_atm": prior.get("ATM"),
                    "prior_map": prior.get("MAP"),
                    "pointing": pointing,
                    "detector": SimpleNamespace(coordinates=coords, full_size=coords.shape[0]),
                    "tods": SimpleNamespace(fs=float(dh.get("FS", 1.0)), units=units),
                    "duration_s": float(dh["TEND"] - dh["TSTART"]) if "TSTART" in dh else 0.0,
                    "mask": None,
                }
            )
    return datasets, meta


def extent_arcmin(space):
    """[xmin, xmax, ymin, ymax] extent (arcmin) of a GridSpace, for imshow/ticks."""
    fov = np.asarray(space.fov, dtype=float) / ARCMIN2RAD
    cen = np.asarray(space.cen, dtype=float).reshape(-1)[:2] / ARCMIN2RAD
    return [cen[0] - 0.5 * fov[0], cen[0] + 0.5 * fov[0], cen[1] - 0.5 * fov[1], cen[1] + 0.5 * fov[1]]


def crop_center_fov(arr, extent, factor):
    """Crop a ``(..., nx, ny)`` array and its ``[x0, x1, y0, y1]`` arcmin
    extent to the central ``factor`` fraction of the field of view, in both
    x and y (e.g. ``factor=0.5`` keeps only the center half of the FOV in
    each direction). ``factor >= 1`` (or <= 0) is a no-op.
    """
    factor = float(factor)
    if factor >= 1.0 or factor <= 0.0:
        return arr, extent
    arr = np.asarray(arr)
    nx, ny = arr.shape[-2], arr.shape[-1]
    new_nx = max(1, int(round(nx * factor)))
    new_ny = max(1, int(round(ny * factor)))
    x0i, y0i = (nx - new_nx) // 2, (ny - new_ny) // 2
    cropped = arr[..., x0i : x0i + new_nx, y0i : y0i + new_ny]

    ext = [float(v) for v in np.asarray(extent, dtype=float).reshape(-1)[:4]]
    cx, cy = 0.5 * (ext[0] + ext[1]), 0.5 * (ext[2] + ext[3])
    half_w = 0.5 * (ext[1] - ext[0]) * (new_nx / nx)
    half_h = 0.5 * (ext[3] - ext[2]) * (new_ny / ny)
    new_extent = [cx - half_w, cx + half_w, cy - half_h, cy + half_h]
    return cropped, new_extent


def compute_rel_std_stats(mean_arr, std_arr):
    """Relative-std (std/|mean|) plus the no-model-support mask and log-scale
    vmin, shared by every relative-std-style panel in this file (see
    ``rel_std_log_plot`` in ``map_reconstruction_plot`` for how they're used).
    """
    abs_mean = np.abs(mean_arr)
    abs_std_raw = np.abs(std_arr)
    eps = max(1e-12, 1e-3 * float(np.percentile(abs_mean, 5)))
    std_eps = max(1e-12, 1e-3 * float(np.percentile(abs_std_raw, 5)))
    rel_std_raw = std_arr / np.maximum(abs_mean, eps)
    # Where both mean and std are effectively zero (no model support at all,
    # e.g. far outside the fitted source) the ratio is a meaningless
    # noise-floor artifact rather than a real relative uncertainty -- force
    # it to that plot's own vmax (fully saturated/uncertain) instead.
    # Compares against the same tiny percentile-based floors as the epsilon
    # above, since continuous model outputs essentially never hit an exact 0.
    no_support = (abs_mean <= eps) & (abs_std_raw <= std_eps)
    # Percentile computed on the no-support-masked ratio (forced to a fixed
    # 1.25 ceiling), not the raw ratio -- otherwise the ambiguous 0/0 region
    # contaminates this plot's vmin with values other rel-std panels never
    # see, and the two "same" quantities end up with different floors.
    positive_raw = np.where(no_support, 1.25, rel_std_raw)
    positive_raw = positive_raw[positive_raw > 0]
    rel_vmin = max(1e-6, float(np.percentile(positive_raw, 1))) if positive_raw.size else 0
    return rel_std_raw, no_support, rel_vmin


def load_fits_map(fits_path, shape, fov_arcmin, center_arcmin, center_phi_theta, hdu=0, data_scale=1.0):
    """Resample an external FITS map onto the NIFTy source-space's own grid.

    ``shape``/``fov_arcmin``/``center_arcmin`` describe the target local
    tangent-plane grid (same convention as ``GridSpace``/``extent_arcmin`` --
    pixel *centers*, not edges); the FITS file's own WCS header is used to look
    up, for every target pixel's (RA, Dec), the corresponding fractional pixel
    position in the FITS data, which is then bilinearly sampled. This puts the
    FITS map on the exact same extent/sky-ticks as the reconstruction panel,
    regardless of the FITS file's own pixel scale or footprint.

    Degenerate (length-1) frequency/Stokes axes, as in CASA images, are
    dropped (only the celestial WCS is used). The data are first multiplied by
    ``data_scale``; if the header's BUNIT is Jy/beam, the result is then
    converted to K_RJ via the header's BMAJ/BMIN (Gaussian beam solid angle)
    and its spectral-axis frequency (or RESTFRQ), so it lands in the same
    native units as the reconstruction.
    """
    from astropy.io import fits as astropy_fits
    from astropy.wcs import WCS
    from maria.coords.transforms import unjitted_offsets_to_phi_theta
    from scipy.ndimage import map_coordinates as sp_map_coordinates

    with astropy_fits.open(fits_path) as hdul:
        header = hdul[hdu].header
        data = np.asarray(hdul[hdu].data, dtype=float)
        wcs = WCS(header).celestial
    data = data.reshape(data.shape[-2:]) * float(data_scale)

    if "jy/beam" in str(header.get("BUNIT", "")).strip().lower().replace(" ", ""):
        freq_hz = None
        for i in range(1, int(header.get("NAXIS", 0)) + 1):
            if str(header.get(f"CTYPE{i}", "")).upper().startswith("FREQ"):
                freq_hz = float(header[f"CRVAL{i}"])
        if freq_hz is None:
            freq_hz = float(header["RESTFRQ"])
        omega_sr = np.pi / (4 * np.log(2)) * np.radians(float(header["BMAJ"])) * np.radians(float(header["BMIN"]))
        c, k_b = 2.99792458e8, 1.380649e-23
        data = data * 1e-26 * c**2 / (2 * k_b * freq_hz**2 * omega_sr)

    nx, ny = int(shape[0]), int(shape[1])
    fov_x, fov_y = (float(v) for v in fov_arcmin)
    cen_x, cen_y = (float(v) for v in center_arcmin)
    xs = (np.arange(nx) + 0.5) / nx * fov_x - 0.5 * fov_x + cen_x
    ys = (np.arange(ny) + 0.5) / ny * fov_y - 0.5 * fov_y + cen_y
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    cphi0, ctheta0 = center_phi_theta
    dX = np.stack([xx, yy], axis=-1) * ARCMIN2RAD
    phi_theta = np.asarray(unjitted_offsets_to_phi_theta(dX, cphi0, ctheta0))
    ra_deg = np.degrees(phi_theta[..., 0])
    dec_deg = np.degrees(phi_theta[..., 1])

    px, py = wcs.all_world2pix(ra_deg.ravel(), dec_deg.ravel(), 0)
    sampled = sp_map_coordinates(data, [py, px], order=1, mode="constant", cval=np.nan)
    return sampled.reshape(nx, ny)


def make_square_grid_figure(nrows, n_img_cols, cbar_col_ratio=0.06, wspace=0.04, hspace=0.04, row_height=7 / 1.5):
    """A ``(fig, gs)`` GridSpec figure: ``n_img_cols`` equal-width image
    columns plus one thin trailing column for a colorbar, repeated over
    ``nrows`` rows.

    Every image panel in these map figures keeps a 1:1 (square) data aspect
    (imshow's default), so matplotlib auto-shrinks whichever of a cell's two
    dimensions is oversized to enforce it -- if a GridSpec column isn't
    already exactly as wide (in inches) as a row is tall, that shrink pads
    one direction's panel-to-panel gaps more than the other's, even with
    ``wspace == hspace``. The figure width is solved for up front so image
    columns and rows come out exactly square, avoiding any such uneven
    padding.
    """
    width_ratios = [1] * n_img_cols + [cbar_col_ratio]
    fig_height = row_height * nrows

    left = plt.rcParams["figure.subplot.left"]
    right = plt.rcParams["figure.subplot.right"]
    top = plt.rcParams["figure.subplot.top"]
    bottom = plt.rcParams["figure.subplot.bottom"]
    row_h_in = (top - bottom) / (nrows + (nrows - 1) * hspace) * fig_height
    n_cols = n_img_cols + 1
    col_w_frac_per_unit = (right - left) / (n_cols + (n_cols - 1) * wspace) * (n_cols / sum(width_ratios))
    fig_width = row_h_in / col_w_frac_per_unit

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)
    gs = GridSpec(nrows, n_cols, figure=fig, width_ratios=width_ratios, wspace=wspace, hspace=hspace)
    return fig, gs


def attach_cell_colorbar(fig, gs, cell, img, label):
    """Attach a colorbar to a GridSpec ``cell`` (e.g. ``gs[0, 3]``), matching
    that cell's exact position -- ``Colorbar()`` installs its own axes
    locator (for "extend" triangle handling) that can nudge the assigned
    rectangle at render time, so it is detached and the exact cell position
    re-applied afterward.
    """
    cax = fig.add_subplot(cell)
    cax.grid(False)
    fig.colorbar(img, cax=cax, label=label)
    cax.set_axes_locator(None)
    cax.set_position(cell.get_position(fig))
    return cax


def attach_axes_colorbar(fig, ax, img, label, **kwargs):
    """Attach a colorbar to a single (non-GridSpec) axes, then re-match its
    height exactly to the axes' own y-extent -- fig.colorbar()'s default
    sizing (via matplotlib's make_axes) comes out slightly smaller than the
    axes' own height even for a single parent axes.
    """
    y0, y1 = ax.get_position().y0, ax.get_position().y1
    cb = fig.colorbar(img, ax=ax, location="right", label=label, **kwargs)
    cb.ax.set_axes_locator(None)
    cpos = cb.ax.get_position()
    cb.ax.set_position([cpos.x0, y0, cpos.width, y1 - y0])
    return cb


@click.command()
@click.option(
    "--fits_dir",
    required=True,
    help="save_fits.py output folder (plot_<run>/save_fits) holding map_mean.fits, map_samples.fits, "
    "tod_mean.fits and config.yaml.",
)
@click.option(
    "--out_dir",
    default=None,
    help="Output folder for the figures. Defaults to <fits_dir>/../<subdir>, i.e. plot_<run>/<subdir>/.",
)
@click.option(
    "--theme",
    type=click.Choice(["light", "dark"]),
    default="light",
    show_default=True,
    help="Tick/label/grid color scheme: 'light' (black ticks) or 'dark' (white ticks). The "
    "figure/axes canvas is always transparent (no solid fill), so this is the only color "
    "choice -- pick whichever contrasts with wherever the PNG gets composited.",
)
@click.option("--grid", default=False, help="Draw grid lines on the plots (True/False).")
@click.option(
    "--subdir",
    default="paper",
    show_default=True,
    help="Name of the output subfolder under plot_<run>/, e.g. 'paper_black' for a separate variant. "
    "Ignored if --out_dir is given.",
)
@click.option(
    "--no_tods",
    default=False,
    show_default=True,
    help="Skip the per-dataset tod_NN/ folders (detector/component/power-spectra/prior plots for each "
    "individual dataset) -- everything else (maps, and the combined-across-all-datasets TOD spectra/"
    "power-spectra/atmosphere-prior plots) still regenerates. Much faster than a full run when only the "
    "maps or the combined plots changed.",
)
@click.option(
    "--master_config",
    default=None,
    help="Path to the original pipeline config (e.g. config/rxj1347_17A.yml) whose data section has the "
    "real scan filenames -- used to derive spectra_data.png's per-dataset 'Scan NN-sNN' legend labels when "
    "this run's own saved config points to an intermediate-stage file (e.g. .../remove_jumps/d1.fits) "
    "instead, matched by dataset key (d1, d2, ...).",
)
@click.option(
    "--fits_map",
    default=None,
    help="Path to an external reference FITS map (e.g. a minkasi/other-pipeline map) to compare the "
    "reconstruction against in map_comparison*.png/map_diff*.png, and to add as the 'Minkasi map' "
    "curve (projected through the exported detector pointing) in spectra_components.png. Resampled (via the "
    "FITS file's own WCS) onto the exact same FOV/center/resolution as the NIFTy reconstruction, so the "
    "two are directly comparable. Without it, those comparison figures are skipped.",
)
@click.option(
    "--extra_fits_map",
    multiple=True,
    help="Additional external FITS map(s) (repeatable), loaded and resampled exactly like --fits_map "
    "(same --fits_hdu/--fits_scale). With --fits_map, writes map_comparison_multi.png: reconstruction, "
    "--fits_map, then each extra map in order, all with the reconstruction's contour overlaid.",
)
@click.option(
    "--fits_hdu",
    default=0,
    show_default=True,
    help="HDU index to read from --fits_map.",
)
@click.option(
    "--fits_scale",
    default=1.0,
    show_default=True,
    help="Multiply the --fits_map data by this factor on load, before any unit conversion (e.g. 1e-3 "
    "for files whose pixels were scaled x1000, i.e. mJy/beam, while BUNIT still says Jy/beam). A "
    "BUNIT of Jy/beam is converted to K_RJ automatically from the header's BMAJ/BMIN and frequency.",
)
@click.option(
    "--rel_vmax",
    default="1.25",
    show_default=True,
    help="Comma-separated upper limits for map_relative_std.png's log color scale. "
    "Each value writes its own file: 1.25 -> map_relative_std.png (the default name), "
    '"auto" -> map_relative_std_auto.png (scaled to the data\'s own max), any other v -> '
    'map_relative_std_vmax{v}.png. e.g. --rel_vmax "auto,1.5,2". Note vmax also fills the '
    "no-model-support pixels, so each file saturates those to its own ceiling.",
)
@click.option(
    "--map_vmin",
    type=float,
    default=None,
    help="Lower color limit for map_comparison.png/map_components.png and the "
    "posterior sample grid, in the map's "
    "NATIVE units (K_RJ, as written to map_mean.fits) -- converted internally to the plots' mK_RJ colorbar. "
    "Omit for the default symmetric auto-scale (+/- the largest |pixel| across the panels).",
)
@click.option(
    "--map_vmax",
    type=float,
    default=None,
    help="Upper color limit for the same three map plots, in native K_RJ. See --map_vmin.",
)
@click.option(
    "--prior_map_vmin",
    type=float,
    default=None,
    help="Lower color limit for map_prior_samples.png specifically, in native K_RJ (same convention "
    "as --map_vmin). Prior draws are typically much wider than the posterior/reconstructed map, so "
    "this is independent of --map_vmin/--map_vmax (which never apply to this plot). Omit for the "
    "default symmetric auto-scale (+/- the largest |pixel| across this plot's own panels).",
)
@click.option(
    "--prior_map_vmax",
    type=float,
    default=None,
    help="Upper color limit for map_prior_samples.png. See --prior_map_vmin.",
)
@click.option(
    "--std_vmax",
    type=float,
    default=None,
    help="Upper color limit for map_std.png (absolute std, linear scale, vmin fixed at 0), in native "
    "K_RJ (same convention as --map_vmin). Omit for the default: the 99.5th percentile of the std "
    "map's own pixels.",
)
@click.option(
    "--rel_std_cmap",
    default="afmhot",
    show_default=True,
    help="Colormap for the std-type map plots: map_relative_std*.png, map_std.png, and the "
    "rel-std row of map_components_relstd.png. The default 'afmhot' saturates to pure white at the "
    "top of its range, which can wash out against the white contour overlay -- 'inferno' is a "
    "similar dark-to-bright warm palette that tops out at a saturated yellow instead.",
)
@click.option(
    "--tod_width_scale",
    default=1.0,
    show_default=True,
    help="Widen the time-stream TOD figures by this factor (tod_components.png, "
    "tod_prior_components.png, tod_atmosphere_prior.png, detector_and_tods.png). Only the width "
    "grows -- every trace sub-plot keeps its height, so a longer scan can be stretched out without "
    "the panels getting taller. In detector_and_tods.png the detector-layout panel keeps its size "
    "and only the trace panel widens. The square power-spectrum figures are unaffected.",
)
@click.option(
    "--all_black",
    default=False,
    show_default=True,
    help="Hide the map plots' axis decorations for compositing onto a black slide/background: "
    "tick marks, tick labels and axis labels are drawn black (invisible against black, though the "
    "canvas itself stays transparent) so only the image, its white frame, the white overlay "
    "contours and the colorbar (with its own white ticks and label) remain visible. Implies "
    "--theme dark. Only affects the map figures, not the TOD/spectrum plots.",
)
@click.option(
    "--sample_ncols",
    default=4,
    show_default=True,
    help="Number of sample panels per row in map_prior_samples.png / map_posterior_samples.png. "
    "The grid grows downward in rows, so panels keep a readable width no matter how many samples "
    "a run stored (a 16-sample run stores 32 antithetic states).",
)
@click.option(
    "--ra_step_sec",
    default=5.0,
    show_default=True,
    help="Spacing (in seconds of time) between RA tick marks on the map plots' bottom x-axis -- "
    "increase for a wider field of view (e.g. MS0735) where 5s ticks would be too crowded.",
)
@click.option(
    "--tick_stride",
    default=1,
    show_default=True,
    help="Blanket multiplier on every map-plot axis's tick spacing at once (RA, Dec, and the "
    "secondary Delta-theta-arcmin offset axis) -- e.g. 2 shows only every second tick everywhere, "
    "without separately tuning --ra_step_sec and the (fixed) Dec/secondary steps.",
)
@click.option(
    "--fov_factor",
    type=float,
    default=1.0,
    show_default=True,
    help="Crop every map plot (map_comparison.png, map_components.png, map_pull.png, "
    "map_mean.png, map_smoothed.png, map_relative_std*.png, and the prior/posterior sample grids) to "
    "the central FRACTION of the full field of view in both x and y -- e.g. 0.5 shows only the center "
    "half of the FOV in each direction (a quarter of the area). The crop is applied to the reconstructed "
    "source map (and the --fits_map comparison map) right after it is computed, so downstream "
    "derived quantities (color-scale auto-limits, relative-uncertainty percentiles, beam smoothing, "
    "contours) are all computed on the cropped region, not the full FOV. Does not affect the fit itself "
    "or any TOD-space plots. 1.0 (default) shows the full FOV.",
)
def main(
    fits_dir,
    out_dir,
    theme,
    grid,
    subdir,
    no_tods,
    master_config,
    fits_map,
    extra_fits_map,
    fits_hdu,
    fits_scale,
    rel_vmax,
    map_vmin,
    map_vmax,
    prior_map_vmin,
    prior_map_vmax,
    std_vmax,
    rel_std_cmap,
    tod_width_scale,
    all_black,
    sample_ncols,
    ra_step_sec,
    tick_stride,
    fov_factor,
):
    """Regenerate talk_plots.py's figures from a save_fits.py export.

    Parameters
    ----------
    fits_dir : str
        save_fits.py output folder (``plot_<run>/save_fits``).
    out_dir : str or None
        Figure output folder; defaults to ``<fits_dir>/../<subdir>``.
    theme : str
        Tick/label/grid color scheme, ``"light"`` (black ticks) or ``"dark"``
        (white ticks). The figure/axes canvas is always transparent.
    grid : bool
        Whether to draw grid lines on the plots.
    rel_vmax : str
        Comma-separated upper color limits for ``map_relative_std.png``; one
        output file per value, ``"auto"`` scaling to the data's own maximum.
    map_vmin, map_vmax : float or None
        Explicit color limits for ``map_comparison.png``,
        ``map_components.png`` and the prior/posterior sample grids, given in
        the map's native K_RJ units and
        rescaled here to the plots' mK_RJ colorbar. ``None`` keeps each plot's
        default symmetric auto-scale.

    Side Effects
    ------------
    - Forces JAX onto the CPU (only used to redraw the prior noise realization).
    - Writes the figures; never touches the FITS files.
    """
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

    import jax
    from tqdm.auto import tqdm

    import maria_nifty as mani

    # The canvas is always transparent (a true alpha hole in the saved PNG); "theme"
    # only picks the tick/label/grid/contour color that contrasts with wherever the
    # PNG gets composited. "dark" reuses matplotlib's "dark_background" style (white
    # text/axes) with the fill overridden to none below; "light" is the default style
    # as-is. ``fg`` is the foreground color used for elements that must contrast with
    # the canvas (scatter edges, map contours), and ``grid_color`` is the grid line
    # color tuned for each theme. The mani.grid_plot / mani.image_plot helpers create
    # their own figures but inherit these rcParams.
    theme = str(theme).lower()
    all_black = bool(all_black)
    if all_black:
        # The whole point is decorations that vanish into a black canvas, which
        # needs the dark theme's white contours/colorbar to still show up.
        theme = "dark"
    if theme == "dark":
        plt.style.use("dark_background")
        fg = "white"
        grid_color = "0.25"
    else:
        plt.style.use("default")
        fg = "black"
        grid_color = "0.85"
    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["savefig.facecolor"] = "none"
    plt.rcParams["savefig.edgecolor"] = "none"
    plt.rcParams["savefig.transparent"] = True
    # Let the grid_plot/image_plot internal figures honor the --grid toggle.
    plt.rcParams["axes.grid"] = bool(grid)

    # Width multiplier for the time-stream figures; heights are never scaled.
    tod_w = max(0.1, float(tod_width_scale))

    def finish_map_fig(fig):
        """Apply --all_black to a finished map figure, just before saving.

        Colorbar axes are identified by matplotlib's own ``_colorbar``
        attribute (set for both the ``ax=`` and ``cax=`` construction paths)
        and left untouched, so the bar keeps its white ticks and label.
        Spines are also left alone, keeping the white frame around each image
        and any white overlay contours.

        The secondary top (offset-arcmin) axis that ``apply_sky_ticks`` adds
        is a ``SecondaryAxis`` living in the parent's ``child_axes``, *not* in
        ``fig.axes``, so it has to be recursed into explicitly -- walking
        ``fig.axes`` alone leaves its "Delta theta [arcmin]" label and ticks
        white.
        """
        if not all_black:
            return

        def blacken(ax):
            if hasattr(ax, "_colorbar"):
                return
            ax.tick_params(axis="both", which="both", colors="black", labelcolor="black")
            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            for lbl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
                lbl.set_color("black")
            for child in getattr(ax, "child_axes", ()):
                blacken(child)

        for ax in fig.axes:
            blacken(ax)

    def set_grid(ax, **line_kwargs):
        """Toggle an axis grid per --grid.

        When the grid is on, the line properties (color, linewidth) are applied;
        when off we call ``ax.grid(False)`` with no line properties -- passing
        them alongside ``False`` makes matplotlib warn and enable the grid anyway.
        """
        if grid:
            ax.grid(True, color=grid_color, **line_kwargs)
        else:
            ax.grid(False)

    # load the save_fits.py export: the fit's own config (for scan labels and
    # the PRNG key), the per-dataset TOD products and -- unless the fit was
    # atmosphere-only -- the map samples.
    config = mani.yaml_load(os.path.join(fits_dir, "config.yaml"))
    # Write paper figures into a dedicated subfolder of the run's plot
    # directory (e.g. plot_0/paper/ or plot_0/paper_black/) so they stay
    # separate from the normal plots and from other --subdir variants.
    pdir = out_dir or os.path.join(os.path.dirname(os.path.abspath(fits_dir)), subdir)
    verbose = config.get("verbose", True)

    datasets, tod_meta = load_tod_fits(fits_dir)
    rec = load_map_fits(fits_dir) if os.path.isfile(os.path.join(fits_dir, "map_samples.fits")) else None
    if verbose:
        n_samp = 0 if rec is None else rec.src_samples.shape[0]
        logger.info(f"\nLoaded {len(datasets)} dataset(s) and {n_samp} posterior map state(s) from {fits_dir}")

    def detector_tod_plot(detector, tod_data, mask=None, odir=".", fs=1.0, unit_label=""):
        """Plot detector coordinates and TOD traces side by side."""
        os.makedirs(odir, exist_ok=True)

        label_fs = 15.0
        tick_fs = 12.5

        coords = np.asarray(detector.coordinates, dtype=float)
        colors = np.asarray(np.arange(detector.full_size), dtype=float)

        tod_arr = np.asarray(tod_data, dtype=float)
        if mask is not None:
            coords = coords[mask]
            colors = colors[mask]
            tod_arr = tod_arr[mask]
        if tod_arr.ndim == 1:
            tod_arr = tod_arr[None, :]

        cmap = plt.colormaps.get_cmap("cmb")
        norm = mcolors.Normalize(
            vmin=float(colors.min()) if colors.size else 0.0, vmax=float(colors.max()) if colors.size else 1.0
        )

        # The detector panel is aspect-locked square, so widening the figure
        # alone would only add empty space beside it. Its cell is held at its
        # original 5 in and the whole increase goes to the trace panel, via
        # both the figure width and the width ratio.
        fig = plt.figure(figsize=(5.0 + 15.0 * tod_w, 5), constrained_layout=False)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 3.0 * tod_w], wspace=0.01)
        fig.subplots_adjust(wspace=0.01)

        ax_det = fig.add_subplot(gs[0, 0])
        ax_tod = fig.add_subplot(gs[0, 1])

        ax_det.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            cmap=cmap,
            norm=norm,
            s=88,
            linewidths=0.35,
            edgecolors=fg,
        )
        ax_det.set_aspect("equal", adjustable="box")
        ax_det.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / ARCMIN2RAD:.0f}"))
        ax_det.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / ARCMIN2RAD:.0f}"))
        ax_det.set_xlabel("Detector x [arcmin]", fontsize=label_fs)
        ax_det.set_ylabel("Detector y [arcmin]", fontsize=label_fs)
        ax_det.tick_params(axis="both", labelsize=tick_fs)
        set_grid(ax_det, linewidth=0.7)
        ax_det.set_axisbelow(True)
        # Colorbar removed per user request

        time_min = np.arange(tod_arr.shape[-1]) / fs / 60.0
        for idx, row in enumerate(tod_arr):
            ax_tod.plot(
                time_min,
                row,
                color=cmap(norm(colors[idx] if idx < colors.size else idx)),
                alpha=0.45,
                linewidth=0.9,
            )
        ax_tod.set_xlabel("Time [minutes]", fontsize=label_fs)
        ax_tod.set_ylabel(f"Observed data {unit_label}".strip(), fontsize=label_fs)
        ax_tod.yaxis.set_label_position("right")
        ax_tod.yaxis.tick_right()
        ax_tod.tick_params(axis="both", which="both", labelsize=tick_fs)
        ax_tod.tick_params(axis="y", which="both", left=False, labelleft=False, right=True, labelright=True)
        set_grid(ax_tod, linewidth=0.7)
        ax_tod.set_axisbelow(True)
        ax_tod.set_xmargin(0)
        half_step_min = 0.5 / fs / 60.0
        ax_tod.set_xlim(time_min[0] - half_step_min, time_min[-1] + half_step_min)

        fig.savefig(os.path.join(odir, "detector_and_tods.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def map_reconstruction_plot(rec, binmap=None, binmap_extent=None, center_phi_theta=None, odir=".", extra_maps=()):
        """Write the map-reconstruction figures for the run.

        ``rec`` is :func:`load_map_fits`'s namespace; ``binmap`` the external
        ``--fits_map`` reference (``None`` skips every comparison figure, i.e.
        all ``map_comparison*.png``/``map_diff*.png`` below). Figures saved to
        ``odir``:

        * ``map_comparison.png`` - reference map (left) and NIFTy source-model
          mean (right) side by side, sharing one symmetric (around 0) colorbar
          on the right. Y-labels only on the left panel, no titles.
        * ``map_comparison_reco_left.png`` - the same two panels with the order
          reversed (reconstruction left, reference right). Only written when a
          reference panel exists.
        * ``map_comparison_multi.png`` - reconstruction, reference map, then
          each of ``extra_maps`` (``(array, extent)`` pairs on the full,
          uncropped grid), all with the reconstruction-mean contour. Only
          written when ``extra_maps`` is non-empty.
        * ``map_comparison_full_fov.png`` - same as ``map_comparison.png``
          (panel order, white reconstruction-mean contour on both panels, like
          ``map_comparison_reco_left.png``), but always at the full field of
          view regardless of ``--fov_factor``.
        * ``map_diff.png`` - the beam-smoothed mean minus the reference map,
          with the white reconstruction-mean contour, in ``map_mean.png``'s
          single-panel layout and colour limits.
        * ``map_diff_full_fov.png`` - the same as ``map_diff.png``, but always
          at the full field of view regardless of ``--fov_factor``.
        * ``map_mean.png`` - the source-model mean alone, on the same colormap
          and colour limits as ``map_comparison.png`` but with single-panel
          axes matching ``map_relative_std.png``.
        * ``map_smoothed.png`` - the same source-model mean, additionally
          convolved with this run's own beam (``map_response.sigma``,
          converted to ``source_space``'s own pixel units) -- otherwise
          identical to ``map_mean.png`` (colormap, colour limits, layout).
        * ``map_std.png`` - absolute standard deviation of the source-model
          sum, in the same mK_RJ colorbar units as ``map_mean.png``, linear
          scale, single-panel axes matching ``map_relative_std.png``.
        * ``map_rel_std.png`` - relative standard deviation (std/|mean|) of the
          source-model sum, dimensionless, with its own colorbar.
        """
        os.makedirs(odir, exist_ok=True)

        source_space = rec.space
        src_mean, src_std = mean_std(rec.src_samples)
        # Kept alongside the (possibly --fov_factor-cropped) src_mean below,
        # for the full-FOV figures (map_comparison_full_fov.png,
        # map_diff_full_fov.png) that always show the full field of view
        # regardless of --fov_factor.
        full_src_mean = np.asarray(src_mean, dtype=float)
        full_src_extent = extent_arcmin(source_space)

        src_extent = extent_arcmin(source_space)
        src_mean, cropped_extent = crop_center_fov(src_mean, src_extent, fov_factor)
        src_std, _ = crop_center_fov(src_std, src_extent, fov_factor)
        src_extent = cropped_extent

        if binmap is not None and binmap_extent is None:
            binmap_extent = [0, np.asarray(binmap).shape[-1], 0, np.asarray(binmap).shape[-2]]
        full_binmap = None if binmap is None else np.asarray(binmap, dtype=float)
        full_binmap_extent = binmap_extent
        if binmap is not None:
            binmap, binmap_extent = crop_center_fov(binmap, binmap_extent, fov_factor)
        extra_maps = [crop_center_fov(np.asarray(arr, dtype=float), ext, fov_factor) for arr, ext in extra_maps]

        def label_axes(axs, extents, ylabel_only_left=True, tick_stride_override=None):
            """Apply arcmin extents and absolute-sky (RA/Dec) tick labels to grid_plot image axes.

            Matches the tick style of the standard (in-run) map plots via
            ``apply_sky_ticks``, and always disables the grid on these image
            panels regardless of the --grid setting (which only applies to the
            TOD-trace plots). ``tick_stride_override``, if given, replaces the
            outer --tick_stride for this call only.
            """
            ts = tick_stride if tick_stride_override is None else tick_stride_override
            for i, ax in enumerate(axs):
                if i >= len(extents) or not ax.images:
                    continue
                ext = [float(v) for v in np.asarray(extents[i], dtype=float).reshape(-1)[:4]]
                ax.images[-1].set_extent(ext)
                ax.set_xlim(ext[0], ext[1])
                ax.set_ylim(ext[2], ext[3])
                apply_sky_ticks(
                    ax,
                    center_phi_theta,
                    ylabel=(not ylabel_only_left) or i == 0,
                    ra_step_sec=ra_step_sec,
                    tick_stride=ts,
                )
                ax.grid(False)

        # Figure 1: external --fits_map reference and source-model mean side
        # by side, one symmetric colorbar. `binmap` is already in the model's
        # own axis0->x, axis1->y display convention (see load_fits_map).
        panels = []
        if binmap is not None:
            panels.append((np.asarray(binmap, dtype=float), binmap_extent))
        panels.append((np.asarray(src_mean, dtype=float), src_extent))

        signal_max = 1e-12
        for arr, _ in panels:
            finite = np.abs(arr[np.isfinite(arr)])
            if finite.size:
                signal_max = max(signal_max, float(np.max(finite)))

        cbar_scale, cbar_label = mani.signal_cbar_label(datasets[0]["tods"], target_unit="mK_RJ")
        cmp_arrays = [arr * cbar_scale for arr, _ in panels]
        cmp_extents = [ext for _, ext in panels]
        signal_max *= cbar_scale
        v_lo = -signal_max if map_vmin is None else map_vmin * cbar_scale
        v_hi = signal_max if map_vmax is None else map_vmax * cbar_scale

        # A single figure-level colorbar (rather than grid_plot's per-panel
        # divider-based one) -- every panel gets exactly the same width with
        # no hidden per-panel reservation, so the only gap between panels is
        # the real wspace below, not a "phantom" stacked colorbar allowance.
        def comparison_figure(
            arrays,
            extents,
            fname,
            contour_arr=None,
            contour_extent=None,
            tick_stride_override=None,
        ):
            """``contour_arr``/``contour_extent``, if given, overlay the same
            white reconstruction-mean contour (matching map_relative_std.png's
            levels/style) on every panel -- not just the panel that array
            itself belongs to. ``tick_stride_override``, if given, replaces
            the outer --tick_stride for this figure only.
            """
            n_panels = len(arrays)
            fig, axes = plt.subplots(
                1,
                n_panels,
                figsize=(7 * n_panels, 7),
                dpi=150,
                gridspec_kw={"wspace": 0.04},
            )
            axes = np.atleast_1d(axes)

            last_img = None
            for ax, arr in zip(axes, arrays):
                last_img = ax.imshow(arr.T, cmap="cmb", origin="lower", vmin=v_lo, vmax=v_hi)
                if contour_arr is not None:
                    ax.contour(
                        contour_arr.T,
                        origin="lower",
                        extent=contour_extent,
                        colors="white",
                        levels=[-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2],
                        linewidths=0.5,
                        linestyles="solid",
                    )

            label_axes(axes, extents, ylabel_only_left=True, tick_stride_override=tick_stride_override)
            fig.colorbar(
                last_img, ax=list(axes), location="right", fraction=0.046, pad=0.01, shrink=0.936, label=cbar_label
            )
            finish_map_fig(fig)
            fig.savefig(os.path.join(odir, fname), dpi=150, bbox_inches="tight")
            plt.close(fig)

        os.makedirs(odir, exist_ok=True)
        if len(cmp_arrays) > 1:
            comparison_figure(cmp_arrays, cmp_extents, "map_comparison.png")
        # Same panels in the opposite order, so the reconstruction leads and the
        # reference map follows. Only meaningful when there are two panels to
        # swap; cmp_arrays itself is left alone because the relative-std contour
        # below indexes it positionally.
        if len(cmp_arrays) > 1:
            comparison_figure(
                cmp_arrays[::-1],
                cmp_extents[::-1],
                "map_comparison_reco_left.png",
                contour_arr=cmp_arrays[-1],
                contour_extent=cmp_extents[-1],
            )

        # Reconstruction, reference map, then every extra reference map, all
        # on one colour scale with the reconstruction-mean contour on each.
        if len(cmp_arrays) > 1 and extra_maps:
            comparison_figure(
                [cmp_arrays[-1], cmp_arrays[0]] + [arr * cbar_scale for arr, _ in extra_maps],
                [cmp_extents[-1], cmp_extents[0]] + [ext for _, ext in extra_maps],
                "map_comparison_multi.png",
                contour_arr=cmp_arrays[-1],
                contour_extent=cmp_extents[-1],
            )

        # Same as map_comparison.png, but always at the full field of view
        # regardless of --fov_factor, with the white reconstruction-mean
        # contour overlaid on both panels (like map_comparison_reco_left.png).
        # Ticks halved (every second one) -- the full FOV packs about twice as
        # many round RA/Dec values into the same panel width.
        full_panels = []
        if full_binmap is not None:
            full_panels.append((full_binmap, full_binmap_extent))
        full_panels.append((full_src_mean, full_src_extent))
        full_cmp_arrays = [arr * cbar_scale for arr, _ in full_panels]
        full_cmp_extents = [ext for _, ext in full_panels]
        if len(full_cmp_arrays) > 1:
            comparison_figure(
                full_cmp_arrays,
                full_cmp_extents,
                "map_comparison_full_fov.png",
                contour_arr=full_cmp_arrays[-1],
                contour_extent=full_cmp_extents[-1],
                tick_stride_override=tick_stride * 2,
            )

        # Figure 1b: the reconstruction's mean on its own, with no reference
        # panel beside it -- same colormap and colour limits as the comparison
        # above (so the two are read on one scale), but the single-panel axes
        # and colorbar geometry of map_relative_std.png.
        fig_mean, ax_mean = plt.subplots(figsize=(7, 7), dpi=150)
        img_mean = ax_mean.imshow(
            (np.asarray(src_mean, dtype=float) * cbar_scale).T,
            cmap="cmb",
            origin="lower",
            extent=src_extent,
            vmin=v_lo,
            vmax=v_hi,
        )
        apply_sky_ticks(ax_mean, center_phi_theta, ra_step_sec=ra_step_sec, tick_stride=tick_stride)
        ax_mean.grid(False)
        attach_axes_colorbar(fig_mean, ax_mean, img_mean, cbar_label, fraction=0.046, pad=0.02)
        finish_map_fig(fig_mean)
        fig_mean.savefig(os.path.join(odir, "map_mean.png"), dpi=150, bbox_inches="tight")
        plt.close(fig_mean)

        # Figure 1c: same source-model mean, beam-smoothed -- this run's own
        # beam (map_response.sigma, exported as BEAMSIG in map pixel units;
        # the exported grid is the model's own map space, so no rescaling).
        if rec.beam_sigma > 0:
            from scipy.ndimage import gaussian_filter

            sigma_source_px = rec.beam_sigma
            smoothed_mean = gaussian_filter(np.asarray(src_mean, dtype=float), sigma_source_px)
            full_smoothed_mean = gaussian_filter(full_src_mean, sigma_source_px)
        else:
            smoothed_mean = np.asarray(src_mean, dtype=float)
            full_smoothed_mean = full_src_mean

        fig_smooth, ax_smooth = plt.subplots(figsize=(7, 7), dpi=150)
        img_smooth = ax_smooth.imshow(
            (smoothed_mean * cbar_scale).T,
            cmap="cmb",
            origin="lower",
            extent=src_extent,
            vmin=v_lo,
            vmax=v_hi,
        )
        apply_sky_ticks(ax_smooth, center_phi_theta, ra_step_sec=ra_step_sec, tick_stride=tick_stride)
        ax_smooth.grid(False)
        attach_axes_colorbar(fig_smooth, ax_smooth, img_smooth, cbar_label, fraction=0.046, pad=0.02)
        finish_map_fig(fig_smooth)
        fig_smooth.savefig(os.path.join(odir, "map_smoothed.png"), dpi=150, bbox_inches="tight")
        plt.close(fig_smooth)

        def diff_alone_figure(diff, diff_extent, contour_arr, contour_extent, stride, fname):
            """One difference map on its own, in map_mean.png's single-panel
            layout and colour limits, with the white reconstruction-mean
            contour of the comparison figures on top."""
            fig_diff, ax_diff = plt.subplots(figsize=(7, 7), dpi=150)
            img_diff = ax_diff.imshow(
                diff.T,
                cmap="cmb",
                origin="lower",
                extent=diff_extent,
                vmin=v_lo,
                vmax=v_hi,
            )
            ax_diff.contour(
                contour_arr.T,
                origin="lower",
                extent=contour_extent,
                colors="white",
                levels=[-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2],
                linewidths=0.5,
                linestyles="solid",
            )
            apply_sky_ticks(ax_diff, center_phi_theta, ra_step_sec=ra_step_sec, tick_stride=stride)
            ax_diff.grid(False)
            attach_axes_colorbar(fig_diff, ax_diff, img_diff, cbar_label, fraction=0.046, pad=0.02)
            finish_map_fig(fig_diff)
            fig_diff.savefig(os.path.join(odir, fname), dpi=150, bbox_inches="tight")
            plt.close(fig_diff)

        # Figure 1c-diff: beam-smoothed mean minus the reference map, on the
        # same colour scale -- the reference (--fits_map) map is already at the
        # instrument's resolution, so the smoothed mean is the like-for-like
        # one to subtract. Shows large-scale disagreements between the two.
        # At the --fov_factor-cropped field of view of the other map plots.
        if len(cmp_arrays) > 1 and cmp_arrays[0].shape == smoothed_mean.shape:
            diff_arr = smoothed_mean * cbar_scale - cmp_arrays[0]
            diff_alone_figure(diff_arr, src_extent, cmp_arrays[-1], cmp_extents[-1], tick_stride, "map_diff.png")

        # The same difference, always at the full field of view regardless of
        # --fov_factor (like map_comparison_full_fov.png), ticks halved for the
        # same reason.
        if len(full_cmp_arrays) > 1 and full_cmp_arrays[0].shape == full_smoothed_mean.shape:
            full_diff_arr = full_smoothed_mean * cbar_scale - full_cmp_arrays[0]
            diff_alone_figure(
                full_diff_arr,
                full_src_extent,
                full_cmp_arrays[-1],
                full_cmp_extents[-1],
                tick_stride * 2,
                "map_diff_full_fov.png",
            )

        # Figure 1d: absolute standard deviation of the source-model sum, in
        # the same mK_RJ colorbar units as map_mean.png. Unlike
        # map_relative_std.png (std/|mean|, log scale -- dividing by a
        # near-zero mean can blow up without bound) this is the std itself,
        # so a linear scale is enough; the ceiling is still a percentile
        # (not the raw max) so a few edge/no-coverage pixels don't wash out
        # the rest of the map.
        abs_std_scaled = np.abs(src_std) * cbar_scale
        finite_std = abs_std_scaled[np.isfinite(abs_std_scaled)]
        if std_vmax is None:
            std_vmax_plot = float(np.percentile(finite_std, 99.5)) if finite_std.size else 1.0
        else:
            std_vmax_plot = std_vmax * cbar_scale
        std_label = f"Std [{_format_unit_label('mK_RJ')}]"

        fig_std, ax_std = plt.subplots(figsize=(7, 7), dpi=150)
        img_std = ax_std.imshow(
            abs_std_scaled.T,
            cmap=rel_std_cmap,
            origin="lower",
            extent=src_extent,
            vmin=0.0,
            vmax=std_vmax_plot,
        )
        ax_std.contour(
            cmp_arrays[-1].T,
            origin="lower",
            extent=src_extent,
            colors="white",
            levels=[-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2],
            linewidths=0.5,
            linestyles="solid",
        )
        apply_sky_ticks(ax_std, center_phi_theta, ra_step_sec=ra_step_sec, tick_stride=tick_stride)
        ax_std.grid(False)
        attach_axes_colorbar(fig_std, ax_std, img_std, std_label, fraction=0.046, pad=0.02)
        finish_map_fig(fig_std)
        fig_std.savefig(os.path.join(odir, "map_std.png"), dpi=150, bbox_inches="tight")
        plt.close(fig_std)

        # Figure 2: relative standard deviation (std/|mean|) of the source-model
        # sum, log-scale, fixed vmax (so pixels at or above it all saturate to
        # the colormap's top instead of a few unusually-noisy in-disc pixels
        # stretching the scale past it).
        rel_std_raw, no_support, rel_vmin = compute_rel_std_stats(src_mean, src_std)

        def rel_std_log_plot(vmax, fname):
            rel_std = np.where(no_support, vmax, rel_std_raw)
            fig_rel, ax_rel = plt.subplots(figsize=(7, 7), dpi=150)
            img_rel = ax_rel.imshow(
                rel_std.T,
                cmap=rel_std_cmap,
                origin="lower",
                extent=src_extent,
                norm=mcolors.LogNorm(vmin=max(rel_vmin, 1e-6), vmax=vmax),
            )
            ax_rel.contour(
                cmp_arrays[-1].T,
                origin="lower",
                extent=src_extent,
                colors="white",
                levels=[-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2],
                linewidths=0.5,
                linestyles="solid",
            )
            apply_sky_ticks(ax_rel, center_phi_theta, ra_step_sec=ra_step_sec, tick_stride=tick_stride)
            ax_rel.grid(False)
            attach_axes_colorbar(fig_rel, ax_rel, img_rel, "Relative uncertainty", fraction=0.046, pad=0.02)
            finish_map_fig(fig_rel)
            fig_rel.savefig(os.path.join(odir, fname), dpi=150, bbox_inches="tight")
            plt.close(fig_rel)

        # One file per requested ceiling; "auto" uses the data's own max.
        finite_raw = rel_std_raw[np.isfinite(rel_std_raw) & (rel_std_raw > 0)]
        data_vmax = float(finite_raw.max()) if finite_raw.size else 1.0
        for tok in str(rel_vmax).split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.lower() in ("auto", "none"):
                v, fname = data_vmax, "map_relative_std_auto.png"
            else:
                v = float(tok)
                fname = "map_relative_std.png" if v == 1.25 else f"map_relative_std_vmax{tok}.png"
            if v <= max(rel_vmin, 1e-6):
                logger.warning(f"Skipping {fname}: vmax={v:.3g} is not above vmin={rel_vmin:.3g}.")
                continue
            rel_std_log_plot(v, fname)

    def compute_two_source_maps(rec):
        """Common per-pixel statistics shared by the two-source map plots.

        Only meaningful when the map model has exactly two source
        components -- callers check ``len(result["source_models"]) == 2``
        themselves and skip/return early otherwise. Returns the full
        (summed) map's mean/std, each individual source component's mean/std,
        the shared extent, and the mK_RJ colorbar scale/label.
        """
        source_space = rec.space
        src_mean, src_std = mean_std(rec.src_samples)
        comp_mean_std = [mean_std(c) for c in rec.comp_samples]
        comp_means = [m for m, _ in comp_mean_std]
        comp_stds = [s for _, s in comp_mean_std]
        cbar_scale, cbar_label = mani.signal_cbar_label(datasets[0]["tods"], target_unit="mK_RJ")

        src_extent = extent_arcmin(source_space)
        src_mean, cropped_extent = crop_center_fov(src_mean, src_extent, fov_factor)
        src_std, _ = crop_center_fov(src_std, src_extent, fov_factor)
        comp_means = [crop_center_fov(c, src_extent, fov_factor)[0] for c in comp_means]
        comp_stds = [crop_center_fov(c, src_extent, fov_factor)[0] for c in comp_stds]
        src_extent = cropped_extent

        return {
            "source_models": rec.names,
            "source_space": source_space,
            "src_extent": src_extent,
            "src_mean": src_mean,
            "src_std": src_std,
            "comp_means": comp_means,
            "comp_stds": comp_stds,
            "cbar_scale": cbar_scale,
            "cbar_label": cbar_label,
        }

    def map_pull_plot(rec, center_phi_theta=None, odir="."):
        """Pull (posterior mean / std) map of the full (summed) source model.

        A significance-style map: how many posterior standard deviations the
        mean is away from zero at each pixel. Only produced when the map
        model has exactly two source components (matches
        map_components.png's scope).
        """
        computed = compute_two_source_maps(rec)
        if len(computed["source_models"]) != 2:
            return
        os.makedirs(odir, exist_ok=True)

        src_mean, src_std = computed["src_mean"], computed["src_std"]
        src_extent = computed["src_extent"]
        cbar_scale = computed["cbar_scale"]

        abs_mean = np.abs(src_mean)
        abs_std_raw = np.abs(src_std)
        eps = max(1e-12, 1e-3 * float(np.percentile(abs_mean, 5)))
        std_eps = max(1e-12, 1e-3 * float(np.percentile(abs_std_raw, 5)))
        pull = src_mean / np.maximum(abs_std_raw, std_eps)
        # Where both mean and std are effectively zero (no model support at
        # all, e.g. far outside the fitted source) the ratio is a
        # meaningless noise-floor artifact rather than a real significance --
        # force it to 0 (no significance) instead.
        pull = np.where((abs_mean <= eps) & (abs_std_raw <= std_eps), 0.0, pull)

        finite = pull[np.isfinite(pull)]
        pull_max = max(1e-6, float(np.percentile(np.abs(finite), 99.5))) if finite.size else 1.0
        contour_arr = src_mean * cbar_scale

        fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
        img = ax.imshow(pull.T, cmap="cmb", origin="lower", extent=src_extent, vmin=-pull_max, vmax=pull_max)
        ax.contour(
            contour_arr.T,
            origin="lower",
            extent=src_extent,
            colors="white",
            levels=[-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2],
            linewidths=0.5,
            linestyles="solid",
        )
        apply_sky_ticks(ax, center_phi_theta, ra_step_sec=ra_step_sec, tick_stride=tick_stride)
        ax.grid(False)
        attach_axes_colorbar(fig, ax, img, "Pull (mean / std)", fraction=0.046, pad=0.02)
        finish_map_fig(fig)
        fig.savefig(os.path.join(odir, "map_pull.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def map_full_and_components_plot(rec, center_phi_theta=None, odir="."):
        """Full (summed) map and its two source components, side by side (1x3
        panels + a shared colorbar column), all on the same color scale --
        taken from the full map's own range (not the max across all three
        panels), so a component may saturate if its own peak exceeds it.

        Only produced when the map model has exactly two source components.
        """
        computed = compute_two_source_maps(rec)
        if len(computed["source_models"]) != 2:
            return
        os.makedirs(odir, exist_ok=True)

        src_extent = computed["src_extent"]
        cbar_scale, cbar_label = computed["cbar_scale"], computed["cbar_label"]
        full_mean = computed["src_mean"] * cbar_scale
        comp_means = [c * cbar_scale for c in computed["comp_means"]]

        finite = np.abs(full_mean[np.isfinite(full_mean)])
        signal_max = max(1e-12, float(np.max(finite))) if finite.size else 1e-12
        v_lo = -signal_max if map_vmin is None else map_vmin * cbar_scale
        v_hi = signal_max if map_vmax is None else map_vmax * cbar_scale

        fig, gs = make_square_grid_figure(nrows=1, n_img_cols=3)
        ax_full = fig.add_subplot(gs[0, 0])
        ax_c0 = fig.add_subplot(gs[0, 1])
        ax_c1 = fig.add_subplot(gs[0, 2])

        def show(ax, arr):
            ax.set_xlim(src_extent[0], src_extent[1])
            ax.set_ylim(src_extent[2], src_extent[3])
            return ax.imshow(arr.T, cmap="cmb", origin="lower", extent=src_extent, vmin=v_lo, vmax=v_hi)

        show(ax_full, full_mean)
        show(ax_c0, comp_means[0])
        img_c1 = show(ax_c1, comp_means[1])

        apply_sky_ticks(
            ax_full,
            center_phi_theta,
            xlabel=True,
            ylabel=True,
            secondary=True,
            ra_step_sec=ra_step_sec,
            tick_stride=tick_stride,
        )
        apply_sky_ticks(
            ax_c0,
            center_phi_theta,
            xlabel=True,
            ylabel=False,
            secondary=True,
            ra_step_sec=ra_step_sec,
            tick_stride=tick_stride,
        )
        apply_sky_ticks(
            ax_c1,
            center_phi_theta,
            xlabel=True,
            ylabel=False,
            secondary=True,
            ra_step_sec=ra_step_sec,
            tick_stride=tick_stride,
        )
        for ax in (ax_full, ax_c0, ax_c1):
            ax.grid(False)

        attach_cell_colorbar(fig, gs, gs[0, 3], img_c1, cbar_label)

        finish_map_fig(fig)
        fig.savefig(os.path.join(odir, "map_components.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def map_components_relstd_plot(rec, center_phi_theta=None, odir="."):
        """Full (summed) map and its two source components in the top row
        (identical panels/colorbar to ``map_components.png``), and each
        component's own relative standard deviation (std/|mean|, log scale,
        vmax fixed at 1.25) in the bottom row, sharing a second colorbar --
        2x3 grid with an empty bottom-left cell.

        Only produced when the map model has exactly two source components.
        """
        computed = compute_two_source_maps(rec)
        if len(computed["source_models"]) != 2:
            return
        os.makedirs(odir, exist_ok=True)

        src_extent = computed["src_extent"]
        cbar_scale, cbar_label = computed["cbar_scale"], computed["cbar_label"]
        full_mean = computed["src_mean"] * cbar_scale
        comp0_mean_raw, comp1_mean_raw = computed["comp_means"]
        comp0_std_raw, comp1_std_raw = computed["comp_stds"]
        comp_means = [comp0_mean_raw * cbar_scale, comp1_mean_raw * cbar_scale]

        finite = np.abs(full_mean[np.isfinite(full_mean)])
        signal_max = max(1e-12, float(np.max(finite))) if finite.size else 1e-12
        v_lo = -signal_max if map_vmin is None else map_vmin * cbar_scale
        v_hi = signal_max if map_vmax is None else map_vmax * cbar_scale

        rel_vmax = 1.25
        rel0_raw, no_support0, rel_vmin0 = compute_rel_std_stats(comp0_mean_raw, comp0_std_raw)
        rel1_raw, no_support1, rel_vmin1 = compute_rel_std_stats(comp1_mean_raw, comp1_std_raw)
        rel0 = np.where(no_support0, rel_vmax, rel0_raw)
        rel1 = np.where(no_support1, rel_vmax, rel1_raw)
        rel_vmin_candidates = [v for v in (rel_vmin0, rel_vmin1) if v > 0]
        rel_vmin = min(rel_vmin_candidates) if rel_vmin_candidates else 0

        fig, gs = make_square_grid_figure(nrows=2, n_img_cols=3)
        ax_full = fig.add_subplot(gs[0, 0])
        ax_c0 = fig.add_subplot(gs[0, 1])
        ax_c1 = fig.add_subplot(gs[0, 2])
        ax_empty = fig.add_subplot(gs[1, 0])
        ax_r0 = fig.add_subplot(gs[1, 1])
        ax_r1 = fig.add_subplot(gs[1, 2])
        ax_empty.axis("off")

        def show(ax, arr, vmin, vmax, cmap="cmb", norm=None):
            ax.set_xlim(src_extent[0], src_extent[1])
            ax.set_ylim(src_extent[2], src_extent[3])
            if norm is not None:
                return ax.imshow(arr.T, cmap=cmap, origin="lower", extent=src_extent, norm=norm)
            return ax.imshow(arr.T, cmap=cmap, origin="lower", extent=src_extent, vmin=vmin, vmax=vmax)

        show(ax_full, full_mean, v_lo, v_hi)
        show(ax_c0, comp_means[0], v_lo, v_hi)
        img_c1 = show(ax_c1, comp_means[1], v_lo, v_hi)

        rel_lognorm = mcolors.LogNorm(vmin=max(rel_vmin, 1e-6), vmax=rel_vmax)
        show(ax_r0, rel0, rel_vmin, rel_vmax, cmap=rel_std_cmap, norm=rel_lognorm)
        img_r1 = show(ax_r1, rel1, rel_vmin, rel_vmax, cmap=rel_std_cmap, norm=rel_lognorm)

        # No contour on ax_r1 (the compact source's own rel-std panel) --
        # it's almost entirely saturated to the colormap's light top color
        # outside its tiny real support, where the contour rings add little.
        ax_r0.contour(
            comp_means[0].T,
            origin="lower",
            extent=src_extent,
            colors="white",
            levels=[-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2],
            linewidths=0.5,
            linestyles="solid",
        )

        apply_sky_ticks(
            ax_full,
            center_phi_theta,
            xlabel=True,
            ylabel=True,
            secondary=True,
            ra_step_sec=ra_step_sec,
            tick_stride=tick_stride,
        )
        apply_sky_ticks(
            ax_c0,
            center_phi_theta,
            xlabel=False,
            ylabel=False,
            secondary=True,
            ra_step_sec=ra_step_sec,
            tick_stride=tick_stride,
        )
        apply_sky_ticks(
            ax_c1,
            center_phi_theta,
            xlabel=False,
            ylabel=False,
            secondary=True,
            ra_step_sec=ra_step_sec,
            tick_stride=tick_stride,
        )
        apply_sky_ticks(
            ax_r0,
            center_phi_theta,
            xlabel=True,
            ylabel=True,
            secondary=False,
            ra_step_sec=ra_step_sec,
            tick_stride=tick_stride,
        )
        apply_sky_ticks(
            ax_r1,
            center_phi_theta,
            xlabel=True,
            ylabel=False,
            secondary=False,
            ra_step_sec=ra_step_sec,
            tick_stride=tick_stride,
        )
        for ax in (ax_full, ax_c0, ax_c1, ax_r0, ax_r1):
            ax.grid(False)

        attach_cell_colorbar(fig, gs, gs[0, 3], img_c1, cbar_label)
        attach_cell_colorbar(fig, gs, gs[1, 3], img_r1, "Relative uncertainty")

        finish_map_fig(fig)
        fig.savefig(os.path.join(odir, "map_components_relstd.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def tod_components_plot(
        detector,
        d,
        mask=None,
        odir=".",
        fs=1.0,
        unit_label="",
        map_scale=1.0,
        map_unit_label="",
    ):
        """Plot stacked TOD reconstruction components for one dataset.

        ``d`` is one of :func:`load_tod_fits`'s dataset dicts (posterior-mean
        TODs). Rows, top to bottom: data, atmosphere reconstruction, map
        reconstruction, residual (model - data). Each panel overlays all
        detector traces colored by detector index, matching the TOD panel of
        ``detector_and_tods.png``. No axis labels; x-ticks only on the bottom
        panel.
        """
        os.makedirs(odir, exist_ok=True)

        def to_2d(arr):
            arr = np.asarray(arr, dtype=float)
            return arr[None, :] if arr.ndim == 1 else arr

        data = to_2d(d["tod_data"])
        rows = [("Observed data", data, unit_label)]
        if d["atm"] is not None:
            rows.append(("Reconstructed atmosphere", to_2d(d["atm"]), unit_label))
        if d["map"] is not None:
            rows.append(("Reconstructed map response", to_2d(d["map"]) * map_scale, map_unit_label))
        rows.append(("Residuals (model - data)", to_2d(d["model"]) - data, unit_label))
        rows = [(f"{name} {ul}".strip(), comp) for name, comp, ul in rows]

        if mask is not None:
            rows = [(name, comp[mask]) for name, comp in rows]

        colors = np.asarray(np.arange(detector.full_size), dtype=float)
        if mask is not None:
            colors = colors[mask]
        cmap = plt.colormaps.get_cmap("cmb")
        norm = mcolors.Normalize(
            vmin=float(colors.min()) if colors.size else 0.0,
            vmax=float(colors.max()) if colors.size else 1.0,
        )

        n_t = data.shape[-1]
        n = len(rows)
        # Each subplot keeps the ~4:1 width:height ratio of the full
        # detector_and_tods.png figure (20 x 5), scaled to 80%.
        fig, axes = plt.subplots(n, 1, figsize=(16 * tod_w, 4.0 * n), sharex=True, constrained_layout=True)
        if n == 1:
            axes = [axes]

        label_fs = 15.0
        time_min = np.arange(n_t) / fs / 60.0
        for ax, (name, comp) in zip(axes, rows):
            for idx, row in enumerate(comp):
                color_val = colors[idx] if idx < colors.size else idx
                ax.plot(time_min, row, color=cmap(norm(color_val)), alpha=0.45, linewidth=0.9)
            ax.set_ylabel(name, fontsize=label_fs)
            ax.tick_params(axis="both", which="both", labelsize=12.5)
            ax.set_xmargin(0)
            half_step_min = 0.5 / fs / 60.0
            ax.set_xlim(time_min[0] - half_step_min, time_min[-1] + half_step_min)
            set_grid(ax, linewidth=0.7)
            ax.set_axisbelow(True)

        # x-ticks and x-label only on the lowest subplot.
        for ax in axes[:-1]:
            ax.tick_params(axis="x", labelbottom=False)
        axes[-1].set_xlabel("Time [minutes]", fontsize=label_fs)

        fig.savefig(os.path.join(odir, "tod_components.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def tod_prior_components_plot(
        detector,
        d,
        noise,
        mask=None,
        n_steps=10000,
        odir=".",
        fs=1.0,
        unit_label="",
    ):
        """Plot stacked TOD components evaluated at a random *prior* sample.

        Mirrors ``tod_components_plot`` but replaces the optimized samples with a
        single random draw from the model prior, and swaps the two data-driven
        rows for purely generative ones. Rows, top to bottom: model data
        (atmosphere + map + noise prior), atmosphere prior, map prior, and a
        random noise realization drawn with the config ``noise_std``. The bottom
        three rows sum to the top row, so the figure shows what the generative
        model produces before any fit. Styling matches ``tod_components.png``.

        Only the first ``n_steps`` time steps are drawn (the components are
        truncated along the time axis), while the figure size is kept at the
        full-plot dimensions so the aspect ratio is unchanged. The prior draw
        is ``d["prior_atm"]``/``d["prior_map"]`` from ``tod_mean.fits``.
        """
        os.makedirs(odir, exist_ok=True)

        def to_2d(arr):
            arr = np.asarray(arr, dtype=float)
            return arr[None, :] if arr.ndim == 1 else arr

        noise = to_2d(noise)
        has_atm = d["prior_atm"] is not None
        has_map = d["prior_map"] is not None
        atm = to_2d(d["prior_atm"]) if has_atm else np.zeros_like(noise)
        map_resp = to_2d(d["prior_map"]) if has_map else np.zeros_like(noise)
        model_data = atm + map_resp + noise

        rows = [("Model data", model_data)]
        if has_atm:
            rows.append(("Atmosphere prior", atm))
        if has_map:
            rows.append(("Map prior", map_resp))
        rows.append(("Noise prior", noise))
        rows = [(f"{name} {unit_label}".strip(), comp) for name, comp in rows]

        if mask is not None:
            rows = [(name, comp[mask]) for name, comp in rows]

        # Truncate to the first n_steps time steps; the figure size is kept at
        # the full-plot dimensions below so the aspect ratio is unchanged.
        n_t = min(int(n_steps), model_data.shape[-1])
        rows = [(name, comp[:, :n_t]) for name, comp in rows]

        colors = np.asarray(np.arange(detector.full_size), dtype=float)
        if mask is not None:
            colors = colors[mask]
        cmap = plt.colormaps.get_cmap("cmb")
        norm = mcolors.Normalize(
            vmin=float(colors.min()) if colors.size else 0.0,
            vmax=float(colors.max()) if colors.size else 1.0,
        )

        n = len(rows)
        fig, axes = plt.subplots(n, 1, figsize=(16 * tod_w, 4.0 * n), sharex=True, constrained_layout=True)
        if n == 1:
            axes = [axes]

        label_fs = 15.0
        time_min = np.arange(n_t) / fs / 60.0
        for ax, (name, comp) in zip(axes, rows):
            for idx, row in enumerate(comp):
                color_val = colors[idx] if idx < colors.size else idx
                ax.plot(time_min, row, color=cmap(norm(color_val)), alpha=0.45, linewidth=0.9)
            ax.set_ylabel(name, fontsize=label_fs)
            ax.tick_params(axis="both", which="both", labelsize=12.5)
            ax.set_xmargin(0)
            half_step_min = 0.5 / fs / 60.0
            ax.set_xlim(time_min[0] - half_step_min, time_min[-1] + half_step_min)
            set_grid(ax, linewidth=0.7)
            ax.set_axisbelow(True)

        # x-ticks and x-label only on the lowest subplot.
        for ax in axes[:-1]:
            ax.tick_params(axis="x", labelbottom=False)
        axes[-1].set_xlabel("Time [minutes]", fontsize=label_fs)

        fig.savefig(os.path.join(odir, "tod_prior_components.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def map_samples_plot(
        rec,
        sample_cube,
        center_phi_theta=None,
        odir=".",
        fname="map_prior_samples.png",
        ncols=None,
        use_shared_vminmax=True,
        own_vmin=None,
        own_vmax=None,
    ):
        """Plot individual map-sample draws side by side (N rows x ``ncols``
        cols + a shared colorbar column spanning all rows).

        The layout is driven by the number of *columns* rather than rows, so
        panels stay a readable width as the sample count grows: a run with 32
        stored samples would otherwise be laid out 16-per-row. ``ncols``
        defaults to --sample_ncols.

        Each panel is one ``(nx, ny)`` frame of ``sample_cube`` (native
        units; ``rec.prior_src`` for prior draws or ``rec.src_samples`` for
        the run's actual posterior samples), filled
        row-major (left to right, top to bottom) and sharing one symmetric
        (around 0) colorbar on the right. Mirrors the source-model panel of
        ``map_comparison.png``.

        ``use_shared_vminmax`` (default True) scales this plot with the
        outer --map_vmin/--map_vmax, matching map_comparison.png and the
        other map plots. If False, --map_vmin/--map_vmax never apply here --
        instead ``own_vmin``/``own_vmax`` (native K_RJ, same convention) are
        used if given, else a symmetric auto-scale to this call's own sample
        arrays. Prior draws are typically much wider than the
        posterior/reconstructed map, so map_prior_samples.png passes False
        (with --prior_map_vmin/--prior_map_vmax as own_vmin/own_vmax) to keep
        its scale independent of the shared one.
        """
        os.makedirs(odir, exist_ok=True)

        src_extent = extent_arcmin(rec.space)
        cbar_scale, cbar_label = mani.signal_cbar_label(datasets[0]["tods"], target_unit="mK_RJ")
        sample_maps = [np.asarray(arr, dtype=float) * cbar_scale for arr in sample_cube]
        cropped_extent = src_extent
        cropped_maps = []
        for arr in sample_maps:
            arr, cropped_extent = crop_center_fov(arr, src_extent, fov_factor)
            cropped_maps.append(arr)
        sample_maps, src_extent = cropped_maps, cropped_extent

        signal_max = 1e-12
        for arr in sample_maps:
            finite = np.abs(arr[np.isfinite(arr)])
            if finite.size:
                signal_max = max(signal_max, float(np.max(finite)))
        if use_shared_vminmax:
            v_lo = -signal_max if map_vmin is None else map_vmin * cbar_scale
            v_hi = signal_max if map_vmax is None else map_vmax * cbar_scale
        else:
            v_lo = -signal_max if own_vmin is None else own_vmin * cbar_scale
            v_hi = signal_max if own_vmax is None else own_vmax * cbar_scale

        n_panels = len(sample_maps)
        n_cols = max(1, min(int(sample_ncols if ncols is None else ncols), n_panels))
        nrows = int(np.ceil(n_panels / n_cols))

        # Which row is the bottom-most *occupied* one in each column (the
        # last row may be a partial one short of n_cols panels) -- that row
        # gets the x label/ticks, matching map_components_relstd_plot's
        # empty-cell-aware convention.
        bottom_row_for_col = {}
        for idx in range(n_panels):
            r, c = divmod(idx, n_cols)
            bottom_row_for_col[c] = r

        fig, gs = make_square_grid_figure(nrows=nrows, n_img_cols=n_cols)

        last_img = None
        for idx, arr in enumerate(sample_maps):
            r, c = divmod(idx, n_cols)
            ax = fig.add_subplot(gs[r, c])
            ax.set_xlim(src_extent[0], src_extent[1])
            ax.set_ylim(src_extent[2], src_extent[3])
            last_img = ax.imshow(arr.T, cmap="cmb", origin="lower", extent=src_extent, vmin=v_lo, vmax=v_hi)
            apply_sky_ticks(
                ax,
                center_phi_theta,
                xlabel=(r == bottom_row_for_col[c]),
                ylabel=(c == 0),
                secondary=(r == 0),
                ra_step_sec=ra_step_sec,
                tick_stride=tick_stride,
            )
            ax.grid(False)

        for idx in range(n_panels, nrows * n_cols):
            r, c = divmod(idx, n_cols)
            fig.add_subplot(gs[r, c]).axis("off")

        attach_cell_colorbar(fig, gs, gs[:, n_cols], last_img, cbar_label)

        finish_map_fig(fig)
        fig.savefig(os.path.join(odir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def tod_power_spectra_plot(d, mask=None, tods=None, fs=1.0, odir="."):
        """Plot the non-binned TOD power spectra (right half of tod_overview_all).

        Computes the unbinned mean periodogram (Tukey-windowed) for the
        reconstructed atmosphere, map response, and residual (model - data),
        and draws all three on a single log-log axis. This reproduces the
        right-hand power-spectrum panel of ``tod_overview_all.png`` as a
        standalone figure ``tod_power_spectra.png``.
        """
        os.makedirs(odir, exist_ok=True)

        if tods is not None and hasattr(tods, "fs"):
            fs = float(tods.fs)
        unit_label = f" {psd_unit_label(getattr(tods, 'units', None))}".rstrip()

        def to_2d(arr):
            arr = np.asarray(arr, dtype=float)
            return arr[None, :] if arr.ndim == 1 else arr

        data = to_2d(d["tod_data"])
        full_rec = to_2d(d["model"])
        atm_rec = np.zeros_like(data) if d["atm"] is None else to_2d(d["atm"])
        map_rec = np.zeros_like(data) if d["map"] is None else to_2d(d["map"])
        residual = full_rec - data

        components = [atm_rec, map_rec, residual]
        labels = ["Atmosphere", "Map", "Residuals"]
        if mask is not None:
            components = [comp[mask] for comp in components]

        def power_spectrum_unbinned(comp):
            f, ps = sp.signal.periodogram(comp, fs=fs, axis=-1, window="tukey")
            ps_mean = np.mean(ps, axis=0)
            use = (f > 0) & np.isfinite(ps_mean) & (ps_mean > 0)
            return f[use], ps_mean[use]

        # Keep atmosphere cool, map warm, residual mid (matches tod_overview).
        cmap = plt.colormaps.get_cmap("cmb")
        comp_colors = [cmap(0.1), cmap(0.9), cmap(0.6)]

        label_fs = 15.0
        tick_fs = 12.5

        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        ps_ymins, ps_ymaxs = [], []
        for comp, label, comp_color in zip(components, labels, comp_colors):
            f_plot, ps_plot = power_spectrum_unbinned(comp)
            if f_plot.size > 0 and ps_plot.size > 0:
                ax.plot(f_plot, ps_plot, lw=1.6, color=comp_color, label=label)
                ps_ymins.append(float(np.nanmin(ps_plot)))
                ps_ymaxs.append(float(np.nanmax(ps_plot)))

        ax.set_xlabel("Frequency [Hz]", fontsize=label_fs)
        ax.set_ylabel(f"Power spectral density{unit_label}", fontsize=label_fs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis="both", which="both", labelsize=tick_fs)
        if ps_ymins and ps_ymaxs:
            y_min, y_max = min(ps_ymins), max(ps_ymaxs)
            if np.isfinite(y_min) and np.isfinite(y_max) and y_min > 0 and y_max > y_min:
                ax.set_ylim(y_min, y_max)
        set_grid(ax, which="both", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="best", fontsize=tick_fs)

        fig.savefig(os.path.join(odir, "tod_power_spectra.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def tod_power_spectra_combined_plot(datasets, binmap=None, odir="."):
        """Combined unbinned TOD power spectra, fused across every dataset at once.

        Mirrors ``tod_power_spectra_plot`` (reconstructed atmosphere/map/
        residual, unbinned Tukey-windowed periodogram) but instead of one
        figure per dataset, concatenates every dataset's rows -- across all
        detectors, truncated to the shortest common time length so datasets
        that differ by a sample or two can still be stacked together -- into
        a single combined periodogram. Writes ``spectra_components.png``:
        the observed data's own average spectrum (drawn as a thick,
        translucent band behind the rest, since it sits above and mostly
        overlaps the atmosphere reconstruction) plus the reconstructed
        atmosphere/map/residual components.

        If ``binmap`` is given (the ``--fits_map`` reference, already
        resampled onto the reconstruction's own map grid), it is pushed through
        each dataset's exported map-response pointing (``Dkk_PNT``, not its beam
        convolution -- minkasi's own map-making has no beam/PSF step of its
        own, so its map already carries the real physical beam and applying
        this run's Gaussian beam model on top would smooth it twice) to get a
        directly comparable TOD-space prediction. Its periodogram is added as
        an extra "Minkasi map" curve (no uncertainty band, since ``binmap``
        carries no posterior) when available.
        """
        os.makedirs(odir, exist_ok=True)

        def to_2d(arr):
            arr = np.asarray(arr, dtype=float)
            return arr[None, :] if arr.ndim == 1 else arr

        binmap_jnp = None if binmap is None else np.nan_to_num(np.asarray(binmap, dtype=float), nan=0.0)

        data_list, atm_list, map_list, resid_list, minkasi_list = [], [], [], [], []
        for d in datasets:
            data = to_2d(d["tod_data"])
            atm_rec = np.zeros_like(data) if d["atm"] is None else to_2d(d["atm"])
            map_rec = np.zeros_like(data) if d["map"] is None else to_2d(d["map"])
            residual = to_2d(d["model"]) - data

            minkasi_rec = None
            pnt = d.get("pointing")
            if binmap_jnp is not None and pnt is not None:
                # Minkasi's own map-making (see mapper.py's run_minkasi_mapper)
                # is a pure pointing solve -- tod2map/map2tod plus a noise
                # weighting, no beam/PSF kernel -- so its map already carries
                # the real, physical telescope beam and nothing more.
                # Re-applying this run's own Gaussian beam model on top would
                # smooth it a second time, so this projects it through the
                # pointing only (sigma_pixels=0).
                pointing = None
                if pnt.kind == "pointing_matrix":
                    pointing = build_pointing_matrix(pnt.coords, binmap_jnp.shape)
                minkasi_rec = to_2d(
                    map2tods(
                        binmap_jnp,
                        pnt.coords,
                        0.0,
                        order=pnt.order,
                        kind=pnt.kind,
                        mode=pnt.mode,
                        cval=pnt.cval,
                        pointing=pointing,
                    )
                )

            mask = d.get("mask")
            if mask is not None:
                data, atm_rec, map_rec, residual = data[mask], atm_rec[mask], map_rec[mask], residual[mask]
                if minkasi_rec is not None:
                    minkasi_rec = minkasi_rec[mask]

            data_list.append(data)
            atm_list.append(atm_rec)
            map_list.append(map_rec)
            resid_list.append(residual)
            if minkasi_rec is not None:
                minkasi_list.append(minkasi_rec)

        t_common = min(arr.shape[-1] for arr in data_list)
        data_cat = np.concatenate([a[:, :t_common] for a in data_list], axis=0)
        atm_cat = np.concatenate([a[:, :t_common] for a in atm_list], axis=0)
        map_cat = np.concatenate([a[:, :t_common] for a in map_list], axis=0)
        resid_cat = np.concatenate([a[:, :t_common] for a in resid_list], axis=0)
        minkasi_cat = (
            np.concatenate([a[:, :t_common] for a in minkasi_list], axis=0)
            if len(minkasi_list) == len(datasets)
            else None
        )

        fs = float(datasets[0]["tods"].fs) if datasets[0].get("tods") is not None else 1.0
        unit_label = f" {psd_unit_label(getattr(datasets[0].get('tods'), 'units', None))}".rstrip()

        def power_spectrum_unbinned(comp):
            f, ps = sp.signal.periodogram(comp, fs=fs, axis=-1, window="tukey")
            ps_mean = np.mean(ps, axis=0)
            use = (f > 0) & np.isfinite(ps_mean) & (ps_mean > 0)
            return f[use], ps_mean[use]

        cmap = plt.colormaps.get_cmap("cmb")
        label_fs = 15.0
        tick_fs = 12.5

        def _make_figure(components, labels, colors, fname, background_index=None):
            """``background_index``, if given, draws that one curve as a thick,
            translucent band drawn first (behind, low zorder) instead of a
            normal line -- visible as a soft presence without competing with
            the other (fully opaque, higher zorder) curves drawn on top of it.
            """
            fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
            ps_ymins, ps_ymaxs = [], []
            freq_mins, freq_maxs = [], []
            order = list(range(len(components)))
            if background_index is not None:
                other = [i for i in order if i != background_index]
                order = [background_index] + other
            for i in order:
                comp, label, color = components[i], labels[i], colors[i]
                f_plot, ps_plot = power_spectrum_unbinned(comp)
                if f_plot.size > 0 and ps_plot.size > 0:
                    if i == background_index:
                        ax.plot(f_plot, ps_plot, lw=4.5, color=color, alpha=0.7, label=label, zorder=1)
                    else:
                        ax.plot(f_plot, ps_plot, lw=1.6, color=color, label=label, zorder=2)
                    ps_ymins.append(float(np.nanmin(ps_plot)))
                    ps_ymaxs.append(float(np.nanmax(ps_plot)))
                    freq_mins.append(float(np.min(f_plot)))
                    freq_maxs.append(float(np.max(f_plot)))

            ax.set_xlabel("Frequency [Hz]", fontsize=label_fs)
            ax.set_ylabel(f"Power spectral density{unit_label}", fontsize=label_fs)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.tick_params(axis="both", which="both", labelsize=tick_fs)
            if ps_ymins and ps_ymaxs:
                y_min, y_max = min(ps_ymins), max(ps_ymaxs)
                if np.isfinite(y_min) and np.isfinite(y_max) and y_min > 0 and y_max > y_min:
                    ax.set_ylim(y_min, y_max)
            if freq_mins and freq_maxs:
                # Matches spectra_data.png's approach: the actual data-driven
                # frequency range, not matplotlib's own auto-margined one.
                ax.set_xlim(min(freq_mins), max(freq_maxs))
            # Matches spectra_data.png's grid style (dotted major gridlines,
            # fixed gray, semi-transparent) rather than the solid major+minor
            # grid used elsewhere in this file.
            if grid:
                ax.grid(True, which="major", axis="both", linestyle=":", color="0.55", alpha=0.5)
            else:
                ax.grid(False)
            ax.set_axisbelow(True)
            ax.legend(loc="best", fontsize=tick_fs)

            fig.savefig(os.path.join(odir, fname), dpi=150, bbox_inches="tight")
            plt.close(fig)

        # All components in one figure: observed data (as a thick translucent
        # band behind the rest), atmosphere/map/residual, and -- when a
        # --fits_map reference is given -- its own TOD-space prediction
        # (pointing only, no uncertainty band since it is a single fixed map,
        # not a posterior).
        components = [data_cat, atm_cat, map_cat, resid_cat]
        labels = ["Observed data", "Atmosphere", "Map", "Residuals"]
        colors = [fg, cmap(0.1), cmap(0.9), cmap(0.6)]
        if minkasi_cat is not None:
            components.append(minkasi_cat)
            labels.append("Minkasi map")
            colors.append("#4CA990")
        _make_figure(components, labels, colors, "spectra_components.png", background_index=0)

    def atmosphere_prior_plot(datasets, odir="."):
        """Single plot of one prior-sample atmosphere draw for one dataset --
        the "Atmosphere prior" row of tod_prior_components.png, but drawn
        once standalone rather than once per dataset.

        Uses whichever dataset has the longest time axis (the representative,
        full-duration one) instead of combining every dataset -- mixing
        datasets of very different lengths into one figure made for a
        visibly two-toned color split at the point where the shorter ones
        ran out.

        The same exported prior draw (``Dkk_PRIOR``) as the per-dataset
        tod_prior_components.png panels.
        """
        os.makedirs(odir, exist_ok=True)

        def to_2d(arr):
            arr = np.asarray(arr, dtype=float)
            return arr[None, :] if arr.ndim == 1 else arr

        d = max(datasets, key=lambda d: np.asarray(d["tod_data"]).shape[-1])
        if d["prior_atm"] is None:
            return

        atm = to_2d(d["prior_atm"])
        mask = d.get("mask")
        if mask is not None:
            atm = atm[mask]

        fs = float(d["tods"].fs) if d.get("tods") is not None else 1.0
        tod_units = getattr(d.get("tods"), "units", None)
        unit_label = f"[{_format_unit_label(tod_units)}]" if tod_units else ""

        colors = np.arange(atm.shape[0], dtype=float)
        cmap = plt.colormaps.get_cmap("cmb")
        norm = mcolors.Normalize(vmin=0.0, vmax=max(1.0, float(colors.max())))

        label_fs = 15.0
        n_t = atm.shape[-1]
        time_min = np.arange(n_t) / fs / 60.0

        fig, ax = plt.subplots(figsize=(16 * tod_w, 4.0), constrained_layout=True)
        for idx, row in enumerate(atm):
            ax.plot(time_min, row, color=cmap(norm(colors[idx])), alpha=0.45, linewidth=0.9)
        ax.set_ylabel(f"Atmosphere prior {unit_label}".strip(), fontsize=label_fs)
        ax.set_xlabel("Time [minutes]", fontsize=label_fs)
        ax.tick_params(axis="both", which="both", labelsize=12.5)
        ax.set_xmargin(0)
        half_step_min = 0.5 / fs / 60.0
        ax.set_xlim(time_min[0] - half_step_min, time_min[-1] + half_step_min)
        set_grid(ax, linewidth=0.7)
        ax.set_axisbelow(True)

        fig.savefig(os.path.join(odir, "tod_atmosphere_prior.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Total observation time across all datasets, summed over each dataset's
    # time axis and reported as hours:min:sec.
    def fmt_hms(seconds):
        """Format a duration in seconds as ``H:MM:SS``."""
        seconds = int(round(max(0.0, seconds)))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:d}:{minutes:02d}:{secs:02d}"

    total_obs_s = sum(d["duration_s"] for d in datasets)
    logger.info(
        f"\nTotal observation time over {len(datasets)} dataset(s): "
        f"{fmt_hms(total_obs_s)} (h:min:sec, {total_obs_s:.1f} s total)"
    )

    # plotting
    logger.info("\nFinal plotting:")

    # map reconstruction plots for the run
    pdir_map = os.path.join(pdir, "map")

    binmap = None
    if rec is None:
        logger.info("No map_samples.fits (atmosphere-only fit): skipping the map plots.")
    else:
        center_phi_theta = rec.center_phi_theta
        binmap_extent, extra_maps = None, []
        if fits_map:
            # Resample the external reference map(s) onto the exact same grid
            # (shape, FOV, center) as the NIFTy reconstruction, so the panels
            # share one extent/set of sky ticks.
            def load_reference(path):
                return load_fits_map(
                    path,
                    shape=np.asarray(rec.space.shape, dtype=int),
                    fov_arcmin=np.asarray(rec.space.fov, dtype=float) / ARCMIN2RAD,
                    center_arcmin=np.asarray(rec.space.cen, dtype=float).reshape(-1)[:2] / ARCMIN2RAD,
                    center_phi_theta=center_phi_theta,
                    hdu=fits_hdu,
                    data_scale=fits_scale,
                )

            binmap = load_reference(fits_map)
            binmap_extent = np.asarray(extent_arcmin(rec.space))
            extra_maps = [(load_reference(path), binmap_extent) for path in extra_fits_map]
            if verbose:
                logger.info(f"\nLoaded reference FITS map from {fits_map} (hdu {fits_hdu})")
        else:
            if extra_fits_map:
                logger.warning("--extra_fits_map needs --fits_map; ignoring it.")
            logger.info("No --fits_map given: skipping the map comparison plots.")

        map_reconstruction_plot(
            rec,
            binmap=binmap,
            binmap_extent=binmap_extent,
            center_phi_theta=center_phi_theta,
            odir=pdir_map,
            extra_maps=extra_maps,
        )

        # Pull map, full-map + component row, and component rel-std -- all
        # no-ops unless the map model has exactly two source components.
        map_pull_plot(rec, center_phi_theta=center_phi_theta, odir=pdir_map)
        map_full_and_components_plot(rec, center_phi_theta=center_phi_theta, odir=pdir_map)
        map_components_relstd_plot(rec, center_phi_theta=center_phi_theta, odir=pdir_map)

        # Actual posterior samples, laid out --sample_ncols per row with one
        # shared colorbar spanning all rows.
        map_samples_plot(
            rec,
            rec.src_samples,
            center_phi_theta=center_phi_theta,
            odir=pdir_map,
            fname="map_posterior_samples.png",
        )

        # Map-prior samples: the draws save_fits.py rejection-sampled (peak
        # source-map pixel within its --prior_vmin_mk/--prior_vmax_mk), on
        # their own color scale (--prior_map_vmin/--prior_map_vmax).
        if rec.prior_src is not None and len(rec.prior_src):
            map_samples_plot(
                rec,
                rec.prior_src,
                center_phi_theta=center_phi_theta,
                odir=pdir_map,
                fname="map_prior_samples.png",
                use_shared_vminmax=False,
                own_vmin=prior_map_vmin,
                own_vmax=prior_map_vmax,
            )

    # Combined (binned + unbinned) mean power spectrum across all datasets,
    # one line per dataset -- an overview of the raw TOD data itself (matches
    # plot_tods.py's pre-fit quick look). Legend labels use each dataset's
    # scan number (e.g. "Scan 03-s34") rather than a bare index, matching the
    # fits_label convention used elsewhere -- but a run's saved config often
    # points to an intermediate-stage file (e.g. ".../remove_jumps/d1.fits")
    # instead of the original raw scan filename, in which case fits_label has
    # no real scan suffix to find and just returns the generic "d1" stem.
    # When --master_config points to the original pipeline config, look the
    # real filename up there instead (matched by dataset key); otherwise fall
    # back to a plain sequential "Scan {i+1}" (detected by the absence of a
    # "-" in the extracted suffix, since every real scan identifier looks
    # like "03-s34").
    data_keys = [k for k in config["data"] if k != "scan_center"]
    master_data = {}
    if master_config:
        master_cfg = mani.yaml_load(master_config)
        master_data = master_cfg.get("data", {})

    def _scan_label(i, key):
        fname = master_data[key]["fname"] if key in master_data else config["data"][key]["fname"]
        suffix = mani.fits_label(fname).split("_")[-1]
        return f"Scan {suffix}" if "-" in suffix else f"Scan {i + 1}"

    scan_labels = [_scan_label(i, k) for i, k in enumerate(data_keys)]
    mani.tod_spectrum_plot(
        datasets, odir=pdir, labels=scan_labels, grid=grid, raw_fname="spectra_data.png", write_binned=False
    )

    # Reconstruction-component power spectra (data, atmosphere, map,
    # residual, and -- with --fits_map -- the reference map), fused across
    # every dataset into one periodogram rather than plotted per-dataset.
    tod_power_spectra_combined_plot(datasets, binmap=binmap, odir=pdir)

    # A single prior-sample atmosphere draw, fused across every dataset into
    # one figure rather than plotted per-dataset.
    atmosphere_prior_plot(datasets, odir=pdir)

    for i, d in enumerate(tqdm(datasets, desc="Plot final results", unit="dataset", disable=no_tods)):  # noqa: B007
        if no_tods:
            continue
        if i < 10:
            pdir_tod_i = os.path.join(pdir, f"tod_{i + 1:02d}")
            fs = float(d["tods"].fs)
            tod_units = d["tods"].units
            unit_label = f"[{_format_unit_label(tod_units)}]" if tod_units else ""
            map_scale = mani.signal_cbar_label(d["tods"], target_unit="mK_RJ")[0] if tod_units else 1.0
            map_unit_label = "[mK$_{\\mathrm{RJ}}$]" if tod_units else ""
            detector_tod_plot(
                d["detector"], d["tod_data"], mask=d.get("mask"), odir=pdir_tod_i, fs=fs, unit_label=unit_label
            )
            tod_components_plot(
                d["detector"],
                d,
                mask=d.get("mask"),
                odir=pdir_tod_i,
                fs=fs,
                unit_label=unit_label,
                map_scale=map_scale,
                map_unit_label=map_unit_label,
            )
            tod_power_spectra_plot(d, mask=d.get("mask"), tods=d.get("tods"), odir=pdir_tod_i)

            # The exported prior draw of the model TODs plus a matching noise
            # realization (config noise_std), keyed exactly as in
            # talk_plots.py: model data = atmosphere + map + noise priors.
            _, _, noise_key = jax.random.split(jax.random.PRNGKey(tod_meta.key), 3)
            noise = tod_meta.noise_std * jax.random.normal(noise_key, d["tod_data"].shape)
            tod_prior_components_plot(
                d["detector"],
                d,
                noise,
                mask=d.get("mask"),
                odir=pdir_tod_i,
                fs=fs,
                unit_label=unit_label,
            )


if __name__ == "__main__":
    main()
