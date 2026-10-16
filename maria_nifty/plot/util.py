"""Shared plotting utilities for figure layout and colorbars."""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

ARCMIN2RAD = np.pi / (180.0 * 60.0)


def _aligned_ticks(lo, hi, step, margin_frac=0.03):
    """All multiples of `step` strictly inside `(lo, hi)`.

    Ticks land on round multiples of `step` (e.g. step=5 -> ..., 25, 30, 35,
    40, ...), not merely evenly spaced -- so ticks stay put (just gaining or
    losing one at the edges) as the view range changes, instead of jumping
    to a new arbitrary offset. `margin_frac` insets the usable range from
    each edge first (as a small fraction of the whole `(lo, hi)` span), so no
    tick ends up sitting exactly at (or right next to) the axis
    limits/corners, while still keeping ticks that merely land a bit close to
    an edge with clear room to spare.
    """
    margin = margin_frac * (hi - lo)
    lo_m, hi_m = lo + margin, hi - margin
    first = np.ceil(lo_m / step - 1e-9) * step
    ticks = []
    v = first
    while v <= hi_m + 1e-9:
        ticks.append(v)
        v += step
    return ticks


def _cascade_sky_labels(angles, *, fields, sep, unit, pattern, alwayssign=False, superscript=False, anchor="first"):
    """Format a sequence of exact angles into abbreviated sexagesimal labels.

    Each tick shows a sexagesimal component (e.g. hour, or degree) only on
    the anchor tick or when it differs from its neighbor towards the anchor;
    the last (finest) component is always shown -- e.g. RA ticks "13h47m40s",
    "20s", "00s", "46m40s" (hour/minute repeated across ticks are dropped,
    seconds always shown), matching standard sky-map tick conventions.

    Parameters
    ----------
    angles : sequence of astropy.coordinates.Angle
        The (already "nice"/round) absolute RA or Dec value for each tick,
        in axis order (i.e. the order the corresponding ticks appear along
        the axis, e.g. left-to-right or bottom-to-top).
    fields, sep, unit :
        Passed to ``Angle.to_string`` (``fields=3`` + hourangle for RA,
        ``fields=2`` + deg for Dec, i.e. no arcsec).
    pattern : str
        Regex (matching ``Angle.to_string``'s output for the given
        ``fields``/``sep``) with one capture group per component, used to
        split the full label into its individual components.
    superscript : bool, optional
        Render each component's unit (``sep``) raised, as mathtext (e.g.
        "13$^\\mathrm{h}$47$^\\mathrm{m}$31$^\\mathrm{s}$") instead of plain
        inline text -- matches the reference MUSTANG-2 map-plot style.
    anchor : {"first", "last", "second"}, optional
        Which entry of `angles` always gets the fully-spelled-out label --
        "first" (default) matches how `angles` is ordered; "last" anchors the
        last entry instead (e.g. for the y-axis, where the top-most tick
        should be the fully-spelled-out one); "second" anchors the second
        entry (e.g. for the x-axis, where the rightmost tick is often crowded
        against the plot edge/secondary axis, so the second-from-edge tick
        carries the full label instead). Cascading then runs outward from the
        anchor in both directions -- towards the far end normally, and
        (for "second") the single remaining entry before the anchor compares
        directly against it too.
    """
    import re

    all_components = []
    for ang in angles:
        s = ang.to_string(unit=unit, sep=sep, precision=0, fields=fields, pad=True, alwayssign=alwayssign)
        m = re.match(pattern, s)
        all_components.append(m.groups())

    n = len(all_components)
    anchor_idx = {"first": 0, "last": n - 1, "second": min(1, n - 1)}[anchor]

    labels = [None] * n

    def emit(i, prev):
        comps = all_components[i]
        show_from = len(comps) - 1
        for k in range(len(comps)):
            if prev is None or comps[k] != prev[k]:
                show_from = k
                break
        shown = zip(comps[show_from:], sep[show_from:])
        if superscript:
            labels[i] = "$" + "".join(f"{c}^{{\\mathrm{{{u}}}}}" for c, u in shown) + "$"
        else:
            labels[i] = "".join(f"{c}{u}" for c, u in shown)
        return comps

    anchor_comps = emit(anchor_idx, None)
    prev = anchor_comps
    for i in range(anchor_idx + 1, n):
        prev = emit(i, prev)
    prev = anchor_comps
    for i in range(anchor_idx - 1, -1, -1):
        prev = emit(i, prev)

    return labels


def apply_sky_ticks(
    ax,
    center_phi_theta,
    xlabel=True,
    ylabel=True,
    secondary=True,
    ra_step_sec=5.0,
    dec_step_arcmin=1.0,
    secondary_step_arcmin=1.0,
    fontsize_scale=1.0,
    tick_stride=1,
):
    """Relabel an offset-arcmin map axes with absolute (RA, Dec) sky ticks.

    Unlike a naive "convert whatever tick the default locator picked", tick
    *locations* are chosen backwards from round absolute sky values: for each
    axis, the current view range is converted to absolute RA (seconds of
    time) / Dec (arcmin), ticks are placed on round multiples of a fixed step
    (see :func:`_aligned_ticks` -- however many land within view, not a
    target count), and each resulting round value is converted back to the
    local-offset-arcmin position it actually sits at (via ``maria``'s
    ``phi_theta_to_offsets``, the exact inverse of
    ``unjitted_offsets_to_phi_theta`` -- the same pair
    :func:`maria_nifty.fits.map2fits` uses). So every label is an exact
    value, not a rounded-for-display approximation of whatever coordinate a
    "nice local arcmin offset" happened to land on.

    ``ra_step_sec``/``dec_step_arcmin`` (default 5s / 1') are the tick
    spacing directly -- tune to whatever grid density you want (e.g. 2 or 3
    arcmin for a sparser Dec grid). ``tick_stride`` is a blanket multiplier on
    top of all three step sizes (RA, Dec, and the secondary offset axis) --
    e.g. ``tick_stride=2`` shows every second tick everywhere at once, without
    having to separately double each step argument. Ticks are inset from the view range's
    edges so none sit at the corners. RA ticks show hour/minute/second
    (arcsec never dropped, units raised as "13$^h$47$^m$31$^s$"); Dec ticks
    show degree/arcmin only (no arcsec). When ``secondary``, a twin top
    x-axis is added showing the original offset scale as "Delta theta
    [arcmin]" (tick spacing fixed to ``secondary_step_arcmin``, default 1',
    same "however many fit" rule as the primary ticks), matching the
    reference MUSTANG-2 map-plot style (absolute RA/Dec on the primary axes,
    local offset on a secondary top axis).

    A no-op if ``center_phi_theta`` is ``None`` (e.g. no dataset was
    available to derive an absolute reference point from).
    """
    if center_phi_theta is None:
        return
    cphi0, ctheta0 = center_phi_theta

    ra_step_sec = ra_step_sec * tick_stride
    dec_step_arcmin = dec_step_arcmin * tick_stride
    secondary_step_arcmin = secondary_step_arcmin * tick_stride

    from astropy import units as u
    from astropy.coordinates import Angle
    from maria.coords.transforms import phi_theta_to_offsets, unjitted_offsets_to_phi_theta

    def _forward(x_arcmin, y_arcmin):
        phi, theta = unjitted_offsets_to_phi_theta(
            np.array([x_arcmin * ARCMIN2RAD, y_arcmin * ARCMIN2RAD]), cphi0, ctheta0
        )
        return float(phi), float(theta)

    def _inverse(phi, theta):
        xy = np.asarray(phi_theta_to_offsets(np.array([phi, theta]), cphi0, ctheta0))
        return float(xy[0]) / ARCMIN2RAD, float(xy[1]) / ARCMIN2RAD

    # RA (x-axis): view range -> absolute RA (seconds of time) -> ticks
    # aligned to round multiples of ra_step_sec -> back to the exact
    # local-offset position each one sits at (at the panel's own Dec center,
    # i.e. y = 0).
    x0, x1 = ax.get_xlim()
    ra0, _ = _forward(x0, 0.0)
    ra1, _ = _forward(x1, 0.0)
    ra_sec_lo, ra_sec_hi = sorted(np.degrees([ra0, ra1]) / 15.0 * 3600.0)
    ra_ticks_sec = _aligned_ticks(ra_sec_lo, ra_sec_hi, ra_step_sec)
    ra_angles = [Angle(sec / 3600.0 * 15.0, unit=u.deg) for sec in ra_ticks_sec]
    x_ticks = [_inverse(ang.rad, ctheta0)[0] for ang in ra_angles]
    # anchor="first": the rightmost tick (first in ra_angles' ascending-RA
    # order, since RA increases right-to-left on screen) carries the fully
    # spelled-out hh mm ss label.
    x_labels = _cascade_sky_labels(
        ra_angles,
        fields=3,
        sep=("h", "m", "s"),
        unit=u.hourangle,
        pattern=r"([+-]?\d+)h(\d+)m(\d+)s",
        superscript=True,
        anchor="first",
    )

    # Dec (y-axis): same idea, in arcmin, at the panel's own RA center (x = 0).
    y0, y1 = ax.get_ylim()
    _, dec0 = _forward(0.0, y0)
    _, dec1 = _forward(0.0, y1)
    dec_arcmin_lo, dec_arcmin_hi = sorted(np.degrees([dec0, dec1]) * 60.0)
    dec_ticks_arcmin = _aligned_ticks(dec_arcmin_lo, dec_arcmin_hi, dec_step_arcmin)
    dec_angles = [Angle(am / 60.0, unit=u.deg) for am in dec_ticks_arcmin]
    y_ticks = [_inverse(cphi0, ang.rad)[1] for ang in dec_angles]
    y_labels = _cascade_sky_labels(
        dec_angles,
        fields=2,
        sep=("\N{DEGREE SIGN}", "'"),
        unit=u.deg,
        pattern=r"([+-]?\d+)\N{DEGREE SIGN}(\d+)'",
        alwayssign=True,
        anchor="last",
    )

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    # Tick *value* labels (not just the axis title) are gated by xlabel/ylabel
    # too -- a shared-axis panel (e.g. a non-left column in a row of images)
    # otherwise still shows redundant tick numbers with no title to explain them.
    ax.set_xticklabels(x_labels if xlabel else [])
    ax.set_yticklabels(y_labels if ylabel else [])

    tick_fontsize = (
        plt.rcParams["xtick.labelsize"]
        if isinstance(plt.rcParams["xtick.labelsize"], int | float)
        else plt.rcParams["font.size"]
    )
    label_fontsize = (
        plt.rcParams["axes.labelsize"]
        if isinstance(plt.rcParams["axes.labelsize"], int | float)
        else plt.rcParams["font.size"]
    )
    ax.tick_params(axis="both", labelsize=tick_fontsize * fontsize_scale)

    if xlabel:
        ax.set_xlabel("Right ascension (J2000)", fontsize=label_fontsize * fontsize_scale)
    if ylabel:
        ax.set_ylabel("Declination (J2000)", labelpad=-6, fontsize=label_fontsize * fontsize_scale)

    if secondary:
        secax = ax.secondary_xaxis("top")
        secax.set_xlabel(r"$\Delta\theta$ [arcmin]", fontsize=label_fontsize * fontsize_scale)
        # Inset from the view edges (like the primary RA/Dec ticks) instead of
        # a blind MultipleLocator -- otherwise a tick lands exactly at the
        # edge whenever the range happens to be an exact multiple of the
        # step (e.g. a +/-3amin view with a 1amin step), crowding the corner.
        secax.set_xticks(_aligned_ticks(min(x0, x1), max(x0, x1), secondary_step_arcmin))
        secax.tick_params(axis="x", labelsize=tick_fontsize * fontsize_scale)

    # Local-offset +x increases with +RA (this pipeline's own tangent-plane
    # convention), but the astronomical display standard shows RA increasing
    # right-to-left (mirrored -- matches maria's own native plots, which
    # encode this via a negative CDELT1 in their WCS header). Reversing the
    # axis *direction* here (not the tick values, not the image data) is
    # what keeps every already-placed tick/image pixel attached to its real
    # position while flipping which screen side reads as increasing RA.
    ax.set_xlim(x1, x0)


_SI_PREFIX_SCALE = {
    "p": 1e-12,
    "n": 1e-9,
    "\N{MICRO SIGN}": 1e-6,
    "u": 1e-6,
    "m": 1e-3,
    "c": 1e-2,
    "": 1.0,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
}


def _split_unit_prefix(unit):
    """Split e.g. "mK_RJ" into (1e-3, "K_RJ") -- an SI-prefix scale and the
    remaining base unit. Tries longer prefixes first (there are none >1
    char here, but keeps this robust to future additions); falls back to
    (1.0, unit) unchanged if no recognized prefix is found.
    """
    for prefix in sorted(_SI_PREFIX_SCALE, key=len, reverse=True):
        if prefix and unit.startswith(prefix):
            return _SI_PREFIX_SCALE[prefix], unit[len(prefix) :]
    return 1.0, unit


def _format_unit_label(unit):
    """Render e.g. "mK_RJ" as "mK$_{\\mathrm{RJ}}$" -- an underscore-separated
    suffix becomes a lowered mathtext subscript instead of a literal
    underscore; units without one (e.g. "Jy/beam") pass through unchanged.
    """
    prefix, sep, suffix = unit.partition("_")
    return f"{prefix}$_{{\\mathrm{{{suffix}}}}}$" if sep else unit


def psd_unit_label(unit):
    """Render a TOD unit as a power-spectral-density axis label, e.g.
    ``"K_RJ"`` -> ``"[K$_{\\mathrm{RJ}}^2$/Hz]"``. Returns ``""`` for a
    falsy ``unit``.
    """
    if not unit:
        return ""
    base = _format_unit_label(unit)
    squared = f"{base[:-1]}^2$" if base.endswith("$") else f"{base}$^2$"
    return f"[{squared}/Hz]"


def signal_cbar_label(tods, target_unit=None):
    """A ``(scale, label)`` colorbar pair from a TOD's native units.

    E.g. for a TOD with ``tods.units == "K_RJ"``: ``(1.0, "signal [K$_{RJ}$]")``
    by default, or ``(1000.0, "signal [mK$_{RJ}$]")`` if
    ``target_unit="mK_RJ"`` -- `scale` is meant to multiply the plotted map
    data (native -> target unit) before display; `label` goes on the
    colorbar, with an underscore-separated suffix (e.g. the "RJ" in "K_RJ")
    rendered as a lowered subscript. `target_unit` must share the same base
    unit as the TOD's native one (only an SI prefix may differ), e.g. can't
    convert "K_RJ" to "Jy/beam" this way.

    Returns ``(1.0, None)`` (no scaling, no label) if `tods` carries no
    `units` attribute (e.g. simulated data without a calibrated physical
    unit) and no `target_unit` override was given.
    """
    native = getattr(tods, "units", None)
    if target_unit is None:
        return 1.0, (f"Signal [{_format_unit_label(native)}]" if native else None)
    if native is None:
        return 1.0, f"Signal [{_format_unit_label(target_unit)}]"

    native_scale, native_base = _split_unit_prefix(native)
    target_scale, target_base = _split_unit_prefix(target_unit)
    if native_base != target_base:
        raise ValueError(f"Cannot convert {native!r} to {target_unit!r} (different base units).")
    return native_scale / target_scale, f"Signal [{_format_unit_label(target_unit)}]"


def plot_figure(
    fname=None,
):
    """Save the current figure to `fname` or display it interactively."""
    if fname != "":
        odir = os.path.dirname(fname)
        odir = "." if odir == "" else odir
        os.makedirs(odir, exist_ok=True)
        if not fname.endswith(".png"):
            fname += ".png"
        plt.savefig(fname)
    else:
        plt.show()
    plt.close()


def set_cbar(
    axes,
    image,
    cbar=True,
    loc="right",
    size="2.5%",
    pad="2.5%",
    label=None,
    labelpad=3,
    labelloc="center",
    n_ticks=5,
):
    """Attach a colorbar to an axes and configure tick placement."""
    # A log-normed image needs its own log-spaced tick locator (matplotlib
    # already picks one by default) -- forcing MaxNLocator on it collapses
    # ticks to just one or two nonsensical values.
    is_log = isinstance(getattr(image, "norm", None), LogNorm)

    div = make_axes_locatable(axes)
    cax_bottom = div.append_axes("bottom", size=size, pad=pad)
    cax_right = div.append_axes("right", size=size, pad=pad)
    cax_left = div.append_axes("left", size=size, pad=pad)
    # A colorbar is a continuous scale, not a data axes -- gridlines on it
    # never mean anything, but they still inherit rcParams["axes.grid"] (e.g.
    # from a --grid CLI flag) like any other new Axes unless disabled here.
    for cax in (cax_bottom, cax_right, cax_left):
        cax.grid(False)

    if cbar and loc == "bottom":
        cbar = plt.colorbar(image, cax=cax_bottom, orientation="horizontal")
        if not is_log:
            cbar.ax.xaxis.set_major_locator(MaxNLocator(n_ticks))
        rot = 0
    else:
        cax_bottom.set_visible(False)

    if cbar and loc == "right":
        cbar = plt.colorbar(image, cax=cax_right)
        if not is_log:
            cbar.ax.yaxis.set_major_locator(MaxNLocator(n_ticks))
        rot = 90
    else:
        cax_right.set_visible(False)

    if cbar and loc == "left":
        cbar = plt.colorbar(image, cax=cax_left)
        if not is_log:
            cbar.ax.yaxis.set_major_locator(MaxNLocator(n_ticks))
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        rot = 90
    else:
        cax_left.set_visible(False)

    if cbar and label:
        cbar.set_label(label, rotation=rot, labelpad=labelpad, loc=labelloc)


def rows_and_cols(
    nums,
    rows=None,
    cols=None,
):
    """Determine a near-square subplot layout for a given panel count."""
    match (rows, cols):
        case (None, None):
            rows = int(np.ceil(np.sqrt(nums)))
            cols = int(np.ceil(nums / rows))
        case (_, None):
            rows = min(rows, nums)
            cols = int(np.ceil(nums / rows))
        case _:
            cols = min(cols, nums)
            rows = int(np.ceil(nums / cols))
    return rows, cols


def to_shape(
    input,
    shape=None,
    rows=None,
    cols=None,
    default=-1,
    dtype=None,
    max_axes=2,
    flip=False,
):
    """Broadcast or reshape an input into a requested subplot-friendly shape."""
    array = np.array(input, dtype=object)

    match (array.ndim, max_axes):
        case (0, 0):
            return array
        case (0, _):
            array = array.reshape((1,))
        case (_, 0):
            raise ValueError("Cannot reshape array to have zero axes.")
        case (m, n) if m < n:
            array = array.reshape((-1,))
        case (n, m) if n == m:
            array = array.reshape((1,) + array.shape)
        case (n, m) if n > m:
            array = array.reshape((-1,) + array.shape[-m:])

    nums = array.shape[0]
    if shape is None:
        shape = rows_and_cols(nums, rows, cols)
    size = int(np.prod(shape))

    try:
        array = np.broadcast_to(array, shape + array.shape[1:])

    except ValueError:
        if nums < size:
            default = array[-1] if default == -1 else default
            if array.ndim == 1:
                for _i in range(size - nums):
                    array = np.append(array, default)
            else:
                array = np.append(array, np.zeros((size - nums,) + array.shape[1:]), axis=0)
        else:
            array = array[:size]
        array = array.reshape(shape + array.shape[1:])

    if flip:
        array = array.transpose(1, 0, *range(2, array.ndim))

    if dtype:
        array = array.astype(dtype)

    return array
