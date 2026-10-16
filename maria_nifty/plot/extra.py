"""High-level diagnostic plots for detectors, TODs, and reconstructions."""

import matplotlib.pyplot as plt
import nifty.re as jft
import numpy as np

from ..model.response import map2tods
from .grid import grid_plot
from .line import line_plot
from .scatter import scatter_plot


def detector_plot(detector, odir="."):
    """Plot detector ordering and detector-group assignments."""

    plot_kwargs = dict(cbar=False, n_ticks=0)

    coords = detector.coordinates
    scatter_plot(coords, np.arange(detector.full_size), fname=f"{odir}/det_ordering.png", **plot_kwargs)
    scatter_plot(coords, detector.groups.indices % 10, fname=f"{odir}/det_groups.png", **plot_kwargs)

    return


def truth_plot(detector, tod_data, truth=None, odir="."):
    """Plot observed TODs and optional truth components for a dataset."""

    _tod_line_plot(tod_data, None, None, fname=f"{odir}/tod_data.png", truth_title="data")

    if truth is not None:
        map_plot(truth["smooth_map"], n_ticks=0, fname=f"{odir}/smooth_map.png")
        _tod_line_plot(truth["tods_atm"], fname=f"{odir}/tod_atm.png", truth_title="atmosphere reconstruction")
        _tod_line_plot(truth["tods_map"], fname=f"{odir}/tod_map.png", truth_title="map reconstruction")

    return


def masked_tod_plot(
    tod_data, bad_mask, odir=".", fname=None, figsize=(20, 5), title="masked regions", label=None, cmap="cmb"
):
    """Plot each detector's TOD with bad (masked-out) samples grayed out.

    Draws two NaN-gapped copies of ``tod_data`` per detector -- kept samples in
    the detector's own color, masked-out samples in a fixed neutral gray --
    so the masked regions stay visually distinct even in a dense, many-line
    overlay (a same-color-but-faded segment can blend into the crowd of other
    full-opacity traces behind it; a flat gray does not).

    Parameters
    ----------
    tod_data : array-like, shape (n_detectors, n_timesamples)
        Cleaned time-ordered data.
    bad_mask : array-like of bool, shape (n_detectors, n_timesamples)
        True where a sample is masked out (jump/spike/high-noise region).
    odir : str, optional
        Output directory; used only when ``fname`` is not given.
    fname : str or None, optional
        Exact output path; defaults to ``{odir}/masked_regions.png``.
    label : str or None, optional
        Identifying label for this dataset (e.g. the source fits file's name,
        see :func:`maria_nifty.fits_label`), shown as a legend in the upper
        left corner. Omitted when ``None``.
    cmap : str, optional
        Colormap used for the (kept-sample) per-detector colors.
    """
    tod_data = np.asarray(tod_data, dtype=float)
    bad_mask = np.asarray(bad_mask, dtype=bool)
    n_det = tod_data.shape[0]

    kept = np.where(bad_mask, np.nan, tod_data)
    masked_out = np.where(bad_mask, tod_data, np.nan)
    combined = np.concatenate([kept, masked_out], axis=0)

    palette = plt.colormaps.get_cmap(cmap)(np.linspace(0, 1, n_det))
    gray = np.tile([0.55, 0.55, 0.55, 1.0], (n_det, 1))
    color = np.concatenate([palette, gray], axis=0)
    row_labels = np.full(2 * n_det, "", dtype=object)
    if label is not None:
        row_labels[0] = label

    # Called directly (not through grid_plot) since this is always a single
    # panel, and grid_plot's own `label` kwarg means something different
    # (per-cell label) than line_plot's per-row legend labels used here.
    line_plot(
        array=combined,
        title=title,
        color=color,
        alpha=1.0,
        label=row_labels,
        legend_loc="upper left",
        figsize=figsize,
        fname=fname or f"{odir}/masked_regions.png",
    )


def tod_plot(
    tod_data,
    sum_response,
    samples,
    mask=None,
    truth=None,
    odir=".",
    plot_extra=False,
    label=None,
):
    """Plot TOD data together with model reconstructions and residuals.

    ``label`` (e.g. the source fits file's identifying name, see
    :func:`maria_nifty.fits_label`), when given, is stamped in the upper-left
    corner of the combined data & reconstructions panel.
    """
    if truth is None:
        truth = dict(smooth_map=None, tods_atm=None, tods_map=None)

    safe_total_response = _build_safe_total_response(sum_response)

    _tod_line_plot(
        tod_data,
        safe_total_response,
        samples,
        mask,
        fname=f"{odir}/tod_residuals.png",
        combined_title="data & reconstructions",
        residual_title="residuals",
        label=label,
    )
    _tod_components_plot(tod_data, sum_response, samples, mask=mask, odir=odir)

    if plot_extra and sum_response.atm_model is not None:
        _tod_line_plot(
            truth["tods_atm"],
            sum_response.atm_model,
            samples,
            mask,
            fname=f"{odir}/tod_atm.png",
            rec_title="atmosphere reconstruction",
            combined_title="atmosphere reconstruction",
            residual_title="residuals",
        )

    if plot_extra and sum_response.map_model is not None:
        _tod_line_plot(
            truth["tods_map"],
            sum_response.map_response,
            samples,
            mask,
            fname=f"{odir}/tod_map.png",
            rec_title="map reconstruction",
            combined_title="map reconstruction",
            residual_title="residuals",
        )

        source_models = tuple(
            getattr(sum_response.map_model, "sources", getattr(sum_response.map_model, "components", ()))
        )
        if sum_response.map_response is not None:
            coords = sum_response.map_response.coords
            sigma = sum_response.map_response.sigma
            background = getattr(sum_response.map_model, "background", None)

            if background is not None:
                background_response = lambda x, m=background: map2tods(m(x, map=True), coords, sigma)
                _tod_line_plot(
                    truth["tods_map"],
                    background_response,
                    samples,
                    mask,
                    fname=f"{odir}/tod_map_background.png",
                    rec_title="map background reconstruction",
                    combined_title="map background reconstruction",
                    residual_title="residuals",
                )

            def source_sum_response(x):
                src_sum = None
                for comp in source_models:
                    val = np.asarray(comp(x, map=True))
                    src_sum = val if src_sum is None else src_sum + val
                if src_sum is None:
                    return np.zeros_like(sum_response.map_response(x))
                return map2tods(src_sum, coords, sigma)

            _tod_line_plot(
                truth["tods_map"],
                source_sum_response,
                samples,
                mask,
                fname=f"{odir}/tod_map_source_model.png",
                rec_title="map source reconstruction",
                combined_title="map source reconstruction",
                residual_title="residuals",
            )

    return


def _eval_model_tods(model, samples, mask=None):
    """Evaluate a TOD model on samples and apply an optional detector mask."""
    if isinstance(samples, jft.Samples):
        rec = jft.mean(tuple(model(s) for s in samples))
    elif isinstance(samples, jft.Vector):
        rec = model(samples)
    else:
        rec = model(samples)

    rec = np.asarray(rec)
    if mask is not None:
        rec = rec[mask]
    return rec


def _build_safe_map_response(sum_response):
    """Build a map-response callable that tolerates missing map components."""
    map_model = getattr(sum_response, "map_model", None)
    map_response = getattr(sum_response, "map_response", None)
    if map_model is None or map_response is None:
        return None

    coords = map_response.coords
    sigma = map_response.sigma
    background = getattr(map_model, "background", None)
    source_models = tuple(getattr(map_model, "sources", getattr(map_model, "components", ())))

    def safe_map_response(x):
        out = None
        if background is not None:
            out = map2tods(background(x, map=True), coords, sigma)
        for comp in source_models:
            comp_tods = map2tods(comp(x, map=True), coords, sigma)
            out = comp_tods if out is None else out + comp_tods
        if out is None:
            return np.zeros_like(map_response(x))
        return out

    return safe_map_response


def _build_safe_total_response(sum_response):
    """Build a total-response callable that tolerates missing components."""
    atm_model = getattr(sum_response, "atm_model", None)
    safe_map_response = _build_safe_map_response(sum_response)

    def safe_total_response(x):
        out = None
        if atm_model is not None:
            out = np.asarray(atm_model(x))
        if safe_map_response is not None:
            map_part = np.asarray(safe_map_response(x))
            out = map_part if out is None else out + map_part
        if out is None:
            raise RuntimeError("No response component available.")
        return out

    return safe_total_response


def _tod_components_plot(tod_data, sum_response, samples, mask=None, odir="."):
    """Plot TOD data, per-component reconstructions, and residuals."""
    data = np.asarray(tod_data)
    if mask is not None:
        data = data[mask]

    rows = []
    titles = []

    # Optional atmosphere row.
    if getattr(sum_response, "atm_model", None) is not None:
        atm_rec = _eval_model_tods(sum_response.atm_model, samples, mask=mask)
        rows.append(atm_rec)
        titles.append("atmosphere reconstruction")

    map_model = getattr(sum_response, "map_model", None)
    map_response = getattr(sum_response, "map_response", None)

    if map_model is not None and map_response is not None:
        coords = map_response.coords
        sigma = map_response.sigma

        background = getattr(map_model, "background", None)
        if background is not None:
            background_response = lambda x, m=background: map2tods(m(x, map=True), coords, sigma)
            bg_rec = _eval_model_tods(background_response, samples, mask=mask)
            rows.append(bg_rec)
            titles.append("map background response")

        source_models = tuple(getattr(map_model, "sources", getattr(map_model, "components", ())))

        def source_sum_response(x):
            src_sum = None
            for comp in source_models:
                val = np.asarray(comp(x, map=True))
                src_sum = val if src_sum is None else src_sum + val
            if src_sum is None:
                return np.zeros_like(map_response(x))
            return map2tods(src_sum, coords, sigma)

        src_rec = _eval_model_tods(source_sum_response, samples, mask=mask)
        rows.append(src_rec)
        titles.append("map source response")

    # Always include residual row.
    safe_total_response = _build_safe_total_response(sum_response)
    rec_total = _eval_model_tods(safe_total_response, samples, mask=mask)
    residual = data - rec_total
    rows.append(residual)
    titles.append("residuals")

    grid_plot(
        array=rows,
        title=titles,
        mode="line",
        figsize=(20, 5),
        rows=len(rows),
        fname=f"{odir}/tod_reconstruction.png",
    )


def map_plot(
    truth=None,
    model=None,
    samples=None,
    vmin=None,
    vmax=None,
    rows=1,
    **kwargs,
):
    """Plot map truth, reconstruction, and optional residuals."""
    arrays, titles = [], []

    if isinstance(truth, np.ndarray):
        arrays.append(truth)
        titles.append("true map (smoothed)")

    if model is not None and samples is not None:
        if isinstance(samples, jft.Samples):
            rec = jft.mean(tuple(model(s) for s in samples))
        elif isinstance(samples, jft.Vector):
            rec = model(samples)
        arrays.append(rec)
        titles.append("reconstructed map")

    if len(titles) == 2:
        arrays.append(rec - truth)
        titles.append("residuals")
        vmin = 2 * [
            truth.min(),
        ] + [
            None,
        ]
        vmax = 2 * [
            truth.max(),
        ] + [
            None,
        ]

    grid_plot(
        array=arrays,
        title=titles,
        vmin=vmin,
        vmax=vmax,
        rows=rows,
        **kwargs,
    )


def _label_callback(label):
    """Build a grid_plot ``callback`` that stamps `label` in the first panel's
    upper-left corner. Used instead of grid_plot's own ``label`` kwarg, which
    means something different (a per-cell title, not a legend entry)."""
    if label is None:
        return None

    def callback(figure, axes):
        axes[0].text(
            0.02,
            0.98,
            label,
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.5", alpha=0.85),
        )

    return callback


def _tod_line_plot(
    truth=None,
    model=None,
    samples=None,
    mask=None,
    figsize=(20, 5),
    truth_title="true tods",
    rec_title="reconstructed tods",
    combined_title="true tods & reconstructed tods",
    residual_title="residuals",
    label=None,
    **kwargs,
):
    """Plot TOD traces, model reconstructions, and residuals as line plots."""
    arrays, titles, linestyles = [], [], []

    if isinstance(truth, np.ndarray):
        if mask is not None:
            truth = truth[mask]
        n_tods = truth.shape[0]
        arrays.append(truth)
        titles.append(truth_title)
        linestyles.append([":"] * n_tods)

    if model is not None and samples is not None:
        if isinstance(samples, jft.Samples):
            rec = jft.mean(tuple(model(s) for s in samples))
        elif isinstance(samples, jft.Vector):
            rec = model(samples)
        if mask is not None:
            rec = rec[mask]
        n_tods = rec.shape[0]
        arrays.append(rec)
        titles.append(rec_title)
        linestyles.append(["-"] * n_tods)

    if len(titles) == 2:
        combined = np.concatenate(arrays)
        residual = arrays[0] - arrays[1]
        arrays = [combined, residual]
        titles = [combined_title, residual_title]
        linestyles[0] = linestyles[0] + linestyles[1]
        alphas = [[0.5] * n_tods + [1.0] * n_tods, [1.0] * n_tods]
    else:
        alphas = [[1.0] * n_tods]

    grid_plot(
        array=arrays,
        title=titles,
        mode="line",
        repeat_kwargs="one",
        color=np.array(
            [
                n_tods,
            ]
            * 2,
            dtype=object,
        ),
        linestyle=np.array(linestyles, dtype=object),
        alpha=np.array(alphas, dtype=object),
        figsize=figsize,
        callback=_label_callback(label),
        **kwargs,
    )
