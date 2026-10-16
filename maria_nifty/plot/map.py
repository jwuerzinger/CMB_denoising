"""Map reconstruction plots and map-component diagnostics."""

import os

import matplotlib.pyplot as plt
import nifty.re as jft
import numpy as np
from matplotlib.colors import LogNorm

from ..model.space import map2space
from ..model.util import ARCMIN2RAD, AS2RAD, DEG2RAD
from .grid import grid_plot
from .util import apply_sky_ticks, set_cbar


def _safe_name(name):
    """Normalize a component name for use in filenames."""
    out = []
    for ch in str(name).strip().lower():
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch in (" ", "/", "\\"):
            out.append("_")
    s = "".join(out).strip("_")
    return s if s else "component"


def _get_mean_std(eval_fn, samples):
    """Evaluate a model on samples and return mean/std arrays."""
    if isinstance(samples, jft.Samples):
        mean, std = jft.mean_and_std(tuple(eval_fn(s) for s in samples))
        return np.asarray(mean), np.asarray(std)

    arr = np.asarray(eval_fn(samples))
    return arr, np.zeros_like(arr)


def _get_sample_maps(eval_fn, samples):
    """Return a list of per-sample map realizations when available."""
    if not isinstance(samples, jft.Samples):
        return []
    return [np.asarray(eval_fn(s)) for s in samples]


def _space_fov_and_center(space):
    """Extract field of view and center information from a space-like object."""
    if space is None:
        return None, None

    fov = None
    center = None

    if hasattr(space, "fov"):
        fov = np.asarray(space.fov, dtype=float)
    elif hasattr(space, "shape") and hasattr(space, "distances"):
        fov = np.asarray(space.shape, dtype=float) * np.asarray(space.distances, dtype=float)

    if hasattr(space, "cen"):
        center = np.asarray(space.cen, dtype=float)
    elif hasattr(space, "center"):
        center = np.asarray(space.center, dtype=float)

    if center is not None and center.ndim > 1:
        center = center[0]

    if fov is None or np.asarray(fov).shape != (2,):
        return None, None
    if center is None or np.asarray(center).shape != (2,):
        center = np.array([0.0, 0.0], dtype=float)
    return np.asarray(fov, dtype=float), np.asarray(center, dtype=float)


def _offsets_fov_and_center(offsets):
    """Infer field of view and center from detector offsets."""
    flat_offsets = np.asarray(offsets, dtype=float).reshape(-1, 2)
    mins = flat_offsets.min(axis=0)
    maxs = flat_offsets.max(axis=0)
    return maxs - mins, 0.5 * (maxs + mins)


def _get_fov_and_center(sum_response, offsets=None, space=None):
    """Resolve map extent information from the response, space, or offsets."""
    fov, center = _space_fov_and_center(space)
    if fov is not None:
        return fov, center

    map_model = getattr(sum_response, "map_model", None)
    if map_model is not None:
        fov, center = _space_fov_and_center(getattr(map_model, "space", None))
        if fov is not None:
            return fov, center

    map_response = getattr(sum_response, "map_response", None)
    if map_response is not None and getattr(map_response, "fov", None) is not None:
        fov = np.asarray(map_response.fov, dtype=float)
        center = np.asarray(getattr(map_response, "center", (0.0, 0.0)), dtype=float)
        if fov.shape == (2,):
            if center.shape != (2,):
                center = np.array([0.0, 0.0], dtype=float)
            return fov, center

    if offsets is None:
        raise ValueError("Could not determine fov/center from model space or response and no offsets were provided")
    return _offsets_fov_and_center(offsets)


def _unit_scale_and_label(unit):
    """Return scaling factor and label for angular display units."""
    unit = str(unit).strip().lower()
    unit_scales = {
        "radian": 1.0,
        "degrees": 1.0 / DEG2RAD,
        "arcmin": 1.0 / ARCMIN2RAD,
        "arcsec": 1.0 / AS2RAD,
    }
    unit_labels = {
        "radian": "rad",
        "degrees": "deg",
        "arcmin": "arcmin",
        "arcsec": "arcsec",
    }
    if unit not in unit_scales:
        raise ValueError("unit has to be one of: radian, arcmin, arcsec, degrees")
    return unit_scales[unit], unit_labels[unit], unit


def _iter_source_models(map_model):
    """Return named source-model pairs in stable order."""
    if map_model is None:
        return []

    components = []
    counts = {}
    for m in tuple(getattr(map_model, "sources", getattr(map_model, "components", ()))):
        base = type(m).__name__.replace("Model", "").lower() or "source"
        counts[base] = counts.get(base, 0) + 1
        name = f"{base}_{counts[base]}"
        components.append((name, m))
    return [(n, m) for n, m in components if m is not None]


def _plot_mean_std_samples(
    *,
    mean_map,
    std_map,
    sample_maps,
    extent,
    unit_label,
    cmap,
    odir,
    base_name,
    title_prefix,
    mean_file=None,
    std_file=None,
    samples_file=None,
    center_phi_theta=None,
    cbar_label=None,
    cbar_scale=1.0,
):
    """Plot a mean map, relative std map, and optional individual samples."""
    # Relative std (std/|mean|) is a unit-invariant ratio -- compute it from
    # the original (unscaled) values, before cbar_scale is applied below,
    # so the fixed absolute epsilon floor doesn't shift which regime
    # dominates depending on the chosen display unit.
    abs_mean = np.abs(mean_map)
    eps = max(1e-12, 1e-3 * float(np.percentile(abs_mean, 5)))
    rel_std = std_map / np.maximum(abs_mean, eps)

    if cbar_scale != 1.0:
        mean_map = mean_map * cbar_scale
        std_map = std_map * cbar_scale
        sample_maps = [m * cbar_scale for m in sample_maps]
    mean_abs_max = float(np.max(np.abs(mean_map)))
    if mean_abs_max == 0.0:
        mean_abs_max = 1e-12

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    img = ax.imshow(
        _plot_transform_xflip(mean_map),
        cmap=cmap,
        origin="lower",
        extent=extent,
        vmin=-mean_abs_max,
        vmax=mean_abs_max,
    )
    if center_phi_theta is not None:
        apply_sky_ticks(ax, center_phi_theta)
    else:
        ax.set_title(f"{title_prefix} mean")
        ax.set_xlabel(f"Right ascension [{unit_label}]")
        ax.set_ylabel(f"Declination [{unit_label}]")
    fig.tight_layout()
    set_cbar(ax, img, cbar=True, loc="right", label=cbar_label)

    positive_rel = rel_std[rel_std > 0]
    if positive_rel.size > 0:
        rel_vmin = max(1e-6, float(np.percentile(positive_rel, 1)))
        rel_vmax = max(rel_vmin * 1.01, float(np.percentile(positive_rel, 99)))
    else:
        rel_vmin, rel_vmax = 1e-6, 1.0

    fig_std, ax_std = plt.subplots(figsize=(7.5, 7.5))
    img_std = ax_std.imshow(
        _plot_transform_xflip(rel_std),
        cmap="cividis",
        origin="lower",
        extent=extent,
        norm=LogNorm(vmin=rel_vmin, vmax=rel_vmax),
    )
    if center_phi_theta is not None:
        apply_sky_ticks(ax_std, center_phi_theta)
    else:
        ax_std.set_title(f"{title_prefix} relative std (std/|mean|)")
        ax_std.set_xlabel(f"Right ascension [{unit_label}]")
        ax_std.set_ylabel(f"Declination [{unit_label}]")
    fig_std.tight_layout()
    set_cbar(ax_std, img_std, cbar=True, loc="right")

    if odir is not None:
        os.makedirs(odir, exist_ok=True)
        mean_out = mean_file if mean_file else f"{base_name}_mean.png"
        std_out = std_file if std_file else f"{base_name}_std.png"
        fig.savefig(os.path.join(odir, mean_out), dpi=150, bbox_inches="tight")
        fig_std.savefig(os.path.join(odir, std_out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close(fig_std)
    else:
        plt.show()

    if sample_maps:
        panel_abs_max = float(np.max([np.max(np.abs(m)) for m in sample_maps]))
        if panel_abs_max == 0.0:
            panel_abs_max = 1e-12
        n_samples = len(sample_maps)
        fig_samples, axes = plt.subplots(
            1,
            n_samples,
            figsize=(4.0 * n_samples + 1.0, 4.2),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        if n_samples == 1:
            axes = [axes]

        last_img = None
        for i, (ax_i, sample_map) in enumerate(zip(axes, sample_maps)):
            last_img = ax_i.imshow(
                _plot_transform_xflip(sample_map),
                cmap=cmap,
                origin="lower",
                extent=extent,
                vmin=-panel_abs_max,
                vmax=panel_abs_max,
            )
            if center_phi_theta is not None:
                apply_sky_ticks(ax_i, center_phi_theta, ylabel=(i == 0))
            else:
                ax_i.set_title(f"{title_prefix} sample {i}")
                ax_i.set_xlabel(f"RA [{unit_label}]")
                if i == 0:
                    ax_i.set_ylabel(f"Dec [{unit_label}]")

        fig_samples.colorbar(last_img, ax=axes, location="right", fraction=0.046, pad=0.02, label=cbar_label)

        if odir is not None:
            samples_out = samples_file if samples_file else f"{base_name}_samples.png"
            fig_samples.savefig(os.path.join(odir, samples_out), dpi=150, bbox_inches="tight")
            plt.close(fig_samples)
        else:
            plt.show()


def _plot_transform_xflip(arr):
    """Transpose map-like data and flip along x-axis in plot coordinates."""
    return np.asarray(arr).T


def _plot_reconstruction_grid(
    arrays,
    titles,
    extents,
    odir,
    unit_label,
    cmap="cmb",
    center_phi_theta=None,
    cbar_label=None,
    cbar_scale=1.0,
):
    """Plot a compact overview grid for reconstructed map components."""
    if len(arrays) == 0:
        return

    # User-facing compact overview: at most five panels (drop any 6th+ panel).
    arrays = list(arrays[:5])
    titles = list(titles[:5])
    extents = list(extents[:5])

    arrays_np = [np.asarray(a) * cbar_scale for a in arrays]

    cols = 2 if len(arrays_np) > 1 else 1
    rows = int(np.ceil(len(arrays_np) / cols))

    def _ticks_from_extent(ext, n_ticks=5):
        x0, x1, y0, y1 = [float(v) for v in ext]
        x_ticks = np.array([x0]) if np.isclose(x0, x1) else np.linspace(x0, x1, n_ticks)
        y_ticks = np.array([y0]) if np.isclose(y0, y1) else np.linspace(y0, y1, n_ticks)
        return x_ticks, y_ticks

    def _callback(_, axes):
        for i, ax in enumerate(axes):
            if i >= len(arrays_np):
                ax.set_visible(False)
                continue

            r = i // cols
            c = i % cols

            if len(ax.images) > 0:
                ext = np.asarray(extents[i], dtype=float).reshape(4)
                ax.images[-1].set_extent(ext.tolist())
                ax.set_xlim(float(ext[0]), float(ext[1]))
                ax.set_ylim(float(ext[2]), float(ext[3]))

                x_ticks, y_ticks = _ticks_from_extent(ext)
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)

                # Axis labels only on bottom row / left column; the Delta-theta
                # offset secondary axis only on the top row (it would otherwise
                # repeat identically -- up to which array's own extent -- on
                # every row).
                if center_phi_theta is not None:
                    ax.set_title("")
                    apply_sky_ticks(
                        ax,
                        center_phi_theta,
                        xlabel=(r == rows - 1),
                        ylabel=(c == 0),
                        secondary=(r == 0),
                    )
                else:
                    ax.set_xlabel(f"Right ascension [{unit_label}]" if r == rows - 1 else "")
                    ax.set_ylabel(f"Declination [{unit_label}]" if c == 0 else "")

                amax = float(np.max(np.abs(arrays_np[i])))
                if amax == 0.0:
                    amax = 1e-12
                ax.images[-1].set_clim(vmin=-amax, vmax=amax)
                set_cbar(ax, ax.images[-1], cbar=True, loc="right", label=cbar_label)

    if odir is not None:
        os.makedirs(odir, exist_ok=True)
        fname = os.path.join(odir, "map_reconstruction.png")
    else:
        fname = ""

    grid_plot(
        array=arrays_np,
        title=titles,
        mode="image",
        transpose=True,
        origin="lower",
        cmap=cmap,
        cbar=False,
        rows=rows,
        cols=cols,
        figsize=(7.5, 7.5),
        callback=_callback,
        fname=fname,
    )


def map_plot(
    sum_response=None,
    offsets=None,
    samples=None,
    mean_map=None,
    space=None,
    odir=None,
    fname="map_mean.png",
    std_fname="map_std.png",
    samples_fname="map_samples.png",
    unit="arcmin",
    cmap="cmb",
    vmin=None,
    vmax=None,
    plot_extra=False,
    plot_reconstruction=True,
    center_phi_theta=None,
    cbar_label=None,
    cbar_scale=1.0,
):
    """Plot map reconstruction overview and optional extra component plots.

    Can operate in two modes:
    1. Model-based mode: provide sum_response (model with map_model), offsets, samples
    2. Direct mean map mode: provide mean_map as numpy array (similar to residual.py pattern)
    """

    # Handle direct mean_map input (e.g., from BinMapper output)
    if mean_map is not None:
        mean_map_array = np.asarray(mean_map, dtype=float) * cbar_scale

        plot_vmin = vmin
        plot_vmax = vmax
        if plot_vmin is None and plot_vmax is None:
            abs_max = float(np.nanmax(np.abs(mean_map_array)))
            if not np.isfinite(abs_max) or abs_max == 0.0:
                abs_max = 1e-12
            plot_vmin = -abs_max
            plot_vmax = abs_max
        elif plot_vmin is None:
            plot_vmin = -abs(float(plot_vmax))
        elif plot_vmax is None:
            plot_vmax = abs(float(plot_vmin))

        if space is not None:
            scale, unit_label, _ = _unit_scale_and_label(unit)
            fov_rad, center_rad = _space_fov_and_center(space)
            fov_u = fov_rad * scale
            center_u = center_rad * scale
            extent = [
                center_u[0] - 0.5 * fov_u[0],
                center_u[0] + 0.5 * fov_u[0],
                center_u[1] - 0.5 * fov_u[1],
                center_u[1] + 0.5 * fov_u[1],
            ]
            xlabel = f"Right ascension [{unit_label}]"
            ylabel = f"Declination [{unit_label}]"
        elif offsets is not None:
            scale, unit_label, _ = _unit_scale_and_label(unit)
            fov_rad, center_rad = _offsets_fov_and_center(offsets)
            fov_u = fov_rad * scale
            center_u = center_rad * scale
            extent = [
                center_u[0] - 0.5 * fov_u[0],
                center_u[0] + 0.5 * fov_u[0],
                center_u[1] - 0.5 * fov_u[1],
                center_u[1] + 0.5 * fov_u[1],
            ]
            xlabel = f"Right ascension [{unit_label}]"
            ylabel = f"Declination [{unit_label}]"
        else:
            extent = [0, mean_map_array.shape[-1], 0, mean_map_array.shape[-2]]
            xlabel = "X [px]"
            ylabel = "Y [px]"

        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        img = ax.imshow(
            _plot_transform_xflip(mean_map_array),
            cmap=cmap,
            origin="lower",
            extent=extent,
            vmin=plot_vmin,
            vmax=plot_vmax,
        )
        if center_phi_theta is not None and (space is not None or offsets is not None):
            apply_sky_ticks(ax, center_phi_theta)
        else:
            ax.set_title("binned map mean")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02, label=cbar_label)
        fig.tight_layout()

        if odir is not None:
            os.makedirs(odir, exist_ok=True)
            out_fname = fname if fname.endswith(".png") else f"{fname}.png"
            fig.savefig(os.path.join(odir, out_fname), dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

        return

    # Model-based mode
    if sum_response is None:
        return

    map_model = getattr(sum_response, "map_model", None)
    if map_model is None:
        return

    source_models = tuple(getattr(map_model, "sources", getattr(map_model, "components", ())))
    bg_model = getattr(map_model, "background", None)
    source_space = getattr(map_model, "source_space", None)
    if source_space is None:
        source_space = getattr(map_model, "space", None)
    source_map_func = map2space(getattr(map_model, "space", None), source_space, order=0)

    scale, unit_label, _ = _unit_scale_and_label(unit)

    def full_eval(s):
        out = None
        if bg_model is not None:
            out = np.asarray(bg_model(s, map=True))
        for m in source_models:
            val = np.asarray(m(s, map=True))
            out = val if out is None else out + val
        if out is None:
            return np.zeros(getattr(map_model.space, "shape", (1, 1)), dtype=float)
        return out

    full_mean, full_std = _get_mean_std(full_eval, samples)
    full_samples = _get_sample_maps(full_eval, samples)

    fov_rad, center_rad = _get_fov_and_center(sum_response, offsets, getattr(map_model, "space", None))
    fov_u = fov_rad * scale
    center_u = center_rad * scale
    full_extent = [
        center_u[0] - 0.5 * fov_u[0],
        center_u[0] + 0.5 * fov_u[0],
        center_u[1] - 0.5 * fov_u[1],
        center_u[1] + 0.5 * fov_u[1],
    ]

    src_fov_rad, src_center_rad = _get_fov_and_center(sum_response, offsets, source_space)
    src_fov_u = src_fov_rad * scale
    src_center_u = src_center_rad * scale
    source_extent = [
        src_center_u[0] - 0.5 * src_fov_u[0],
        src_center_u[0] + 0.5 * src_fov_u[0],
        src_center_u[1] - 0.5 * src_fov_u[1],
        src_center_u[1] + 0.5 * src_fov_u[1],
    ]

    full_base = "map"

    full_mean_file = fname if fname else "map_mean.png"
    full_std_file = std_fname if std_fname else "map_std.png"
    full_samples_file = samples_fname if samples_fname else "map_samples.png"
    if not full_mean_file.endswith(".png"):
        full_mean_file += ".png"
    if not full_std_file.endswith(".png"):
        full_std_file += ".png"
    if not full_samples_file.endswith(".png"):
        full_samples_file += ".png"

    if plot_extra:
        _plot_mean_std_samples(
            mean_map=full_mean,
            std_map=full_std,
            sample_maps=full_samples,
            extent=full_extent,
            unit_label=unit_label,
            cmap=cmap,
            odir=odir,
            base_name=full_base,
            title_prefix="full map",
            mean_file=full_mean_file,
            std_file=full_std_file,
            samples_file=full_samples_file,
            center_phi_theta=center_phi_theta,
            cbar_label=cbar_label,
            cbar_scale=cbar_scale,
        )

    rec_arrays = []
    rec_titles = []
    rec_extents = []

    # Background: include only if available.
    if bg_model is not None:
        bg_eval = lambda s, m=bg_model: m(s, map=True)
        bg_mean, bg_std = _get_mean_std(bg_eval, samples)
        bg_samples = _get_sample_maps(bg_eval, samples)

        bg_fov_rad, bg_center_rad = _get_fov_and_center(
            sum_response,
            offsets,
            getattr(bg_model, "space", None),
        )

        if bg_mean.shape != tuple(getattr(getattr(bg_model, "space", None), "shape", bg_mean.shape)):
            bg_extent = full_extent
        else:
            bg_fov_u = bg_fov_rad * scale
            bg_center_u = bg_center_rad * scale
            bg_extent = [
                bg_center_u[0] - 0.5 * bg_fov_u[0],
                bg_center_u[0] + 0.5 * bg_fov_u[0],
                bg_center_u[1] - 0.5 * bg_fov_u[1],
                bg_center_u[1] + 0.5 * bg_fov_u[1],
            ]

        if plot_extra:
            _plot_mean_std_samples(
                mean_map=bg_mean,
                std_map=bg_std,
                sample_maps=bg_samples,
                extent=bg_extent,
                unit_label=unit_label,
                cmap=cmap,
                odir=odir,
                base_name="map_background",
                title_prefix="background",
                center_phi_theta=center_phi_theta,
                cbar_label=cbar_label,
                cbar_scale=cbar_scale,
            )
        rec_arrays.append(bg_mean)
        rec_titles.append("background")
        rec_extents.append(bg_extent)

    # Source model: always include when map_model exists.
    def _source_sum_eval(s):
        out = None
        for m in source_models:
            val = np.asarray(m(s, map=True))
            out = val if out is None else out + val
        if out is None:
            return np.zeros(getattr(source_space, "shape", full_mean.shape), dtype=float)
        return np.asarray(source_map_func(out, x=s))

    src_sum_mean, src_sum_std = _get_mean_std(_source_sum_eval, samples)
    src_sum_samples = _get_sample_maps(_source_sum_eval, samples)

    if plot_extra:
        _plot_mean_std_samples(
            mean_map=src_sum_mean,
            std_map=src_sum_std,
            sample_maps=src_sum_samples,
            extent=source_extent,
            unit_label=unit_label,
            cmap=cmap,
            odir=odir,
            base_name="map_source_model",
            title_prefix="source model",
            center_phi_theta=center_phi_theta,
            cbar_label=cbar_label,
            cbar_scale=cbar_scale,
        )
    rec_arrays.append(src_sum_mean)
    rec_titles.append("source model")
    rec_extents.append(source_extent)

    # Individual source components only if there is more than one source.
    if len(source_models) <= 1:
        if plot_reconstruction:
            _plot_reconstruction_grid(
                rec_arrays,
                rec_titles,
                rec_extents,
                odir,
                unit_label,
                cmap=cmap,
                center_phi_theta=center_phi_theta,
                cbar_label=cbar_label,
                cbar_scale=cbar_scale,
            )
        return

    for comp_name, comp_model in _iter_source_models(map_model):
        comp_eval = lambda s, m=comp_model: source_map_func(np.asarray(m(s, map=True)), x=s)
        comp_mean, comp_std = _get_mean_std(comp_eval, samples)
        comp_samples = _get_sample_maps(comp_eval, samples)
        comp_extent = source_extent

        base_name = f"map_{_safe_name(comp_name)}"
        if plot_extra:
            _plot_mean_std_samples(
                mean_map=comp_mean,
                std_map=comp_std,
                sample_maps=comp_samples,
                extent=comp_extent,
                unit_label=unit_label,
                cmap=cmap,
                odir=odir,
                base_name=base_name,
                title_prefix=comp_name,
                center_phi_theta=center_phi_theta,
                cbar_label=cbar_label,
                cbar_scale=cbar_scale,
            )

        rec_arrays.append(comp_mean)
        rec_titles.append(comp_name)
        rec_extents.append(comp_extent)

    if plot_reconstruction:
        _plot_reconstruction_grid(
            rec_arrays,
            rec_titles,
            rec_extents,
            odir,
            unit_label,
            cmap=cmap,
            center_phi_theta=center_phi_theta,
            cbar_label=cbar_label,
            cbar_scale=cbar_scale,
        )
