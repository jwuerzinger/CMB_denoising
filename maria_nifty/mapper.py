"""Thin wrapper around Maria's bin mapper used by the steering scripts."""

import os

import numpy as np
from maria.mappers import BinMapper
from nifty.re import logger

from .model.space import GridSpace
from .plot.map import map_plot
from .plot.util import signal_cbar_label


def _finalize_map(
    mean_map_array,
    weight_array,
    offsets,
    mapper_kwargs,
    odir,
    fname,
    center_phi_theta=None,
    cbar_label=None,
):
    """Apply shared map postprocessing/plotting to a binned or solved map.

    Common tail used by both :func:`run_bin_mapper` and
    :func:`run_minkasi_mapper`: optional low-coverage masking, robust
    (percentile-based) color scaling, building the :class:`GridSpace` implied
    by ``width``/``height``/``resolution``/``center``, and rendering via
    :func:`~maria_nifty.plot.map.map_plot`.

    Parameters
    ----------
    mean_map_array : numpy.ndarray
        2D map array, shape ``(n_y, n_x)`` (declination axis first).
    weight_array : numpy.ndarray or None
        2D weight/hits map matching ``mean_map_array``'s shape, or ``None``.
    offsets : numpy.ndarray or None
        Detector offsets, forwarded to :func:`map_plot` as a plotting fallback.
    mapper_kwargs : dict
        Mapper keyword arguments; reads ``center``, ``width``, ``height``,
        ``resolution``, and ``map_postprocessing``.
    odir : str
        Output directory for the plot.
    fname : str
        Plot filename.

    Returns
    -------
    mean_map_array : numpy.ndarray
        The map array (flipped along the RA axis for plotting).
    space : GridSpace or None
        The grid space describing ``mean_map_array``'s extent, or ``None`` when
        no width/height/resolution were provided to derive one.
    """
    # Optional low-coverage masking + robust (percentile-based) color scale: a
    # few noisy scan-turnaround-edge pixels with little coverage otherwise set
    # the whole plot's vmin/vmax via a hard nanmax, washing out faint interior
    # structure. Both are opt-in via map_postprocessing so existing configs
    # (with no such keys) plot exactly as before.
    map_postprocessing = mapper_kwargs.get("map_postprocessing", {}) or {}
    plot_vmin = plot_vmax = None

    well_covered = None
    if weight_array is not None:
        coverage_cfg = map_postprocessing.get("coverage_mask")
        if coverage_cfg:
            min_frac = float(coverage_cfg.get("min_frac", 0.2))
            peak_weight = np.nanmax(weight_array)
            well_covered = weight_array >= (min_frac * peak_weight) if peak_weight > 0 else None
            if well_covered is not None:
                mean_map_array = np.where(well_covered, mean_map_array, np.nan)

    robust_cfg = map_postprocessing.get("robust_color_scale")
    if robust_cfg:
        percentile = float(robust_cfg.get("percentile", 99.5))
        scale = float(
            robust_cfg.get("scale", 1.0)
        )  # multiplies the percentile-based bound, e.g. 0.5 to saturate bright peaks and boost contrast on fainter structure
        region = mean_map_array if well_covered is None else np.where(well_covered, mean_map_array, np.nan)
        abs_vals = np.abs(region)
        if np.isfinite(abs_vals).any():
            abs_max = float(np.nanpercentile(abs_vals, percentile)) * scale
            if np.isfinite(abs_max) and abs_max > 0:
                plot_vmin, plot_vmax = -abs_max, abs_max

    space = None
    plot_shape = tuple(int(s) for s in mean_map_array.shape)
    center = tuple(np.asarray(mapper_kwargs.get("center", (0.0, 0.0)), dtype=float).tolist())

    width = mapper_kwargs.get("width")
    height = mapper_kwargs.get("height")
    resolution = mapper_kwargs.get("resolution")

    if resolution is not None:
        distances = (np.deg2rad(float(resolution)), np.deg2rad(float(resolution)))
    elif width is not None or height is not None:
        width = float(width if width is not None else height)
        height = float(height if height is not None else width)
        distances = (
            np.deg2rad(height / plot_shape[0]),
            np.deg2rad(width / plot_shape[1]),
        )
    else:
        distances = (1.0 / plot_shape[0], 1.0 / plot_shape[1])

    if width is not None or height is not None or resolution is not None:
        space = GridSpace.build(
            shape=plot_shape,
            distances=distances,
            center=np.deg2rad(center),
        )

    mean_map_array = np.flip(mean_map_array, axis=1)

    # Use map_plot for visualization
    map_plot(
        mean_map=mean_map_array,
        space=space,
        offsets=offsets,
        odir=odir,
        fname=fname,
        unit="arcmin",
        vmin=plot_vmin,
        vmax=plot_vmax,
        plot_extra=False,
        plot_reconstruction=True,
        center_phi_theta=center_phi_theta,
        cbar_label=cbar_label,
    )

    return mean_map_array, space


def run_bin_mapper(datasets, odir, verbose=True, **mapper_kwargs):
    """Run the Maria bin mapper and save its preview plot to ``odir``.

    Parameters
    ----------
    datasets : list of dict
        List of dataset dictionaries, each containing a "tods" key with the corresponding TODs.
    odir : str
        Output directory; created if it does not exist.
    verbose : bool, optional
        Whether to log progress via the NIFTy logger.
    **mapper_kwargs
        Additional keyword arguments forwarded to :class:`maria.mappers.BinMapper`.

    Returns
    -------
    mean_map_array : numpy.ndarray
        The 2D binned map (flipped along the RA axis for plotting).
    space : GridSpace or None
        The grid space describing ``mean_map_array``'s extent, or ``None`` when
        no width/height/resolution were provided to derive one.
    """
    if verbose:
        logger.info("\nRun bin mapper:")

    tods = [d["tods"] for d in datasets]
    offsets = datasets[0].get("offsets") if datasets else None

    os.makedirs(odir, exist_ok=True)

    mapper = BinMapper(
        **mapper_kwargs,
        tods=tods,
    )
    output_map = mapper.run()

    def _extract_2d(arr):
        if hasattr(arr, "compute"):
            arr = arr.compute()
        arr = np.squeeze(np.asarray(arr, dtype=float))
        if arr.ndim != 2:
            raise ValueError(f"BinMapper output must reduce to a 2D array for plotting, got shape {arr.shape}")
        return arr

    if hasattr(output_map, "package") and hasattr(output_map, "squeeze"):
        squeezed = output_map.squeeze()
        mean_map_array = _extract_2d(squeezed.data)
        weight_array = _extract_2d(squeezed.weight) if getattr(squeezed, "weight", None) is not None else None
    else:
        mean_map_array = _extract_2d(output_map)
        weight_array = None

    return _finalize_map(
        mean_map_array,
        weight_array,
        offsets,
        mapper_kwargs,
        odir,
        "bin_mapper.png",
        center_phi_theta=datasets[0].get("center_phi_theta") if datasets else None,
        cbar_label=(signal_cbar_label(datasets[0]["tods"])[1] if datasets else None),
    )


def run_minkasi_mapper(datasets, odir, verbose=True, **mapper_kwargs):
    """Run a minkasi maximum-likelihood mapper and save its preview plot to ``odir``.

    A second, independent reconstruction alongside :func:`run_bin_mapper`: each
    dataset's TOD is optionally cleaned with the same
    :meth:`maria.tod.TOD.process` pipeline used by the bin mapper's
    ``tod_preprocessing``, wrapped into a `minkasi <https://github.com/sievers/minkasi>`_
    :class:`~minkasi.tods.Tod` (pointing taken from the already-recentered
    ``offsets``), given a `minkasi.mapmaking.noise.NoiseSmoothedSVD` noise
    model, and solved for the maximum-likelihood map via preconditioned
    conjugate gradient (:func:`minkasi.mapmaking.run_pcg`) instead of Maria's
    simple binning.

    Parameters
    ----------
    datasets : list of dict
        List of dataset dictionaries as returned by
        :func:`~maria_nifty.util.load_tod_datasets` (needs "tods", "offsets",
        "time", and optionally "bad_mask").
    odir : str
        Output directory; created if it does not exist.
    verbose : bool, optional
        Whether to log progress via the NIFTy logger.
    **mapper_kwargs
        ``width``, ``height`` (defaults to ``width``), and ``resolution`` (all
        degrees) set the map extent/pixel size, matching
        :class:`maria.mappers.BinMapper`'s conventions; ``center`` (RA/Dec
        degrees, default ``(0, 0)``) and ``map_postprocessing`` are forwarded
        to the shared plotting tail. ``tod_preprocessing`` is forwarded to
        ``TOD.process`` per dataset (same schema as the bin mapper's).
        ``noise`` (dict) is forwarded to ``NoiseSmoothedSVD`` (``fwhm``,
        ``prewhiten``, ``fit_powlaw``). ``maxiter`` (default 25) sets the PCG
        iteration count.

    Returns
    -------
    mean_map_array : numpy.ndarray
        The 2D solved map (flipped along the RA axis for plotting).
    space : GridSpace or None
        The grid space describing ``mean_map_array``'s extent, or ``None`` when
        no width/height/resolution were provided to derive one.
    """
    import minkasi

    if verbose:
        logger.info("\nRun minkasi mapper:")

    os.makedirs(odir, exist_ok=True)

    tod_preprocessing = mapper_kwargs.get("tod_preprocessing", {}) or {}
    noise_kwargs = dict(mapper_kwargs.get("noise", {}) or {})
    maxiter = int(mapper_kwargs.get("maxiter", 25))

    width = float(mapper_kwargs["width"])
    height = float(mapper_kwargs.get("height", width))
    resolution = float(mapper_kwargs["resolution"])

    offsets = datasets[0].get("offsets") if datasets else None

    todvec = minkasi.tods.TodVec()
    for i, d in enumerate(datasets):
        ptod = d["tods"].process(config=tod_preprocessing) if tod_preprocessing else d["tods"]
        dat = ptod.signal
        if hasattr(dat, "compute"):
            dat = dat.compute()
        dat = np.asarray(dat, dtype="float64")

        d_offsets = d["offsets"]
        n_t = min(dat.shape[1], d_offsets.shape[1])
        dat = dat[:, :n_t].copy()

        bad_mask = d.get("bad_mask")
        if bad_mask is not None and np.any(bad_mask):
            dat[np.asarray(bad_mask)[:, :n_t]] = 0.0

        info = {
            "dat_calib": dat,
            "dx": np.asarray(d_offsets[:, :n_t, 0], dtype="float64"),
            "dy": np.asarray(d_offsets[:, :n_t, 1], dtype="float64"),
            "ctime": np.asarray(d["time"][:n_t], dtype="float64"),
        }
        minkasi_tod = minkasi.tods.Tod(info)
        # NoiseSmoothedSVD's default path runs np.linalg.svd(dat, full_matrices=True),
        # which for (n_det, n_time) with n_time >> n_det computes a huge and entirely
        # unused (n_time, n_time) right-singular-vector matrix -- O(n_time**3) and
        # impractically slow for TOD lengths like ours. Passing the economical
        # (n_det, n_det) U ourselves via u_in skips that; only U is ever used.
        u_econ, _, _ = np.linalg.svd(dat, full_matrices=False)
        minkasi_tod.set_noise(minkasi.mapmaking.noise.NoiseSmoothedSVD, u_in=u_econ, **noise_kwargs)
        # The already-bandpass-filtered/windowed input (tod_preprocessing) can
        # drive the smoothed power spectrum to exactly 0 at some frequency bins,
        # so NoiseSmoothedSVD's own `1/spec_smooth` blows up to inf there. Since
        # apply_noise later multiplies these weights against data that is often
        # ~0 at the very same suppressed frequencies, `0 * inf = nan` leaks in
        # and poisons the whole PCG solve a few iterations in. Sanitize: treat
        # a zero-power (no information) bin as zero weight, not infinite
        # weight, and cap pathologically large-but-finite weights so no single
        # bin can numerically dominate.
        mywt = minkasi_tod.noise.mywt
        finite = np.isfinite(mywt) & (mywt > 0)
        cap = 1.0e3 * np.median(mywt[finite]) if finite.any() else 0.0
        minkasi_tod.noise.mywt = np.where(finite, np.minimum(mywt, cap), 0.0)
        todvec.add_tod(minkasi_tod)
        if verbose:
            logger.info(f"  minkasi tod {i + 1:02d} shape: {dat.shape}")

    pixsize = np.deg2rad(resolution)

    # minkasi's tod2map/map2tod call raw C extensions with unchecked pixel
    # indexing: any sample landing outside the map grid writes out of bounds
    # and segfaults. The configured width/height is a science-box footprint
    # for plotting (matching BinMapper), not necessarily wide enough to
    # contain every scanned sample (e.g. raster turnarounds), so the solve
    # grid must instead cover the *actual* data extent -- shrink-wrapped via
    # todvec.lims(), same as minkasi's own examples do -- and get cropped
    # down to the configured science box only afterward, for plotting.
    full_lims = todvec.lims()
    sky_map = minkasi.maps.SkyMapCar(full_lims, pixsize)

    mapset = minkasi.maps.Mapset()
    mapset.add_map(sky_map)

    rhs = mapset.copy()
    todvec.make_rhs(rhs)

    x0 = mapset.copy()
    x0.clear()

    hits = minkasi.mapmaking.make_hits(todvec, sky_map, do_weights=True)
    precon = mapset.copy()
    precon.maps[0].map[:] = hits.map[:]
    precon.maps[0].invert()

    if verbose:
        logger.info(f"  running PCG for {maxiter} iterations")
    result = minkasi.mapmaking.run_pcg(rhs, x0, todvec, precon=precon, maxiter=maxiter)

    # Crop the (data-extent-sized) solved map down to the configured science
    # box, using the same cos(dec)-flat pixelization SkyMapCar itself uses, so
    # the cropped array lines up with sky_map's own pixel grid.
    center = np.asarray(mapper_kwargs.get("center", (0.0, 0.0)), dtype=float)
    center_rad = np.deg2rad(center)
    half_width = np.deg2rad(width) / 2.0
    half_height = np.deg2rad(height) / 2.0
    cosdec = np.cos(0.5 * (full_lims[2] + full_lims[3]))

    def _xpix(ra):
        return int(round((ra - full_lims[0]) * cosdec / pixsize))

    def _ypix(dec):
        return int(round((dec - full_lims[2]) / pixsize))

    x_lo = np.clip(_xpix(center_rad[0] - half_width), 0, sky_map.nx)
    x_hi = np.clip(_xpix(center_rad[0] + half_width), 0, sky_map.nx)
    y_lo = np.clip(_ypix(center_rad[1] - half_height), 0, sky_map.ny)
    y_hi = np.clip(_ypix(center_rad[1] + half_height), 0, sky_map.ny)

    # minkasi's SkyMap.map is (nx, ny) (RA axis first); crop to the science
    # box, then transpose to this package's (n_y, n_x) convention (Dec axis
    # first, matching BinMapper).
    mean_map_array = np.asarray(result.maps[0].map, dtype=float)[x_lo:x_hi, y_lo:y_hi].T
    weight_array = np.asarray(hits.map, dtype=float)[x_lo:x_hi, y_lo:y_hi].T

    # Pixels with zero hits were never touched by PCG (still at their x0=0
    # initial guess) -- mark them as no-data rather than a genuine zero, so
    # they plot as background instead of blending in with real near-zero
    # signal (matching BinMapper's own nan-for-zero-weight convention).
    mean_map_array = np.where(weight_array > 0, mean_map_array, np.nan)

    return _finalize_map(
        mean_map_array,
        weight_array,
        offsets,
        mapper_kwargs,
        odir,
        "minkasi_mapper.png",
        center_phi_theta=datasets[0].get("center_phi_theta") if datasets else None,
        cbar_label=(signal_cbar_label(datasets[0]["tods"])[1] if datasets else None),
    )
