"""Run a maria simulation and write talk-quality plots of each TOD component.

A trimmed-down sibling of ``simulate.py``: instead of a YAML config it takes the
simulation setup from CLI options (defaults below), runs the maria simulation
once, and writes a small set of presentation figures styled like
``talk_plots.py`` -- black background, no grid lines, the ``cmb`` colormap and
the same ``.T`` + ``origin="lower"`` display convention.

Figures written to ``<odir>/``:

* ``detectors.png``        -- focal-plane layout, colored by detector number.
* ``tods_atmosphere.png``  -- simulated atmosphere TODs, colored by detector.
* ``tods_map.png``         -- simulated map TODs, colored by detector.
* ``tods_noise.png``       -- simulated white-noise TODs, colored by detector.
* ``tods_total.png``       -- their sum, i.e. what the mapper actually sees.
* ``input_map.png``        -- the truth map fed into the simulation.
* ``bin_map.png``          -- bin-mapper reconstruction of the simulated TODs.

Example::

    pixi run -e cpu python scripts/sim_plots.py --odir runs/SIM/talk
"""

import os

import click

# Matches talk_plots.py's tick formatting for focal-plane coordinates.
ARCMIN2RAD = 2.0 * 3.141592653589793 / 360.0 / 60.0


@click.command()
@click.option("--odir", default="runs/SIM/talk", show_default=True, help="Directory for the output figures.")
@click.option(
    "--map_file",
    default="maps/cluster3.fits",
    show_default=True,
    help="Input truth map: a maria-data path fetched via maria.io.fetch (same map simulate.py uses). "
    "Pass --local_map to read a local FITS file instead.",
)
@click.option("--local_map", is_flag=True, help="Treat --map_file as a local FITS path, skipping the fetch.")
@click.option(
    "--map_center",
    default="144.0,12.0",
    show_default=True,
    help="Map center 'ra,dec' in degrees. Must be above the horizon at --site for --start_time.",
)
@click.option(
    "--map_res",
    default=8.714e-05,
    show_default=True,
    help="Input-map pixel size in degrees; 0 keeps the FITS file's own WCS scale. With the default "
    "--map_width this gives a 1000x1000 grid over 5.23 arcmin.",
)
@click.option(
    "--map_width",
    default=0.08714,
    show_default=True,
    help="Input-map angular width in degrees (0 to leave it to the file's WCS).",
)
@click.option("--map_units", default="Jy/pixel", show_default=True, help="Units of the input map's pixel values.")
@click.option("--map_nu", default=93.0e9, show_default=True, help="Reference frequency of the input map in Hz.")
@click.option("--map_frame", default="ra_dec", show_default=True, help="Coordinate frame of the input map.")
@click.option(
    "--map_scale",
    default=1.0 / 8.0,
    show_default=True,
    help="Multiply the input map by this before simulating (simulate.py's own 1/8 scaling).",
)
@click.option("--instrument", default="mustang-2", show_default=True, help="maria instrument name.")
@click.option("--site", default="green_bank", show_default=True, help="maria site name.")
@click.option("--scan_pattern", default="daisy", show_default=True, help="maria scan pattern.")
@click.option(
    "--duration",
    default=450.0,
    show_default=True,
    help="Scan duration in seconds. Map noise scales as 1/sqrt(duration); too short and the "
    "bin-mapper panel is pure noise.",
)
@click.option("--sample_rate", default=50.0, show_default=True, help="Sample rate in Hz.")
@click.option("--scan_radius", default=2.0, show_default=True, help="Scan radius in arcmin.")
@click.option(
    "--start_time",
    default="2017-04-15T04:00:00",
    show_default=True,
    help="UTC start of the scan. Must place the target above the horizon for --site, or maria "
    "raises PointingError; the default has the RXJ1347 field up at Green Bank.",
)
@click.option(
    "--atmosphere",
    default="2d",
    show_default=True,
    help="maria atmosphere model, or 'none' for a noiseless-sky run. The simulated atmosphere is "
    "~3.9 K rms against a ~1e-3 K cluster, so removing it is the only way the bin-mapper panel "
    "shows the cluster without aggressive filtering that also destroys it.",
)
@click.option(
    "--binmap_atm",
    default=False,
    show_default=True,
    help="Include the simulated atmosphere in the TODs handed to the bin mapper. The atmosphere is "
    "~3.9 K rms against a ~1e-3 K cluster, so leaving it out is what lets one run produce both an "
    "atmosphere TOD panel and a bin map that actually shows the input map.",
)
@click.option(
    "--binmap_preprocessing",
    type=click.Choice(["full", "light", "none"]),
    default="full",
    show_default=True,
    help="'full' is talk_plots' real-data chain (spline + mode 1 + 0.1-10 Hz bandpass + tukey "
    "window), which removes ~94%% of a slowly-scanned cluster. 'light' is spline + mode 0 only. "
    "'none' does no preprocessing at all -- straight binning, which is the honest best case for "
    "map+noise TODs since every step in the other chains removes some of the cluster.",
)
@click.option("--noise_std", default=2.0e-2, show_default=True, help="White-noise std in K_RJ.")
@click.option("--seed", default=4, show_default=True, help="PRNG seed for the noise realization.")
@click.option(
    "--width",
    default=0.0,
    show_default=True,
    help="Bin-mapper map width in degrees; 0 uses the input map's realized field of view so the "
    "reconstruction covers exactly the same sky as the input-map panel.",
)
@click.option("--resolution", default=2.0 / 3600, show_default=True, help="Bin-mapper pixel size in degrees.")
@click.option(
    "--map_vmin",
    type=float,
    default=None,
    help="Lower color limit for the map panels (K_RJ). Default: the data's 0.5th percentile, "
    "which ignores the handful of very deep bin-mapper edge pixels.",
)
@click.option(
    "--map_vmax",
    type=float,
    default=0.0,
    show_default=True,
    help="Upper color limit for the map panels (K_RJ). 0 suits a pure-decrement map -- the whole "
    "color range then covers the negative signal instead of half of it being wasted on positives. "
    "Pass a negative/positive value to override, and note the cmb colormap's neutral point no "
    "longer lands on zero once the scale is asymmetric.",
)
@click.option(
    "--remove_offset",
    default=True,
    show_default=True,
    help="Subtract each detector's own time-mean from the atmosphere and total TOD panels "
    "(simulate.py's `atm_offset` step). The atmosphere's DC level is far larger than its "
    "structure, so without this the traces collapse onto a few flat lines. Map and noise "
    "panels are left untouched -- their offsets are signal.",
)
@click.option(
    "--ra_step_sec",
    default=5.0,
    show_default=True,
    help="Spacing between RA ticks in seconds of time; increase for a wider field.",
)
@click.option("--max_dets", default=0, show_default=True, help="Plot only the first N detectors (0 = all).")
@click.option("--background", default="black", show_default=True, help="Figure background: 'black' or 'white'.")
@click.option("--grid", default=False, show_default=True, help="Draw grid lines on the plots.")
@click.option(
    "--cuda_device",
    default="",
    help="CUDA device selection such as '0'; leave empty to force CPU mode.",
)
def main(
    odir,
    map_file,
    local_map,
    map_center,
    map_res,
    map_width,
    map_units,
    map_nu,
    map_frame,
    map_scale,
    instrument,
    site,
    scan_pattern,
    duration,
    sample_rate,
    scan_radius,
    start_time,
    atmosphere,
    binmap_atm,
    binmap_preprocessing,
    noise_std,
    seed,
    width,
    resolution,
    map_vmin,
    map_vmax,
    remove_offset,
    ra_step_sec,
    max_dets,
    background,
    grid,
    cuda_device,
):
    """Simulate TODs once and write the presentation figures for them."""
    if str(cuda_device) == "":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    os.environ.setdefault("MARIA_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "maria-data"))
    os.makedirs(os.environ["MARIA_CACHE_DIR"], exist_ok=True)

    import dask.array as da
    import jax
    import maria
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from nifty.re import logger

    import maria_nifty as mani
    from maria_nifty.plot.util import apply_sky_ticks

    # ---------------------------------------------------------------- style
    # Same treatment as talk_plots.py: pin the facecolors so bbox_inches="tight"
    # cannot reintroduce a white border, and route the grid through one toggle.
    background = str(background).lower()
    if background == "black":
        plt.style.use("dark_background")
        fg, grid_color = "white", "0.25"
    else:
        plt.style.use("default")
        fg, grid_color = "black", "0.85"
    for key in ("figure.facecolor", "axes.facecolor", "savefig.facecolor", "savefig.edgecolor"):
        plt.rcParams[key] = background
    plt.rcParams["axes.grid"] = bool(grid)

    def set_grid(ax, **line_kwargs):
        """Toggle an axis grid per --grid (see talk_plots.set_grid)."""
        if grid:
            ax.grid(True, color=grid_color, **line_kwargs)
        else:
            ax.grid(False)

    label_fs, tick_fs = 15.0, 12.5
    os.makedirs(odir, exist_ok=True)

    # ----------------------------------------------------------- simulation
    logger.info("\nStart simulation:")
    ra, dec = (float(v) for v in str(map_center).split(","))
    path = map_file if local_map else maria.io.fetch(map_file)
    load_kwargs = {"center": (ra, dec), "frame": map_frame, "units": map_units, "nu": float(map_nu)}
    if float(map_res) > 0:
        load_kwargs["resolution"] = float(map_res)
    if float(map_width) > 0:
        load_kwargs["width"] = float(map_width)
    input_map = maria.map.load(filename=path, **load_kwargs)
    input_map.data = input_map.data * float(map_scale)

    plan = maria.get_plan(
        scan_pattern=scan_pattern,
        scan_center=(ra, dec),
        frame=map_frame,
        duration=duration,
        sample_rate=sample_rate,
        start_time=start_time,
        scan_options={"radius": scan_radius / 60.0},
    )
    sim = maria.Simulation(
        maria.get_instrument(instrument),
        plans=plan,
        site=site,
        map=input_map,
        atmosphere=None if str(atmosphere).lower() in ("none", "") else atmosphere,
    )
    tods = sim.run()[0]

    # Rebuild the map component through this repo's own beam-convolved
    # projection (as simulate.py does) so the "map TODs" panel shows exactly
    # what the forward model would produce, not maria's internal version.
    sim_map = np.array(sim.map.data[0, 0])
    offsets = tods.coords.offsets(frame=sim.map.frame)
    fov = np.array([x.max() - x.min() for x in (sim.map.y_side[::-1], sim.map.x_side)])
    coords = mani.offsets2coords(offsets, sim_map.shape, fov)
    sigma_pixels = float(tods.dets.fwhm[0] / np.sqrt(8 * np.log(2)) / sim.map.resolution)
    tods_map = np.asarray(mani.map2tods(sim_map, coords, sigma_pixels))
    logger.info(f"input map {sim_map.shape}, fov {fov}, beam sigma {sigma_pixels:.3f} px")

    has_atm = "atmosphere" in tods.data
    tods_atm = np.array(tods.data.pop("atmosphere")) if has_atm else np.zeros_like(tods_map)
    rng = jax.random.PRNGKey(int(seed))
    tods_noise = np.asarray(float(noise_std) * jax.random.normal(rng, np.array(tods.data["noise"]).shape))

    # Component amplitudes -- if map_rms is well below noise_rms/sqrt(hits per
    # map pixel), the bin-mapper panel will be noise and needs either a bigger
    # --map_scale, a smaller --noise_std, or a longer --duration.
    n_pix = ((float(width) if float(width) > 0 else float(np.rad2deg(fov[0]))) / float(resolution)) ** 2
    hits = tods_map.size / max(n_pix, 1.0)
    map_rms, noise_rms = float(tods_map.std()), float(tods_noise.std())
    logger.info(
        f"component rms [K_RJ]: atmosphere={float(tods_atm.std()):.4e}  map={map_rms:.4e}  noise={noise_rms:.4e}\n"
        f"  map peak={float(np.abs(tods_map).max()):.4e}; ~{hits:.0f} samples/map-pixel "
        f"-> binned noise ~{noise_rms / max(np.sqrt(hits), 1.0):.4e}, "
        f"map/noise per pixel ~{map_rms / max(noise_rms / max(np.sqrt(hits), 1.0), 1e-30):.2f}"
    )

    # TOD.signal sums every field, so the mapper below sees atm + map + noise.
    if has_atm:
        tods.data["atmosphere"] = da.from_array(tods_atm)
    tods.data["map"] = da.from_array(tods_map)
    tods.data["noise"] = da.from_array(tods_noise)

    # ------------------------------------------------------------- plotting
    coords_det = np.asarray(tods.dets.offsets, dtype=float)
    n_det = coords_det.shape[0]
    sel = slice(None) if int(max_dets) <= 0 else slice(0, int(max_dets))
    colors = np.arange(n_det, dtype=float)[sel]
    cmap = plt.colormaps.get_cmap("cmb")
    norm = mcolors.Normalize(vmin=float(colors.min()), vmax=float(colors.max()))
    fs = float(1.0 / tods.dt)

    def detector_plot():
        """Focal-plane scatter, one color per detector number."""
        fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
        ax.scatter(
            coords_det[sel, 0],
            coords_det[sel, 1],
            c=colors,
            cmap=cmap,
            norm=norm,
            s=88,
            linewidths=0.35,
            edgecolors=fg,
        )
        ax.set_aspect("equal", adjustable="box")
        # Tick at whole arcmin, so labels are unique (a plain ".0f" formatter on
        # matplotlib's own radian ticks rounds neighbours to the same integer).
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MultipleLocator(ARCMIN2RAD))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / ARCMIN2RAD:.0f}"))
        ax.set_xlabel("Detector x [arcmin]", fontsize=label_fs)
        ax.set_ylabel("Detector y [arcmin]", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        set_grid(ax, linewidth=0.7)
        ax.set_axisbelow(True)
        fig.savefig(os.path.join(odir, "detectors.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def tod_plot(arr, ylabel, fname, drop_offset=False):
        """One TOD panel, traces colored by detector number.

        ``drop_offset`` removes each detector's own time-mean, so the panel
        shows structure rather than the per-detector DC level.
        """
        arr = np.asarray(arr, dtype=float)[sel]
        if drop_offset:
            arr = arr - arr.mean(axis=1, keepdims=True)
        t_min = np.arange(arr.shape[-1]) / fs / 60.0
        fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
        for i, row in enumerate(arr):
            ax.plot(t_min, row, color=cmap(norm(colors[i])), alpha=0.45, linewidth=0.9)
        ax.set_xlabel("Time [minutes]", fontsize=label_fs)
        ax.set_ylabel(ylabel, fontsize=label_fs)
        ax.tick_params(axis="both", which="both", labelsize=tick_fs)
        set_grid(ax, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xmargin(0)
        half = 0.5 / fs / 60.0
        ax.set_xlim(t_min[0] - half, t_min[-1] + half)
        fig.savefig(os.path.join(odir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def map_plot(arr, extent, fname, center_phi_theta=None, lims=None):
        """Sky map in talk_plots' display convention (.T + origin="lower").

        Limits come from --map_vmin/--map_vmax; the defaults (data minimum to
        0) suit a decrement-only map rather than forcing a symmetric scale.
        """
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        # A robust low percentile, not the raw minimum: bin-mapper edge pixels
        # sit several times deeper than the source and would otherwise set the
        # whole scale, washing the real signal out to a flat color.
        lo = float(np.percentile(finite, 0.5)) if finite.size else -1.0
        vmin = lo if map_vmin is None else float(map_vmin)
        vmax = 0.0 if map_vmax is None else float(map_vmax)
        if not vmax > vmin:  # e.g. an all-positive array against a 0 ceiling
            hi = float(np.max(finite)) if finite.size else 1.0
            vmin, vmax = lo, max(hi, lo + 1e-12)
            logger.warning(f"{fname}: --map_vmax={map_vmax} is not above the data minimum; using the data range.")
        fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
        img = ax.imshow(arr.T, cmap="cmb", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
        if lims is not None:
            # Set before apply_sky_ticks: it reads the current limits to place
            # the sky ticks and ends by inverting xlim to flip RA, so setting
            # them afterwards would undo that flip.
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[2], lims[3])
        # Absolute RA/Dec ticks plus the secondary offset-arcmin axis, matching
        # talk_plots' map figures (apply_sky_ticks also flips the RA axis).
        if center_phi_theta is not None:
            apply_sky_ticks(
                ax,
                center_phi_theta,
                xlabel=True,
                ylabel=True,
                secondary=True,
                ra_step_sec=ra_step_sec,
            )
        else:
            ax.set_xlabel(r"$\Delta\theta$ [arcmin]", fontsize=label_fs)
            ax.set_ylabel(r"$\Delta\theta$ [arcmin]", fontsize=label_fs)
        ax.grid(False)
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02, label=r"Signal [K$_\mathrm{RJ}$]")
        fig.savefig(os.path.join(odir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("\nPlotting...")
    detector_plot()
    if has_atm:
        tod_plot(tods_atm, r"Atmosphere [K$_\mathrm{RJ}$]", "tods_atmosphere.png", drop_offset=remove_offset)
    tod_plot(tods_map, r"Map signal [K$_\mathrm{RJ}$]", "tods_map.png")
    tod_plot(tods_noise, r"Noise [K$_\mathrm{RJ}$]", "tods_noise.png")
    tod_plot(
        tods_atm + tods_map + tods_noise,
        r"Total [K$_\mathrm{RJ}$]",
        "tods_total.png",
        drop_offset=remove_offset,
    )

    center_phi_theta = (np.deg2rad(ra), np.deg2rad(dec))
    half_fov_amin = 0.5 * np.rad2deg(fov) * 60.0
    # One shared field of view, taken from the input map, so the truth and
    # reconstruction panels are directly comparable pixel-for-pixel on screen.
    shared_lims = [-half_fov_amin[1], half_fov_amin[1], -half_fov_amin[0], half_fov_amin[0]]
    map_plot(
        sim_map.T,
        shared_lims,
        "input_map.png",
        center_phi_theta=center_phi_theta,
        lims=shared_lims,
    )

    # -------------------------------------------------------- bin-mapper run
    # Use the input map's *realized* fov, not --map_width: maria keeps the file's
    # native pixel count at --map_res, so the loaded map is generally a little
    # wider than the requested width (1024 x 8.714e-05 deg = 0.08923, not
    # 0.08714). Matching the realized fov is what makes the two panels fill the
    # same frame instead of leaving a margin around the reconstruction.
    binmap_width = float(width) if float(width) > 0 else float(np.rad2deg(fov[0]))
    logger.info(f"\nRun bin mapper (width={binmap_width:.5f} deg):")
    if has_atm and not binmap_atm:
        # TOD.signal sums every field, so dropping this one is what excludes
        # the atmosphere from the mapper -- the plotted atmosphere panel above
        # already has its own copy of the array.
        tods.data.pop("atmosphere", None)
        logger.info("  atmosphere excluded from the mapper (--binmap_atm to include it)")
    datasets = [{"tods": tods, "offsets": offsets, "center_phi_theta": center_phi_theta}]
    binmap, binmap_space = mani.run_bin_mapper(
        datasets,
        odir,
        True,
        center=(ra, dec),
        width=binmap_width,
        resolution=resolution,
        tod_preprocessing=(
            {}
            if binmap_preprocessing == "none"
            else {
                "remove_spline": {"knot_spacing": 6, "remove_el_gradient": True},
                "remove_modes": {"modes_to_remove": [1]},
                "filter": {"f_lower": 0.1, "f_upper": 10},
                "window": {"name": "tukey", "kwargs": {"alpha": 0.1}},
            }
            if binmap_preprocessing == "full"
            else {
                # No bandpass and no window: both attack the slow, large-scale
                # signal the cluster lives in. Mode 0 is the strongest common
                # mode -- note "full" removes mode *1*, so it never took the
                # common mode out at all.
                "remove_spline": {"knot_spacing": 6, "remove_el_gradient": True},
                "remove_modes": {"modes_to_remove": [0]},
            }
        ),
        map_postprocessing={"gaussian_filter": {"sigma": 1.0}},
    )
    # run_bin_mapper hands back an array with an internal flip baked in for its
    # own standalone preview. Which axis to undo differs from talk_plots here:
    # this script's input_map panel displays maria's native (y, x) layout, so
    # the bin map needs axis 1 flipped, not axis 0. Verified by matching the
    # sky position of the deepest pixel against the input map -- axis 1 lands
    # 0.12 arcmin away, axis 0 lands 1.63 arcmin away.
    binmap = np.flip(np.asarray(binmap, dtype=float), axis=1)
    if os.path.exists(os.path.join(odir, "bin_mapper.png")):
        os.remove(os.path.join(odir, "bin_mapper.png"))
    extent = None
    if binmap_space is not None:
        f = np.rad2deg(np.asarray(binmap_space.fov)) * 60.0
        extent = [-0.5 * f[0], 0.5 * f[0], -0.5 * f[1], 0.5 * f[1]]
    else:
        extent = [0, binmap.shape[-1], 0, binmap.shape[-2]]
    map_plot(binmap, extent, "bin_map.png", center_phi_theta=center_phi_theta, lims=shared_lims)

    logger.info(f"\nWrote figures to: {odir}")


if __name__ == "__main__":
    main()
