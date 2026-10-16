"""Simulate TOD data and export truth products and plots.

The script runs a Maria simulation for the configured instrument, map, site,
and observing plan, then exports the resulting TODs, truth arrays, and figures.

The ``instrument`` section is either the name of a pre-defined Maria
instrument (e.g. ``mustang-2``) or a dict. A dict may name a pre-defined
instrument under ``name`` and overrides or replaces its settings with the
remaining keys, which are passed to ``maria.get_instrument`` (e.g. a custom
``array`` with its own ``bands``; see ``config/sim_atlast.yml``).

The ``map`` section is passed to ``maria.map.load``, apart from ``filename``
(a local path, or else a path in the maria-data repository that is fetched)
and ``scale`` (a factor applied to the input map, default 1).

The optional ``image`` section replaces the true sky by an image from an
``.npz`` (array ``key``) or ``.npy`` file, centered on the map; ``fov`` is the
image width as a fraction of the map width and ``scale`` a factor applied to
the image (both default 1).

The optional ``plot`` section selects the detectors drawn in the TOD plots:
``dets`` is a number of detectors spread evenly over the array or a list of
detector indices (default: all). ``plot: false`` skips the plots.
"""

import os

import click


def get_instrument(instrument_cfg):
    """Build a Maria instrument from a name or a config dict.

    Parameters
    ----------
    instrument_cfg : str or dict
        Name of a pre-defined Maria instrument, or a dict with an optional
        ``name`` of a pre-defined instrument plus keyword arguments for
        ``maria.get_instrument`` (e.g. ``array``).

    Returns
    -------
    maria.Instrument
        The configured instrument.
    """
    import maria

    if isinstance(instrument_cfg, str):
        return maria.get_instrument(instrument_cfg)
    instrument_cfg = dict(instrument_cfg)
    return maria.get_instrument(instrument_cfg.pop("name", None), **instrument_cfg)


def load_image(image_cfg, map_shape):
    """Load an image and resample it onto the simulation's map grid.

    Parameters
    ----------
    image_cfg : dict
        ``fname`` of an ``.npz`` or ``.npy`` file, ``key`` of the array in an
        ``.npz`` file (default ``original``), ``fov`` (image width as a
        fraction of the map width, default 1) and ``scale`` (factor applied
        to the image, default 1).
    map_shape : tuple of int
        Pixel shape of the simulation's map.

    Returns
    -------
    numpy.ndarray
        The image on the map grid, zero outside the image.
    """
    import numpy as np

    import maria_nifty as mani

    data = np.load(image_cfg["fname"])
    arr = data[image_cfg.get("key", "original")] if isinstance(data, np.lib.npyio.NpzFile) else data
    fov = float(image_cfg.get("fov", 1.0))
    in_space = mani.GridSpace.build(shape=arr.shape, fov=(fov, fov))
    out_space = mani.GridSpace.build(shape=map_shape, fov=(1.0, 1.0))
    image = mani.map2space(in_space, out_space)(np.asarray(arr, dtype="float64"), x=None)
    return np.asarray(image) * image_cfg.get("scale", 1.0)


def detector_subset(dets, n_dets):
    """Indices of the detectors to plot.

    Parameters
    ----------
    dets : int, list of int or None
        Number of detectors spread evenly over the array, or explicit
        detector indices. `None` selects all detectors.
    n_dets : int
        Total number of detectors.

    Returns
    -------
    numpy.ndarray
        Sorted detector indices.
    """
    import numpy as np

    if dets is None:
        return np.arange(n_dets)
    if isinstance(dets, int):
        return np.unique(np.linspace(0, n_dets - 1, min(dets, n_dets)).round().astype(int))
    return np.asarray(sorted(dets), dtype=int)


@click.command()
@click.option("--config", required=True, help="Path to the simulation YAML config file.")
@click.option(
    "--cuda_device",
    default="",
    help="CUDA device selection such as '0' or '0,1'; leave empty to force CPU mode.",
)
def main(config, cuda_device):
    """Run the synthetic TOD simulation workflow.

    Parameters
    ----------
    config : str
        Path to the simulation YAML configuration file.
    cuda_device : str
        CUDA device selection passed through to JAX. An empty string enables
        CPU execution.

    Side Effects
    ------------
    - Sets JAX/CUDA environment variables before importing libraries.
    - Creates simulation output directories.
    - Writes simulated TODs, truth arrays, and diagnostic plots.
    """
    if str(cuda_device) == "":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # Use a user-writable cache directory for maria downloads.
    user_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "maria-data")
    os.environ.setdefault("MARIA_CACHE_DIR", user_cache_dir)
    os.makedirs(os.environ["MARIA_CACHE_DIR"], exist_ok=True)

    import dask.array as da
    import jax
    import maria
    import numpy as np
    from nifty.re import logger

    import maria_nifty as mani

    rng = jax.random.PRNGKey(4)

    REQUIRED_SECTIONS = ["tdir", "odir", "instrument", "site", "map", "plan", "noise"]
    OPTIONAL_SECTIONS = ["run", "atm", "image", "plot", "verbose"]

    # load simulation config
    config = mani.yaml_load(config)
    mani.check_dict(config, REQUIRED_SECTIONS, OPTIONAL_SECTIONS)

    tdir = os.path.join(config["tdir"], f"sim_{config.get('run', 0)}")
    sdir = os.path.join(config["odir"], f"sim_{config.get('run', 0)}")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(sdir, exist_ok=True)
    mani.yaml_save(config, f"{tdir}/config.yaml")
    mani.yaml_save(config, f"{sdir}/config.yaml")

    # run maria simulation
    logger.info("\nStart simulation:")
    instrument = get_instrument(config.pop("instrument"))
    logger.info(f"instrument: {instrument.n} detectors")

    map_fname = config["map"].pop("filename")
    if not os.path.exists(map_fname):
        map_fname = maria.io.fetch(map_fname)
    map_scale = config["map"].pop("scale", 1.0)
    input_map = maria.map.load(filename=map_fname, **config["map"])
    input_map.data = input_map.data * map_scale

    plan = maria.get_plan(scan_center=config["map"]["center"], frame=config["map"]["frame"], **config["plan"])
    sim = maria.Simulation(
        instrument,
        plans=plan,
        site=config["site"],
        map=input_map,
        atmosphere="2d",
    )
    tods = sim.run()[0]

    # jaxify map tods
    sim_map = np.array(sim.map.data[0, 0])

    # switch to another input image if specified in config
    if config.get("image"):
        sim_map = load_image(config["image"], sim_map.shape)

    offsets = tods.coords.offsets(frame=sim.map.frame)
    fov = np.array([x.max() - x.min() for x in (sim.map.y_side[::-1], sim.map.x_side)])
    coords = mani.offsets2coords(offsets, sim_map.shape, fov)
    logger.info(f"offsets shape: {offsets.shape}")
    logger.info(f"map fov: {fov}")

    sigma_rad = tods.dets.fwhm[0] / np.sqrt(8 * np.log(2))
    sigma_pixels = float(sigma_rad / sim.map.resolution)
    tods_map = mani.map2tods(sim_map, coords, sigma_pixels)
    logger.info(f"sigma rad: {sigma_rad}")
    logger.info(f"map resolution: {sim.map.resolution}")
    logger.info(f"sigma pixels: {sigma_pixels}")

    tods.data["map"] = da.from_array(tods_map)

    # option to simplify atmosphere tods
    tods_atm = np.array(tods.data.pop("atmosphere"))
    if "atm" in config:
        atm_config = config["atm"]

        if not atm_config.get("offsets", False):
            atm_offset = tods_atm.mean(axis=1, keepdims=True)
            tods_atm -= atm_offset

        if not atm_config.get("slopes", False):
            atm_slopes = (tods_atm / tods_atm[0]).mean(axis=1, keepdims=True)
            tods_atm /= atm_slopes

        tods.data["atmosphere"] = da.from_array(tods_atm)

    # jaxify noise tods
    tods_noise = np.array(tods.data["noise"])

    noise_std = config["noise"]
    rng, key = jax.random.split(rng)
    tods_noise = noise_std * jax.random.normal(key, tods_noise.shape)

    tods.data["noise"] = da.from_array(tods_noise)

    # save tods and true components; maria writes an all-zero ELEV column, so
    # stamp the simulated boresight elevation (in degrees) into it
    tods_fname = os.path.join(tdir, "tods.fits")
    tods.to_fits(tods_fname)
    boresight_el = np.degrees(np.asarray(tods.boresight.el, dtype="float64"))
    mani.write_boresight_elevation(tods_fname, boresight_el, tods_map.shape[0])
    logger.info(f"boresight elevation: {boresight_el.min():.2f} to {boresight_el.max():.2f} deg")

    np.savez(
        file=os.path.join(tdir, "truth.npz"),
        map=sim_map,
        tods_map=np.array(tods_map),
        tods_atm=np.array(tods_atm),
        tods_noise=np.array(tods_noise),
        allow_pickle=False,
    )

    # plotting
    plot_cfg = config.get("plot", {})
    if plot_cfg is False:
        return
    logger.info("\nPlotting...")
    mani.image_plot(
        array=sim_map,
        title="input map",
        fname=os.path.join(sdir, "map", "map.png"),
        vmin=-np.max(np.abs(sim_map)),
        vmax=np.max(np.abs(sim_map)),
    )

    dets = detector_subset((plot_cfg or {}).get("dets"), tods_map.shape[0])
    logger.info(f"plotting {dets.size} of {tods_map.shape[0]} detectors")
    components = {"map": np.asarray(tods_map)[dets], "noise": np.asarray(tods_noise)[dets]}
    if "atm" in config:
        components = {"atmosphere": tods_atm[dets], **components}

    sdir_tods = os.path.join(sdir, "tods")
    mani.line_plot(
        array=sum(components.values()),
        title="tods",
        figsize=(10, 2.5),
        fname=os.path.join(sdir_tods, "tods.png"),
    )
    mani.grid_plot(
        array=list(components.values()),
        rows=len(components),
        mode="line",
        title=[f"{name} tods" for name in components],
        figsize=(10, 2.5),
        fname=os.path.join(sdir_tods, "tod_components.png"),
    )


if __name__ == "__main__":
    main()
