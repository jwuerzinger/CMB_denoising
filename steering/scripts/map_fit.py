"""Fit a map-only model (no atmosphere) and checkpoint the posterior samples.

Use this once the atmosphere has already been removed at the TOD level (by
``clean_tods.py``); any ``atmosphere`` section in the config is ignored. For
fitting the atmosphere jointly with the map instead, see ``joint_fit.py``.

Example::

    python map_fit.py --config <odir>/run_0/flag_spikes/config.yaml
"""

import os

import click

REQUIRED_SECTIONS = ["odir", "data", "detector", "response", "noise_std", "minimization", "map"]
OPTIONAL_SECTIONS = [
    "run",
    "key",
    "resume",
    "plot_every",
    "plot_extra",
    "plot_tods",
    "clear_every",
    "fused",
    "verbose",
    "kl_device_map",
    "residual_device_map",
    "residual_map",
]


@click.command()
@click.option("--config", required=True, help="Path to the input YAML config file.")
@click.option(
    "--cuda_device",
    default="",
    help="CUDA device selection such as '0' or '0,1'; leave empty to force CPU mode.",
)
def main(config, cuda_device):
    """Fit a map-only model and write `last.pkl` + periodic map/tod plots."""
    if str(cuda_device) == "":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from nifty.re import logger

    import maria_nifty as mani

    config = mani.load_config(config, REQUIRED_SECTIONS, OPTIONAL_SECTIONS)
    verbose = config.get("verbose", True)
    if "fname" in config["data"]:
        config["data"] = {"d1": config["data"]}
    data_keys = [k for k in config["data"] if k != "scan_center"]

    odir = config["odir"]
    run = config.get("run", 0)
    rdir, pdir = mani.get_stage_dirs(odir, run, "map_fit")

    datasets = mani.load_tod_datasets(config, verbose)
    datasets = mani.build_detector(config, datasets, verbose)
    datasets = mani.load_noise_masks(odir, run, datasets, data_keys)

    logger.info(f"\nMap-only fit ({config['minimization']['n_total_iterations']} iterations):")
    mani.run_kl_fit(
        config,
        datasets,
        rdir=rdir,
        pdir=pdir,
        with_map=True,
        with_atmosphere=False,
        plot_kind="map",
        resume=bool(config.get("resume", False)),
        verbose=verbose,
    )

    # Save the *effective* (atmosphere-free) config alongside `last.pkl`, so
    # `plot_results.py`/`save_fits.py` can rebuild the exact same model later
    # from `--fit_dir` alone.
    effective_config = dict(config)
    effective_config.pop("atmosphere", None)
    mani.yaml_save(effective_config, os.path.join(rdir, "config.yaml"))
    logger.info(f"\nWrote map-fit samples to: {rdir}/last.pkl")


if __name__ == "__main__":
    main()
