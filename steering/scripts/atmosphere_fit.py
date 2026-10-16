"""Fit an atmosphere-only model (no map) and checkpoint the posterior samples.

Standalone diagnostic reconstruction, and/or a warm-start seed for
``joint_fit.py`` (pass ``--warm_start <this run's rdir>``).

Example::

    python atmosphere_fit.py --config <odir>/run_0/flag_spikes/config.yaml
"""

import os

import click

REQUIRED_SECTIONS = ["odir", "data", "detector", "response", "noise_std", "minimization", "atmosphere"]
OPTIONAL_SECTIONS = [
    "run",
    "key",
    "resume",
    "plot_every",
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
    """Fit an atmosphere-only model and write `last.pkl` + periodic tod plots."""
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
    rdir, pdir = mani.get_stage_dirs(odir, run, "atmosphere_fit")

    datasets = mani.load_tod_datasets(config, verbose)
    datasets = mani.build_detector(config, datasets, verbose)
    datasets = mani.load_noise_masks(odir, run, datasets, data_keys)

    logger.info(f"\nAtmosphere-only fit ({config['minimization']['n_total_iterations']} iterations):")
    mani.run_kl_fit(
        config,
        datasets,
        rdir=rdir,
        pdir=pdir,
        with_map=False,
        with_atmosphere=True,
        plot_kind="atmosphere",
        resume=bool(config.get("resume", False)),
        verbose=verbose,
    )

    # Save the *effective* (map-free) config alongside `last.pkl`, so
    # `plot_results.py`/`save_fits.py` can rebuild the exact same model later
    # from `--fit_dir` alone.
    effective_config = dict(config)
    effective_config.pop("map", None)
    mani.yaml_save(effective_config, os.path.join(rdir, "config.yaml"))
    logger.info(f"\nWrote atmosphere-only samples to: {rdir}/last.pkl")


if __name__ == "__main__":
    main()
