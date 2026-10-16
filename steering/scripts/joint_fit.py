"""Fit the atmosphere and map jointly, and checkpoint the posterior samples.

Optionally warm-starts from a prior ``atmosphere_fit.py`` run: the atmosphere
mean and its per-sample residuals are carried over verbatim, with a fresh
random map latent injected per sample.

Example::

    python joint_fit.py --config <odir>/run_0/flag_spikes/config.yaml \\
        --warm_start <odir>/run_0/atmosphere_fit
"""

import os

import click

REQUIRED_SECTIONS = ["odir", "data", "detector", "response", "noise_std", "minimization", "map", "atmosphere"]
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
@click.option(
    "--warm_start",
    default=None,
    help="Path to an atmosphere_fit.py run dir (containing last.pkl) to seed the joint fit's atmosphere from.",
)
def main(config, cuda_device, warm_start):
    """Fit atmosphere+map jointly and write `last.pkl` + periodic map/tod plots."""
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
    rdir, pdir = mani.get_stage_dirs(odir, run, "joint_fit")

    datasets = mani.load_tod_datasets(config, verbose)
    datasets = mani.build_detector(config, datasets, verbose)
    datasets = mani.load_noise_masks(odir, run, datasets, data_keys)

    warm_start_atm = None
    if warm_start:
        logger.info(f"\nWarm-starting the atmosphere from: {warm_start}/last.pkl")
        warm_start_atm = mani.load_samples(warm_start)

    logger.info(f"\nJoint atmosphere+map fit ({config['minimization']['n_total_iterations']} iterations):")
    mani.run_kl_fit(
        config,
        datasets,
        rdir=rdir,
        pdir=pdir,
        with_map=True,
        with_atmosphere=True,
        plot_kind="map",
        warm_start_atm=warm_start_atm,
        resume=bool(config.get("resume", False)),
        verbose=verbose,
    )

    # Save the effective config (both `map` and `atmosphere` present)
    # alongside `last.pkl`, so `plot_results.py`/`save_fits.py` can rebuild
    # the exact same model later from `--fit_dir` alone.
    mani.yaml_save(config, os.path.join(rdir, "config.yaml"))
    logger.info(f"\nWrote joint-fit samples to: {rdir}/last.pkl")


if __name__ == "__main__":
    main()
