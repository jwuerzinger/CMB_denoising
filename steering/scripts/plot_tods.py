"""Quick-look plots of raw TODs: detector layout and per-detector data.

Runs before any dejumping/cleaning/flagging/fitting -- a first look at exactly
what's in the input FITS files.

Example::

    python plot_tods.py --config config/ms0735_21A.yml
"""

import os

import click

REQUIRED_SECTIONS = ["odir", "data", "detector"]
OPTIONAL_SECTIONS = ["run", "plot_tods", "verbose"]


@click.command()
@click.option("--config", required=True, help="Path to the input YAML config file.")
@click.option(
    "--cuda_device",
    default="",
    help="CUDA device selection such as '0' or '0,1'; leave empty to force CPU mode.",
)
def main(config, cuda_device):
    """Plot raw detector layout and TOD data, before any processing."""
    if str(cuda_device) == "":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from nifty.re import logger

    import maria_nifty as mani

    config = mani.load_config(config, REQUIRED_SECTIONS, OPTIONAL_SECTIONS)
    verbose = config.get("verbose", True)

    odir = config["odir"]
    run = config.get("run", 0)
    _, pdir = mani.get_stage_dirs(odir, run, "plot_tods")

    datasets = mani.load_tod_datasets(config, verbose)
    datasets = mani.build_detector(config, datasets, verbose)

    plot_tods = int(config.get("plot_tods", len(datasets)))
    plot_datasets = datasets[:plot_tods]

    logger.info("\nPlot raw detectors and TODs:")
    mani.tod_spectrum_plot(plot_datasets, odir=pdir)
    for i, d in enumerate(plot_datasets):
        pdir_tod_i = os.path.join(pdir, f"tod_{i + 1:02d}")
        mani.detector_plot(d["detector"], odir=pdir_tod_i)
        mani.truth_plot(d["detector"], d["tod_data"], odir=pdir_tod_i)


if __name__ == "__main__":
    main()
