"""Run Maria's bin mapper on whatever TOD inputs the config points to.

A simple, independent diagnostic map product -- not part of the NIFTy
fit chain, so it can be pointed at raw, dejumped, cleaned, or flagged data.

Example::

    python bin_mapper.py --config <odir>/run_0/flag_spikes/config.yaml
"""

import os

import click

REQUIRED_SECTIONS = ["odir", "data", "bin_mapper"]
OPTIONAL_SECTIONS = ["run", "verbose"]


@click.command()
@click.option("--config", required=True, help="Path to the input YAML config file.")
@click.option(
    "--cuda_device",
    default="",
    help="CUDA device selection such as '0' or '0,1'; leave empty to force CPU mode.",
)
def main(config, cuda_device):
    """Run the Maria bin-mapper and save its map + preview plot."""
    if str(cuda_device) == "":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import maria_nifty as mani

    config = mani.load_config(config, REQUIRED_SECTIONS, OPTIONAL_SECTIONS)
    verbose = config.get("verbose", True)

    odir = config["odir"]
    run = config.get("run", 0)
    _, pdir = mani.get_stage_dirs(odir, run, "bin_mapper")

    datasets = mani.load_tod_datasets(config, verbose)
    mani.run_bin_mapper(datasets, pdir, verbose, **config["bin_mapper"])


if __name__ == "__main__":
    main()
