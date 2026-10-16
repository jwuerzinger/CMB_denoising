"""Remove jumps from raw TODs with minkasi (fit and subtract the jump steps).

Writes the dejumped FITS files and an updated config (``data.*.fname``
repointed at the dejumped files) that the next stage (``clean_tods.py``) reads
directly. Settings come from the config's ``dejump`` section (defaults:
``maria_nifty.DEJUMP_DEFAULTS``). Spikes are detected later, by
``flag_spikes.py``.

Example::

    python remove_jumps.py --config config/ms0735_21A.yml
"""

import os

import click
import numpy as np

REQUIRED_SECTIONS = ["odir", "data"]
OPTIONAL_SECTIONS = ["run", "dejump", "plot_tods", "verbose"]


@click.command()
@click.option("--config", required=True, help="Path to the input YAML config file.")
@click.option(
    "--cuda_device",
    default="",
    help="CUDA device selection such as '0' or '0,1'; leave empty to force CPU mode.",
)
def main(config, cuda_device):
    """Dejump raw TODs and write the result as a new config + FITS files."""
    if str(cuda_device) == "":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from dask import array as da
    from nifty.re import logger

    import maria_nifty as mani

    config = mani.load_config(config, REQUIRED_SECTIONS, OPTIONAL_SECTIONS)
    verbose = config.get("verbose", True)
    if "fname" in config["data"]:
        config["data"] = {"d1": config["data"]}
    data_keys = [k for k in config["data"] if k != "scan_center"]

    odir = config["odir"]
    run = config.get("run", 0)
    rdir, pdir = mani.get_stage_dirs(odir, run, "remove_jumps")

    dejump_cfg = {**mani.DEJUMP_DEFAULTS, **(config.get("dejump") or {})}
    plot_tods = int(config.get("plot_tods", len(data_keys)))

    raw_datasets = mani.load_tod_datasets(config, verbose=False)

    logger.info("\nDejump raw TODs:")
    for i, (dkey, d) in enumerate(zip(data_keys, raw_datasets)):
        o_tod = d["tods"]
        fs = float(getattr(o_tod, "fs", 1.0) or 1.0)
        signal = np.asarray(o_tod.data["data"].compute(), dtype=np.float64)
        dat_clean, n_jumps = mani.dejump(signal, fs, dejump_cfg)
        if verbose:
            logger.info(
                f"  {dkey}: dejumped {int(n_jumps.sum())} jump(s) across {int((n_jumps > 0).sum())} detector(s)"
            )

        dj_tod = o_tod.copy()
        dj_tod.data["data"] = da.asarray(dat_clean, dtype=dj_tod.dtype)
        dj_fname = os.path.join(rdir, f"{dkey}.fits")
        dj_tod.to_fits(dj_fname, format="MUSTANG-2", overwrite=True)
        mani.write_boresight_elevation(dj_fname, d.get("boresight_el"), signal.shape[0])
        config["data"][dkey]["fname"] = dj_fname

        if i < plot_tods:
            mani.grid_plot(
                array=[signal, dat_clean],
                rows=2,
                mode="line",
                title=["raw tods", "dejumped tods"],
                figsize=(10, 2.5),
                fname=os.path.join(pdir, f"tod_{i + 1:02d}_dejumped.png"),
            )

    mani.yaml_save(config, os.path.join(rdir, "config.yaml"))
    logger.info(f"\nWrote dejumped config to: {os.path.join(rdir, 'config.yaml')}")


if __name__ == "__main__":
    main()
