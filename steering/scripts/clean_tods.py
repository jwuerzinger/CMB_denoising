"""Clean TODs by fitting and subtracting a short atmosphere-only model.

This is ``clean_and_run.py``'s cleaning stage as its own script: a short fused
atmosphere-only NIFTy fit (``cleaning`` config section), then subtract the
posterior-mean atmosphere reconstruction from each raw TOD and write the
result as the cleaned FITS file. (Not the same mechanism as the old
``steering/clean_tods.py``, which ran Maria's own ``BinMapper`` DSP
preprocessing -- that path isn't used here.)

Example::

    python clean_tods.py --config <odir>/run_0/remove_jumps/config.yaml
"""

import os

import click
import numpy as np
from dask import array as da

REQUIRED_SECTIONS = ["odir", "data", "detector", "response", "noise_std", "minimization", "cleaning"]
OPTIONAL_SECTIONS = ["run", "key", "plot_tods", "plot_every", "clear_every", "resume", "verbose"]


def _clean_fit_config(config):
    """Build the atmosphere-only spline-fit config used to clean the TODs."""
    clean_cfg = config.get("cleaning") or {}
    if "spline_params" not in clean_cfg:
        raise ValueError("clean_tods.py requires a `cleaning.spline_params` section (see module docstring).")

    fit_config = dict(config)
    fit_config["fused"] = True
    fit_config["atmosphere"] = {"mode": "spline", "spline_params": clean_cfg["spline_params"]}
    fit_config["minimization"] = dict(config["minimization"])
    fit_config["minimization"]["sample_mode"] = clean_cfg.get("sample_mode", "linear_resample")
    fit_config["minimization"]["n_total_iterations"] = int(clean_cfg.get("n_iterations", 4))
    return fit_config


@click.command()
@click.option("--config", required=True, help="Path to the input YAML config file.")
@click.option(
    "--cuda_device",
    default="",
    help="CUDA device selection such as '0' or '0,1'; leave empty to force CPU mode.",
)
def main(config, cuda_device):
    """Fit and subtract a short atmosphere-only model; write cleaned TODs."""
    if str(cuda_device) == "":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import nifty.re as jft
    from nifty.re import logger

    import maria_nifty as mani

    config = mani.load_config(config, REQUIRED_SECTIONS, OPTIONAL_SECTIONS)
    verbose = config.get("verbose", True)
    if "fname" in config["data"]:
        config["data"] = {"d1": config["data"]}
    data_keys = [k for k in config["data"] if k != "scan_center"]

    odir = config["odir"]
    run = config.get("run", 0)
    rdir, pdir = mani.get_stage_dirs(odir, run, "clean_tods")

    fit_config = _clean_fit_config(config)
    datasets = mani.load_tod_datasets(fit_config, verbose)
    original_tods = [d["tods"] for d in datasets]
    datasets = mani.build_detector(fit_config, datasets, verbose)

    logger.info(
        f"\nClean TODs (atmosphere-only NIFTy fit, {fit_config['minimization']['n_total_iterations']} iterations):"
    )
    samples, datasets = mani.run_kl_fit(
        fit_config,
        datasets,
        rdir=rdir,
        pdir=pdir,
        with_map=False,
        with_atmosphere=True,
        plot_kind="atmosphere",
        resume=bool(config.get("resume", False)),
        verbose=verbose,
    )

    plot_tods = int(config.get("plot_tods", len(data_keys)))
    for i, dkey in enumerate(data_keys):
        d = datasets[i]
        atm_rec = np.asarray(jft.mean(tuple(d["model"](s) for s in samples)))  # (n_valid, T)
        atm_full = mani.scatter_to_raw(atm_rec, d["detector"])  # (n_raw, T)

        o_tod = original_tods[i]
        o_signal = np.asarray(o_tod.data["data"].compute(), dtype=np.float64)
        c_signal = o_signal - atm_full

        c_tods = o_tod.copy()
        c_tods.data["data"] = da.asarray(c_signal, dtype=c_tods.dtype)
        cleaned_fname = os.path.join(rdir, f"{dkey}.fits")
        c_tods.to_fits(cleaned_fname, format="MUSTANG-2", overwrite=True)
        mani.write_boresight_elevation(cleaned_fname, d.get("boresight_el"), o_signal.shape[0])
        config["data"][dkey]["fname"] = cleaned_fname

        if i < plot_tods:
            mani.grid_plot(
                array=[o_signal, c_signal],
                rows=2,
                mode="line",
                title=["original tods", "cleaned tods"],
                figsize=(10, 2.5),
                fname=os.path.join(pdir, f"tod_{i + 1:02d}_cleaned.png"),
            )

    mani.yaml_save(config, os.path.join(rdir, "config.yaml"))
    logger.info(f"\nWrote cleaned config to: {os.path.join(rdir, 'config.yaml')}")


if __name__ == "__main__":
    main()
