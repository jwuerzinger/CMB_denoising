"""Regenerate the final map/detector/tod diagnostic plots for a completed fit.

Reads a ``atmosphere_fit.py``/``map_fit.py``/``joint_fit.py`` run's own
``config.yaml`` + ``last.pkl`` (from ``--fit_dir``), rebuilds the exact model
that was fit (respecting ``fused``, unlike the old ``steering/plot_rec.py``,
which always rebuilt the non-fused path), and redraws the full plot suite,
including ``masked_tod_plot`` when applicable.

Example::

    python plot_results.py --fit_dir <odir>/run_0/joint_fit --plot_extra
"""

import os

import click


@click.command()
@click.option(
    "--fit_dir",
    required=True,
    help="A atmosphere_fit.py/map_fit.py/joint_fit.py run dir (containing config.yaml + last.pkl).",
)
@click.option(
    "--cuda_device",
    default="",
    help="CUDA device selection such as '0' or '0,1'; leave empty to force CPU mode.",
)
@click.option(
    "--plot_extra", is_flag=True, default=False, help="Also draw truth/overview/residual/coefficient diagnostics."
)
@click.option(
    "--cbar_unit",
    default=None,
    help="Colorbar display unit, e.g. 'mK_RJ' (must share the TOD's native unit's base, only the SI prefix may differ). Defaults to the TOD's own native unit.",
)
def main(fit_dir, cuda_device, plot_extra, cbar_unit):
    """Rebuild the fit's model and redraw its final plots."""
    if str(cuda_device) == "":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from nifty.re import logger

    import maria_nifty as mani

    config = mani.yaml_load(os.path.join(fit_dir, "config.yaml"))
    verbose = config.get("verbose", True)
    if "fname" in config["data"]:
        config["data"] = {"d1": config["data"]}
    data_keys = [k for k in config["data"] if k != "scan_center"]

    odir = config["odir"]
    run = config.get("run", 0)
    _, pdir = mani.get_stage_dirs(odir, run, "plot_results")

    datasets = mani.load_tod_datasets(config, verbose)
    datasets = mani.build_detector(config, datasets, verbose)
    datasets = mani.load_noise_masks(odir, run, datasets, data_keys)
    _, datasets, _ = mani.build_model(config, datasets, verbose)

    samples = mani.load_samples(fit_dir)

    plot_tods = int(config.get("plot_tods", len(datasets)))
    plot_datasets = datasets[:plot_tods]
    has_map = datasets[0]["model"].map_model is not None

    logger.info("\nFinal plotting:")
    if has_map:
        cbar_scale, cbar_label = mani.signal_cbar_label(datasets[0]["tods"], target_unit=cbar_unit)
        pdir_map = os.path.join(pdir, "map")
        mani.map_plot(
            datasets[0]["model"],
            datasets[0]["offsets"],
            samples,
            odir=pdir_map,
            center_phi_theta=datasets[0]["center_phi_theta"],
            cbar_label=cbar_label,
            cbar_scale=cbar_scale,
        )
        if plot_extra:
            pdir_map_extra = os.path.join(pdir, "extra", "map")
            mani.map_plot(
                datasets[0]["model"],
                datasets[0]["offsets"],
                samples,
                odir=pdir_map_extra,
                plot_extra=True,
                center_phi_theta=datasets[0]["center_phi_theta"],
                cbar_label=cbar_label,
                cbar_scale=cbar_scale,
            )

    for i, d in enumerate(plot_datasets):
        pdir_tod_i = os.path.join(pdir, f"tod_{i + 1:02d}")
        label = mani.fits_label(config["data"][data_keys[i]]["fname"])
        mani.detector_plot(d["detector"], odir=pdir_tod_i)
        mani.tod_plot(d["tod_data"], d["model"], samples, d["mask"], odir=pdir_tod_i, label=label)
        if d.get("bad_mask") is not None and d["bad_mask"].any():
            mani.masked_tod_plot(d["tod_data"], d["bad_mask"], fname=f"{pdir_tod_i}/masked_regions.png", label=label)

        if plot_extra:
            pdir_extra_i = os.path.join(pdir, "extra", f"tod_{i + 1:02d}")
            mani.truth_plot(d["detector"], d["tod_data"], odir=pdir_extra_i)
            mani.tod_plot(d["tod_data"], d["model"], samples, d["mask"], odir=pdir_extra_i, plot_extra=True)
            mani.tod_overview_plot(d["tod_data"], d["model"], samples, d["tods"], odir=pdir_extra_i)
            mani.detector_residual_plots(d["detector"], d["tod_data"], d["model"], samples, odir=pdir_extra_i)
            if d["model"].atm_model is not None:
                mani.coefficient_histograms(d["model"], samples, odir=pdir_extra_i)


if __name__ == "__main__":
    main()
