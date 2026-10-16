"""Export a fit's reconstruction to FITS, so plots no longer need ``last.pkl``.

Reads a ``atmosphere_fit.py``/``map_fit.py``/``joint_fit.py`` run's own
``config.yaml`` + ``last.pkl`` (from ``--fit_dir``), rebuilds the model, and
writes three files to ``plot_{run}/save_fits/``:

* ``map_mean.fits`` -- posterior mean of the full map (primary HDU) and of
  each map component (``S1_MEAN``, ``S2_MEAN``, ...), with an RA/Dec WCS.
* ``map_samples.fits`` -- every stored posterior state as ``(n_samp, ny, nx)``
  cubes (``FULL``, ``S1``, ``S2``, ...), plus ``--n_prior`` prior draws
  (``PRIOR_FULL``, ``PRIOR_S1``, ...). Std/rel-std/pull maps follow from these.
* ``tod_mean.fits`` -- per dataset: data + posterior-mean atmosphere/map/model
  TODs, the flagged-sample mask, the detector table, and one prior draw of the
  atmosphere/map TODs.

The two map files are skipped for an atmosphere-only fit. Prior draws use the
same PRNG keys as ``talk_plots.py``, so its prior figures are reproduced
exactly: map draws are rejection-sampled on their peak source-map pixel lying
in ``[--prior_vmin_mk, --prior_vmax_mk]`` mK_RJ (``talk_plots.py``'s
``--filtered_prior_*``), TOD draws use ``config["key"]``.

Example::

    python save_fits.py --fit_dir <odir>/run_0/joint_fit
"""

import os
import shutil

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
@click.option("--n_prior", default=16, show_default=True, help="Number of map prior draws to store.")
@click.option(
    "--prior_vmin_mk",
    default=1.0,
    show_default=True,
    help="Keep only map prior draws whose peak source-map pixel is >= this (mK_RJ).",
)
@click.option(
    "--prior_vmax_mk",
    default=5.0,
    show_default=True,
    help="Keep only map prior draws whose peak source-map pixel is <= this (mK_RJ).",
)
def main(fit_dir, cuda_device, n_prior, prior_vmin_mk, prior_vmax_mk):
    """Rebuild the fit's model and export map/TOD means, map samples and prior draws to FITS."""
    if str(cuda_device) == "":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import jax
    import nifty.re as jft
    import numpy as np
    from nifty.re import logger

    import maria_nifty as mani

    config = mani.yaml_load(os.path.join(fit_dir, "config.yaml"))
    verbose = config.get("verbose", True)
    if "fname" in config["data"]:
        config["data"] = {"d1": config["data"]}
    data_keys = [k for k in config["data"] if k != "scan_center"]

    odir = config["odir"]
    run = config.get("run", 0)
    _, pdir = mani.get_stage_dirs(odir, run, "save_fits")

    datasets = mani.load_tod_datasets(config, verbose)
    datasets = mani.build_detector(config, datasets, verbose)
    datasets = mani.load_noise_masks(odir, run, datasets, data_keys)
    _, datasets, fused_response = mani.build_model(config, datasets, verbose)

    samples = mani.load_samples(fit_dir)
    model = datasets[0]["model"]
    shutil.copy(os.path.join(fit_dir, "config.yaml"), os.path.join(pdir, "config.yaml"))

    if model.map_model is not None:
        names = mani.map_component_names(config)
        mani.map2fits(model, samples, datasets=datasets, odir=pdir, component_names=names, overwrite=True)
        logger.info(f"\nWrote {pdir}/map_mean.fits")

        prior_positions = filtered_map_prior_positions(
            fused_response if fused_response is not None else model,
            model.map_model,
            datasets[0]["tods"],
            n_prior,
            prior_vmin_mk,
            prior_vmax_mk,
            base_key=int(config.get("key", 42)) + 1000,
        )
        mani.map_samples2fits(
            model,
            samples,
            os.path.join(pdir, "map_samples.fits"),
            datasets=datasets,
            component_names=names,
            prior_positions=prior_positions,
        )
        logger.info(f"Wrote {pdir}/map_samples.fits ({len(prior_positions)} prior draws)")

    # One prior latent for the TOD-space prior draw, keyed exactly as in
    # talk_plots.py's tod_prior_components.png / tod_atmosphere_prior.png. Fused
    # runs share it across datasets; non-fused ones init each dataset's own view.
    key = int(config.get("key", 123))
    _, prior_key, _ = jax.random.split(jax.random.PRNGKey(key), 3)
    if fused_response is not None:
        prior_positions = [jft.Vector(fused_response.init(key=prior_key))] * len(datasets)
    else:
        prior_positions = [jft.Vector(d["model"].init(key=prior_key)) for d in datasets]

    labels = [mani.fits_label(config["data"][k]["fname"]) for k in data_keys]
    mani.tods2fits(
        datasets,
        samples,
        os.path.join(pdir, "tod_mean.fits"),
        prior_positions=prior_positions,
        labels=labels if len(labels) == len(datasets) else None,
        header_cards={
            "NOISESTD": (float(config.get("noise_std", np.nan)), "config noise_std (prior noise draw)"),
            "KEY": (key, "config PRNG key"),
        },
    )
    logger.info(f"Wrote {pdir}/tod_mean.fits")


def filtered_map_prior_positions(init_src, map_model, tods, n_accept, vmin_mk, vmax_mk, base_key):
    """Prior latents whose summed source map peaks within ``[vmin_mk, vmax_mk]`` mK_RJ.

    Same rejection sampling (one ``PRNGKey(base_key + attempt)`` per try, at most
    ``200 * n_accept`` tries) as ``talk_plots.py``'s ``sample_filtered_prior_positions``.
    """
    import jax
    import nifty.re as jft
    import numpy as np
    from nifty.re import logger

    import maria_nifty as mani

    source_space = map_model.source_space if map_model.source_space is not None else map_model.space
    source_map_func = mani.map2space(map_model.space, source_space, order=0)
    cbar_scale, _ = mani.signal_cbar_label(tods, target_unit="mK_RJ")

    def peak_mk(pos):
        out = sum(np.asarray(m(pos, map=True)) for m in map_model.sources)
        return float(np.max(source_map_func(out, x=pos)) * cbar_scale)

    accepted, attempt, max_attempts = [], 0, 200 * n_accept
    while len(accepted) < n_accept and attempt < max_attempts:
        pos = 1 * jft.Vector(init_src.init(key=jax.random.PRNGKey(base_key + attempt)))
        if vmin_mk <= peak_mk(pos) <= vmax_mk:
            accepted.append(pos)
        attempt += 1
    if len(accepted) < n_accept:
        logger.warning(
            f"Only found {len(accepted)}/{n_accept} map prior draws with peak in "
            f"[{vmin_mk}, {vmax_mk}] mK_RJ after {attempt} attempts."
        )
    return accepted


if __name__ == "__main__":
    main()
