"""Detect and mask spikes, and estimate per-detector noise.

Spikes (narrow impulsive glitches) are detected on ``--config``'s data with
``maria_nifty.find_spike_mask`` (minkasi's difference-of-Gaussians filter,
thresholded per detector), configured by the config's ``despike`` section
(defaults: ``maria_nifty.DESPIKE_DEFAULTS``). Run this on dejumped data
(``remove_jumps.py``, usually followed by ``clean_tods.py``), so a leftover
jump isn't flagged as a run of spikes at its edge. Detectors with too large a
spike-masked fraction
(``bad_regions.drop_detector_if_bad_frac``) are dropped entirely via
``mask_dets`` (rather than physically rewriting the FITS files -- the
existing ``mask_dets`` mechanism the whole codebase already uses is
sufficient and avoids subsetting detectors out of a FITS file by hand);
everything else gets a per-sample inverse-noise-variance array instead of a
whole-detector drop.

Writes an updated config (``mask_dets`` extended for dropped detectors) plus
a ``masks.npz`` sidecar (per-dataset ``bad_mask``/``noise_cov_inv``) that
``atmosphere_fit.py``/``map_fit.py``/``joint_fit.py`` load for noise
weighting -- the underlying TOD FITS files are unchanged from ``clean_tods.py``.

Masks/noise are always computed from ``--config``'s (cleaned) data -- the
per-detector noise estimate assumes the shared atmosphere is already removed,
the same way ``flag_tods.py``'s ``detect_bad_samples`` does. Pass
``--fit_config <path>`` (e.g. ``remove_jumps.py``'s output) to instead write
*that* config's ``data.*.fname`` values into the output ``config.yaml`` --
for pipeline modes that want to fit the atmosphere jointly with the map (or
on its own) rather than have it removed by ``clean_tods.py``, while still
carrying over the ``mask_dets``/noise-weighting decisions computed here.

Example::

    python flag_spikes.py --config <odir>/run_0/clean_tods/config.yaml
    python flag_spikes.py --config <odir>/run_0/clean_tods/config.yaml \\
        --fit_config <odir>/run_0/remove_jumps/config.yaml
"""

import os

import click
import numpy as np

REQUIRED_SECTIONS = ["odir", "data", "detector"]
OPTIONAL_SECTIONS = ["run", "despike", "bad_regions", "plot_tods", "verbose"]

BAD_REGION_DEFAULTS = {
    "drop_detector_if_bad_frac": 0.9,
    "bad_sample_inv_var": 1.0e-6,
}


@click.command()
@click.option("--config", required=True, help="Path to the input YAML config file (the cleaned data to analyze).")
@click.option(
    "--fit_config",
    default=None,
    help=(
        "Optional path to an alternate YAML config (e.g. remove_jumps.py's output) whose data.*.fname "
        "values replace --config's own in the written output config, for fitting the pre-cleaning data "
        "instead of the cleaned data this script analyzed."
    ),
)
@click.option(
    "--cuda_device",
    default="",
    help="CUDA device selection such as '0' or '0,1'; leave empty to force CPU mode.",
)
def main(config, fit_config, cuda_device):
    """Detect spikes, drop badly corrupted detectors, and estimate per-detector
    noise for the fit stages."""
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
    rdir, pdir = mani.get_stage_dirs(odir, run, "flag_spikes")

    despike_cfg = {**mani.DESPIKE_DEFAULTS, **(config.get("despike") or {})}
    bad_cfg = {**BAD_REGION_DEFAULTS, **(config.get("bad_regions") or {})}

    datasets = mani.load_tod_datasets(config, verbose)
    datasets = mani.build_detector(config, datasets, verbose)

    # Detected on the data after build_detector, so the mask already follows its
    # detector selection/ordering. find_spikes thresholds every detector against
    # its own MAD, so re-detecting after detectors are dropped (below) gives the
    # same mask for the ones that remain.
    def spike_mask_of(d):
        fs = float(getattr(d["tods"], "fs", 1.0) or 1.0)
        return mani.find_spike_mask(d["tod_data"], fs, despike_cfg)

    logger.info("\nDetecting spikes:")
    fully_bad_input_idx = {dkey: [] for dkey in data_keys}
    for dkey, d in zip(data_keys, datasets):
        spike_mask = spike_mask_of(d)
        bad_mask = spike_mask | d["bad_mask"]
        frac_bad = spike_mask.mean(axis=1)
        fully_bad = np.nonzero(frac_bad > bad_cfg["drop_detector_if_bad_frac"])[0]
        if fully_bad.size:
            fully_bad_input_idx[dkey] = mani.input_indices(d["detector"], fully_bad)
        d["bad_mask"] = bad_mask
        if verbose:
            logger.info(
                f"  {dkey}: {int(bad_mask.any(axis=1).sum())}/{bad_mask.shape[0]} detectors have bad samples "
                f"({bad_mask.mean() * 100:.2f}% of samples); {fully_bad.size} dropped entirely"
            )

    if any(fully_bad_input_idx.values()):
        for dkey, bad_idx in fully_bad_input_idx.items():
            if not bad_idx:
                continue
            existing = list(config["data"][dkey].get("mask_dets", []))
            config["data"][dkey]["mask_dets"] = sorted(set(existing) | set(bad_idx))
        datasets = mani.load_tod_datasets(config, verbose)
        datasets = mani.build_detector(config, datasets, verbose)
        for d in datasets:
            d["bad_mask"] = spike_mask_of(d) | d["bad_mask"]

    plot_tods = int(config.get("plot_tods", len(data_keys)))
    logger.info("\nMasked-region plot + per-detector noise estimate:")
    masks_npz = {}
    for i, (dkey, d) in enumerate(zip(data_keys, datasets)):
        sigma_det = mani.per_detector_sigma(d["tod_data"], d["bad_mask"])
        inv_var = sigma_det[:, None] ** -2 * np.ones_like(d["tod_data"])
        inv_var[d["bad_mask"]] = bad_cfg["bad_sample_inv_var"]
        masks_npz[f"{dkey}_bad_mask"] = d["bad_mask"]
        masks_npz[f"{dkey}_noise_cov_inv"] = inv_var

        if verbose:
            finite = sigma_det[np.isfinite(sigma_det)]
            med = np.median(finite) if finite.size else float("nan")
            logger.info(f"  {dkey}: median detector sigma = {med:.4g}")

        if i < plot_tods:
            label = mani.fits_label(config["data"][dkey]["fname"])
            mani.masked_tod_plot(d["tod_data"], d["bad_mask"], fname=f"{pdir}/tod_{i + 1:02d}.png", label=label)

    np.savez(os.path.join(rdir, "masks.npz"), **masks_npz)

    output_config = config
    if fit_config:
        fit_source = mani.yaml_load(fit_config)
        if "fname" in fit_source["data"]:
            fit_source["data"] = {"d1": fit_source["data"]}
        output_config = dict(config)
        output_config["data"] = dict(config["data"])  # keeps a sibling `scan_center`, if any
        for dkey in data_keys:
            output_config["data"][dkey] = {**config["data"][dkey], "fname": fit_source["data"][dkey]["fname"]}
        logger.info(f"\nWriting output config with data fnames from: {fit_config}")

    mani.yaml_save(output_config, os.path.join(rdir, "config.yaml"))
    logger.info(f"\nWrote flagged config to: {os.path.join(rdir, 'config.yaml')}")


if __name__ == "__main__":
    main()
