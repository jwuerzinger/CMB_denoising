"""Shared CLI plumbing for the single-purpose ``steering/scripts/`` stage scripts.

Every stage script is a thin ``(config in) -> (config + data/samples out)``
node: load a YAML config, do its one job, and write its outputs under a
``run_{run}/<name>`` / ``plot_{run}/<name>`` pair of directories so the next
stage can chain off it. This module factors out that boilerplate -- except
the CUDA-device env-var setup and the ``--config``/``--cuda_device`` click
options themselves, which every script must still set up inline *before*
importing ``maria_nifty`` (see each script's top): merely importing this
package pulls in ``jax`` (via ``enforce_float64``), which locks in JAX's
device list on first import -- setting ``JAX_PLATFORM_NAME``/
``CUDA_VISIBLE_DEVICES`` any later has no effect. Everything in this module
is safe to import only *after* that env setup has already happened.
"""

import os

import numpy as np

from .yml import yaml_load


def load_config(config, required, optional=None) -> dict:
    """Load a stage config and check that its required sections are present.

    Unlike :func:`maria_nifty.check_dict` (built for a config's single final
    consumer), this does **not** filter the dict down to ``required |
    optional`` -- every ``steering/scripts/`` stage writes its output config
    back out for the *next* stage to read, so sections a given stage doesn't
    itself touch (e.g. ``map``/``atmosphere`` as seen by ``remove_jumps.py``)
    must survive the round-trip unchanged. ``optional`` is documentation only
    (the section names a stage additionally reads), not enforced.

    Parameters
    ----------
    config : str or dict
        Path to a YAML config, or an already-loaded config dict.
    required : list of str
        Keys that must be present.
    optional : list of str, optional
        Additional keys this stage reads, if present (documentation only).

    Returns
    -------
    dict
        The loaded config, unfiltered.
    """
    if isinstance(config, str):
        config = yaml_load(config)
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing required config section(s): {missing}")
    return config


def get_stage_dirs(odir: str, run, name: str) -> tuple[str, str]:
    """Return (and create) this stage's ``run_{run}/<name>`` / ``plot_{run}/<name>`` dirs.

    Parameters
    ----------
    odir : str
        The config's top-level output directory.
    run : int or str
        The config's ``run`` index.
    name : str
        This stage's own name (e.g. ``"remove_jumps"``), used as the
        subfolder both directories are namespaced under.

    Returns
    -------
    (rdir, pdir) : tuple of str
        ``rdir`` holds anything a later stage needs to read back in (data
        files, ``last.pkl``, an updated ``config.yaml``); ``pdir`` holds
        plots and other final artifacts nothing reads back in.
    """
    rdir = os.path.join(odir, f"run_{run}", name)
    pdir = os.path.join(odir, f"plot_{run}", name)
    os.makedirs(rdir, exist_ok=True)
    os.makedirs(pdir, exist_ok=True)
    return rdir, pdir


def load_noise_masks(odir, run, datasets, data_keys):
    """Attach a prior ``flag_spikes.py`` run's per-dataset ``noise_cov_inv``/``bad_mask``.

    Looks for ``odir/run_{run}/flag_spikes/masks.npz``; a no-op (``datasets``
    unchanged) if it doesn't exist, so the fit scripts work identically
    whether or not ``flag_spikes.py`` ran first.
    """
    masks_path = os.path.join(odir, f"run_{run}", "flag_spikes", "masks.npz")
    if not os.path.isfile(masks_path):
        return datasets
    with np.load(masks_path) as npz:
        masks = dict(npz.items())
    for dkey, d in zip(data_keys, datasets):
        if f"{dkey}_noise_cov_inv" in masks:
            d["noise_cov_inv"] = masks[f"{dkey}_noise_cov_inv"]
        if f"{dkey}_bad_mask" in masks:
            d["bad_mask"] = masks[f"{dkey}_bad_mask"]
    return datasets
