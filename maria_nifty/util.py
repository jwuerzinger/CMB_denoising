"""High-level data loading and model assembly utilities.

These helpers bridge YAML configs, Maria TOD files, detector construction, and
forward-model setup so the steering scripts can stay compact.
"""

import logging
import os
from functools import reduce

import numpy as np
from maria.coords.transforms import phi_theta_to_offsets
from maria.tod import TOD
from nifty.re import Gaussian, logger

from .detector import Detector
from .model.atmosphere import AtmosphereModel
from .model.map import MapModel
from .model.response import SumResponse
from .model.util import str2rad


def raw_indices_of_valid(detector) -> np.ndarray:
    """Original input-detector indices for ``detector``'s current valid (unmasked)
    positions, in valid-axis order."""
    valid_positions = np.nonzero(~detector.mask)[0]
    if detector.indices is not None:
        return np.asarray(detector.indices)[valid_positions]
    return valid_positions


def input_indices(detector, valid_idx) -> list[int]:
    """Map positions along the valid-detector axis back to original input-detector indices."""
    raw_idx = raw_indices_of_valid(detector)
    return [int(x) for x in raw_idx[np.asarray(valid_idx, dtype=int)]]


def scatter_to_raw(values: np.ndarray, detector) -> np.ndarray:
    """Place ``(n_valid, T)`` model values onto ``detector.full_size`` raw detector axis.

    Positions of detectors the model never saw (masked out before the fit)
    are left at zero.
    """
    raw_idx = raw_indices_of_valid(detector)
    out = np.zeros((detector.full_size, values.shape[1]), dtype=values.dtype)
    out[raw_idx] = values
    return out


def per_detector_sigma(tod_data: np.ndarray, bad_mask: np.ndarray) -> np.ndarray:
    """Robust per-detector noise sigma (MAD of the first difference), excluding bad samples."""
    tod_data = np.asarray(tod_data, dtype=float)
    dr = np.diff(tod_data, axis=1)
    dr_bad = bad_mask[:, :-1] | bad_mask[:, 1:]
    sigma = np.full(tod_data.shape[0], np.inf)
    for i in range(tod_data.shape[0]):
        good = dr[i][~dr_bad[i]]
        if good.size < 2:
            continue
        mad = np.median(np.abs(good - np.median(good)))
        if mad > 0:
            sigma[i] = 1.4826 * mad / np.sqrt(2.0)
    return sigma


def fits_label(fname: str) -> str:
    """Return the identifying part of a TOD fits filename, for plot labels.

    Strips the directory and extension, then drops everything up to and
    including the first ``-`` (e.g. ``data/MC/cleaned/Signal_TOD-AGBT22B_068_01
    -s19.fits`` -> ``AGBT22B_068_01-s19``). Falls back to the full stem when
    there is no ``-`` to split on.
    """
    stem = os.path.splitext(os.path.basename(fname))[0]
    _, _, rest = stem.partition("-")
    return rest if rest else stem


def _normalize_atmosphere_config(atmosphere_cfg):
    """Normalize atmosphere config keys to the current build API."""
    cfg = dict(atmosphere_cfg)

    # Align with AtmosphereModel.build keyword names.
    if "spline" in cfg and "spline_params" not in cfg:
        cfg["spline_params"] = cfg.pop("spline")
    if "gp" in cfg and "gp_params" not in cfg:
        cfg["gp_params"] = cfg.pop("gp")
    if "poly" in cfg and "poly_params" not in cfg:
        cfg["poly_params"] = cfg.pop("poly")

    mode = cfg.get("mode")
    if isinstance(mode, str):
        m = mode.strip().lower()
        if "gp" in m and "1d" in m:
            cfg["mode"] = "gp-1d"
        elif "gp" in m and "3d" in m:
            cfg["mode"] = "gp-3d"

    return cfg


def _get_atmosphere_downsampling(config):
    """Return the effective atmosphere downsampling factor for a config."""
    atmosphere_cfg = _normalize_atmosphere_config(config.get("atmosphere", {}))
    if not atmosphere_cfg:
        return 1

    mode = atmosphere_cfg.get("mode", None)

    if mode in ("gp-1d", "gp-3d"):
        gp_cfg = atmosphere_cfg.get("gp_params", {})
        ds_factor = gp_cfg.get("downsampling", 1)
    else:
        ds_factor = 1

    return ds_factor


def _ephemeris_offsets(body: str, time: np.ndarray, earth_location) -> np.ndarray:
    """Project a solar-system body's true topocentric position onto the offset frame.

    Computes ``body``'s apparent RA/Dec at each sample time and site, then maps
    it through the same ``(cphi, ctheta) = (0, 0)`` projection used by
    :meth:`maria.coords.coordinates.Coordinates.offsets`, so the result can be
    subtracted directly from a dataset's ``offsets`` array (moving to a frame
    co-moving with the body, e.g. the Moon).

    Parameters
    ----------
    body : str
        Name resolved by :func:`astropy.coordinates.get_body`, e.g. ``"moon"``.
    time : array-like, shape (n_time,)
        Absolute sample times (Unix seconds).
    earth_location : astropy.coordinates.EarthLocation
        Observing site, for the topocentric (parallax-corrected) position.

    Returns
    -------
    numpy.ndarray, shape (n_time, 2)
        Offset (radians) matching the layout of ``Coordinates.offsets``.
    """
    from astropy.coordinates import get_body
    from astropy.time import Time

    t = Time(np.asarray(time, dtype="float64"), format="unix")
    coord = get_body(body, t, location=earth_location)
    pt = np.stack([coord.ra.rad, coord.dec.rad], axis=-1)
    return np.asarray(phi_theta_to_offsets(pt, 0.0, 0.0))


def _read_boresight_elevation(fname: str, format: str, n_dets: int, n_samp: int) -> np.ndarray | None:
    """Boresight elevation per time sample, read directly from the FITS file.

    ``maria``'s own ``TOD.boresight.el`` computes elevation via an internal
    RA/Dec -> Alt/Az transform (``maria.coords.coordinates.Coordinates.compute_transform``)
    that has been found to return physically wrong values (verified against
    the astronomically-expected elevation for this site/target -- e.g.
    ``boresight.el`` gave ~-62.6 deg for a file whose FITS `ELEV` column, and
    the known RA/Dec of the target vs. Green Bank's latitude, both agree on
    ~34 deg). The underlying RA/Dec themselves (``raw["DX"]``/``raw["DY"]``)
    are correct -- confirmed against the target's known coordinates -- so this
    is a bug in maria's own transform, not the input data. The map-projection
    path is unaffected (it works directly in the ra/dec frame and never calls
    ``.el``); only elevation-dependent code
    (:class:`maria_nifty.model.spline.TimeSplineModel`'s ``remove_el_gradient``)
    is. Re-reading the FITS file's own `ELEV` column sidesteps the broken
    transform entirely.

    Returns `None` (rather than raising) for any non-MUSTANG-2 format or any
    read failure, so callers without this metadata (e.g. simulated TODs)
    degrade gracefully instead of blocking.
    """
    if format.strip().lower() != "mustang-2":
        return None
    try:
        from astropy.io import fits

        with fits.open(fname) as hdul:
            elev = np.asarray(hdul[1].data["ELEV"], dtype=np.float64)
    except Exception:
        return None

    if elev.size != n_dets * n_samp:
        logger.warning(
            f"  {fname}: ELEV size ({elev.size}) doesn't match the loaded detector/sample "
            f"count ({n_dets}x{n_samp}); dropping boresight elevation for this dataset."
        )
        return None
    # Per-(detector, time); elevation is a shared boresight quantity, essentially
    # identical across detectors at a given sample -- average over detectors.
    return elev.reshape(n_dets, n_samp).mean(axis=0)


def write_boresight_elevation(fname: str, boresight_el, n_dets: int, format: str = "mustang-2") -> None:
    """Patch a just-written MUSTANG-2 FITS file's `ELEV` column in place.

    ``maria``'s own ``TOD.to_fits`` does not preserve or regenerate a
    meaningful `ELEV` column (it writes all zeros) -- so every pipeline stage
    that rewrites a TOD as a new FITS file (``remove_jumps.py``,
    ``clean_tods.py``) must re-stamp the real elevation back in itself, via
    this function, right after calling ``to_fits``. Otherwise every later
    stage's :func:`_read_boresight_elevation` silently loses elevation for
    that dataset (reading back the written zeros), and ``remove_el_gradient``
    then fails with a "constant elevation" error the next time the model is
    built from that file.

    `boresight_el` (shape ``(n_samp,)``, as returned in
    ``dataset["boresight_el"]``) is broadcast identically across all
    `n_dets` detectors, matching the layout :func:`_read_boresight_elevation`
    assumes when reading it back (``reshape(n_dets, n_samp)``).

    A no-op if `boresight_el` is `None` or `format` isn't MUSTANG-2.
    """
    if boresight_el is None or format.strip().lower() != "mustang-2":
        return
    from astropy.io import fits

    n_samp = np.asarray(boresight_el).shape[0]
    with fits.open(fname, mode="update") as hdul:
        if hdul[1].data["ELEV"].size != n_dets * n_samp:
            logger.warning(
                f"  {fname}: cannot stamp ELEV -- expected {n_dets * n_samp} rows, found {hdul[1].data['ELEV'].size}."
            )
            return
        hdul[1].data["ELEV"] = np.tile(np.asarray(boresight_el, dtype=np.float32), n_dets)
        hdul.flush()


def load_tod_file(fname: str, format: str = "mustang-2", mask_dets: list = None, truth: str | None = None):
    """Load a single TOD file and optionally its truth array.

    Parameters
    ----------
    fname : str
        Path to the TOD FITS file.
    format : str, optional
        Maria TOD format passed to :meth:`maria.tod.TOD.from_fits`.
    mask_dets : list[int], optional
        Detector indices to exclude from downstream processing.
    truth : str or None, optional
        Optional path to a `.npz` truth file.

    Returns
    -------
    tods : maria.tod.TOD
        Loaded TOD object.
    mask : numpy.ndarray or None
        Boolean keep-mask aligned with the detector axis, or `None`.
    truth : dict or None
        Loaded truth data if a truth file was provided.
    boresight_el : numpy.ndarray or None
        Boresight elevation (degrees) per time sample, read directly from the
        FITS file's ``ELEV`` column (see :func:`_read_boresight_elevation`) --
        `None` if unavailable. Use this instead of ``tods.boresight.el``,
        which has been found to return physically wrong values for this
        format (see :func:`_read_boresight_elevation` for details).
    """
    if mask_dets is None:
        mask_dets = []
    tods = TOD.from_fits(fname, format)
    n_dets, n_samp = tods.data["data"].shape
    boresight_el = _read_boresight_elevation(fname, format, n_dets, n_samp)

    # load simulation output
    if truth is not None and os.path.exists(truth):
        truth = dict(np.load(truth))

    # remove bad tods
    if len(mask_dets) > 0:
        mask = np.ones(tods.data["data"].shape[0], dtype=bool)
        mask[mask_dets] = False
    else:
        mask = None

    return tods, mask, truth, boresight_el


def load_tod_datasets(config: dict, verbose: bool = True) -> list[dict]:
    """Load all TOD datasets defined in a reconstruction config.

    Also computes detector offsets, trims time samples to match the atmosphere
    downsampling factor, and recenters all datasets to a common map center.

    Parameters
    ----------
    config : dict
        Reconstruction configuration containing at least a ``data`` section
        with per-dataset file paths and optional by-hand masking. ``data``
        may also set ``scan_center`` as a sibling of the per-dataset entries:
        ``"auto"`` (the default) derives the projection's tangent point
        directly from the data's own true (ra, dec) mean (a proper spherical
        mean over every sample/detector/dataset), so the model's local
        ``(0, 0)`` sits at that mean pointing center -- this is a per-run
        value, since it depends on exactly which datasets are included, but
        (unlike projecting around a literal ``(ra, dec) = (0, 0)`` tangent
        point and hoping the target happens to be nearby) it keeps every
        dataset's offsets well-conditioned regardless of where on the sky the
        target actually is. Alternatively, an explicit ``[ra, dec]`` (same
        angular-unit-suffix strings as ``space_params.center``, e.g.
        ``["115.446deg", "74.248deg"]``) is used *directly* as the
        projection's reference point, with no further derivation -- gives a
        *fixed* frame that stays identical across configs/runs regardless of
        which datasets are included (unlike ``"auto"``); prefer this when you
        want that stability, e.g. for comparing multiple runs pixel-for-pixel.
        Map/source
        ``space_params.center`` values should then be small (near ``(0, 0)``
        or a few arcmin), matching this local frame -- *not* the same
        absolute degrees given here, which would double-apply the shift.
        Each per-dataset entry may set
        ``mask_dets`` (list[int], default ``[]``) to
        drop specific detectors entirely (same as before, just renamed from
        ``mask_indices`` to avoid confusion with the unrelated
        ``detector.mask_indices`` config key); ``mask_fits`` (bool, default
        ``False``) to exclude the whole file from the fit while still keeping
        it around (faded) for plotting; and ``mask_time`` (list of
        ``[t0, t1]`` pairs of *sample indices* -- matching the x-axis of the
        tod/masked-region plots -- default ``[]``) to mask specific time
        ranges for every detector in the file. An optional top-level
        ``ephemeris_center`` (e.g. ``"moon"``) subtracts that solar-system
        body's true per-sample topocentric position (via
        :func:`astropy.coordinates.get_body`) from every dataset's offsets
        before the global recentering below, moving to the body's co-moving
        frame deterministically -- useful for sources whose ephemeris is
        well known.
    verbose : bool, optional
        Whether to log dataset shapes and the inferred map center.

    Returns
    -------
    list of dict
        One dict per dataset with keys ``tods``, ``tod_data``, ``mask``,
        ``offsets``, ``bad_mask`` (a ``(n_det, n_time)`` boolean array,
        by-hand masked samples set from ``mask_fits``/``mask_time``; all
        ``False`` when neither is given), and ``center_phi_theta`` (a ``(ra, dec)`` tuple in
        radians, identical across every dataset in the list: the absolute
        sky position of the model's local ``(0, 0)`` origin, for recovering
        true sky coordinates downstream, e.g. in
        :func:`maria_nifty.fits.map2fits`).
    """
    if verbose:
        logger.info("\nLoad datasets:")

    datasets = []
    # Work on a shallow copy of config["data"] rather than mutating it in
    # place: callers (e.g. clean_and_run.py) reuse the same config dict
    # across several load_tod_datasets calls, and a destructive pop here
    # would silently make scan_center vanish (falling back to "auto") after
    # the first call.
    data_cfg = dict(config["data"])
    scan_center = data_cfg.pop("scan_center", "auto")
    if isinstance(scan_center, str):
        if scan_center != "auto":
            raise ValueError(f'`data.scan_center` must be `"auto"` or an `[ra, dec]` pair, got {scan_center!r}.')
        fixed_center = None
    else:
        cphi, ctheta = (v if isinstance(v, int | float) else str2rad(v) for v in scan_center)
        fixed_center = np.array([cphi, ctheta], dtype="float64")

    if "fname" in data_cfg:
        data_cfg = {"d1": data_cfg}

    ephemeris_body = config.get("ephemeris_center")

    # Pass 1: load every TOD file (no offsets yet -- those need `fixed_center`,
    # which for "auto" is only determined below, from these very TODs).
    loaded = []
    for i, d in enumerate(data_cfg.values()):
        d = dict(d)
        mask_fits = bool(d.pop("mask_fits", False))
        mask_time = d.pop("mask_time", None) or []
        tods, mask, _, boresight_el = load_tod_file(**d)
        loaded.append((tods, mask, boresight_el, mask_fits, mask_time))
        if verbose:
            logger.info(f"  tods {i + 1:02d} shape: {tods.data['data'].shape}")

    # "auto" mode: derive the projection's tangent point directly from the
    # data's own true (ra, dec) mean (a proper spherical mean: average unit
    # vectors, not raw angles, so it's exact regardless of where on the sky
    # the target sits), rather than anchoring at a literal (ra, dec) = (0, 0)
    # tangent point and hoping the target happens to be nearby. The old
    # approach was fine for targets near (0, 0), but for one e.g. ~200 deg
    # away in RA (RXJ1347), a gnomonic projection anchored that far from its
    # own tangent point is degenerate well before 90 deg out (see the
    # `scan_center` docs above) -- offsets computed against it are already
    # corrupted, and inverting back through it (as the old code did) recovers
    # the wrong absolute RA (verified: recovered Dec matched RXJ1347's real
    # declination to 3 decimal places, since Dec happens to stay well-behaved
    # in this projection, but recovered RA was off by ~234 deg). Deriving the
    # tangent point from the real (ra, dec) values up front keeps every
    # dataset's offsets small/well-conditioned by construction, and this
    # derived point is then usable directly and exactly as `center_phi_theta`.
    if fixed_center is None:
        x = y = z = 0.0
        n = 0
        for tods, *_ in loaded:
            ra = np.asarray(tods.coords.ra, dtype=np.float64).ravel()
            dec = np.asarray(tods.coords.dec, dtype=np.float64).ravel()
            x += np.sum(np.cos(dec) * np.cos(ra))
            y += np.sum(np.cos(dec) * np.sin(ra))
            z += np.sum(np.sin(dec))
            n += ra.size
        # atan2 returns (-pi, pi]; normalize to the conventional [0, 2*pi) RA
        # range (matching how an explicit `scan_center` RA is normally given).
        auto_ra = float(np.arctan2(y / n, x / n)) % (2 * np.pi)
        auto_dec = float(np.arctan2(z / n, np.hypot(x / n, y / n)))
        fixed_center = np.array([auto_ra, auto_dec], dtype="float64")
        if verbose:
            logger.info(
                f"  auto scan center (mean sky position of the data): "
                f"ra={np.degrees(auto_ra):.5f} deg, dec={np.degrees(auto_dec):.5f} deg"
            )

    # Pass 2: every dataset's offsets, relative to `fixed_center` -- for
    # "auto" this is now the just-derived data-mean center; for an explicit
    # `scan_center` it's the user's own reference point. Either way, offsets
    # are already correctly centered by construction -- no further
    # mean-subtraction step needed (unlike the old "auto" path).
    for tods, mask, boresight_el, mask_fits, mask_time in loaded:
        tod_data = np.array(tods.data["data"], dtype="float64")
        offsets = tods.coords.offsets(center=fixed_center, frame=tods.coords.frame)
        # Absolute sample time (Unix seconds); maria adds the FITS JDSTART epoch,
        # so gaps between datasets are represented naturally on a common timeline.
        time = np.asarray(tods.time, dtype="float64")

        if ephemeris_body:
            body_offset = _ephemeris_offsets(ephemeris_body, time, tods.coords.earth_location)
            offsets = offsets - body_offset[None, :, :]

        # By-hand bad-sample mask from mask_fits / mask_time. mask_dets is
        # still a hard per-detector drop applied later in build_detector.
        bad_mask = np.zeros(tod_data.shape, dtype=bool)
        if mask_fits:
            bad_mask[:, :] = True
        elif mask_time:
            for t0, t1 in mask_time:
                bad_mask[:, int(t0) : int(t1)] = True

        # Trim offsets to match the atmosphere model's effective downsampling.
        ds_factor = _get_atmosphere_downsampling(config)
        if ds_factor > 1:
            n_timesteps = tod_data.shape[1] - (tod_data.shape[1] % ds_factor)
            tod_data = tod_data[:, :n_timesteps]
            offsets = offsets[:, :n_timesteps, :]
            time = time[:n_timesteps]
            bad_mask = bad_mask[:, :n_timesteps]
            if boresight_el is not None:
                boresight_el = boresight_el[:n_timesteps]

        datasets.append(
            {
                "tods": tods,
                "tod_data": tod_data,
                "mask": mask,
                "offsets": offsets,
                "time": time,
                "bad_mask": bad_mask,
                "boresight_el": boresight_el,
            }
        )

    if verbose:
        logger.info(f"  min offset: {np.min([dct['offsets'].min(axis=(0, 1)) for dct in datasets], axis=0)}")
        logger.info(f"  max offset: {np.max([dct['offsets'].max(axis=(0, 1)) for dct in datasets], axis=0)}")

    # Stashed on every dataset (identical value each time) so downstream
    # code -- e.g. fits.map2fits -- can recover the absolute sky position of
    # the model's local (0, 0) origin exactly, without re-deriving an
    # approximation of it from scratch.
    cphi0, ctheta0 = float(fixed_center[0]), float(fixed_center[1])
    for dct in datasets:
        dct["center_phi_theta"] = (cphi0, ctheta0)

    return datasets


def build_detector(config: dict, datasets: list[dict], verbose: bool = True) -> list[dict]:
    """Construct one detector object per dataset and apply detector masks.

    Parameters
    ----------
    config : dict
        Reconstruction configuration containing a ``detector`` section.
    datasets : list of dict
        Dataset list as returned by :func:`load_tod_datasets`; modified
        in-place to add a ``detector`` key and update ``tod_data`` / ``offsets``
        / ``bad_mask`` to reflect reordering and masking.
    verbose : bool, optional
        Whether to log detector shapes and masked-detector counts.

    Returns
    -------
    list of dict
        The same ``datasets`` list with ``detector``, ``tod_data``, and
        ``offsets`` updated for each entry.
    """
    if verbose:
        logger.info("\nBuild detector:")

    kept = []
    for i, d in enumerate(datasets):
        detector_cfg = dict(config["detector"])

        # d["mask"] is a keep-mask from data loading (True = keep).
        if d["mask"] is not None:
            data_mask = ~np.asarray(d["mask"], dtype=bool)
            cfg_mask = detector_cfg.pop("mask", None)
            cfg_mask_indices = detector_cfg.pop("mask_indices", None)

            if cfg_mask is not None:
                cfg_mask = np.asarray(cfg_mask, dtype=bool)
            elif cfg_mask_indices is not None:
                cfg_mask = np.zeros(data_mask.shape[0], dtype=bool)
                cfg_mask[np.asarray(cfg_mask_indices, dtype=int)] = True
            else:
                cfg_mask = np.zeros(data_mask.shape[0], dtype=bool)

            detector_cfg["mask"] = np.logical_or(cfg_mask, data_mask)

        detector = Detector.build(tods=d["tods"], **detector_cfg)

        if detector.indices is not None:
            d["tod_data"] = d["tod_data"][detector.indices]
            d["offsets"] = d["offsets"][detector.indices]
            if d.get("bad_mask") is not None:
                d["bad_mask"] = d["bad_mask"][detector.indices]

        if detector.n_masked > 0:
            valid = ~detector.mask
            d["tod_data"] = d["tod_data"][valid]
            d["offsets"] = d["offsets"][valid]
            if d.get("bad_mask") is not None:
                d["bad_mask"] = d["bad_mask"][valid]

        # Masking is now handled directly by detector/response models.
        d["mask"] = None

        d["detector"] = detector
        if verbose:
            logger.info(f"  det {i + 1:02d} shape: {d['tod_data'].shape} | masked: {detector.n_masked}")

        # A dataset whose detectors are all flagged collapses to shape (0, T): it
        # carries no information and would otherwise feed a degenerate size-0
        # entry into the (summed or fused) forward model and the plotting. Drop it.
        if d["tod_data"].shape[0] == 0:
            logger.warning(f"  dataset {i + 1:02d}: all detectors flagged (0 unmasked); dropping it.")
            continue

        kept.append(d)

    return kept


def setup_forward_model(config: dict, datasets: list[dict], verbose: bool = True):
    """Assemble the likelihood and forward-model objects for all datasets.

    Builds the map model, optional atmosphere model, response chain, and
    Gaussian likelihood for each dataset, then returns the summed likelihood.

    Parameters
    ----------
    config : dict
        Reconstruction configuration with optional ``map``, ``atmosphere``,
        ``response``, and ``noise_std`` sections.
    datasets : list of dict
        Dataset list as returned by :func:`build_detector`; modified in-place
        to add ``likelihood`` and ``model`` keys for each entry. An optional
        per-entry ``noise_cov_inv`` (array broadcastable against
        ``tod_data``, e.g. a per-detector or per-(detector, time) inverse
        noise variance) overrides the global scalar ``config["noise_std"]``
        for that dataset.
    verbose : bool, optional
        Whether to log model shapes and the likelihood domain.

    Returns
    -------
    likelihood : nifty.re.Likelihood
        Sum of per-dataset Gaussian likelihoods.
    datasets : list of dict
        The same ``datasets`` list with ``likelihood`` and ``model`` added.
    """
    if verbose:
        logger.info("\nSetup forward model:")

    map_model = None
    if "map" in config:
        map_model = MapModel.build(**config["map"])
        if verbose:
            logger.info(f"  map shape: {map_model.target.shape}")
    elif verbose:
        logger.info("  map model disabled")

    atmosphere_cfg = _normalize_atmosphere_config(config.get("atmosphere", {}))
    has_atmosphere = len(atmosphere_cfg) > 0

    for i, d in enumerate(datasets):
        atm_model = None
        if has_atmosphere:
            cfg = dict(atmosphere_cfg)
            cfg["prefix"] = cfg.get("prefix", "atm").strip() + f" {i + 1:02d}"
            t_steps = d["tod_data"].shape[1]
            time = np.asarray(d["tods"].time[:t_steps], dtype=np.float64)
            boresight_el = d["boresight_el"][:t_steps] if d.get("boresight_el") is not None else None
            atm_model = AtmosphereModel.build(
                d["detector"],
                d["tod_data"],
                time=time,
                boresight_el=boresight_el,
                **cfg,
            )
            if verbose:
                logger.info(f"  atm {i + 1:02d} shape: {atm_model.target.shape}")
        elif verbose:
            logger.info("  atm model disabled")

        sum_response = SumResponse.build(
            atm_model=atm_model,
            map_model=map_model,
            offsets=d["offsets"],
            detector=d.get("detector"),
            **config["response"],
        )

        if d.get("noise_cov_inv") is not None:
            # Per-detector/per-sample inverse-noise-variance array (e.g. from a
            # noise estimate plus a bad-sample mask), broadcastable against
            # `d["tod_data"]`; overrides the global scalar below.
            inv_var = np.asarray(d["noise_cov_inv"])
            noise_cov_inv = lambda x, _inv=inv_var: _inv * x  # noqa: E731
        elif d.get("bad_mask") is not None and np.any(d["bad_mask"]):
            # By-hand mask_fits/mask_time masking (see load_tod_datasets) with
            # no dedicated noise estimate available: fall back to the global
            # scalar noise_std everywhere except the masked samples, which get
            # a near-zero weight so they drop out of the likelihood.
            noise_std = config.get("noise_std")
            inv_var = np.full(d["tod_data"].shape, float(noise_std) ** -2)
            inv_var[d["bad_mask"]] = float(config.get("bad_sample_inv_var", 1.0e-6))
            noise_cov_inv = lambda x, _inv=inv_var: _inv * x  # noqa: E731
        else:
            noise_std = config.get("noise_std")
            noise_cov_inv = lambda x, _std=noise_std: _std**-2 * x  # noqa: E731

        logger.setLevel(logging.ERROR)
        lh = Gaussian(d["tod_data"], noise_cov_inv).amend(sum_response)
        logger.setLevel(logging.DEBUG)

        d["likelihood"] = lh
        d["model"] = sum_response

    likelihood = reduce(lambda a, b: a + b, [d["likelihood"] for d in datasets])
    if verbose:
        logger.info("\nLikelihood domain:")
        for k, v in sorted(likelihood.domain.tree.items()):
            logger.info(f"  {k}: {v.shape}")

    return likelihood, datasets


def setup_sample_distribution(config: dict, mini_dct: dict, devices=None, verbose: bool = True):
    """Configure NIFTy's built-in Monte-Carlo sample distribution across devices.

    With more than one visible device, NIFTy spreads the samples (and the
    residual sampling / KL minimization mapped over them) across devices. The
    model is replicated per device, so this distributes the per-sample work
    rather than the model memory.

    Parameters
    ----------
    config : dict
        Reconstruction config; optional keys ``residual_map``, ``kl_device_map``,
        ``residual_device_map`` override the corresponding ``optimize_kl`` maps.
    mini_dct : dict
        Parsed minimization kwargs (from :func:`parse_minimization`).
    devices : list, optional
        Devices to distribute over; defaults to ``jax.devices()``.
    verbose : bool, optional
        Whether to log the chosen distribution.

    Returns
    -------
    device_kwargs : dict
        Extra kwargs for ``optimize_kl`` (empty when <= 1 device).
    mini_dct : dict
        The minimization kwargs, with static solvers injected on the device path.

    Notes
    -----
    The device path traces (jit/vmap/shard_map) the sample solvers, so they must
    use static control flow:

    - ``residual_map`` must be ``"vmap"`` (the default ``"lmap"`` calls
      ``.devices()`` on a traced array -> ``ConcretizationTypeError``);
    - the linear ``cg`` must be ``static_cg`` (the default ``cg``'s ``float()``
      raises under tracing);
    - the geoVI ``minimize`` must be the private ``_static_newton_cg``, which
      returns an ``OptimizeResults`` (with ``.x``) as expected by
      ``nonlinearly_update_residual`` — the public ``static_newton_cg`` returns a
      ``Vector`` and crashes with ``'Vector' object has no attribute 'x'``.
    """
    import jax

    if devices is None:
        devices = jax.devices()
    if len(devices) <= 1:
        return {}, mini_dct

    from nifty.re import static_cg
    from nifty.re.optimize import _static_newton_cg

    device_kwargs = {"devices": devices, "residual_map": config.get("residual_map", "vmap")}
    if "kl_device_map" in config:
        device_kwargs["kl_device_map"] = config["kl_device_map"]
    if "residual_device_map" in config:
        device_kwargs["residual_device_map"] = config["residual_device_map"]

    def _add_static(kw, update):
        if kw is None:
            return update
        if callable(kw):
            # Per-iteration callable: return a *single-argument* closure, because
            # NIFTy only resolves callables with exactly one arg (it inspects the
            # arg count); a 2-arg lambda would be passed through verbatim and then
            # `**`-unpacked as a function.
            def per_iter(i):
                return dict(kw(i)) | update

            return per_iter
        return dict(kw) | update

    mini_dct = dict(mini_dct)
    mini_dct["draw_linear_kwargs"] = _add_static(mini_dct.get("draw_linear_kwargs"), {"cg": static_cg})
    mini_dct["nonlinearly_update_kwargs"] = _add_static(
        mini_dct.get("nonlinearly_update_kwargs"), {"minimize": _static_newton_cg}
    )

    if verbose:
        logger.info(
            f"\nDistributing samples over {len(devices)} devices: {devices} "
            f"(residual_map={device_kwargs['residual_map']}, "
            f"kl_device_map={config.get('kl_device_map', 'shard_map')}, "
            f"residual_device_map={config.get('residual_device_map', 'pmap')}, static solvers)"
        )
    return device_kwargs, mini_dct
